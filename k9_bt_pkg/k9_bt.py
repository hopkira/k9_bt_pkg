#!/usr/bin/env python3
"""Visible, non-blocking behaviour-tree shell for K9.

This version implements the persistent LLM conversation loop:

    WAITING_FOR_HOTWORD
        -> hotword
        -> LISTENING
        -> utterance / intent
        -> GENERAL_CONVERSATION
        -> /conversation/response
        -> speak response
        -> LISTENING
        -> ...
        -> STOP_LISTENING
        -> reset conversation history
        -> WAITING_FOR_HOTWORD

The desired audio mode remains LISTENING for the lifetime of an active
conversation. While K9 is physically speaking, /voice/is_talking temporarily
forces the effective audio mode to NOT_LISTENING. When speech finishes, the
effective mode therefore returns to LISTENING and the STT node begins a fresh
listening session.

Raw STT text is retained for diagnostics only. Dialogue sequencing waits for
/intent/result so STOP_LISTENING cannot race the normal conversation path.
The separate /conversation node subscribes to the same IntentResult and
publishes generated text on /conversation/response.
"""

from __future__ import annotations

from dataclasses import dataclass

import threading
import py_trees
import py_trees_ros
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from k9_interfaces_pkg.action import SpeakText
from k9_interfaces_pkg.msg import (
    IntentResult,
    RecognisedFaceArray,
)
try:
    # Normal installed-package / ros2 run path.
    from k9_bt_pkg.k9_blackboard import (
        AudioMode,
        BlackboardKey,
        DialogueState,
        Intent,
        K9Blackboard,
    )
except ModuleNotFoundError:
    # Convenient direct execution from the source directory.
    from k9_blackboard import (
        AudioMode,
        BlackboardKey,
        DialogueState,
        Intent,
        K9Blackboard,
    )


# ---------------------------------------------------------------------------
# Small blackboard helpers
# ---------------------------------------------------------------------------

def register_read_write(
    client: py_trees.blackboard.Client,
    key: str,
) -> None:
    """Register both READ and WRITE access for one blackboard key."""
    client.register_key(
        key=key,
        access=py_trees.common.Access.READ,
    )
    client.register_key(
        key=key,
        access=py_trees.common.Access.WRITE,
    )


def clear_dialogue_turn(
    blackboard: py_trees.blackboard.Client,
) -> None:
    """Clear transient state for one conversational turn.

    Deliberately does not change:
      * DIALOGUE_CONVERSATION_ACTIVE
      * AUDIO_DESIRED_MODE

    Those two values describe the lifetime of the conversation, not an
    individual utterance.
    """

    blackboard.set(
        BlackboardKey.DIALOGUE_COMMAND,
        "",
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.DIALOGUE_INTENT,
        Intent.NONE,
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
        0.0,
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.DIALOGUE_PENDING_RESPONSE,
        "",
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.DIALOGUE_STOP_LISTENING_REQUESTED,
        False,
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.DIALOGUE_STATE,
        DialogueState.IDLE,
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.DIALOGUE_ERROR,
        "",
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.AUDIO_HEARD_TEXT,
        "",
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# Generic shell behaviours
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaceholderResult:
    """Configuration for a shell leaf."""

    status: py_trees.common.Status
    feedback: str


class Placeholder(py_trees.behaviour.Behaviour):
    """A deterministic leaf used while the real behaviour is not implemented."""

    def __init__(
        self,
        name: str,
        result: PlaceholderResult,
    ) -> None:
        super().__init__(name=name)
        self.result = result

    def update(self) -> py_trees.common.Status:
        self.feedback_message = self.result.feedback
        return self.result.status


class BlackboardEquals(py_trees.behaviour.Behaviour):
    """SUCCESS when a K9 blackboard value equals the requested value."""

    def __init__(self, name: str, key: str, expected) -> None:
        super().__init__(name=name)

        self.key = key
        self.expected = expected

        self.blackboard = py_trees.blackboard.Client(
            name=name,
            namespace="k9",
        )
        self.blackboard.register_key(
            key=key,
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        actual = self.blackboard.get(self.key)

        self.feedback_message = (
            f"{actual!r} {'==' if actual == self.expected else '!='} "
            f"{self.expected!r}"
        )

        if actual == self.expected:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# ROS -> blackboard event bridge
# ---------------------------------------------------------------------------

class ProcessAudioEvents(py_trees.behaviour.Behaviour):
    """Receive ROS audio/dialogue events and reflect them onto the blackboard."""

    def __init__(self, node: Node) -> None:
        super().__init__(name="Process Audio Events")

        self.node = node

        self.blackboard = py_trees.blackboard.Client(
            name="Process Audio Events",
            namespace="k9",
        )

        # Audio event/state fields.
        for key in [
            BlackboardKey.AUDIO_HOTWORD_DETECTED,
            BlackboardKey.AUDIO_IS_LISTENING,
            BlackboardKey.AUDIO_IS_TALKING,
            BlackboardKey.AUDIO_HEARD_TEXT,
            BlackboardKey.AUDIO_LAST_EVENT,
        ]:
            register_read_write(self.blackboard, key)

        # Dialogue fields populated by IntentResult and conversation responses.
        for key in [
            BlackboardKey.DIALOGUE_COMMAND,
            BlackboardKey.DIALOGUE_INTENT,
            BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            BlackboardKey.DIALOGUE_STOP_LISTENING_REQUESTED,
            BlackboardKey.DIALOGUE_STATE,
            BlackboardKey.DIALOGUE_ERROR,
        ]:
            register_read_write(self.blackboard, key)

        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            access=py_trees.common.Access.READ,
        )

        self.hotword_subscription = node.create_subscription(
            Bool,
            "/hotword_detected",
            self._hotword_callback,
            10,
        )

        self.stt_state_subscription = node.create_subscription(
            String,
            "/speech_to_text/state",
            self._stt_state_callback,
            10,
        )

        self.stt_text_subscription = node.create_subscription(
            String,
            "/speech_to_text/text",
            self._stt_text_callback,
            10,
        )

        self.intent_subscription = node.create_subscription(
            IntentResult,
            "/intent/result",
            self._intent_callback,
            10,
        )

        self.conversation_response_subscription = node.create_subscription(
            String,
            "/conversation/response",
            self._conversation_response_callback,
            10,
        )

        # This is what makes the audio manager's Talking Override real.
        self.voice_talking_subscription = node.create_subscription(
            Bool,
            "/voice/is_talking",
            self._voice_talking_callback,
            10,
        )

    def _hotword_callback(self, msg: Bool) -> None:
        # Treat True as an event and latch it until BeginConversation consumes it.
        if not msg.data:
            return

        self.blackboard.set(
            BlackboardKey.AUDIO_HOTWORD_DETECTED,
            True,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_LAST_EVENT,
            "HOTWORD_DETECTED",
            overwrite=True,
        )

    def _stt_state_callback(self, msg: String) -> None:
        state = msg.data.strip().lower()

        is_listening = state in {
            "listening",
            "speech",
        }

        self.blackboard.set(
            BlackboardKey.AUDIO_IS_LISTENING,
            is_listening,
            overwrite=True,
        )

    def _stt_text_callback(self, msg: String) -> None:
        """Record raw STT text for diagnostics only.

        The dialogue manager deliberately does NOT act on this field. It waits
        for /intent/result so STOP_LISTENING cannot race the normal response
        path.
        """
        text = msg.data.strip()

        if not text:
            return

        self.blackboard.set(
            BlackboardKey.AUDIO_HEARD_TEXT,
            text,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_LAST_EVENT,
            "UTTERANCE_RECEIVED",
            overwrite=True,
        )

    def _intent_callback(self, msg: IntentResult) -> None:
        """Commit one complete interpreted utterance to the dialogue manager."""

        text = msg.text.strip()
        intent = msg.intent.strip().upper() or Intent.NONE

        if not text:
            return

        # Ignore stray/stale intent results outside a conversation.
        if not self.blackboard.get(
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
        ):
            self.node.get_logger().debug(
                f"Ignoring intent outside active conversation: "
                f"{intent} / {text!r}"
            )
            return

        # A new interpreted utterance begins a new turn. Clear any previous
        # response before making the new command visible to the tree.
        self.blackboard.set(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            "",
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_ERROR,
            "",
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_COMMAND,
            text,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_INTENT,
            intent,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
            float(msg.confidence),
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_STOP_LISTENING_REQUESTED,
            intent == Intent.STOP_LISTENING,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_STATE,
            DialogueState.PROCESSING,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_LAST_EVENT,
            "INTENT_RECEIVED",
            overwrite=True,
        )

        self.node.get_logger().info(
            f"Intent received: {intent} "
            f"({float(msg.confidence):.2f}) / {text!r} "
            f"requires_response={bool(msg.requires_response)}"
        )

        if (
            intent == Intent.GENERAL_CONVERSATION
            and not msg.requires_response
        ):
            self.node.get_logger().warning(
                "GENERAL_CONVERSATION was marked requires_response=false; "
                "the conversation node will not produce a reply"
            )

    def _conversation_response_callback(self, msg: String) -> None:
        """Latch a generated LLM response for the current general turn.

        The response is written here, rather than held locally by a waiting
        leaf, so it cannot be lost if Ollama replies between BT ticks.
        """
        text = msg.data.strip()

        if not text:
            return

        active = bool(
            self.blackboard.get(
                BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
            )
        )
        current_intent = self.blackboard.get(
            BlackboardKey.DIALOGUE_INTENT
        )
        current_command = self.blackboard.get(
            BlackboardKey.DIALOGUE_COMMAND
        ).strip()

        if (
            not active
            or current_intent != Intent.GENERAL_CONVERSATION
            or not current_command
        ):
            self.node.get_logger().debug(
                "Ignoring conversation response with no matching active "
                "GENERAL_CONVERSATION turn"
            )
            return

        self.blackboard.set(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            text,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_STATE,
            DialogueState.WAITING_TO_SPEAK,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_LAST_EVENT,
            "CONVERSATION_RESPONSE_RECEIVED",
            overwrite=True,
        )

        self.node.get_logger().info(
            f"Conversation response received: {text!r}"
        )

    def _voice_talking_callback(self, msg: Bool) -> None:
        self.blackboard.set(
            BlackboardKey.AUDIO_IS_TALKING,
            bool(msg.data),
            overwrite=True,
        )

    def update(self) -> py_trees.common.Status:
        self.feedback_message = "monitoring ROS audio/dialogue events"
        return py_trees.common.Status.RUNNING

class ProcessPerceptionEvents(py_trees.behaviour.Behaviour):
    """Bridge recognised-face ROS state into the K9 blackboard."""

    def __init__(self, node: Node) -> None:
        super().__init__(name="Process Perception Events")

        self.node = node

        # The ROS callback writes only to this local cache.
        # Blackboard changes are made synchronously from update().
        self._lock = threading.Lock()
        self._latest_faces = {}
        self._new_message = False

        # Previous state is retained so that transitions can be derived.
        self._previous_faces = {}

        self.blackboard = py_trees.blackboard.Client(
            name="Process Perception Events",
            namespace="k9",
        )

        for key in [
            BlackboardKey.PERCEPTION_PERSON_COUNT,
            BlackboardKey.PERCEPTION_PERSON_VISIBLE,
            BlackboardKey.PERCEPTION_KNOWN_PERSON_VISIBLE,
            BlackboardKey.PERCEPTION_VISIBLE_TRACK_IDS,
            BlackboardKey.PERCEPTION_VISIBLE_IDENTITIES,
            BlackboardKey.PERCEPTION_LAST_EVENT,
            BlackboardKey.PERCEPTION_EVENT_TRACK_ID,
            BlackboardKey.PERCEPTION_EVENT_IDENTITY,
            BlackboardKey.PERCEPTION_ERROR,
        ]:
            register_read_write(
                self.blackboard,
                key,
            )

        self.subscription = node.create_subscription(
            RecognisedFaceArray,
            "/k9/perception/recognised_faces",
            self._faces_callback,
            10,
        )

    def _faces_callback(
        self,
        msg: RecognisedFaceArray,
    ) -> None:
        """Cache the latest recognition state.

        Do not mutate the BT blackboard from the asynchronous ROS callback.
        """

        faces = {}

        for face in msg.faces:
            track_id = int(
                face.track_id
            )

            faces[track_id] = {
                "recognised": bool(
                    face.recognised
                ),
                "identity": (
                    face.identity.strip()
                    if face.recognised
                    else ""
                ),
                "confidence": float(
                    face.recognition_confidence
                ),
            }

        with self._lock:
            self._latest_faces = faces
            self._new_message = True

    def _set_event(
        self,
        event: str,
        track_id: int,
        identity: str = "",
    ) -> None:
        self.blackboard.set(
            BlackboardKey.PERCEPTION_LAST_EVENT,
            event,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.PERCEPTION_EVENT_TRACK_ID,
            track_id,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.PERCEPTION_EVENT_IDENTITY,
            identity,
            overwrite=True,
        )

    def update(self) -> py_trees.common.Status:

        # Events last for one BT tick unless a new transition occurs.
        self.blackboard.set(
            BlackboardKey.PERCEPTION_LAST_EVENT,
            "",
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.PERCEPTION_EVENT_TRACK_ID,
            0,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.PERCEPTION_EVENT_IDENTITY,
            "",
            overwrite=True,
        )

        with self._lock:
            if not self._new_message:
                self.feedback_message = (
                    "no new perception message"
                )
                return py_trees.common.Status.RUNNING

            current_faces = dict(
                self._latest_faces
            )

            self._new_message = False

        previous_faces = self._previous_faces

        current_ids = set(
            current_faces.keys()
        )

        previous_ids = set(
            previous_faces.keys()
        )

        # --------------------------------------------------------
        # Persistent world state
        # --------------------------------------------------------

        visible_track_ids = sorted(
            current_ids
        )

        visible_identities = sorted(
            {
                face["identity"]
                for face in current_faces.values()
                if (
                    face["recognised"]
                    and face["identity"]
                )
            }
        )

        self.blackboard.set(
            BlackboardKey.PERCEPTION_PERSON_COUNT,
            len(current_faces),
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.PERCEPTION_PERSON_VISIBLE,
            bool(current_faces),
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.PERCEPTION_KNOWN_PERSON_VISIBLE,
            bool(visible_identities),
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.PERCEPTION_VISIBLE_TRACK_IDS,
            visible_track_ids,
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.PERCEPTION_VISIBLE_IDENTITIES,
            visible_identities,
            overwrite=True,
        )

        # --------------------------------------------------------
        # Derive transitions
        # --------------------------------------------------------

        appeared = (
            current_ids - previous_ids
        )

        disappeared = (
            previous_ids - current_ids
        )

        event_generated = False

        # First priority: completely new person/track.
        if appeared:
            track_id = min(
                appeared
            )

            face = current_faces[
                track_id
            ]

            self._set_event(
                "PERSON_APPEARED",
                track_id,
                face["identity"],
            )

            event_generated = True

        # Second priority: an existing unknown track has just
        # become recognised.
        if not event_generated:
            for track_id in sorted(
                current_ids & previous_ids
            ):
                current = current_faces[
                    track_id
                ]
                previous = previous_faces[
                    track_id
                ]

                if (
                    current["recognised"]
                    and
                    not previous["recognised"]
                ):
                    self._set_event(
                        "KNOWN_PERSON_APPEARED",
                        track_id,
                        current["identity"],
                    )

                    event_generated = True
                    break

        # Third priority: somebody left.
        if (
            not event_generated
            and disappeared
        ):
            track_id = min(
                disappeared
            )

            previous = previous_faces[
                track_id
            ]

            event = (
                "KNOWN_PERSON_LEFT"
                if previous["recognised"]
                else "PERSON_LEFT"
            )

            self._set_event(
                event,
                track_id,
                previous["identity"],
            )

        self._previous_faces = (
            current_faces
        )

        if visible_identities:
            self.feedback_message = (
                f"{len(current_faces)} visible: "
                + ", ".join(
                    visible_identities
                )
            )
        else:
            self.feedback_message = (
                f"{len(current_faces)} visible; "
                "none recognised"
            )

        return py_trees.common.Status.RUNNING

# ---------------------------------------------------------------------------
# Audio state arbitration
# ---------------------------------------------------------------------------

class MaintainAudioMode(py_trees.behaviour.Behaviour):
    """Maintain one effective audio mode and publish it to ROS."""

    def __init__(
        self,
        node: Node,
        name: str,
        mode: str,
    ) -> None:
        super().__init__(name=name)

        self.node = node
        self.mode = mode

        self.blackboard = py_trees.blackboard.Client(
            name=name,
            namespace="k9",
        )
        register_read_write(
            self.blackboard,
            BlackboardKey.AUDIO_EFFECTIVE_MODE,
        )

        self.publisher = node.create_publisher(
            String,
            "/audio/effective_state",
            10,
        )

    def update(self) -> py_trees.common.Status:
        current = self.blackboard.get(
            BlackboardKey.AUDIO_EFFECTIVE_MODE
        )

        if current != self.mode:
            self.blackboard.set(
                BlackboardKey.AUDIO_EFFECTIVE_MODE,
                self.mode,
                overwrite=True,
            )

            self.node.get_logger().info(
                f"Effective audio state: {current} -> {self.mode}"
            )

        # Publish every BT tick. A hotword/STT node which restarts therefore
        # quickly receives the current state even though the topic is volatile.
        self.publisher.publish(String(data=self.mode))

        self.feedback_message = f"maintaining {self.mode}"
        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# Persistent conversation leaves
# ---------------------------------------------------------------------------

class WaitForHotword(py_trees.behaviour.Behaviour):
    """SUCCESS when a latched hotword event is ready to be consumed."""

    def __init__(self, name: str = "Wait For Hotword") -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )
        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_HOTWORD_DETECTED,
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        detected = bool(
            self.blackboard.get(
                BlackboardKey.AUDIO_HOTWORD_DETECTED
            )
        )

        if detected:
            self.feedback_message = "hotword pending"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = "waiting for hotword"
        return py_trees.common.Status.FAILURE


class BeginConversation(py_trees.behaviour.Behaviour):
    """Consume the hotword and enter persistent LISTENING mode."""

    def __init__(self, name: str = "Begin Conversation") -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        for key in [
            BlackboardKey.AUDIO_HOTWORD_DETECTED,
            BlackboardKey.AUDIO_DESIRED_MODE,
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            BlackboardKey.DIALOGUE_COMMAND,
            BlackboardKey.DIALOGUE_INTENT,
            BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            BlackboardKey.DIALOGUE_STOP_LISTENING_REQUESTED,
            BlackboardKey.DIALOGUE_STATE,
            BlackboardKey.DIALOGUE_ERROR,
            BlackboardKey.AUDIO_HEARD_TEXT,
        ]:
            register_read_write(self.blackboard, key)

    def update(self) -> py_trees.common.Status:
        # Clear anything left from the previous conversation before activating
        # the new one.
        clear_dialogue_turn(self.blackboard)

        self.blackboard.set(
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            True,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_DESIRED_MODE,
            AudioMode.LISTENING,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_HOTWORD_DETECTED,
            False,
            overwrite=True,
        )

        self.feedback_message = "conversation active; requested LISTENING"
        return py_trees.common.Status.SUCCESS


class ConversationActive(py_trees.behaviour.Behaviour):
    """SUCCESS while the persistent conversation is active."""

    def __init__(self, name: str = "Conversation Active?") -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )
        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        active = bool(
            self.blackboard.get(
                BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
            )
        )

        self.feedback_message = (
            "conversation active" if active else "conversation inactive"
        )

        if active:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE


class WaitForCommand(py_trees.behaviour.Behaviour):
    """Wait for the intent node to provide a complete interpreted utterance."""

    def __init__(self, name: str = "Wait For Interpreted Utterance") -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )
        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_COMMAND,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_INTENT,
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        command = self.blackboard.get(
            BlackboardKey.DIALOGUE_COMMAND
        ).strip()

        if not command:
            self.feedback_message = "waiting for /intent/result"
            return py_trees.common.Status.RUNNING

        intent = self.blackboard.get(
            BlackboardKey.DIALOGUE_INTENT
        )
        self.feedback_message = f"{intent}: {command}"
        return py_trees.common.Status.SUCCESS


class IsIntent(py_trees.behaviour.Behaviour):
    """SUCCESS when the current dialogue intent equals the requested intent."""

    def __init__(
        self,
        expected_intent: str,
        name: str | None = None,
    ) -> None:
        super().__init__(
            name=name or f"Intent = {expected_intent}?"
        )

        self.expected_intent = expected_intent

        self.blackboard = self.attach_blackboard_client(
            name=self.name,
            namespace="k9",
        )
        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_INTENT,
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        actual = self.blackboard.get(
            BlackboardKey.DIALOGUE_INTENT
        )

        self.feedback_message = (
            f"{actual} "
            f"{'==' if actual == self.expected_intent else '!='} "
            f"{self.expected_intent}"
        )

        if actual == self.expected_intent:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE


class WaitForConversationResponse(py_trees.behaviour.Behaviour):
    """Wait for /conversation to populate the pending LLM response."""

    def __init__(
        self,
        name: str = "Wait For Conversation Response",
    ) -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )
        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_INTENT,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        if not self.blackboard.get(
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
        ):
            self.feedback_message = "conversation no longer active"
            return py_trees.common.Status.FAILURE

        intent = self.blackboard.get(
            BlackboardKey.DIALOGUE_INTENT
        )
        if intent != Intent.GENERAL_CONVERSATION:
            self.feedback_message = f"not a general conversation turn: {intent}"
            return py_trees.common.Status.FAILURE

        response = self.blackboard.get(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE
        ).strip()

        if not response:
            self.feedback_message = "waiting for /conversation/response"
            return py_trees.common.Status.RUNNING

        self.feedback_message = response
        return py_trees.common.Status.SUCCESS


class GenerateUnsupportedIntentResponse(py_trees.behaviour.Behaviour):
    """Provide a temporary safe response for recognised but unimplemented intents.

    PLAY_CHESS and CHESS_SETUP_ANSWER already exist in the intent schema, but
    their real dialogue handlers are still placeholders. This leaf prevents an
    unimplemented intent from leaving the active conversation permanently stuck.
    """

    def __init__(
        self,
        name: str = "Handle Unimplemented Intent",
    ) -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )
        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_INTENT,
            access=py_trees.common.Access.READ,
        )
        register_read_write(
            self.blackboard,
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
        )
        register_read_write(
            self.blackboard,
            BlackboardKey.DIALOGUE_STATE,
        )

    def update(self) -> py_trees.common.Status:
        intent = self.blackboard.get(
            BlackboardKey.DIALOGUE_INTENT
        )

        response = (
            f"I recognised the {intent} command, but that behaviour is not "
            "yet connected."
        )

        self.blackboard.set(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            response,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_STATE,
            DialogueState.WAITING_TO_SPEAK,
            overwrite=True,
        )

        self.feedback_message = response
        return py_trees.common.Status.SUCCESS


class ResetConversationHistory(py_trees.behaviour.Behaviour):
    """Best-effort reset of /conversation history when a session ends."""

    def __init__(
        self,
        *,
        node: Node,
        name: str = "Reset Conversation History",
    ) -> None:
        super().__init__(name=name)

        self.node = node
        self.client = node.create_client(
            Trigger,
            "/conversation/reset",
        )
        self.future = None
        self.warned_unavailable = False

    def initialise(self) -> None:
        self.future = None

    def update(self) -> py_trees.common.Status:
        if self.future is None:
            if not self.client.service_is_ready():
                if not self.warned_unavailable:
                    self.node.get_logger().warning(
                        "/conversation/reset unavailable; ending conversation "
                        "without resetting LLM history"
                    )
                    self.warned_unavailable = True

                # Reset is useful, but must never prevent STOP_LISTENING.
                self.feedback_message = "reset unavailable; continuing"
                return py_trees.common.Status.SUCCESS

            self.future = self.client.call_async(Trigger.Request())
            self.feedback_message = "conversation reset requested"
            return py_trees.common.Status.RUNNING

        if not self.future.done():
            self.feedback_message = "waiting for conversation reset"
            return py_trees.common.Status.RUNNING

        try:
            result = self.future.result()
        except Exception as exc:
            self.node.get_logger().warning(
                f"Conversation reset failed: {exc}"
            )
            self.feedback_message = f"reset failed: {exc}; continuing"
            return py_trees.common.Status.SUCCESS

        if result.success:
            self.feedback_message = result.message or "conversation reset"
        else:
            self.node.get_logger().warning(
                f"Conversation reset rejected: {result.message}"
            )
            self.feedback_message = (
                result.message or "conversation reset rejected; continuing"
            )

        # A failed reset must not trap K9 in LISTENING.
        return py_trees.common.Status.SUCCESS


class SpeakPendingResponse(py_trees.behaviour.Behaviour):
    """Speak the pending dialogue response using the priority action server."""

    def __init__(
        self,
        *,
        node: Node,
        name: str = "Speak Pending Response",
    ) -> None:
        super().__init__(name=name)

        self.node = node

        self.client = ActionClient(
            node,
            SpeakText,
            "/voice/speak",
        )

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            access=py_trees.common.Access.READ,
        )
        register_read_write(
            self.blackboard,
            BlackboardKey.DIALOGUE_STATE,
        )
        register_read_write(
            self.blackboard,
            BlackboardKey.DIALOGUE_ERROR,
        )

        self.goal_future = None
        self.goal_handle = None
        self.result_future = None
        self.response_text = ""

    def initialise(self) -> None:
        self.goal_future = None
        self.goal_handle = None
        self.result_future = None

        self.response_text = self.blackboard.get(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE
        ).strip()

        if not self.response_text:
            self.feedback_message = "no pending response"
            return

        if not self.client.server_is_ready():
            self.feedback_message = "voice action server unavailable"
            return

        goal = SpeakText.Goal()
        goal.text = self.response_text
        goal.owner = "dialogue"
        goal.priority = 100
        goal.interrupt_lower_priority = True
        goal.clear_lower_priority = True

        self.goal_future = self.client.send_goal_async(goal)
        self.feedback_message = "speech goal submitted"

    def update(self) -> py_trees.common.Status:
        if not self.response_text:
            return py_trees.common.Status.FAILURE

        if self.goal_future is None:
            # Returning FAILURE lets the conversation sequence retry on a later
            # BT tick without losing the command or pending response.
            self.feedback_message = "voice action server unavailable"
            return py_trees.common.Status.FAILURE

        # Waiting for the action server to accept the goal.
        if self.goal_handle is None:
            if not self.goal_future.done():
                self.feedback_message = "waiting for speech goal acceptance"
                return py_trees.common.Status.RUNNING

            try:
                self.goal_handle = self.goal_future.result()
            except Exception as exc:  # rclpy future exception
                self.blackboard.set(
                    BlackboardKey.DIALOGUE_ERROR,
                    f"speech goal failed: {exc}",
                    overwrite=True,
                )
                self.feedback_message = f"speech goal failed: {exc}"
                return py_trees.common.Status.FAILURE

            if self.goal_handle is None or not self.goal_handle.accepted:
                self.feedback_message = "speech goal rejected"
                return py_trees.common.Status.FAILURE

            self.result_future = self.goal_handle.get_result_async()

            self.blackboard.set(
                BlackboardKey.DIALOGUE_STATE,
                DialogueState.SPEAKING,
                overwrite=True,
            )

            self.feedback_message = "speaking"
            return py_trees.common.Status.RUNNING

        # Voice is still speaking.
        if self.result_future is None or not self.result_future.done():
            self.feedback_message = "speaking"
            return py_trees.common.Status.RUNNING

        # Speech completed.
        try:
            wrapped_result = self.result_future.result()
            result = wrapped_result.result
        except Exception as exc:  # rclpy future exception
            self.blackboard.set(
                BlackboardKey.DIALOGUE_ERROR,
                f"speech result failed: {exc}",
                overwrite=True,
            )
            self.feedback_message = f"speech result failed: {exc}"
            return py_trees.common.Status.FAILURE

        if not result.success:
            self.blackboard.set(
                BlackboardKey.DIALOGUE_ERROR,
                result.message,
                overwrite=True,
            )
            self.feedback_message = result.message
            return py_trees.common.Status.FAILURE

        # Do NOT return to WAITING_FOR_HOTWORD here. The conversation is still
        # active and AUDIO_DESIRED_MODE remains LISTENING.
        self.feedback_message = "speech complete; resume conversation"
        return py_trees.common.Status.SUCCESS


class ClearConversationTurn(py_trees.behaviour.Behaviour):
    """Finish one normal turn and remain in the active conversation."""

    def __init__(self, name: str = "Clear Conversation Turn") -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        for key in [
            BlackboardKey.DIALOGUE_COMMAND,
            BlackboardKey.DIALOGUE_INTENT,
            BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            BlackboardKey.DIALOGUE_STOP_LISTENING_REQUESTED,
            BlackboardKey.DIALOGUE_STATE,
            BlackboardKey.DIALOGUE_ERROR,
            BlackboardKey.AUDIO_HEARD_TEXT,
            BlackboardKey.AUDIO_DESIRED_MODE,
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
        ]:
            register_read_write(self.blackboard, key)

    def update(self) -> py_trees.common.Status:
        clear_dialogue_turn(self.blackboard)

        # A successfully completed normal turn always returns to listening,
        # never to the hotword detector.
        self.blackboard.set(
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            True,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_DESIRED_MODE,
            AudioMode.LISTENING,
            overwrite=True,
        )

        self.feedback_message = "turn cleared; still LISTENING"
        return py_trees.common.Status.SUCCESS


class EndConversation(py_trees.behaviour.Behaviour):
    """End the conversation only after an explicit STOP_LISTENING intent."""

    def __init__(self, name: str = "End Conversation") -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        for key in [
            BlackboardKey.DIALOGUE_COMMAND,
            BlackboardKey.DIALOGUE_INTENT,
            BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            BlackboardKey.DIALOGUE_STOP_LISTENING_REQUESTED,
            BlackboardKey.DIALOGUE_STATE,
            BlackboardKey.DIALOGUE_ERROR,
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            BlackboardKey.AUDIO_HEARD_TEXT,
            BlackboardKey.AUDIO_DESIRED_MODE,
            BlackboardKey.AUDIO_LAST_EVENT,
        ]:
            register_read_write(self.blackboard, key)

    def update(self) -> py_trees.common.Status:
        clear_dialogue_turn(self.blackboard)

        self.blackboard.set(
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            False,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_DESIRED_MODE,
            AudioMode.WAITING_FOR_HOTWORD,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_LAST_EVENT,
            "CONVERSATION_ENDED",
            overwrite=True,
        )

        self.feedback_message = "STOP_LISTENING; waiting for hotword"
        return py_trees.common.Status.SUCCESS


class DialogueIdle(py_trees.behaviour.Behaviour):
    """Visible idle leaf used while waiting for the next hotword."""

    def __init__(self, name: str = "Dialogue Idle") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        self.feedback_message = "waiting for hotword"
        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# Shell helper constructors
# ---------------------------------------------------------------------------

RUNNING = PlaceholderResult(
    status=py_trees.common.Status.RUNNING,
    feedback="shell: maintaining state",
)

INACTIVE = PlaceholderResult(
    status=py_trees.common.Status.FAILURE,
    feedback="shell: inactive",
)

NO_WORK = PlaceholderResult(
    status=py_trees.common.Status.FAILURE,
    feedback="shell: no matching work",
)


def running(name: str) -> Placeholder:
    return Placeholder(name=name, result=RUNNING)


def inactive(name: str) -> Placeholder:
    return Placeholder(name=name, result=INACTIVE)


def no_work(name: str) -> Placeholder:
    return Placeholder(name=name, result=NO_WORK)


def selector(name: str) -> py_trees.composites.Selector:
    return py_trees.composites.Selector(
        name=name,
        memory=False,
    )


def sequence(name: str) -> py_trees.composites.Sequence:
    return py_trees.composites.Sequence(
        name=name,
        memory=False,
    )


def parallel(name: str) -> py_trees.composites.Parallel:
    return py_trees.composites.Parallel(
        name=name,
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(
            synchronise=False,
        ),
    )


# ---------------------------------------------------------------------------
# Subtrees
# ---------------------------------------------------------------------------

def create_audio_state_manager(
    node: Node,
) -> py_trees.behaviour.Behaviour:
    process_audio_events = ProcessAudioEvents(node)

    # Highest priority: while the voice node says K9 is talking, force the
    # effective mode away from STT regardless of the persistent desired mode.
    talking_override = sequence("Talking Override")
    talking_override.add_children(
        [
            BlackboardEquals(
                "K9 Talking?",
                BlackboardKey.AUDIO_IS_TALKING,
                True,
            ),
            MaintainAudioMode(
                node,
                "Ensure NotListening",
                AudioMode.NOT_LISTENING,
            ),
        ]
    )

    listening_state = sequence("Listening State")
    listening_state.add_children(
        [
            BlackboardEquals(
                "Desired = Listening?",
                BlackboardKey.AUDIO_DESIRED_MODE,
                AudioMode.LISTENING,
            ),
            MaintainAudioMode(
                node,
                "Maintain Listening",
                AudioMode.LISTENING,
            ),
        ]
    )

    hotword_state = sequence("Hotword State")
    hotword_state.add_children(
        [
            BlackboardEquals(
                "Desired = WaitingForHotword?",
                BlackboardKey.AUDIO_DESIRED_MODE,
                AudioMode.WAITING_FOR_HOTWORD,
            ),
            MaintainAudioMode(
                node,
                "Maintain Hotword Detector",
                AudioMode.WAITING_FOR_HOTWORD,
            ),
        ]
    )

    maintain_effective_audio_state = selector(
        "Maintain Effective Audio State"
    )
    maintain_effective_audio_state.add_children(
        [
            talking_override,
            listening_state,
            hotword_state,
            MaintainAudioMode(
                node,
                "Maintain NotListening",
                AudioMode.NOT_LISTENING,
            ),
        ]
    )

    audio_state_manager = parallel("Audio State Manager")
    audio_state_manager.add_children(
        [
            process_audio_events,
            maintain_effective_audio_state,
        ]
    )

    return audio_state_manager


def create_perception_state_manager(
    node: Node,
) -> py_trees.behaviour.Behaviour:
    return ProcessPerceptionEvents(node)

    return perception_manager


def create_dialogue_manager(
    node: Node,
) -> py_trees.behaviour.Behaviour:
    """Build the persistent hotword -> LLM conversation -> stop loop."""

    # Highest-priority turn path. STOP_LISTENING is consumed by the BT and is
    # never sent to speech. Reset the LLM session before returning to hotword.
    stop_conversation = py_trees.composites.Sequence(
        name="Stop Conversation",
        memory=True,
    )
    stop_conversation.add_children(
        [
            IsIntent(Intent.STOP_LISTENING),
            ResetConversationHistory(node=node),
            EndConversation(),
        ]
    )

    # Normal conversational turn. The /conversation node has already received
    # the same /intent/result independently and is generating asynchronously.
    # We simply wait for its latched /conversation/response, speak it, then
    # clear the turn while leaving the conversation active.
    general_conversation = py_trees.composites.Sequence(
        name="General Conversation",
        memory=True,
    )
    general_conversation.add_children(
        [
            IsIntent(Intent.GENERAL_CONVERSATION),
            WaitForConversationResponse(),
            SpeakPendingResponse(node=node),
            ClearConversationTurn(),
        ]
    )

    # PLAY_CHESS and CHESS_SETUP_ANSWER exist already but are not yet connected
    # to real dialogue behaviours. Do not wait for an LLM response that the
    # conversation node will never publish for those intents.
    unimplemented_intent = py_trees.composites.Sequence(
        name="Unimplemented Intent",
        memory=True,
    )
    unimplemented_intent.add_children(
        [
            GenerateUnsupportedIntentResponse(),
            SpeakPendingResponse(node=node),
            ClearConversationTurn(),
        ]
    )

    handle_turn = py_trees.composites.Selector(
        name="Handle Conversation Turn",
        memory=True,
    )
    handle_turn.add_children(
        [
            stop_conversation,
            general_conversation,
            unimplemented_intent,
        ]
    )

    # While a conversation is active this branch remains RUNNING in
    # WaitForCommand, then handles exactly one interpreted utterance.
    conversation = py_trees.composites.Sequence(
        name="Active Conversation",
        memory=True,
    )
    conversation.add_children(
        [
            ConversationActive(),
            WaitForCommand(),
            handle_turn,
        ]
    )

    # This branch is only reached when no conversation is active.
    start_conversation = py_trees.composites.Sequence(
        name="Start Conversation",
        memory=True,
    )
    start_conversation.add_children(
        [
            WaitForHotword(),
            BeginConversation(),
        ]
    )

    dialogue_manager = py_trees.composites.Selector(
        name="Dialogue Manager",
        memory=False,
    )
    dialogue_manager.add_children(
        [
            conversation,
            start_conversation,
            DialogueIdle(),
        ]
    )

    return dialogue_manager


def create_chess_manager() -> py_trees.behaviour.Behaviour:
    chess_setup = sequence("Chess Setup")
    chess_setup.add_children(
        [
            no_work("Ask Player Name"),
            no_work("Wait For Player Name"),
            no_work("Ask Preferred Colour"),
            no_work("Wait For Preferred Colour"),
            no_work("Start Lichess / Phantom Game"),
        ]
    )

    chess_manager = selector("Chess Manager")
    chess_manager.add_children(
        [
            no_work("Suspended Chess Session"),
            no_work("Active Chess Game"),
            chess_setup,
            no_work("Start Chess Setup"),
            running("Chess Idle"),
        ]
    )
    return chess_manager


def create_expression_manager() -> py_trees.behaviour.Behaviour:
    expression_manager = selector("Expression Manager")
    expression_manager.add_children(
        [
            inactive("Emergency Expression"),
            inactive("Talking Eye Animation"),
            inactive("Listening Expression"),
            inactive("Waiting Expression"),
            running("NotListening Expression"),
        ]
    )
    return expression_manager


def create_tree(node: Node) -> py_trees.behaviour.Behaviour:
    """Construct the complete K9 hierarchy."""

    emergency_mode = sequence("Emergency Mode")
    emergency_mode.add_children(
        [
            inactive("Emergency Button Pressed?"),
            running("Maintain Emergency State"),
        ]
    )

    normal_operation = parallel("Normal Operation")
    normal_operation.add_children(
        [
            create_audio_state_manager(node),
            create_perception_state_manager(node),
            create_dialogue_manager(node),
            create_chess_manager(),
            create_expression_manager(),
        ]
    )

    safety_executive = selector("Safety Executive")
    safety_executive.add_children(
        [
            emergency_mode,
            normal_operation,
        ]
    )

    root = parallel("K9 Root")
    root.add_children(
        [
            running("Battery Supervisor"),
            safety_executive,
        ]
    )

    return root


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class K9BehaviourTreeShell(Node):
    """ROS 2 custodian for the K9 behaviour-tree and shared blackboard."""

    def __init__(self) -> None:
        super().__init__("k9_bt_shell")

        self.declare_parameter("tick_period_ms", 200.0)
        tick_period_ms = float(
            self.get_parameter("tick_period_ms").value
        )
        if tick_period_ms <= 0.0:
            raise ValueError("tick_period_ms must be greater than zero")

        # Create and initialise the central /k9 blackboard before the tree ticks.
        self.blackboard = K9Blackboard()
        self.blackboard.set(
            BlackboardKey.SYSTEM_STATUS,
            "INITIALISING_TREE",
        )

        root = create_tree(self)

        self.tree = py_trees_ros.trees.BehaviourTree(
            root=root,
            unicode_tree_debug=False,
        )
        self.tree.setup(
            node=self,
            timeout=15.0,
        )

        parameter_results = self.set_parameters(
            [
                Parameter(
                    "default_snapshot_period",
                    Parameter.Type.DOUBLE,
                    0.5,
                ),
                Parameter(
                    "default_snapshot_stream",
                    Parameter.Type.BOOL,
                    True,
                ),
                Parameter(
                    "default_snapshot_blackboard_data",
                    Parameter.Type.BOOL,
                    True,
                ),
            ]
        )
        for result in parameter_results:
            if not result.successful:
                self.get_logger().warning(
                    "Could not configure a tree snapshot parameter: "
                    f"{result.reason}"
                )

        self.blackboard.set(
            BlackboardKey.SYSTEM_READY,
            True,
        )
        self.blackboard.set(
            BlackboardKey.SYSTEM_STATUS,
            "RUNNING",
        )

        # Tick once immediately so a newly opened viewer does not need to wait
        # for the first timer callback.
        self.tree.tick()

        # py_trees_ros implements this with an rclpy timer.
        self.tree.tick_tock(period_ms=tick_period_ms)

        self.get_logger().info("K9 behaviour-tree shell is running")
        self.get_logger().info(
            f"Initialised {self.blackboard.field_count} blackboard fields "
            "beneath /k9"
        )
        self.get_logger().info(
            f"Tick period: {tick_period_ms:.0f} ms"
        )
        self.get_logger().info(
            "Conversation flow: persistent LISTENING; "
            "/conversation/response -> voice; STOP_LISTENING exits"
        )
        self.get_logger().info(
            "Tree snapshots: /k9_bt_shell/snapshots"
        )
        self.get_logger().info(
            "Inspect state with: py-trees-blackboard-watcher"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node: K9BehaviourTreeShell | None = None

    try:
        node = K9BehaviourTreeShell()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.blackboard.set(
                BlackboardKey.SYSTEM_READY,
                False,
            )
            node.blackboard.set(
                BlackboardKey.SYSTEM_STATUS,
                "STOPPED",
            )

            # Installed py_trees_ros BehaviourTree.shutdown() takes no
            # destroy_node keyword in the current K9 environment.
            node.tree.shutdown()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
