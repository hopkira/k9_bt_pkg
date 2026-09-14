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

Face enrolment is an executive conversational branch:

    ENROL_FACE -> ask name -> family/friend -> optional form of address
               -> front/left/right capture -> commit -> LISTENING

The desired and effective audio modes remain LISTENING for the lifetime of an
active conversation. PROCESSING and SPEAKING are temporary interaction
activities published on /interaction/activity; the STT node inhibits recognition
during those activities, so the persistent interaction mode does not need to
change while K9 is thinking or speaking.

The physical back panel publishes /interaction/mode_request with one of
NOT_LISTENING, WAITING_FOR_HOTWORD, or LISTENING. Those requests are folded into
the same AUDIO_DESIRED_MODE state used by the behaviour tree, so the panel
remains a physical control surface rather than a parallel audio controller.

Raw STT text is retained for diagnostics only. Dialogue sequencing waits for
/intent/result so STOP_LISTENING cannot race the normal conversation path.
For GENERAL_CONVERSATION, the behaviour tree publishes an authorised request
on /conversation/request and then waits for generated text on
/conversation/response.
"""

from __future__ import annotations

from dataclasses import dataclass

import json
import re
import threading
import py_trees
import py_trees_ros
import rclpy
import time
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from k9_interfaces_pkg.action import CaptureFace, SpeakText
from k9_interfaces_pkg.srv import (
    CommitFaceEnrollment,
    RetrieveKnowledge,
)
from k9_interfaces_pkg.msg import (
    IntentResult,
    RecognisedFaceArray,
)
try:
    # Normal installed-package / ros2 run path.
    from k9_bt_pkg.k9_blackboard import (
        AudioMode,
        BlackboardKey,
        ChessSetupStep,
        ChessState,
        DialogueState,
        EmotionalState,
        Intent,
        K9Blackboard,
    )
except ModuleNotFoundError:
    # Convenient direct execution from the source directory.
    from k9_blackboard import (
        AudioMode,
        BlackboardKey,
        ChessSetupStep,
        ChessState,
        DialogueState,
        EmotionalState,
        Intent,
        K9Blackboard,
    )

try:
    from k9_bt_pkg.chess_behaviours import (
        BeginChessSetup,
        ChessControlCommand,
        ChessRuntimeManager,
        ContinueChessSetup,
    )
except ModuleNotFoundError:
    from chess_behaviours import (
        BeginChessSetup,
        ChessControlCommand,
        ChessRuntimeManager,
        ContinueChessSetup,
    )



# ---------------------------------------------------------------------------
# Small blackboard helpers
# ---------------------------------------------------------------------------

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


def set_emotional_event(
    blackboard: py_trees.blackboard.Client,
    *,
    state: str,
    event: str,
    trigger: str,
) -> None:
    """Set K9's emotional state and post one consumable emotional event."""

    event_id = int(
        blackboard.get(
            BlackboardKey.EMOTIONAL_EVENT_ID
        )
    ) + 1

    blackboard.set(
        BlackboardKey.EMOTIONAL_STATE,
        state,
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.EMOTIONAL_EVENT,
        event,
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.EMOTIONAL_TRIGGER,
        trigger,
        overwrite=True,
    )
    blackboard.set(
        BlackboardKey.EMOTIONAL_EVENT_ID,
        event_id,
        overwrite=True,
    )


PRAISE_PATTERNS = (
    r"\bgood (?:boy|dog|k9|k nine|kay nine)\b",
    r"\bclever (?:boy|dog|k9|k nine|kay nine)\b",
    r"\bwell done(?: k9| k nine| kay nine)?\b",
    r"\bgreat (?:job|work)(?: k9| k nine| kay nine)?\b",
    r"\bnice (?:job|work)(?: k9| k nine| kay nine)?\b",
    r"\bthank you(?: k9| k nine| kay nine)?\b",
    r"\bthanks(?: k9| k nine| kay nine)?\b",
    r"\bbrilliant(?: k9| k nine| kay nine)?\b",
    r"\bexcellent(?: k9| k nine| kay nine)?\b",
)


def is_praise(text: str) -> bool:
    """Return True for an utterance that praises or thanks K9."""

    normal = text.lower().replace("’", "'")
    normal = re.sub(
        r"[^a-z0-9'\s]",
        " ",
        normal,
    )
    normal = re.sub(
        r"\s+",
        " ",
        normal,
    ).strip()

    return any(
        re.search(
            pattern,
            normal,
        )
        is not None
        for pattern in PRAISE_PATTERNS
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

        for key in [
            BlackboardKey.EMOTIONAL_STATE,
            BlackboardKey.EMOTIONAL_EVENT,
            BlackboardKey.EMOTIONAL_EVENT_ID,
            BlackboardKey.EMOTIONAL_TRIGGER,
        ]:
            register_read_write(
                self.blackboard,
                key,
            )

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

        register_read_write(
            self.blackboard,
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
        )
        register_read_write(
            self.blackboard,
            BlackboardKey.AUDIO_DESIRED_MODE,
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

        # Voice state remains useful for diagnostics/blackboard visibility.
        self.voice_talking_subscription = node.create_subscription(
            Bool,
            "/voice/is_talking",
            self._voice_talking_callback,
            10,
        )

        # Physical back-panel requests are deliberately routed into the same
        # persistent mode state used by the BT rather than controlling the
        # hotword/STT nodes directly.
        self.mode_request_subscription = node.create_subscription(
            String,
            "/interaction/mode_request",
            self._mode_request_callback,
            10,
        )

        # Ending a live conversation from the physical panel should have the
        # same LLM-memory semantics as an explicit STOP_LISTENING intent.
        # This reset is best-effort and asynchronous so a panel button can
        # never block an audio-mode transition.
        self.conversation_reset_client = node.create_client(
            Trigger,
            "/conversation/reset",
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

        if is_praise(text):
            set_emotional_event(
                self.blackboard,
                state=EmotionalState.HAPPY,
                event="PRAISE",
                trigger=text,
            )

            self.node.get_logger().info(
                "Emotional state -> HAPPY: praise detected"
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
                "treating it as requiring a response"
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

    def _mode_request_callback(self, msg: String) -> None:
        """Apply one debounced physical back-panel mode request.

        LISTENING starts or maintains a persistent conversation without
        requiring a hotword. WAITING_FOR_HOTWORD and NOT_LISTENING both end
        any active conversation.

        The callback only changes BT/blackboard state. MaintainAudioMode then
        publishes the authoritative /audio/effective_state in the usual way.
        """

        value = msg.data.strip().upper()

        aliases = {
            "NOTLISTENING": AudioMode.NOT_LISTENING,
            "NOT_LISTENING": AudioMode.NOT_LISTENING,
            "WAITINGFORHOTWORD": AudioMode.WAITING_FOR_HOTWORD,
            "WAITING_FOR_HOTWORD": AudioMode.WAITING_FOR_HOTWORD,
            "LISTENING": AudioMode.LISTENING,
        }

        requested = aliases.get(value)

        if requested is None:
            self.node.get_logger().warning(
                f"Ignoring unknown back-panel mode request: {msg.data!r}"
            )
            return

        current = self.blackboard.get(
            BlackboardKey.AUDIO_DESIRED_MODE
        )

        was_active = bool(
            self.blackboard.get(
                BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
            )
        )

        conversation_active = (
            requested == AudioMode.LISTENING
        )

        # LISTENING while conversation_active=False is not a no-op:
        # the green button is the explicit "listen now" control.
        if (
            requested == current
            and conversation_active == was_active
        ):
            return

        # Discard any partially completed conversational turn before changing
        # the persistent interaction mode.
        clear_dialogue_turn(self.blackboard)

        # A manual panel choice supersedes any previously latched hotword.
        self.blackboard.set(
            BlackboardKey.AUDIO_HOTWORD_DETECTED,
            False,
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            conversation_active,
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.AUDIO_DESIRED_MODE,
            requested,
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.AUDIO_LAST_EVENT,
            f"BACK_PANEL_MODE_{requested}",
            overwrite=True,
        )

        self.node.get_logger().info(
            f"Back-panel mode request: {current} -> {requested}; "
            f"conversation_active={conversation_active}"
        )

        # A red/blue request that ends an active conversation should also
        # discard its LLM history. Do this asynchronously and best-effort.
        if (
            was_active
            and not conversation_active
        ):
            self._request_conversation_reset()

    def _request_conversation_reset(self) -> None:
        """Best-effort asynchronous reset of /conversation history."""

        if not self.conversation_reset_client.service_is_ready():
            self.node.get_logger().warning(
                "/conversation/reset unavailable after back-panel mode change"
            )
            return

        future = self.conversation_reset_client.call_async(
            Trigger.Request()
        )
        future.add_done_callback(
            self._conversation_reset_done
        )

    def _conversation_reset_done(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.node.get_logger().warning(
                f"Back-panel conversation reset failed: {exc}"
            )
            return

        if result.success:
            self.node.get_logger().info(
                result.message
                or "Conversation history reset after back-panel mode change"
            )
        else:
            self.node.get_logger().warning(
                result.message
                or "Conversation reset rejected after back-panel mode change"
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
            BlackboardKey.PERCEPTION_EVENT_RELATIONSHIP,
            BlackboardKey.PERCEPTION_EVENT_PREFERRED_ADDRESS,
            BlackboardKey.PERCEPTION_ERROR,            
            BlackboardKey.EMOTIONAL_STATE,
            BlackboardKey.EMOTIONAL_EVENT,
            BlackboardKey.EMOTIONAL_EVENT_ID,
            BlackboardKey.EMOTIONAL_TRIGGER,

        ]:
            register_read_write(
                self.blackboard,
                key,
            )

        perception_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.subscription = node.create_subscription(
            RecognisedFaceArray,
            "/k9/perception/recognised_faces",
            self._faces_callback,
            perception_qos,
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
                "relationship": (
                    face.relationship.strip().lower()
                    if face.recognised
                    else ""
                ),
                "preferred_address": (
                    face.preferred_address.strip()
                    if face.recognised
                    else ""
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
        relationship: str = "",
        preferred_address: str = "",
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
        self.blackboard.set(
            BlackboardKey.PERCEPTION_EVENT_RELATIONSHIP,
            relationship,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.PERCEPTION_EVENT_PREFERRED_ADDRESS,
            preferred_address,
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
        self.blackboard.set(
            BlackboardKey.PERCEPTION_EVENT_RELATIONSHIP,
            "",
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.PERCEPTION_EVENT_PREFERRED_ADDRESS,
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

            event = (
                "KNOWN_PERSON_APPEARED"
                if face["recognised"]
                else "PERSON_APPEARED"
            )

            self._set_event(
                event,
                track_id,
                face["identity"],
                face["relationship"],
                face["preferred_address"],
            )

            if (
                face["recognised"]
                and face["relationship"] == "family"
            ):
                set_emotional_event(
                    self.blackboard,
                    state=EmotionalState.HAPPY,
                    event="FAMILY_RECOGNISED",
                    trigger=face["identity"],
                )

                self.node.get_logger().info(
                    "Emotional state -> HAPPY: "
                    f"family member recognised "
                    f"({face['identity']})"
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
                        current["relationship"],
                        current["preferred_address"],
                    )

                    if (
                        current["relationship"]
                        == "family"
                    ):
                        set_emotional_event(
                            self.blackboard,
                            state=EmotionalState.HAPPY,
                            event="FAMILY_RECOGNISED",
                            trigger=current["identity"],
                        )

                        self.node.get_logger().info(
                            "Emotional state -> HAPPY: "
                            f"family member recognised "
                            f"({current['identity']})"
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
                previous["relationship"],
                previous["preferred_address"],
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
# Recognition-triggered social greeting
# ---------------------------------------------------------------------------

class KnownPersonGreetingManager(py_trees.behaviour.Behaviour):
    """Greet each recognised identity once during a conversation session.

    Recognition can start a conversation from WAITING_FOR_HOTWORD, but it
    deliberately does not override NOT_LISTENING. If another known person
    appears during an active conversation, the greeting is queued until the
    current dialogue turn and any speech have finished.

    The per-session greeted set is cleared when the conversation ends. A person
    who remains continuously visible after STOP_LISTENING therefore does not
    immediately restart the conversation; a fresh recognition transition is
    required.
    """

    def __init__(
        self,
        node: Node,
        name: str = "Known Person Greeting Manager",
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

        for key in [
            BlackboardKey.PERCEPTION_LAST_EVENT,
            BlackboardKey.PERCEPTION_EVENT_IDENTITY,
            BlackboardKey.PERCEPTION_EVENT_RELATIONSHIP,
            BlackboardKey.PERCEPTION_EVENT_PREFERRED_ADDRESS,
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            BlackboardKey.DIALOGUE_STATE,
            BlackboardKey.DIALOGUE_COMMAND,
            BlackboardKey.DIALOGUE_INTENT,
            BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            BlackboardKey.DIALOGUE_STOP_LISTENING_REQUESTED,
            BlackboardKey.DIALOGUE_ERROR,
            BlackboardKey.AUDIO_HEARD_TEXT,
            BlackboardKey.AUDIO_DESIRED_MODE,
            BlackboardKey.AUDIO_IS_TALKING,
            BlackboardKey.AUDIO_HOTWORD_DETECTED,
            BlackboardKey.AUDIO_LAST_EVENT,
        ]:
            register_read_write(
                self.blackboard,
                key,
            )

        self._greeted_identities: set[str] = set()
        self._pending_greetings: list[dict[str, str]] = []

        self._goal_future = None
        self._goal_handle = None
        self._result_future = None
        self._current_greeting = None

        self._conversation_was_active = bool(
            self.blackboard.get(
                BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
            )
        )

    def _queue_recognition_event(self) -> None:
        event = self.blackboard.get(
            BlackboardKey.PERCEPTION_LAST_EVENT
        )

        if event != "KNOWN_PERSON_APPEARED":
            return

        identity = self.blackboard.get(
            BlackboardKey.PERCEPTION_EVENT_IDENTITY
        ).strip()

        if not identity:
            return

        if identity in self._greeted_identities:
            return

        if (
            self._current_greeting is not None
            and self._current_greeting["identity"] == identity
        ):
            return

        if any(
            item["identity"] == identity
            for item in self._pending_greetings
        ):
            return

        active = bool(
            self.blackboard.get(
                BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
            )
        )

        desired_mode = self.blackboard.get(
            BlackboardKey.AUDIO_DESIRED_MODE
        )

        # An explicit NOT_LISTENING selection remains authoritative.
        if (
            not active
            and desired_mode == AudioMode.NOT_LISTENING
        ):
            self.node.get_logger().debug(
                f"Known person '{identity}' appeared while NOT_LISTENING; "
                "automatic greeting suppressed"
            )
            return

        relationship = self.blackboard.get(
            BlackboardKey.PERCEPTION_EVENT_RELATIONSHIP
        ).strip().lower()

        preferred_address = self.blackboard.get(
            BlackboardKey.PERCEPTION_EVENT_PREFERRED_ADDRESS
        ).strip()

        self._pending_greetings.append(
            {
                "identity": identity,
                "relationship": relationship,
                "preferred_address": preferred_address,
            }
        )

        self.node.get_logger().info(
            f"Queued greeting for recognised person '{identity}'"
        )

    def _handle_session_boundary(self) -> bool:
        active = bool(
            self.blackboard.get(
                BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
            )
        )

        if (
            self._conversation_was_active
            and not active
        ):
            if self._greeted_identities:
                self.node.get_logger().debug(
                    "Conversation ended; clearing greeted identities: "
                    + ", ".join(
                        sorted(self._greeted_identities)
                    )
                )

            self._greeted_identities.clear()
            self._pending_greetings.clear()

            # A recognition greeting is deliberately not allowed to resurrect a
            # conversation after STOP_LISTENING unless a fresh recognition
            # transition occurs.
            if self._current_greeting is None:
                self._reset_action_state()

        self._conversation_was_active = active
        return active

    def _dialogue_is_idle(self) -> bool:
        if self.blackboard.get(
            BlackboardKey.DIALOGUE_STATE
        ) != DialogueState.IDLE:
            return False

        if self.blackboard.get(
            BlackboardKey.DIALOGUE_COMMAND
        ).strip():
            return False

        if self.blackboard.get(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE
        ).strip():
            return False

        if bool(
            self.blackboard.get(
                BlackboardKey.AUDIO_IS_TALKING
            )
        ):
            return False

        return True

    def _start_conversation_if_needed(self) -> bool:
        active = bool(
            self.blackboard.get(
                BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
            )
        )

        if active:
            return True

        desired_mode = self.blackboard.get(
            BlackboardKey.AUDIO_DESIRED_MODE
        )

        if desired_mode == AudioMode.NOT_LISTENING:
            return False

        clear_dialogue_turn(
            self.blackboard
        )

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
        self.blackboard.set(
            BlackboardKey.AUDIO_LAST_EVENT,
            "KNOWN_PERSON_STARTED_CONVERSATION",
            overwrite=True,
        )

        self._conversation_was_active = True

        self.node.get_logger().info(
            "Recognised person started a conversation; requested LISTENING"
        )

        return True

    @staticmethod
    def _greeting_text(
        greeting: dict[str, str],
    ) -> str:
        identity = greeting["identity"]
        relationship = greeting["relationship"]
        preferred_address = greeting["preferred_address"]

        if (
            relationship == "family"
            and preferred_address
        ):
            address = preferred_address
        else:
            address = identity

        return f"Greetings, {address}."

    def _start_greeting(
        self,
        greeting: dict[str, str],
    ) -> bool:
        if not self.client.server_is_ready():
            self.feedback_message = (
                "waiting for /voice/speak before greeting"
            )
            return False

        goal = SpeakText.Goal()
        goal.text = self._greeting_text(
            greeting
        )
        goal.owner = "recognition_greeting"
        goal.priority = 90
        goal.interrupt_lower_priority = False
        goal.clear_lower_priority = False

        self._current_greeting = greeting
        self._goal_future = self.client.send_goal_async(
            goal
        )
        self._goal_handle = None
        self._result_future = None

        self.blackboard.set(
            BlackboardKey.DIALOGUE_STATE,
            DialogueState.WAITING_TO_SPEAK,
            overwrite=True,
        )

        self.feedback_message = (
            f"submitted greeting for {greeting['identity']}"
        )
        return True

    def _reset_action_state(self) -> None:
        self._goal_future = None
        self._goal_handle = None
        self._result_future = None
        self._current_greeting = None

    def _update_active_greeting(self) -> None:
        if self._current_greeting is None:
            return

        if self._goal_handle is None:
            if (
                self._goal_future is None
                or not self._goal_future.done()
            ):
                self.feedback_message = "waiting for greeting acceptance"
                return

            try:
                self._goal_handle = self._goal_future.result()
            except Exception as exc:
                self.node.get_logger().warning(
                    f"Greeting speech goal failed: {exc}"
                )
                self.blackboard.set(
                    BlackboardKey.DIALOGUE_STATE,
                    DialogueState.IDLE,
                    overwrite=True,
                )
                self._pending_greetings.pop(0)
                self._reset_action_state()
                return

            if (
                self._goal_handle is None
                or not self._goal_handle.accepted
            ):
                self.node.get_logger().warning(
                    "Greeting speech goal was rejected"
                )
                self.blackboard.set(
                    BlackboardKey.DIALOGUE_STATE,
                    DialogueState.IDLE,
                    overwrite=True,
                )
                self._pending_greetings.pop(0)
                self._reset_action_state()
                return

            self._result_future = (
                self._goal_handle.get_result_async()
            )

            self.blackboard.set(
                BlackboardKey.DIALOGUE_STATE,
                DialogueState.SPEAKING,
                overwrite=True,
            )

            self.feedback_message = (
                f"greeting {self._current_greeting['identity']}"
            )
            return

        if (
            self._result_future is None
            or not self._result_future.done()
        ):
            self.feedback_message = (
                f"greeting {self._current_greeting['identity']}"
            )
            return

        identity = self._current_greeting["identity"]

        try:
            wrapped_result = self._result_future.result()
            result = wrapped_result.result
            success = bool(
                result.success
            )
            message = result.message

        except Exception as exc:
            success = False
            message = str(exc)

        conversation_active = bool(
            self.blackboard.get(
                BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
            )
        )

        if success and conversation_active:
            self._greeted_identities.add(
                identity
            )

            self.node.get_logger().info(
                f"Greeted recognised person '{identity}'"
            )
        elif success:
            self.node.get_logger().debug(
                f"Greeting for '{identity}' completed after conversation ended; "
                "not recording it in the new-session greeted set"
            )
        else:
            self.node.get_logger().warning(
                f"Greeting for '{identity}' failed: {message}"
            )

        if self._pending_greetings:
            self._pending_greetings.pop(0)

        self.blackboard.set(
            BlackboardKey.DIALOGUE_STATE,
            DialogueState.IDLE,
            overwrite=True,
        )

        self._reset_action_state()
        self.feedback_message = (
            f"greeting complete for {identity}"
            if success
            else f"greeting failed for {identity}"
        )

    def update(self) -> py_trees.common.Status:
        # Apply a just-ended conversation boundary before consuming any fresh
        # recognition event from this BT tick.
        self._handle_session_boundary()
        self._queue_recognition_event()

        if self._current_greeting is not None:
            # If a panel action ended the conversation while a greeting was
            # already speaking, let the speech action finish but do not start
            # any further queued greeting.
            self._update_active_greeting()
            return py_trees.common.Status.RUNNING

        if not self._pending_greetings:
            self.feedback_message = (
                "no recognised person awaiting greeting"
            )
            return py_trees.common.Status.RUNNING

        if not self._dialogue_is_idle():
            self.feedback_message = (
                "recognised greeting queued; dialogue busy"
            )
            return py_trees.common.Status.RUNNING

        if not self._start_conversation_if_needed():
            self._pending_greetings.clear()
            self.feedback_message = (
                "automatic greeting suppressed by NOT_LISTENING"
            )
            return py_trees.common.Status.RUNNING

        # Starting conversation clears the transient dialogue fields; it is now
        # safe to submit a deterministic greeting without involving the LLM.
        self._start_greeting(
            self._pending_greetings[0]
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

class SendConversationRequest(
    py_trees.behaviour.Behaviour
):
    """Send the current general-conversation turn for generation."""

    def __init__(
        self,
        node: Node,
        name: str = "Send Conversation Request",
    ) -> None:
        super().__init__(name=name)

        self.node = node

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_COMMAND,
            access=py_trees.common.Access.READ,
        )

        self.publisher = node.create_publisher(
            String,
            "/conversation/request",
            10,
        )

    def update(self) -> py_trees.common.Status:
        text = self.blackboard.get(
            BlackboardKey.DIALOGUE_COMMAND
        ).strip()

        if not text:
            self.feedback_message = (
                "no conversation text"
            )
            return py_trees.common.Status.FAILURE

        payload = {
            "text": text,
            "rag_context": "",
            "rag_source": "",
            "rag_score": 0.0,
        }

        self.publisher.publish(
            String(
                data=json.dumps(
                    payload,
                    separators=(",", ":"),
                )
            )
        )

        self.feedback_message = (
            f"conversation request sent: {text!r}"
        )

        return py_trees.common.Status.SUCCESS


class RetrieveKnowledgeAndRequestConversation(
    py_trees.behaviour.Behaviour
):
    """
    Retrieve one relevant long-term memory, then ask the conversation
    node to generate K9's reply.

    RAG is optional. If the service is unavailable, fails, times out,
    or returns no document, normal conversation still proceeds.
    """

    def __init__(
        self,
        node: Node,
        name: str = "Retrieve Knowledge",
    ) -> None:

        super().__init__(
            name=name
        )

        self.node = node

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_COMMAND,
            access=py_trees.common.Access.READ,
        )

        self.rag_client = node.create_client(
            RetrieveKnowledge,
            "/k9/rag/retrieve",
        )

        self.conversation_request_pub = (
            node.create_publisher(
                String,
                "/conversation/request",
                10,
            )
        )

        self.future = None
        self.query = ""
        self.started_at = 0.0
        self.dispatched = False

        # Loading the 4B embedding model can take several seconds
        # because it is deliberately unloaded after each query.
        self.timeout_seconds = 20.0

    def initialise(self) -> None:

        self.future = None

        self.query = (
            self.blackboard.get(
                BlackboardKey.DIALOGUE_COMMAND
            )
            or ""
        ).strip()

        self.started_at = time.monotonic()
        self.dispatched = False

    def _dispatch(
        self,
        rag_context: str = "",
        rag_source: str = "",
        rag_score: float = 0.0,
        rag_metadata: str = "",
    ) -> None:

        payload = {
            "text": self.query,
            "rag_context": rag_context,
            "rag_source": rag_source,
            "rag_score": rag_score,
            "rag_metadata": rag_metadata,
        }

        message = String()

        message.data = json.dumps(
            payload,
            separators=(",", ":"),
        )

        self.conversation_request_pub.publish(
            message
        )

        self.dispatched = True

    def update(
        self,
    ) -> py_trees.common.Status:

        if not self.query:

            self.feedback_message = (
                "no conversation text"
            )

            return py_trees.common.Status.FAILURE

        if self.dispatched:

            return py_trees.common.Status.SUCCESS

        # Start the asynchronous RAG request.
        if self.future is None:

            if not self.rag_client.service_is_ready():

                self.node.get_logger().warning(
                    "RAG service unavailable; "
                    "continuing without long-term memory"
                )

                self._dispatch()

                return py_trees.common.Status.SUCCESS

            request = RetrieveKnowledge.Request()

            request.query = self.query

            # Chroma top 5 -> reranker -> top 1.
            request.max_results = 5

            self.future = (
                self.rag_client.call_async(
                    request
                )
            )

            self.feedback_message = (
                "retrieving long-term memory"
            )

            return py_trees.common.Status.RUNNING

        # Wait non-blockingly for the RAG service.
        if not self.future.done():

            elapsed = (
                time.monotonic()
                - self.started_at
            )

            if elapsed < self.timeout_seconds:

                self.feedback_message = (
                    "waiting for long-term memory"
                )

                return py_trees.common.Status.RUNNING

            self.node.get_logger().warning(
                "RAG lookup timed out; "
                "continuing without long-term memory"
            )

            self._dispatch()

            return py_trees.common.Status.SUCCESS

        try:

            result = self.future.result()

        except Exception as exc:

            self.node.get_logger().warning(
                f"RAG lookup failed: {exc}; "
                "continuing without long-term memory"
            )

            self._dispatch()

            return py_trees.common.Status.SUCCESS

        if (
            result is None
            or not result.success
        ):

            error = (
                result.error
                if result is not None
                else "no result"
            )

            self.node.get_logger().warning(
                f"RAG lookup failed: {error}; "
                "continuing without long-term memory"
            )

            self._dispatch()

            return py_trees.common.Status.SUCCESS

        # No sufficiently relevant memory.
        if not result.documents:

            self.node.get_logger().info(
                "RAG: no relevant long-term memory "
                f"for {self.query!r}"
            )

            self._dispatch()

            return py_trees.common.Status.SUCCESS

        # k9_rag deliberately returns no more than one
        # reranked document.
        document = result.documents[0]

        source = (
            result.sources[0]
            if result.sources
            else ""
        )

        score = (
            float(result.scores[0])
            if result.scores
            else 0.0
        )

        metadata = (
            result.metadata_json[0]
            if result.metadata_json
            else ""
        )

        self.node.get_logger().info(
            "RAG memory selected: "
            f"source={source or 'unknown'}, "
            f"score={score:.3f}"
        )

        self._dispatch(
            rag_context=document,
            rag_source=source,
            rag_score=score,
            rag_metadata=metadata,
        )

        self.feedback_message = (
            "conversation request dispatched "
            "with long-term memory"
        )

        return py_trees.common.Status.SUCCESS

    def terminate(
        self,
        new_status: py_trees.common.Status,
    ) -> None:

        if (
            new_status
            == py_trees.common.Status.INVALID
            and self.future is not None
            and not self.future.done()
        ):

            try:
                self.future.cancel()
            except Exception:
                pass

class WaitForConversationResponse(py_trees.behaviour.Behaviour):
    """Wait for /conversation to populate the pending LLM response."""

    def __init__(
        self,
        name: str = "Wait For Conversation Response",
        timeout_seconds: float = 35.0,
    ) -> None:
        super().__init__(name=name)

        self.timeout_seconds = float(
            timeout_seconds
        )

        # None means that this particular wait has not started yet.
        self.started_at = None

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        register_read_write(
            self.blackboard,
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
        )

        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_INTENT,
            access=py_trees.common.Access.READ,
        )

        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
            access=py_trees.common.Access.READ,
        )

        register_read_write(
            self.blackboard,
            BlackboardKey.DIALOGUE_STATE,
        )

    def initialise(self) -> None:
        # Start a fresh timer every time this behaviour is entered.
        self.started_at = None

    def update(self) -> py_trees.common.Status:
        if not self.blackboard.get(
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
        ):
            self.feedback_message = (
                "conversation no longer active"
            )
            return py_trees.common.Status.FAILURE

        intent = self.blackboard.get(
            BlackboardKey.DIALOGUE_INTENT
        )

        if intent != Intent.GENERAL_CONVERSATION:
            self.feedback_message = (
                f"not a general conversation turn: {intent}"
            )
            return py_trees.common.Status.FAILURE

        response = self.blackboard.get(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE
        ).strip()

        # A response may already have arrived between BT ticks.
        if response:
            self.started_at = None
            self.feedback_message = response
            return py_trees.common.Status.SUCCESS

        # Start timing only when we genuinely begin waiting.
        if self.started_at is None:
            self.started_at = time.monotonic()

            self.feedback_message = (
                "waiting for /conversation/response"
            )

            return py_trees.common.Status.RUNNING

        elapsed = (
            time.monotonic()
            - self.started_at
        )

        if elapsed < self.timeout_seconds:
            self.feedback_message = (
                "waiting for /conversation/response "
                f"({elapsed:.1f}s)"
            )

            return py_trees.common.Status.RUNNING

        fallback = (
            "Apologies. My response generator did not answer."
        )

        self.blackboard.set(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            fallback,
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.DIALOGUE_STATE,
            DialogueState.WAITING_TO_SPEAK,
            overwrite=True,
        )

        self.feedback_message = (
            f"conversation response timed out "
            f"after {elapsed:.1f}s"
        )

        self.started_at = None

        return py_trees.common.Status.SUCCESS

    def terminate(
        self,
        new_status: py_trees.common.Status,
    ) -> None:
        if new_status != py_trees.common.Status.RUNNING:
            self.started_at = None

class GenerateUnsupportedIntentResponse(py_trees.behaviour.Behaviour):
    """Provide a safe fallback for recognised but unhandled intents.

    PLAY_CHESS and CHESS_SETUP_ANSWER are handled by dedicated chess branches.
    This leaf is only a final guard so an unexpected executive intent cannot
    leave the active conversation permanently stuck.
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
        self.warned_unavailable = False
        self.started_at = time.monotonic()

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


class FaceEnrollmentDialogue(py_trees.behaviour.Behaviour):
    """Run K9's spoken face-enrolment dialogue as one persistent BT leaf.

    The leaf is entered after the initial ENROL_FACE intent. It then publishes
    /intent/context so subsequent STT utterances are classified as enrolment
    answers rather than GENERAL_CONVERSATION.

    Face capture itself is delegated to the face_recogniser action server;
    this behaviour owns only dialogue, sequencing, retry/cancel policy and the
    final metadata commit.
    """

    FRONT_SAMPLES = 4
    LEFT_SAMPLES = 3
    RIGHT_SAMPLES = 3
    MAX_CAPTURE_ATTEMPTS = 2

    def __init__(
        self,
        *,
        node: Node,
        name: str = "Face Enrolment Dialogue",
    ) -> None:
        super().__init__(name=name)

        self.node = node

        self.speech_client = ActionClient(
            node,
            SpeakText,
            "/voice/speak",
        )
        self.capture_client = ActionClient(
            node,
            CaptureFace,
            "/face_recogniser/capture_face",
        )
        self.commit_client = node.create_client(
            CommitFaceEnrollment,
            "/face_recogniser/commit_enrolment",
        )
        self.discard_client = node.create_client(
            Trigger,
            "/face_recogniser/discard_enrolment",
        )
        self.intent_context_publisher = node.create_publisher(
            String,
            "/intent/context",
            10,
        )

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        for key in [
            BlackboardKey.DIALOGUE_COMMAND,
            BlackboardKey.DIALOGUE_INTENT,
            BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
            BlackboardKey.DIALOGUE_STATE,
            BlackboardKey.DIALOGUE_ERROR,
            BlackboardKey.AUDIO_HEARD_TEXT,
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
        ]:
            register_read_write(
                self.blackboard,
                key,
            )

        self.stage = "IDLE"
        self.next_stage = ""

        self.identity = ""
        self.relationship = ""
        self.preferred_address = ""

        self.speech_goal_future = None
        self.speech_goal_handle = None
        self.speech_result_future = None

        self.capture_goal_future = None
        self.capture_goal_handle = None
        self.capture_result_future = None
        self.capture_pose = ""
        self.capture_samples = 0
        self.capture_attempts = {}
        self.capture_feedback_state = ""

        self.commit_future = None
        self.discard_future = None
        self.end_message = ""

    # ------------------------------------------------------------------
    # Lifecycle / small helpers
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        self.stage = "IDLE"
        self.next_stage = ""

        self.identity = ""
        self.relationship = ""
        self.preferred_address = ""

        self.speech_goal_future = None
        self.speech_goal_handle = None
        self.speech_result_future = None

        self.capture_goal_future = None
        self.capture_goal_handle = None
        self.capture_result_future = None
        self.capture_pose = ""
        self.capture_samples = 0
        self.capture_attempts = {}
        self.capture_feedback_state = ""

        self.commit_future = None
        self.discard_future = None
        self.end_message = ""

        # Consume the original "remember me" turn. The surrounding sequence
        # is memory=True, so later ENROL_FACE_ANSWER intents will not make it
        # re-check the initial IsIntent leaf.
        self._consume_input()

        self._start_speech(
            "Certainly. What is your name?",
            "WAIT_NAME",
        )

    def terminate(
        self,
        new_status: py_trees.common.Status,
    ) -> None:
        if new_status == py_trees.common.Status.RUNNING:
            return

        self._publish_intent_context("")

        if (
            self.capture_goal_handle is not None
            and self.capture_result_future is not None
            and not self.capture_result_future.done()
        ):
            try:
                self.capture_goal_handle.cancel_goal_async()
            except Exception:
                pass

    def _publish_intent_context(
        self,
        enrolment_state: str,
    ) -> None:
        if enrolment_state:
            payload = {
                "enrolment_state": enrolment_state,
            }
        else:
            payload = {}

        self.intent_context_publisher.publish(
            String(
                data=json.dumps(
                    payload,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        )

    def _consume_input(self) -> None:
        self.blackboard.set(
            BlackboardKey.DIALOGUE_COMMAND,
            "",
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_INTENT,
            Intent.NONE,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
            0.0,
            overwrite=True,
        )
        self.blackboard.set(
            BlackboardKey.AUDIO_HEARD_TEXT,
            "",
            overwrite=True,
        )

    def _current_input(self):
        command = self.blackboard.get(
            BlackboardKey.DIALOGUE_COMMAND
        ).strip()
        intent = self.blackboard.get(
            BlackboardKey.DIALOGUE_INTENT
        )
        return intent, command

    @staticmethod
    def _normalise_text(text: str) -> str:
        normal = text.lower().replace("’", "'")
        normal = re.sub(r"[^a-z0-9'\s-]", " ", normal)
        normal = normal.replace("-", " ")
        return re.sub(r"\s+", " ", normal).strip()

    @staticmethod
    def _extract_name(text: str) -> str:
        cleaned = text.strip(" \t\r\n.,!?")

        patterns = [
            r"(?i)\bmy name is\s+([a-z][a-z' -]{0,40})$",
            r"(?i)\bi am\s+([a-z][a-z' -]{0,40})$",
            r"(?i)\bi'm\s+([a-z][a-z' -]{0,40})$",
            r"(?i)\bcall me\s+([a-z][a-z' -]{0,40})$",
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                return FaceEnrollmentDialogue._title_words(
                    match.group(1)
                )

        if "?" not in text:
            words = re.findall(
                r"[A-Za-z][A-Za-z'-]*",
                cleaned,
            )
            if 1 <= len(words) <= 3 and len(cleaned) <= 40:
                return FaceEnrollmentDialogue._title_words(
                    " ".join(words)
                )

        return ""

    @staticmethod
    def _extract_relationship(text: str) -> str:
        normal = FaceEnrollmentDialogue._normalise_text(text)

        has_family = re.search(
            r"\b(?:family|relative|relation)\b",
            normal,
        ) is not None
        has_friend = re.search(
            r"\bfriend\b",
            normal,
        ) is not None

        if has_family and not has_friend:
            return "family"
        if has_friend and not has_family:
            return "friend"
        return ""

    @staticmethod
    def _extract_preferred_address(text: str) -> str:
        cleaned = text.strip(" \t\r\n.,!?")

        patterns = [
            r"(?i)^call me\s+(.+)$",
            r"(?i)^address me as\s+(.+)$",
            r"(?i)^you can call me\s+(.+)$",
            r"(?i)^please call me\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, cleaned)
            if match:
                cleaned = match.group(1).strip(" .,!?")
                break

        words = re.findall(
            r"[A-Za-z][A-Za-z'-]*",
            cleaned,
        )

        if not (1 <= len(words) <= 4):
            return ""

        value = " ".join(words)
        if len(value) > 40:
            return ""

        return FaceEnrollmentDialogue._title_words(value)

    @staticmethod
    def _title_words(text: str) -> str:
        return " ".join(
            part.capitalize()
            for part in text.split()
        )

    def _address_for_speech(self) -> str:
        if self.relationship == "family":
            return self.preferred_address or self.identity
        return self.identity

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------

    def _start_speech(
        self,
        text: str,
        next_stage: str,
    ) -> None:
        self._publish_intent_context("SPEAKING")

        self.stage = "SPEAKING"
        self.next_stage = next_stage

        self.speech_goal_future = None
        self.speech_goal_handle = None
        self.speech_result_future = None

        if not self.speech_client.server_is_ready():
            self.blackboard.set(
                BlackboardKey.DIALOGUE_ERROR,
                "voice action server unavailable during face enrolment",
                overwrite=True,
            )
            self.stage = "FINISHED"
            return

        goal = SpeakText.Goal()
        goal.text = text
        goal.owner = "face_enrolment"
        goal.priority = 100
        goal.interrupt_lower_priority = True
        goal.clear_lower_priority = True

        self.speech_goal_future = self.speech_client.send_goal_async(
            goal
        )

        self.feedback_message = text

    def _update_speech(self) -> None:
        if self.speech_goal_future is None:
            self.stage = "FINISHED"
            return

        if self.speech_goal_handle is None:
            if not self.speech_goal_future.done():
                return

            try:
                self.speech_goal_handle = self.speech_goal_future.result()
            except Exception as exc:
                self.blackboard.set(
                    BlackboardKey.DIALOGUE_ERROR,
                    f"enrolment speech goal failed: {exc}",
                    overwrite=True,
                )
                self.stage = "FINISHED"
                return

            if (
                self.speech_goal_handle is None
                or not self.speech_goal_handle.accepted
            ):
                self.blackboard.set(
                    BlackboardKey.DIALOGUE_ERROR,
                    "enrolment speech goal rejected",
                    overwrite=True,
                )
                self.stage = "FINISHED"
                return

            self.speech_result_future = (
                self.speech_goal_handle.get_result_async()
            )
            self.blackboard.set(
                BlackboardKey.DIALOGUE_STATE,
                DialogueState.SPEAKING,
                overwrite=True,
            )
            return

        if (
            self.speech_result_future is None
            or not self.speech_result_future.done()
        ):
            return

        try:
            wrapped = self.speech_result_future.result()
            result = wrapped.result
        except Exception as exc:
            self.blackboard.set(
                BlackboardKey.DIALOGUE_ERROR,
                f"enrolment speech failed: {exc}",
                overwrite=True,
            )
            self.stage = "FINISHED"
            return

        if not result.success:
            self.blackboard.set(
                BlackboardKey.DIALOGUE_ERROR,
                result.message,
                overwrite=True,
            )
            self.stage = "FINISHED"
            return

        self.stage = self.next_stage
        self.next_stage = ""

        if self.stage.startswith("WAIT_"):
            self._publish_intent_context(self.stage)
            self.blackboard.set(
                BlackboardKey.DIALOGUE_STATE,
                DialogueState.IDLE,
                overwrite=True,
            )
        elif self.stage.startswith("START_"):
            self._publish_intent_context("CAPTURING")
        elif self.stage == "FINISHED":
            self._publish_intent_context("")

    # ------------------------------------------------------------------
    # Face capture action
    # ------------------------------------------------------------------

    def _start_capture(
        self,
        pose: str,
        samples: int,
    ) -> None:
        if not self.capture_client.server_is_ready():
            self._begin_end(
                "My face recognition system is unavailable at present."
            )
            return

        self.capture_pose = pose
        self.capture_samples = samples
        self.capture_feedback_state = ""

        self.capture_attempts[pose] = (
            self.capture_attempts.get(pose, 0) + 1
        )

        goal = CaptureFace.Goal()
        goal.identity = self.identity
        goal.pose = pose
        goal.samples_required = int(samples)

        self.capture_goal_future = self.capture_client.send_goal_async(
            goal,
            feedback_callback=self._capture_feedback_callback,
        )
        self.capture_goal_handle = None
        self.capture_result_future = None
        self.stage = "CAPTURING"
        self._publish_intent_context("CAPTURING")

        self.feedback_message = (
            f"capturing {pose} face samples "
            f"attempt {self.capture_attempts[pose]}"
        )

    def _capture_feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self.capture_feedback_state = feedback.state

    def _retry_capture_message(
        self,
        pose: str,
        failure_message: str,
    ) -> str:
        if pose == "front":
            instruction = "Please look directly at me and we shall try again."
        elif pose == "left":
            instruction = (
                "Please turn your head slightly to your left and we shall "
                "try again."
            )
        else:
            instruction = (
                "Please turn your head slightly to your right and we shall "
                "try again."
            )

        return f"{failure_message} {instruction}".strip()

    def _update_capture(self) -> None:
        if self.capture_goal_handle is None:
            if not self.capture_goal_future.done():
                return

            try:
                self.capture_goal_handle = self.capture_goal_future.result()
            except Exception as exc:
                self._begin_end(
                    f"Face capture failed: {exc}"
                )
                return

            if (
                self.capture_goal_handle is None
                or not self.capture_goal_handle.accepted
            ):
                self._begin_end(
                    "My face recognition system rejected the capture request."
                )
                return

            self.capture_result_future = (
                self.capture_goal_handle.get_result_async()
            )
            return

        if (
            self.capture_result_future is None
            or not self.capture_result_future.done()
        ):
            return

        try:
            wrapped = self.capture_result_future.result()
            result = wrapped.result
        except Exception as exc:
            self._begin_end(
                f"Face capture failed: {exc}"
            )
            return

        pose = self.capture_pose

        self.capture_goal_future = None
        self.capture_goal_handle = None
        self.capture_result_future = None

        if result.success:
            if pose == "front":
                self._start_speech(
                    "Good. Now turn your head slightly to your left.",
                    "START_LEFT",
                )
            elif pose == "left":
                self._start_speech(
                    "Thank you. Now turn slightly to your right.",
                    "START_RIGHT",
                )
            else:
                self.stage = "START_COMMIT"
                self._publish_intent_context("COMMITTING")
            return

        attempts = self.capture_attempts.get(pose, 1)

        if attempts < self.MAX_CAPTURE_ATTEMPTS:
            retry_stage = {
                "front": "START_FRONT",
                "left": "START_LEFT",
                "right": "START_RIGHT",
            }[pose]

            self._start_speech(
                self._retry_capture_message(
                    pose,
                    result.message,
                ),
                retry_stage,
            )
            return

        self._begin_end(
            result.message
            or "I am unable to obtain suitable face samples at present."
        )

    # ------------------------------------------------------------------
    # Commit / discard
    # ------------------------------------------------------------------

    def _start_commit(self) -> None:
        if not self.commit_client.service_is_ready():
            self._begin_end(
                "I cannot save the face enrolment at present."
            )
            return

        request = CommitFaceEnrollment.Request()
        request.identity = self.identity
        request.relationship = self.relationship
        request.preferred_address = self.preferred_address

        self.commit_future = self.commit_client.call_async(
            request
        )
        self.stage = "COMMITTING"
        self._publish_intent_context("COMMITTING")

    def _update_commit(self) -> None:
        if self.commit_future is None or not self.commit_future.done():
            return

        try:
            result = self.commit_future.result()
        except Exception as exc:
            self._begin_end(
                f"I could not save your identity: {exc}"
            )
            return

        self.commit_future = None

        if not result.success:
            self._begin_end(
                result.message
                or "I could not save your identity."
            )
            return

        address = self._address_for_speech()
        self._start_speech(
            f"Excellent. I shall remember you, {address}.",
            "FINISHED",
        )

    def _begin_end(self, message: str) -> None:
        """Cancel any live capture, discard staging, then speak message."""

        self.end_message = message
        self._consume_input()
        self._publish_intent_context("CANCELLING")

        if (
            self.capture_goal_handle is not None
            and self.capture_result_future is not None
            and not self.capture_result_future.done()
        ):
            try:
                self.capture_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.stage = "END_WAIT_CAPTURE"
            return

        if (
            self.capture_goal_future is not None
            and not self.capture_goal_future.done()
        ):
            self.stage = "END_WAIT_GOAL"
            return

        self._start_discard()

    def _update_end_wait_goal(self) -> None:
        if not self.capture_goal_future.done():
            return

        try:
            self.capture_goal_handle = self.capture_goal_future.result()
        except Exception:
            self.capture_goal_handle = None

        if (
            self.capture_goal_handle is not None
            and self.capture_goal_handle.accepted
        ):
            try:
                self.capture_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.capture_result_future = (
                self.capture_goal_handle.get_result_async()
            )
            self.stage = "END_WAIT_CAPTURE"
            return

        self._start_discard()

    def _update_end_wait_capture(self) -> None:
        if (
            self.capture_result_future is not None
            and not self.capture_result_future.done()
        ):
            return

        self.capture_goal_future = None
        self.capture_goal_handle = None
        self.capture_result_future = None
        self._start_discard()

    def _start_discard(self) -> None:
        if not self.discard_client.service_is_ready():
            self._start_speech(
                self.end_message,
                "FINISHED",
            )
            return

        self.discard_future = self.discard_client.call_async(
            Trigger.Request()
        )
        self.stage = "DISCARDING"

    def _update_discard(self) -> None:
        if self.discard_future is None or not self.discard_future.done():
            return

        try:
            result = self.discard_future.result()
            if not result.success:
                self.node.get_logger().warning(
                    result.message
                    or "Face enrolment staging could not be discarded"
                )
        except Exception as exc:
            self.node.get_logger().warning(
                f"Face enrolment discard failed: {exc}"
            )

        self.discard_future = None
        self._start_speech(
            self.end_message,
            "FINISHED",
        )

    # ------------------------------------------------------------------
    # BT update
    # ------------------------------------------------------------------

    def update(self) -> py_trees.common.Status:
        if not self.blackboard.get(
            BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE
        ):
            self.feedback_message = "conversation ended during enrolment"
            self._publish_intent_context("")
            return py_trees.common.Status.SUCCESS

        current_intent, current_command = self._current_input()

        if (
            current_intent == "ENROL_FACE_CANCEL"
            and current_command
            and not self.stage.startswith("END_")
            and self.stage not in {"DISCARDING", "FINISHED"}
        ):
            self._begin_end(
                "Very well. Enrolment cancelled."
            )

        if self.stage == "SPEAKING":
            self._update_speech()

        elif self.stage == "WAIT_NAME":
            if current_command:
                name = self._extract_name(current_command)
                self._consume_input()

                if not name:
                    self._start_speech(
                        "I did not catch your name. Please give me just your name.",
                        "WAIT_NAME",
                    )
                else:
                    self.identity = name
                    self._start_speech(
                        "Are you a member of my family, or a friend?",
                        "WAIT_RELATIONSHIP",
                    )

        elif self.stage == "WAIT_RELATIONSHIP":
            if current_command:
                relationship = self._extract_relationship(
                    current_command
                )
                self._consume_input()

                if not relationship:
                    self._start_speech(
                        "Please say family or friend.",
                        "WAIT_RELATIONSHIP",
                    )
                elif relationship == "family":
                    self.relationship = "family"
                    self._start_speech(
                        "And how should I address you?",
                        "WAIT_ADDRESS",
                    )
                else:
                    self.relationship = "friend"
                    self.preferred_address = self.identity
                    self._start_speech(
                        f"Very good, {self.identity}. "
                        "Please look directly at me.",
                        "START_FRONT",
                    )

        elif self.stage == "WAIT_ADDRESS":
            if current_command:
                preferred_address = self._extract_preferred_address(
                    current_command
                )
                self._consume_input()

                if not preferred_address:
                    self._start_speech(
                        "I did not catch that. How should I address you?",
                        "WAIT_ADDRESS",
                    )
                else:
                    self.preferred_address = preferred_address
                    self._start_speech(
                        f"Very good, {preferred_address}. "
                        "Please look directly at me.",
                        "START_FRONT",
                    )

        elif self.stage == "START_FRONT":
            self._start_capture(
                "front",
                self.FRONT_SAMPLES,
            )

        elif self.stage == "START_LEFT":
            self._start_capture(
                "left",
                self.LEFT_SAMPLES,
            )

        elif self.stage == "START_RIGHT":
            self._start_capture(
                "right",
                self.RIGHT_SAMPLES,
            )

        elif self.stage == "CAPTURING":
            self._update_capture()

        elif self.stage == "START_COMMIT":
            self._start_commit()

        elif self.stage == "COMMITTING":
            self._update_commit()

        elif self.stage == "END_WAIT_GOAL":
            self._update_end_wait_goal()

        elif self.stage == "END_WAIT_CAPTURE":
            self._update_end_wait_capture()

        elif self.stage == "DISCARDING":
            self._update_discard()

        elif self.stage == "FINISHED":
            self._publish_intent_context("")
            self.blackboard.set(
                BlackboardKey.DIALOGUE_STATE,
                DialogueState.IDLE,
                overwrite=True,
            )
            self.feedback_message = "face enrolment complete"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = (
            f"face enrolment: {self.stage}"
            + (
                f" ({self.capture_feedback_state})"
                if self.capture_feedback_state
                else ""
            )
        )
        return py_trees.common.Status.RUNNING


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
    """End the conversation after an explicit STOP_LISTENING intent.

    In addition to returning audio to WAITING_FOR_HOTWORD, this clears any
    incomplete chess setup dialogue and removes /intent/context.  Without this
    cleanup an abandoned "Who am I playing?" prompt can cause utterances in the
    next conversation to be misclassified as CHESS_SETUP_ANSWER.
    """

    def __init__(
        self,
        *,
        node: Node,
        name: str = "End Conversation",
    ) -> None:
        super().__init__(name=name)

        self.node = node

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
            BlackboardKey.CHESS_STATE,
            BlackboardKey.CHESS_SETUP_STEP,
        ]:
            register_read_write(
                self.blackboard,
                key,
            )

        self.intent_context_publisher = node.create_publisher(
            String,
            "/intent/context",
            10,
        )

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

        # A STOP_LISTENING command abandons any incomplete spoken chess setup.
        # An already active chess game is left alone; only the SETUP state is
        # reset.
        chess_state = str(
            self.blackboard.get(
                BlackboardKey.CHESS_STATE
            )
            or ""
        ).upper()

        if chess_state in {
            "SETUP",
            "SETTING_UP",
        }:
            self.blackboard.set(
                BlackboardKey.CHESS_SETUP_STEP,
                ChessSetupStep.NONE,
                overwrite=True,
            )
            self.blackboard.set(
                BlackboardKey.CHESS_STATE,
                ChessState.IDLE,
                overwrite=True,
            )

        # Clear contextual intent classification regardless of chess state.
        # This also safely removes stale WAIT_NAME/WAIT_COLOUR context.
        self.intent_context_publisher.publish(
            String(data="{}")
        )

        self.feedback_message = (
            "STOP_LISTENING; cleared setup context; waiting for hotword"
        )
        return py_trees.common.Status.SUCCESS


class DialogueIdle(py_trees.behaviour.Behaviour):
    """Visible idle leaf while no persistent conversation is active."""

    def __init__(self, name: str = "Dialogue Idle") -> None:
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )
        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_DESIRED_MODE,
            access=py_trees.common.Access.READ,
        )

    def update(self) -> py_trees.common.Status:
        mode = self.blackboard.get(
            BlackboardKey.AUDIO_DESIRED_MODE
        )

        if mode == AudioMode.NOT_LISTENING:
            self.feedback_message = "not listening"
        elif mode == AudioMode.WAITING_FOR_HOTWORD:
            self.feedback_message = "waiting for hotword"
        else:
            self.feedback_message = f"idle; desired mode={mode}"

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

    # PROCESSING and SPEAKING are temporary interaction activities.
    # k9_stt inhibits recognition from /interaction/activity, so the
    # persistent effective mode can remain LISTENING throughout a conversation.

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


def create_known_person_greeting_manager(
    node: Node,
) -> py_trees.behaviour.Behaviour:
    return KnownPersonGreetingManager(node)


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
            EndConversation(node=node),
        ]
    )

    chess_resign = py_trees.composites.Sequence(
        name="Chess Resign",
        memory=True,
    )
    chess_resign.add_children(
        [
            IsIntent("CHESS_RESIGN"),
            ChessControlCommand(
                node=node,
                command="HUMAN_RESIGN",
                name="Accept Human Resignation",
            ),
            ClearConversationTurn(),
        ]
    )

    chess_draw_offer = py_trees.composites.Sequence(
        name="Chess Draw Offer",
        memory=True,
    )
    chess_draw_offer.add_children(
        [
            IsIntent("CHESS_DRAW_OFFER"),
            ChessControlCommand(
                node=node,
                command="OFFER_DRAW",
                name="Consider Draw Offer",
            ),
            ClearConversationTurn(),
        ]
    )

    # Face enrolment is a stateful spoken workflow. Once entered, the sequence
    # remains on FaceEnrollmentDialogue while the intent node classifies the
    # user's name / relationship / preferred-address answers contextually.
    face_enrolment = py_trees.composites.Sequence(
        name="Face Enrolment",
        memory=True,
    )
    face_enrolment.add_children(
        [
            IsIntent("ENROL_FACE"),
            FaceEnrollmentDialogue(node=node),
            ClearConversationTurn(),
        ]
    )

    # Chess setup is split over separate conversational turns. If the
    # player is already known from perception, PLAY_CHESS arms immediately.
    # Otherwise K9 asks the name and returns control so unrelated conversation
    # remains available while waiting for CHESS_SETUP_ANSWER.
    play_chess = py_trees.composites.Sequence(
        name="Play Chess",
        memory=True,
    )
    play_chess.add_children(
        [
            IsIntent(Intent.PLAY_CHESS),
            BeginChessSetup(node=node),
            ClearConversationTurn(),
        ]
    )

    chess_setup_answer = py_trees.composites.Sequence(
        name="Chess Setup Answer",
        memory=True,
    )
    chess_setup_answer.add_children(
        [
            IsIntent(Intent.CHESS_SETUP_ANSWER),
            ContinueChessSetup(node=node),
            ClearConversationTurn(),
        ]
    )

    # Normal conversational turn. The intent result is owned by the BT.
    # Retrieve optional long-term memory first, then submit one authorised
    # request to the conversation node. RAG failure is deliberately non-fatal:
    # normal conversation proceeds without memory if retrieval is unavailable.
    general_conversation = py_trees.composites.Sequence(
        name="General Conversation",
        memory=True,
    )
    general_conversation.add_children(
        [
            IsIntent(Intent.GENERAL_CONVERSATION),
            RetrieveKnowledgeAndRequestConversation(
                node=node
            ),
            SendConversationRequest(node=node),
            WaitForConversationResponse(),
            SpeakPendingResponse(node=node),
            ClearConversationTurn(),
        ]
    )
    # Final guard for any recognised executive intent not handled above.
    # Do not wait for an LLM response that the conversation node will never
    # publish for those intents.
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
            chess_resign,
            chess_draw_offer,
            face_enrolment,
            play_chess,
            chess_setup_answer,
            general_conversation,
            unimplemented_intent,
        ]
    )

    # While a conversation is active this branch remains RUNNING in
    # WaitForCommand, then handles exactly one interpreted utterance.
    conversation = py_trees.composites.Sequence(
        name="Active Conversation",
        memory=False,
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


def create_chess_manager(
    node: Node,
) -> py_trees.behaviour.Behaviour:
    """Run chess as a non-blocking overlay on normal K9 behaviour."""
    return ChessRuntimeManager(node)

class EmotionalExpressionManager(
    py_trees.behaviour.Behaviour
):
    """Turn new emotional events into physical K9 expressions."""

    def __init__(
        self,
        node: Node,
        name: str = "Emotional Expression Manager",
    ) -> None:
        super().__init__(name=name)

        self.node = node

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        for key in [
            BlackboardKey.EMOTIONAL_STATE,
            BlackboardKey.EMOTIONAL_EVENT,
            BlackboardKey.EMOTIONAL_EVENT_ID,
            BlackboardKey.EMOTIONAL_TRIGGER,
        ]:
            self.blackboard.register_key(
                key=key,
                access=py_trees.common.Access.READ,
            )

        self.tail_up_client = node.create_client(
            Trigger,
            "/tail_up",
        )

        self.tail_wag_v_client = node.create_client(
            Trigger,
            "/tail_wag_v",
        )

        # Event zero is the initial/default state and must not cause motion.
        self._handled_event_id = 0

    def _call_tail_service(
        self,
        client,
        service_name: str,
    ) -> bool:
        if not client.service_is_ready():
            self.feedback_message = (
                f"waiting for {service_name}"
            )
            return False

        future = client.call_async(
            Trigger.Request()
        )

        future.add_done_callback(
            lambda completed:
            self._tail_service_done(
                completed,
                service_name,
            )
        )

        return True

    def _tail_service_done(
        self,
        future,
        service_name: str,
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.node.get_logger().warning(
                f"{service_name} failed: {exc}"
            )
            return

        if result.success:
            self.node.get_logger().info(
                f"Expression completed: {service_name}"
            )
        else:
            self.node.get_logger().warning(
                result.message
                or f"{service_name} rejected"
            )

    def update(self) -> py_trees.common.Status:
        event_id = int(
            self.blackboard.get(
                BlackboardKey.EMOTIONAL_EVENT_ID
            )
        )

        if event_id == self._handled_event_id:
            self.feedback_message = "no new emotional event"
            return py_trees.common.Status.RUNNING

        event = self.blackboard.get(
            BlackboardKey.EMOTIONAL_EVENT
        )

        trigger = self.blackboard.get(
            BlackboardKey.EMOTIONAL_TRIGGER
        )

        if event == "FAMILY_RECOGNISED":
            accepted = self._call_tail_service(
                self.tail_up_client,
                "/tail_up",
            )

        elif event == "PRAISE":
            accepted = self._call_tail_service(
                self.tail_wag_v_client,
                "/tail_wag_v",
            )

        else:
            # Unknown emotional events do not block future events.
            self._handled_event_id = event_id
            self.feedback_message = (
                f"no expression for {event}"
            )
            return py_trees.common.Status.RUNNING

        # If the Pi service is temporarily unavailable, retain the event and
        # try it again on the next BT tick.
        if not accepted:
            return py_trees.common.Status.RUNNING

        self._handled_event_id = event_id

        self.feedback_message = (
            f"{event}: {trigger}"
        )

        return py_trees.common.Status.RUNNING

def create_expression_manager(
    node: Node,
) -> py_trees.behaviour.Behaviour:
    return EmotionalExpressionManager(
        node
    )

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
            create_known_person_greeting_manager(node),
            create_dialogue_manager(node),
            create_chess_manager(node),
            create_expression_manager(node),
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
            "/interaction/activity inhibits STT while processing/speaking; "
            "back-panel mode requests supported"
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
