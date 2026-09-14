#!/usr/bin/env python3
"""Chess behaviours for the central K9 behaviour tree.

Chess is deliberately an overlay on K9's persistent conversation rather than a
separate operating mode.  This module provides:

* BeginChessSetup
    Handles PLAY_CHESS.  It uses an attended/uniquely recognised face when one
    is available, otherwise asks who K9 is playing, then asks the chess
    manager to initialise the directly connected Phantom board.

* ContinueChessSetup
    Handles a later CHESS_SETUP_ANSWER without monopolising the conversation
    branch while K9 is waiting for the answer.

* ChessRuntimeManager
    Mirrors /chess/status onto the K9 blackboard and reacts to /chess/event.
    It never blocks normal conversation.  Chess move announcements use lower
    speech priorities than ordinary dialogue, so conversation can pre-empt
    them.

The chess subsystem remains authoritative for board facts.  These behaviours
only coordinate executive state, speech and simple physical reactions.
"""

from __future__ import annotations

from collections import deque
import json
import re
import threading
import time
from typing import Deque, Optional

import py_trees
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
from std_srvs.srv import Trigger

from k9_interfaces_pkg.action import SpeakText
from k9_interfaces_pkg.msg import ChessEvent, ChessStatus
from k9_interfaces_pkg.srv import ControlChessGame, StartChessGame

try:
    from k9_bt_pkg.k9_blackboard import (
        BlackboardKey,
        ChessSetupStep,
        ChessState,
        DialogueState,
        Intent,
    )
except ModuleNotFoundError:
    from k9_blackboard import (
        BlackboardKey,
        ChessSetupStep,
        ChessState,
        DialogueState,
        Intent,
    )


# Chess setup speech should yield to ordinary dialogue and urgent social speech.
CHESS_SETUP_SPEECH_PRIORITY = 80

# Move instructions should be audible but must not monopolise conversation.
CHESS_MOVE_SPEECH_PRIORITY = 50

# "Your move" is useful but deliberately very low priority.
CHESS_TURN_SPEECH_PRIORITY = 20

# End-of-game personality speech should normally be heard, while still
# yielding to higher-priority executive dialogue.
CHESS_RESULT_SPEECH_PRIORITY = 70

# "Check" and especially "Checkmate" are chess facts, not personality.
# Give them a slightly higher priority than result commentary so a generated
# reaction can never obscure the deterministic declaration.
CHESS_DECLARATION_SPEECH_PRIORITY = 75

CHESS_START_TIMEOUT_SECONDS = 3.0


def _register_read_write(
    blackboard: py_trees.blackboard.Client,
    key: str,
) -> None:
    blackboard.register_key(
        key=key,
        access=py_trees.common.Access.READ,
    )
    blackboard.register_key(
        key=key,
        access=py_trees.common.Access.WRITE,
    )


def _title_words(text: str) -> str:
    return " ".join(
        part[:1].upper() + part[1:]
        for part in text.split()
    )


def _extract_name(text: str) -> str:
    """Extract a short human name from a chess setup reply."""

    cleaned = text.strip(" \t\r\n.,!?")

    patterns = [
        r"(?i)\bmy name is\s+([a-z][a-z' -]{0,40})$",
        r"(?i)\bi am\s+([a-z][a-z' -]{0,40})$",
        r"(?i)\bi'm\s+([a-z][a-z' -]{0,40})$",
        r"(?i)\bcall me\s+([a-z][a-z' -]{0,40})$",
        r"(?i)\byou are playing\s+([a-z][a-z' -]{0,40})$",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return _title_words(
                re.sub(r"\s+", " ", match.group(1)).strip()
            )

    # A bare reply is accepted only when it consists of one plausible
    # name. Multi-word names require an explicit form such as
    # "My name is Richard Hopkins" or "I am Richard Hopkins".
    if re.fullmatch(
        r"[A-Za-z][A-Za-z'-]{1,39}",
        cleaned,
    ):
        return _title_words(cleaned)

    return ""


class _ChessDialogueBase(py_trees.behaviour.Behaviour):
    """Shared asynchronous speech/service machinery for chess setup leaves."""

    def __init__(
        self,
        *,
        node: Node,
        name: str,
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
            BlackboardKey.DIALOGUE_STATE,
            BlackboardKey.DIALOGUE_ERROR,
            BlackboardKey.CHESS_STATE,
            BlackboardKey.CHESS_SETUP_STEP,
            BlackboardKey.CHESS_PLAYER_NAME,
            BlackboardKey.CHESS_GAME_ACTIVE,
            BlackboardKey.CHESS_ERROR,
            BlackboardKey.PERCEPTION_ATTENDED_PERSON,
            BlackboardKey.PERCEPTION_VISIBLE_IDENTITIES,
        ]:
            _register_read_write(
                self.blackboard,
                key,
            )

        self.start_client = node.create_client(
            StartChessGame,
            "/chess/start_game",
        )
        self.speech_client = ActionClient(
            node,
            SpeakText,
            "/voice/speak",
        )
        self.intent_context_publisher = node.create_publisher(
            String,
            "/intent/context",
            10,
        )

        self.phase = "IDLE"
        self.player_name = ""
        self.service_future = None

        self.service_wait_started = None
        self.service_call_started = None

        self.speech_goal_future = None
        self.speech_goal_handle = None
        self.speech_result_future = None

    def _publish_setup_context(
        self,
        *,
        waiting_for_name: bool,
    ) -> None:
        if waiting_for_name:
            payload = {
                "chess_state": "SETUP",
                "chess_setup_step": "WAIT_NAME",
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

    def _known_player_name(self) -> str:
        """Use social attention first, then one unambiguous recognised face."""

        attended = str(
            self.blackboard.get(
                BlackboardKey.PERCEPTION_ATTENDED_PERSON
            )
            or ""
        ).strip()

        if attended:
            return attended

        visible = self.blackboard.get(
            BlackboardKey.PERCEPTION_VISIBLE_IDENTITIES
        )

        names = sorted(
            {
                str(name).strip()
                for name in (visible or [])
                if str(name).strip()
            }
        )

        if len(names) == 1:
            return names[0]

        return ""


    def _fail_game_start(
        self,
        message: str,
    ) -> None:
        """Abandon chess setup without blocking the central BT."""

        self.blackboard.set(
            BlackboardKey.CHESS_ERROR,
            message,
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.DIALOGUE_ERROR,
            message,
            overwrite=True,
        )

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

        # Remove WAIT_NAME/START_GAME context from the intent node.
        self._publish_setup_context(
            waiting_for_name=False
        )

        self.feedback_message = message


    def _request_game_start(
        self,
    ) -> Optional[bool]:
        """
        Start the asynchronous StartChessGame call.

        Returns:
            True  - request has been sent.
            None  - still waiting briefly for the service.
            False - service did not appear before the timeout.
        """

        if self.service_future is not None:
            return True

        now = time.monotonic()

        if self.service_wait_started is None:
            self.service_wait_started = now

        if not self.start_client.service_is_ready():

            elapsed = (
                now
                - self.service_wait_started
            )

            if elapsed >= CHESS_START_TIMEOUT_SECONDS:

                self._fail_game_start(
                    "/chess/start_game is unavailable"
                )

                self.node.get_logger().error(
                    "Chess setup abandoned: "
                    "/chess/start_game was not available "
                    f"after {elapsed:.1f}s"
                )

                return False

            self.feedback_message = (
                "waiting for /chess/start_game"
            )

            return None

        request = StartChessGame.Request()
        request.player_name = self.player_name

        self.service_future = (
            self.start_client.call_async(
                request
            )
        )

        self.service_call_started = (
            time.monotonic()
        )

        self.blackboard.set(
            BlackboardKey.CHESS_SETUP_STEP,
            ChessSetupStep.START_GAME,
            overwrite=True,
        )

        self.feedback_message = (
            f"arming chess for {self.player_name}"
        )

        return True

    def _game_start_result(
        self,
    ) -> Optional[bool]:
        """Return None while waiting, otherwise success/failure."""

        if self.service_future is None:
            return None

        if not self.service_future.done():

            if self.service_call_started is not None:

                elapsed = (
                    time.monotonic()
                    - self.service_call_started
                )

                if elapsed >= CHESS_START_TIMEOUT_SECONDS:

                    self._fail_game_start(
                        "/chess/start_game did not respond"
                    )

                    self.node.get_logger().error(
                        "Chess setup abandoned: "
                        "/chess/start_game did not respond "
                        f"after {elapsed:.1f}s"
                    )

                    return False

            self.feedback_message = (
                "waiting for chess manager to arm "
                f"for {self.player_name}"
            )

            return None

        try:
            response = self.service_future.result()

        except Exception as exc:

            self._fail_game_start(
                f"/chess/start_game failed: {exc}"
            )

            return False

        if not response.success:

            error = (
                response.message
                or "Chess manager rejected the start request"
            )

            self._fail_game_start(
                error
            )

            return False

        self.blackboard.set(
            BlackboardKey.CHESS_SETUP_STEP,
            ChessSetupStep.NONE,
            overwrite=True,
        )

        return True

    def _start_speech(
        self,
        text: str,
    ) -> bool:
        if not self.speech_client.server_is_ready():
            self.feedback_message = "waiting for /voice/speak"
            return False

        goal = SpeakText.Goal()
        goal.text = text
        goal.owner = "chess_setup"
        goal.priority = CHESS_SETUP_SPEECH_PRIORITY
        goal.interrupt_lower_priority = True
        goal.clear_lower_priority = False

        self.speech_goal_future = (
            self.speech_client.send_goal_async(goal)
        )
        self.speech_goal_handle = None
        self.speech_result_future = None

        self.feedback_message = text
        return True

    def _speech_finished(self) -> Optional[bool]:
        """Return None while speaking, else whether speech completed normally."""

        if self.speech_goal_future is None:
            return False

        if self.speech_goal_handle is None:
            if not self.speech_goal_future.done():
                return None

            try:
                handle = self.speech_goal_future.result()
            except Exception as exc:
                self.blackboard.set(
                    BlackboardKey.DIALOGUE_ERROR,
                    f"Chess setup speech failed: {exc}",
                    overwrite=True,
                )
                return False

            if not handle.accepted:
                self.blackboard.set(
                    BlackboardKey.DIALOGUE_ERROR,
                    "Chess setup speech goal rejected",
                    overwrite=True,
                )
                return False

            self.speech_goal_handle = handle
            self.speech_result_future = handle.get_result_async()
            return None

        if not self.speech_result_future.done():
            return None

        try:
            wrapped = self.speech_result_future.result()
            result = wrapped.result
        except Exception as exc:
            self.blackboard.set(
                BlackboardKey.DIALOGUE_ERROR,
                f"Chess setup speech result failed: {exc}",
                overwrite=True,
            )
            return False

        return bool(result.success)


class BeginChessSetup(_ChessDialogueBase):
    """Handle PLAY_CHESS without blocking later ordinary conversation turns."""

    def __init__(
        self,
        *,
        node: Node,
        name: str = "Begin Chess Setup",
    ) -> None:
        super().__init__(
            node=node,
            name=name,
        )

    def initialise(self) -> None:
        self.phase = "PREPARE"
        self.player_name = ""
        self.service_future = None
        self.service_wait_started = None
        self.service_call_started = None
        self.speech_goal_future = None
        self.speech_goal_handle = None
        self.speech_result_future = None

    def update(self) -> py_trees.common.Status:
        if self.phase == "PREPARE":
            current_state = str(
                self.blackboard.get(
                    BlackboardKey.CHESS_STATE
                )
                or ""
            ).upper()

            if current_state in {
                "WAITING_FOR_CHALLENGE",
                "ACTIVE",
                "SUSPENDED",
                "STARTING",
            }:
                if self._start_speech(
                    "We already have a chess game in progress."
                ):
                    self.phase = "SPEAK_ALREADY_ACTIVE"
                return py_trees.common.Status.RUNNING

            self.player_name = self._known_player_name()

            self.blackboard.set(
                BlackboardKey.CHESS_STATE,
                ChessState.SETUP,
                overwrite=True,
            )

            if self.player_name:
                self.blackboard.set(
                    BlackboardKey.CHESS_PLAYER_NAME,
                    self.player_name,
                    overwrite=True,
                )
                self._publish_setup_context(
                    waiting_for_name=False
                )
                self.phase = "START_GAME"
            else:
                self.blackboard.set(
                    BlackboardKey.CHESS_SETUP_STEP,
                    ChessSetupStep.WAIT_PLAYER_NAME,
                    overwrite=True,
                )
                self._publish_setup_context(
                    waiting_for_name=True
                )
                if self._start_speech(
                    "Who am I playing?"
                ):
                    self.phase = "SPEAK_ASK_NAME"

            return py_trees.common.Status.RUNNING

        if self.phase == "SPEAK_ASK_NAME":
            complete = self._speech_finished()
            if complete is None:
                return py_trees.common.Status.RUNNING

            # Deliberately leave the intent chess setup context in WAIT_NAME.
            # The next unrelated utterance can still be GENERAL_CONVERSATION;
            # a valid name will be classified as CHESS_SETUP_ANSWER.
            self.feedback_message = "waiting for player name on a later turn"
            return py_trees.common.Status.SUCCESS

        if self.phase == "START_GAME":

            request_started = (
                self._request_game_start()
            )

            if request_started is None:
                return py_trees.common.Status.RUNNING

            if request_started is False:

                if self._start_speech(
                    "I cannot start the chess subsystem at present."
                ):
                    self.phase = "SPEAK_START_FAILED"
                    return py_trees.common.Status.RUNNING

                # Even if speech is unavailable, never wedge the BT.
                return py_trees.common.Status.SUCCESS

            started = self._game_start_result()

            if started is None:
                return py_trees.common.Status.RUNNING

            if not started:

                if self._start_speech(
                    "I cannot start the chess subsystem at present."
                ):
                    self.phase = "SPEAK_START_FAILED"
                    return py_trees.common.Status.RUNNING

                # Speech itself must not become another blocking dependency.
                return py_trees.common.Status.SUCCESS

            text = (
                f"Affirmative, {self.player_name}. "
                "I am setting up the chessboard."
            )
            if self._start_speech(text):
                self.phase = "SPEAK_PHANTOM_PROMPT"
            return py_trees.common.Status.RUNNING

        if self.phase in {
            "SPEAK_ALREADY_ACTIVE",
            "SPEAK_START_FAILED",
            "SPEAK_PHANTOM_PROMPT",
        }:
            complete = self._speech_finished()
            if complete is None:
                return py_trees.common.Status.RUNNING

            self._publish_setup_context(
                waiting_for_name=False
            )
            return py_trees.common.Status.SUCCESS

        self.feedback_message = f"unexpected phase {self.phase}"
        return py_trees.common.Status.FAILURE


class ContinueChessSetup(_ChessDialogueBase):
    """Handle CHESS_SETUP_ANSWER on a later conversational turn."""

    def __init__(
        self,
        *,
        node: Node,
        name: str = "Continue Chess Setup",
    ) -> None:
        super().__init__(
            node=node,
            name=name,
        )

    def initialise(self) -> None:
        self.phase = "READ_NAME"
        self.player_name = ""
        self.service_future = None
        self.service_wait_started = None
        self.service_call_started = None
        self.speech_goal_future = None
        self.speech_goal_handle = None
        self.speech_result_future = None

    def update(self) -> py_trees.common.Status:
        if self.phase == "READ_NAME":
            setup_step = str(
                self.blackboard.get(
                    BlackboardKey.CHESS_SETUP_STEP
                )
                or ""
            ).upper()

            if setup_step not in {
                "WAIT_PLAYER_NAME",
                "WAIT_NAME",
            }:
                self.feedback_message = (
                    f"not waiting for a chess player name: {setup_step}"
                )
                return py_trees.common.Status.FAILURE

            command = str(
                self.blackboard.get(
                    BlackboardKey.DIALOGUE_COMMAND
                )
                or ""
            ).strip()

            self.player_name = _extract_name(command)

            if not self.player_name:
                self._publish_setup_context(
                    waiting_for_name=True
                )
                if self._start_speech(
                    "I did not catch the name. Who am I playing?"
                ):
                    self.phase = "SPEAK_RETRY"
                return py_trees.common.Status.RUNNING

            self.blackboard.set(
                BlackboardKey.CHESS_PLAYER_NAME,
                self.player_name,
                overwrite=True,
            )
            self._publish_setup_context(
                waiting_for_name=False
            )
            self.phase = "START_GAME"

        if self.phase == "START_GAME":

            request_started = (
                self._request_game_start()
            )

            if request_started is None:
                return py_trees.common.Status.RUNNING

            if request_started is False:

                if self._start_speech(
                    "I cannot start the chess subsystem at present."
                ):
                    self.phase = "SPEAK_START_FAILED"
                    return py_trees.common.Status.RUNNING

                # Even if speech is unavailable, never wedge the BT.
                return py_trees.common.Status.SUCCESS

            started = self._game_start_result()

            if started is None:
                return py_trees.common.Status.RUNNING

            if not started:

                if self._start_speech(
                    "I cannot start the chess subsystem at present."
                ):
                    self.phase = "SPEAK_START_FAILED"
                    return py_trees.common.Status.RUNNING

                # Speech itself must not become another blocking dependency.
                return py_trees.common.Status.SUCCESS

            text = (
                f"Affirmative, {self.player_name}. "
                "I am setting up the chessboard."
            )
            if self._start_speech(text):
                self.phase = "SPEAK_PHANTOM_PROMPT"
            return py_trees.common.Status.RUNNING

        if self.phase == "SPEAK_RETRY":
            complete = self._speech_finished()
            if complete is None:
                return py_trees.common.Status.RUNNING

            # Keep waiting for another later CHESS_SETUP_ANSWER.
            self.blackboard.set(
                BlackboardKey.CHESS_SETUP_STEP,
                ChessSetupStep.WAIT_PLAYER_NAME,
                overwrite=True,
            )
            self._publish_setup_context(
                waiting_for_name=True
            )
            return py_trees.common.Status.SUCCESS

        if self.phase in {
            "SPEAK_START_FAILED",
            "SPEAK_PHANTOM_PROMPT",
        }:
            complete = self._speech_finished()
            if complete is None:
                return py_trees.common.Status.RUNNING

            return py_trees.common.Status.SUCCESS

        self.feedback_message = f"unexpected phase {self.phase}"
        return py_trees.common.Status.FAILURE


class ChessControlCommand(py_trees.behaviour.Behaviour):
    """Send one non-blocking executive chess command to the chess manager."""

    def __init__(
        self,
        *,
        node: Node,
        command: str,
        name: str,
    ) -> None:
        super().__init__(name=name)
        self.node = node
        self.command = str(command).strip().upper()
        self.client = node.create_client(
            ControlChessGame,
            "/chess/control",
        )
        self.future = None
        self.started_at = 0.0

    def initialise(self) -> None:
        self.future = None
        self.started_at = time.monotonic()

    def update(self) -> py_trees.common.Status:
        if self.future is None:
            if not self.client.service_is_ready():
                if time.monotonic() - self.started_at >= 3.0:
                    self.node.get_logger().error(
                        "/chess/control is unavailable"
                    )
                    return py_trees.common.Status.SUCCESS

                self.feedback_message = "waiting for /chess/control"
                return py_trees.common.Status.RUNNING

            request = ControlChessGame.Request()
            request.command = self.command
            self.future = self.client.call_async(
                request
            )
            self.feedback_message = self.command
            return py_trees.common.Status.RUNNING

        if not self.future.done():
            return py_trees.common.Status.RUNNING

        try:
            response = self.future.result()
            self.feedback_message = str(
                response.message
            )
        except Exception as exc:
            self.node.get_logger().error(
                f"Chess command {self.command} failed: {exc}"
            )

        # Command rejection/decision speech is published as chess events by
        # the authoritative manager, so the dialogue turn can always clear.
        return py_trees.common.Status.SUCCESS


class ChessRuntimeManager(py_trees.behaviour.Behaviour):
    """Run chess beside normal conversation and turn events into behaviour.

    Responsibilities are intentionally separated:

    * /chess/status -> durable authoritative blackboard state;
    * normal move events -> deterministic move instructions;
    * terminal/social events -> concise English event prompts sent to the
      conversation node for a K9-style reaction;
    * GAME_FINISHED supersedes redundant final-move commentary.

    The LLM is never required for chess correctness.  If a reaction cannot be
    generated promptly, a deterministic fallback is spoken.
    """

    REACTION_TIMEOUT_SEC = 8.0

    def __init__(
        self,
        node: Node,
        name: str = "Chess Runtime Manager",
    ) -> None:
        super().__init__(name=name)
        self.node = node

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        for key in [
            BlackboardKey.CHESS_STATE,
            BlackboardKey.CHESS_SETUP_STEP,
            BlackboardKey.CHESS_PLAYER_NAME,
            BlackboardKey.CHESS_HUMAN_COLOUR,
            BlackboardKey.CHESS_K9_COLOUR,
            BlackboardKey.CHESS_GAME_ID,
            BlackboardKey.CHESS_GAME_ACTIVE,
            BlackboardKey.CHESS_GAME_SUSPENDED,
            BlackboardKey.CHESS_SIDE_TO_MOVE,
            BlackboardKey.CHESS_PENDING_MOVE,
            BlackboardKey.CHESS_LAST_MOVE,
            BlackboardKey.CHESS_RESULT,
            BlackboardKey.CHESS_ERROR,
            BlackboardKey.CHESS_FEN,
            BlackboardKey.CHESS_PLY,
            BlackboardKey.CHESS_ENGINE_BUSY,
            BlackboardKey.CHESS_EVALUATION_VALID,
            BlackboardKey.CHESS_EVALUATION_PAWNS,
            BlackboardKey.CHESS_EVALUATION_IS_MATE,
            BlackboardKey.CHESS_MATE_IN,
            BlackboardKey.CHESS_LAST_EVENT,
        ]:
            _register_read_write(
                self.blackboard,
                key,
            )

        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._lock = threading.Lock()
        self._latest_status = None
        self._status_dirty = False
        self._events: Deque[dict] = deque(
            maxlen=32
        )

        # Cached move events allow GAME_FINISHED to be interpreted even when
        # its ROS event reaches the BT before an earlier queued move event has
        # been ticked.
        self._pending_terminal_move = None
        self._last_human_move = None

        self.status_subscription = node.create_subscription(
            ChessStatus,
            "/chess/status",
            self._status_callback,
            status_qos,
        )
        self.event_subscription = node.create_subscription(
            ChessEvent,
            "/chess/event",
            self._event_callback,
            30,
        )

        self.speech_client = ActionClient(
            node,
            SpeakText,
            "/voice/speak",
        )
        self._speech_goal_future = None
        self._speech_goal_handle = None
        self._speech_result_future = None

        # Personality reactions are generated by the existing serialised
        # conversation/Ollama node, but on a separate channel that never enters
        # normal conversation history.
        self.reaction_request_publisher = node.create_publisher(
            String,
            "/conversation/chess_reaction/request",
            10,
        )
        self.reaction_response_subscription = node.create_subscription(
            String,
            "/conversation/chess_reaction/response",
            self._reaction_response_callback,
            10,
        )

        self._reaction_sequence = 0
        self._reaction_request_id = ""
        self._reaction_requested_at = 0.0
        self._reaction_fallback = ""
        self._reaction_response = None

        # When a game ends by mate, "Checkmate." is spoken deterministically
        # before any personality reaction.  The LLM may generate in parallel,
        # but its speech is held until this declaration has completed.
        self._terminal_declaration_active = False

        self.tail_wag_v = node.create_client(
            Trigger,
            "/tail_wag_v",
        )
        self.tail_up = node.create_client(
            Trigger,
            "/tail_up",
        )
        self.tail_down = node.create_client(
            Trigger,
            "/tail_down",
        )
        self.tail_centre = node.create_client(
            Trigger,
            "/tail_centre",
        )

    # ------------------------------------------------------------------
    # ROS callbacks only cache state. Blackboard mutation remains in update().
    # ------------------------------------------------------------------

    def _status_callback(
        self,
        msg: ChessStatus,
    ) -> None:
        status = {
            "state": msg.state,
            "player_name": msg.player_name,
            "human_colour": msg.human_colour,
            "k9_colour": msg.k9_colour,
            "game_id": msg.game_id,
            "game_active": bool(msg.game_active),
            "game_suspended": bool(msg.game_suspended),
            "side_to_move": msg.side_to_move,
            "pending_move": msg.pending_move,
            "last_move": msg.last_move,
            "result": msg.result,
            "error": msg.error,
            "fen": msg.fen,
            "ply": int(msg.ply),
            "engine_busy": bool(msg.engine_busy),
            "evaluation_valid": bool(msg.evaluation_valid),
            "evaluation_pawns": float(msg.evaluation_pawns),
            "evaluation_is_mate": bool(
                msg.evaluation_is_mate
            ),
            "mate_in": int(msg.mate_in),
        }

        with self._lock:
            self._latest_status = status
            self._status_dirty = True

    def _event_callback(
        self,
        msg: ChessEvent,
    ) -> None:
        event = {
            "type": msg.type,
            "game_id": msg.game_id,
            "player_name": msg.player_name,
            "status": msg.status,
            "colour": msg.colour,
            "uci": msg.uci,
            "san": msg.san,
            "piece": msg.piece,
            "from_square": msg.from_square,
            "to_square": msg.to_square,
            "captured_piece": msg.captured_piece,
            "gives_check": bool(msg.gives_check),
            "gives_mate": bool(msg.gives_mate),
            "source": msg.source,
            "evaluation_before_valid": bool(
                msg.evaluation_before_valid
            ),
            "evaluation_before_pawns": float(
                msg.evaluation_before_pawns
            ),
            "evaluation_after_valid": bool(
                msg.evaluation_after_valid
            ),
            "evaluation_after_pawns": float(
                msg.evaluation_after_pawns
            ),
            "evaluation_delta_pawns": float(
                msg.evaluation_delta_pawns
            ),
            "is_mate": bool(msg.is_mate),
            "mate_in": int(msg.mate_in),
            "speech_hint": msg.speech_hint,
            "message": msg.message,
        }

        event_type = str(
            event["type"]
        ).upper()

        with self._lock:
            # Cache terminal/recent moves immediately so a terminal event can
            # safely jump ahead of lower-priority queued commentary.
            if (
                event_type == "K9_MOVE_SELECTED"
                and event["gives_mate"]
            ):
                self._pending_terminal_move = dict(
                    event
                )

            if event_type == "HUMAN_MOVE":
                self._last_human_move = dict(
                    event
                )

            self._events.append(event)

    def _reaction_response_callback(
        self,
        msg: String,
    ) -> None:
        try:
            payload = json.loads(
                msg.data
            )
        except json.JSONDecodeError:
            self.node.get_logger().warning(
                "Ignoring malformed chess reaction response"
            )
            return

        request_id = str(
            payload.get(
                "request_id",
                "",
            )
        )
        text = str(
            payload.get(
                "text",
                "",
            )
        ).strip()

        with self._lock:
            if (
                not request_id
                or request_id != self._reaction_request_id
            ):
                return

            self._reaction_response = text

    # ------------------------------------------------------------------
    # Blackboard/status handling
    # ------------------------------------------------------------------

    def _apply_status(
        self,
        status: dict,
    ) -> None:
        # While waiting for an unknown player's name, the manager correctly
        # still reports IDLE because it has not yet been armed. Preserve the
        # local conversational SETUP state until the name is supplied.
        current_setup_step = str(
            self.blackboard.get(
                BlackboardKey.CHESS_SETUP_STEP
            )
            or ""
        ).upper()

        local_name_setup = (
            current_setup_step
            in {
                "WAIT_PLAYER_NAME",
                "WAIT_NAME",
            }
            and str(
                status["state"]
            ).upper() == "IDLE"
        )

        if not local_name_setup:
            self.blackboard.set(
                BlackboardKey.CHESS_STATE,
                status["state"] or ChessState.IDLE,
                overwrite=True,
            )

            if status["state"] not in {
                ChessState.SETUP,
            }:
                self.blackboard.set(
                    BlackboardKey.CHESS_SETUP_STEP,
                    ChessSetupStep.NONE,
                    overwrite=True,
                )

        if status["player_name"]:
            self.blackboard.set(
                BlackboardKey.CHESS_PLAYER_NAME,
                status["player_name"],
                overwrite=True,
            )

        mappings = [
            (
                BlackboardKey.CHESS_HUMAN_COLOUR,
                "human_colour",
            ),
            (
                BlackboardKey.CHESS_K9_COLOUR,
                "k9_colour",
            ),
            (
                BlackboardKey.CHESS_GAME_ID,
                "game_id",
            ),
            (
                BlackboardKey.CHESS_GAME_ACTIVE,
                "game_active",
            ),
            (
                BlackboardKey.CHESS_GAME_SUSPENDED,
                "game_suspended",
            ),
            (
                BlackboardKey.CHESS_SIDE_TO_MOVE,
                "side_to_move",
            ),
            (
                BlackboardKey.CHESS_PENDING_MOVE,
                "pending_move",
            ),
            (
                BlackboardKey.CHESS_LAST_MOVE,
                "last_move",
            ),
            (
                BlackboardKey.CHESS_RESULT,
                "result",
            ),
            (
                BlackboardKey.CHESS_ERROR,
                "error",
            ),
            (
                BlackboardKey.CHESS_FEN,
                "fen",
            ),
            (
                BlackboardKey.CHESS_PLY,
                "ply",
            ),
            (
                BlackboardKey.CHESS_ENGINE_BUSY,
                "engine_busy",
            ),
            (
                BlackboardKey.CHESS_EVALUATION_VALID,
                "evaluation_valid",
            ),
            (
                BlackboardKey.CHESS_EVALUATION_PAWNS,
                "evaluation_pawns",
            ),
            (
                BlackboardKey.CHESS_EVALUATION_IS_MATE,
                "evaluation_is_mate",
            ),
            (
                BlackboardKey.CHESS_MATE_IN,
                "mate_in",
            ),
        ]

        for key, field in mappings:
            self.blackboard.set(
                key,
                status[field],
                overwrite=True,
            )

    # ------------------------------------------------------------------
    # Speech and physical expression
    # ------------------------------------------------------------------

    def _speak(
        self,
        text: str,
        *,
        priority: int,
        owner: str = "chess_commentary",
        interrupt_lower_priority: bool = False,
        clear_lower_priority: bool = False,
    ) -> bool:
        if not text:
            return False

        if not self.speech_client.server_is_ready():
            self.node.get_logger().warning(
                "Dropping chess speech because "
                f"/voice/speak is unavailable: {text!r}"
            )
            return False

        goal = SpeakText.Goal()
        goal.text = text
        goal.owner = owner
        goal.priority = int(
            priority
        )
        goal.interrupt_lower_priority = bool(
            interrupt_lower_priority
        )
        goal.clear_lower_priority = bool(
            clear_lower_priority
        )

        self._speech_goal_future = (
            self.speech_client.send_goal_async(
                goal
            )
        )
        self._speech_goal_handle = None
        self._speech_result_future = None
        return True

    def _speech_busy(
        self,
    ) -> bool:
        if self._speech_goal_future is None:
            return False

        if self._speech_goal_handle is None:
            if not self._speech_goal_future.done():
                return True

            try:
                handle = (
                    self._speech_goal_future.result()
                )
            except Exception as exc:
                self.node.get_logger().warning(
                    f"Chess speech goal failed: {exc}"
                )
                self._clear_speech()
                return False

            if not handle.accepted:
                self._clear_speech()
                return False

            self._speech_goal_handle = handle
            self._speech_result_future = (
                handle.get_result_async()
            )
            return True

        if not self._speech_result_future.done():
            return True

        self._clear_speech()
        return False

    def _clear_speech(
        self,
    ) -> None:
        self._speech_goal_future = None
        self._speech_goal_handle = None
        self._speech_result_future = None

    @staticmethod
    def _trigger(
        client,
    ) -> None:
        if client.service_is_ready():
            client.call_async(
                Trigger.Request()
            )

    @staticmethod
    def _with_declaration(
        text: str,
        declaration: str,
    ) -> str:
        """Append a fixed chess declaration without saying it twice.

        ``move_instruction()`` may evolve independently and could itself start
        including "check".  Keeping duplicate suppression here makes the BT
        robust to that change while retaining a deterministic declaration.
        """
        base = str(text or "").strip()
        declaration = str(declaration or "").strip()

        if not declaration:
            return base

        keyword = declaration.rstrip(".!?").casefold()

        if keyword and keyword in base.casefold():
            return base

        if not base:
            return declaration

        if base[-1] not in ".!?":
            base += "."

        return f"{base} {declaration}"

    @staticmethod
    def _without_checkmate(text: str) -> str:
        """Remove an embedded checkmate declaration from a move instruction.

        ``move_instruction()`` currently includes ``Checkmate.`` when a move
        mates.  The BT deliberately announces checkmate later, only after the
        Phantom mechanism has completed and GAME_FINISHED confirms the result.
        Without removing it here the user hears "Checkmate" twice.
        """
        value = str(text or "").strip()

        value = re.sub(
            r"(?i)(?:\s*[,;:-]?\s*)\bcheckmate\b[.!?]*\s*$",
            "",
            value,
        ).strip()

        if value and value[-1] not in ".!?":
            value += "."

        return value

    # ------------------------------------------------------------------
    # English event interpretation / personality reaction
    # ------------------------------------------------------------------

    @staticmethod
    def _move_english(
        move: Optional[dict],
        *,
        actor: str,
    ) -> str:
        if not move:
            return ""

        piece = str(
            move.get("piece", "")
            or "piece"
        )
        from_square = str(
            move.get("from_square", "")
            or "?"
        )
        to_square = str(
            move.get("to_square", "")
            or "?"
        )
        san = str(
            move.get("san", "")
            or ""
        )

        text = (
            f"{actor} moved the {piece.lower()} "
            f"from {from_square} to {to_square}"
        )

        captured = str(
            move.get("captured_piece", "")
            or ""
        )
        if captured:
            text += (
                f", capturing the {captured.lower()}"
            )

        if san:
            text += f" ({san})"

        if move.get("gives_mate"):
            text += ", delivering checkmate"
        elif move.get("gives_check"):
            text += ", giving check"

        return text + "."

    def _game_result_details(
        self,
        event: dict,
    ) -> tuple[str, str, str]:
        result = str(
            event.get(
                "message",
                "",
            )
            or self.blackboard.get(
                BlackboardKey.CHESS_RESULT
            )
            or ""
        ).upper()

        k9_colour = str(
            self.blackboard.get(
                BlackboardKey.CHESS_K9_COLOUR
            )
            or ""
        ).upper()

        human_colour = str(
            self.blackboard.get(
                BlackboardKey.CHESS_HUMAN_COLOUR
            )
            or ""
        ).upper()

        winner = (
            result.split(
                ":",
                1,
            )[0]
            if ":" in result
            else ""
        )

        if winner in {
            "WHITE",
            "BLACK",
        }:
            if winner == k9_colour:
                outcome = "K9_WIN"
            elif winner == human_colour:
                outcome = "HUMAN_WIN"
            else:
                outcome = "UNKNOWN_WIN"
        else:
            outcome = "DRAW"

        return result, winner, outcome

    def _build_terminal_prompt(
        self,
        event: dict,
    ) -> tuple[str, str]:
        result, winner, outcome = (
            self._game_result_details(
                event
            )
        )

        player_name = str(
            event.get(
                "player_name",
                "",
            )
            or self.blackboard.get(
                BlackboardKey.CHESS_PLAYER_NAME
            )
            or "the other player"
        )

        k9_colour = str(
            self.blackboard.get(
                BlackboardKey.CHESS_K9_COLOUR
            )
            or "unknown"
        )

        human_colour = str(
            self.blackboard.get(
                BlackboardKey.CHESS_HUMAN_COLOUR
            )
            or "unknown"
        )

        terminal_move = None
        with self._lock:
            if (
                self._pending_terminal_move is not None
                and (
                    not event.get("game_id")
                    or self._pending_terminal_move.get(
                        "game_id"
                    ) == event.get("game_id")
                )
            ):
                terminal_move = dict(
                    self._pending_terminal_move
                )
            elif (
                self._last_human_move is not None
                and (
                    not event.get("game_id")
                    or self._last_human_move.get(
                        "game_id"
                    ) == event.get("game_id")
                )
            ):
                terminal_move = dict(
                    self._last_human_move
                )

        facts = [
            "A physical chess game has just ended.",
            f"Your opponent is {player_name}.",
            f"You were playing {k9_colour}.",
            f"{player_name} was playing {human_colour}.",
        ]

        move_sentence = self._move_english(
            terminal_move,
            actor=(
                "You"
                if (
                    terminal_move
                    and terminal_move.get(
                        "type"
                    ) == "K9_MOVE_SELECTED"
                )
                else player_name
            ),
        )
        if move_sentence:
            facts.append(
                move_sentence
            )

        reason = ""
        if ":" in result:
            reason = result.split(
                ":",
                1,
            )[1].lower()

        if outcome == "K9_WIN":
            facts.append(
                f"You have won the game"
                + (
                    f" by {reason}"
                    if reason
                    else ""
                )
                + "."
            )

            if reason == "mate":
                instruction = (
                    f"You, K9, won. {player_name} lost. "
                    "The chess system has already announced 'Checkmate.' "
                    "Say one short first-person follow-up sentence reacting to "
                    "YOUR victory. Do not repeat the word checkmate. "
                    f"Do NOT congratulate {player_name}; {player_name} did not "
                    "win. Do not say 'Congratulations'. Sound pleased and "
                    "slightly smug in K9's normal character, but not rude. "
                    "Do not analyse the position. Do not describe these "
                    "instructions."
                )
            else:
                instruction = (
                    f"You, K9, won. {player_name} lost. "
                    "Say one short first-person sentence reacting to YOUR "
                    f"victory. Do NOT congratulate {player_name}; "
                    f"{player_name} did not win. Do not say 'Congratulations'. "
                    "Sound pleased and slightly smug in K9's normal character, "
                    "but not rude. Do not analyse the position. Do not describe "
                    "these instructions."
                )

        elif outcome == "HUMAN_WIN":
            facts.append(
                f"{player_name} has won the game"
                + (
                    f" by {reason}"
                    if reason
                    else ""
                )
                + "."
            )

            if reason == "mate":
                instruction = (
                    f"{player_name} won. You, K9, lost. "
                    "The chess system has already announced 'Checkmate.' "
                    "Say one short follow-up sentence acknowledging the human's "
                    "victory. Do not repeat the word checkmate. Be gracious but "
                    "recognisably K9: mildly disappointed and dignified. "
                    "Prefer 'Well played' over 'Congratulations'. "
                    "Do not analyse the position. Do not describe these "
                    "instructions."
                )
            else:
                instruction = (
                    f"{player_name} won. You, K9, lost. "
                    "Say one short sentence acknowledging the human's victory. "
                    "Be gracious but recognisably K9: mildly disappointed and "
                    "dignified. Prefer 'Well played' over 'Congratulations'. "
                    "Do not analyse the position. Do not describe these "
                    "instructions."
                )
        else:
            facts.append(
                "The game has ended in a draw"
                + (
                    f" by {reason}"
                    if reason
                    else ""
                )
                + "."
            )
            instruction = (
                "Say one short sentence aloud to your opponent reacting to "
                "the draw in K9's normal character. Do not analyse the "
                "position. Do not describe these instructions."
            )

        prompt = (
            "RECENT CHESS EVENT\n\n"
            + "\n".join(facts)
            + "\n\nREACTION REQUIRED\n\n"
            + instruction
        )

        # Deterministic fallback: the result is always acknowledged even if
        # Ollama is unavailable or busy.
        base_hint = str(
            event.get(
                "speech_hint",
                "",
            )
            or ""
        ).strip()

        if outcome == "K9_WIN":
            if reason == "mate":
                # "Checkmate." has already been spoken deterministically.
                fallback = f"A satisfactory result, {player_name}. I have won."
            elif (
                terminal_move
                and terminal_move.get(
                    "type"
                ) == "K9_MOVE_SELECTED"
            ):
                move_hint = str(
                    terminal_move.get(
                        "speech_hint",
                        "",
                    )
                    or ""
                ).strip()

                if move_hint:
                    fallback = (
                        f"{move_hint} I have won."
                    )
                else:
                    fallback = f"A satisfactory result, {player_name}. I have won."
            else:
                fallback = (
                    base_hint
                    or "I have won."
                )
        elif outcome == "HUMAN_WIN":
            fallback = (
                base_hint
                or f"Well played, {player_name}. You have won."
            )
        else:
            fallback = (
                base_hint
                or "The game is drawn."
            )

        return prompt, fallback

    def _request_personality_reaction(
        self,
        *,
        prompt: str,
        fallback: str,
        game_id: str,
    ) -> None:
        self._reaction_sequence += 1
        request_id = (
            f"{game_id or 'chess'}:"
            f"{self._reaction_sequence}"
        )

        with self._lock:
            self._reaction_request_id = (
                request_id
            )
            self._reaction_requested_at = (
                time.monotonic()
            )
            self._reaction_fallback = (
                fallback
            )
            self._reaction_response = None

        payload = {
            "request_id": request_id,
            "prompt": prompt,
        }

        self.reaction_request_publisher.publish(
            String(
                data=json.dumps(
                    payload,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )
        )

        self.node.get_logger().info(
            "Requested chess personality reaction "
            f"{request_id}"
        )

    def _deliver_reaction_if_ready(
        self,
    ) -> None:
        with self._lock:
            request_id = (
                self._reaction_request_id
            )
            response = (
                self._reaction_response
            )
            requested_at = (
                self._reaction_requested_at
            )
            fallback = (
                self._reaction_fallback
            )

        if not request_id:
            return

        # A checkmate declaration is a deterministic chess fact and must be
        # heard before personality commentary.  Ollama is still allowed to
        # generate the follow-up while "Checkmate." is being spoken.
        if self._terminal_declaration_active:
            if self._speech_busy():
                return

            self._terminal_declaration_active = False

        timed_out = (
            requested_at > 0.0
            and (
                time.monotonic()
                - requested_at
            ) >= self.REACTION_TIMEOUT_SEC
        )

        if response is None and not timed_out:
            return

        text = (
            response.strip()
            if response
            else fallback.strip()
        )

        if not text:
            text = (
                "The chess game is over."
            )

        if timed_out and not response:
            self.node.get_logger().warning(
                "Chess personality reaction timed out; "
                "using deterministic fallback"
            )

        # A result should supersede stale low-priority chess commentary, but
        # must still yield to the normal dialogue priority (100).
        self._speak(
            text,
            priority=CHESS_RESULT_SPEECH_PRIORITY,
            owner="chess_result",
            interrupt_lower_priority=True,
            clear_lower_priority=True,
        )

        with self._lock:
            self._reaction_request_id = ""
            self._reaction_requested_at = 0.0
            self._reaction_fallback = ""
            self._reaction_response = None
            self._pending_terminal_move = None
            self._terminal_declaration_active = False

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _discard_redundant_terminal_events(
        self,
        game_id: str,
    ) -> None:
        with self._lock:
            kept = deque(
                maxlen=self._events.maxlen
            )

            for queued in self._events:
                same_game = (
                    not game_id
                    or queued.get(
                        "game_id"
                    ) == game_id
                )

                redundant = (
                    same_game
                    and (
                        (
                            queued.get(
                                "type",
                                "",
                            ).upper()
                            == "K9_MOVE_SELECTED"
                            and queued.get(
                                "gives_mate"
                            )
                        )
                        or queued.get(
                            "type",
                            "",
                        ).upper()
                        == "YOUR_MOVE"
                    )
                )

                if not redundant:
                    kept.append(
                        queued
                    )

            self._events = kept

    def _pop_next_event(
        self,
        *,
        speech_busy: bool,
    ) -> Optional[dict]:
        with self._lock:
            if not self._events:
                return None

            if not speech_busy:
                return self._events.popleft()

            # GAME_FINISHED is allowed to jump ahead of low-priority chess
            # speech. Its generated result utterance can then pre-empt/clear
            # that old commentary in VoicePiper.
            for index, candidate in enumerate(
                self._events
            ):
                if (
                    str(
                        candidate.get(
                            "type",
                            "",
                        )
                    ).upper()
                    == "GAME_FINISHED"
                ):
                    event = candidate
                    del self._events[
                        index
                    ]
                    return event

        return None

    def _react_to_event(
        self,
        event: dict,
    ) -> None:
        event_type = str(
            event.get(
                "type",
                "",
            )
        ).upper()

        self.blackboard.set(
            BlackboardKey.CHESS_LAST_EVENT,
            event_type,
            overwrite=True,
        )

        if event_type == "K9_MOVE_SELECTED":
            gives_mate = bool(
                event.get(
                    "gives_mate"
                )
            )
            gives_check = bool(
                event.get(
                    "gives_check"
                )
            )

            if gives_mate:
                # The move itself may be announced while Phantom executes it,
                # but "Checkmate." is deliberately deferred until
                # GAME_FINISHED, after physical completion has been confirmed.
                self._trigger(
                    self.tail_wag_v
                )

            elif gives_check:
                self._trigger(
                    self.tail_up
                )

            speech = str(
                event.get(
                    "speech_hint",
                    "",
                )
                or ""
            ).strip()

            if gives_mate:
                # move_instruction() may already contain "Checkmate.", but
                # mate is announced exactly once later, after physical
                # completion has been confirmed by GAME_FINISHED.
                speech = self._without_checkmate(
                    speech
                )

            elif gives_check:
                speech = self._with_declaration(
                    speech,
                    "Check.",
                )

            # Physical move instructions remain deterministic and precise.
            if speech:
                self._speak(
                    speech,
                    priority=CHESS_MOVE_SPEECH_PRIORITY,
                )

            return

        if event_type == "HUMAN_MOVE":
            # A human check is also a chess fact worth stating explicitly.
            # Checkmate itself is announced by GAME_FINISHED so it is spoken
            # exactly once, irrespective of who delivered mate.
            if (
                event.get(
                    "gives_check"
                )
                and not event.get(
                    "gives_mate"
                )
            ):
                self._speak(
                    "Check.",
                    priority=CHESS_DECLARATION_SPEECH_PRIORITY,
                    owner="chess_check",
                )

            return

        if event_type == "ILLEGAL_HUMAN_MOVE":
            self._speak(
                event.get(
                    "speech_hint",
                    "",
                )
                or (
                    "Negative. That move is not legal. "
                    "I shall restore the board."
                ),
                priority=CHESS_MOVE_SPEECH_PRIORITY,
                owner="chess_illegal_move",
                interrupt_lower_priority=True,
            )
            return

        if event_type == "YOUR_MOVE":
            self._speak(
                event.get(
                    "speech_hint",
                    "",
                )
                or "Your turn to move.",
                priority=CHESS_TURN_SPEECH_PRIORITY,
            )
            return

        if event_type in {
            "DRAW_OFFER_DECLINED",
            "CHESS_COMMAND_REJECTED",
        }:
            self._speak(
                str(
                    event.get(
                        "speech_hint",
                        "",
                    )
                    or event.get(
                        "message",
                        "",
                    )
                ),
                priority=CHESS_RESULT_SPEECH_PRIORITY,
                owner="chess_decision",
                interrupt_lower_priority=True,
            )
            return

        if event_type == "GAME_FINISHED":
            result, winner, outcome = (
                self._game_result_details(
                    event
                )
            )

            reason = (
                result.split(
                    ":",
                    1,
                )[1].lower()
                if ":" in result
                else str(
                    event.get(
                        "status",
                        "",
                    )
                    or ""
                ).lower()
            )

            is_checkmate = (
                reason == "mate"
            )

            if is_checkmate:
                # Deterministic and higher priority than personality speech.
                # This can interrupt stale move commentary but still yields to
                # normal priority-100 dialogue.
                self._terminal_declaration_active = (
                    self._speak(
                        "Checkmate.",
                        priority=CHESS_DECLARATION_SPEECH_PRIORITY,
                        owner="chess_checkmate",
                        interrupt_lower_priority=True,
                        clear_lower_priority=True,
                    )
                )

            if outcome == "K9_WIN":
                self._trigger(
                    self.tail_wag_v
                )
            elif outcome == "HUMAN_WIN":
                self._trigger(
                    self.tail_down
                )
            else:
                self._trigger(
                    self.tail_centre
                )

            prompt, fallback = (
                self._build_terminal_prompt(
                    event
                )
            )

            self._discard_redundant_terminal_events(
                str(
                    event.get(
                        "game_id",
                        "",
                    )
                )
            )

            self._request_personality_reaction(
                prompt=prompt,
                fallback=fallback,
                game_id=str(
                    event.get(
                        "game_id",
                        "",
                    )
                ),
            )
            return

        # HUMAN_MOVE and POSITION_EVALUATED are retained as recent context by
        # ChessConversationContext. K9_MOVE_SENT/CONFIRMED and engine activity
        # are intentionally silent. ENGINE_THINKING is NOT mapped to
        # /interaction/activity=PROCESSING because PROCESSING inhibits STT and
        # would prevent conversation while Stockfish thinks.

    def update(
        self,
    ) -> py_trees.common.Status:
        status = None

        with self._lock:
            if self._status_dirty:
                status = dict(
                    self._latest_status
                )
                self._status_dirty = False

        if status is not None:
            self._apply_status(
                status
            )

        speech_busy = self._speech_busy()

        event = self._pop_next_event(
            speech_busy=speech_busy,
        )

        if event is not None:
            self._react_to_event(
                event
            )

        # A generated result is allowed to supersede tracked low-priority chess
        # speech. VoicePiper itself will still protect priority-100 dialogue.
        self._deliver_reaction_if_ready()

        state = self.blackboard.get(
            BlackboardKey.CHESS_STATE
        )
        self.feedback_message = (
            f"chess overlay: {state}"
        )

        return py_trees.common.Status.RUNNING
