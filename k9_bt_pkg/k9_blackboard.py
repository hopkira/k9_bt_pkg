#!/usr/bin/env python3
"""Shared blackboard schema and initialisation for the K9 executive.

All values are deliberately fundamental Python types so that py_trees_ros can
pickle, compare and display them through the blackboard watcher.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import py_trees


class SystemMode:
    NORMAL = "NORMAL"
    EMERGENCY = "EMERGENCY"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class BatteryState:
    UNKNOWN = "UNKNOWN"
    NOMINAL = "NOMINAL"
    LOW = "LOW"
    CRITICAL = "CRITICAL"
    CHARGING = "CHARGING"


class AudioMode:
    NOT_LISTENING = "NOT_LISTENING"
    WAITING_FOR_HOTWORD = "WAITING_FOR_HOTWORD"
    LISTENING = "LISTENING"


class DialogueState:
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    WAITING_TO_SPEAK = "WAITING_TO_SPEAK"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


class Intent:
    NONE = "NONE"
    STOP_LISTENING = "STOP_LISTENING"
    PLAY_CHESS = "PLAY_CHESS"
    CHESS_SETUP_ANSWER = "CHESS_SETUP_ANSWER"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"


class SpeechState:
    IDLE = "IDLE"
    QUEUED = "QUEUED"
    SPEAKING = "SPEAKING"
    CANCELLING = "CANCELLING"
    ERROR = "ERROR"


class ChessState:
    IDLE = "IDLE"
    SETUP = "SETUP"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class ChessSetupStep:
    NONE = "NONE"
    ASK_PLAYER_NAME = "ASK_PLAYER_NAME"
    WAIT_PLAYER_NAME = "WAIT_PLAYER_NAME"
    ASK_COLOUR = "ASK_COLOUR"
    WAIT_COLOUR = "WAIT_COLOUR"
    START_GAME = "START_GAME"


class ExpressionState:
    EMERGENCY = "EMERGENCY"
    TALKING = "TALKING"
    LISTENING = "LISTENING"
    WAITING = "WAITING"
    NOT_LISTENING = "NOT_LISTENING"


class BlackboardKey:
    """Canonical relative keys beneath the /k9 namespace."""

    # Executive lifecycle
    SYSTEM_MODE = "system/mode"
    SYSTEM_READY = "system/ready"
    SYSTEM_STATUS = "system/status"
    SYSTEM_LAST_ERROR = "system/last_error"
    SYSTEM_SHUTDOWN_REQUESTED = "system/shutdown_requested"

    # Safety
    SAFETY_EMERGENCY_BUTTON_PRESSED = "safety/emergency_button_pressed"
    SAFETY_EMERGENCY_ACTIVE = "safety/emergency_active"
    SAFETY_EMERGENCY_REASON = "safety/emergency_reason"
    SAFETY_MOTION_INHIBITED = "safety/motion_inhibited"

    # Battery
    BATTERY_PERCENTAGE = "battery/percentage"
    BATTERY_VOLTAGE = "battery/voltage"
    BATTERY_STATE = "battery/state"
    BATTERY_IS_CHARGING = "battery/is_charging"
    BATTERY_LOW_WARNING = "battery/low_warning"
    BATTERY_CRITICAL_WARNING = "battery/critical_warning"

    # Audio arbitration and recognition
    AUDIO_DESIRED_MODE = "audio/desired_mode"
    AUDIO_EFFECTIVE_MODE = "audio/effective_mode"
    AUDIO_IS_TALKING = "audio/is_talking"
    AUDIO_IS_LISTENING = "audio/is_listening"
    AUDIO_HOTWORD_DETECTED = "audio/hotword_detected"
    AUDIO_HEARD_TEXT = "audio/heard_text"
    AUDIO_HEARD_CONFIDENCE = "audio/heard_confidence"
    AUDIO_LAST_EVENT = "audio/last_event"
    AUDIO_ERROR = "audio/error"

    # Dialogue and intent processing
    DIALOGUE_STATE = "dialogue/state"
    DIALOGUE_COMMAND = "dialogue/command"
    DIALOGUE_INTENT = "dialogue/intent"
    DIALOGUE_INTENT_CONFIDENCE = "dialogue/intent_confidence"
    DIALOGUE_PENDING_RESPONSE = "dialogue/pending_response"
    DIALOGUE_RESPONSE_PRIORITY = "dialogue/response_priority"
    DIALOGUE_RESPONSE_ID = "dialogue/response_id"
    DIALOGUE_CONVERSATION_ACTIVE = "dialogue/conversation_active"
    DIALOGUE_STOP_LISTENING_REQUESTED = "dialogue/stop_listening_requested"
    DIALOGUE_ERROR = "dialogue/error"

    # Speech action state
    SPEECH_STATE = "speech/state"
    SPEECH_CURRENT_TEXT = "speech/current_text"
    SPEECH_GOAL_ID = "speech/goal_id"
    SPEECH_CANCEL_REQUESTED = "speech/cancel_requested"
    SPEECH_ERROR = "speech/error"

    # Chess workflow
    CHESS_STATE = "chess/state"
    CHESS_SETUP_STEP = "chess/setup_step"
    CHESS_PLAYER_NAME = "chess/player_name"
    CHESS_PREFERRED_COLOUR = "chess/preferred_colour"
    CHESS_GAME_ID = "chess/game_id"
    CHESS_GAME_ACTIVE = "chess/game_active"
    CHESS_GAME_SUSPENDED = "chess/game_suspended"
    CHESS_SIDE_TO_MOVE = "chess/side_to_move"
    CHESS_PENDING_MOVE = "chess/pending_move"
    CHESS_LAST_MOVE = "chess/last_move"
    CHESS_RESULT = "chess/result"
    CHESS_ERROR = "chess/error"

    # Physical expression requested by the executive
    EXPRESSION_DESIRED = "expression/desired"
    EXPRESSION_EFFECTIVE = "expression/effective"
    EXPRESSION_EYES_LEVEL = "expression/eyes_level"
    EXPRESSION_EYE_ANIMATION = "expression/eye_animation"
    EXPRESSION_TAIL_MODE = "expression/tail_mode"
    EXPRESSION_EARS_MODE = "expression/ears_mode"


@dataclass(frozen=True)
class BlackboardField:
    key: str
    expected_type: type
    default: Any
    description: str


K9_BLACKBOARD_FIELDS: tuple[BlackboardField, ...] = (
    # Executive lifecycle
    BlackboardField(
        BlackboardKey.SYSTEM_MODE,
        str,
        SystemMode.NORMAL,
        "Top-level executive operating mode.",
    ),
    BlackboardField(
        BlackboardKey.SYSTEM_READY,
        bool,
        False,
        "True when the executive has completed initialisation.",
    ),
    BlackboardField(
        BlackboardKey.SYSTEM_STATUS,
        str,
        "STARTING",
        "Human-readable lifecycle status.",
    ),
    BlackboardField(
        BlackboardKey.SYSTEM_LAST_ERROR,
        str,
        "",
        "Most recent executive-level error.",
    ),
    BlackboardField(
        BlackboardKey.SYSTEM_SHUTDOWN_REQUESTED,
        bool,
        False,
        "Requests an orderly shutdown.",
    ),

    # Safety
    BlackboardField(
        BlackboardKey.SAFETY_EMERGENCY_BUTTON_PRESSED,
        bool,
        False,
        "Raw emergency-button state.",
    ),
    BlackboardField(
        BlackboardKey.SAFETY_EMERGENCY_ACTIVE,
        bool,
        False,
        "Latched/effective emergency state.",
    ),
    BlackboardField(
        BlackboardKey.SAFETY_EMERGENCY_REASON,
        str,
        "",
        "Reason for the current emergency state.",
    ),
    BlackboardField(
        BlackboardKey.SAFETY_MOTION_INHIBITED,
        bool,
        False,
        "Prevents motion-producing behaviours from operating.",
    ),

    # Battery
    BlackboardField(
        BlackboardKey.BATTERY_PERCENTAGE,
        float,
        100.0,
        "Estimated remaining battery percentage.",
    ),
    BlackboardField(
        BlackboardKey.BATTERY_VOLTAGE,
        float,
        0.0,
        "Measured battery voltage.",
    ),
    BlackboardField(
        BlackboardKey.BATTERY_STATE,
        str,
        BatteryState.UNKNOWN,
        "Derived battery state.",
    ),
    BlackboardField(
        BlackboardKey.BATTERY_IS_CHARGING,
        bool,
        False,
        "Whether external charging is active.",
    ),
    BlackboardField(
        BlackboardKey.BATTERY_LOW_WARNING,
        bool,
        False,
        "Low-battery warning flag.",
    ),
    BlackboardField(
        BlackboardKey.BATTERY_CRITICAL_WARNING,
        bool,
        False,
        "Critical-battery protective-action flag.",
    ),

    # Audio
    BlackboardField(
        BlackboardKey.AUDIO_DESIRED_MODE,
        str,
        AudioMode.WAITING_FOR_HOTWORD,
        "Audio mode requested by dialogue or another manager.",
    ),
    BlackboardField(
        BlackboardKey.AUDIO_EFFECTIVE_MODE,
        str,
        AudioMode.NOT_LISTENING,
        "Audio mode selected after safety and talking overrides.",
    ),
    BlackboardField(
        BlackboardKey.AUDIO_IS_TALKING,
        bool,
        False,
        "Whether K9 is currently producing speech.",
    ),
    BlackboardField(
        BlackboardKey.AUDIO_IS_LISTENING,
        bool,
        False,
        "Whether speech recognition is currently active.",
    ),
    BlackboardField(
        BlackboardKey.AUDIO_HOTWORD_DETECTED,
        bool,
        False,
        "Unconsumed hotword event.",
    ),
    BlackboardField(
        BlackboardKey.AUDIO_HEARD_TEXT,
        str,
        "",
        "Most recent recognised utterance.",
    ),
    BlackboardField(
        BlackboardKey.AUDIO_HEARD_CONFIDENCE,
        float,
        0.0,
        "Confidence associated with the recognised utterance.",
    ),
    BlackboardField(
        BlackboardKey.AUDIO_LAST_EVENT,
        str,
        "",
        "Most recently processed audio event.",
    ),
    BlackboardField(
        BlackboardKey.AUDIO_ERROR,
        str,
        "",
        "Most recent audio subsystem error.",
    ),

    # Dialogue
    BlackboardField(
        BlackboardKey.DIALOGUE_STATE,
        str,
        DialogueState.IDLE,
        "Current dialogue workflow state.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_COMMAND,
        str,
        "",
        "Utterance currently being handled as a command.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_INTENT,
        str,
        Intent.NONE,
        "Classified intent for the current command.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_INTENT_CONFIDENCE,
        float,
        0.0,
        "Confidence associated with the current intent.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_PENDING_RESPONSE,
        str,
        "",
        "Response waiting to be spoken.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_RESPONSE_PRIORITY,
        int,
        0,
        "Priority assigned to the pending response.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_RESPONSE_ID,
        str,
        "",
        "Correlation identifier for the pending response.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_CONVERSATION_ACTIVE,
        bool,
        False,
        "Whether K9 is in an active conversational exchange.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_STOP_LISTENING_REQUESTED,
        bool,
        False,
        "Requests that listening and hotword detection stop.",
    ),
    BlackboardField(
        BlackboardKey.DIALOGUE_ERROR,
        str,
        "",
        "Most recent dialogue or LLM error.",
    ),

    # Speech
    BlackboardField(
        BlackboardKey.SPEECH_STATE,
        str,
        SpeechState.IDLE,
        "Current speech-action state.",
    ),
    BlackboardField(
        BlackboardKey.SPEECH_CURRENT_TEXT,
        str,
        "",
        "Text accepted by the active speech action.",
    ),
    BlackboardField(
        BlackboardKey.SPEECH_GOAL_ID,
        str,
        "",
        "Identifier for the active speech goal.",
    ),
    BlackboardField(
        BlackboardKey.SPEECH_CANCEL_REQUESTED,
        bool,
        False,
        "Requests cancellation of active speech.",
    ),
    BlackboardField(
        BlackboardKey.SPEECH_ERROR,
        str,
        "",
        "Most recent speech-action error.",
    ),

    # Chess
    BlackboardField(
        BlackboardKey.CHESS_STATE,
        str,
        ChessState.IDLE,
        "Current chess workflow state.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_SETUP_STEP,
        str,
        ChessSetupStep.NONE,
        "Current question/action in chess setup.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_PLAYER_NAME,
        str,
        "",
        "Player name captured during setup.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_PREFERRED_COLOUR,
        str,
        "RANDOM",
        "Requested chess colour: WHITE, BLACK or RANDOM.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_GAME_ID,
        str,
        "",
        "Lichess or Phantom game identifier.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_GAME_ACTIVE,
        bool,
        False,
        "Whether a chess game is currently active.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_GAME_SUSPENDED,
        bool,
        False,
        "Whether an active game is temporarily suspended.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_SIDE_TO_MOVE,
        str,
        "",
        "Side whose turn it is: WHITE or BLACK.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_PENDING_MOVE,
        str,
        "",
        "Move waiting to be submitted or announced.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_LAST_MOVE,
        str,
        "",
        "Most recently completed chess move.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_RESULT,
        str,
        "",
        "Final or current game result/status.",
    ),
    BlackboardField(
        BlackboardKey.CHESS_ERROR,
        str,
        "",
        "Most recent chess subsystem error.",
    ),

    # Expression
    BlackboardField(
        BlackboardKey.EXPRESSION_DESIRED,
        str,
        ExpressionState.NOT_LISTENING,
        "Expression requested by the active behaviour.",
    ),
    BlackboardField(
        BlackboardKey.EXPRESSION_EFFECTIVE,
        str,
        ExpressionState.NOT_LISTENING,
        "Expression selected after priority overrides.",
    ),
    BlackboardField(
        BlackboardKey.EXPRESSION_EYES_LEVEL,
        float,
        0.0,
        "Requested normalised eye level from 0.0 to 1.0.",
    ),
    BlackboardField(
        BlackboardKey.EXPRESSION_EYE_ANIMATION,
        str,
        "IDLE",
        "Requested eye animation name.",
    ),
    BlackboardField(
        BlackboardKey.EXPRESSION_TAIL_MODE,
        str,
        "CENTRE",
        "Requested tail behaviour.",
    ),
    BlackboardField(
        BlackboardKey.EXPRESSION_EARS_MODE,
        str,
        "CENTRE",
        "Requested ear behaviour.",
    ),
)


class K9Blackboard:
    """Owns schema registration and safe access to the shared K9 blackboard."""

    def __init__(
        self,
        *,
        client_name: str = "K9 Executive Initialiser",
        namespace: str = "k9",
    ) -> None:
        self.namespace = namespace.strip("/")
        self.client = py_trees.blackboard.Client(
            name=client_name,
            namespace=self.namespace,
        )
        self._field_by_key = {
            field.key: field for field in K9_BLACKBOARD_FIELDS
        }

        # READ and WRITE are registered separately. This client initialises the
        # values and can also provide diagnostics. Future behaviours should
        # register only the access they actually need.
        for field in K9_BLACKBOARD_FIELDS:
            self.client.register_key(
                key=field.key,
                access=py_trees.common.Access.WRITE,
            )
            self.client.register_key(
                key=field.key,
                access=py_trees.common.Access.READ,
            )

        self.initialise_defaults()

    @property
    def field_count(self) -> int:
        return len(K9_BLACKBOARD_FIELDS)

    def absolute_name(self, key: str) -> str:
        self._require_known_key(key)
        return self.client.absolute_name(key)

    def initialise_defaults(self) -> None:
        """Initialise every registered key without overwriting existing data."""
        for field in K9_BLACKBOARD_FIELDS:
            self.client.set(
                field.key,
                deepcopy(field.default),
                overwrite=False,
            )

    def get(self, key: str) -> Any:
        self._require_known_key(key)
        return self.client.get(key)

    def set(self, key: str, value: Any) -> None:
        field = self._require_known_key(key)
        coerced_value = self._coerce_value(field, value)
        self.client.set(key, coerced_value, overwrite=True)

    def snapshot(self) -> dict[str, Any]:
        """Return a simple diagnostic copy of the current K9 state."""
        return {
            self.absolute_name(field.key): deepcopy(self.client.get(field.key))
            for field in K9_BLACKBOARD_FIELDS
        }

    def _require_known_key(self, key: str) -> BlackboardField:
        try:
            return self._field_by_key[key]
        except KeyError as exc:
            raise KeyError(f"Unknown K9 blackboard key: {key}") from exc

    @staticmethod
    def _coerce_value(
        field: BlackboardField,
        value: Any,
    ) -> Any:
        """Apply lightweight type safety while keeping ROS callback use easy."""
        expected = field.expected_type

        if expected is bool:
            if type(value) is not bool:
                raise TypeError(
                    f"{field.key} expects bool, got {type(value).__name__}"
                )
            return value

        if expected is int:
            if type(value) is not int:
                raise TypeError(
                    f"{field.key} expects int, got {type(value).__name__}"
                )
            return value

        if expected is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"{field.key} expects float, got {type(value).__name__}"
                )
            return float(value)

        if expected is str:
            if not isinstance(value, str):
                raise TypeError(
                    f"{field.key} expects str, got {type(value).__name__}"
                )
            return value

        if not isinstance(value, expected):
            raise TypeError(
                f"{field.key} expects {expected.__name__}, "
                f"got {type(value).__name__}"
            )
        return value