#!/usr/bin/env python3
"""Visible, non-blocking behaviour-tree shell for K9.

Increment 2 adds:
  * a namespaced /k9 blackboard;
  * registered and initialised executive state fields;
  * simple fundamental values suitable for ROS blackboard introspection.

The behaviour leaves remain placeholders. There are still no K9 service
clients, hardware calls or blocking waits.
"""

from __future__ import annotations

from dataclasses import dataclass
from platform import node

import py_trees
import py_trees_ros
import rclpy
from std_msgs.msg import Bool, String
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.action import ActionClient
from k9_interfaces_pkg.action import SpeakText

try:
    # Normal installed-package / ros2 run path.
    from k9_bt_pkg.k9_blackboard import (
        AudioMode,
        BlackboardKey,
        K9Blackboard,
    )
except ModuleNotFoundError:
    # Convenient direct execution from the source directory.
    from k9_blackboard import (
        AudioMode,
        BlackboardKey,
        K9Blackboard,
    )


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


class ProcessAudioEvents(py_trees.behaviour.Behaviour):
    """Receive ROS audio events and reflect them onto the K9 blackboard."""

    def __init__(self, node: Node) -> None:
        super().__init__(name="Process Audio Events")

        self.node = node

        self.blackboard = py_trees.blackboard.Client(
            name="Process Audio Events",
            namespace="k9",
        )

        for key in [
            BlackboardKey.AUDIO_HOTWORD_DETECTED,
            BlackboardKey.AUDIO_IS_LISTENING,
            BlackboardKey.AUDIO_HEARD_TEXT,
            BlackboardKey.AUDIO_LAST_EVENT,
        ]:
            self.blackboard.register_key(
                key=key,
                access=py_trees.common.Access.WRITE,
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

    def _hotword_callback(self, msg: Bool) -> None:
        # Treat True as an event and latch it until a BT behaviour consumes it.
        if msg.data:
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

    def update(self) -> py_trees.common.Status:
        self.feedback_message = "monitoring ROS audio events"
        return py_trees.common.Status.RUNNING


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
        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_EFFECTIVE_MODE,
            access=py_trees.common.Access.WRITE,
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

        # Deliberately publish every BT tick.
        #
        # This means a hotword/STT node that restarts will quickly receive
        # the current effective state even though the ROS topic is volatile.
        self.publisher.publish(String(data=self.mode))

        self.feedback_message = f"maintaining {self.mode}"

        return py_trees.common.Status.RUNNING


class HandleHotwordDetected(py_trees.behaviour.Behaviour):
    """Convert a latched hotword event into a request to listen."""

    def __init__(self) -> None:
        super().__init__(name="Handle Hotword Detected")

        self.blackboard = py_trees.blackboard.Client(
            name="Handle Hotword Detected",
            namespace="k9",
        )

        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_HOTWORD_DETECTED,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_HOTWORD_DETECTED,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_DESIRED_MODE,
            access=py_trees.common.Access.WRITE,
        )

    def update(self) -> py_trees.common.Status:
        detected = self.blackboard.get(
            BlackboardKey.AUDIO_HOTWORD_DETECTED
        )

        if not detected:
            self.feedback_message = "no hotword pending"
            return py_trees.common.Status.FAILURE

        self.blackboard.set(
            BlackboardKey.AUDIO_HOTWORD_DETECTED,
            False,
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.AUDIO_DESIRED_MODE,
            AudioMode.LISTENING,
            overwrite=True,
        )

        self.feedback_message = "requested LISTENING"

        return py_trees.common.Status.SUCCESS


class HandleUtteranceReceived(py_trees.behaviour.Behaviour):

    def __init__(self, name="Handle Utterance Received"):
        super().__init__(name=name)

        self.blackboard = self.attach_blackboard_client(
            name=name,
            namespace="k9",
        )

        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_HEARD_TEXT,
            access=py_trees.common.Access.WRITE,
        )

        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_DESIRED_MODE,
            access=py_trees.common.Access.WRITE,
        )

        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_COMMAND,
            access=py_trees.common.Access.WRITE,
        )

        self.blackboard.register_key(
            key=BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            access=py_trees.common.Access.WRITE,
        )

    def update(self):

        text = self.blackboard.get(
            BlackboardKey.AUDIO_HEARD_TEXT
        ).strip()

        if not text:
            return py_trees.common.Status.FAILURE

        # Preserve what the user said for the dialogue manager.
        self.blackboard.set(
            BlackboardKey.DIALOGUE_COMMAND,
            text,
            overwrite=True,
        )

        # For this first end-to-end test, generate a simple response.
        self.blackboard.set(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            f"Affirmative. I heard you say {text}",
            overwrite=True,
        )

        # The utterance is now consumed.
        self.blackboard.set(
            BlackboardKey.AUDIO_HEARD_TEXT,
            "",
            overwrite=True,
        )

        # Stop STT while we formulate/speak the response.
        self.blackboard.set(
            BlackboardKey.AUDIO_DESIRED_MODE,
            AudioMode.NOT_LISTENING,
            overwrite=True,
        )

        self.feedback_message = f"consumed: {text}"

        return py_trees.common.Status.SUCCESS

class SpeakPendingResponse(py_trees.behaviour.Behaviour):

    def __init__(self, *, node, name="Speak Pending Response"):
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
            access=py_trees.common.Access.WRITE,
        )

        self.blackboard.register_key(
            key=BlackboardKey.AUDIO_DESIRED_MODE,
            access=py_trees.common.Access.WRITE,
        )

        self.goal_future = None
        self.goal_handle = None
        self.result_future = None
        self.response_text = ""

    def initialise(self):

        self.goal_future = None
        self.goal_handle = None
        self.result_future = None

        self.response_text = self.blackboard.get(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE
        ).strip()

        if not self.response_text:
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

    def update(self):

        if not self.response_text:
            return py_trees.common.Status.FAILURE

        if self.goal_future is None:
            self.feedback_message = "voice server unavailable"
            return py_trees.common.Status.FAILURE

        #
        # Waiting for voice server to accept goal
        #

        if self.goal_handle is None:

            if not self.goal_future.done():
                self.feedback_message = "waiting for speech goal acceptance"
                return py_trees.common.Status.RUNNING

            self.goal_handle = self.goal_future.result()

            if not self.goal_handle.accepted:
                self.feedback_message = "speech goal rejected"
                return py_trees.common.Status.FAILURE

            self.result_future = self.goal_handle.get_result_async()

            self.feedback_message = "speaking"
            return py_trees.common.Status.RUNNING

        #
        # Voice is still speaking
        #

        if not self.result_future.done():
            self.feedback_message = "speaking"
            return py_trees.common.Status.RUNNING

        #
        # Speech completed
        #

        result = self.result_future.result().result

        if not result.success:
            self.feedback_message = result.message
            return py_trees.common.Status.FAILURE

        self.blackboard.set(
            BlackboardKey.DIALOGUE_PENDING_RESPONSE,
            "",
            overwrite=True,
        )

        self.blackboard.set(
            BlackboardKey.AUDIO_DESIRED_MODE,
            AudioMode.WAITING_FOR_HOTWORD,
            overwrite=True,
        )

        self.feedback_message = "speech complete"

        return py_trees.common.Status.SUCCESS


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

def create_audio_state_manager(
    node: Node,
) -> py_trees.behaviour.Behaviour:
    process_audio_events = ProcessAudioEvents(node)

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



def create_dialogue_manager() -> py_trees.behaviour.Behaviour:
    dialogue_manager = selector("Dialogue Manager")
    dialogue_manager.add_children(
        [
            HandleHotwordDetected(),
            HandleUtteranceReceived(),
            no_work("Handle StopListening Intent"),
            no_work("Handle PlayChess Intent"),
            no_work("Handle Chess Setup Answer"),
            no_work("Handle General Conversation"),
            SpeakPendingResponse(node=node),
            running("Dialogue Idle"),
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
            create_dialogue_manager(),
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
        # The wrapper provides canonical keys and lightweight type checking.
        self.blackboard = K9Blackboard()
        self.blackboard.set(
            BlackboardKey.SYSTEM_STATUS,
            "INITIALISING_TREE",
        )

        root = create_tree(self)

        # py_trees_ros adds snapshot-stream services and blackboard
        # introspection around the ordinary py_trees hierarchy.
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=root,
            unicode_tree_debug=False,
        )
        self.tree.setup(
            node=self,
            timeout=15.0,
        )

        # Keep a predictable snapshot topic available as well as the dynamic
        # snapshot-stream services used by the viewer.
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
            node.tree.shutdown(destroy_node=False)
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
