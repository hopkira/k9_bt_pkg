#!/usr/bin/env python3
"""Visible, non-blocking behaviour-tree shell for K9.

This first increment deliberately contains:
  * the complete intended tree hierarchy;
  * placeholder leaf behaviours only;
  * no service clients;
  * no hardware access;
  * no blocking waits.

The tree ticks continuously and exposes py_trees_ros snapshot services plus a
default snapshot topic for py-trees-tree-viewer / py-trees-tree-watcher.
"""

from __future__ import annotations

from dataclasses import dataclass

import py_trees
import py_trees_ros
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


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


def create_audio_state_manager() -> py_trees.behaviour.Behaviour:
    process_audio_events = running("Process Audio Events")

    talking_override = sequence("Talking Override")
    talking_override.add_children(
        [
            inactive("K9 Talking?"),
            running("Ensure NotListening"),
        ]
    )

    listening_state = sequence("Listening State")
    listening_state.add_children(
        [
            inactive("Desired = Listening?"),
            running("Maintain Listening"),
        ]
    )

    hotword_state = sequence("Hotword State")
    hotword_state.add_children(
        [
            inactive("Desired = WaitingForHotword?"),
            running("Maintain Hotword Detector"),
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
            running("Maintain NotListening"),
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
            no_work("Handle StopListening Intent"),
            no_work("Handle PlayChess Intent"),
            no_work("Handle Chess Setup Answer"),
            no_work("Handle General Conversation"),
            no_work("Speak Pending Response"),
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


def create_tree() -> py_trees.behaviour.Behaviour:
    """Construct the complete first-increment K9 hierarchy."""

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
            create_audio_state_manager(),
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
    """ROS 2 custodian for the visible K9 behaviour-tree shell."""

    def __init__(self) -> None:
        super().__init__("k9_bt_shell")

        self.declare_parameter("tick_period_ms", 200.0)
        tick_period_ms = float(
            self.get_parameter("tick_period_ms").value
        )
        if tick_period_ms <= 0.0:
            raise ValueError("tick_period_ms must be greater than zero")

        root = create_tree()

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
            ]
        )
        for result in parameter_results:
            if not result.successful:
                self.get_logger().warning(
                    "Could not configure a tree snapshot parameter: "
                    f"{result.reason}"
                )

        # Tick once immediately so a newly opened viewer does not need to wait
        # for the first timer callback.
        self.tree.tick()

        # py_trees_ros implements this with an rclpy timer. It does not start a
        # second blocking loop or require a node argument here.
        self.tree.tick_tock(period_ms=tick_period_ms)

        self.get_logger().info(
            "K9 visible tree shell is running"
        )
        self.get_logger().info(
            f"Tick period: {tick_period_ms:.0f} ms"
        )
        self.get_logger().info(
            "Default snapshot topic: /k9_bt_shell/snapshots"
        )
        self.get_logger().info(
            "Open py-trees-tree-viewer or run py-trees-tree-watcher"
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
            node.tree.shutdown(destroy_node=False)
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
