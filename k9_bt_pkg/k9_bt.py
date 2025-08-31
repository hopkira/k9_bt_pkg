#!/usr/bin/env python3
"""
K9 Behavior Tree Node
---------------------

This file defines the main behavior tree (BT) that drives K9's
speech interaction and eye panel control.

It uses py_trees and py_trees_ros to:
- Wait for a hotword
- Start listening (speech-to-text)
- Wait for a command
- Send the command to an LLM service
- Speak the response via TTS
- Resume listening
- Run an "eyes" behavior tree in parallel that reflects robot state

The BT is ticked at 10Hz and snapshots are published for visualization
via py_trees_tree_viewer or rqt_py_trees.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from k9_interfaces_pkg.srv import GenerateUtterance, EmptySrv

import py_trees
import py_trees_ros
from py_trees.common import Status, ParallelPolicy

# Import the eye-related behaviors from a separate module
from k9_bt_pkg.eyes_bt import EyesTalkingRMS, EyesListening, EyesIdle


class WaitForHotword(py_trees.behaviour.Behaviour):
    """
    Waits for a Bool message on "hotword_detected".

    - SUCCESS: when hotword is received
    - RUNNING: otherwise
    - Resets its internal state on each (re)entry to the tree
    """

    def __init__(self, node: Node, name="WaitForHotword"):
        super().__init__(name)
        self.node = node
        self.hotword = False
        # Subscribe to "hotword_detected" topic
        self.sub = self.node.create_subscription(Bool, "hotword_detected", self.cb, 10)

    def cb(self, msg):
        # Callback fires when the hotword node publishes a Bool
        self.node.get_logger().info("Hotword detected" if msg.data else "Hotword cleared")
        self.hotword = msg.data

    def initialise(self):
        # py_trees convention: reset state on entry
        self.hotword = False

    def update(self):
        # Behaviours always return one of [RUNNING, SUCCESS, FAILURE]
        if self.hotword:
            # Hotword "consumed" -> succeed once, then reset
            self.hotword = False
            return Status.SUCCESS
        return Status.RUNNING


class StartListening(py_trees.behaviour.Behaviour):
    """
    Calls the "start_listening" EmptySrv service to tell
    the STT node to begin listening.
    """

    def __init__(self, node: Node, name="StartListening"):
        super().__init__(name)
        self.node = node
        self.client = self.node.create_client(EmptySrv, "start_listening")
        self.future = None

    def initialise(self):
        self.future = None

    def update(self):
        # If service not available yet, keep trying
        if not self.client.service_is_ready():
            return Status.RUNNING

        if self.future is None:
            # First tick -> send async request
            self.node.get_logger().info("Sending start_listening request")
            self.future = self.client.call_async(EmptySrv.Request())
            return Status.RUNNING

        if self.future.done():
            # Service responded -> succeed
            self.future = None
            return Status.SUCCESS

        # Still waiting -> keep running
        return Status.RUNNING


class CommandReceived(py_trees.behaviour.Behaviour):
    """
    Waits until a transcription is received on "speech_to_text/text".
    """

    def __init__(self, node: Node, name="CommandReceived"):
        super().__init__(name)
        self.node = node
        self.command = None
        # Subscribes to STT output
        self.sub = self.node.create_subscription(String, "speech_to_text/text", self.cb, 10)

    def cb(self, msg: String):
        self.node.get_logger().info(f"Command received: {msg.data}")
        self.command = msg.data

    def initialise(self):
        # Reset command on each entry
        self.command = None

    def update(self):
        # If we’ve heard something, succeed, otherwise keep waiting
        return Status.SUCCESS if self.command else Status.RUNNING


class GenerateResponse(py_trees.behaviour.Behaviour):
    """
    Sends the received command to the "generate_utterance" LLM service.
    """

    def __init__(self, node: Node, command_getter, name="GenerateResponse"):
        super().__init__(name)
        self.node = node
        # We don’t store command here, instead we "pull" it
        self.command_getter = command_getter
        self.response_text = None
        self.client = self.node.create_client(GenerateUtterance, "generate_utterance")
        self.future = None

    def initialise(self):
        self.response_text = None
        self.future = None

    def update(self):
        if not self.client.service_is_ready():
            return Status.RUNNING

        if self.future is None:
            # First tick -> send request with last command
            cmd = self.command_getter()
            if not cmd:
                self.node.get_logger().warn("No command available to send to LLM")
                return Status.FAILURE

            req = GenerateUtterance.Request()
            req.input = cmd
            self.node.get_logger().info(f"Sending command to LLM: {cmd}")
            self.future = self.client.call_async(req)
            return Status.RUNNING

        if self.future.done():
            # Service responded -> store result + succeed
            result = self.future.result()
            self.response_text = getattr(result, "output", "Apologies, cognitive faculties impaired.")
            self.node.get_logger().info(f"LLM response: {self.response_text}")
            self.future = None
            return Status.SUCCESS

        return Status.RUNNING


class SpeakResponse(py_trees.behaviour.Behaviour):
    """
    Publishes the response text to "voice/tts_input" for TTS playback.
    """

    def __init__(self, node: Node, text_getter, wait_until_done=True, name="SpeakResponse"):
        super().__init__(name)
        self.node = node
        self.text_getter = text_getter
        self.wait_until_done = wait_until_done

        # Publisher sends the text to TTS node
        self.pub = self.node.create_publisher(String, "voice/tts_input", 10)
        # Subscriber listens to "is_talking" to know when playback ends
        self.sub = self.node.create_subscription(Bool, "is_talking", self._status_cb, 10)

        self.sent = False
        self.is_talking = False

    def initialise(self):
        self.sent = False
        self.is_talking = False

    def _status_cb(self, msg: Bool):
        self.is_talking = msg.data

    def update(self):
        if not self.sent:
            # First tick -> publish the response
            text = self.text_getter()
            if not text:
                self.node.get_logger().warn("No response text available for TTS")
                return Status.FAILURE

            msg = String()
            msg.data = text
            self.pub.publish(msg)
            self.node.get_logger().info(f"Speaking: {text}")
            self.sent = True

            # If we’re not waiting for playback, succeed immediately
            return Status.RUNNING if self.wait_until_done else Status.SUCCESS

        if not self.wait_until_done:
            return Status.SUCCESS

        # Wait until is_talking goes False
        return Status.RUNNING if self.is_talking else Status.SUCCESS


class ResumeListening(py_trees.behaviour.Behaviour):
    """
    Re-starts the STT listening loop by calling "start_listening".
    """

    def __init__(self, node: Node, name="ResumeListening"):
        super().__init__(name)
        self.node = node
        self.client = self.node.create_client(EmptySrv, "start_listening")
        self.future = None

    def initialise(self):
        self.future = None

    def update(self):
        if not self.client.service_is_ready():
            return Status.RUNNING

        if self.future is None:
            # Call service again to re-arm hotword listening
            self.node.get_logger().info("Resuming listening...")
            self.future = self.client.call_async(EmptySrv.Request())
            return Status.RUNNING

        if self.future.done():
            self.future = None
            return Status.SUCCESS

        return Status.RUNNING


def build_main_sequence(node: Node):
    """
    Main pipeline of speech interaction.

    A py_trees Sequence ticks each child in order:
      - If a child returns SUCCESS → move to next
      - If a child returns RUNNING → tick same child next cycle
      - If a child returns FAILURE → abort sequence

    Structure:
        [WaitForHotword] -> [StartListening] -> [CommandReceived]
        -> [GenerateResponse] -> [SpeakResponse] -> [ResumeListening]
    """
    root_seq = py_trees.composites.Sequence("MainSequence", memory=False)

    # Hotword detection sequence (first stage)
    hotword_seq = py_trees.composites.Sequence("HotwordSequence", memory=False)
    hotword_seq.add_children([WaitForHotword(node), StartListening(node)])

    # Command processing sequence (second stage)
    command_seq = py_trees.composites.Sequence("CommandSequence", memory=False)
    command = CommandReceived(node)
    generate = GenerateResponse(node, lambda: command.command)
    speak = SpeakResponse(node, lambda: generate.response_text)
    resume = ResumeListening(node)
    command_seq.add_children([command, generate, speak, resume])

    # Chain both sequences
    root_seq.add_children([hotword_seq, command_seq])
    return root_seq


def build_full_bt(node: Node):
    """
    Top-level tree:

    Parallel composite ticks both children *simultaneously*:
      - Succeeds if ONE child succeeds (SuccessOnOne policy)

    Children:
      - MainSequence (speech pipeline)
      - EyeSelector (manages idle/listening/talking eyes)

    Selector composite chooses the first child that succeeds.
    """
    root_parallel = py_trees.composites.Parallel(
        name="K9RootParallel",
        policy=ParallelPolicy.SuccessOnOne()
    )

    main_seq = build_main_sequence(node)

    # EyeSelector: only one of these should succeed at a time
    eye_selector = py_trees.composites.Selector("EyesSelector", memory=False)
    eye_selector.add_children([
        EyesTalkingRMS(node),   # If robot is speaking
        EyesListening(node),    # If robot is listening
        EyesIdle(node)          # Default fallback
    ])

    root_parallel.add_children([main_seq, eye_selector])
    return root_parallel


class K9BTNode(Node):
    """ROS 2 Node wrapper for the behavior tree."""
    def __init__(self):
        super().__init__("k9_bt")


def main(args=None):
    """
    Entry point:
    - Initialise ROS 2
    - Build and setup the tree
    - Tick at 10Hz with automatic snapshot publishing
    """
    rclpy.init(args=args)
    node = K9BTNode()

    root = build_full_bt(node)
    tree = py_trees_ros.trees.BehaviourTree(root=root, unicode_tree_debug=False)

    # Sets up publishers, subscribers, snapshot streaming
    tree.setup(node=node, timeout=15.0)

    try:
        # Tick tree at 100ms intervals
        tree.tick_tock(period_ms=100, node=node)
    except KeyboardInterrupt:
        pass
    finally:
        tree.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()