#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from k9_interfaces_pkg.srv import GenerateUtterance, EmptySrv

import py_trees
import py_trees_ros
from py_trees.common import Status, ParallelPolicy

# Import the eye leaves
from k9_bt_pkg.eyes_bt import EyesTalkingRMS, EyesListening, EyesIdle


# -------------------
# Hotword
# -------------------
class WaitForHotword(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="WaitForHotword"):
        super().__init__(name)
        self.node = node
        self.hotword = False
        self.sub = self.node.create_subscription(Bool, "hotword_detected", self.cb, 10)

    def cb(self, msg):
        self.node.get_logger().info("Hotword detected" if msg.data else "Hotword cleared")
        self.hotword = msg.data

    def initialise(self):
        self.hotword = False

    def update(self):
        if self.hotword:
            self.hotword = False
            return Status.SUCCESS
        return Status.RUNNING


# -------------------
# STT
# -------------------
class StartListening(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="StartListening"):
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
            self.node.get_logger().info("Sending start_listening request")
            self.future = self.client.call_async(EmptySrv.Request())
            return Status.RUNNING
        if self.future.done():
            self.future = None
            return Status.SUCCESS
        return Status.RUNNING


class CommandReceived(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="CommandReceived"):
        super().__init__(name)
        self.node = node
        self.command = None
        self.sub = self.node.create_subscription(String, "speech_to_text/text", self.cb, 10)

    def cb(self, msg: String):
        self.node.get_logger().info(f"Command received: {msg.data}")
        self.command = msg.data

    def initialise(self):
        self.command = None

    def update(self):
        return Status.SUCCESS if self.command else Status.RUNNING


# -------------------
# Ollama LLM
# -------------------
class GenerateResponse(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, command_getter, name="GenerateResponse"):
        super().__init__(name)
        self.node = node
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
            result = self.future.result()
            self.response_text = getattr(result, "output", "Apologies, cognitive faculties impaired.")
            self.node.get_logger().info(f"LLM response: {self.response_text}")
            self.future = None
            return Status.SUCCESS
        return Status.RUNNING


# -------------------
# TTS
# -------------------
class SpeakResponse(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, text_getter, wait_until_done=True, name="SpeakResponse"):
        super().__init__(name)
        self.node = node
        self.text_getter = text_getter
        self.wait_until_done = wait_until_done
        self.pub = self.node.create_publisher(String, "voice/tts_input", 10)
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
            text = self.text_getter()
            if not text:
                self.node.get_logger().warn("No response text available for TTS")
                return Status.FAILURE
            msg = String()
            msg.data = text
            self.pub.publish(msg)
            self.node.get_logger().info(f"Speaking: {text}")
            self.sent = True
            return Status.RUNNING if self.wait_until_done else Status.SUCCESS
        if not self.wait_until_done:
            return Status.SUCCESS
        return Status.RUNNING if self.is_talking else Status.SUCCESS


# -------------------
# Resume Listening
# -------------------
class ResumeListening(py_trees.behaviour.Behaviour):
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
            self.node.get_logger().info("Resuming listening...")
            self.future = self.client.call_async(EmptySrv.Request())
            return Status.RUNNING
        if self.future.done():
            self.future = None
            return Status.SUCCESS
        return Status.RUNNING


# -------------------
# Build BT
# -------------------
def build_main_sequence(node: Node):
    root_seq = py_trees.composites.Sequence("MainSequence", memory=False)

    hotword_seq = py_trees.composites.Sequence("HotwordSequence", memory=False)
    hotword_seq.add_children([WaitForHotword(node), StartListening(node)])

    command_seq = py_trees.composites.Sequence("CommandSequence", memory=False)
    command = CommandReceived(node)
    generate = GenerateResponse(node, lambda: command.command)
    speak = SpeakResponse(node, lambda: generate.response_text)
    resume = ResumeListening(node)
    command_seq.add_children([command, generate, speak, resume])

    root_seq.add_children([hotword_seq, command_seq])
    return root_seq


def build_full_bt(node: Node):
    root_parallel = py_trees.composites.Parallel(
        name="K9RootParallel",
        policy=ParallelPolicy.SuccessOnOne()
    )

    main_seq = build_main_sequence(node)
    eye_selector = py_trees.composites.Selector("EyesSelector", memory=False)
    eye_selector.add_children([
        EyesTalkingRMS(node),
        EyesListening(node),
        EyesIdle(node)
    ])

    root_parallel.add_children([main_seq, eye_selector])
    return root_parallel


# -------------------
# Node Wrapper
# -------------------
class K9BTNode(Node):
    def __init__(self):
        super().__init__("k9_bt")


# -------------------
# Main
# -------------------
def main(args=None):
    rclpy.init(args=args)
    node = K9BTNode()

    root = build_full_bt(node)
    tree = py_trees_ros.trees.BehaviourTree(root=root, unicode_tree_debug=False)

    tree.setup(node=node, timeout=15.0)

    try:
        # Tick automatically every 100ms and publish snapshots
        tree.tick_tock(period_ms=100, node=node)
    except KeyboardInterrupt:
        pass
    finally:
        tree.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()