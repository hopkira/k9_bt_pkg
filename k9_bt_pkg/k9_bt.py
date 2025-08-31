#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
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
        self.sub = self.node.create_subscription(Bool, "/hotword_detected", self.cb, 10)

    def cb(self, msg):
        self.hotword = msg.data

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

    def update(self):
        if not self.client.service_is_ready():
            return Status.RUNNING
        if self.future is None:
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
        self.sub = self.node.create_subscription(String, "/speech_to_text/text", self.cb, 10)

    def cb(self, msg: String):
        self.command = msg.data

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

    def update(self):
        if not self.client.service_is_ready():
            return Status.RUNNING
        if self.future is None:
            cmd = self.command_getter()
            if not cmd:
                return Status.FAILURE
            req = GenerateUtterance.Request()
            req.input = cmd
            self.future = self.client.call_async(req)
            return Status.RUNNING
        if self.future.done():
            result = self.future.result()
            self.response_text = getattr(result, "output", "Apologies, cognitive faculties impaired.")
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
        self.pub = self.node.create_publisher(String, "/voice/tts_input", 10)
        self.sub = self.node.create_subscription(Bool, "is_talking", self._status_cb, 10)
        self.sent = False
        self.is_talking = False

    def _status_cb(self, msg: Bool):
        self.is_talking = msg.data

    def update(self):
        if not self.sent:
            msg = String()
            msg.data = self.text_getter()
            self.pub.publish(msg)
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

    def update(self):
        if not self.client.service_is_ready():
            return Status.RUNNING
        if self.future is None:
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
# Main
# -------------------
def main(args=None):
    rclpy.init(args=args)
    node = Node("k9_bt_node")

    tree = py_trees.trees.BehaviourTree(build_full_bt(node))

    # Tick tree periodically
    tree.setup()
    try:
        while rclpy.ok():
            tree.tick()
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
