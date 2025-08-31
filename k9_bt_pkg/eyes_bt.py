import py_trees
from py_trees.common import Status
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String
import numpy as np

# ------------------------
# Idle eyes (waiting for hotword)
# ------------------------
class EyesIdle(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="EyesIdle"):
        super().__init__(name)
        self.node = node
        self.eye_pub = self.node.create_publisher(Float32, "/eyes/brightness", 10)
        self.hotword_detected = False
        self.sub = self.node.create_subscription(Bool, "/hotword_detected", self.hotword_cb, 10)

    def hotword_cb(self, msg: Bool):
        self.hotword_detected = msg.data

    def update(self):
        # Eyes off while waiting for hotword
        if not self.hotword_detected:
            msg = Float32()
            msg.data = 0.0
            self.eye_pub.publish(msg)
            return Status.RUNNING
        return Status.FAILURE  # allow other states to take over

# ------------------------
# Listening eyes (full brightness)
# ------------------------
class EyesListening(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="EyesListening"):
        super().__init__(name)
        self.node = node
        self.eye_pub = self.node.create_publisher(Float32, "/eyes/brightness", 10)
        self.is_listening = False
        self.sub = self.node.create_subscription(String, "/speech_to_text/state", self.state_cb, 10)

    def state_cb(self, msg: String):
        self.is_listening = (msg.data == "listening")

    def update(self):
        if self.is_listening:
            msg = Float32()
            msg.data = 1.0  # full brightness
            self.eye_pub.publish(msg)
            return Status.RUNNING
        return Status.FAILURE

# ------------------------
# Talking eyes (RMS-driven)
# ------------------------
class EyesTalkingRMS(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, smoothing_alpha=0.2, name="EyesTalkingRMS"):
        super().__init__(name)
        self.node = node
        self.current_rms = 0.0
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_rms = 0.0
        self.is_talking = False
        self.rms_sub = self.node.create_subscription(Float32, "/voice/rms_level", self.rms_cb, 10)
        self.talking_sub = self.node.create_subscription(Bool, "/voice/is_talking", self.talking_cb, 10)
        self.eye_pub = self.node.create_publisher(Float32, "/eyes/brightness", 10)

    def rms_cb(self, msg: Float32):
        self.current_rms = msg.data

    def talking_cb(self, msg: Bool):
        self.is_talking = msg.data

    def update(self):
        if not self.is_talking:
            return Status.FAILURE
        self.smoothed_rms = self.smoothing_alpha * self.current_rms + (1 - self.smoothing_alpha) * self.smoothed_rms
        brightness = max(0.0, min(1.0, self.smoothed_rms))
        msg = Float32()
        msg.data = brightness
        self.eye_pub.publish(msg)
        return Status.RUNNING

# ------------------------
# Combine in selector for parallel eye branch
# ------------------------
def build_eye_selector(node: Node):
    selector = py_trees.composites.Selector("Eyes")
    selector.add_children([
        EyesTalkingRMS(node),
        EyesListening(node),
        EyesIdle(node)
    ])
    return selector
