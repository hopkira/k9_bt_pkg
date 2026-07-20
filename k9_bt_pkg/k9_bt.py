#!/usr/bin/env python3

# ROS 2 Behavior Tree

#import serial
#import ast
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String, Bool
from k9_interfaces_pkg.srv import LightsControl, SwitchState  # Custom interfaces
from k9_interfaces_pkg.srv import Speak, CancelSpeech  # Updated to k9_voice package
from k9_interfaces_pkg.srv import SetBrightness, GetBrightness

import py_trees
import py_trees_ros
#from py_trees_ros.trees import BehaviourTree
#from py_trees.blackboard import Blackboard
from rclpy.parameter import Parameter

# Initialize components
#
k9stt=None
k9qa=None
ChessGame=None

'''
k9stt = Listen()
k9qa = Respond()
ChessGame = Backhistory()
'''

'''
blackboard = Blackboard()
blackboard.command = None
blackboard.intent = None
blackboard.speaking = 0.0
blackboard.hotword_detected = False
'''

class NotListening(py_trees.behaviour.Behaviour):
    def __init__(self, node, name="NotListening"):
        super().__init__(name)
        self.node = node

    def initialise(self):
        self.node.eyes.set_level(0.0)
        self.node.back_lights.off()
        self.node.back_lights.cmd('computer')
        self.node.back_lights.turn_on([1, 3, 7, 10, 12])
        self.node.tail.center()
        self.node.ears.stop()

    def update(self):
        return py_trees.common.Status.SUCCESS


class WaitForHotword(py_trees.behaviour.Behaviour):
    def __init__(self, node, name="WaitForHotword"):
        super().__init__(name)
        self.node = node

        self.blackboard = attach_blackboard_client(
            self,
            read_keys=("hotword_detected",),
            write_keys=("hotword_detected",),
        )

    def initialise(self):
        if self.node.is_talking:
            self.logger.info("Still speaking, waiting...")
        #if mem.retrieveState("speaking") == 1.0:
        #    self.logger.info("Still speaking, waiting...")
        self.node.back_lights.turn_on([1,3,6,8,9,12])
        self.node.tail.center()
        self.node.eyes.set_level(0.001)
        self.feedback_message = "Waiting for hotword"

    def update(self):
        if self.blackboard.hotword_detected:
            return py_trees.common.Status.RUNNING
        # Consume the event so it triggers only one interaction.
        self.blackboard.hotword_detected = False
        self.feedback_message = "Hotword detected"
        return py_trees.common.Status.SUCCESS


class Listening(py_trees.behaviour.Behaviour):
    def __init__(self, node, name="Listening"):
        super().__init__(name)
        self.node = node

        self.blackboard = attach_blackboard_client(
            self,
            read_keys=("command",),
            write_keys=("command",),
        )

    def initialise(self):
        if self.node.is_talking:
            self.logger.info("Still speaking, waiting...")
        self.node.back_lights.cmd('computer')
        self.node.back_lights.off()
        self.node.back_lights.turn_on([1,2,5,9,12])
        self.node.eyes.set_level(0.01)
        if k9stt is None:
            self.logger.error("STT interface not configured")
            self.blackboard.command = None
            return
        self.blackboard.command = k9stt.listen_for_command()
        self.node.eyes.set_level(0.0)

    def update(self):
        if self.blackboard.command:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class Responding(py_trees.behaviour.Behaviour):
    SPECIAL_INTENTS = {
        "ShowOff",
        "PlayChess",
    }

    def __init__(self, node, name="Interpret Command"):
        super().__init__(name)
        self.node = node
        self.result = py_trees.common.Status.FAILURE

    def initialise(self):
        self.result = py_trees.common.Status.FAILURE

        command = self.blackboard.command or ""

        if not command:
            self.logger.warning(
                "Interpret Command invoked without a command"
            )
            self.feedback_message = "No command"
            return

        self.node.back_lights.on()
        self.node.eyes.set_level(0.5)
        self.node.ears.think()

        intent = None
        answer = None

        normalised_command = command.casefold()

        try:
            # Local high-confidence intents can be detected without the LLM.
            if "play chess" in normalised_command:
                intent = "PlayChess"

            elif (
                "demonstration" in normalised_command
                or "show me what you can do" in normalised_command
                or "show off" in normalised_command
                or "demo" in normalised_command
            ):
                intent = "ShowOff"

            elif "thank" in normalised_command:
                answer = "Affirmative. Your gratitude has been noted."

            else:
                if k9qa is None:
                    self.logger.error(
                        "Question-answer interface is not configured"
                    )
                    self.feedback_message = "QA unavailable"
                    return

                intent, answer = k9qa.robot_response(command)

            self.blackboard.command = None
            self.blackboard.intent = intent

            # A specialist branch will process these on the next tree tick.
            if intent in self.SPECIAL_INTENTS:
                self.feedback_message = f"Queued intent: {intent}"
                self.result = py_trees.common.Status.SUCCESS
                return

            # Ordinary question or conversational response.
            if answer:
                self.node.voice.speak(answer)
                self._wait_for_speech()
            else:
                self.logger.warning(
                    f"No answer generated for intent {intent!r}"
                )

            # No specialist behaviour needs to handle this intent.
            self.blackboard.intent = None

            self.feedback_message = "Response completed"
            self.result = py_trees.common.Status.SUCCESS

        finally:
            self.node.ears.stop()
            self.node.back_lights.off()
            self.node.eyes.set_level(0.1)

    def _wait_for_speech(self):
        deadline = time.monotonic() + 30.0

        while self.node.is_talking:
            if time.monotonic() >= deadline:
                self.logger.warning(
                    "Timed out waiting for speech to finish"
                )
                break

            rclpy.spin_once(
                self.node,
                timeout_sec=0.1,
            )

        time.sleep(0.75)

    def update(self):
        return self.result


class IntentIs(py_trees.behaviour.Behaviour):
    """Succeeds only when the current intent matches expected_intent."""

    def __init__(self, expected_intent: str, name=None):
        super().__init__(
            name=name or f"Intent is {expected_intent}"
        )
        self.expected_intent = expected_intent

    def update(self):
        current_intent = self.blackboard.intent

        self.feedback_message = (
            f"expected={self.expected_intent!r}, "
            f"actual={current_intent!r}"
        )

        if current_intent == self.expected_intent:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE

class Demonstration(py_trees.behaviour.Behaviour):
    def __init__(self, node, name="Perform Demonstration"):
        super().__init__(name)
        self.node = node
        self.result = py_trees.common.Status.FAILURE

    def initialise(self):
        self.result = py_trees.common.Status.FAILURE

        # Defensive check: the sequence guard should already guarantee this.
        if self.blackboard.intent != "ShowOff":
            self.feedback_message = "ShowOff intent not present"
            return

        try:
            self.node.voice.speak("Starting demonstration")
            self._wait_for_speech()

            #
            # Put the actual demonstration actions here.
            #
            # self.node.tail.wag_h()
            # self.node.ears.scan()
            # self.node.back_lights.turn_on([...])
            #

            self.feedback_message = "Demonstration completed"
            self.result = py_trees.common.Status.SUCCESS

        finally:
            # Consume the intent even if part of the demonstration failed.
            self.blackboard.intent = None
            self.blackboard.command = None

    def _wait_for_speech(self):
        deadline = time.monotonic() + 30.0

        while self.node.is_talking:
            if time.monotonic() >= deadline:
                self.logger.warning(
                    "Timed out waiting for speech to finish"
                )
                break

            rclpy.spin_once(
                self.node,
                timeout_sec=0.1,
            )

    def update(self):
        return self.result


class PlayChess(py_trees.behaviour.Behaviour):
    def __init__(self, node, name="Start Chess"):
        super().__init__(name)
        self.node = node
        self.game = None
        self.result = py_trees.common.Status.FAILURE

    def initialise(self):
        self.result = py_trees.common.Status.FAILURE

        if self.blackboard.intent != "PlayChess":
            self.feedback_message = "PlayChess intent not present"
            return

        try:
            if ChessGame is None:
                self.logger.error("ChessGame is not configured")
                self.feedback_message = "Chess unavailable"
                return

            self.game = ChessGame()
            self.feedback_message = "Chess started"
            self.result = py_trees.common.Status.SUCCESS

        finally:
            self.blackboard.intent = None
            self.blackboard.command = None

    def update(self):
        return self.result


def create_behavior_tree(node):
    root = py_trees.composites.Selector(
        name="K9",
        memory=False,
    )

    demonstration_branch = py_trees.composites.Sequence(
        name="Handle Demonstration",
        memory=False,
        children=[
            IntentIs(
                expected_intent="ShowOff",
                name="Demonstration Requested?",
            ),
            Demonstration(node),
        ],
    )

    chess_branch = py_trees.composites.Sequence(
        name="Handle Chess",
        memory=False,
        children=[
            IntentIs(
                expected_intent="PlayChess",
                name="Chess Requested?",
            ),
            PlayChess(node),
        ],
    )

    interaction_branch = py_trees.composites.Sequence(
        name="Handle Voice Interaction",
        memory=True,
        children=[
            NotListening(node),
            WaitForHotword(node),
            Listening(node),
            Responding(node),
        ],
    )

    root.add_children([
        demonstration_branch,
        chess_branch,
        interaction_branch,
    ])

    return root


class K9BTNode(Node):
    def __init__(self):
        super().__init__('k9_bt_node')

        self.service_helper = ServiceClientHelper(self)

        # Create the Eyes, Tail, Ears, and BackLights client instances
        self.eyes = Eyes(self, self.service_helper)
        self.tail = Tail(self, self.service_helper)
        self.ears = Ears(self, self.service_helper)
        self.back_lights = BackLights(self, self.service_helper)
        self.voice = Voice(self, self.service_helper)

        self.is_talking = False

        self.blackboard_initialiser = initialise_blackboard()

        # Publisher for is_talking status
        # self.publisher = self.create_publisher(Bool, 'is_talking', 10)

        # Subscribe to the 'is_talking' and 'hotword_detected' topics
        self.subscription = self.create_subscription(
            Bool,
            'is_talking',
            self.is_talking_callback,
            10
        )

        self.hotword_subscription = self.create_subscription(
            Bool,
            'hotword_detected',
            self.hotword_detected_callback,
            10,
        )
        
        root = create_behavior_tree(self)
        self.bt = py_trees_ros.trees.BehaviourTree(
            root=root,
            unicode_tree_debug=False,
        )
        self.bt.setup(
            node=self,
            timeout=15.0,
        )
        # Configure the standard snapshot stream.
        #
        # Set the configuration parameters first, then enable the stream.
        # The order matters because enabling the stream reads the other
        # parameter values.
        self.set_parameters([
            Parameter(
                "default_snapshot_period",
                Parameter.Type.DOUBLE,
                0.5,
            ),
            Parameter(
                "default_snapshot_blackboard_data",
                Parameter.Type.BOOL,
                True,
            ),
            Parameter(
                "default_snapshot_blackboard_activity",
                Parameter.Type.BOOL,
                True,
            ),
        ])

        self.set_parameters([
            Parameter(
                "default_snapshot_stream",
                Parameter.Type.BOOL,
                True,
            ),
        ])
        # self.bt.tick_tock(period_ms=500)
        # Tree behaves syncrhonously bit tick_tock() uses callback
        # behaviours need to be rewritten to use asyn futures
    
    def is_talking_callback(self, msg):
        self.is_talking = msg.data

    def hotword_detected_callback(self, msg: Bool):
        if msg.data:
            self.blackboard.hotword_detected = True
            self.get_logger().info("Hotword detected")


class ServiceClientHelper:
    def __init__(self, node: Node):
        self.node = node
    def create_client(self, service_type, service_name):
        """Helper method to create service clients."""
        client = self.node.create_client(service_type, service_name)
        while not client.wait_for_service(timeout_sec=1.0):
            print(f"Waiting for service {service_name} to be available...")
        return client

    def call_service(self, client, request):
        """Call a service and return its response."""

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        if not future.done():
            self.node.get_logger().error("Service call did not complete")
            return None

        exception = future.exception()
        if exception is not None:
            self.node.get_logger().error(
                f"Service call failed: {exception}"
            )
            return None

        result = future.result()

        if result is None:
            self.node.get_logger().error("Service returned no response")
            return None

        self.node.get_logger().debug(
            f"Service response: {result}"
        )
        return result

def attach_blackboard_client(
    behaviour,
    read_keys=(),
    write_keys=(),
):
    """Attach a tracked blackboard client to a behaviour."""

    client = behaviour.attach_blackboard_client(
        name=f"{behaviour.name} Blackboard",
    )

    for key in read_keys:
        client.register_key(
            key=key,
            access=py_trees.common.Access.READ,
        )

    for key in write_keys:
        client.register_key(
            key=key,
            access=py_trees.common.Access.WRITE,
        )

    return client

def initialise_blackboard():
    client = py_trees.blackboard.Client(
        name="K9 Blackboard Initialiser",
    )

    initial_values = {
        "command": None,
        "intent": None,
        "speaking": 0.0,
        "hotword_detected": False,
    }

    for key, value in initial_values.items():
        client.register_key(
            key=key,
            access=py_trees.common.Access.WRITE,
        )
        client.set(
            name=key,
            value=value,
            overwrite=True,
        )

    return client

class Voice:
    def __init__(self, node: Node, service_helper: ServiceClientHelper):
        self.node = node
        self.service_helper = service_helper
        # Create clients for TTS services
        self.client_speak = self.service_helper.create_client(Speak, 'speak_now')
        self.client_cancel = self.service_helper.create_client(CancelSpeech, 'cancel_speech')

    def speak(self, text: str):
        """Call the Speak service to immediately speak the text."""
        request = Speak.Request()
        request.text = text
        response = self.service_helper.call_service(self.client_speak, request)
        if response and response.success:
            self.node.get_logger().info(f"Speaking: {text}")
        else:
            self.node.get_logger().error(f"Failed to speak: {text}")

    def cancel_speech(self):
        """Call the CancelSpeech service to cancel ongoing speech."""
        request = CancelSpeech.Request()
        response = self.service_helper.call_service(self.client_cancel, request)
        if response and response.success:
            self.node.get_logger().info("Speech canceled.")
        else:
            self.node.get_logger().error("Failed to cancel speech.")


class Eyes:
    def __init__(self, node: Node, service_helper: ServiceClientHelper):
        self.node = node
        self.service_helper = service_helper
        self.client_set_level = self.service_helper.create_client(SetBrightness, 'eyes_set_level')
        self.client_get_level = self.service_helper.create_client(GetBrightness, 'eyes_get_level')
        self.client_on = self.service_helper.create_client(Trigger, 'eyes_on')
        self.client_off = self.service_helper.create_client(Trigger, 'eyes_off')

        self._is_talking = False
        self._stored_level = 0.0  # Saved level before talking began

        # Subscribers
        self.subscription = self.node.create_subscription(
            Bool,
            'is_talking',
            self.talking_cb,
            10,
        )

    def talking_cb(self, msg: Bool):
        """Handles is_talking state and adjusts brightness."""
        if msg.data and not self._is_talking:
            self._stored_level = self.get_level()
            self.set_level(1.0)  # Eyes on with full brightness
            self._is_talking = True
            self.node.get_logger().info("Talking detected: eyes set to 100%")
        elif not msg.data and self._is_talking:
            # Stop talking: restore previous level
            self.set_level(self._stored_level)
            self._is_talking = False
            self.node.get_logger().info(f"Stopped talking: eyes restored to {self._stored_level:.2f}")

    def set_level(self, level: float):
        """Sets the brightness level of the eyes."""
        request = SetBrightness.Request()
        request.level = level
        self.service_helper.call_service(self.client_set_level, request)

    def get_level(self) -> float:
        """Gets the current brightness level of the eyes."""
        request = GetBrightness.Request()
        result = self.service_helper.call_service(self.client_get_level, request)
        return result.level if result else 0.0

    def on(self):
        """Turns the eyes on."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_on, request)

    def off(self):
        """Turns the eyes off."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_off, request)


class Tail:
    def __init__(self, node: Node, service_helper: ServiceClientHelper):
        self.node = node
        self.service_helper = service_helper
        self.client_wag_h = self.service_helper.create_client(Trigger, 'tail_wag_h')
        self.client_wag_v = self.service_helper.create_client(Trigger, 'tail_wag_v')
        self.client_center = self.service_helper.create_client(Trigger, 'tail_center')
        self.client_up = self.service_helper.create_client(Trigger, 'tail_up')
        self.client_down = self.service_helper.create_client(Trigger, 'tail_down')

    def wag_h(self):
        """Wag the tail horizontally."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_wag_h, request)

    def wag_v(self):
        """Wag the tail vertically."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_wag_v, request)

    def center(self):
        """Center the tail."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_center, request)

    def up(self):
        """Raise the tail."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_up, request)

    def down(self):
        """Lower the tail."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_down, request)


class Ears:
    def __init__(self, node: Node, service_helper: ServiceClientHelper):
        self.node = node
        self.service_helper = service_helper
        self.client_stop = self.service_helper.create_client(Trigger, 'ears_stop')
        self.client_scan = self.service_helper.create_client(Trigger, 'ears_scan')
        self.client_fast = self.service_helper.create_client(Trigger, 'ears_fast')
        self.client_think = self.service_helper.create_client(Trigger, 'ears_think')
        self.client_follow_read = self.service_helper.create_client(Trigger, 'ears_follow_read')
        self.client_safe_rotate = self.service_helper.create_client(Trigger, 'ears_safe_rotate')

    def stop(self):
        """Stop the ears from following."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_stop, request)

    def scan(self):
        """Start the ears scanning."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_scan, request)

    def fast(self):
        """Set ears to fast mode."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_fast, request)

    def think(self):
        """Set ears to think mode."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_think, request)

    def follow_read(self) -> float:
        request = Trigger.Request()
        response = self.service_helper.call_service(
            self.client_follow_read,
            request,
        )
        if response is None:
            return 0.0
        try:
            _, value = response.message.split(":", 1)
            return float(value.strip())

        except (AttributeError, IndexError, ValueError) as error:
            self.node.get_logger().warning(
                f"Invalid ears_follow_read response: {error}"
            )
            return 0.0

    def safe_rotate(self) -> bool:
        """Perform safe rotation check."""
        request = Trigger.Request()
        response = self.service_helper.call_service(self.client_safe_rotate, request)
        return response.success if response else False


class BackLights:
    def __init__(self, node: Node, service_helper: ServiceClientHelper):
        self.node = node
        self.service_helper = service_helper
        self.client_on = self.service_helper.create_client(Trigger, 'back_lights_on')
        self.client_off = self.service_helper.create_client(Trigger, 'back_lights_off')
        self.client_turn_on = self.service_helper.create_client(LightsControl, 'back_lights_turn_on')
        self.client_turn_off = self.service_helper.create_client(LightsControl, 'back_lights_turn_off')
        self.client_toggle = self.service_helper.create_client(LightsControl, 'back_lights_toggle')
        self.client_get_switch_state = self.service_helper.create_client(SwitchState, 'back_lights_get_switch_state')

    def on(self):
        """Turn the back lights on."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_on, request)

    def off(self):
        """Turn the back lights off."""
        request = Trigger.Request()
        self.service_helper.call_service(self.client_off, request)

    def turn_on(self, lights: list):
        """Turn specific lights on."""
        request = LightsControl.Request()
        request.lights = lights
        self.service_helper.call_service(self.client_turn_on, request)

    def turn_off(self, lights: list):
        """Turn specific lights off."""
        request = LightsControl.Request()
        request.lights = lights
        self.service_helper.call_service(self.client_turn_off, request)

    def toggle(self, lights: list):
        """Toggle specific back_lights."""
        request = LightsControl.Request()
        request.lights = lights
        self.service_helper.call_service(self.client_toggle, request)

    def get_switch_state(self) -> list:
        """Get the switch state of the back back_lights."""
        request = SwitchState.Request()
        response = self.service_helper.call_service(self.client_get_switch_state, request)
        return response.states if response else []

    def cmd(self, command:str):
        """Temporary compatibility method."""
        self.node.get_logger().warning(
            f"BackLights.cmd({command!r}) is not implemented; ignoring"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = K9BTNode()
        node.get_logger().info("K9 behaviour tree running")

        while rclpy.ok():
            # Process subscriptions such as is_talking.
            rclpy.spin_once(node, timeout_sec=0.1)

            # Tick outside an executor callback, allowing the current
            # synchronous service helper to spin for service responses.
            node.bt.tick()

            time.sleep(0.4)

            # when each behaviour uses non-blocking service futures:
            # self.bt.tick_tock(period_ms=500)
            # rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(
                f"Behaviour tree crashed: {error}"
            )
        raise

    finally:
        if node is not None:
            if hasattr(node, "bt"):
                node.bt.shutdown(destroy_node=False)

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
