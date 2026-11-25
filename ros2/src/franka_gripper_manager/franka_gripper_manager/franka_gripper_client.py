"""Control a Franka Hand with the two ends of the GELLO trigger travel."""

import math
import time
from ctypes import c_float

import rclpy
from action_msgs.msg import GoalStatus
from franka_msgs.action import Grasp, Homing, Move
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

DEFAULT_MOVE_ACTION_TOPIC = "franka_gripper/move"
DEFAULT_GRASP_ACTION_TOPIC = "franka_gripper/grasp"
DEFAULT_HOMING_ACTION_TOPIC = "franka_gripper/homing"
DEFAULT_JOINT_STATES_TOPIC = "franka_gripper/joint_states"
DEFAULT_GRIPPER_COMMAND_TOPIC = "gripper/gripper_client/target_gripper_width_percent"
# Compare against the same precision as the incoming ROS Float32.
CLOSE_THRESHOLD = c_float(0.10).value
OPEN_THRESHOLD = 0.25
FINGER_SPEED = 0.03  # m/s


class GripperClient(Node):
    """Latch one grasp or full-open command per trigger transition."""

    def __init__(self):
        super().__init__("gripper_client")

        move_action_topic = self.declare_parameter(
            "move_action_topic", DEFAULT_MOVE_ACTION_TOPIC
        ).value
        grasp_action_topic = self.declare_parameter(
            "grasp_action_topic", DEFAULT_GRASP_ACTION_TOPIC
        ).value
        homing_action_topic = self.declare_parameter(
            "homing_action_topic", DEFAULT_HOMING_ACTION_TOPIC
        ).value
        gripper_command_topic = self.declare_parameter(
            "gripper_command_topic", DEFAULT_GRIPPER_COMMAND_TOPIC
        ).value
        joint_states_topic = self.declare_parameter(
            "joint_states_topic", DEFAULT_JOINT_STATES_TOPIC
        ).value
        self.add_on_set_parameters_callback(self._validate_force)
        self.declare_parameter(
            "grasp_force",
            30.0,
            ParameterDescriptor(
                description="Grasping force in newtons, applied to the next grasp.",
                additional_constraints="30.0 to 70.0 N, inclusive.",
            ),
        )

        self._ACTION_SERVER_TIMEOUT = 10.0
        self._requested_action = Move  # Homing leaves the hand open.
        self._active_action = None
        self._goal_handle = None
        self._cancel_requested = False

        self.get_logger().info("Initializing gripper client...")
        self._home_gripper(homing_action_topic)
        self._get_max_gripper_width(joint_states_topic)
        if not math.isfinite(self._max_width) or self._max_width <= 0.0:
            raise RuntimeError(f"Invalid gripper width after homing: {self._max_width}")

        self._move_client = ActionClient(self, Move, move_action_topic)
        self._grasp_client = ActionClient(self, Grasp, grasp_action_topic)
        for action, topic in (
            (self._move_client, move_action_topic),
            (self._grasp_client, grasp_action_topic),
        ):
            if not action.wait_for_server(timeout_sec=self._ACTION_SERVER_TIMEOUT):
                raise RuntimeError(f"Gripper action server unavailable: {topic}")

        self.get_logger().info("Subscribing to gripper commands...")
        self._gripper_command_subscription = self.create_subscription(
            Float32, gripper_command_topic, self._gripper_command_callback, 10
        )
        self.get_logger().info("Gripper client initialized!")

    @staticmethod
    def _validate_force(parameters):
        for parameter in parameters:
            if parameter.name == "grasp_force" and (
                not isinstance(parameter.value, float) or not 30.0 <= parameter.value <= 70.0
            ):
                return SetParametersResult(successful=False, reason="grasp_force must be 30–70 N")
        return SetParametersResult(successful=True)

    def _home_gripper(self, homing_action_topic: str) -> None:
        self.get_logger().info("Starting gripper homing...")
        homing_client = ActionClient(self, Homing, homing_action_topic)

        self.get_logger().info(f"Waiting for homing action server {homing_action_topic}...")
        if not homing_client.wait_for_server(timeout_sec=self._ACTION_SERVER_TIMEOUT):
            raise RuntimeError(
                f"Homing action server not available after {self._ACTION_SERVER_TIMEOUT} seconds!"
            )

        self.get_logger().info("Homing action server found!")
        goal_msg = Homing.Goal()
        future = homing_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError("Homing action rejected!")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        time.sleep(2)

        if result.result.success:
            self.get_logger().info("Gripper homing successful!")
        else:
            raise RuntimeError("Gripper homing failed!")

    def _get_max_gripper_width(self, joint_states_topic: str) -> None:
        self.get_logger().info("Readout maximum gripper width...")
        future = rclpy.task.Future()

        def joint_state_callback(msg):
            _INDEX_FINGER_LEFT = 0
            self._max_width = 2 * msg.position[_INDEX_FINGER_LEFT]
            self.get_logger().info(f"Maximum gripper width determined: {self._max_width}")
            future.set_result(True)

        self.get_logger().info(f"Subscribing to {joint_states_topic}...")
        gripper_subscription = self.create_subscription(
            JointState, joint_states_topic, joint_state_callback, 10
        )

        self.get_logger().info(f"Waiting for {joint_states_topic}...")
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Unsubscribing from {joint_states_topic}")
        self.destroy_subscription(gripper_subscription)

    def _gripper_command_callback(self, msg: Float32) -> None:
        if 0.0 <= msg.data <= CLOSE_THRESHOLD:
            requested = Grasp
        elif OPEN_THRESHOLD <= msg.data <= 1.0:
            requested = Move
        else:
            return
        if requested is self._requested_action:
            return
        self._requested_action = requested
        if self._active_action is None:
            self._send_requested_goal()
        else:
            self._cancel_grasp_if_opening()

    def _send_requested_goal(self) -> None:
        self._active_action = self._requested_action
        if self._active_action is Grasp:
            client = self._grasp_client
            goal = Grasp.Goal()
            # Accept contact throughout the measured stroke for objects of unknown width.
            goal.width = self._max_width / 2
            goal.epsilon.inner = goal.width
            goal.epsilon.outer = goal.width
            goal.force = self.get_parameter("grasp_force").value
        else:
            client = self._move_client
            goal = Move.Goal()
            goal.width = self._max_width
        goal.speed = FINGER_SPEED
        client.send_goal_async(goal).add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future: rclpy.task.Future) -> None:
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().warning(f"{self._active_action.__name__} goal rejected")
            self._finish_action()
            return
        self._goal_handle.get_result_async().add_done_callback(self._on_result)
        self._cancel_grasp_if_opening()

    def _cancel_grasp_if_opening(self) -> None:
        if (
            self._active_action is Grasp
            and self._requested_action is Move
            and self._goal_handle is not None
            and not self._cancel_requested
        ):
            self._cancel_requested = True
            # The cancel response is only an acknowledgement. Wait for the grasp result
            # before sending Move, so the server cannot stop the new opening motion.
            self._goal_handle.cancel_goal_async()

    def _on_result(self, future: rclpy.task.Future) -> None:
        response = future.result()
        succeeded = response.status == GoalStatus.STATUS_SUCCEEDED and response.result.success
        if not succeeded and response.status != GoalStatus.STATUS_CANCELED:
            self.get_logger().warning(
                f"{self._active_action.__name__} failed (status {response.status}): "
                f"{response.result.error or 'unsuccessful gripper result'}"
            )
        self._finish_action(retry_canceled=self._cancel_requested and not succeeded)

    def _finish_action(self, retry_canceled: bool = False) -> None:
        # If the user squeezed again during cancellation, honor the new squeeze.
        send_next = self._requested_action is not self._active_action or retry_canceled
        self._goal_handle = None
        self._active_action = None
        self._cancel_requested = False
        if send_next:
            self._send_requested_goal()


def main(args=None):
    """Run the real Franka Hand client until interrupted."""
    rclpy.init(args=args)
    try:
        gripper_client = GripperClient()
        try:
            rclpy.spin(gripper_client)
        finally:
            gripper_client.destroy_node()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
