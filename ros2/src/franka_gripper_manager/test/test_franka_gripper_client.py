"""Exercise the Franka Hand client's two-position control without hardware."""

from concurrent.futures import Future
from ctypes import c_float
from types import SimpleNamespace
from unittest.mock import Mock

from action_msgs.msg import GoalInfo, GoalStatus
from action_msgs.srv import CancelGoal
from franka_msgs.action import Grasp, Move
import pytest
import rclpy
from rclpy.exceptions import InvalidParameterValueException
from rclpy.parameter import Parameter
from std_msgs.msg import Float32

from franka_gripper_manager import franka_gripper_client as client


class PendingGoal:
    """Allow each stage of an action to complete at a time chosen by a test."""

    def __init__(self, goal):
        self.goal = goal
        self.response = Future()
        self.result = Future()
        self.handle = Mock()
        self.handle.get_result_async.return_value = self.result
        cancel_reply = Future()
        cancel_reply.set_result(CancelGoal.Response(goals_canceling=[GoalInfo()]))
        self.handle.cancel_goal_async.return_value = cancel_reply

    def respond(self, accepted=True):
        """Deliver the server's acceptance or rejection."""
        self.handle.accepted = accepted
        self.response.set_result(self.handle)

    def finish(self, status=GoalStatus.STATUS_SUCCEEDED):
        """Deliver the terminal action status and matching typed result."""
        action_type = Grasp if isinstance(self.goal, Grasp.Goal) else Move
        result = action_type.Result(success=status == GoalStatus.STATUS_SUCCEEDED)
        self.result.set_result(SimpleNamespace(status=status, result=result))


@pytest.fixture
def gripper(monkeypatch):
    """Run the node with recorded actions and a homed 80 mm hand."""
    goals = []

    def send_goal(goal):
        pending = PendingGoal(goal)
        goals.append(pending)
        return pending.response

    def fake_client(node, action_type, action_name):
        action = Mock()
        action.send_goal_async.side_effect = send_goal
        return action

    def read_max_width(node, topic):
        node._max_width = 0.08

    monkeypatch.setattr(client, "ActionClient", fake_client)
    monkeypatch.setattr(client.GripperClient, "_home_gripper", lambda self, topic: None)
    monkeypatch.setattr(client.GripperClient, "_get_max_gripper_width", read_max_width)
    rclpy.init(args=[], domain_id=187)
    try:
        node = client.GripperClient()
        try:
            yield node, goals
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def send_openings(node, *openings):
    """Deliver trigger readings with the same precision as the Float32 wire format."""
    for value in openings:
        node._gripper_command_callback(Float32(data=c_float(value).value))


def test_thresholds_and_holding(gripper):
    """Crossing each threshold once issues exactly one action despite trigger jitter."""
    node, goals = gripper
    send_openings(node, 0.249, 0.101)
    assert not goals
    send_openings(node, 0.10)
    grasp, = goals
    assert isinstance(grasp.goal, Grasp.Goal)
    assert grasp.goal.force == 30.0
    assert grasp.goal.width == pytest.approx(0.04)
    assert grasp.goal.epsilon.inner == pytest.approx(0.04)
    assert grasp.goal.epsilon.outer == pytest.approx(0.04)
    grasp.respond()
    grasp.finish()

    send_openings(node, 0.0, 0.09, 0.101, 0.24, 0.11)
    assert goals == [grasp]
    send_openings(node, 0.25)
    _, release = goals
    assert isinstance(release.goal, Move.Goal)
    assert release.goal.width == pytest.approx(0.08)
    release.respond()
    release.finish()
    send_openings(node, 0.24, 0.7, 1.0, 0.101)
    assert goals == [grasp, release]


@pytest.mark.parametrize("release_before_acceptance", [True, False])
def test_release_waits_for_grasp_to_stop(gripper, release_before_acceptance):
    """A release during a grasp cancels it once and waits for its terminal result."""
    node, goals = gripper
    send_openings(node, 0.01)
    grasp, = goals
    if release_before_acceptance:
        send_openings(node, 0.8)
    grasp.respond()
    send_openings(node, 0.8, 1.0)
    grasp.handle.cancel_goal_async.assert_called_once_with()
    assert goals == [grasp]

    grasp.finish(GoalStatus.STATUS_CANCELED)
    _, release = goals
    assert isinstance(release.goal, Move.Goal)
    assert release.goal.width == pytest.approx(0.08)


def test_squeeze_again_while_canceling(gripper):
    """A fresh squeeze replaces the pending open after the old grasp is canceled."""
    node, goals = gripper
    send_openings(node, 0.0)
    first, = goals
    first.respond()
    send_openings(node, 1.0, 0.0)
    first.handle.cancel_goal_async.assert_called_once_with()
    assert goals == [first]
    first.finish(GoalStatus.STATUS_CANCELED)
    _, second = goals
    assert isinstance(second.goal, Grasp.Goal)
    second.respond()
    second.finish()
    send_openings(node, 0.01)
    assert goals == [first, second]


@pytest.mark.parametrize("accepted", [False, True])
def test_grasp_failure_requires_another_squeeze(gripper, accepted):
    """Both rejection and execution failure leave the squeezed trigger latched."""
    node, goals = gripper
    send_openings(node, 0.05)
    grasp, = goals
    grasp.respond(accepted=accepted)
    if accepted:
        grasp.finish(GoalStatus.STATUS_ABORTED)
    send_openings(node, 0.1, 0.0, 0.2)
    assert goals == [grasp]
    send_openings(node, 0.3)
    _, release = goals
    release.respond()
    release.finish()
    send_openings(node, 0.05)
    assert len(goals) == 3
    assert isinstance(goals[-1].goal, Grasp.Goal)


def test_squeeze_while_opening_waits_for_terminal_result(gripper):
    """New squeeze and open goals must never operate the same motors concurrently."""
    node, goals = gripper
    send_openings(node, 0.05)
    grasp, = goals
    grasp.respond()
    grasp.finish()
    send_openings(node, 0.9)
    _, release = goals
    release.respond()
    send_openings(node, 0.01, 0.05)
    assert goals == [grasp, release]
    release.finish()
    assert len(goals) == 3
    assert isinstance(goals[-1].goal, Grasp.Goal)


def test_force_changes_only_on_next_grasp(gripper):
    """The force range is inclusive, and updates never disturb a held object."""
    node, goals = gripper
    send_openings(node, 0.05)
    grasp, = goals
    grasp.respond()
    grasp.finish()
    for value in (29.999999, 70.000001, float("nan"), float("inf")):
        result, = node.set_parameters([Parameter("grasp_force", value=value)])
        assert not result.successful
        assert node.get_parameter("grasp_force").value == 30.0
    for value in (30.0, 70.0, 40.0):
        result, = node.set_parameters([Parameter("grasp_force", value=value)])
        assert result.successful
    send_openings(node, 0.0, 0.05)
    assert goals == [grasp]
    assert grasp.goal.force == 30.0
    send_openings(node, 0.4)
    _, release = goals
    release.respond()
    release.finish()
    send_openings(node, 0.0)
    assert len(goals) == 3
    assert goals[-1].goal.force == 40.0


def test_invalid_startup_force_is_rejected_before_homing(monkeypatch):
    """An invalid parameter override must fail before any hardware operation."""
    homing = Mock()
    monkeypatch.setattr(client.GripperClient, "_home_gripper", homing)
    rclpy.init(args=["--ros-args", "-p", "grasp_force:=75.0"], domain_id=187)
    try:
        with pytest.raises(InvalidParameterValueException):
            client.GripperClient()
        homing.assert_not_called()
    finally:
        rclpy.shutdown()
