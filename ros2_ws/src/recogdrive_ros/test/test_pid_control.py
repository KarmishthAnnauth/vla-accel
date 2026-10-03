import math

import numpy as np

import os

import pytest

from recogdrive_ros.pid_control import compute_control, load_pid_controller, rebase_plan, to_pid_frame

TEAM_CODE = os.environ.get("PID_CONTROLLER_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../../Orion/team_code"))
if not os.path.isfile(os.path.join(TEAM_CODE, "pid_controller.py")):
    pytest.skip("no Bench2Drive pid_controller.py (set PID_CONTROLLER_DIR)", allow_module_level=True)
PIDController = load_pid_controller(TEAM_CODE)


def straight(speed, n=8, dt=0.5):
    return np.array([[speed * dt * (j + 1), 0.0] for j in range(n)])


def arc(speed, yaw_rate, n=8, dt=0.5):
    r = speed / yaw_rate
    return np.array([[r * math.sin(yaw_rate * dt * (j + 1)), r * (1 - math.cos(yaw_rate * dt * (j + 1)))]
                     for j in range(n)])


def test_pid_frame_is_right_forward():
    assert np.allclose(to_pid_frame(np.array([[2.0, 0.5]])), [[-0.5, 2.0]])


def test_straight_plan_accelerates_without_steering():
    steer, throttle, brake, meta = compute_control(PIDController(), straight(4.0), speed=1.0)
    assert abs(steer) < 1e-6 and throttle > 0.0 and brake == 0.0
    assert abs(meta["desired_speed"] - 4.0) < 1e-6


def test_left_turn_steers_left_right_turn_steers_right():
    left, _, _, _ = compute_control(PIDController(), arc(4.0, 0.4), speed=4.0)
    right, _, _, _ = compute_control(PIDController(), arc(4.0, -0.4), speed=4.0)
    assert left < -0.02 and right > 0.02
    assert abs(left + right) < 1e-9


def test_brakes_when_faster_than_the_plan_and_when_the_plan_stops():
    _, throttle, brake, _ = compute_control(PIDController(), straight(2.0), speed=4.0)
    assert throttle == 0.0 and brake == 1.0
    _, throttle, brake, _ = compute_control(PIDController(), np.zeros((8, 2)), speed=0.0)
    assert throttle == 0.0 and brake == 1.0


def test_speed_cap_cuts_throttle_only_above_it():
    _, capped, _, _ = compute_control(PIDController(), straight(8.0), speed=6.0)
    _, free, _, _ = compute_control(PIDController(), straight(8.0), speed=6.0, speed_cap_mps=0.0)
    assert capped == 0.0 and free > 0.0


def test_rebase_follows_the_ego_along_the_plan():
    plan = straight(4.0)
    frame = np.array([10.0, -3.0, 0.7])
    now = np.array([10.0 + 4.0 * math.cos(0.7), -3.0 + 4.0 * math.sin(0.7), 0.7])
    out = rebase_plan(plan, 1.0, frame, now)
    assert np.allclose(out, straight(4.0), atol=1e-9)
    assert np.allclose(rebase_plan(plan, 5.0, frame, frame)[0], [22.0, 0.0])


def test_rebase_shows_lateral_error_when_the_ego_drifted():
    frame = np.zeros(3)
    out = rebase_plan(straight(4.0), 0.0, frame, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(out[:, 1], -1.0)
