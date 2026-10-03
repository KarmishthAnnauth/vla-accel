"""Tests for recogdrive_ros.agent_inputs.

The first group needs numpy only.  The last test compares ``build_agent_input``
against NAVSIM's own ``AgentInput.from_scene_dict_list`` and is skipped where
the ReCogDrive checkout / nuplan-devkit are not importable (run it in the
container: start_recogdrive.sh --shell, then
``python3 -m pytest /benchmarking/alpamayo-autoware/src/recogdrive_ros/test``).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recogdrive_ros import agent_inputs as ai


def _drive(history, t0, t1, hz=20.0, speed=4.0, yaw=0.0, x0=0.0, y0=0.0):
    """Constant speed along `yaw` from (x0, y0), sampled at `hz`."""
    n = int(round((t1 - t0) * hz))
    for k in range(n + 1):
        t = t0 + k / hz
        d = speed * (t - t0)
        history.add(t, x0 + d * math.cos(yaw), y0 + d * math.sin(yaw), yaw, speed, 0.0)


def test_road_option_mapping():
    assert ai.road_option_to_command(1) == ai.CMD_LEFT
    assert ai.road_option_to_command(2) == ai.CMD_RIGHT
    assert ai.road_option_to_command(3) == ai.CMD_STRAIGHT
    assert ai.road_option_to_command(4) == ai.CMD_STRAIGHT
    assert ai.road_option_to_command(5) == ai.CMD_LEFT
    assert ai.road_option_to_command(6) == ai.CMD_RIGHT
    assert ai.road_option_to_command(-1) == ai.CMD_STRAIGHT
    assert ai.road_option_to_command(255) == ai.CMD_STRAIGHT


def test_command_one_hot_never_unknown():
    assert ai.command_one_hot(ai.CMD_LEFT).tolist() == [1, 0, 0, 0]
    assert ai.command_one_hot(ai.CMD_STRAIGHT).tolist() == [0, 1, 0, 0]
    assert ai.command_one_hot(ai.CMD_RIGHT).tolist() == [0, 0, 1, 0]
    with pytest.raises(ValueError):
        ai.command_one_hot(ai.CMD_UNKNOWN)


def test_window_is_four_frames_half_a_second_apart():
    h = ai.PoseHistory()
    _drive(h, 0.0, 3.0, speed=4.0)
    poses, vels, held = h.window(3.0)
    assert held == 0
    assert poses.shape == (4, 3) and vels.shape == (4, 2)
    np.testing.assert_allclose(poses[:, 0], [6.0, 8.0, 10.0, 12.0], atol=1e-9)
    np.testing.assert_allclose(vels[:, 0], 4.0)


def test_window_interpolates_between_samples():
    h = ai.PoseHistory()
    _drive(h, 0.0, 3.0, hz=20.0, speed=4.0)
    poses, _, held = h.window(2.975)
    assert held == 0
    np.testing.assert_allclose(poses[-1, 0], 4.0 * 2.975, atol=1e-9)
    np.testing.assert_allclose(poses[0, 0], 4.0 * 1.475, atol=1e-9)


def test_window_holds_the_oldest_pose_when_history_is_short():
    h = ai.PoseHistory()
    _drive(h, 10.0, 10.6, speed=2.0, x0=5.0)
    poses, _, held = h.window(10.6)
    assert held == 2
    np.testing.assert_allclose(poses[0], [5.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(poses[1], [5.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(poses[2, 0], 5.2, atol=1e-9)
    np.testing.assert_allclose(poses[3, 0], 6.2, atol=1e-9)


def test_yaw_interpolates_along_the_shorter_arc():
    h = ai.PoseHistory()
    h.add(0.0, 0.0, 0.0, math.radians(179.0), 0.0, 0.0)
    h.add(1.0, 0.0, 0.0, math.radians(-179.0), 0.0, 0.0)
    pose, _ = h.sample(0.5)
    assert abs(abs(pose[2]) - math.pi) < 1e-9


def test_time_running_backwards_restarts_the_buffer():
    h = ai.PoseHistory()
    _drive(h, 100.0, 102.0)
    assert h.add(3.0, 0.0, 0.0, 0.0, 0.0, 0.0) is True
    assert len(h) == 1 and h.oldest_time == 3.0


def test_buffer_is_trimmed_to_max_age():
    h = ai.PoseHistory(max_age_sec=2.0)
    _drive(h, 0.0, 10.0)
    assert h.newest_time == pytest.approx(10.0)
    assert h.oldest_time >= 8.0 - 0.051


def test_finite_difference_velocity_is_in_the_ego_frame():
    h = ai.PoseHistory()
    _drive(h, 0.0, 2.0, speed=3.0, yaw=math.pi / 2)
    v = h.finite_difference_velocity(2.0)
    np.testing.assert_allclose(v, [3.0, 0.0], atol=1e-6)


def test_shift_along_heading():
    p = ai.shift_along_heading(np.array([1.0, 2.0, math.pi / 2]), -1.5)
    np.testing.assert_allclose(p, [1.0, 0.5, math.pi / 2], atol=1e-12)


def test_trajectory_frame_round_trip():
    rng = np.random.default_rng(0)
    poses = rng.normal(size=(8, 3))
    frame = np.array([12.0, -7.0, 2.1])
    world = ai.trajectory_to_world(poses, frame)
    back = ai.world_to_frame(world, frame)
    np.testing.assert_allclose(back[:, :2], poses[:, :2], atol=1e-9)
    np.testing.assert_allclose(np.sin(back[:, 2]), np.sin(poses[:, 2]), atol=1e-9)
    np.testing.assert_allclose(np.cos(back[:, 2]), np.cos(poses[:, 2]), atol=1e-9)


def test_rebase_moves_the_plan_by_the_distance_driven():
    poses = np.array([[2.0 * (i + 1), 0.0, 0.0] for i in range(8)])
    then = np.array([0.0, 0.0, 0.0])
    now = np.array([3.0, 0.0, 0.0])
    local = ai.world_to_frame(ai.trajectory_to_world(poses, then), now)
    np.testing.assert_allclose(local[:, 0], poses[:, 0] - 3.0, atol=1e-12)


def test_segment_speeds():
    pts = np.array([[1.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(ai.segment_speeds(pts, np.zeros(2)), [2.0, 4.0, 8.0])


def test_build_agent_input_matches_navsim_from_scene_dict_list(tmp_path):
    repo = os.environ.get("RECOGDRIVE_REPO", "/benchmarking/recogdrive")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    pytest.importorskip("nuplan")
    pytest.importorskip("navsim.common.dataclasses")
    from pyquaternion import Quaternion
    from navsim.common.dataclasses import AgentInput, SensorConfig

    h = ai.PoseHistory()
    for k in range(61):
        t = k * 0.05
        yaw = 0.2 * t
        h.add(t, 30.0 + 5.0 * t * math.cos(yaw), -4.0 + 5.0 * t * math.sin(yaw), yaw, 5.0, 0.1)
    poses, vels, held = h.window(3.0)
    assert held == 0
    accel = np.array([0.3, -0.2])
    command = ai.command_one_hot(ai.CMD_LEFT)
    image_path = str(tmp_path / "cam_f0.bmp")
    ours = ai.build_agent_input(poses, vels, accel, command, image_path)

    scene = []
    for i in range(4):
        q = Quaternion(axis=[0, 0, 1], radians=poses[i, 2])
        scene.append({
            "ego2global_translation": [poses[i, 0], poses[i, 1], 0.0],
            "ego2global_rotation": [q.w, q.x, q.y, q.z],
            "ego_dynamic_state": [vels[i, 0], vels[i, 1], accel[0], accel[1]],
            "driving_command": command,
            "cams": {},
            "lidar_path": "none.pcd",
        })
    no_sensors = SensorConfig.build_no_sensors()
    scene_cams = {name.upper(): {"data_path": "x.jpg", "sensor2lidar_rotation": None,
                                 "sensor2lidar_translation": None, "cam_intrinsic": None,
                                 "distortion": None}
                  for name in ("cam_f0", "cam_l0", "cam_l1", "cam_l2",
                               "cam_r0", "cam_r1", "cam_r2", "cam_b0")}
    for frame in scene:
        frame["cams"] = scene_cams
    ref = AgentInput.from_scene_dict_list(scene, tmp_path, 4, no_sensors)

    assert len(ours.ego_statuses) == len(ref.ego_statuses) == 4
    for a, b in zip(ours.ego_statuses, ref.ego_statuses):
        assert a.ego_pose.dtype == b.ego_pose.dtype == np.float32
        np.testing.assert_allclose(a.ego_pose, b.ego_pose, atol=1e-5)
        np.testing.assert_array_equal(a.ego_velocity, b.ego_velocity)
        np.testing.assert_array_equal(a.ego_acceleration, b.ego_acceleration)
        np.testing.assert_array_equal(a.driving_command, b.driving_command)
    assert ours.cameras[-1].cam_f0.image == image_path
    np.testing.assert_allclose(ours.ego_statuses[-1].ego_pose, 0.0, atol=1e-6)
