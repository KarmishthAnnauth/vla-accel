"""Constants copied VERBATIM from MindDrive/team_code/minddrive_b2d_agent.py.

They are part of the model's input contract (the calibration is hardcoded for
the Bench2Drive six-camera rig at 1600x900) and of its control law.  Do not
edit the values; if the upstream agent changes, re-copy.  Kept in one module so
the ROS node, the smoke test and the parity tests all read the same numbers.
"""
import numpy as np

CAMERA_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

LIDAR2IMG = {
    "CAM_FRONT": np.array([[1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                           [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                           [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    "CAM_FRONT_LEFT": np.array([[6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                                [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    "CAM_FRONT_RIGHT": np.array([[1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                                 [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                 [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    "CAM_BACK": np.array([[-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
                          [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                          [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    "CAM_BACK_LEFT": np.array([[-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                               [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                               [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    "CAM_BACK_RIGHT": np.array([[3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                                [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
}
LIDAR2CAM = {
    "CAM_FRONT": np.array([[1., 0., 0., 0.],
                           [0., 0., -1., -0.24],
                           [0., 1., 0., -1.19],
                           [0., 0., 0., 1.]]),
    "CAM_FRONT_LEFT": np.array([[0.57357644, 0.81915204, 0., -0.22517331],
                                [0., 0., -1., -0.24],
                                [-0.81915204, 0.57357644, 0., -0.82909407],
                                [0., 0., 0., 1.]]),
    "CAM_FRONT_RIGHT": np.array([[0.57357644, -0.81915204, 0., 0.22517331],
                                 [0., 0., -1., -0.24],
                                 [0.81915204, 0.57357644, 0., -0.82909407],
                                 [0., 0., 0., 1.]]),
    "CAM_BACK": np.array([[-1., 0., 0., 0.],
                          [0., 0., -1., -0.24],
                          [0., -1., 0., -1.61],
                          [0., 0., 0., 1.]]),
    "CAM_BACK_LEFT": np.array([[-0.34202014, 0.93969262, 0., -0.25388956],
                               [0., 0., -1., -0.24],
                               [-0.93969262, -0.34202014, 0., -0.49288953],
                               [0., 0., 0., 1.]]),
    "CAM_BACK_RIGHT": np.array([[-0.34202014, -0.93969262, 0., 0.25388956],
                                [0., 0., -1., -0.24],
                                [0.93969262, -0.34202014, 0., -0.49288953],
                                [0., 0., 0., 1.]]),
}
LIDAR2EGO = np.array([[0., 1., 0., -0.39],
                      [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84],
                      [0., 0., 0., 1.]])

AGENT_GNSS_MOUNT_X = -1.4

SPEED_CAP_MPS = 5.0

ROUTE_MIN_DIST = 4.0
ROUTE_MAX_DIST = 50.0

TRAJ_DT = 0.5
FUT_TS = 6
FUT_PS = 20

MEMORY_MAX_DT = 2.0
AGENT_HZ = 20.0


def command2hot(command: int, max_dim: int = 6) -> np.ndarray:
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot


def command2nohot(command: int, max_dim: int = 6) -> int:
    if command < 0:
        command = 4
    command -= 1
    return command


def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


_CUSTOM_FP16 = dict(map_head=False, pts_bbox_head=False)


def custom_wrap_fp16_model(model) -> None:
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True
    for module_name, v in _CUSTOM_FP16.items():
        model._modules[module_name].fp16_enabled = v


def build_agent_results(images, can_bus, ego_theta, command, scene_token, frame_idx,
                        timestamp, get_box_type):
    """The `results` dict of MinddriveAgent.run_step, field for field, from
    already-decoded BGR images and an already-built can_bus.  `ego_theta` is
    the agent's ego_theta (vehicle->world yaw, ENU)."""
    from pyquaternion import Quaternion

    results = {}
    results["lidar2img"] = []
    results["lidar2cam"] = []
    results["cam_intrinsic"] = []
    results["img"] = []
    results["folder"] = " "
    results["scene_token"] = scene_token
    results["frame_idx"] = frame_idx
    results["timestamp"] = timestamp
    results["box_type_3d"], _ = get_box_type("LiDAR")
    for cam in CAMERA_ORDER:
        results["lidar2img"].append(LIDAR2IMG[cam])
        results["lidar2cam"].append(LIDAR2CAM[cam])
        results["cam_intrinsic"].append(np.matmul(LIDAR2IMG[cam], np.linalg.inv(LIDAR2CAM[cam])))
        results["img"].append(images[cam])
    results["lidar2img"] = np.stack(results["lidar2img"], axis=0)
    results["lidar2cam"] = np.stack(results["lidar2cam"], axis=0)
    results["can_bus"] = can_bus
    results["command"] = command2nohot(command)
    results["ego_fut_cmd"] = command2hot(command)

    ego2world = np.eye(4)
    ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
    ego2world[0:2, 3] = can_bus[0:2]
    lidar2global = ego2world @ LIDAR2EGO
    ego_pose = lidar2global
    ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
    results["ego_pose"] = ego_pose
    results["ego_pose_inv"] = ego_pose_inv
    results["lidar2ego"] = LIDAR2EGO
    results["l2g_r_mat"] = lidar2global[0:3, 0:3]
    results["l2g_t"] = lidar2global[0:3, 3]
    stacked_imgs = np.stack(results["img"], axis=-1)
    results["img_shape"] = stacked_imgs.shape
    results["ori_shape"] = stacked_imgs.shape
    results["pad_shape"] = stacked_imgs.shape
    return results


def batch_to_device(batch, device) -> None:
    """MinddriveAgent.run_step's exact per-key H2D loop (incl. nested input_ids)."""
    import torch
    for key, data in batch.items():
        if key != "img_metas":
            if torch.is_tensor(data[0]):
                data[0] = data[0].to(device)
        if key == "input_ids":
            for i in range(len(data[0])):
                for k in range(len(data[0][i])):
                    data[0][i][k] = data[0][i][k].to(device)
