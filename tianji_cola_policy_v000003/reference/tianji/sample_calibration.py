"""Sample-derived initial arm state and explicit, uncertainty-labelled camera models.

The configuration embeds small calibration/state values, never episode images or
runtime object poses. Camera poses use only public calibration and commanded FK.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/tianji-rgb/sample_calibration.json"


def _transform(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid {name}: expected a finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"Invalid homogeneous row in {name}")
    if (not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f"Invalid rotation in {name}")
    return matrix


@lru_cache(maxsize=4)
def load_calibration(path: Path = DEFAULT_CONFIG) -> dict:
    """Load an immutable-for-the-process configuration; source data need not be mounted."""
    result = json.loads(Path(path).read_text())
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported sample calibration schema")
    if result["units"]["arm"] != "radian" or result["units"]["gripper"] != (
        "normalized_opening_0_closed_1_open"
    ):
        raise ValueError("Unexpected sample action/state units")
    for name, row in result["cameras"].items():
        _transform(row["T_parent_camera"], name)
        intrinsic = np.asarray(row["intrinsic"], dtype=float)
        if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
            raise ValueError("Invalid sample camera intrinsic matrix: " + name)
    return result


def calibration_fingerprint(calibration: dict | None = None) -> str:
    """Hash the complete fixed configuration for run provenance."""
    calibration = calibration if calibration is not None else load_calibration()
    return hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()


def initial_action(episode: int | None = None, *, calibration: dict | None = None) -> np.ndarray:
    """Return the frame-zero MEASURED state, not the first commanded action."""
    calibration = calibration if calibration is not None else load_calibration()
    episode = calibration["default_episode"] if episode is None else episode
    profile = next(
        (row for row in calibration["initial_profiles"] if row["episode"] == episode), None,
    )
    if profile is None:
        raise ValueError(f"Unknown sample episode {episode}")
    return _action(profile["state"])


def _action(value) -> np.ndarray:
    action = np.asarray(value, dtype=float)
    if action.shape != (16,) or not np.isfinite(action).all():
        raise ValueError("Sample action must contain exactly 16 finite values")
    if np.any(action[[7, 15]] < 0) or np.any(action[[7, 15]] > 1):
        raise ValueError("Grippers are normalized openings, not raw driver radians")
    return action.copy()


def joint_angles(action=None, *, episode: int | None = None,
                 calibration: dict | None = None, camera_profile="urdf") -> dict[str, float]:
    """Map measured/commanded L7,opening,R7,opening into the current URDF joints.

    The 14 arm signs/offsets come from the dataset's delivered mapping. Gripper
    normalization is verified, but the Rhino transmission is still approximate.
    Unrecorded body/waist/head joints are intentionally not invented here.
    """
    calibration = calibration if calibration is not None else load_calibration()
    action = initial_action(episode, calibration=calibration) if action is None else _action(action)
    q14 = np.r_[action[:7], action[8:15]]
    result = {
        row["model_joint"]: float(row["sign"] * q14[row["index"]] + row["zero_offset_rad"])
        for row in calibration["joint_mapping"]
    }
    from .wrist_profile import drive_targets

    result.update(drive_targets(float(action[7]), "left", camera_profile))
    result.update(drive_targets(float(action[15]), "right", camera_profile))
    return result


def camera_model(description, command, camera: str, *, calibration: dict | None = None,
                 resolution: tuple[int, int] | None = None,
                 world_from_chassis=None, camera_profile="urdf") -> dict:
    """Compose world-from-OPTICAL using own commands; never measured simulator state.

    Selected physical URDF optical mounts use sample device intrinsics. Raw
    sample extrinsics and the rejected approximate PnP remain in provenance;
    this model does not claim independently calibrated hardware extrinsics.
    """
    calibration = calibration if calibration is not None else load_calibration()
    if camera not in calibration["cameras"]:
        raise ValueError("Unknown sample camera: " + camera)
    row = calibration["cameras"][camera]
    if camera_profile != "urdf":
        from .wrist_profile import PROFILES, alignment_profile

        if camera_profile not in PROFILES:
            raise ValueError("Unknown camera profile")
        if camera in ("left_wrist", "right_wrist"):
            hand = camera.split("_")[0]
            row = {**row, "parent_frame": hand + "_tool0",
                   "T_parent_camera": alignment_profile()["hands"][hand]["T_tool_camera"],
                   "calibration_status": "provisional_sample_finger_registration"}
    angles = description.horizontal_home()
    angles.update(joint_angles(command, calibration=calibration))
    poses = description.forward(angles)
    parent = row["parent_frame"]
    if parent == "dataset_common_base":
        # Both mounting matrices are known; this cancels their arbitrary common
        # base convention and preserves current robot morphology/stand height.
        parent_pose = poses["Base_L"] @ np.linalg.inv(
            _transform(calibration["dataset_common_base_from_Base_L"], "dataset Base_L mount")
        )
    elif parent in (
        "left_tool0", "right_tool0", "cam_high_color_optical_frame",
        "cam_left_wrist_color_optical_frame", "cam_right_wrist_color_optical_frame",
    ):
        parent_pose = poses[parent]
    else:
        raise ValueError("Unverified camera parent frame: " + str(parent))
    pose = parent_pose @ _transform(row["T_parent_camera"], camera)
    if world_from_chassis is not None:
        pose = _transform(world_from_chassis, "world_from_chassis") @ pose
    intrinsic = np.asarray(row["intrinsic"], dtype=float).copy()
    width, height = row["image_size"]
    if resolution is not None:
        target_width, target_height = resolution
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Camera resolution must be positive")
        intrinsic[0] *= target_width / width
        intrinsic[1] *= target_height / height
        width, height = target_width, target_height
    return dict(
        intrinsic=intrinsic.tolist(), optical_pose_from_command=pose.tolist(),
        image_size=[width, height], distortion_model=row["distortion_model"],
        distortion_coefficients=row["distortion_coefficients"],
        calibration_status=row["calibration_status"],
        source=("sample device intrinsics + selected static mount + own command FK; "
                "NOT measured pose"),
    )
