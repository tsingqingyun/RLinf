"""
Constants for robocasa v1.0.0 ("365") environment.

vs robocasa v0.2.0:
  - Default image_space: "3views" (adds robot0_agentview_right)
  - Default camera resolution: 256x256 (was 224x224)
  - Raw obs keys (robot0_eef_pos, etc.) are unchanged
  - Action space: still 12D, same layout
  - State vector: still 25D, same layout
"""

import os
from typing import Union

import numpy as np


# Default RoboCasa v1.0 checkout path.
# Override with:
#   export ROBOCASA365_PATH=/path/to/robocasa
DEFAULT_ROBOCASA365_PATH = "/home/njc/robocasa"


def get_robocasa365_source_path() -> str:
    """
    Absolute path to the RoboCasa v1.0 source tree (prepended to ``sys.path``).

    Uses ``ROBOCASA365_PATH`` if set.

    Otherwise, uses ``DEFAULT_ROBOCASA365_PATH`` (default: ``/home/njc/robocasa``)
    if it exists; if not, falls back to ``~/robocasa``.

    This avoids accidentally importing the optional editable install under
    RLinf's ``.venv/robocasa``.
    """
    p = os.environ.get("ROBOCASA365_PATH", "").strip()
    if p and os.path.isdir(p):
        return os.path.abspath(p)

    if os.path.isdir(DEFAULT_ROBOCASA365_PATH):
        return os.path.abspath(DEFAULT_ROBOCASA365_PATH)

    return os.path.abspath(os.path.expanduser("~/robocasa"))


# ---------------------------------------------------------------------------
# STATE — identical to v0.2.0
# ---------------------------------------------------------------------------

STATE_DIM_365 = 25

ROBOCASA365_STATES = {
    "robot0_eef_pos":          np.arange(0, 3),
    "robot0_eef_quat":         np.arange(3, 7),
    "robot0_gripper_qpos":     np.arange(7, 9),
    "robot0_gripper_qvel":     np.arange(9, 11),
    "robot0_base_to_eef_pos":  np.arange(11, 14),
    "robot0_base_to_eef_quat": np.arange(14, 18),
    "robot0_base_pos":         np.arange(18, 21),
    "robot0_base_quat":        np.arange(21, 25),
}

STATE_SPACE_STR_MAPPING_365 = {
    "25d": list(ROBOCASA365_STATES.keys()),
}


def get_state_space_365(state_space: Union[str, list]) -> list:
    if isinstance(state_space, str):
        return STATE_SPACE_STR_MAPPING_365.get(state_space, [])
    return state_space


# ---------------------------------------------------------------------------
# ACTION — identical to v0.2.0
# ---------------------------------------------------------------------------

ROBOCASA365_ALL_ACTION_DIM = 12

ROBOCASA365_ACTIONS = {
    "rel_pose_6d": np.arange(0, 6),
    "gripper":     np.arange(6, 7),
    "base":        np.arange(7, 10),
    "torso":       np.arange(10, 11),
    "base_mode":   np.arange(11, 12),
}

ROBOCASA365_DEFAULT_ACTION = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
)

ACTION_SPACE_STR_MAPPING_365 = {
    "12d": list(ROBOCASA365_ACTIONS.keys()),
    "7d":  ["rel_pose_6d", "gripper"],
}


def get_action_space_365(action_space: Union[str, list]) -> list:
    if isinstance(state_space := action_space, str):
        return ACTION_SPACE_STR_MAPPING_365.get(state_space, [])
    return action_space


def get_action_ids_365(action_space: list) -> list:
    ids = []
    for name in action_space:
        ids.extend(ROBOCASA365_ACTIONS[name].tolist())
    return ids


# ---------------------------------------------------------------------------
# IMAGE / CAMERA
#
# v1.0.0 default: 3 cameras at 256x256
#   robot0_agentview_left   (same as v0.2)
#   robot0_agentview_right  (NEW — was optional in v0.2, now standard)
#   robot0_eye_in_hand      (same as v0.2)
# ---------------------------------------------------------------------------

# obs_key (in RLinf) -> robosuite camera name passed to robosuite.make()
OBS_KEY_CAMERA_NAME_MAPPING_365 = {
    "observation/image":            "robot0_agentview_left",
    "observation/wrist_image":      "robot0_eye_in_hand",
    "observation/extra_view_image": "robot0_agentview_right",
}

# internal obs dict key (in RLinf) -> raw key in robosuite obs dict
OBS_KEY_ROBOCASA365_IMAGE_MAPPING = {
    "main_images":       "robot0_agentview_left_image",
    "wrist_images":      "robot0_eye_in_hand_image",
    "extra_view_images": "robot0_agentview_right_image",  # added in 3views
}

# Preset image spaces:
IMAGE_SPACE_STR_MAPPING_365 = {
    "2views": [                       # backward compat with v0.2 configs
        "observation/image",
        "observation/wrist_image",
    ],
    "3views": [                       # v1.0.0 new default
        "observation/image",
        "observation/wrist_image",
        "observation/extra_view_image",
    ],
}


def get_image_space_365(image_space: Union[str, list]) -> list:
    if isinstance(image_space, str):
        return IMAGE_SPACE_STR_MAPPING_365.get(image_space, [])
    return image_space
