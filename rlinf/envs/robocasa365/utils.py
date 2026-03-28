"""
Constants for robocasa365 environment.

After running probe_robocasa365.py on the new environment, update:
  - STATE_DIM_365          : total state vector length
  - ROBOCASA365_STATES     : mapping from field name -> slice in state vector
  - ROBOCASA365_ALL_ACTION_DIM : total action dimensions
  - OBS_KEY_CAMERA_NAME_MAPPING_365 : obs key -> robosuite camera name
  - OBS_KEY_ROBOCASA365_IMAGE_MAPPING : internal key -> obs key name in env
  - IMAGE_SPACE_STR_MAPPING_365 : preset name -> list of obs keys
"""

from typing import Union
import numpy as np


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

# Current assumption: same 25D layout as robocasa v0.2.
# Update after running probe_robocasa365.py if dims differ.
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
    # If robocasa365 adds joint positions, uncomment and update STATE_DIM_365:
    # "robot0_joint_pos":      np.arange(25, 32),
}

STATE_SPACE_STR_MAPPING_365 = {
    "25d": list(ROBOCASA365_STATES.keys()),
    # Add new presets here if robocasa365 has different standard layouts
}


def get_state_space_365(state_space: Union[str, list]) -> list:
    if isinstance(state_space, str):
        result = STATE_SPACE_STR_MAPPING_365.get(state_space)
        if result is None:
            # Fall back to base robocasa mapping
            from rlinf.envs.robocasa.utils import STATE_SPACE_STR_MAPPING
            result = STATE_SPACE_STR_MAPPING.get(state_space)
        return result
    return state_space


# ---------------------------------------------------------------------------
# ACTION
# ---------------------------------------------------------------------------

ROBOCASA365_ALL_ACTION_DIM = 12  # Update if new version changes action space

ROBOCASA365_ACTIONS = {
    "rel_pose_6d": np.arange(0, 6),
    "gripper":     np.arange(6, 7),
    "base":        np.arange(7, 10),
    "torso":       np.arange(10, 11),
    "base_mode":   np.arange(11, 12),
}

# Default values when model does not output all dims
ROBOCASA365_DEFAULT_ACTION = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
)

ACTION_SPACE_STR_MAPPING_365 = {
    "12d": list(ROBOCASA365_ACTIONS.keys()),
    "7d": ["rel_pose_6d", "gripper"],
}


def get_action_space_365(action_space: Union[str, list]) -> list:
    if isinstance(action_space, str):
        return ACTION_SPACE_STR_MAPPING_365.get(action_space, [])
    return action_space


def get_action_ids_365(action_space: list) -> list:
    ids = []
    for name in action_space:
        ids.extend(ROBOCASA365_ACTIONS[name].tolist())
    return ids


# ---------------------------------------------------------------------------
# IMAGE / CAMERA
# ---------------------------------------------------------------------------

# Mapping: internal obs key name -> robosuite camera name for robosuite.make()
# Update camera names if robocasa365 renames cameras.
OBS_KEY_CAMERA_NAME_MAPPING_365 = {
    "observation/image":            "robot0_agentview_left",
    "observation/wrist_image":      "robot0_eye_in_hand",
    "observation/extra_view_image": "robot0_agentview_right",
}

# Mapping: internal key used in obs dict -> raw key from robosuite obs
# These are the keys that appear in the obs returned by env.reset()/step()
OBS_KEY_ROBOCASA365_IMAGE_MAPPING = {
    "main_images":       "robot0_agentview_left_image",
    "wrist_images":      "robot0_eye_in_hand_image",
    "extra_view_images": "robot0_agentview_right_image",
}

# Preset image spaces: name -> list of obs_key strings
IMAGE_SPACE_STR_MAPPING_365 = {
    "2views": [
        "observation/image",
        "observation/wrist_image",
    ],
    "3views": [
        "observation/image",
        "observation/wrist_image",
        "observation/extra_view_image",
    ],
}


def get_image_space_365(image_space: Union[str, list]) -> list:
    if isinstance(image_space, str):
        result = IMAGE_SPACE_STR_MAPPING_365.get(image_space)
        if result is None:
            from rlinf.envs.robocasa.utils import IMAGE_SPACE_STR_MAPPING
            result = IMAGE_SPACE_STR_MAPPING.get(image_space)
        return result
    return image_space
