"""
RoboCasa v1.0.0 ("365") environment adapter for RLinf.

Key API changes vs robocasa v0.2.0:
  - Use create_env() from robocasa.utils.env_utils instead of robosuite.make() directly
  - New split / layout_ids / style_ids / obj_instance_split params for domain randomization
  - Default camera resolution: 256x256, 3 views
  - Raw obs keys (robot0_eef_pos, etc.) are unchanged → venv.py / _check_success unchanged
  - Requires: mujoco==3.3.1, numpy==2.2.5, robosuite>=1.5.2
"""

import inspect
import sys

import numpy as np

from rlinf.envs.robocasa.robocasa_env import RobocasaEnv
from rlinf.envs.robocasa365.venv365 import Robocasa365SubprocEnv
from rlinf.envs.robocasa365.utils import (
    IMAGE_SPACE_STR_MAPPING_365,
    OBS_KEY_CAMERA_NAME_MAPPING_365,
    OBS_KEY_ROBOCASA365_IMAGE_MAPPING,
    STATE_DIM_365,
    get_image_space_365,
    get_robocasa365_source_path,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor


class Robocasa365Env(RobocasaEnv):
    """
    RoboCasa v1.0.0 environment for RLinf.

    New config keys (set in yaml under env.train / env.eval):
      split: null | "all" | "pretrain" | "target"
          Controls which layout/style/object splits are used.
          null  → free random (default, same as old behavior)
          "all" → all layouts and styles
          "pretrain" / "target" → for train/eval split generalisation

      layout_ids: null | int | list[int]
          Fix specific layout IDs (overridden by split if split != null)

      style_ids: null | int | list[int]
          Fix specific style IDs (overridden by split if split != null)

      obj_instance_split: null | "pretrain" | "target"
          Object instance-level generalisation split

      generative_textures: null | float
          Probability of using generative (random) textures (0.0-1.0)

      randomize_cameras: false
          Whether to randomly jitter camera poses each episode

      camera_heights: 256   (new default in v1.0.0; was 224)
      camera_widths:  256
      image_space: "3views" (new default; was 2views — adds agentview_right)
    """

    # -----------------------------------------------------------------------
    # 1. Import RoboCasa v1.0 from /home/njc/robocasa by default (override: export ROBOCASA365_PATH=...)
    # -----------------------------------------------------------------------

    def _ensure_robocasa365_on_path(self):
        """Add robocasa v1.0.0 to sys.path so its package takes priority."""
        rc = get_robocasa365_source_path()
        if rc not in sys.path:
            sys.path.insert(0, rc)

    # -----------------------------------------------------------------------
    # 2. Camera names — new default adds agentview_right
    # -----------------------------------------------------------------------
    @property
    def camera_names(self):
        image_space = get_image_space_365(self.cfg.image_space)
        return [
            OBS_KEY_CAMERA_NAME_MAPPING_365[obs_key]
            for obs_key in image_space
            if obs_key in OBS_KEY_CAMERA_NAME_MAPPING_365
        ]

    # -----------------------------------------------------------------------
    # 3. _init_env — ensure robocasa365 is importable, then build subprocess envs
    # -----------------------------------------------------------------------
    def _init_env(self):
        self._ensure_robocasa365_on_path()
        import robocasa  # noqa: F401 — registers all envs in robosuite

        self.task_ids = []
        for env_id in range(self.num_envs):
            self.task_ids.append(env_id % self.num_tasks)
        self.task_ids = np.array(self.task_ids)

        env_fns = self.get_env_fns()
        self.env = Robocasa365SubprocEnv(env_fns)

    # -----------------------------------------------------------------------
    # 4. get_env_fns — use create_env() with new v1.0.0 parameters
    # -----------------------------------------------------------------------
    def get_env_fns(self):
        env_fns = []

        # Read new v1.0.0 params from config (all optional, default to None/False)
        split            = self.cfg.get("split", None)
        layout_ids       = self.cfg.get("layout_ids", None)
        style_ids        = self.cfg.get("style_ids", None)
        obj_instance_split  = self.cfg.get("obj_instance_split", None)
        generative_textures = self.cfg.get("generative_textures", None)
        randomize_cameras   = self.cfg.get("randomize_cameras", False)

        for env_id in range(self.num_envs):
            task_name  = self.task_names[self.task_ids[env_id]]
            env_seed   = int(self.env_seeds[env_id])
            cam_h      = self.cfg.init_params.camera_heights
            cam_w      = self.cfg.init_params.camera_widths
            robot_name = self.cfg.robot_name
            cameras    = self.camera_names

            obj_registries = tuple(
                getattr(self.cfg, "obj_registries", None) or ("lightwheel",)
            )

            def env_fn(
                task=task_name,
                seed=env_seed,
                width=cam_w,
                height=cam_h,
                robot=robot_name,
                _cameras=cameras,
                _split=split,
                _layout_ids=layout_ids,
                _style_ids=style_ids,
                _obj_instance_split=obj_instance_split,
                _generative_textures=generative_textures,
                _randomize_cameras=randomize_cameras,
                _obj_registries=obj_registries,
            ):
                # Ensure robocasa365 is on path inside the subprocess too
                import sys as _sys
                _rc365_path = get_robocasa365_source_path()
                if _rc365_path not in _sys.path:
                    _sys.path.insert(0, _rc365_path)

                import robocasa  # noqa: F401
                from robocasa.utils.env_utils import create_env

                create_kw = dict(
                    env_name=task,
                    robots=robot,
                    camera_names=_cameras,
                    camera_widths=width,
                    camera_heights=height,
                    seed=seed,
                    render_onscreen=False,
                    # v1.0.0 new params ↓
                    split=_split,
                    layout_ids=_layout_ids,
                    style_ids=_style_ids,
                    obj_instance_split=_obj_instance_split,
                    generative_textures=_generative_textures,
                    randomize_cameras=_randomize_cameras,
                    # objaverse assets shipped with repo lack reg_bbox geom;
                    # restrict to lightwheel until assets are re-downloaded
                    obj_registries=_obj_registries,
                )
                # Older robocasa create_env() does not accept translucent_robot / split / obj_registries (v1.0+ only)
                sig = inspect.signature(create_env)
                params = sig.parameters
                if "translucent_robot" in params:
                    create_kw["translucent_robot"] = False
                if any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in params.values()
                ):
                    env = create_env(**create_kw)
                else:
                    allowed = {k: v for k, v in create_kw.items() if k in params}
                    env = create_env(**allowed)
                return env

            env_fns.append(env_fn)

        return env_fns

    # -----------------------------------------------------------------------
    # 5. _extract_image_and_state
    #    Raw obs keys from robosuite are unchanged in v1.0.0 — state format same.
    #    Only image mapping differs if image_space = "3views".
    # -----------------------------------------------------------------------
    def _extract_image_and_state(self, obs):
        images_by_key = {k: [] for k in OBS_KEY_ROBOCASA365_IMAGE_MAPPING}
        states = []

        for env_id in range(len(obs)):
            env_obs = obs[env_id]

            # --- Images (OpenGL coords: flip vertically) ---
            for img_key, raw_name in OBS_KEY_ROBOCASA365_IMAGE_MAPPING.items():
                img = env_obs.get(raw_name)
                if img is not None:
                    img = img[::-1].copy()
                images_by_key[img_key].append(img)

            # --- State vector (same 25D layout as v0.2.0) ---
            state = np.zeros(STATE_DIM_365, dtype=np.float32)
            state[0:3]   = env_obs["robot0_eef_pos"]
            state[3:7]   = env_obs["robot0_eef_quat"]
            state[7:9]   = env_obs["robot0_gripper_qpos"]
            state[9:11]  = env_obs["robot0_gripper_qvel"]
            state[11:14] = env_obs["robot0_base_to_eef_pos"]
            state[14:18] = env_obs["robot0_base_to_eef_quat"]
            state[18:21] = env_obs["robot0_base_pos"]
            state[21:25] = env_obs["robot0_base_quat"]
            states.append(state)

        result = {"state": np.array(states)}
        for img_key in OBS_KEY_ROBOCASA365_IMAGE_MAPPING:
            result[img_key] = np.array(images_by_key[img_key])
        return result

    # -----------------------------------------------------------------------
    # 6. _wrap_obs — rebuild obs dict supporting 2views or 3views
    # -----------------------------------------------------------------------
    def _wrap_obs(self, obs_list, info_list):
        import torch

        extracted_obs = self._extract_image_and_state(obs_list)
        task_description_list = self._extract_task_description(info_list)

        images_and_states_list = []
        for idx in range(self.num_envs):
            entry = {"state": extracted_obs["state"][idx]}
            for img_key in OBS_KEY_ROBOCASA365_IMAGE_MAPPING:
                entry[img_key] = extracted_obs[img_key][idx]
            images_and_states_list.append(entry)

        images_and_states_tensor = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        obs = {
            "states": images_and_states_tensor["state"],
            "task_descriptions": task_description_list,
        }

        for obs_key_name in OBS_KEY_ROBOCASA365_IMAGE_MAPPING:
            imgs = images_and_states_tensor[obs_key_name]
            if isinstance(imgs, list) and any(v is None for v in imgs):
                obs[obs_key_name] = None
            elif isinstance(imgs, list):
                obs[obs_key_name] = torch.stack([v.clone() for v in imgs])
            else:
                obs[obs_key_name] = imgs
        return obs
