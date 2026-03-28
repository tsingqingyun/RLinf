"""
RoboCasa-365 environment adapter for RLinf.

Inherits from RobocasaEnv and overrides only the parts that differ
in the robocasa365 version:
  - Package import (robocasa365 vs robocasa)
  - Observation extraction (if obs keys / state dims differ)
  - Action space (if dims differ)
  - Reward (if multi-stage / progress reward is available)

Before modifying, run the probe script to confirm actual obs keys:
  python /tmp/probe_robocasa365.py <TaskName>
"""

import numpy as np

from rlinf.envs.robocasa.robocasa_env import RobocasaEnv
from rlinf.envs.robocasa365.utils import (
    OBS_KEY_CAMERA_NAME_MAPPING_365,
    OBS_KEY_ROBOCASA365_IMAGE_MAPPING,
    ROBOCASA365_ALL_ACTION_DIM,
    ROBOCASA365_STATES,
    STATE_DIM_365,
    get_image_space_365,
)
from rlinf.envs.utils import to_tensor, list_of_dict_to_dict_of_list


class Robocasa365Env(RobocasaEnv):
    """
    RoboCasa-365 environment.

    Key differences from RobocasaEnv (fill in after running probe script):
      - Package name: robocasa365 (or same robocasa with updated version)
      - State dim: STATE_DIM_365 (update utils.py after probing)
      - Camera names: may include a third view
      - Reward: supports partial/progress reward if available
    """

    # -----------------------------------------------------------------------
    # 1. Package import — override if new version uses a different package name
    # -----------------------------------------------------------------------
    def _init_env(self):
        """Initialize robocasa365 environments using subprocess isolation."""
        # ↓ Change import name if the new package is called robocasa365
        try:
            import robocasa365  # noqa: F401
        except ImportError:
            import robocasa  # noqa: F401  fall back to existing install

        from rlinf.envs.robocasa.venv import RobocasaSubprocEnv

        self.task_ids = []
        for env_id in range(self.num_envs):
            task_idx = env_id % self.num_tasks
            self.task_ids.append(task_idx)
        self.task_ids = np.array(self.task_ids)

        env_fns = self.get_env_fns()
        self.env = RobocasaSubprocEnv(env_fns)

    # -----------------------------------------------------------------------
    # 2. Camera names — update if new version exposes different camera keys
    # -----------------------------------------------------------------------
    @property
    def camera_names(self):
        """Camera names for robocasa365 (may differ from v0.2)."""
        image_space = get_image_space_365(self.cfg.image_space)
        return [
            OBS_KEY_CAMERA_NAME_MAPPING_365[obs_key]
            for obs_key in image_space
            if obs_key in OBS_KEY_CAMERA_NAME_MAPPING_365
        ]

    # -----------------------------------------------------------------------
    # 3. get_env_fns — add any extra robosuite.make() kwargs for new version
    # -----------------------------------------------------------------------
    def get_env_fns(self):
        """Create env factory functions; adds new-version-specific kwargs."""
        env_fns = []

        for env_id in range(self.num_envs):
            task_idx = self.task_ids[env_id]
            task_name = self.task_names[task_idx]
            env_seed = self.env_seeds[env_id]

            camera_widths = self.cfg.init_params.camera_widths
            camera_heights = self.cfg.init_params.camera_heights
            robot_name = self.cfg.robot_name

            # Optional: layout/style control if robocasa365 supports it
            layout_id = self.cfg.get("layout_id", None)
            style_id = self.cfg.get("style_id", None)

            def env_fn(
                task=task_name,
                seed=env_seed,
                width=camera_widths,
                height=camera_heights,
                robot=robot_name,
                layout_id=layout_id,
                style_id=style_id,
            ):
                import robosuite
                from robosuite.controllers import load_composite_controller_config

                controller_config = load_composite_controller_config(
                    controller=None, robot=robot
                )

                make_kwargs = dict(
                    env_name=task,
                    robots=robot,
                    controller_configs=controller_config,
                    camera_names=self.camera_names,
                    camera_widths=width,
                    camera_heights=height,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    ignore_done=True,
                    use_object_obs=True,
                    use_camera_obs=True,
                    camera_depths=False,
                    seed=seed,
                    translucent_robot=False,
                    render_camera="robot0_agentview_center",
                )

                # ↓ Add robocasa365-specific kwargs only if provided
                if layout_id is not None:
                    make_kwargs["layout_id"] = layout_id
                if style_id is not None:
                    make_kwargs["style_id"] = style_id

                env = robosuite.make(**make_kwargs)
                return env

            env_fns.append(env_fn)

        return env_fns

    # -----------------------------------------------------------------------
    # 4. _extract_image_and_state — the most critical adaptation point
    #    Update field slices after running probe_robocasa365.py
    # -----------------------------------------------------------------------
    def _extract_image_and_state(self, obs):
        """
        Extract images and states matching robocasa365's obs format.

        Update STATE_DIM_365 and field slices in utils.py after probing.
        Current assumption: same 25D layout as v0.2 — change if needed.
        """
        images_by_key = {k: [] for k in OBS_KEY_ROBOCASA365_IMAGE_MAPPING}
        states = []

        for env_id in range(len(obs)):
            env_obs = obs[env_id]

            # --- Images (flip vertically: OpenGL coords are upside-down) ---
            for img_key in OBS_KEY_ROBOCASA365_IMAGE_MAPPING:
                raw_name = OBS_KEY_ROBOCASA365_IMAGE_MAPPING[img_key]
                img = env_obs.get(raw_name)
                if img is not None:
                    img = img[::-1].copy()
                images_by_key[img_key].append(img)

            # --- State vector ---
            # ↓ Modify slices here if robocasa365 has different state layout.
            # Run probe_robocasa365.py first to confirm.
            state = np.zeros(STATE_DIM_365, dtype=np.float32)
            ptr = 0

            def fill(field, dim):
                nonlocal ptr
                val = env_obs.get(field)
                if val is not None:
                    state[ptr : ptr + dim] = val[:dim]
                ptr += dim

            fill("robot0_eef_pos", 3)         # [0:3]
            fill("robot0_eef_quat", 4)        # [3:7]
            fill("robot0_gripper_qpos", 2)    # [7:9]
            fill("robot0_gripper_qvel", 2)    # [9:11]
            fill("robot0_base_to_eef_pos", 3) # [11:14]
            fill("robot0_base_to_eef_quat", 4)# [14:18]
            fill("robot0_base_pos", 3)        # [18:21]
            fill("robot0_base_quat", 4)       # [21:25]
            # ↓ If robocasa365 adds joint positions, uncomment:
            # fill("robot0_joint_pos", 7)     # [25:32]

            states.append(state)

        result = {"state": np.array(states)}
        for img_key in OBS_KEY_ROBOCASA365_IMAGE_MAPPING:
            result[img_key] = np.array(images_by_key[img_key])

        return result

    # -----------------------------------------------------------------------
    # 5. _wrap_obs — rebuild obs dict with new image key names
    # -----------------------------------------------------------------------
    def _wrap_obs(self, obs_list, info_list):
        import torch
        from rlinf.envs.utils import list_of_dict_to_dict_of_list

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
            if isinstance(imgs, list) and imgs[0] is None:
                obs[obs_key_name] = None
            else:
                obs[obs_key_name] = torch.stack(
                    [v.clone() for v in imgs]
                ) if isinstance(imgs, list) else imgs
        return obs

    # -----------------------------------------------------------------------
    # 6. _calc_step_reward — extend for progress/partial reward if available
    # -----------------------------------------------------------------------
    def _calc_step_reward(self, terminations, infos=None):
        """
        Reward computation for robocasa365.

        If the new version provides partial/progress rewards through
        info["partial_reward"], use them; otherwise fall back to binary.

        To use partial reward:
          1. Add partial_reward extraction in venv.py _worker()
          2. Set use_partial_reward: True in yaml
        """
        use_partial = getattr(self.cfg, "use_partial_reward", False)

        if use_partial and infos is not None:
            # infos here is a list of per-env info dicts
            partial = np.array(
                [info.get("partial_reward", float(t)) for info, t in
                 zip(infos, terminations)],
                dtype=np.float32,
            )
            reward = self.cfg.reward_coef * partial
        else:
            reward = self.cfg.reward_coef * terminations.astype(np.float32)

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        return reward

    # Override step to pass infos into reward calc
    def step(self, actions=None, auto_reset=True):
        if actions is None:
            assert self._is_start
        if self.is_start:
            obs, infos = self.reset()
            import torch
            zeros = np.zeros(self.num_envs, dtype=np.float32)
            return (
                obs,
                to_tensor(zeros),
                to_tensor(zeros.astype(bool)),
                to_tensor(zeros.astype(bool)),
                infos,
            )

        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, rewards, dones, info_lists = self.env.step(actions)
        infos = list_of_dict_to_dict_of_list(info_lists)

        terminations = np.array(
            [info.get("success", False) for info in info_lists]
        ).astype(bool)
        truncations = self._elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs, info_lists)

        # Pass raw info_lists for partial reward support
        step_reward = self._calc_step_reward(terminations, infos=info_lists)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )
