"""
Subprocess vectorized environment for robocasa v1.0.0.

Difference from robocasa/venv.py:
  - Workers patch sys.path to load robocasa v1.0.0 from ``/home/njc/robocasa`` by
    default (override: ``export ROBOCASA365_PATH=...``) BEFORE any other
    robocasa import, so the new package takes priority over an editable install
    under RLinf's ``.venv``.
  - If a separate venv is used (ROBOCASA365_PYTHON env var), workers are
    spawned using that Python interpreter via subprocess.Popen instead of
    multiprocessing.Process.

Usage (choose one):

  A) sys.path patching only (simpler, works if robosuite >= 1.5.2 is installed
     and numpy/mujoco version assertions are patched):
       Robocasa365SubprocEnv(env_fns)   ← default

  B) Separate venv (no version conflicts at all):
       export ROBOCASA365_PYTHON=/root/venvs/robocasa365_venv/bin/python
       Robocasa365SubprocEnv(env_fns)   ← auto-detected via env var
"""

import os
import sys
from multiprocessing import Pipe, Process, connection
from typing import Any, Callable, Optional, Union

import numpy as np

from rlinf.envs.robocasa365.utils import get_robocasa365_source_path
from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)

# Path to robocasa v1.0.0 source (inserted at the start of sys.path in workers)
ROBOCASA365_SRC = get_robocasa365_source_path()

# Optional: path to a separate venv Python that has robocasa365 deps installed
# Set via: export ROBOCASA365_PYTHON=/root/venvs/robocasa365_venv/bin/python
ROBOCASA365_PYTHON = os.environ.get("ROBOCASA365_PYTHON", None)


def _worker_365(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    """
    Worker function for robocasa v1.0.0 subprocess environment.

    Inserts ``ROBOCASA365_SRC`` (default ``/home/njc/robocasa``) at the front of
    sys.path so the v1.0 package is imported before any older version.

    Identical protocol to robocasa/venv.py _worker():
      step  → calls env.step(), env._check_success(), env.get_ep_meta()
      reset → calls env.reset(), env.get_ep_meta()
    """
    # ── Patch sys.path to load robocasa v1.0.0 ──────────────────────────
    if ROBOCASA365_SRC not in sys.path:
        sys.path.insert(0, ROBOCASA365_SRC)

    def _encode_obs(obs, buffer):
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])

    def _check_success(env, env_return):
        success = env._check_success()
        env_return = list(env_return)
        info = env_return[-1]
        info["success"] = success
        env_return[-1] = info
        return tuple(env_return)

    def get_ep_meta(env, env_return):
        ep_meta = env.get_ep_meta()
        env_return = list(env_return)
        info = env_return[-1]
        info["ep_meta"] = ep_meta
        env_return[-1] = info
        return tuple(env_return)

    parent.close()
    env = env_fn_wrapper.data()

    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:
                p.close()
                break

            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                if hasattr(env, "_check_success"):
                    env_return = _check_success(env, env_return)
                if hasattr(env, "get_ep_meta"):
                    env_return = get_ep_meta(env, env_return)
                p.send(env_return)

            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs, info = retval, {}
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if hasattr(env, "get_ep_meta"):
                    info = get_ep_meta(env, (info,))[-1]
                p.send((obs, info))

            elif cmd == "close":
                p.send(env.close())
                p.close()
                break

            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)

            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)

            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)

            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])

            else:
                p.close()
                raise NotImplementedError(f"Unknown command: {cmd}")

    except KeyboardInterrupt:
        p.close()


class Robocasa365SubprocEnvWorker(SubprocEnvWorker):
    """Subprocess worker for robocasa v1.0.0; uses _worker_365."""

    def __init__(self, env_fn: Callable, share_memory: bool = False):
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None

        if self.share_memory:
            dummy = env_fn()
            self.buffer = _setup_buf(dummy.observation_space)
            dummy.close()
            del dummy

        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker_365, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)


class Robocasa365SubprocEnv(SubprocVectorEnv):
    """
    Subprocess vectorized environment for robocasa v1.0.0.

    Each subprocess worker patches sys.path to import robocasa v1.0.0 from
    ``/home/njc/robocasa`` by default (or ``ROBOCASA365_PATH``) before any other version.
    """

    def __init__(self, env_fns: list[Callable], **kwargs: Any) -> None:
        def worker_fn(fn: Callable) -> Robocasa365SubprocEnvWorker:
            return Robocasa365SubprocEnvWorker(fn, share_memory=False)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)
