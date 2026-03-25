# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .utils.omega_resolver import omegaconf_register

omegaconf_register()

# ---- Numpy safety patch (NaN probabilities in choice) ----
# Some environments/tools (e.g. reset/randomization logic) may call:
#   np.random.Generator.choice(..., p=probabilities)
# and occasionally produce NaN in `p`, which would hard-crash the worker.
# For embodied training we prefer a robust fallback over a hard failure.
#
# Policy:
# - Replace NaN in p with 0
# - Re-normalize p to sum to 1
# - If sum is invalid/non-positive, fall back to uniform distribution
#
# This avoids: ValueError: probabilities contain NaN
try:
    import numpy as _np

    _orig_choice = _np.random.Generator.choice

    def _safe_choice(self, *args, **kwargs):  # noqa: ANN001
        p = None
        if "p" in kwargs:
            p = kwargs.get("p")
        elif len(args) > 3:
            # Generator.choice(a, size=None, replace=True, p=None, ...)
            p = args[3]

        if p is not None:
            p_arr = _np.asarray(p, dtype=float)
            if _np.isnan(p_arr).any() or not _np.isfinite(p_arr).all():
                p_arr = _np.nan_to_num(p_arr, nan=0.0, posinf=0.0, neginf=0.0)

            total = float(p_arr.sum()) if p_arr.size > 0 else 0.0
            if not _np.isfinite(total) or total <= 0.0:
                # Fall back to uniform over the provided p shape.
                if p_arr.size > 0:
                    p_arr = _np.full_like(p_arr, 1.0 / p_arr.size, dtype=float)
                else:
                    # Degenerate but keep behavior: leave p as None.
                    p_arr = None
            else:
                p_arr = p_arr / total

            if p_arr is not None:
                if "p" in kwargs:
                    kwargs["p"] = p_arr
                elif len(args) > 3:
                    args = list(args)
                    args[3] = p_arr
                    args = tuple(args)

        return _orig_choice(self, *args, **kwargs)

    _np.random.Generator.choice = _safe_choice  # type: ignore[assignment]

except Exception:
    # Safety patch must never break imports.
    pass
