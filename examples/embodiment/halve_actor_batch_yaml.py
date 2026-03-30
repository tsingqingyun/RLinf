#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Halve actor.micro_batch_size and actor.global_batch_size in an RLinf Hydra yaml.

Keeps FSDP constraint: global_batch_size % (micro_batch_size * world_size) == 0.
Preserves comments and unrelated lines (line-oriented edit).

Usage:
  WORLD_SIZE=8 python3 halve_actor_batch_yaml.py /path/to/config.yaml
Exit 0 on success; 1 if nothing could be reduced or file invalid.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: halve_actor_batch_yaml.py <config.yaml>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1]).resolve()
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        sys.exit(1)

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("NUM_GPUS", "8")))

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    micro_idx = global_idx = None
    micro = glob = None
    for i, line in enumerate(lines):
        if re.match(r"\s+micro_batch_size:\s*\d+\s*$", line):
            micro_idx = i
            micro = int(re.search(r"\d+", line).group())
        if re.match(r"\s+global_batch_size:\s*\d+\s*$", line):
            global_idx = i
            glob = int(re.search(r"\d+", line).group())

    if micro_idx is None or global_idx is None or micro is None or glob is None:
        print(
            "Could not find actor micro_batch_size / global_batch_size lines.",
            file=sys.stderr,
        )
        sys.exit(1)

    m2 = max(1, micro // 2)
    g_raw = glob // 2
    step = m2 * world_size
    if step < 1:
        step = 1
    g2 = (g_raw // step) * step
    if g2 < step:
        g2 = step

    if m2 == micro and g2 == glob:
        print(
            "Already at minimum batch (cannot halve further under FSDP constraints).",
            file=sys.stderr,
        )
        sys.exit(1)

    def repl_num(line: str, new_val: int) -> str:
        return re.sub(r"\d+", str(new_val), line, count=1)

    lines[micro_idx] = repl_num(lines[micro_idx], m2)
    lines[global_idx] = repl_num(lines[global_idx], g2)
    path.write_text("".join(lines), encoding="utf-8")
    print(
        f"Updated {path}: micro_batch_size {micro} -> {m2}, "
        f"global_batch_size {glob} -> {g2} (WORLD_SIZE={world_size})"
    )


if __name__ == "__main__":
    main()
