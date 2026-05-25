# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env Warp RNG state initialiser.

Cross-task reusable. PyTorch has ``torch.Generator``; Warp has no equivalent
so each env owns a ``wp.uint32`` state advanced via ``wp.randf`` / ``wp.rand_init``.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def initialize_rng_state(state: wp.array(dtype=wp.uint32), seed: wp.int32):
    """Per-env Warp RNG state initialiser.

    Launch ``dim=num_envs``. Typical use: separate streams for reset / push
    (``rng_state``) and obs noise (``obs_noise_rng_state``) to avoid entropy
    sharing — call once per stream with different ``seed`` values.
    """
    env_id = wp.tid()
    state[env_id] = wp.rand_init(seed, env_id)
