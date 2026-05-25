# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""History circular-buffer kernels — replacement for
:class:`isaaclab.utils.buffers.CircularBuffer` for Warp envs.

Cross-task reusable. ``num_envs`` and per-frame ``feature_dim`` come from
launch dim / array shape; ``history_length`` is a runtime ``wp.int32`` so
cfg-driven overrides work without recompiling.

Usage pattern:

* ``set_first_push_kernel`` flags resetting envs (broadcast first frame to
  all H slots on next append).
* ``append_or_fill_history_kernel`` reads ``first_push_mask`` and either
  broadcasts (first push) or shifts+writes (steady state).
* ``clear_first_push_kernel`` clears the flag after the append.
* ``flatten_history_kernel`` flattens history into the term-stacked layout
  expected by Manager (``torch.cat([term_history.reshape(N, -1) for term])``).
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def append_or_fill_history_kernel(
    first_push_mask: wp.array(dtype=wp.bool),
    frame: wp.array2d(dtype=wp.float32),
    history_length: wp.int32,
    history: wp.array3d(dtype=wp.float32),
):
    """Replacement for CircularBuffer.append (with first-push broadcast).

    cfg:     g1/29dof/velocity_env_cfg.py:260-262 (PolicyCfg.history_length=5)
    manager: isaaclab/utils/buffers/circular_buffer.py:112-141 (append)
           + isaaclab/utils/buffers/circular_buffer.py:99-110  (reset)
    First-push branch is required for Manager parity: after reset, Manager
    broadcasts the first frame into all H history slots, not [0,0,0,0,obs1].
    """
    env_id, feature_id = wp.tid()
    if first_push_mask[env_id]:
        for history_id in range(history_length):
            history[env_id, history_id, feature_id] = frame[env_id, feature_id]
    else:
        for history_id in range(history_length - 1):
            history[env_id, history_id, feature_id] = history[env_id, history_id + 1, feature_id]
        history[env_id, history_length - 1, feature_id] = frame[env_id, feature_id]


@wp.kernel
def set_first_push_kernel(
    mask: wp.array(dtype=wp.bool),
    first_push_mask: wp.array(dtype=wp.bool),
):
    """Set first_push_mask=True for envs being reset (lazy broadcast trigger).

    manager: isaaclab/utils/buffers/circular_buffer.py:99-110 (reset zeros _num_pushes)
    The next ``append_or_fill_history_kernel`` reads this flag and broadcasts.
    """
    env_id = wp.tid()
    if mask[env_id]:
        first_push_mask[env_id] = True


@wp.kernel
def clear_first_push_kernel(first_push_mask: wp.array(dtype=wp.bool)):
    """Clear first_push_mask after append (so next append takes shift+write path).

    manager: isaaclab/utils/buffers/circular_buffer.py:112-141 (append increments _num_pushes)
    Separate kernel to avoid race on first_push_mask[env_id] across frame_dim threads.
    """
    env_id = wp.tid()
    first_push_mask[env_id] = False


@wp.kernel
def flatten_history_kernel(
    history: wp.array3d(dtype=wp.float32),
    obs_to_slot: wp.array(dtype=wp.int32),
    obs_to_feat: wp.array(dtype=wp.int32),
    observations: wp.array2d(dtype=wp.float32),
):
    """Flatten history into term-stacked layout (NOT frame-stacked).

    manager: isaaclab/managers/observation_manager.py:411-430 (per-term reshape + cat)
    Term-stacked = [bav_h0..h4, pg_h0..h4, ..., la_h0..h4]; frame-stacked = [frame_h0, ..., frame_h4].
    Layout MUST match Manager — PPO's first-layer weights can't learn the difference out.
    Lookup arrays ``obs_to_slot`` / ``obs_to_feat`` precomputed host-side.
    """
    env_id, obs_id = wp.tid()
    slot = obs_to_slot[obs_id]
    feat = obs_to_feat[obs_id]
    observations[env_id, obs_id] = history[env_id, slot, feat]
