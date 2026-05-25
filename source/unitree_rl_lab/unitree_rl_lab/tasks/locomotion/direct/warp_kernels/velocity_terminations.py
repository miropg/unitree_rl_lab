# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-task termination kernel.

Reusable across velocity-tracking tasks (G1 / Go2 / H1) that share the
3-condition termination pattern: time_out OR base_height OR bad_orientation.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def get_dones_kernel(
    episode_length_buf: wp.array(dtype=wp.int32),
    root_pose_w: wp.array(dtype=wp.transformf),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    max_episode_length: wp.int32,
    termination_base_height: wp.float32,
    termination_bad_orientation: wp.float32,
    reset_terminated: wp.array(dtype=wp.bool),
    reset_time_outs: wp.array(dtype=wp.bool),
    reset_buf: wp.array(dtype=wp.bool),
):
    """Fused replacement: TerminationManager.compute (3 G1 termination terms).

    cfg:     g1/29dof/velocity_env_cfg.py:395-401 (TerminationsCfg)
    manager: isaaclab/envs/mdp/terminations.py:32 (mdp.time_out)
           + isaaclab/envs/mdp/terminations.py:52 (mdp.bad_orientation)
           + isaaclab/envs/mdp/terminations.py:64 (mdp.root_height_below_minimum)
    Trade-off: collapses base_height + bad_orientation into single ``reset_terminated``
    bit (per-term ``term_dones`` buffer not replicated); both surface as
    ``Episode_Termination/bad_orientation`` in TB log.
    """
    env_id = wp.tid()
    root_height = wp.transform_get_translation(root_pose_w[env_id])[2]
    gravity_z = wp.clamp(-projected_gravity_b[env_id][2], -1.0, 1.0)
    base_height = root_height < termination_base_height
    bad_orientation = wp.acos(gravity_z) > termination_bad_orientation
    time_out = episode_length_buf[env_id] >= max_episode_length
    reset_terminated[env_id] = base_height or bad_orientation
    reset_time_outs[env_id] = time_out
    reset_buf[env_id] = reset_terminated[env_id] or time_out
