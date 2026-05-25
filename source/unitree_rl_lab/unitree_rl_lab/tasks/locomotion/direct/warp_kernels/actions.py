# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position action kernels — replacement for
:class:`isaaclab.envs.mdp.actions.JointPositionAction` and
:class:`isaaclab.managers.ActionManager` reset / process_action paths.

Cross-task reusable for any env using ``JointPositionAction`` (G1 / Go2 / H1
share this action mode). ``num_joints`` comes from launch dim, no special
handling needed.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def pre_physics_step_kernel(
    input_actions: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    action_scale: wp.float32,
    raw_actions: wp.array2d(dtype=wp.float32),
    prev_actions: wp.array2d(dtype=wp.float32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
):
    """Fused replacement: ActionManager.process_action + JointPositionAction.process_actions.

    cfg:     g1/29dof/velocity_env_cfg.py:235-240 (ActionsCfg)
    manager: isaaclab/managers/action_manager.py:374-389       (prev_action / action update)
           + isaaclab/envs/mdp/actions/joint_actions.py:170-179 (raw * scale + default_joint_pos)
    G1 cfg has no action_clip, so no per-joint clip applied (matches Manager no-op).
    Launch ``dim=(num_envs, num_joints)``; ``num_joints`` inferred from array shape.
    """
    env_id, joint_id = wp.tid()
    prev_actions[env_id, joint_id] = raw_actions[env_id, joint_id]
    raw = input_actions[env_id, joint_id]
    raw_actions[env_id, joint_id] = raw
    joint_pos_target[env_id, joint_id] = default_joint_pos[env_id, joint_id] + action_scale * raw


@wp.kernel
def clear_reset_buffers_kernel(
    mask: wp.array(dtype=wp.bool),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    raw_actions: wp.array2d(dtype=wp.float32),
    prev_actions: wp.array2d(dtype=wp.float32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
):
    """Replacement: ActionManager.reset (zero buffers) + post-reset target = default pose.

    manager: isaaclab/managers/action_manager.py:367-368 (zero action / prev_action)
    Setting joint_pos_target = default_joint_pos ensures first sub-step after reset
    commands home pose, not 0 rad. Launch ``dim=(num_envs, num_joints)``.
    """
    env_id, joint_id = wp.tid()
    if mask[env_id]:
        raw_actions[env_id, joint_id] = 0.0
        prev_actions[env_id, joint_id] = 0.0
        joint_pos_target[env_id, joint_id] = default_joint_pos[env_id, joint_id]
