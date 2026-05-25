# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-time event kernels — replacement for the ``mode=reset`` ``mdp.*``
event functions called from :class:`isaaclab.managers.EventManager.reset`.

Cross-task reusable. ``num_joints`` is a runtime ``wp.int32`` parameter so
G1 (29) / Go2 (12) / H1 (19) all use the same kernels.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def reset_root_kernel(
    mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    default_root_pose: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    reset_x_range: wp.vec2f,
    reset_y_range: wp.vec2f,
    reset_yaw_range: wp.vec2f,
):
    """Mirror mdp.reset_root_state_uniform for cfg ``reset_base``.

    cfg:     g1/29dof/velocity_env_cfg.py:149-163 (EventCfg.reset_base, mode=reset)
    manager: isaaclab/envs/mdp/events.py:1680
    G1 / Go2 / H1 cfgs only randomise x/y/yaw (z, roll, pitch absent);
    velocity_range all zero so this kernel writes zero velocity.
    """
    env_id = wp.tid()
    if mask[env_id]:
        rand_x = wp.randf(rng_state[env_id], reset_x_range[0], reset_x_range[1])
        rng_state[env_id] += wp.uint32(1)
        rand_y = wp.randf(rng_state[env_id], reset_y_range[0], reset_y_range[1])
        rng_state[env_id] += wp.uint32(1)
        rand_yaw = wp.randf(rng_state[env_id], reset_yaw_range[0], reset_yaw_range[1])
        rng_state[env_id] += wp.uint32(1)

        pos = wp.transform_get_translation(default_root_pose[env_id]) + env_origins[env_id]
        pos = pos + wp.vec3f(rand_x, rand_y, 0.0)
        yaw_quat = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), rand_yaw)
        root_pose_w[env_id] = wp.transform(pos, yaw_quat)
        root_vel_w[env_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def reset_joints_kernel(
    mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    reset_joint_velocity_range: wp.vec2f,
    num_joints: wp.int32,
):
    """Mirror mdp.reset_joints_by_scale for cfg ``reset_robot_joints``.

    cfg:     g1/29dof/velocity_env_cfg.py:165-172 (EventCfg.reset_robot_joints)
    manager: isaaclab/envs/mdp/events.py:1846
    Note: Manager interprets velocity_range as MULTIPLICATIVE scale on default_joint_vel,
    NOT absolute sample.  G1 / Go2 / H1 default_joint_vel=0 → cfg velocity_range
    is effectively no-op; reading as absolute sample would silently break parity
    (~±0.4 rad/s per joint). 1D launch (per-joint serial inside) avoids RNG race
    on rng_state[env_id].
    """
    env_id = wp.tid()
    if mask[env_id]:
        state = rng_state[env_id]
        for joint_id in range(num_joints):
            # Position: Manager scales default by uniform(position_range).
            # cfg currently fixes scale = 1.0 (Manager cfg
            # ``position_range = (1.0, 1.0)``); mirror that exactly.
            joint_pos[env_id, joint_id] = default_joint_pos[env_id, joint_id]
            # Velocity: per-joint independent uniform scale, same as
            # Manager's per-element ``torch.empty(N, J).uniform_(...)``.
            scale = wp.randf(
                state, reset_joint_velocity_range[0], reset_joint_velocity_range[1]
            )
            state += wp.uint32(1)
            joint_vel[env_id, joint_id] = default_joint_vel[env_id, joint_id] * scale
        rng_state[env_id] = state


@wp.kernel
def apply_external_force_torque_kernel(
    mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    body_id: wp.int32,
    force_range: wp.vec2f,
    torque_range: wp.vec2f,
    external_force_b: wp.array3d(dtype=wp.float32),
    external_torque_b: wp.array3d(dtype=wp.float32),
):
    """Mirror mdp.apply_external_force_torque for cfg ``base_external_force_torque``.

    cfg:     g1/29dof/velocity_env_cfg.py:139-147 (EventCfg.base_external_force_torque)
    manager: isaaclab/envs/mdp/events.py:1614
    Activation: caller gates on ``cfg.enable_external_force_torque``. When False
    (G1 / H1 / Go2 default), the kernel is NOT launched and Newton's
    ``has_external_wrench`` stays False. When True, the kernel runs on every
    reset regardless of whether the ranges are (0, 0).
    """
    env_id = wp.tid()
    if mask[env_id]:
        state = rng_state[env_id]
        fx = wp.randf(state, force_range[0], force_range[1])
        state += wp.uint32(1)
        fy = wp.randf(state, force_range[0], force_range[1])
        state += wp.uint32(1)
        fz = wp.randf(state, force_range[0], force_range[1])
        state += wp.uint32(1)
        tx = wp.randf(state, torque_range[0], torque_range[1])
        state += wp.uint32(1)
        ty = wp.randf(state, torque_range[0], torque_range[1])
        state += wp.uint32(1)
        tz = wp.randf(state, torque_range[0], torque_range[1])
        state += wp.uint32(1)
        rng_state[env_id] = state
        external_force_b[env_id, body_id, 0] = fx
        external_force_b[env_id, body_id, 1] = fy
        external_force_b[env_id, body_id, 2] = fz
        external_torque_b[env_id, body_id, 0] = tx
        external_torque_b[env_id, body_id, 1] = ty
        external_torque_b[env_id, body_id, 2] = tz
