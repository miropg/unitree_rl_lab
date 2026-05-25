# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-term reward ``wp.func`` library — one per RewardsCfg term.

Reusable across velocity-tracking tasks (G1 / Go2 / H1) that share the
core 19-term reward set. Joint-aggregating wp.funcs take ``num_joints``
as a runtime ``wp.int32`` parameter so each robot dispatches the same
function with its own joint count.

Naming: ``_reward_<cfg_name>`` matches RewardsCfg field name (NOT mdp.*
function name — e.g. cfg ``base_linear_velocity`` is ``mdp.lin_vel_z_l2``).
Each returns the RAW per-step quantity (BEFORE weight × step_dt); the
robot-specific fusion kernel applies weight × step_dt at the call site,
mirroring :meth:`isaaclab.managers.RewardManager.compute` (line 129-159).

19 G1 terms come from 3 repos:
    isaaclab        → general kinematic / dynamic terms (envs/mdp/rewards.py)
    isaaclab_tasks  → upstream locomotion-specific (manager_based/locomotion/.../rewards.py)
    unitree_rl_lab  → project-internal terms (tasks/locomotion/mdp/rewards.py)
"""

from __future__ import annotations

import warp as wp


# ============================================================================
# Helper wp.funcs (algorithmic primitives shared across reward terms)
# ============================================================================


@wp.func
def _sum_joint_l1(
    joint_pos: wp.array(dtype=wp.float32),
    default_joint_pos: wp.array(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    count: wp.int32,
) -> wp.float32:
    """Sum |joint_pos - default_joint_pos| over selected joints. Helper for joint_deviation_l1."""
    value = wp.float32(0.0)
    for i in range(count):
        joint_id = joint_ids[i]
        value += wp.abs(joint_pos[joint_id] - default_joint_pos[joint_id])
    return value


@wp.func
def _yaw_quat_inverse_apply_xy(quat: wp.quatf, vec: wp.vec3f) -> wp.vec2f:
    """Project vec into gravity-aligned yaw frame, return xy. Helper for track_lin_vel_xy.

    Equivalent: ``quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w)[:, :2]``
    used by isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py:92.
    z component of vec is ignored (reward only looks at xy).
    """
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]
    yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cy = wp.cos(yaw)
    sy = wp.sin(yaw)
    out_x = cy * vec[0] + sy * vec[1]
    out_y = -sy * vec[0] + cy * vec[1]
    return wp.vec2f(out_x, out_y)


@wp.func
def _max_history_force_norm(
    net_forces_w_history: wp.array3d(dtype=wp.vec3f),
    env_id: wp.int32,
    body_id: wp.int32,
    history_length: wp.int32,
) -> wp.float32:
    """Max L2 force norm across history dim. Helper for feet_slide / undesired_contacts.

    Equivalent: ``net_forces_w_history[:, :, body_ids, :].norm(dim=-1).max(dim=1)[0]``
    history_length = contact_sensor.cfg.history_length (3 for G1 default).
    """
    max_sq = wp.float32(0.0)
    for h in range(history_length):
        f = net_forces_w_history[env_id, h, body_id]
        norm_sq = f[0] * f[0] + f[1] * f[1] + f[2] * f[2]
        if norm_sq > max_sq:
            max_sq = norm_sq
    return wp.sqrt(max_sq)


# ============================================================================
# Per-term reward wp.func (19 G1 terms; cross-robot reusable)
# ============================================================================


@wp.func
def _reward_track_lin_vel_xy(
    env_id: wp.int32,
    root_quat_w: wp.array(dtype=wp.quatf),
    root_lin_vel_w: wp.array(dtype=wp.vec3f),
    commands: wp.array2d(dtype=wp.float32),
    std_sq: wp.float32,
) -> wp.float32:
    """Mirror mdp.track_lin_vel_xy_yaw_frame_exp for cfg ``track_lin_vel_xy``.

    cfg:     g1/29dof/velocity_env_cfg.py:297-301 (weight=1.0, std=sqrt(0.25))
    manager: isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py:92
    Note: std_sq is precomputed std**2 host-side.
    """
    vel_yaw = _yaw_quat_inverse_apply_xy(root_quat_w[env_id], root_lin_vel_w[env_id])
    err_x = commands[env_id, 0] - vel_yaw[0]
    err_y = commands[env_id, 1] - vel_yaw[1]
    lin_vel_error = err_x * err_x + err_y * err_y
    return wp.exp(-lin_vel_error / std_sq)


@wp.func
def _reward_track_ang_vel_z(
    env_id: wp.int32,
    root_ang_vel_b: wp.array(dtype=wp.vec3f),
    commands: wp.array2d(dtype=wp.float32),
    std_sq: wp.float32,
) -> wp.float32:
    """Mirror mdp.track_ang_vel_z_exp for cfg ``track_ang_vel_z``.

    cfg:     g1/29dof/velocity_env_cfg.py:302-304 (weight=0.5, std=sqrt(0.25))
    manager: isaaclab/envs/mdp/rewards.py:332
    """
    err = commands[env_id, 2] - root_ang_vel_b[env_id][2]
    return wp.exp(-(err * err) / std_sq)


@wp.func
def _reward_alive(
    env_id: wp.int32,
    reset_terminated: wp.array(dtype=wp.bool),
) -> wp.float32:
    """Mirror mdp.is_alive for cfg ``alive``.  Returns 1.0 if alive else 0.0.

    cfg:     g1/29dof/velocity_env_cfg.py:306 (weight=0.15)
    manager: isaaclab/envs/mdp/rewards.py:33
    """
    if reset_terminated[env_id]:
        return wp.float32(0.0)
    return wp.float32(1.0)


@wp.func
def _reward_lin_vel_z_l2(
    env_id: wp.int32,
    root_lin_vel_b: wp.array(dtype=wp.vec3f),
) -> wp.float32:
    """Mirror mdp.lin_vel_z_l2 for cfg ``base_linear_velocity``.

    cfg:     g1/29dof/velocity_env_cfg.py:309 (weight=-2.0)
    manager: isaaclab/envs/mdp/rewards.py:78
    """
    z = root_lin_vel_b[env_id][2]
    return z * z


@wp.func
def _reward_ang_vel_xy_l2(
    env_id: wp.int32,
    root_ang_vel_b: wp.array(dtype=wp.vec3f),
) -> wp.float32:
    """Mirror mdp.ang_vel_xy_l2 for cfg ``base_angular_velocity``.

    cfg:     g1/29dof/velocity_env_cfg.py:310 (weight=-0.05)
    manager: isaaclab/envs/mdp/rewards.py:85
    """
    x = root_ang_vel_b[env_id][0]
    y = root_ang_vel_b[env_id][1]
    return x * x + y * y


@wp.func
def _reward_joint_vel_l2(
    env_id: wp.int32,
    joint_vel: wp.array2d(dtype=wp.float32),
    num_joints: wp.int32,
) -> wp.float32:
    """Mirror mdp.joint_vel_l2 for cfg ``joint_vel`` (sum over all joints).

    cfg:     g1/29dof/velocity_env_cfg.py:311 (weight=-0.001)
    manager: isaaclab/envs/mdp/rewards.py:157
    Cross-robot reuse: ``num_joints`` is a runtime ``wp.int32``.
    """
    s = wp.float32(0.0)
    for j in range(num_joints):
        v = joint_vel[env_id, j]
        s += v * v
    return s


@wp.func
def _reward_joint_acc_l2(
    env_id: wp.int32,
    joint_acc: wp.array2d(dtype=wp.float32),
    num_joints: wp.int32,
) -> wp.float32:
    """Mirror mdp.joint_acc_l2 for cfg ``joint_acc``.

    cfg:     g1/29dof/velocity_env_cfg.py:312 (weight=-2.5e-7)
    manager: isaaclab/envs/mdp/rewards.py:169
    Cross-robot reuse: ``num_joints`` is a runtime ``wp.int32``.
    """
    s = wp.float32(0.0)
    for j in range(num_joints):
        a = joint_acc[env_id, j]
        s += a * a
    return s


@wp.func
def _reward_action_rate_l2(
    env_id: wp.int32,
    raw_actions: wp.array2d(dtype=wp.float32),
    prev_actions: wp.array2d(dtype=wp.float32),
    num_joints: wp.int32,
) -> wp.float32:
    """Mirror mdp.action_rate_l2 for cfg ``action_rate``.

    cfg:     g1/29dof/velocity_env_cfg.py:313 (weight=-0.05)
    manager: isaaclab/envs/mdp/rewards.py:259
    Cross-robot reuse: ``num_joints`` is a runtime ``wp.int32``.
    """
    s = wp.float32(0.0)
    for j in range(num_joints):
        d = raw_actions[env_id, j] - prev_actions[env_id, j]
        s += d * d
    return s


@wp.func
def _reward_joint_pos_limits(
    env_id: wp.int32,
    joint_pos: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    num_joints: wp.int32,
) -> wp.float32:
    """Mirror mdp.joint_pos_limits for cfg ``dof_pos_limits``.

    cfg:     g1/29dof/velocity_env_cfg.py:314 (weight=-5.0)
    manager: isaaclab/envs/mdp/rewards.py:193
    Cross-robot reuse: ``num_joints`` is a runtime ``wp.int32``.
    """
    s = wp.float32(0.0)
    for j in range(num_joints):
        lower = soft_joint_pos_limits[env_id, j][0]
        upper = soft_joint_pos_limits[env_id, j][1]
        pos = joint_pos[env_id, j]
        if pos < lower:
            s += lower - pos
        elif pos > upper:
            s += pos - upper
    return s


@wp.func
def _reward_energy(
    env_id: wp.int32,
    joint_vel: wp.array2d(dtype=wp.float32),
    applied_torque: wp.array2d(dtype=wp.float32),
    num_joints: wp.int32,
) -> wp.float32:
    """Mirror mdp.energy for cfg ``energy`` (project-internal mdp).

    cfg:     g1/29dof/velocity_env_cfg.py:315 (weight=-2e-5)
    manager: unitree_rl_lab/tasks/locomotion/mdp/rewards.py:23
    Cross-robot reuse: ``num_joints`` is a runtime ``wp.int32``.
    """
    s = wp.float32(0.0)
    for j in range(num_joints):
        s += wp.abs(joint_vel[env_id, j]) * wp.abs(applied_torque[env_id, j])
    return s


@wp.func
def _reward_joint_deviation_l1(
    env_id: wp.int32,
    joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    count: wp.int32,
) -> wp.float32:
    """Mirror mdp.joint_deviation_l1 for cfg ``joint_deviation_{arms,waists,legs}``.

    cfg:     g1/29dof/velocity_env_cfg.py:317-347
             (arms weight=-0.1, waists weight=-1, legs weight=-1.0)
    manager: isaaclab/envs/mdp/rewards.py:181
    Invoked 3 times in fusion kernel with different joint_ids arrays.
    """
    return _sum_joint_l1(joint_pos[env_id], default_joint_pos[env_id], joint_ids, count)


@wp.func
def _reward_flat_orientation_l2(
    env_id: wp.int32,
    projected_gravity_b: wp.array(dtype=wp.vec3f),
) -> wp.float32:
    """Mirror mdp.flat_orientation_l2 for cfg ``flat_orientation_l2``.

    cfg:     g1/29dof/velocity_env_cfg.py:350 (weight=-5.0)
    manager: isaaclab/envs/mdp/rewards.py:92
    """
    g = projected_gravity_b[env_id]
    return g[0] * g[0] + g[1] * g[1]


@wp.func
def _reward_base_height_l2(
    env_id: wp.int32,
    root_pose_w: wp.array(dtype=wp.transformf),
    target_height: wp.float32,
) -> wp.float32:
    """Mirror mdp.base_height_l2 for cfg ``base_height``.

    cfg:     g1/29dof/velocity_env_cfg.py:351 (weight=-10, target_height=0.78)
    manager: isaaclab/envs/mdp/rewards.py:102
    Flat-terrain only (terrain-offset path omitted; G1 / Go2 / H1 currently
    use flat ground so target_height is the absolute z target).
    """
    z = wp.transform_get_translation(root_pose_w[env_id])[2]
    err = z - target_height
    return err * err


@wp.func
def _reward_feet_gait(
    env_id: wp.int32,
    episode_length_buf: wp.array(dtype=wp.int32),
    commands: wp.array2d(dtype=wp.float32),
    current_contact_time: wp.array2d(dtype=wp.float32),
    foot_body_ids: wp.array(dtype=wp.int32),
    num_feet: wp.int32,
    foot_phase_offsets: wp.array(dtype=wp.float32),
    step_dt: wp.float32,
    gait_period: wp.float32,
    gait_threshold: wp.float32,
    gait_command_threshold: wp.float32,
) -> wp.float32:
    """Mirror mdp.feet_gait for cfg ``gait`` (project-internal mdp).

    cfg:     g1/29dof/velocity_env_cfg.py:354-364
             (weight=0.5, period=0.8, offset=[0,0.5], threshold=0.55)
    manager: unitree_rl_lab/tasks/locomotion/mdp/rewards.py:179
    Bonus when foot stance/swing phase agrees with actual contact, gated by command norm.
    Uses single-frame current_contact_time (different from history-max force used by
    feet_slide / undesired_contacts) — matches Manager's per-term signal choice.
    """
    cmd_norm = wp.sqrt(
        commands[env_id, 0] * commands[env_id, 0]
        + commands[env_id, 1] * commands[env_id, 1]
        + commands[env_id, 2] * commands[env_id, 2]
    )
    if cmd_norm <= gait_command_threshold:
        return wp.float32(0.0)
    episode_time = wp.float32(episode_length_buf[env_id]) * step_dt
    global_phase = (episode_time - gait_period * wp.floor(episode_time / gait_period)) / gait_period
    bonus = wp.float32(0.0)
    for foot_idx in range(num_feet):
        body_id = foot_body_ids[foot_idx]
        is_contact_gait = current_contact_time[env_id, body_id] > 0.0
        leg_phase = global_phase + foot_phase_offsets[foot_idx]
        leg_phase = leg_phase - wp.floor(leg_phase)
        is_stance = leg_phase < gait_threshold
        if is_stance == is_contact_gait:
            bonus += 1.0
    return bonus


@wp.func
def _reward_feet_slide(
    env_id: wp.int32,
    body_lin_vel_w: wp.array2d(dtype=wp.vec3f),
    foot_body_ids: wp.array(dtype=wp.int32),
    num_feet: wp.int32,
    net_forces_w_history: wp.array3d(dtype=wp.vec3f),
    contact_history_length: wp.int32,
    feet_slide_force_threshold: wp.float32,
) -> wp.float32:
    """Mirror mdp.feet_slide for cfg ``feet_slide``.

    cfg:     g1/29dof/velocity_env_cfg.py:365-372 (weight=-0.2, ankle_roll bodies)
    manager: isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py:72
    Manager's hard-coded ``> 1.0`` threshold exposed as cfg.feet_slide_force_threshold.
    """
    s = wp.float32(0.0)
    for foot_idx in range(num_feet):
        body_id = foot_body_ids[foot_idx]
        force_norm = _max_history_force_norm(
            net_forces_w_history, env_id, body_id, contact_history_length
        )
        is_contact = force_norm > feet_slide_force_threshold
        vx = body_lin_vel_w[env_id, body_id][0]
        vy = body_lin_vel_w[env_id, body_id][1]
        if is_contact:
            s += wp.sqrt(vx * vx + vy * vy)
    return s


@wp.func
def _reward_foot_clearance(
    env_id: wp.int32,
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_lin_vel_w: wp.array2d(dtype=wp.vec3f),
    foot_body_ids: wp.array(dtype=wp.int32),
    num_feet: wp.int32,
    target_height: wp.float32,
    std: wp.float32,
    tanh_mult: wp.float32,
) -> wp.float32:
    """Mirror mdp.foot_clearance_reward for cfg ``feet_clearance`` (project-internal mdp).

    cfg:     g1/29dof/velocity_env_cfg.py:373-382
             (weight=1.0, std=0.05, tanh_mult=2.0, target_height=0.1)
    manager: unitree_rl_lab/tasks/locomotion/mdp/rewards.py:125
    Returns INNER SUM ``Σ foot_z_err² · tanh(tanh_mult * |v_xy|)``; caller applies
    ``exp(-sum / std)`` envelope so the wp.func stays composable with weight × dt.
    """
    s = wp.float32(0.0)
    for foot_idx in range(num_feet):
        body_id = foot_body_ids[foot_idx]
        vx = body_lin_vel_w[env_id, body_id][0]
        vy = body_lin_vel_w[env_id, body_id][1]
        z_err = body_pos_w[env_id, body_id][2] - target_height
        s += z_err * z_err * wp.tanh(tanh_mult * wp.sqrt(vx * vx + vy * vy))
    return s


@wp.func
def _reward_undesired_contacts(
    env_id: wp.int32,
    net_forces_w_history: wp.array3d(dtype=wp.vec3f),
    undesired_body_ids: wp.array(dtype=wp.int32),
    num_undesired: wp.int32,
    contact_history_length: wp.int32,
    threshold: wp.float32,
) -> wp.float32:
    """Mirror mdp.undesired_contacts for cfg ``undesired_contacts``.

    cfg:     g1/29dof/velocity_env_cfg.py:385-392 (weight=-1, threshold=1.0 N)
    manager: isaaclab/envs/mdp/rewards.py:274
    Body regex ``"(?!.*ankle.*).*"`` excludes both ankle_pitch and ankle_roll;
    resolved host-side in __init__ and passed as undesired_body_ids.
    """
    n = wp.float32(0.0)
    for body_idx in range(num_undesired):
        body_id = undesired_body_ids[body_idx]
        force_norm = _max_history_force_norm(
            net_forces_w_history, env_id, body_id, contact_history_length
        )
        if force_norm > threshold:
            n += 1.0
    return n
