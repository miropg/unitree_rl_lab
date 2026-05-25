# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1-specific fusion reward kernels — bind ``RewardsCfg`` term order to
the per-term ``wp.func`` library in :mod:`.warp_kernels.velocity_rewards`.

Why this file is robot-specific
================================

The per-term reward ``wp.func`` library is fully cross-robot (G1 / Go2 /
H1 share the same 19-term core), but the *fusion* kernel that dispatches
those wp.funcs in cfg-declared order is robot-specific because:

1. Term order MUST match :class:`RewardsCfg` declaration to keep the
   floating-point ``+=`` accumulation bit-stable across runs (Manager
   parity).
2. Each robot's ``RewardsCfg`` declares its weights / targets / std
   parameters; the fusion kernel parameter list mirrors those.
3. Warp does not currently support generic dispatch over a variable
   parameter list; a future Go2 / H1 fusion kernel will copy this file's
   structure with its own ``RewardsCfg`` order and weights.

When adding a new robot:

* Copy this file → ``velocity_warp_env_<robot>_kernels.py``.
* Reorder the wp.func dispatch calls to match the new robot's
  ``RewardsCfg`` field declaration.
* Drop / add ``wp.func`` imports for terms the robot adds or removes.
* Pass the robot's ``num_joints`` (e.g. ``GO2_NUM_JOINTS = 12``) into
  every joint-aggregating wp.func.
"""

from __future__ import annotations

import warp as wp

from .warp_kernels.velocity_rewards import (
    _reward_action_rate_l2,
    _reward_alive,
    _reward_ang_vel_xy_l2,
    _reward_base_height_l2,
    _reward_energy,
    _reward_feet_gait,
    _reward_feet_slide,
    _reward_flat_orientation_l2,
    _reward_foot_clearance,
    _reward_joint_acc_l2,
    _reward_joint_deviation_l1,
    _reward_joint_pos_limits,
    _reward_joint_vel_l2,
    _reward_lin_vel_z_l2,
    _reward_track_ang_vel_z,
    _reward_track_lin_vel_xy,
    _reward_undesired_contacts,
)


@wp.kernel
def compute_rewards_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    root_quat_w: wp.array(dtype=wp.quatf),
    root_lin_vel_w: wp.array(dtype=wp.vec3f),
    root_lin_vel_b: wp.array(dtype=wp.vec3f),
    root_ang_vel_b: wp.array(dtype=wp.vec3f),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
    applied_torque: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    raw_actions: wp.array2d(dtype=wp.float32),
    prev_actions: wp.array2d(dtype=wp.float32),
    commands: wp.array2d(dtype=wp.float32),
    reset_terminated: wp.array(dtype=wp.bool),
    num_joints: wp.int32,
    arm_joint_ids: wp.array(dtype=wp.int32),
    num_arm_joints: wp.int32,
    waist_joint_ids: wp.array(dtype=wp.int32),
    num_waist_joints: wp.int32,
    leg_joint_ids: wp.array(dtype=wp.int32),
    num_leg_joints: wp.int32,
    step_dt: wp.float32,
    rew_track_lin_vel_xy: wp.float32,
    track_lin_vel_xy_std_sq: wp.float32,
    rew_track_ang_vel_z: wp.float32,
    track_ang_vel_z_std_sq: wp.float32,
    rew_alive: wp.float32,
    rew_base_linear_velocity: wp.float32,
    rew_base_angular_velocity: wp.float32,
    rew_flat_orientation_l2: wp.float32,
    rew_base_height: wp.float32,
    base_height_target: wp.float32,
    rew_joint_vel: wp.float32,
    rew_joint_acc: wp.float32,
    rew_action_rate: wp.float32,
    rew_dof_pos_limits: wp.float32,
    rew_energy: wp.float32,
    rew_joint_deviation_arms: wp.float32,
    rew_joint_deviation_waists: wp.float32,
    rew_joint_deviation_legs: wp.float32,
    rewards: wp.array(dtype=wp.float32),
    track_lin_vel_xy_step: wp.array(dtype=wp.float32),
    track_ang_vel_z_step: wp.array(dtype=wp.float32),
    reward_terms_step: wp.array2d(dtype=wp.float32),
):
    """Fused replacement: RewardManager.compute (15 non-contact RewardsCfg terms).

    cfg:     g1/29dof/velocity_env_cfg.py:293-392 (RewardsCfg)
    manager: isaaclab/managers/reward_manager.py:129-159 (compute loop)

    Per-term: ``total += _reward_<name>(...) * weight * step_dt``.
    Term order MUST match RewardsCfg declaration (FP ``+=`` non-commutative
    → reordering introduces ~1e-6 noise vs Manager).

    reward_terms_step buffer layout (15 lanes here, 4 more from
    ``add_contact_rewards_kernel`` for lanes [15:18]):
      [0] track_lin_vel_xy   [5] joint_vel        [10] joint_dev_arms   [13] flat_orient_l2
      [1] track_ang_vel_z    [6] joint_acc        [11] joint_dev_waists [14] base_height
      [2] alive              [7] action_rate      [12] joint_dev_legs   [15-18] contact terms
      [3] base_lin_vel       [8] dof_pos_limits
      [4] base_ang_vel       [9] energy

    Also writes track_lin_vel_xy_step / track_ang_vel_z_step (consumed
    host-side by curriculum to mirror mdp.lin_vel_cmd_levels).
    """
    env_id = wp.tid()
    rew = wp.float32(0.0)

    # [0] RewardsCfg.track_lin_vel_xy → ``mdp.track_lin_vel_xy_yaw_frame_exp``.
    val = _reward_track_lin_vel_xy(env_id, root_quat_w, root_lin_vel_w, commands, track_lin_vel_xy_std_sq)
    step_val = val * rew_track_lin_vel_xy * step_dt
    track_lin_vel_xy_step[env_id] = step_val
    reward_terms_step[env_id, 0] = step_val
    rew += step_val

    # [1] RewardsCfg.track_ang_vel_z → ``mdp.track_ang_vel_z_exp``.
    val = _reward_track_ang_vel_z(env_id, root_ang_vel_b, commands, track_ang_vel_z_std_sq)
    step_val = val * rew_track_ang_vel_z * step_dt
    track_ang_vel_z_step[env_id] = step_val
    reward_terms_step[env_id, 1] = step_val
    rew += step_val

    # [2] RewardsCfg.alive → ``mdp.is_alive``.
    val = _reward_alive(env_id, reset_terminated)
    step_val = val * rew_alive * step_dt
    reward_terms_step[env_id, 2] = step_val
    rew += step_val

    # [3] RewardsCfg.base_linear_velocity → ``mdp.lin_vel_z_l2``.
    val = _reward_lin_vel_z_l2(env_id, root_lin_vel_b)
    step_val = val * rew_base_linear_velocity * step_dt
    reward_terms_step[env_id, 3] = step_val
    rew += step_val

    # [4] RewardsCfg.base_angular_velocity → ``mdp.ang_vel_xy_l2``.
    val = _reward_ang_vel_xy_l2(env_id, root_ang_vel_b)
    step_val = val * rew_base_angular_velocity * step_dt
    reward_terms_step[env_id, 4] = step_val
    rew += step_val

    # [5] RewardsCfg.joint_vel → ``mdp.joint_vel_l2``.
    val = _reward_joint_vel_l2(env_id, joint_vel, num_joints)
    step_val = val * rew_joint_vel * step_dt
    reward_terms_step[env_id, 5] = step_val
    rew += step_val

    # [6] RewardsCfg.joint_acc → ``mdp.joint_acc_l2``.
    val = _reward_joint_acc_l2(env_id, joint_acc, num_joints)
    step_val = val * rew_joint_acc * step_dt
    reward_terms_step[env_id, 6] = step_val
    rew += step_val

    # [7] RewardsCfg.action_rate → ``mdp.action_rate_l2``.
    val = _reward_action_rate_l2(env_id, raw_actions, prev_actions, num_joints)
    step_val = val * rew_action_rate * step_dt
    reward_terms_step[env_id, 7] = step_val
    rew += step_val

    # [8] RewardsCfg.dof_pos_limits → ``mdp.joint_pos_limits``.
    val = _reward_joint_pos_limits(env_id, joint_pos, soft_joint_pos_limits, num_joints)
    step_val = val * rew_dof_pos_limits * step_dt
    reward_terms_step[env_id, 8] = step_val
    rew += step_val

    # [9] RewardsCfg.energy → ``mdp.energy`` (project-internal).
    val = _reward_energy(env_id, joint_vel, applied_torque, num_joints)
    step_val = val * rew_energy * step_dt
    reward_terms_step[env_id, 9] = step_val
    rew += step_val

    # [10] RewardsCfg.joint_deviation_arms → ``mdp.joint_deviation_l1`` (arms).
    val = _reward_joint_deviation_l1(env_id, joint_pos, default_joint_pos, arm_joint_ids, num_arm_joints)
    step_val = val * rew_joint_deviation_arms * step_dt
    reward_terms_step[env_id, 10] = step_val
    rew += step_val

    # [11] RewardsCfg.joint_deviation_waists → ``mdp.joint_deviation_l1`` (waists).
    val = _reward_joint_deviation_l1(env_id, joint_pos, default_joint_pos, waist_joint_ids, num_waist_joints)
    step_val = val * rew_joint_deviation_waists * step_dt
    reward_terms_step[env_id, 11] = step_val
    rew += step_val

    # [12] RewardsCfg.joint_deviation_legs → ``mdp.joint_deviation_l1`` (legs).
    val = _reward_joint_deviation_l1(env_id, joint_pos, default_joint_pos, leg_joint_ids, num_leg_joints)
    step_val = val * rew_joint_deviation_legs * step_dt
    reward_terms_step[env_id, 12] = step_val
    rew += step_val

    # [13] RewardsCfg.flat_orientation_l2 → ``mdp.flat_orientation_l2``.
    val = _reward_flat_orientation_l2(env_id, projected_gravity_b)
    step_val = val * rew_flat_orientation_l2 * step_dt
    reward_terms_step[env_id, 13] = step_val
    rew += step_val

    # [14] RewardsCfg.base_height → ``mdp.base_height_l2``.
    val = _reward_base_height_l2(env_id, root_pose_w, base_height_target)
    step_val = val * rew_base_height * step_dt
    reward_terms_step[env_id, 14] = step_val
    rew += step_val

    rewards[env_id] = rew


@wp.kernel
def add_contact_rewards_kernel(
    episode_length_buf: wp.array(dtype=wp.int32),
    commands: wp.array2d(dtype=wp.float32),
    net_forces_w_history: wp.array3d(dtype=wp.vec3f),
    current_contact_time: wp.array2d(dtype=wp.float32),
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_lin_vel_w: wp.array2d(dtype=wp.vec3f),
    foot_body_ids: wp.array(dtype=wp.int32),
    num_feet: wp.int32,
    foot_phase_offsets: wp.array(dtype=wp.float32),
    undesired_body_ids: wp.array(dtype=wp.int32),
    num_undesired: wp.int32,
    contact_history_length: wp.int32,
    step_dt: wp.float32,
    gait_period: wp.float32,
    gait_threshold: wp.float32,
    gait_command_threshold: wp.float32,
    feet_slide_force_threshold: wp.float32,
    rew_gait: wp.float32,
    rew_feet_slide: wp.float32,
    rew_feet_clearance: wp.float32,
    feet_clearance_target_height: wp.float32,
    feet_clearance_std: wp.float32,
    feet_clearance_tanh_mult: wp.float32,
    undesired_contacts_threshold: wp.float32,
    rew_undesired_contacts: wp.float32,
    rewards: wp.array(dtype=wp.float32),
    reward_terms_step: wp.array2d(dtype=wp.float32),
):
    """Fused replacement: RewardManager.compute (4 contact-dependent RewardsCfg terms).

    cfg:     g1/29dof/velocity_env_cfg.py:354-392 (gait / feet_slide /
                                                    feet_clearance / undesired_contacts)
    manager: isaaclab/managers/reward_manager.py:129-159 (compute loop)

    Why a 2nd kernel: these 4 terms read contact_sensor data
    (net_forces_w_history / current_contact_time / body_pos_w / body_lin_vel_w),
    only available when self.contact_sensor != None.  Splitting keeps
    ``compute_rewards_kernel`` callable on minimal scenes (parity tests).

    Term-add order mirrors Manager :class:`RewardsCfg`
    (gait → feet_slide → feet_clearance → undesired_contacts) so the
    fused total is bit-stable across runs and matches Manager
    floating-point accumulation order.

    Per-term contributions are written to ``reward_terms_step`` lanes
    [15:19] (15=gait, 16=feet_slide, 17=feet_clearance, 18=undesired_contacts)
    for the host-side TB log emitter (see ``compute_rewards_kernel``
    docstring for the full 19-slot layout).  Host only reads this buffer
    via zero-copy ``wp.to_torch`` in :meth:`step`, never writes inside
    the captured graph, so it does not break CUDA graph capture.
    """
    env_id = wp.tid()

    # [15] RewardsCfg.gait → ``mdp.feet_gait`` (project-internal).
    val = _reward_feet_gait(
        env_id,
        episode_length_buf,
        commands,
        current_contact_time,
        foot_body_ids,
        num_feet,
        foot_phase_offsets,
        step_dt,
        gait_period,
        gait_threshold,
        gait_command_threshold,
    )
    gait_step = rew_gait * val * step_dt
    reward_terms_step[env_id, 15] = gait_step
    rewards[env_id] += gait_step

    # [16] RewardsCfg.feet_slide → ``mdp.feet_slide`` (upstream task mdp).
    val = _reward_feet_slide(
        env_id,
        body_lin_vel_w,
        foot_body_ids,
        num_feet,
        net_forces_w_history,
        contact_history_length,
        feet_slide_force_threshold,
    )
    slide_step = rew_feet_slide * val * step_dt
    reward_terms_step[env_id, 16] = slide_step
    rewards[env_id] += slide_step

    # [17] RewardsCfg.feet_clearance → ``mdp.foot_clearance_reward`` (project-internal).
    # The wp.func returns the inner-sum; apply the ``exp(-sum / std)``
    # envelope here (mirrors Manager which folds it into the same expression).
    inner = _reward_foot_clearance(
        env_id,
        body_pos_w,
        body_lin_vel_w,
        foot_body_ids,
        num_feet,
        feet_clearance_target_height,
        feet_clearance_std,
        feet_clearance_tanh_mult,
    )
    clearance_step = rew_feet_clearance * wp.exp(-inner / feet_clearance_std) * step_dt
    reward_terms_step[env_id, 17] = clearance_step
    rewards[env_id] += clearance_step

    # [18] RewardsCfg.undesired_contacts → ``mdp.undesired_contacts``.
    val = _reward_undesired_contacts(
        env_id,
        net_forces_w_history,
        undesired_body_ids,
        num_undesired,
        contact_history_length,
        undesired_contacts_threshold,
    )
    undesired_step = rew_undesired_contacts * val * step_dt
    reward_terms_step[env_id, 18] = undesired_step
    rewards[env_id] += undesired_step
