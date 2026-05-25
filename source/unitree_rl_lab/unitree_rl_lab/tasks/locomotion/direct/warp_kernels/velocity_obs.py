# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-task observation kernels and per-term ``wp.func`` helpers.

Reusable across velocity-tracking tasks (G1 / Go2 / H1) that follow the
shared Manager ``ObservationsCfg.PolicyCfg`` field order:

    [0:3]                       base_ang_vel * scale     (3 lanes)
    [3:6]                       projected_gravity_b      (3 lanes)
    [6:9]                       velocity_commands        (3 lanes)
    [9:9+N]                     joint_pos_rel            (N lanes)
    [9+N:9+2N]                  joint_vel_rel * scale    (N lanes)
    [9+2N:9+3N]                 last_action              (N lanes)

The base section (lanes 0:9) is dimension-independent and reused as-is.
The joint section (lanes 9:9+3N) takes ``num_joints`` and ``joint_lanes``
as runtime ``wp.int32`` parameters so each robot launches the same kernels
with its own ``num_joints``. Critic frame is the same layout shifted by
+3 (extra ``base_lin_vel`` prepended at [0:3]).

RNG: noise-drawing wp.funcs take state by value and return advanced state;
parent kernel threads the state through call sites in cfg-declaration
order so RNG-advance count matches Manager NoiseModelWithAdditiveBias.
"""

from __future__ import annotations

import warp as wp


# ============================================================================
# Per-term policy observation wp.func — one per ObservationsCfg.PolicyCfg term.
# Each wp.func fuses raw + noise + scale (the 3 Manager ops driven by
# ObservationManager.compute_group:392-407) into one inline write.
# ============================================================================


@wp.func
def _obs_base_ang_vel_noisy(
    env_id: wp.int32,
    state: wp.uint32,
    root_ang_vel_b: wp.array(dtype=wp.vec3f),
    scale: wp.float32,
    noise_range: wp.vec2f,
    frame: wp.array2d(dtype=wp.float32),
    frame_offset: wp.int32,
) -> wp.uint32:
    """Fused replacement: PolicyCfg.base_ang_vel (raw + noise + scale).

    cfg:     g1/29dof/velocity_env_cfg.py:252
    manager: isaaclab/envs/mdp/observations.py:65            (mdp.base_ang_vel)
           + isaaclab/utils/noise/noise_model.py:46          (Unoise -> uniform_noise)
           + isaaclab/managers/observation_manager.py:407    (obs.mul_(scale))
    """
    s = state
    for i in range(3):
        raw = root_ang_vel_b[env_id][i] + wp.randf(s, noise_range[0], noise_range[1])
        s += wp.uint32(1)
        frame[env_id, frame_offset + i] = raw * scale
    return s


@wp.func
def _obs_projected_gravity_noisy(
    env_id: wp.int32,
    state: wp.uint32,
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    noise_range: wp.vec2f,
    frame: wp.array2d(dtype=wp.float32),
    frame_offset: wp.int32,
) -> wp.uint32:
    """Fused replacement: PolicyCfg.projected_gravity (raw + noise, no scale).

    cfg:     g1/29dof/velocity_env_cfg.py:253
    manager: isaaclab/envs/mdp/observations.py:75            (mdp.projected_gravity)
           + isaaclab/utils/noise/noise_model.py:46          (Unoise -> uniform_noise)
    """
    s = state
    for i in range(3):
        raw = projected_gravity_b[env_id][i] + wp.randf(s, noise_range[0], noise_range[1])
        s += wp.uint32(1)
        frame[env_id, frame_offset + i] = raw
    return s


@wp.func
def _obs_velocity_commands(
    env_id: wp.int32,
    commands: wp.array2d(dtype=wp.float32),
    frame: wp.array2d(dtype=wp.float32),
    frame_offset: wp.int32,
):
    """Fused replacement: PolicyCfg.velocity_commands (no noise, no scale).

    cfg:     g1/29dof/velocity_env_cfg.py:254
    manager: isaaclab/envs/mdp/observations.py:677           (mdp.generated_commands)
    Reads the env's own ``commands`` buffer, replacing CommandManager.
    """
    for i in range(3):
        frame[env_id, frame_offset + i] = commands[env_id, i]


@wp.func
def _obs_joint_pos_rel_noisy(
    env_id: wp.int32,
    state: wp.uint32,
    joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    num_joints: wp.int32,
    noise_range: wp.vec2f,
    frame: wp.array2d(dtype=wp.float32),
    frame_offset: wp.int32,
) -> wp.uint32:
    """Fused replacement: PolicyCfg.joint_pos_rel (raw + noise, no scale).

    cfg:     g1/29dof/velocity_env_cfg.py:255
    manager: isaaclab/envs/mdp/observations.py:213           (mdp.joint_pos_rel)
           + isaaclab/utils/noise/noise_model.py:46          (Unoise -> uniform_noise)
    Cross-robot reuse: ``num_joints`` is a runtime ``wp.int32`` (29 for G1,
    12 for Go2, 19 for H1).
    """
    s = state
    for j in range(num_joints):
        raw = joint_pos[env_id, j] - default_joint_pos[env_id, j] + wp.randf(
            s, noise_range[0], noise_range[1]
        )
        s += wp.uint32(1)
        frame[env_id, frame_offset + j] = raw
    return s


@wp.func
def _obs_joint_vel_rel_noisy(
    env_id: wp.int32,
    state: wp.uint32,
    joint_vel: wp.array2d(dtype=wp.float32),
    num_joints: wp.int32,
    scale: wp.float32,
    noise_range: wp.vec2f,
    frame: wp.array2d(dtype=wp.float32),
    frame_offset: wp.int32,
) -> wp.uint32:
    """Fused replacement: PolicyCfg.joint_vel_rel (raw + noise + scale).

    cfg:     g1/29dof/velocity_env_cfg.py:256
    manager: isaaclab/envs/mdp/observations.py:261           (mdp.joint_vel_rel)
           + isaaclab/utils/noise/noise_model.py:46          (Unoise -> uniform_noise)
           + isaaclab/managers/observation_manager.py:407    (obs.mul_(scale))
    Note: G1 / Go2 / H1 all have ``default_joint_vel = 0``, so the subtraction
    against default_joint_vel is omitted (bit-equivalent for these robots).
    """
    s = state
    for j in range(num_joints):
        raw = joint_vel[env_id, j] + wp.randf(s, noise_range[0], noise_range[1])
        s += wp.uint32(1)
        frame[env_id, frame_offset + j] = raw * scale
    return s


@wp.func
def _obs_last_action(
    env_id: wp.int32,
    raw_actions: wp.array2d(dtype=wp.float32),
    num_joints: wp.int32,
    frame: wp.array2d(dtype=wp.float32),
    frame_offset: wp.int32,
):
    """Fused replacement: PolicyCfg.last_action (no noise, no scale).

    cfg:     g1/29dof/velocity_env_cfg.py:257
    manager: isaaclab/envs/mdp/observations.py:659           (mdp.last_action)
    """
    for j in range(num_joints):
        frame[env_id, frame_offset + j] = raw_actions[env_id, j]


# ============================================================================
# Frame kernels — invoke per-term wp.funcs in cfg-declared order. Base frame
# is dimension-independent; joint frame takes ``num_joints`` / ``joint_lanes``
# so the same kernel binary serves any robot.
# ============================================================================


@wp.kernel
def compute_policy_base_frame_kernel(
    root_ang_vel_b: wp.array(dtype=wp.vec3f),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    commands: wp.array2d(dtype=wp.float32),
    base_ang_vel_scale: wp.float32,
    frame: wp.array2d(dtype=wp.float32),
):
    """Policy frame BASE section, no-noise variant: lanes [0:9].

    cfg:     g1/29dof/velocity_env_cfg.py:252-254 (PolicyCfg fields 1-3)
    manager: isaaclab/managers/observation_manager.py:392-407 (compute_group loop)
    Layout: [0:3] base_ang_vel * 0.2 | [3:6] projected_gravity | [6:9] velocity_commands
    Launched ``dim=(num_envs, 4)`` (lane 3 no-op).  Used when
    cfg.enable_observation_noise=False; noisy variant: ``compute_policy_base_frame_noisy_kernel``.
    """
    env_id, lane = wp.tid()
    if lane < 3:
        frame[env_id, lane] = root_ang_vel_b[env_id][lane] * base_ang_vel_scale
        frame[env_id, 3 + lane] = projected_gravity_b[env_id][lane]
        frame[env_id, 6 + lane] = commands[env_id, lane]


@wp.kernel
def compute_policy_joint_frame_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    raw_actions: wp.array2d(dtype=wp.float32),
    joint_vel_scale: wp.float32,
    num_joints: wp.int32,
    joint_lanes: wp.int32,
    frame: wp.array2d(dtype=wp.float32),
):
    """Policy frame JOINT section, no-noise variant: lanes [9:9+3N].

    cfg:     g1/29dof/velocity_env_cfg.py:255-257 (PolicyCfg fields 4-6)
    manager: isaaclab/managers/observation_manager.py:392-407 (compute_group loop)
    Layout: [9:9+N] joint_pos_rel | [9+N:9+2N] joint_vel * scale | [9+2N:9+3N] last_action
    Launched ``dim=(num_envs, joint_lanes)``; lane[i] handles joint[i] via stride
    loop (forward-compatible if num_joints > joint_lanes). Cross-robot reuse:
    ``num_joints`` and ``joint_lanes`` are runtime ``wp.int32`` parameters.
    """
    env_id, lane = wp.tid()
    joint_pos_offset = 9
    joint_vel_offset = joint_pos_offset + num_joints
    action_offset = joint_vel_offset + num_joints
    joint_id = lane
    while joint_id < num_joints:
        frame[env_id, joint_pos_offset + joint_id] = joint_pos[env_id, joint_id] - default_joint_pos[env_id, joint_id]
        frame[env_id, joint_vel_offset + joint_id] = joint_vel[env_id, joint_id] * joint_vel_scale
        frame[env_id, action_offset + joint_id] = raw_actions[env_id, joint_id]
        joint_id += joint_lanes


@wp.kernel
def compute_policy_base_frame_noisy_kernel(
    root_ang_vel_b: wp.array(dtype=wp.vec3f),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    commands: wp.array2d(dtype=wp.float32),
    base_ang_vel_scale: wp.float32,
    obs_noise_rng_state: wp.array(dtype=wp.uint32),
    noise_base_ang_vel_range: wp.vec2f,
    noise_projected_gravity_range: wp.vec2f,
    frame: wp.array2d(dtype=wp.float32),
):
    """Policy frame BASE section, noisy variant: lanes [0:9].

    cfg:     g1/29dof/velocity_env_cfg.py:252-254 (PolicyCfg fields 1-3)
    manager: isaaclab/managers/observation_manager.py:392-407 (compute_group loop)
    Dispatches: _obs_base_ang_vel_noisy + _obs_projected_gravity_noisy + _obs_velocity_commands.
    Launched ``dim=num_envs`` single-thread (lane parallel would race on per-env RNG state).
    """
    env_id = wp.tid()
    state = obs_noise_rng_state[env_id]
    state = _obs_base_ang_vel_noisy(
        env_id, state, root_ang_vel_b, base_ang_vel_scale, noise_base_ang_vel_range, frame, 0
    )
    state = _obs_projected_gravity_noisy(
        env_id, state, projected_gravity_b, noise_projected_gravity_range, frame, 3
    )
    _obs_velocity_commands(env_id, commands, frame, 6)
    obs_noise_rng_state[env_id] = state


@wp.kernel
def compute_policy_joint_frame_noisy_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    raw_actions: wp.array2d(dtype=wp.float32),
    joint_vel_scale: wp.float32,
    num_joints: wp.int32,
    obs_noise_rng_state: wp.array(dtype=wp.uint32),
    noise_joint_pos_range: wp.vec2f,
    noise_joint_vel_range: wp.vec2f,
    frame: wp.array2d(dtype=wp.float32),
):
    """Policy frame JOINT section, noisy variant: lanes [9:9+3N].

    cfg:     g1/29dof/velocity_env_cfg.py:255-257 (PolicyCfg fields 4-6)
    manager: isaaclab/managers/observation_manager.py:392-407 (compute_group loop)
    Dispatches: _obs_joint_pos_rel_noisy + _obs_joint_vel_rel_noisy + _obs_last_action.
    Launched ``dim=num_envs`` single-thread (RNG-race reason same as base noisy variant).
    Cross-robot reuse: ``num_joints`` is a runtime ``wp.int32`` parameter.
    """
    env_id = wp.tid()
    joint_pos_offset = 9
    joint_vel_offset = joint_pos_offset + num_joints
    action_offset = joint_vel_offset + num_joints
    state = obs_noise_rng_state[env_id]
    state = _obs_joint_pos_rel_noisy(
        env_id, state, joint_pos, default_joint_pos, num_joints,
        noise_joint_pos_range, frame, joint_pos_offset,
    )
    state = _obs_joint_vel_rel_noisy(
        env_id, state, joint_vel, num_joints, joint_vel_scale,
        noise_joint_vel_range, frame, joint_vel_offset,
    )
    _obs_last_action(env_id, raw_actions, num_joints, frame, action_offset)
    obs_noise_rng_state[env_id] = state


@wp.kernel
def compute_critic_base_frame_kernel(
    root_lin_vel_b: wp.array(dtype=wp.vec3f),
    root_ang_vel_b: wp.array(dtype=wp.vec3f),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    commands: wp.array2d(dtype=wp.float32),
    base_ang_vel_scale: wp.float32,
    frame: wp.array2d(dtype=wp.float32),
):
    """Critic frame BASE section: lanes [0:12].  Privileged (no noise).

    cfg:     g1/29dof/velocity_env_cfg.py:272-275 (CriticCfg fields 1-4)
    manager: isaaclab/managers/observation_manager.py:392-407 (compute_group loop)
    Layout: [0:3] base_lin_vel | [3:6] base_ang_vel * 0.2 | [6:9] projected_gravity
            | [9:12] velocity_commands.  Extra base_lin_vel vs policy frame.
    No rng_state param — structurally cannot apply noise.
    """
    env_id, lane = wp.tid()
    if lane < 3:
        frame[env_id, lane] = root_lin_vel_b[env_id][lane]
        frame[env_id, 3 + lane] = root_ang_vel_b[env_id][lane] * base_ang_vel_scale
        frame[env_id, 6 + lane] = projected_gravity_b[env_id][lane]
        frame[env_id, 9 + lane] = commands[env_id, lane]


@wp.kernel
def compute_critic_joint_frame_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    raw_actions: wp.array2d(dtype=wp.float32),
    joint_vel_scale: wp.float32,
    num_joints: wp.int32,
    joint_lanes: wp.int32,
    frame: wp.array2d(dtype=wp.float32),
):
    """Critic frame JOINT section: lanes [12:12+3N].  Privileged (no noise).

    cfg:     g1/29dof/velocity_env_cfg.py:276-278 (CriticCfg fields 5-7)
    manager: isaaclab/managers/observation_manager.py:392-407 (compute_group loop)
    Layout: [12:12+N] joint_pos_rel | [12+N:12+2N] joint_vel * scale | [12+2N:12+3N] last_action
    Same logic as policy joint frame but offsets shifted by +3 (critic base is 12 vs 9).
    Cross-robot reuse: ``num_joints`` and ``joint_lanes`` are runtime ``wp.int32`` parameters.
    """
    env_id, lane = wp.tid()
    joint_pos_offset = 12
    joint_vel_offset = joint_pos_offset + num_joints
    action_offset = joint_vel_offset + num_joints
    joint_id = lane
    while joint_id < num_joints:
        frame[env_id, joint_pos_offset + joint_id] = joint_pos[env_id, joint_id] - default_joint_pos[env_id, joint_id]
        frame[env_id, joint_vel_offset + joint_id] = joint_vel[env_id, joint_id] * joint_vel_scale
        frame[env_id, action_offset + joint_id] = raw_actions[env_id, joint_id]
        joint_id += joint_lanes
