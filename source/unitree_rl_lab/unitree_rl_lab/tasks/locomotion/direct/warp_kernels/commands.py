# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-command sample / update kernels — replacement for
:class:`isaaclab.envs.mdp.commands.UniformVelocityCommand`.

Cross-task reusable for any env that emits a 3-d ``[lin_vel_x, lin_vel_y,
ang_vel_z]`` command from uniform ranges. Used by G1 / Go2 / H1 velocity
trackers; will support future :mod:`unitree_rl_lab.tasks.locomotion` envs.

Range params are passed as ``wp.array(dtype=wp.float32)`` (length 2)
rather than ``wp.vec2f`` so curriculum can mutate ranges in-place across
CUDA-graph replays — ``wp.vec2f`` is captured by value and freezes once
the graph is recorded.

Limitation: only supports ``cfg.heading_command=False`` (velocity mode).
Heading mode (where ang_vel_z is auto-derived from a sampled target heading)
is not implemented — kernel signatures have no heading-related params and
``cfg.rel_heading_envs`` is therefore a dead field. Setting
``cfg.heading_command=True`` will silently fall back to velocity mode here.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def sample_reset_commands_kernel(
    mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    command_time_left: wp.array(dtype=wp.float32),
    commands: wp.array2d(dtype=wp.float32),
    resampling_time: wp.float32,
    rel_standing_envs: wp.float32,
    lin_vel_x_range: wp.array(dtype=wp.float32),
    lin_vel_y_range: wp.array(dtype=wp.float32),
    ang_vel_z_range: wp.array(dtype=wp.float32),
):
    """Replacement for UniformVelocityCommand.reset — sample fresh commands.

    cfg:     g1/29dof/velocity_env_cfg.py:215-231 (CommandsCfg.base_velocity)
    manager: unitree_rl_lab/tasks/locomotion/mdp/commands/velocity_command.py:10
             (UniformLevelVelocityCommandCfg, this repo's subclass with limit_ranges)
           + isaaclab/envs/mdp/commands/velocity_command.py (UniformVelocityCommand base)
    """
    env_id = wp.tid()
    if mask[env_id]:
        stand_sample = wp.randf(rng_state[env_id], 0.0, 1.0)
        rng_state[env_id] += wp.uint32(1)
        if stand_sample < rel_standing_envs:
            commands[env_id, 0] = 0.0
            commands[env_id, 1] = 0.0
            commands[env_id, 2] = 0.0
        else:
            commands[env_id, 0] = wp.randf(rng_state[env_id], lin_vel_x_range[0], lin_vel_x_range[1])
            rng_state[env_id] += wp.uint32(1)
            commands[env_id, 1] = wp.randf(rng_state[env_id], lin_vel_y_range[0], lin_vel_y_range[1])
            rng_state[env_id] += wp.uint32(1)
            commands[env_id, 2] = wp.randf(rng_state[env_id], ang_vel_z_range[0], ang_vel_z_range[1])
            rng_state[env_id] += wp.uint32(1)
        command_time_left[env_id] = resampling_time


@wp.kernel
def update_interval_commands_kernel(
    rng_state: wp.array(dtype=wp.uint32),
    command_time_left: wp.array(dtype=wp.float32),
    commands: wp.array2d(dtype=wp.float32),
    step_dt: wp.float32,
    resampling_time: wp.float32,
    rel_standing_envs: wp.float32,
    lin_vel_x_range: wp.array(dtype=wp.float32),
    lin_vel_y_range: wp.array(dtype=wp.float32),
    ang_vel_z_range: wp.array(dtype=wp.float32),
):
    """Replacement for UniformVelocityCommand.compute — resample on timer expiry.

    cfg:     g1/29dof/velocity_env_cfg.py:215-231
    manager: isaaclab/envs/mdp/commands/velocity_command.py (UniformVelocityCommand.compute)
           + isaaclab/envs/manager_based_rl_env.py:239-246  (called from step, between
                                                             _reset_idx and interval event)
    Same wp.array range argument as ``sample_reset_commands_kernel`` for
    curriculum mutability.
    """
    env_id = wp.tid()
    command_time_left[env_id] -= step_dt
    if command_time_left[env_id] <= 0.0:
        stand_sample = wp.randf(rng_state[env_id], 0.0, 1.0)
        rng_state[env_id] += wp.uint32(1)
        if stand_sample < rel_standing_envs:
            commands[env_id, 0] = 0.0
            commands[env_id, 1] = 0.0
            commands[env_id, 2] = 0.0
        else:
            commands[env_id, 0] = wp.randf(rng_state[env_id], lin_vel_x_range[0], lin_vel_x_range[1])
            rng_state[env_id] += wp.uint32(1)
            commands[env_id, 1] = wp.randf(rng_state[env_id], lin_vel_y_range[0], lin_vel_y_range[1])
            rng_state[env_id] += wp.uint32(1)
            commands[env_id, 2] = wp.randf(rng_state[env_id], ang_vel_z_range[0], ang_vel_z_range[1])
            rng_state[env_id] += wp.uint32(1)
        command_time_left[env_id] = resampling_time
