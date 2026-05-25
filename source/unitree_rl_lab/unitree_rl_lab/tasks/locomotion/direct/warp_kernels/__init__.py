# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable Warp kernels and per-term ``wp.func`` helpers for Direct-Warp envs.

This package collects the kernels that any robot's Direct-Warp env can
reuse without modification, organized by capability:

* :mod:`.rng`                 - Per-env Warp RNG state initialiser.
* :mod:`.actions`             - JointPositionAction process / reset kernels.
* :mod:`.buffers`             - Observation history (CircularBuffer + flatten).
* :mod:`.commands`            - UniformVelocityCommand sample / update kernels.
* :mod:`.reset_events`        - mdp.reset_root_state / reset_joints / apply_external_force kernels.
* :mod:`.interval_events`     - EventManager interval-mode push kernels.
* :mod:`.velocity_obs`        - Velocity-task observation frame kernels and per-term wp.func.
* :mod:`.velocity_rewards`    - Velocity-task reward wp.func library (G1 / Go2 / H1).
* :mod:`.velocity_terminations` - Velocity-task termination kernel.

Joint-aggregating kernels and wp.funcs accept ``num_joints`` and (where
relevant) ``joint_lanes`` as runtime ``wp.int32`` parameters so a single
kernel binary serves G1 (29 DoF), Go2 (12 DoF), H1 (19 DoF), etc.

Usage from a robot-specific env file::

    from .warp_kernels.rng import initialize_rng_state
    from .warp_kernels.actions import pre_physics_step_kernel, clear_reset_buffers_kernel
    from .warp_kernels.buffers import (
        append_or_fill_history_kernel, set_first_push_kernel,
        clear_first_push_kernel, flatten_history_kernel,
    )
    from .warp_kernels.commands import (
        sample_reset_commands_kernel, update_interval_commands_kernel,
    )
    from .warp_kernels.reset_events import (
        reset_root_kernel, reset_joints_kernel, apply_external_force_torque_kernel,
    )
    from .warp_kernels.interval_events import init_push_timer_kernel, update_push_kernel
    from .warp_kernels.velocity_obs import (
        compute_policy_base_frame_kernel, compute_policy_joint_frame_kernel,
        compute_policy_base_frame_noisy_kernel, compute_policy_joint_frame_noisy_kernel,
        compute_critic_base_frame_kernel, compute_critic_joint_frame_kernel,
    )
    from .warp_kernels.velocity_rewards import (
        _reward_track_lin_vel_xy, _reward_track_ang_vel_z, _reward_alive,
        _reward_lin_vel_z_l2, _reward_ang_vel_xy_l2, _reward_joint_vel_l2,
        _reward_joint_acc_l2, _reward_action_rate_l2, _reward_joint_pos_limits,
        _reward_energy, _reward_joint_deviation_l1, _reward_flat_orientation_l2,
        _reward_base_height_l2, _reward_feet_gait, _reward_feet_slide,
        _reward_foot_clearance, _reward_undesired_contacts,
    )
    from .warp_kernels.velocity_terminations import get_dones_kernel

The robot-specific file then writes its own fusion kernel that dispatches
the per-term wp.funcs in ``RewardsCfg`` declaration order (see
``velocity_warp_env_g1_kernels.py`` for the G1 reference).
"""

from __future__ import annotations
