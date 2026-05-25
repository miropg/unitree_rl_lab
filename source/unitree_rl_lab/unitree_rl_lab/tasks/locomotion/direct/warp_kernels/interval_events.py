# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interval-mode event kernels — replacement for the ``mode=interval``
trigger logic in :class:`isaaclab.managers.EventManager` plus the
``mdp.push_by_setting_velocity`` event function.

Cross-task reusable for any env that pushes the root via interval-mode
random velocity perturbation.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def init_push_timer_kernel(
    mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    push_time_left: wp.array(dtype=wp.float32),
    push_interval_range_s: wp.vec2f,
):
    """Seed push_time_left for resetting envs (initial interval sample).

    cfg:     g1/29dof/velocity_env_cfg.py:175-180 (EventCfg.push_robot, mode=interval)
    manager: isaaclab/managers/event_manager.py:142-147 (interval-mode timer init)
    Without this, push_time_left stays 0 → push triggers on step 1 (Manager waits ~5s).

    Note: ``wp.randf`` here intentionally advances ``rng_state`` 1× per reset env,
    mirroring Manager ``EventManager.reset`` line 147 (``torch.rand(num_envs, ...)``)
    which advances the torch.Generator state by ``num_envs`` even when
    ``interval_range_s`` is degenerate (e.g., ``(5.0, 5.0)``). The pre-refactor
    baseline used a const path that omitted this advance, which caused a hidden
    baseline-vs-Manager RNG sequence skew (downstream reset / push / command
    resample saw a different RNG sequence than the corresponding Manager run).
    Refactored kernel mirrors Manager exactly. See
    ``agent_tmp/doc/refactor_residual_0.1pct_2026_05_07.md`` §3 for the
    cross-env tensorboard parity (4-backend max_diff 6.90% < 10%) and §4 for
    the parity_dump CP-level evidence (T1 graph_off bit-equal at 1e-9).
    """
    env_id = wp.tid()
    if mask[env_id]:
        new_interval = wp.randf(
            rng_state[env_id], push_interval_range_s[0], push_interval_range_s[1]
        )
        rng_state[env_id] += wp.uint32(1)
        push_time_left[env_id] = new_interval


@wp.kernel
def update_push_kernel(
    rng_state: wp.array(dtype=wp.uint32),
    push_time_left: wp.array(dtype=wp.float32),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    step_dt: wp.float32,
    push_interval_range_s: wp.vec2f,
    push_velocity_x_range: wp.vec2f,
    push_velocity_y_range: wp.vec2f,
    push_velocity_z_range: wp.vec2f,
    push_velocity_roll_range: wp.vec2f,
    push_velocity_pitch_range: wp.vec2f,
    push_velocity_yaw_range: wp.vec2f,
):
    """Mirror EventManager interval-trigger + mdp.push_by_setting_velocity.

    cfg:     g1/29dof/velocity_env_cfg.py:175-180 (EventCfg.push_robot, mode=interval)
    manager: isaaclab/managers/event_manager.py:205-229 (interval timer + dispatch)
           + isaaclab/envs/mdp/events.py:1652-1677     (push_by_setting_velocity)
    Critical: trigger epsilon ``< 1e-6``; resample interval after trigger; sample 6 axis
    deltas even if z/roll/pitch/yaw cfg = (0,0) so RNG count matches Manager; ``+=`` accumulate
    (NOT overwrite — would reset xyz every push).  Sample order matches Manager:
    interval first, then [x,y,z,roll,pitch,yaw].
    wp.spatial_vectorf layout (verified in isaaclab_newton/.../rigid_object_data.py:920+):
    - lane[0] = linear x
    - lane[1] = linear y
    - lane[2] = linear z
    - lane[3] = angular roll
    - lane[4] = angular pitch
    - lane[5] = angular yaw
    """
    env_id = wp.tid()
    push_time_left[env_id] -= step_dt
    if push_time_left[env_id] < 1e-6:
        state = rng_state[env_id]
        # 1. Sample new interval (Manager event_manager.py:222-226).
        new_interval = wp.randf(
            state, push_interval_range_s[0], push_interval_range_s[1]
        )
        state += wp.uint32(1)
        # 2. Sample 6 axis vel deltas (Manager events.py:993-997).
        #    Order matches the Manager dict key order
        #    ["x", "y", "z", "roll", "pitch", "yaw"].
        dvx = wp.randf(state, push_velocity_x_range[0], push_velocity_x_range[1])
        state += wp.uint32(1)
        dvy = wp.randf(state, push_velocity_y_range[0], push_velocity_y_range[1])
        state += wp.uint32(1)
        dvz = wp.randf(state, push_velocity_z_range[0], push_velocity_z_range[1])
        state += wp.uint32(1)
        droll = wp.randf(state, push_velocity_roll_range[0], push_velocity_roll_range[1])
        state += wp.uint32(1)
        dpitch = wp.randf(state, push_velocity_pitch_range[0], push_velocity_pitch_range[1])
        state += wp.uint32(1)
        dyaw = wp.randf(state, push_velocity_yaw_range[0], push_velocity_yaw_range[1])
        state += wp.uint32(1)
        rng_state[env_id] = state
        # 3. Accumulate (``vel_w += delta``, Manager events.py:997).
        cur = root_vel_w[env_id]
        root_vel_w[env_id] = wp.spatial_vectorf(
            cur[0] + dvx,    # linear x
            cur[1] + dvy,    # linear y
            cur[2] + dvz,    # linear z
            cur[3] + droll,  # angular roll
            cur[4] + dpitch, # angular pitch
            cur[5] + dyaw,   # angular yaw
        )
        # 4. Reset timer to the freshly sampled interval.
        push_time_left[env_id] = new_interval
