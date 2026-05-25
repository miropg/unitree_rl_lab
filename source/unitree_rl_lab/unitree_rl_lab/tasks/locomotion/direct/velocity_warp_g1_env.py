# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRLEnvWarp implementation for Unitree G1 velocity tracking.

This is the Warp counterpart of :mod:`velocity_base_env` (Direct env)
and :mod:`unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg`
(Manager-based env).  Currently G1-only; the Direct Newton task remains
the canonical parity reference for Warp output.

Cross-robot module layout
=========================

The reusable Warp kernels and per-term ``wp.func`` library now live in
:mod:`.warp_kernels`; the G1-specific reward fusion kernels live in
:mod:`.velocity_warp_env_g1_kernels`.  Adding Go2 / H1:

* Reuse :mod:`.warp_kernels` as-is — joint-aggregating kernels accept
  ``num_joints`` (and where relevant ``joint_lanes``) as runtime
  ``wp.int32`` parameters so a single kernel binary serves any robot.
* Copy :mod:`.velocity_warp_env_g1_kernels` to
  ``velocity_warp_env_<robot>_kernels.py`` and reorder the dispatch to
  match the new robot's :class:`RewardsCfg` declaration.
* Copy :class:`G1VelocityWarpEnv` to a sibling class
  ``Go2VelocityWarpEnv`` / ``H1VelocityWarpEnv`` (NOT a subclass — see
  the class docstring for why), swap the joint-group regex / foot-body
  regex / fusion kernel imports, and adjust the module-level dim
  constants below.

Why is everything fused into Warp kernels?
==========================================

Manager-based env composes obs / reward / events as Python dispatch
chains; on G1 with 4096 envs at 50 Hz this becomes the dominant CPU
overhead and limits throughput well before GPU is saturated.  The fused
kernels:

* run inside a CUDA Graph (captured by parent class
  :class:`DirectRLEnvWarp` once and replayed every step);
* read directly from Newton's ``wp.array`` storage with no
  ``wp.to_torch`` conversion;
* draw RNG samples on-device, removing ``torch.rand_like`` host calls.

Result: per-step CPU overhead drops from ~3 ms (Manager) to < 100 µs
(Warp). See ``agent_tmp/feedback/g1_4backend_review_*`` and
``agent_tmp/doc/refactor_residual_0.1pct_2026_05_07.md`` for the
parity numbers.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp
from isaaclab_experimental.envs import DirectRLEnvWarp
from isaaclab_experimental.envs.direct_rl_env_warp import zero_mask_int32

from .velocity_warp_env_g1_kernels import (
    add_contact_rewards_kernel,
    compute_rewards_kernel,
)
from .warp_kernels.actions import (
    clear_reset_buffers_kernel,
    pre_physics_step_kernel,
)
from .warp_kernels.buffers import (
    append_or_fill_history_kernel,
    clear_first_push_kernel,
    flatten_history_kernel,
    set_first_push_kernel,
)
from .warp_kernels.commands import (
    sample_reset_commands_kernel,
    update_interval_commands_kernel,
)
from .warp_kernels.interval_events import (
    init_push_timer_kernel,
    update_push_kernel,
)
from .warp_kernels.reset_events import (
    apply_external_force_torque_kernel,
    reset_joints_kernel,
    reset_root_kernel,
)
from .warp_kernels.rng import initialize_rng_state
from .warp_kernels.velocity_obs import (
    compute_critic_base_frame_kernel,
    compute_critic_joint_frame_kernel,
    compute_policy_base_frame_kernel,
    compute_policy_base_frame_noisy_kernel,
    compute_policy_joint_frame_kernel,
    compute_policy_joint_frame_noisy_kernel,
)
from .warp_kernels.velocity_terminations import get_dones_kernel


G1_NUM_JOINTS = 29
G1_POLICY_FRAME_DIM = 9 + 3 * G1_NUM_JOINTS  # base 9 + (joint_pos + joint_vel + last_action) * N

# Per-term dim within a single frame, declared in Manager
# :class:`ObservationsCfg.PolicyCfg` field order:
# base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) +
# joint_pos_rel(N) + joint_vel_rel(N) + last_action(N).  Sums to
# ``G1_POLICY_FRAME_DIM``.  Used by :func:`_build_term_stacked_layout`
# to produce the per-obs lookup arrays that flatten the Warp history
# buffer into Manager's term-stacked obs layout.
G1_POLICY_TERM_DIMS = (3, 3, 3, G1_NUM_JOINTS, G1_NUM_JOINTS, G1_NUM_JOINTS)
G1_CRITIC_TERM_DIMS = (3, 3, 3, 3, G1_NUM_JOINTS, G1_NUM_JOINTS, G1_NUM_JOINTS)
"""Critic adds ``base_lin_vel`` (3) at the front; otherwise same as policy."""

G1_CRITIC_FRAME_DIM = 12 + 3 * G1_NUM_JOINTS  # critic base 12 (extra base_lin_vel) + 3N joint
"""Single-frame critic observation dim. Critic obs has no noise injection;
critic frame kernels do not take noise rng / range parameters, so an
accidental "noise everywhere" change cannot contaminate the critic path."""

G1_HISTORY_LENGTH = 5
"""Default policy obs history length, mirrors Manager
``ObservationsCfg.PolicyCfg.history_length=5``. Cfg-overridable via
:attr:`G1VelocityWarpEnvCfg.policy_history_length`; history kernels take
the actual length as a runtime ``wp.int32`` so cfg-driven overrides work
without recompiling the kernels."""

G1_POLICY_OBS_DIM = G1_POLICY_FRAME_DIM * G1_HISTORY_LENGTH
"""Default policy obs dim. If a cfg overrides ``policy_history_length``,
also override ``observation_space`` to match."""

G1_JOINT_LANES = 32
"""Logical lane count for joint-feature kernels. Intentionally larger than
``G1_NUM_JOINTS`` (29) so each env maps to a warp-friendly 32-lane group;
joint-frame kernels use a stride loop so they remain correct if a future
robot has more than 32 joints."""

# Order MUST match the Manager ``RewardsCfg`` dataclass field declaration order
# in ``robots/g1/29dof/velocity_env_cfg.py``. Both ``RewardManager.compute`` and
# the fused Warp kernel accumulate ``total = sum(term_i * weight_i * dt)`` in
# this order; floating-point addition is non-commutative so reordering would
# introduce ~1e-6 differences and make parity scripts noisy.
REWARD_TERM_NAMES = (
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "alive",
    "base_linear_velocity",
    "base_angular_velocity",
    "joint_vel",
    "joint_acc",
    "action_rate",
    "dof_pos_limits",
    "energy",
    "joint_deviation_arms",
    "joint_deviation_waists",
    "joint_deviation_legs",
    "flat_orientation_l2",
    "base_height",
    "gait",
    "feet_slide",
    "feet_clearance",
    "undesired_contacts",
)
NUM_REWARD_TERMS = len(REWARD_TERM_NAMES)
"""Reward term naming for fusion kernels. These indices index
:attr:`G1VelocityWarpEnv._reward_terms_step` (the per-step kernel-side
per-term buffer) and the host-side per-episode ``_reward_terms_sum``
accumulator emitted to ``Episode_Reward/<name>`` on reset. The order MUST
match :class:`RewardsCfg` declaration order so the fused FP ``+=``
accumulation is bit-stable across runs."""


class G1VelocityWarpEnv(DirectRLEnvWarp):
    """G1 29-DoF velocity tracker (Warp-native, Newton-only).

    **G1-specific implementation.**  ~90% of this class is hardcoded for
    G1 (``G1_NUM_JOINTS=29``, biped gait offsets ``(0.0, 0.5)``,
    ``ankle_roll`` foot regex, arm / waist / leg group regex, fusion
    kernel with 15+4 reward term order matching G1 :class:`RewardsCfg`).

    Future Go2 / H1 envs should be **independent sibling classes**
    (e.g. ``Go2VelocityWarpEnv``, ``H1VelocityWarpEnv``), copied from
    this file as a template and customised for the new robot's DoF
    count, joint groups, foot regex, fusion kernel, and gait pattern.
    **Do NOT subclass this class** — there is no shared init / step
    scaffolding worth inheriting (joint-group regex is per-robot,
    fusion-kernel reward-term order is :class:`RewardsCfg`-bound).

    Cross-robot reuse goes through:

    * :mod:`.warp_kernels` — kernel-level reusable ``wp.func`` /
      ``wp.kernel``, accept ``num_joints`` (and where applicable
      ``joint_lanes``) as runtime ``wp.int32`` parameters.
    * :func:`unitree_rl_lab.tasks.locomotion.mdp.events
      .newton_randomize_rigid_body_material` — robot-agnostic startup
      friction event impl, called by both Manager / Direct (via
      ``EventManager``) and Direct-Warp (via
      :meth:`_randomize_friction_at_startup`).
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Task-graph capture always on (parent class). No cfg toggle — see
        # file-level docstring for the host-vs-graph rules that prevent 906.

        self.robot = self.scene["robot"]
        self.contact_sensor = self.scene.sensors.get("contact_forces", None)

        # Newton simulation data views (zero-copy wp.array, kernels read these).
        # root_pose_w / root_quat_w / root_lin_vel_w are non-TimestampedBuffer
        # — re-fetched in _step_warp_end_pre on every step. Others are
        # TimestampedBuffer-backed (refresh on property access).
        self.root_pose_w = self.robot.data.root_pose_w           # wp.transformf (translation+quat)
        self.root_quat_w = self.robot.data.root_quat_w           # wp.quatf (qx,qy,qz,qw) for yaw-frame projection
        self.root_lin_vel_w = self.robot.data.root_lin_vel_w     # wp.vec3f, world-frame for track_lin_vel_xy_yaw_frame
        self.root_vel_w = self.robot.data.root_vel_w             # wp.spatial_vectorf (ang xyz, lin xyz)
        self.root_lin_vel_b = self.robot.data.root_lin_vel_b     # wp.vec3f, base frame
        self.root_ang_vel_b = self.robot.data.root_ang_vel_b     # wp.vec3f, base frame
        self.projected_gravity_b = self.robot.data.projected_gravity_b  # wp.vec3f
        self.joint_pos = self.robot.data.joint_pos               # (num_envs, 29) wp.float32
        self.joint_vel = self.robot.data.joint_vel
        self.joint_acc = self.robot.data.joint_acc
        self.applied_torque = self.robot.data.applied_torque
        self.default_joint_pos = self.robot.data.default_joint_pos
        # Note: G1 default_joint_vel = 0 for all joints, so cfg
        # reset_joint_velocity_range is effectively no-op (matches Manager).
        self.default_joint_vel = self.robot.data.default_joint_vel
        self.soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits  # wp.vec2f [lower, upper]
        self.body_pos_w = self.robot.data.body_pos_w             # (num_envs, num_bodies) wp.vec3f
        self.body_lin_vel_w = self.robot.data.body_lin_vel_w
        self.env_origins = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)

        # Action buffers
        self.raw_actions = wp.zeros((self.num_envs, G1_NUM_JOINTS), dtype=wp.float32, device=self.device)
        self.prev_actions = wp.zeros_like(self.raw_actions)
        self.joint_pos_target = wp.zeros_like(self.raw_actions)
        # cfg-driven history length (runtime, no kernel recompile on override)
        self.policy_history_length = int(getattr(self.cfg, "policy_history_length", G1_HISTORY_LENGTH))
        if self.policy_history_length < 1:
            raise ValueError(
                f"cfg.policy_history_length must be >= 1, got {self.policy_history_length}"
            )
        if self.policy_history_length * G1_POLICY_FRAME_DIM != self.cfg.observation_space:
            raise ValueError(
                f"cfg.observation_space ({self.cfg.observation_space}) must equal "
                f"policy_history_length ({self.policy_history_length}) * frame_dim "
                f"({G1_POLICY_FRAME_DIM}) = {self.policy_history_length * G1_POLICY_FRAME_DIM}."
            )

        # Policy obs buffers: frame (single step) -> history (5 frames) -> observations (480-dim flattened)
        self.policy_frame = wp.zeros((self.num_envs, G1_POLICY_FRAME_DIM), dtype=wp.float32, device=self.device)
        self.history = wp.zeros(
            (self.num_envs, self.policy_history_length, G1_POLICY_FRAME_DIM),
            dtype=wp.float32,
            device=self.device,
        )
        # first_push_mask: set by set_first_push_kernel in _reset_idx; consumed by
        # next append_or_fill_history_kernel to broadcast first frame to all H slots
        # (mirrors CircularBuffer._num_pushes == 0).
        self.first_push_mask = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self.observations = wp.zeros(
            (self.num_envs, self.cfg.observation_space), dtype=wp.float32, device=self.device
        )

        # Critic obs (privileged, never deployed). Same H, frame_dim = 99
        # (policy 96 + base_lin_vel 3 prepended).
        if self.cfg.state_space != G1_CRITIC_FRAME_DIM * self.policy_history_length:
            raise ValueError(
                f"cfg.state_space ({self.cfg.state_space}) must equal "
                f"G1_CRITIC_FRAME_DIM ({G1_CRITIC_FRAME_DIM}) * "
                f"policy_history_length ({self.policy_history_length}) = "
                f"{G1_CRITIC_FRAME_DIM * self.policy_history_length}."
            )
        self.critic_frame = wp.zeros(
            (self.num_envs, G1_CRITIC_FRAME_DIM), dtype=wp.float32, device=self.device
        )
        self.critic_history = wp.zeros(
            (self.num_envs, self.policy_history_length, G1_CRITIC_FRAME_DIM),
            dtype=wp.float32,
            device=self.device,
        )
        # critic_first_push_mask MUST be marked together with first_push_mask
        # (asymmetric marking shifts value baseline first H steps).
        self.critic_first_push_mask = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self.critic_observations = wp.zeros(
            (self.num_envs, self.cfg.state_space), dtype=wp.float32, device=self.device
        )
        # Per-env total reward (per-term contributions in _reward_terms_step).
        self.rewards = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        # Two RNG streams: rng_state for reset/command/push, obs_noise_rng_state
        # for obs noise. Separate so noise's ~64 samples/step don't shift the
        # reset/push RNG sequence (mirrors Manager NoiseModel torch.Generator isolation).
        self.rng_state = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        self.obs_noise_rng_state = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)

        # Command system buffers
        self.commands = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)  # [vx, vy, ωz]
        self.command_time_left = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.push_time_left = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        # Curriculum / log per-step buffers (kernel-written, host-read).
        # Slot layout of _reward_terms_step (matches RewardsCfg declaration order):
        #   [0] track_lin_vel_xy    [5] joint_vel        [10] joint_dev_arms     [15] gait
        #   [1] track_ang_vel_z     [6] joint_acc        [11] joint_dev_waists   [16] feet_slide
        #   [2] alive               [7] action_rate      [12] joint_dev_legs     [17] feet_clearance
        #   [3] base_linear_vel     [8] dof_pos_limits   [13] flat_orientation   [18] undesired_contacts
        #   [4] base_angular_vel    [9] energy           [14] base_height
        # Lanes [0:15] from compute_rewards_kernel, [15:19] from add_contact_rewards_kernel.
        self._track_lin_vel_xy_step = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._track_ang_vel_z_step = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._reward_terms_step = wp.zeros((self.num_envs, NUM_REWARD_TERMS), dtype=wp.float32, device=self.device)

        # Host-side episode sums (torch tensors). _reward_terms_sum drives TB
        # log emit; _track_*_sum drives curriculum threshold check. Both reset
        # for an env at every reset (mirrors RewardManager.reset zeroing).
        self._reward_terms_sum = torch.zeros(
            self.num_envs, NUM_REWARD_TERMS, device=self.device, dtype=torch.float32
        )
        self._track_lin_vel_xy_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._track_ang_vel_z_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Curriculum-mutable command ranges: must be wp.array (NOT wp.vec2f) so
        # captured graph dereferences pointer at replay time, picking up host
        # mutations between replays. Init values copy cfg.lin_vel_*_range;
        # _curriculum_step widens them toward cfg.lin_vel_*_limit_range.
        self._lin_vel_x_range_wp = wp.from_torch(
            torch.tensor(list(self.cfg.lin_vel_x_range), dtype=torch.float32, device=self.device),
            dtype=wp.float32,
        )
        self._lin_vel_y_range_wp = wp.from_torch(
            torch.tensor(list(self.cfg.lin_vel_y_range), dtype=torch.float32, device=self.device),
            dtype=wp.float32,
        )
        self._ang_vel_z_range_wp = wp.from_torch(
            torch.tensor(list(self.cfg.ang_vel_z_range), dtype=torch.float32, device=self.device),
            dtype=wp.float32,
        )
        # Host torch views of the same wp.array memory for curriculum mutation.
        self._lin_vel_x_range_t = wp.to_torch(self._lin_vel_x_range_wp)
        self._lin_vel_y_range_t = wp.to_torch(self._lin_vel_y_range_wp)
        self._ang_vel_z_range_t = wp.to_torch(self._ang_vel_z_range_wp)

        # RNG init: separate seeds (seed / seed+1) so obs noise and reset/push
        # don't share entropy (would halve effective entropy).
        seed = -1 if self.cfg.seed is None else self.cfg.seed
        wp.launch(initialize_rng_state, dim=self.num_envs, inputs=[self.rng_state, seed], device=self.device)
        obs_noise_seed = (-1 if seed == -1 else seed + 1)
        wp.launch(
            initialize_rng_state,
            dim=self.num_envs,
            inputs=[self.obs_noise_rng_state, obs_noise_seed],
            device=self.device,
        )

        # Resolve joint groups on host once, push to device int arrays.
        arm_joint_ids = self.robot.find_joints([".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"])[0]
        waist_joint_ids = self.robot.find_joints(["waist.*"])[0]
        leg_joint_ids = self.robot.find_joints([".*_hip_roll_joint", ".*_hip_yaw_joint"])[0]
        self.num_arm_joints = len(arm_joint_ids)
        self.num_waist_joints = len(waist_joint_ids)
        self.num_leg_joints = len(leg_joint_ids)
        self.arm_joint_ids = self._make_int_array(arm_joint_ids)
        self.waist_joint_ids = self._make_int_array(waist_joint_ids)
        self.leg_joint_ids = self._make_int_array(leg_joint_ids)

        # Contact body groups: feet (ankle_roll) for gait/slide/clearance,
        # undesired (everything except ankle*) for undesired_contacts. Manager
        # regex "(?!.*ankle.*).*" excludes BOTH ankle_pitch and ankle_roll —
        # using only ankle_roll here would bias undesired_contacts.
        foot_ids, _ = self.robot.find_bodies(".*ankle_roll.*")
        ankle_ids, _ = self.robot.find_bodies(".*ankle.*")
        ankle_id_set = set(int(i) for i in ankle_ids)
        all_body_ids, _ = self.robot.find_bodies(".*")
        undesired_ids = [int(i) for i in all_body_ids if int(i) not in ankle_id_set]
        self.num_feet = len(foot_ids)
        self.num_undesired_bodies = len(undesired_ids)
        self.foot_body_ids = self._make_int_array(foot_ids)
        self.undesired_body_ids = self._make_int_array(undesired_ids)

        # Push event: per-env interval mode only (see update_push_kernel docstring).
        # is_global_time mode is not implemented (would need host-side timer check
        # or single-thread atomic kernel — neither worth doing for G1/H1/Go2).
        if self.cfg.push_is_global_time:
            raise NotImplementedError(
                "cfg.push_is_global_time=True is not implemented; "
                "G1/H1/Go2 cfgs use is_global_time=False (per-env timers)."
            )

        # External wrench: gated by cfg.enable_external_force_torque (default False
        # leaves Newton's has_external_wrench=False, no per-sub-step cost).
        ext_body_ids, _ = self.robot.find_bodies(self.cfg.external_force_body_name)
        if len(ext_body_ids) == 0:
            raise ValueError(
                f"cfg.external_force_body_name='{self.cfg.external_force_body_name}' "
                f"did not match any body on the articulation. Available bodies: "
                f"{list(self.robot.body_names)}"
            )
        self._external_force_body_id: int = int(ext_body_ids[0])

        if self.cfg.enable_external_force_torque:
            zeros = torch.zeros(
                (self.num_envs, 1, 3), device=self.device, dtype=torch.float32
            )
            self.robot.set_external_force_and_torque(
                forces=zeros,
                torques=zeros,
                body_ids=[self._external_force_body_id],
            )
            self.robot.has_external_wrench = True
            # Zero-copy wp.array views into articulation's torch wrench buffers
            # (written by apply_external_force_torque_kernel each reset).
            self._external_force_b_wp = wp.from_torch(
                self.robot._external_force_b, dtype=wp.float32
            )
            self._external_torque_b_wp = wp.from_torch(
                self.robot._external_torque_b, dtype=wp.float32
            )
        # Per-foot phase offset for feet_gait (cfg [0, 0.5] for biped trot).
        if len(self.cfg.feet_gait_offsets) != self.num_feet:
            raise ValueError(
                f"cfg.feet_gait_offsets has length {len(self.cfg.feet_gait_offsets)} "
                f"but num_feet={self.num_feet}.  Both must agree."
            )
        self.foot_phase_offsets = wp.array(
            [float(o) for o in self.cfg.feet_gait_offsets], dtype=wp.float32, device=self.device
        )

        # Cache contact sensor history length (kernel uses it as runtime int).
        if self.contact_sensor is not None:
            self.contact_history_length = int(self.contact_sensor.cfg.history_length)
        else:
            self.contact_history_length = 0

        # Term-stacked flatten lookup (matches Manager torch.cat([reshape, ...]) layout).
        self.policy_obs_to_slot, self.policy_obs_to_feat = self._build_term_stacked_layout(
            G1_POLICY_TERM_DIMS
        )
        self.critic_obs_to_slot, self.critic_obs_to_feat = self._build_term_stacked_layout(
            G1_CRITIC_TERM_DIMS
        )

        # Zero-copy torch views of wp.array buffers (returned through Gym/RSL-RL).
        self.torch_obs_buf = wp.to_torch(self.observations)
        self.torch_critic_obs_buf = wp.to_torch(self.critic_observations)
        self.torch_reward_buf = wp.to_torch(self.rewards)
        self.torch_reset_terminated = wp.to_torch(self.reset_terminated)
        self.torch_reset_time_outs = wp.to_torch(self.reset_time_outs)
        self.torch_episode_length_buf = self.episode_length_buf

        # Per-shape friction / restitution randomization, mirroring the
        # Manager ``physics_material`` startup event.  Invoked once before
        # the first ``_reset_idx`` (DirectRLEnvWarp has no EventManager).
        if self.cfg.enable_friction_randomization:
            self._randomize_friction_at_startup()

        self._reset_idx(self._ALL_ENV_MASK)

    def _randomize_friction_at_startup(self) -> None:
        """Mirror Manager / Direct ``physics_material`` startup event.

        Delegates to :func:`mdp.newton_randomize_rigid_body_material` so the
        randomization logic (μ sampling, ground-solmix-zero PhysX-multiply
        emulation, ``add_model_change(SHAPE_PROPERTIES)`` propagation) lives
        in **one** place — robot-agnostic, future Go2 / H1 sibling classes
        get the same behaviour for free.
        """
        from isaaclab.managers import SceneEntityCfg

        from unitree_rl_lab.tasks.locomotion.mdp.events import (
            newton_randomize_rigid_body_material,
        )

        newton_randomize_rigid_body_material(
            env=self,
            env_ids=None,  # all envs
            static_friction_range=self.cfg.friction_range,
            restitution_range=self.cfg.restitution_range,
            asset_cfg=SceneEntityCfg("robot", body_names=".*"),
        )

    def _make_int_array(self, values) -> wp.array:
        return wp.array([int(v) for v in values], dtype=wp.int32, device=self.device)

    def _build_term_stacked_layout(
        self, term_dims: tuple[int, ...]
    ) -> tuple["wp.array", "wp.array"]:
        """Build ``(obs_to_slot, obs_to_feat)`` lookup for term-stacked flatten.

        Maps obs_id → (history slot, feat-within-frame) so the final layout
        matches Manager's ``torch.cat([term_history.reshape(N, -1) for term])``.
        Output length = history_length × sum(term_dims).
        """
        H = self.policy_history_length
        obs_to_slot: list[int] = []
        obs_to_feat: list[int] = []
        feat_offset = 0
        for term_dim in term_dims:
            for slot in range(H):
                for d in range(term_dim):
                    obs_to_slot.append(slot)
                    obs_to_feat.append(feat_offset + d)
            feat_offset += term_dim
        expected_total = H * sum(term_dims)
        if len(obs_to_slot) != expected_total:
            raise RuntimeError(
                f"layout build size mismatch: got {len(obs_to_slot)} cells, expected {expected_total}"
            )
        return (
            wp.array(obs_to_slot, dtype=wp.int32, device=self.device),
            wp.array(obs_to_feat, dtype=wp.int32, device=self.device),
        )

    def _setup_scene(self) -> None:
        # The cfg scene already declares the robot, terrain, sensors, and lights.
        pass

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset all envs and fill obs history (broadcasts first frame to all H slots).

        Override parent reset (which uses _get_observations without update_history)
        because G1 needs Manager-equivalent first-push broadcast: after _reset_idx
        sets first_push_mask=True for every env, _compute_observations(True) below
        triggers the broadcast branch in append_or_fill_history_kernel.
        """
        if seed is not None:
            self.seed(seed)

        self._reset_idx(self._ALL_ENV_MASK)
        self.scene.write_data_to_sim()
        # sim.forward() propagates reset kinematics into TimestampedBuffer-backed
        # views before first _compute_observations reads them (mirror Direct env).
        self.sim.forward()

        if hasattr(self.sim, "has_rtx_sensors") and self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        self._compute_observations(update_history=True)
        return {
            "policy": self.torch_obs_buf.clone(),
            "critic": self.torch_critic_obs_buf.clone(),
        }, self.extras

    def step(self, action: torch.Tensor):
        """Run one env step. Returns obs dict with both ``policy`` and ``critic`` keys.

        Parent ``DirectRLEnvWarp.step`` (direct_rl_env_warp.py:463-469) only
        returns ``{"policy": ...}``; we add ``"critic"`` after super returns.
        Then run host-side post-processing (curriculum + log emit) OUTSIDE the
        captured graph (see file-level docstring "host vs graph" rule).
        """
        obs, rewards, terminated, time_outs, extras = super().step(action)
        obs["critic"] = self.torch_critic_obs_buf.clone()
        # Fresh dict every step so RSL-RL's 24-step ep_extras gets distinct
        # references (alias would collapse iter-end TB mean to last step).
        extras["log"] = {}
        self._curriculum_step()
        self._emit_termination_log(extras)
        self._emit_per_term_reward_log(extras)
        return obs, rewards, terminated, time_outs, extras

    def _emit_per_term_reward_log(self, extras: dict) -> None:
        """Emit ``Episode_Reward/<term>`` to extras["log"] on reset steps.

        Host-side (outside captured graph). Pipeline:
          1. Accumulate _reward_terms_step (kernel-written) into _reward_terms_sum.
          2. On reset steps: emit (resetting envs' sum) / max_episode_length_s
             to log_dict, then zero the sum for those envs.
          3. Non-reset steps: return early (no emit, mirrors Direct env behaviour).
        """
        self._reward_terms_sum += wp.to_torch(self._reward_terms_step)

        reset_buf = wp.to_torch(self.reset_buf).bool()
        n_reset = int(reset_buf.sum().item())
        if n_reset == 0:
            return

        log_dict = extras["log"]
        resetting_sums = self._reward_terms_sum[reset_buf]  # (n_reset, NUM_REWARD_TERMS)
        per_term_means = (resetting_sums.mean(dim=0) / self.max_episode_length_s).cpu().numpy()
        for i, name in enumerate(REWARD_TERM_NAMES):
            log_dict[f"Episode_Reward/{name}"] = float(per_term_means[i])
        self._reward_terms_sum[reset_buf] = 0.0

    def _emit_termination_log(self, extras: dict) -> None:
        """Emit ``Episode_Termination/{bad_orientation,time_out}`` ratios to log on reset steps.

        Host-side. Returns early on non-reset steps (mirrors Manager
        TerminationManager.compute which only logs on reset).
        Note: bad_orientation OR base_height collapse into ``reset_terminated``
        (single bit), so both surface as Episode_Termination/bad_orientation.
        """
        reset_buf = wp.to_torch(self.reset_buf).bool()
        n_reset = int(reset_buf.sum().item())
        if n_reset == 0:
            return

        term_buf = wp.to_torch(self.reset_terminated).bool()
        timeout_buf = wp.to_torch(self.reset_time_outs).bool()
        orientation_ratio = float((reset_buf & term_buf).sum().item() / n_reset)
        timeout_ratio = float((reset_buf & timeout_buf).sum().item() / n_reset)

        log_dict = extras["log"]
        # Use the same key as Manager (``bad_orientation`` covers both
        # bad_orientation and base_height on warp because get_dones_kernel
        # combines them into ``reset_terminated``; per-term split would
        # require splitting the kernel — not worth the graph-capture
        # overhead for a diagnostic metric).
        log_dict["Episode_Termination/bad_orientation"] = orientation_ratio
        log_dict["Episode_Termination/time_out"] = timeout_ratio

    def _curriculum_step(self) -> None:
        """Host-side curriculum: extend cmd ranges when tracking reward exceeds threshold.

        Manager equivalent: mdp.lin_vel_cmd_levels / mdp.ang_vel_cmd_levels.
        Runs OUTSIDE captured graph (host torch ops).

        Pipeline:
          1. Accumulate per-step _track_*_step into host _track_*_sum.
          2. On reset steps: zero the resetting envs' sums (mirrors
             RewardManager.reset which zeroes regardless of curriculum trigger;
             prevents partial-episode bias from mid-episode crashes).
          3. Every max_episode_length step: if avg_reward > rew_track_* × threshold_factor,
             extend range ±delta clamped to limit_range.

        Range mutation: host torch view (_lin_vel_x_range_t) shares memory with
        wp.array consumed by kernels; graph replay reads the latest value via pointer.
        """
        if not self.cfg.enable_command_curriculum:
            return
        # Per-step accumulate: tensor view zero-copies the wp.array.
        self._track_lin_vel_xy_sum += wp.to_torch(self._track_lin_vel_xy_step)
        self._track_ang_vel_z_sum += wp.to_torch(self._track_ang_vel_z_step)

        # Zero per-env sums for every resetting env on every step,
        # INDEPENDENTLY of the curriculum trigger.  Must read sums for
        # resetting envs BEFORE zeroing so the curriculum trigger branch
        # below can still use the just-completed episode's tracking sum.
        reset_buf_torch = wp.to_torch(self.reset_buf).bool()
        n_reset = int(reset_buf_torch.sum().item())

        # Snapshot sums for resetting envs before zeroing (only needed for
        # the curriculum trigger; can skip otherwise).
        trigger_curriculum = (
            self.common_step_counter % self.max_episode_length == 0 and n_reset > 0
        )
        if trigger_curriculum:
            # Average over resetting envs / max_episode_length_s, exactly
            # like ``mdp.lin_vel_cmd_levels:21``.
            avg_lin = (
                self._track_lin_vel_xy_sum[reset_buf_torch].mean().item()
                / self.max_episode_length_s
            )
            avg_ang = (
                self._track_ang_vel_z_sum[reset_buf_torch].mean().item()
                / self.max_episode_length_s
            )

        # Zero EVERY resetting env's sum (manager-equivalent semantics).
        if n_reset > 0:
            self._track_lin_vel_xy_sum[reset_buf_torch] = 0.0
            self._track_ang_vel_z_sum[reset_buf_torch] = 0.0

        if not trigger_curriculum:
            return

        delta = self.cfg.curriculum_delta
        # lin_vel curriculum (extends both lin_vel_x and lin_vel_y; mirrors
        # ``mdp.lin_vel_cmd_levels:24-35`` which extends both axes when the
        # x-tracking reward triggers).
        threshold_lin = self.cfg.rew_track_lin_vel_xy * self.cfg.curriculum_track_lin_vel_xy_threshold_factor
        if avg_lin > threshold_lin:
            self._extend_range_t_(self._lin_vel_x_range_t, delta, self.cfg.lin_vel_x_limit_range)
            self._extend_range_t_(self._lin_vel_y_range_t, delta, self.cfg.lin_vel_y_limit_range)

        # ang_vel curriculum (mirror ``mdp.ang_vel_cmd_levels:53-59``).
        threshold_ang = self.cfg.rew_track_ang_vel_z * self.cfg.curriculum_track_ang_vel_z_threshold_factor
        if avg_ang > threshold_ang:
            self._extend_range_t_(self._ang_vel_z_range_t, delta, self.cfg.ang_vel_z_limit_range)


    def _extend_range_t_(
        self,
        range_t: torch.Tensor,
        delta: float,
        limit_range: tuple[float, float],
    ) -> None:
        """In-place extend (lo, hi) by ±delta, clamped to limit_range.

        range_t is a host view aliasing wp.array consumed by command kernels;
        in-place mutation propagates to next graph replay automatically.
        """
        cur_lo = float(range_t[0].item())
        cur_hi = float(range_t[1].item())
        range_t[0] = max(limit_range[0], cur_lo - delta)
        range_t[1] = min(limit_range[1], cur_hi + delta)

    def get_observations(self) -> dict[str, torch.Tensor]:
        """Public read-only obs accessor.  Returns CPU-safe clones.

        Returns both ``policy`` (``observation_space`` dim, history-stacked)
        and ``critic`` (``state_space`` dim, history-stacked, includes
        ``base_lin_vel`` privileged signal).
        """
        obs = self._get_observations()
        # Public reads can happen immediately before the first graph capture.
        # Keep those eager kernels from leaving a stream dependency before
        # returning a cloned tensor to the caller.
        wp.synchronize()
        return {"policy": obs["policy"].clone(), "critic": obs["critic"].clone()}

    def _get_observations(self) -> dict:
        """Read-only obs path: recompute frame + flatten history, do NOT push history.

        Signature matches DirectRLEnvWarp._get_observations (no update_history arg).
        History advance happens only in :meth:`reset` and :meth:`_step_warp_end_post`
        (both call _compute_observations(update_history=True)).
        """
        self._compute_observations(update_history=False)
        return {"policy": self.torch_obs_buf, "critic": self.torch_critic_obs_buf}

    def _pre_physics_step(self, actions: wp.array) -> None:
        wp.launch(
            pre_physics_step_kernel,
            dim=(self.num_envs, G1_NUM_JOINTS),
            inputs=[
                actions,
                self.default_joint_pos,
                self.cfg.action_scale,
                self.raw_actions,
                self.prev_actions,
                self.joint_pos_target,
            ],
            device=self.device,
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_mask(target=self.joint_pos_target)

    def _compute_observations(self, update_history: bool) -> None:
        # Trigger lazy-property refresh (TimestampedBuffer) before kernel launch.
        # Cached __init__ aliases would otherwise feed init-time data forever.
        _ = self.robot.data.root_lin_vel_b
        _ = self.robot.data.root_ang_vel_b
        _ = self.robot.data.projected_gravity_b

        # Policy frame: noisy variant single-thread per env (RNG race), non-noisy
        # multi-lane parallel.
        if self.cfg.enable_observation_noise:
            wp.launch(
                compute_policy_base_frame_noisy_kernel,
                dim=self.num_envs,
                inputs=[
                    self.root_ang_vel_b,
                    self.projected_gravity_b,
                    self.commands,
                    self.cfg.obs_base_ang_vel_scale,
                    self.obs_noise_rng_state,
                    self.cfg.obs_noise_base_ang_vel_range,
                    self.cfg.obs_noise_projected_gravity_range,
                    self.policy_frame,
                ],
                device=self.device,
            )
            wp.launch(
                compute_policy_joint_frame_noisy_kernel,
                dim=self.num_envs,
                inputs=[
                    self.joint_pos,
                    self.joint_vel,
                    self.default_joint_pos,
                    self.raw_actions,
                    self.cfg.obs_joint_vel_scale,
                    G1_NUM_JOINTS,
                    self.obs_noise_rng_state,
                    self.cfg.obs_noise_joint_pos_range,
                    self.cfg.obs_noise_joint_vel_range,
                    self.policy_frame,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                compute_policy_base_frame_kernel,
                dim=(self.num_envs, 4),
                inputs=[
                    self.root_ang_vel_b,
                    self.projected_gravity_b,
                    self.commands,
                    self.cfg.obs_base_ang_vel_scale,
                    self.policy_frame,
                ],
                device=self.device,
            )
            wp.launch(
                compute_policy_joint_frame_kernel,
                dim=(self.num_envs, G1_JOINT_LANES),
                inputs=[
                    self.joint_pos,
                    self.joint_vel,
                    self.default_joint_pos,
                    self.raw_actions,
                    self.cfg.obs_joint_vel_scale,
                    G1_NUM_JOINTS,
                    G1_JOINT_LANES,
                    self.policy_frame,
                ],
                device=self.device,
            )
        # ----- Critic frame (privileged: extra base_lin_vel; never noisy) -----
        wp.launch(
            compute_critic_base_frame_kernel,
            dim=(self.num_envs, 4),
            inputs=[
                self.root_lin_vel_b,
                self.root_ang_vel_b,
                self.projected_gravity_b,
                self.commands,
                self.cfg.obs_base_ang_vel_scale,
                self.critic_frame,
            ],
            device=self.device,
        )
        wp.launch(
            compute_critic_joint_frame_kernel,
            dim=(self.num_envs, G1_JOINT_LANES),
            inputs=[
                self.joint_pos,
                self.joint_vel,
                self.default_joint_pos,
                self.raw_actions,
                self.cfg.obs_joint_vel_scale,
                G1_NUM_JOINTS,
                G1_JOINT_LANES,
                self.critic_frame,
            ],
            device=self.device,
        )
        if update_history:
            # Policy + critic history each has its own first_push_mask;
            # _reset_idx marks both for the same set of envs (asymmetry
            # would shift value baseline first H steps).
            wp.launch(
                append_or_fill_history_kernel,
                dim=(self.num_envs, G1_POLICY_FRAME_DIM),
                inputs=[
                    self.first_push_mask,
                    self.policy_frame,
                    self.policy_history_length,
                    self.history,
                ],
                device=self.device,
            )
            wp.launch(
                append_or_fill_history_kernel,
                dim=(self.num_envs, G1_CRITIC_FRAME_DIM),
                inputs=[
                    self.critic_first_push_mask,
                    self.critic_frame,
                    self.policy_history_length,
                    self.critic_history,
                ],
                device=self.device,
            )
            # Clear after both appends finished (same race-avoidance reason
            # split out from append kernel — see clear_first_push_kernel docstring).
            wp.launch(
                clear_first_push_kernel,
                dim=self.num_envs,
                inputs=[self.first_push_mask],
                device=self.device,
            )
            wp.launch(
                clear_first_push_kernel,
                dim=self.num_envs,
                inputs=[self.critic_first_push_mask],
                device=self.device,
            )
        wp.launch(
            flatten_history_kernel,
            dim=(self.num_envs, self.policy_history_length * G1_POLICY_FRAME_DIM),
            inputs=[
                self.history,
                self.policy_obs_to_slot,
                self.policy_obs_to_feat,
                self.observations,
            ],
            device=self.device,
        )
        wp.launch(
            flatten_history_kernel,
            dim=(self.num_envs, self.policy_history_length * G1_CRITIC_FRAME_DIM),
            inputs=[
                self.critic_history,
                self.critic_obs_to_slot,
                self.critic_obs_to_feat,
                self.critic_observations,
            ],
            device=self.device,
        )

    def _get_rewards(self) -> None:
        # Trigger lazy-property refresh (TimestampedBuffer) before reward kernel.
        # __init__-cached aliases would feed init-time zeros otherwise — silently
        # zeroes base_linear_velocity / flat_orientation_l2 etc.
        _ = self.robot.data.root_lin_vel_b
        _ = self.robot.data.root_ang_vel_b
        _ = self.robot.data.projected_gravity_b
        _ = self.robot.data.body_pos_w
        _ = self.robot.data.body_lin_vel_w
        # joint_acc: scene.update(dt) refreshes it implicitly today, but force
        # explicit access — undocumented dependency, easy to break on reorder.
        _ = self.robot.data.joint_acc

        # root_pose_w / root_quat_w / root_lin_vel_w (non-TimestampedBuffer)
        # are re-fetched at top of _step_warp_end_pre, before this method runs.

        wp.launch(
            compute_rewards_kernel,
            dim=self.num_envs,
            inputs=[
                self.root_pose_w,
                self.root_quat_w,
                self.root_lin_vel_w,
                self.root_lin_vel_b,
                self.root_ang_vel_b,
                self.projected_gravity_b,
                self.joint_pos,
                self.joint_vel,
                self.joint_acc,
                self.applied_torque,
                self.default_joint_pos,
                self.soft_joint_pos_limits,
                self.raw_actions,
                self.prev_actions,
                self.commands,
                self.reset_terminated,
                G1_NUM_JOINTS,
                self.arm_joint_ids,
                self.num_arm_joints,
                self.waist_joint_ids,
                self.num_waist_joints,
                self.leg_joint_ids,
                self.num_leg_joints,
                self.step_dt,
                self.cfg.rew_track_lin_vel_xy,
                self.cfg.track_lin_vel_xy_std_sq,
                self.cfg.rew_track_ang_vel_z,
                self.cfg.track_ang_vel_z_std_sq,
                self.cfg.rew_alive,
                self.cfg.rew_base_linear_velocity,
                self.cfg.rew_base_angular_velocity,
                self.cfg.rew_flat_orientation_l2,
                self.cfg.rew_base_height,
                self.cfg.base_height_target,
                self.cfg.rew_joint_vel,
                self.cfg.rew_joint_acc,
                self.cfg.rew_action_rate,
                self.cfg.rew_dof_pos_limits,
                self.cfg.rew_energy,
                self.cfg.rew_joint_deviation_arms,
                self.cfg.rew_joint_deviation_waists,
                self.cfg.rew_joint_deviation_legs,
                self.rewards,
                self._track_lin_vel_xy_step,
                self._track_ang_vel_z_step,
                self._reward_terms_step,
            ],
            device=self.device,
        )
        if self.contact_sensor is not None:
            wp.launch(
                add_contact_rewards_kernel,
                dim=self.num_envs,
                inputs=[
                    self._episode_length_buf_wp,
                    self.commands,
                    self.contact_sensor.data.net_forces_w_history,
                    self.contact_sensor.data.current_contact_time,
                    self.body_pos_w,
                    self.body_lin_vel_w,
                    self.foot_body_ids,
                    self.num_feet,
                    self.foot_phase_offsets,
                    self.undesired_body_ids,
                    self.num_undesired_bodies,
                    self.contact_history_length,
                    self.step_dt,
                    self.cfg.gait_period,
                    self.cfg.gait_threshold,
                    self.cfg.gait_command_threshold,
                    self.cfg.feet_slide_force_threshold,
                    self.cfg.rew_gait,
                    self.cfg.rew_feet_slide,
                    self.cfg.rew_feet_clearance,
                    self.cfg.feet_clearance_target_height,
                    self.cfg.feet_clearance_std,
                    self.cfg.feet_clearance_tanh_mult,
                    self.cfg.undesired_contacts_threshold,
                    self.cfg.rew_undesired_contacts,
                    self.rewards,
                    self._reward_terms_step,
                ],
                device=self.device,
            )

    def _step_warp_end_pre(self) -> None:
        """Re-fetch 3 derived lazy properties before parent's end_pre graph runs.

        ``root_pose_w`` / ``root_quat_w`` / ``root_lin_vel_w`` are NOT
        TimestampedBuffer-backed — each access returns a FRESH wp.array (new
        pointer). __init__-cached aliases would forever point to init-time
        zeros, silently zeroing reward/dones/reset's view of root state.

        Python attr assignment runs at first capture only; subsequent kernels
        bake the new pointer, and Newton's in-place buffer update keeps it valid.
        """
        self.root_pose_w = self.robot.data.root_pose_w
        self.root_quat_w = self.robot.data.root_quat_w
        self.root_lin_vel_w = self.robot.data.root_lin_vel_w
        super()._step_warp_end_pre()

    def _get_dones(self) -> None:
        # Trigger lazy-property refresh (TimestampedBuffer); without this
        # the cached alias lags 1 sim step behind, letting near-fall
        # orientations slip past dones check.
        _ = self.robot.data.projected_gravity_b

        wp.launch(
            get_dones_kernel,
            dim=self.num_envs,
            inputs=[
                self._episode_length_buf_wp,
                self.root_pose_w,
                self.projected_gravity_b,
                self.max_episode_length,
                self.cfg.termination_base_height,
                self.cfg.termination_bad_orientation,
                self.reset_terminated,
                self.reset_time_outs,
                self.reset_buf,
            ],
            device=self.device,
        )

    def _reset_idx(self, mask: wp.array | None = None) -> None:
        """Reset envs flagged by ``mask``. Pure-Warp path (no host torch ops).

        We do NOT call super()._reset_idx because InteractiveSceneWarp.reset
        drops env_mask for sensor resets, which lets NewtonContactSensor.reset
        fall into wp.full() (default stream) and trigger 906 mid-capture.
        Workaround: inline DirectRLEnvWarp._reset_idx with explicit env_mask
        forwarding to all asset types (articulation / deformable / rigid_object /
        gripper / rigid_object_collection / sensor).
        """
        if mask is None:
            mask = self._ALL_ENV_MASK

        # Forward env_mask to all asset types (mirror InteractiveSceneWarp.reset
        # but explicitly for sensors too). Order matches upstream so divergence
        # is easy to spot. G1 cfg has no deformable/gripper currently, but loops
        # are zero-iter then — future cfgs adding them work without changes.
        for articulation in self.scene._articulations.values():
            articulation.reset(env_ids=None, env_mask=mask)
        for deformable_object in self.scene._deformable_objects.values():
            deformable_object.reset(env_ids=None, env_mask=mask)
        for rigid_object in self.scene._rigid_objects.values():
            rigid_object.reset(env_ids=None, env_mask=mask)
        for surface_gripper in self.scene._surface_grippers.values():
            surface_gripper.reset(env_ids=None, env_mask=mask)
        for rigid_object_collection in self.scene._rigid_object_collections.values():
            rigid_object_collection.reset(env_ids=None, env_mask=mask)
        # Sensors: env_mask required to avoid wp.full() 906 path. TypeError
        # means a sensor doesn't support env_mask yet — raise loudly rather
        # than silently fall back to env_ids=None (would reset ALL envs).
        for sensor in self.scene._sensors.values():
            try:
                sensor.reset(env_ids=None, env_mask=mask)
            except TypeError as e:
                raise RuntimeError(
                    f"Sensor {type(sensor).__name__} does not support env_mask "
                    f"kwarg in reset(); G1 Direct Warp requires per-env "
                    f"reset semantics for CUDA-graph-safe partial reset. "
                    f"Either upgrade the sensor to accept env_mask, or use "
                    f"the Manager / Direct workflow that takes env_ids."
                ) from e

        # Zero episode_length_buf for resetting envs.
        wp.launch(
            zero_mask_int32,
            dim=self.num_envs,
            inputs=[mask, self._episode_length_buf_wp],
        )

        wp.launch(
            reset_root_kernel,
            dim=self.num_envs,
            inputs=[
                mask,
                self.rng_state,
                self.robot.data.default_root_pose,
                self.env_origins,
                self.root_pose_w,
                self.root_vel_w,
                self.cfg.reset_root_x_range,
                self.cfg.reset_root_y_range,
                self.cfg.reset_root_yaw_range,
            ],
            device=self.device,
        )
        wp.launch(
            reset_joints_kernel,
            dim=self.num_envs,
            inputs=[
                mask,
                self.rng_state,
                self.default_joint_pos,
                self.default_joint_vel,
                self.joint_pos,
                self.joint_vel,
                self.cfg.reset_joint_velocity_range,
                G1_NUM_JOINTS,
            ],
            device=self.device,
        )
        wp.launch(
            clear_reset_buffers_kernel,
            dim=(self.num_envs, G1_NUM_JOINTS),
            inputs=[mask, self.default_joint_pos, self.raw_actions, self.prev_actions, self.joint_pos_target],
            device=self.device,
        )
        # Set first_push_mask=True for resetting envs (broadcast frame to all
        # H history slots on next _compute_observations, mirrors CircularBuffer
        # first-push). Policy + critic must mark together for the SAME mask —
        # splitting would create asymmetric value-baseline shift first H steps.
        wp.launch(
            set_first_push_kernel,
            dim=self.num_envs,
            inputs=[mask, self.first_push_mask],
            device=self.device,
        )
        wp.launch(
            set_first_push_kernel,
            dim=self.num_envs,
            inputs=[mask, self.critic_first_push_mask],
            device=self.device,
        )
        wp.launch(
            sample_reset_commands_kernel,
            dim=self.num_envs,
            inputs=[
                mask,
                self.rng_state,
                self.command_time_left,
                self.commands,
                self.cfg.command_resampling_time,
                self.cfg.rel_standing_envs,
                # Host-mutable wp.array ranges (NOT cfg tuples) so curriculum
                # mutation between replays takes effect.
                self._lin_vel_x_range_wp,
                self._lin_vel_y_range_wp,
                self._ang_vel_z_range_wp,
            ],
            device=self.device,
        )
        # Seed push timer (mirror EventManager init); without this push fires step 1
        # while Manager waits the sampled interval.
        wp.launch(
            init_push_timer_kernel,
            dim=self.num_envs,
            inputs=[
                mask,
                self.rng_state,
                self.push_time_left,
                self.cfg.push_interval_range_s,
            ],
            device=self.device,
        )

        # External wrench (gated by cfg.enable_external_force_torque; when False
        # kernel never enters captured graph, see file-level docstring).
        if self.cfg.enable_external_force_torque:
            wp.launch(
                apply_external_force_torque_kernel,
                dim=self.num_envs,
                inputs=[
                    mask,
                    self.rng_state,
                    self._external_force_body_id,
                    self.cfg.external_force_range,
                    self.cfg.external_torque_range,
                    self._external_force_b_wp,
                    self._external_torque_b_wp,
                ],
                device=self.device,
            )

    def _step_warp_end_post(self) -> None:
        wp.launch(
            update_interval_commands_kernel,
            dim=self.num_envs,
            inputs=[
                self.rng_state,
                self.command_time_left,
                self.commands,
                self.step_dt,
                self.cfg.command_resampling_time,
                self.cfg.rel_standing_envs,
                # Pass host-mutable wp.array ranges — see
                # ``sample_reset_commands_kernel`` launch comment.
                self._lin_vel_x_range_wp,
                self._lin_vel_y_range_wp,
                self._ang_vel_z_range_wp,
            ],
            device=self.device,
        )
        wp.launch(
            update_push_kernel,
            dim=self.num_envs,
            inputs=[
                self.rng_state,
                self.push_time_left,
                self.root_vel_w,
                self.step_dt,
                self.cfg.push_interval_range_s,
                self.cfg.push_velocity_x_range,
                self.cfg.push_velocity_y_range,
                self.cfg.push_velocity_z_range,
                self.cfg.push_velocity_roll_range,
                self.cfg.push_velocity_pitch_range,
                self.cfg.push_velocity_yaw_range,
            ],
            device=self.device,
        )
        self._compute_observations(update_history=True)
