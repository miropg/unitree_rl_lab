# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRLEnvWarp cfg for the G1-29dof flat velocity tracker."""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.direct.velocity_warp_g1_env import (
    G1_CRITIC_FRAME_DIM,
    G1_HISTORY_LENGTH,
    G1_NUM_JOINTS,
    G1_POLICY_FRAME_DIM,
)

from .velocity_env_cfg import RobotSceneCfg


@configclass
class G1WarpSceneCfg(RobotSceneCfg):
    """G1 flat scene for the Newton-only Direct Warp task."""

    contact_forces: NewtonContactSensorCfg = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


@configclass
class G1WarpEventsCfg:
    """Startup-only events for the G1 Direct Warp task.

    DirectRLEnvWarp's current implementation auto-applies ``prestartup`` and
    ``startup`` events from :class:`~isaaclab.managers.EventManager` (see
    ``direct_rl_env_warp.py:185-190,262-267``).  ``reset`` and ``interval``
    modes are intentionally commented out in the base class
    (``direct_rl_env_warp.py:511-513,730-734``), so this cfg only exposes the
    one event we need on this code path: torso mass randomization.

    The matching Manager / Direct cfg
    (:class:`unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg
    .ResetEventCfg.add_base_mass`) randomizes torso mass by ``Uniform(-1, 3)``
    kg.  Without this term, Warp training sees a fixed-mass torso while
    Manager sees a 4 kg spread, which makes training-curve parity impossible
    even after every other reward is aligned.

    Reset-time pose / joint randomization and interval push velocity are
    re-implemented as Warp kernels in
    :class:`G1VelocityWarpEnv` (``_reset_root_kernel`` /
    ``_reset_joints_kernel`` / ``_update_push_kernel``); we do NOT register
    the Manager event terms for those modes here, otherwise we would risk a
    silent double-apply if DirectRLEnvWarp ever turns those branches back on.
    """

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )


@configclass
class G1VelocityWarpEnvCfg(DirectRLEnvCfg):
    """Warp-native cfg for ``Unitree-G1-29dof-Velocity-Flat-Direct-Warp``.

    Layout convention: flat + prefix-naming
    ----------------------------------------

    All fields below are declared at the top level of this class (flat
    layout), with related fields grouped via shared prefixes.  Logical
    groups present:

    * ``action_*``         (action processing)
    * ``policy_*`` / ``observation_*`` / ``state_*``  (obs / critic dims)
    * ``command_*`` / ``rel_standing_envs`` / ``lin_vel_*_range``
                           (command sampling)
    * ``enable_command_curriculum`` / ``*_limit_range`` /
      ``curriculum_*``    (command curriculum)
    * ``reset_*``          (root pose / joint reset randomization)
    * ``push_*``           (interval-mode push event)
    * ``enable_external_force_torque`` / ``external_*``
                           (external wrench event)
    * ``obs_*_scale`` / ``enable_observation_noise`` / ``obs_noise_*_range``
                           (obs scale + noise corruption)
    * ``termination_*``    (termination thresholds)
    * ``rew_*`` / ``track_*_std_sq`` / ``base_height_target`` / ``gait_*`` /
      ``feet_*_threshold`` / ``feet_clearance_*`` / ``undesired_contacts_*``
                           (reward weights + reward params)

    Why flat + prefix instead of nested ``@configclass`` groups?

    1. **Hydra CLI override convenience**: flat allows
       ``env.external_force_range=[-50,50]``; nested would need
       ``env.external_force_torque.range=[-50,50]``.
    2. **DirectRLEnvCfg base class is flat**: inherited fields
       (``action_space``, ``observation_space``, ``decimation``, ``sim``,
       ``scene``) are all flat; nesting subclass fields would mix styles.
    3. **configclass + Hydra OmegaConf nested merging has had quirks
       historically** (dev3 era hit nested-config serialisation issues);
       flat is the safe path until upstream isaaclab settles its own
       nested cfg pattern.

    Future work: nested cfg groups
    -------------------------------

    The flat layout is functional but not great for discoverability —
    related fields are scattered across ~50 lines, and adding a new
    field to a group requires finding the right insertion point by
    searching for the prefix.  A future refactor SHOULD convert the
    groups above into nested ``@configclass`` containers, e.g.::

        @configclass
        class ExternalForceTorqueCfg:
            enabled: bool = False
            force_range: tuple[float, float] = (0.0, 0.0)
            torque_range: tuple[float, float] = (0.0, 0.0)
            body_name: str = "torso_link"

        @configclass
        class CommandCurriculumCfg:
            enabled: bool = True
            lin_vel_x_limit_range: tuple[float, float] = (-0.5, 1.0)
            ...

        class G1VelocityWarpEnvCfg(DirectRLEnvCfg):
            external_force_torque: ExternalForceTorqueCfg = ExternalForceTorqueCfg()
            command_curriculum: CommandCurriculumCfg = CommandCurriculumCfg()
            ...

    When doing this refactor, follow these rules:

    * **Convert ALL groups in one PR.** Half-flat / half-nested is
      worse than either pure form — readers can't predict which style
      a given field uses.
    * **Update all in-repo references**: ``self.cfg.external_force_range``
      becomes ``self.cfg.external_force_torque.force_range`` etc.;
      grep ``self.cfg\.[a-z_]+_range`` and similar to find them.
    * **Re-run full parity test** after refactor (4-backend Manager /
      Direct / DirectWarp / Warp parity must remain bit-stable).
    * **Document Hydra override migration** for users with existing
      override scripts: ``env.external_force_range`` → ``env.external_force_torque.force_range``.
    * **Historical training_parity snapshots** under ``agent_tmp/training_parity/**/``
      will not load against the new cfg shape; that is expected (snapshots
      are immutable historical artefacts, not for resume training).
    * **Coordinate with H1 / Go2 cfg files** — if they exist or get added,
      they should adopt the same nested structure.
    """

    action_space: int = G1_NUM_JOINTS
    policy_history_length: int = G1_HISTORY_LENGTH
    """Number of policy obs frames stacked into the history buffer.

    Mirrors Manager :class:`ObservationsCfg.PolicyCfg.history_length=5`.
    Forwarded as a runtime ``wp.int32`` to the history kernels so cfg
    overrides do not require recompiling — same pattern as the official
    ``num_dof`` parameter in
    ``isaaclab_tasks_experimental/direct/locomotion/locomotion_env_warp.py``.

    Override implies overriding :attr:`observation_space` to match
    ``policy_history_length * G1_POLICY_FRAME_DIM`` (96).
    :class:`G1VelocityWarpEnv.__init__` raises ``ValueError`` if they
    disagree.
    """
    observation_space: int = G1_POLICY_FRAME_DIM * G1_HISTORY_LENGTH
    state_space: int = G1_CRITIC_FRAME_DIM * G1_HISTORY_LENGTH
    """Critic / privileged observation dimension.

    ``G1_CRITIC_FRAME_DIM (99) * policy_history_length (5) = 495``, mirroring
    Manager :class:`ObservationsCfg.CriticCfg` which adds ``base_lin_vel``
    (3 dim) on top of the policy obs to feed PPO's critic with a
    privileged signal.

    The ``"critic"`` group is emitted by
    :meth:`G1VelocityWarpEnv._get_observations` and
    :meth:`G1VelocityWarpEnv.step`; the matching
    :class:`BasePPORunnerAsymmetricCfg` registered as the
    ``rsl_rl_cfg_entry_point`` for ``Unitree-G1-29dof-Velocity-Flat-Direct-Warp``
    consumes it via :class:`RslRlPpoActorCriticAsymmetricCfg`.

    Strict check in :meth:`G1VelocityWarpEnv.__init__`:
    ``cfg.state_space == G1_CRITIC_FRAME_DIM * cfg.policy_history_length``
    or ``ValueError``.  Critic obs is **noise-free** by design (Manager
    :class:`CriticCfg.__post_init__` does not enable corruption).
    """
    ui_window_class_type = None

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=95,
                nconmax=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                ls_parallel=True,
                ls_iterations=15,
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        ),
    )
    scene: G1WarpSceneCfg = G1WarpSceneCfg(num_envs=4096, env_spacing=2.5)
    events: G1WarpEventsCfg = G1WarpEventsCfg()
    """Startup events (only ``add_base_mass`` is active; see :class:`G1WarpEventsCfg`)."""

    action_scale: float = 0.25
    command_resampling_time: float = 10.0
    rel_standing_envs: float = 0.02
    lin_vel_x_range: tuple[float, float] = (-0.1, 0.1)
    """Initial sampling range for lin_vel_x command (matches Manager
    :class:`CommandsCfg.base_velocity.ranges.lin_vel_x`).  At runtime,
    :meth:`G1VelocityWarpEnv._curriculum_step` mutates the active
    range up to :attr:`lin_vel_x_limit_range` when the policy is
    consistently tracking lin_vel_x reward above
    ``rew_track_lin_vel_xy * 0.8``.  See dev6 root-cause-fix
    ``agent_tmp/review/reviewer12/g1_direct_warp_review_v6.md`` §"P0-1".
    """
    lin_vel_y_range: tuple[float, float] = (-0.1, 0.1)
    """Initial sampling range for lin_vel_y command.  Curriculum target
    is :attr:`lin_vel_y_limit_range`.
    """
    ang_vel_z_range: tuple[float, float] = (-0.1, 0.1)
    """Initial sampling range for ang_vel_z command.  Curriculum target
    is :attr:`ang_vel_z_limit_range`.
    """

    # Command curriculum (mirror Manager :func:`mdp.lin_vel_cmd_levels` /
    # :func:`mdp.ang_vel_cmd_levels`).  Without these fields the warp env
    # never widens its cmd ranges so the tracking task stays trivially
    # easy compared to manager / direct (which extend to limit_ranges as
    # the policy improves).  This was the real root cause of the +80%
    # reward divergence in Item 6 v3 (reviewer12 v6 §0).
    enable_command_curriculum: bool = True
    """Enable command-range curriculum.  Default ``True`` matches Manager
    :class:`CurriculumCfg.lin_vel_cmd_levels` / ``ang_vel_cmd_levels``.

    Set to ``False`` for parity tests against pinned-range trajectories or
    for play / deploy cfgs (which already start with ranges == limit
    via :class:`G1VelocityWarpPlayEnvCfg`).
    """
    lin_vel_x_limit_range: tuple[float, float] = (-0.5, 1.0)
    """Upper bound for lin_vel_x curriculum extension.  Mirrors Manager
    :class:`CommandsCfg.base_velocity.limit_ranges.lin_vel_x`.
    """
    lin_vel_y_limit_range: tuple[float, float] = (-0.3, 0.3)
    """Upper bound for lin_vel_y curriculum extension.  Mirrors Manager
    :class:`CommandsCfg.base_velocity.limit_ranges.lin_vel_y`.
    """
    ang_vel_z_limit_range: tuple[float, float] = (-0.2, 0.2)
    """Upper bound for ang_vel_z curriculum extension.  Mirrors Manager
    :class:`CommandsCfg.base_velocity.limit_ranges.ang_vel_z`.
    """
    curriculum_track_lin_vel_xy_threshold_factor: float = 0.8
    """Multiplier on :attr:`rew_track_lin_vel_xy` for the curriculum
    extension trigger.  Mirrors Manager :func:`mdp.lin_vel_cmd_levels`
    L24 ``reward > reward_term.weight * 0.8``.
    """
    curriculum_track_ang_vel_z_threshold_factor: float = 0.8
    """Same factor for the ang_vel_z curriculum (Manager:
    :func:`mdp.ang_vel_cmd_levels` L53 — note Manager hardcodes 0.8).
    """
    curriculum_delta: float = 0.1
    """Per-trigger range extension (Manager hardcodes ``0.1``)."""

    reset_root_x_range: tuple[float, float] = (-0.5, 0.5)
    reset_root_y_range: tuple[float, float] = (-0.5, 0.5)
    reset_root_yaw_range: tuple[float, float] = (-3.14, 3.14)
    reset_joint_velocity_range: tuple[float, float] = (-1.0, 1.0)

    # ----- Interval push event (mirror Manager ``push_robot``) -----
    # Manager cfg (``robots/g1/29dof/velocity_env_cfg.py:175-180``):
    #   push_robot = EventTerm(
    #       func=mdp.push_by_setting_velocity, mode="interval",
    #       interval_range_s=(5.0, 5.0),
    #       params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    #   )
    #
    # Manager's ``mdp.push_by_setting_velocity`` (``events.py:993-997``):
    #     range_list = [velocity_range.get(key, (0.0, 0.0))
    #                   for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    #
    # → missing keys default to ``(0, 0)``.  Below the same 6 axis are
    # exposed as cfg fields so :func:`_update_push_kernel` can sample
    # all 6 (matching Manager's RNG advance count exactly), even though
    # G1 reference cfg only sets x / y.
    push_interval_range_s: tuple[float, float] = (5.0, 5.0)
    """Interval (seconds) between pushes, sampled fresh after every push.

    Mirrors Manager ``EventTermCfg(push_robot).interval_range_s``
    (``robots/g1/29dof/velocity_env_cfg.py:178``).  G1 reference uses
    ``(5.0, 5.0)`` (degenerate range → constant 5 s); set ``(3.0, 7.0)``
    for randomised intervals matching Manager's ``torch.rand * (upper -
    lower) + lower`` semantics (``event_manager.py:222-226``).
    """
    push_velocity_x_range: tuple[float, float] = (-0.5, 0.5)
    """Linear x velocity delta range [m/s]; mirrors Manager
    ``velocity_range['x']`` for ``push_by_setting_velocity``."""
    push_velocity_y_range: tuple[float, float] = (-0.5, 0.5)
    """Linear y velocity delta range [m/s]."""
    push_velocity_z_range: tuple[float, float] = (0.0, 0.0)
    """Linear z velocity delta range [m/s].  Default ``(0, 0)`` matches
    Manager G1 cfg (``z`` key not in ``velocity_range`` dict; Manager
    defaults missing keys to ``(0, 0)``, ``events.py:995``)."""
    push_velocity_roll_range: tuple[float, float] = (0.0, 0.0)
    """Angular roll velocity delta range [rad/s]; default ``(0, 0)``."""
    push_velocity_pitch_range: tuple[float, float] = (0.0, 0.0)
    """Angular pitch velocity delta range [rad/s]; default ``(0, 0)``."""
    push_velocity_yaw_range: tuple[float, float] = (0.0, 0.0)
    """Angular yaw velocity delta range [rad/s]; default ``(0, 0)``."""

    push_is_global_time: bool = False
    """If ``True``, all envs share a single timer and push synchronously.

    Mirrors Manager ``EventTermCfg.is_global_time``
    (``event_manager.py:213-220``).  G1/H1/Go2 reference cfgs do not set
    this flag so it defaults ``False`` (per-env timers, all envs push
    asynchronously when each env's own timer fires).

    .. warning::
        :meth:`G1VelocityWarpEnv.__init__` raises
        ``NotImplementedError`` when this is set to ``True``.  A clean
        Warp implementation would need either (a) a host-side timer
        check that breaks CUDA Graph capture, or (b) a single-thread
        atomic timer-update kernel.  G1/H1/Go2 reference cfgs all use
        per-env mode so this code path has not been wired up.  File an
        issue if your task needs the synchronous-push semantics.
    """

    # External wrench randomization (mirror Manager
    # ``base_external_force_torque`` / :func:`mdp.apply_external_force_torque`,
    # ``isaaclab.envs.mdp.events`` line 943).
    #
    # Manager cfg for G1 / H1 / Go2 (``robots/{g1/29dof,h1,go2}/velocity_env_cfg.py``
    # line 139-145 / 139 / 167) all hardcode ``force_range = (0.0, 0.0)``,
    # ``torque_range = (-0.0, 0.0)``, making the term a NO-OP at the
    # canonical reference cfg.  Upstream IsaacLab official velocity
    # tasks (``isaaclab_tasks/manager_based/locomotion/velocity/config/{a1,
    # anymal_b/c/d, digit, g1, go1, ...}/{flat,rough}_env_cfg.py``) simply
    # disable the term with ``self.events.base_external_force_torque = None``.
    #
    # Defaults below mirror that NO-OP behaviour exactly: when both
    # ranges are ``(0.0, 0.0)``, :class:`G1VelocityWarpEnv.__init__`
    # leaves Newton's ``has_external_wrench`` flag at ``False`` and
    # :meth:`_reset_idx` skips the
    # :func:`_apply_external_force_torque_kernel` launch so the captured
    # CUDA graph stays minimal.
    #
    # Set both ranges to a non-zero range to actually apply external
    # wrench at every reset.  The kernel writes into
    # ``articulation._external_force_b`` / ``_external_torque_b`` and
    # Newton's ``_sim_bind_body_external_wrench`` propagates them to
    # ``sim.step`` at every ``write_data_to_sim``.
    enable_external_force_torque: bool = False
    """Whether to enable external force/torque randomization on reset.

    Default ``False`` matches Manager NO-OP behaviour for G1/H1/Go2
    (canonical Manager cfgs set ``force_range=(0, 0)``, ``torque_range=(0, 0)``).

    When ``True``: :meth:`__init__` allocates the wrench buffers and
    sets ``robot.has_external_wrench=True``; :meth:`_reset_idx` launches
    :func:`_apply_external_force_torque_kernel` on every reset.  The
    kernel runs even when :attr:`external_force_range` /
    :attr:`external_torque_range` are ``(0, 0)`` (writes zeros) — opt-in
    is explicit, no implicit "all-zero range = disabled" detection.

    When ``False``: kernel is NOT launched; ``has_external_wrench`` stays
    ``False``, Newton skips the per-sub-step wrench buffer read.  Saves
    HBM bandwidth in the physics hot path (~1-3% of physics step time).
    """
    external_force_range: tuple[float, float] = (0.0, 0.0)
    """Per-component external force sample range [N], applied to the body
    named :attr:`external_force_body_name` at every reset.  Only used
    when :attr:`enable_external_force_torque` is ``True``."""
    external_torque_range: tuple[float, float] = (0.0, 0.0)
    """Per-component external torque sample range [N⋅m]; see
    :attr:`external_force_range` for full semantics."""
    external_force_body_name: str = "torso_link"
    """Body that receives the external wrench.  Mirrors Manager
    ``base_external_force_torque.params["asset_cfg"].body_names="torso_link"``
    (G1/H1/Go2 cfg).  Resolved to a body_id via
    :meth:`Articulation.find_bodies` in
    :meth:`G1VelocityWarpEnv.__init__`."""

    # Friction / restitution randomization at startup.  Mirrors Manager
    # ``StartupEventCfg.physics_material``; DirectRLEnvWarp does not use
    # EventManager so we invoke the randomization from
    # :meth:`G1VelocityWarpEnv._randomize_friction_at_startup`.
    enable_friction_randomization: bool = True
    """Whether to randomize per-shape ``mu`` / ``restitution`` at startup.

    Default ``True`` mirrors the Manager / Direct ``physics_material``
    startup event.  Set ``False`` for deterministic-material parity tests.
    """
    friction_range: tuple[float, float] = (0.3, 1.0)
    """Per-shape ``mu`` sample range, ``Uniform(low, high)``.  Matches G1
    Manager ``static_friction_range`` (Go2 uses ``(0.3, 1.2)``)."""
    restitution_range: tuple[float, float] = (0.0, 0.0)
    """Per-shape restitution sample range, ``Uniform(low, high)``.  Matches
    G1 Manager ``restitution_range`` (Go2 uses ``(0.0, 0.15)``)."""

    # Observation scales from `ObservationsCfg.PolicyCfg`.
    obs_base_ang_vel_scale: float = 0.2
    obs_joint_vel_scale: float = 0.05

    enable_observation_noise: bool = True
    """Whether to inject Manager-style observation corruption noise on the
    policy obs.  Default ``True`` matches Manager
    :class:`ObservationsCfg.PolicyCfg.enable_corruption=True`.

    Critic obs is **always** noise-free (Manager
    :class:`CriticCfg.__post_init__` does not enable corruption); the
    critic frame kernels do not even take the noise rng / range params,
    so this flag only affects the policy frame kernels.

    Set to ``False`` for deterministic parity comparisons with Direct env
    (parity script must also disable Direct's noise via
    ``cfg.observations.policy.enable_corruption=False``).

    .. warning::
        **Init-time-only.** :meth:`G1VelocityWarpEnv._compute_observations`
        branches on ``self.cfg.enable_observation_noise`` to pick the noisy
        vs. deterministic policy frame kernels.  After the first
        ``end_post`` step the parent's :class:`WarpGraphCache` captures
        whichever branch was taken into the CUDA graph and replays it for
        every subsequent step — flipping this flag at runtime will silently
        keep using the originally-captured kernel.  Set this in cfg before
        ``gym.make()`` and treat it as immutable.  reviewer12 v3 §5 V1.
    """
    obs_noise_base_ang_vel_range: tuple[float, float] = (-0.2, 0.2)
    """Uniform noise range for ``base_ang_vel`` in policy obs (pre-scale).

    Mirrors Manager
    ``ObservationsCfg.PolicyCfg.base_ang_vel.noise=Unoise(-0.2, 0.2)``.
    Noise is added to the RAW value BEFORE
    :attr:`obs_base_ang_vel_scale` multiplies, matching
    :meth:`ObservationManager.compute_group` pipeline order
    ``func → modifiers → noise → clip → scale``
    (``observation_manager.py:393-407``).
    """
    obs_noise_projected_gravity_range: tuple[float, float] = (-0.05, 0.05)
    """Uniform noise range for ``projected_gravity`` in policy obs (no scale)."""
    obs_noise_joint_pos_range: tuple[float, float] = (-0.01, 0.01)
    """Uniform noise range for ``joint_pos_rel`` in policy obs (no scale on joint_pos)."""
    obs_noise_joint_vel_range: tuple[float, float] = (-1.5, 1.5)
    """Uniform noise range for ``joint_vel_rel`` in policy obs (pre-scale).

    Mirrors Manager
    ``ObservationsCfg.PolicyCfg.joint_vel_rel.noise=Unoise(-1.5, 1.5)``.
    Noise is added BEFORE :attr:`obs_joint_vel_scale` multiplies (so the
    ~1.5 rad/s noise becomes ~0.075 once scaled, same as Manager).
    """

    # Termination thresholds from `TerminationsCfg`.
    termination_base_height: float = 0.2
    termination_bad_orientation: float = 0.8

    # Reward weights and parameters from `RewardsCfg`.
    rew_track_lin_vel_xy: float = 1.0
    track_lin_vel_xy_std_sq: float = 0.25
    rew_track_ang_vel_z: float = 0.5
    track_ang_vel_z_std_sq: float = 0.25
    rew_alive: float = 0.15
    rew_base_linear_velocity: float = -2.0
    rew_base_angular_velocity: float = -0.05
    rew_joint_vel: float = -0.001
    rew_joint_acc: float = -2.5e-7
    rew_action_rate: float = -0.05
    rew_dof_pos_limits: float = -5.0
    rew_energy: float = -2.0e-5
    rew_joint_deviation_arms: float = -0.1
    rew_joint_deviation_waists: float = -1.0
    rew_joint_deviation_legs: float = -1.0
    rew_flat_orientation_l2: float = -5.0
    rew_base_height: float = -10.0
    base_height_target: float = 0.78
    rew_gait: float = 0.5
    gait_period: float = 0.8
    gait_threshold: float = 0.55
    gait_command_threshold: float = 0.1
    feet_gait_offsets: tuple[float, ...] = (0.0, 0.5)
    """Per-foot phase offsets for ``feet_gait``.

    Length must equal ``num_feet`` (2 for G1 biped trot).  Mirrors
    Manager :class:`RewardsCfg.gait` ``params['offset']``.  Exposed as a cfg
    so reward kernels stop hardcoding ``foot_idx == 1``.
    """
    rew_feet_slide: float = -0.2
    feet_slide_force_threshold: float = 1.0
    """Contact-force threshold for ``feet_slide`` history-max test [N].

    Manager :func:`mdp.feet_slide` hardcodes ``> 1.0``; mirrors that here so
    cfg overrides also work for parity sweeps.
    """
    rew_feet_clearance: float = 1.0
    feet_clearance_std: float = 0.05
    feet_clearance_tanh_mult: float = 2.0
    feet_clearance_target_height: float = 0.1
    rew_undesired_contacts: float = -1.0
    undesired_contacts_threshold: float = 1.0

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None


@configclass
class G1VelocityWarpPlayEnvCfg(G1VelocityWarpEnvCfg):
    """Light-weight play cfg for the G1 Warp task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.lin_vel_x_range = (-0.5, 1.0)
        self.lin_vel_y_range = (-0.3, 0.3)
        self.ang_vel_z_range = (-0.2, 0.2)
        # Play cfg starts at limit_ranges; disable curriculum so deploy
        # rollouts use the full range from step 0.  Mirrors Manager
        # ``RobotFlatPlayEnvCfg`` which sets ``ranges = limit_ranges``.
        self.enable_command_curriculum = False
