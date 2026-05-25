# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cfg containers for :class:`UnitreeVelocityDirectEnv`.

These cfg classes intentionally re-use the **same** ``RewardTermCfg`` /
``ObservationTermCfg`` / ``TerminationTermCfg`` / ``EventTermCfg`` containers
that the Manager-based env uses, so per-robot cfgs can be a near copy-paste
of the existing :mod:`unitree_rl_lab.tasks.locomotion.robots.*.velocity_env_cfg`.
The Direct env class :class:`UnitreeVelocityDirectEnv` parses these containers
in its ``__init__`` and dispatches each term itself (no ``RewardManager``).

Notes for next reader:
    * we do not subclass :class:`~isaaclab.managers.RewardTermCfg` etc.; the
      Direct env iterates ``__dataclass_fields__`` and skips ``None``, so a
      subclass cfg can disable a term inherited from the base by setting it
      to ``None``;
    * ``observation_space`` / ``state_space`` / ``action_space`` are
      ``int`` (totals after history flattening); each per-robot cfg is
      responsible for computing them in ``__post_init__`` because
      :meth:`~isaaclab.envs.DirectRLEnv._configure_gym_env_spaces` runs before
      the env can introspect any tensor shape.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class VelocityRewardsCfg:
    """Container for reward terms.

    Each field is :class:`isaaclab.managers.RewardTermCfg` (or ``None`` to
    disable a term inherited from a parent cfg).  The Direct env iterates
    ``__dataclass_fields__`` in declaration order and accumulates
    ``func(env, **params) * weight * step_dt`` (matches
    :meth:`~isaaclab.managers.RewardManager.compute`).
    """

    # Field declarations are *not* set here; per-robot subclasses fill them via
    # ``__post_init__`` to match the existing Manager cfgs verbatim.  Using a
    # bare ``configclass`` with no fields lets subclasses set arbitrary
    # attributes (each becomes an entry in ``__dataclass_fields__`` because
    # configclass turns dynamic attributes into dataclass fields).
    pass


@configclass
class VelocityObservationsCfg:
    """Container for observation groups.

    Two attributes ``policy`` and ``critic`` (each
    :class:`isaaclab.managers.ObservationGroupCfg`).  The Direct env honours:

    * ``group.history_length`` (per-term :class:`~isaaclab.utils.buffers.CircularBuffer`),
    * ``group.enable_corruption`` (set ``False`` to skip ``noise`` injection),
    * ``group.flatten_history_dim`` (default ``True``: ``(N, H, D) -> (N, H*D)``),
    * ``group.concatenate_terms`` (default ``True``: cat all terms along last dim).
    """

    policy: object = MISSING
    critic: object = MISSING


@configclass
class VelocityTerminationsCfg:
    """Container for termination terms.

    Each field is :class:`isaaclab.managers.TerminationTermCfg`.  The Direct
    env iterates them in declaration order and OR-merges; ``time_out=True``
    terms go into ``truncated``, others into ``terminated``.
    """

    pass


@configclass
class VelocityCommandsCfg:
    """Container for command terms.

    Held verbatim because we keep :class:`~isaaclab.managers.CommandManager`
    (the resample timer / heading_command / standing-envs logic is too
    intricate to re-implement just to save a microsecond).  The Direct env
    wraps it in the same way as Manager env.
    """

    pass


@configclass
class VelocityEventsCfg:
    """Container for event terms (startup / reset / interval).

    Held verbatim because :class:`~isaaclab.envs.DirectRLEnv` already invokes
    :class:`~isaaclab.managers.EventManager` automatically when ``cfg.events``
    is non-None (see ``direct_rl_env.py`` lines 170-175 / 252-253 / 441-444).
    We do not need to call it ourselves.
    """

    pass


@configclass
class UnitreeVelocityDirectEnvCfg(DirectRLEnvCfg):
    """Base cfg for G1 / Go2 / H1 Direct-workflow velocity tracking.

    Per-robot subclasses must:

    1. Set :attr:`scene` (with ``robot=...`` and the right ``ContactSensor``
       :class:`~isaaclab_tasks.utils.PresetCfg`).
    2. Set :attr:`sim` (typically ``SimulationCfg(physics=PhysicsCfg())``
       sharing the same ``PresetCfg`` Sharron added in commit ``e063e1a``).
    3. Set :attr:`rewards` / :attr:`observations` / :attr:`terminations` /
       :attr:`events` / :attr:`commands` / :attr:`curriculum`.
    4. Compute :attr:`action_space` (number of joints), :attr:`observation_space`
       (per-step policy obs dim x ``policy.history_length``) and
       :attr:`state_space` (per-step critic obs dim x ``critic.history_length``).
    """

    # ------------------------------------------------------------------
    # DirectRLEnv-required fields
    # ------------------------------------------------------------------
    decimation: int = 4
    episode_length_s: float = 20.0

    action_space: int = MISSING
    observation_space: int = MISSING
    state_space: int = MISSING

    sim: SimulationCfg = MISSING
    scene: InteractiveSceneCfg = MISSING

    # ------------------------------------------------------------------
    # Direct-workflow specific fields
    # ------------------------------------------------------------------
    action_scale: float = 0.25
    """Multiplier applied to the raw action before adding ``default_joint_pos``.

    Matches ``mdp.JointPositionActionCfg(scale=0.25, use_default_offset=True)``
    used in every Manager-version cfg in this repo.
    """

    action_clip: tuple[float, float] | None = None
    """Per-joint clip applied to the **processed** joint position target
    (``raw * scale + default_joint_pos``), matching ``JointPositionAction``
    semantics in joint_actions.py:170-179.

    Default is ``None`` (no clip) to match G1 and H1 Manager cfgs which omit
    the ``clip`` arg.  Go2's Manager cfg sets
    ``clip={".*": (-100.0, 100.0)}`` so its Direct cfg overrides this to
    ``(-100.0, 100.0)``.
    """

    # ------------------------------------------------------------------
    # MDP cfg containers (Manager-style, parsed by Direct env)
    # ------------------------------------------------------------------
    rewards: VelocityRewardsCfg = MISSING
    observations: VelocityObservationsCfg = MISSING
    terminations: VelocityTerminationsCfg = MISSING

    # ------------------------------------------------------------------
    # Manager-managed pieces we keep as-is
    # ------------------------------------------------------------------
    events: VelocityEventsCfg | None = None
    """Auto-handled by :class:`~isaaclab.envs.DirectRLEnv` (startup in
    ``_init_sim``, reset in ``_reset_idx``, interval at the end of ``step``)."""

    commands: VelocityCommandsCfg = MISSING
    """We instantiate :class:`~isaaclab.managers.CommandManager` in our env
    ``__init__`` and call ``compute(dt=step_dt)`` once per ``step``."""

    curriculum: object = None
    """Optional :class:`~isaaclab.managers.CurriculumManager` cfg; if set we
    instantiate the manager and call ``compute(env_ids)`` from ``_reset_idx``
    (matches Manager-version :meth:`ManagerBasedRLEnv._reset_idx` order)."""
