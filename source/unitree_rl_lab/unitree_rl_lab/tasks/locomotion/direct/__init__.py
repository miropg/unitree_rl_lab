# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow locomotion environments shared by G1 / Go2 / H1.

These environments mirror the semantics of their Manager-based counterparts in
:mod:`unitree_rl_lab.tasks.locomotion.robots` (same reward weights, same
observation shapes, same events / commands / curriculum) but bypass
:class:`isaaclab.managers.RewardManager` /
:class:`~isaaclab.managers.ObservationManager` /
:class:`~isaaclab.managers.TerminationManager` to remove per-term Python
dispatch overhead from the hot path.

Goals (Phase 2):

* keep cfg-driven reward / observation / termination so users can still tune
  weights or add a term by editing cfg only;
* keep :class:`~isaaclab.managers.EventManager` /
  :class:`~isaaclab.managers.CommandManager` /
  :class:`~isaaclab.managers.CurriculumManager` because they are non-trivial
  to re-implement and are not the perf bottleneck;
* be a clean baseline for Phase 3 where the per-term dispatch is fused into a
  single ``@wp.kernel``.
"""

from .velocity_base_env import UnitreeVelocityDirectEnv
from .velocity_base_env_cfg import (
    UnitreeVelocityDirectEnvCfg,
    VelocityCommandsCfg,
    VelocityEventsCfg,
    VelocityObservationsCfg,
    VelocityRewardsCfg,
    VelocityTerminationsCfg,
)
from .velocity_warp_g1_env import G1VelocityWarpEnv  # noqa: F401
