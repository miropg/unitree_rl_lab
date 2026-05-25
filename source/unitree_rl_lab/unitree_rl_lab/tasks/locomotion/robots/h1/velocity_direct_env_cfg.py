# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow cfg for the H1 (19 dof) velocity tracker.

Mirrors :class:`.velocity_env_cfg.RobotFlatEnvCfg`.  Notable bits:

* H1 Manager cfg also comments out ``history_length=5`` -> single-step obs.
* Both groups have an extra ``gait_phase`` term (2 dims) and the critic has
  ``joint_effort`` (19 dims).
"""

from __future__ import annotations

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.direct import UnitreeVelocityDirectEnvCfg

from .velocity_env_cfg import (
    CommandsCfg,
    CurriculumCfg,
    EventCfgPreset,
    ObservationsCfg,
    PhysicsCfg,
    RewardsCfg,
    RobotSceneCfg,
    TerminationsCfg,
)


_H1_NUM_JOINTS = 19
# Per-step obs widths from H1 Manager cfg PolicyCfg / CriticCfg term order:
#   policy = ang_vel(3) + grav(3) + cmd(3) + jpos(N) + jvel(N) + last(N) + gait_phase(2)  = 11 + 3*N
#   critic = lin_vel(3) + ang_vel(3) + grav(3) + cmd(3)
#          + jpos(N) + jvel(N) + effort(N) + last(N) + gait_phase(2)                       = 14 + 4*N
_POLICY_DIM_PER_STEP = 11 + 3 * _H1_NUM_JOINTS  # 68
_CRITIC_DIM_PER_STEP = 14 + 4 * _H1_NUM_JOINTS  # 90


@configclass
class H1VelocityDirectEnvCfg(UnitreeVelocityDirectEnvCfg):
    """Direct cfg for ``Unitree-H1-Velocity-Flat-Direct``."""

    action_space: int = _H1_NUM_JOINTS
    observation_space: int = _POLICY_DIM_PER_STEP
    state_space: int = _CRITIC_DIM_PER_STEP

    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    events: EventCfgPreset = EventCfgPreset()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # H1 Manager cfg sets ``self.gait_f = 2.0`` here for downstream
        # consumption; we mirror it for any reward/obs term that may read it.
        self.gait_f = 2.0

        self.scene.contact_forces.physx.update_period = self.sim.dt
        self.scene.contact_forces.newton.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Flat variant
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None


@configclass
class H1VelocityDirectPlayEnvCfg(H1VelocityDirectEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
