# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow cfg for the Go2 velocity tracker.

Mirrors :class:`.velocity_env_cfg.RobotFlatEnvCfg` (Go2 Manager Flat).  Two
notable differences vs G1:

* Go2's :class:`ObservationsCfg.PolicyCfg` / ``CriticCfg`` do **not** set a
  ``history_length`` (the Manager cfg explicitly comments it out), so
  ``observation_space`` / ``state_space`` here are the per-step dims.
* Critic has an extra ``joint_effort`` term -> per-step dim differs by 12.
* Go2 Manager cfg sets ``clip={".*": (-100.0, 100.0)}`` on the
  ``JointPositionAction``.  The base Direct cfg has no clip (matches G1 / H1);
  we override :attr:`action_clip` below so the joint position target is
  clamped to the same range as in Manager Go2.
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


_GO2_NUM_JOINTS = 12  # 4 legs x (hip + thigh + calf)
# Per-step obs widths (Go2 Manager cfg observation order):
#   policy = ang_vel(3) + grav(3) + cmd(3) + jpos(N) + jvel(N) + last(N)        = 9 + 3*N
#   critic = lin_vel(3) + ang_vel(3) + grav(3) + cmd(3)
#          + jpos(N) + jvel(N) + effort(N) + last(N)                            = 12 + 4*N
_POLICY_DIM_PER_STEP = 9 + 3 * _GO2_NUM_JOINTS  # 45
_CRITIC_DIM_PER_STEP = 12 + 4 * _GO2_NUM_JOINTS  # 60


@configclass
class Go2VelocityDirectEnvCfg(UnitreeVelocityDirectEnvCfg):
    """Direct cfg for ``Unitree-Go2-Velocity-Flat-Direct``."""

    action_space: int = _GO2_NUM_JOINTS
    observation_space: int = _POLICY_DIM_PER_STEP  # no history
    state_space: int = _CRITIC_DIM_PER_STEP  # no history

    # Go2 Manager cfg sets ``clip={".*": (-100.0, 100.0)}`` (velocity_env_cfg.py
    # line 264-265 of go2/), applied to the **processed** joint position
    # target inside JointPositionAction.process_actions.  The base Direct cfg
    # defaults to no clip (matches G1 / H1); override here.
    action_clip: tuple[float, float] | None = (-100.0, 100.0)

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

        # Sharron's note: armature must be non-zero for Newton numerical
        # stability.  PhysX keeps 0.  Same selection logic as Manager Go2 cfg.
        from isaaclab_tasks.utils import preset

        self.scene.robot.actuators["GO2HV"].armature = preset(default=0.0, newton=0.01, physx=0.0)

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
class Go2VelocityDirectPlayEnvCfg(Go2VelocityDirectEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
