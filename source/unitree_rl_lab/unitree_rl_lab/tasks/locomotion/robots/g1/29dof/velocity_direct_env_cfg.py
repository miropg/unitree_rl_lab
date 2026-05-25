# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow cfg for the G1-29dof velocity tracker.

We **deliberately** re-use the Manager-version cfg sub-classes (``RewardsCfg``,
``ObservationsCfg``, ``TerminationsCfg``, ``EventCfgPreset``, ``CommandsCfg``,
``CurriculumCfg``, ``RobotSceneCfg``, ``PhysicsCfg``) from
:mod:`.velocity_env_cfg`.  Sharing the cfg classes guarantees:

* identical reward weights / observation pipeline / event randomisation,
* drift-free behaviour when Sharron's Manager cfg gets a tweak,
* zero copy-paste maintenance burden.

The cfg instance is fresh per ``gym.make`` call, so any in-place mutation our
Direct env may perform (e.g. dropping noise on the critic group) does not
leak into the Manager env.
"""

from __future__ import annotations

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.direct import (
    UnitreeVelocityDirectEnvCfg,
)

# Re-export the Manager-version cfg classes; sharing them is intentional.
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


# G1-29dof joint count.  Hardcoded because :meth:`DirectRLEnv._configure_gym_env_spaces`
# runs in ``_init_sim`` *before* the env can introspect the articulation, so
# we cannot derive it from ``robot.num_joints`` at cfg-time.  The number must
# match :class:`unitree_rl_lab.assets.robots.unitree.UNITREE_G1_29DOF_CFG`.
_G1_29DOF_NUM_JOINTS = 29
# Per-step observation widths derived from each ObsTerm's tensor shape:
#   policy = base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
#          + joint_pos_rel(N) + joint_vel_rel(N) + last_action(N) = 9 + 3*N
#   critic = base_lin_vel(3) + (everything in policy) = 12 + 3*N
# The base env runs each obs term once at startup and asserts these match.
_POLICY_DIM_PER_STEP = 9 + 3 * _G1_29DOF_NUM_JOINTS  # 96
_CRITIC_DIM_PER_STEP = 12 + 3 * _G1_29DOF_NUM_JOINTS  # 99
# The PolicyCfg / CriticCfg in the Manager cfg both set history_length=5 in
# their __post_init__; if you flip that, update these constants too.
_HISTORY_LENGTH = 5


@configclass
class G1VelocityDirectEnvCfg(UnitreeVelocityDirectEnvCfg):
    """Direct-workflow cfg for ``Unitree-G1-29dof-Velocity-Flat-Direct``.

    Mirrors :class:`.velocity_env_cfg.RobotFlatEnvCfg` (the ``-Flat`` variant
    of the Manager env).  Rough-terrain Direct task is deferred until Newton
    supports terrain generators; ``-Flat`` works on both PhysX and Newton via
    Sharron's :class:`~isaaclab_tasks.utils.PresetCfg` machinery.
    """

    # ----- Direct workflow gym spaces -----
    action_space: int = _G1_29DOF_NUM_JOINTS
    observation_space: int = _POLICY_DIM_PER_STEP * _HISTORY_LENGTH  # 480
    state_space: int = _CRITIC_DIM_PER_STEP * _HISTORY_LENGTH  # 495

    # ----- Sim / scene (identical to Manager Flat cfg) -----
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    # ----- MDP cfg containers -----
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # ----- Manager-managed pieces -----
    events: EventCfgPreset = EventCfgPreset()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # ``DirectRLEnvCfg`` does not call ``super().__post_init__`` (no parent
        # post_init exists); we set sim/scene defaults here in the same shape
        # as Manager ``RobotFlatEnvCfg``.
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.contact_forces.physx.update_period = self.sim.dt
        self.scene.contact_forces.newton.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Force flat terrain (matches RobotFlatEnvCfg).  Newton does not
        # support terrain generators yet (Sharron's note in
        # validate_config of RobotEnvCfg).
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # ``terrain_levels`` curriculum requires a terrain generator; disable
        # for the flat task (matches RobotFlatEnvCfg which sets it to None).
        self.curriculum.terrain_levels = None


@configclass
class G1VelocityDirectPlayEnvCfg(G1VelocityDirectEnvCfg):
    """Light-weight play variant: 32 envs, full command range from the start."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        # Match Manager Flat-Play: bypass curriculum warm-up and use full
        # operating velocity range.
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
