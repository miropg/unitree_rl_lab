# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow base environment for Unitree velocity-tracking tasks.

This class replicates the externally observable behaviour of the Manager
version (same MDP semantics, same logged metrics, same reward magnitudes
within numeric noise) while bypassing
:class:`~isaaclab.managers.RewardManager`,
:class:`~isaaclab.managers.ObservationManager` and
:class:`~isaaclab.managers.TerminationManager`.

It still uses :class:`~isaaclab.managers.EventManager` (auto-invoked by
:class:`~isaaclab.envs.DirectRLEnv`),
:class:`~isaaclab.managers.CommandManager` (we drive ``.compute`` and
``.reset`` ourselves) and :class:`~isaaclab.managers.CurriculumManager`
(driven from ``_reset_idx``).

Why we keep ``CircularBuffer``, ``noise.func`` and the per-term ``func`` calls
intact: they are *exactly* what :class:`ObservationManager.compute_group`
runs.  Re-implementing them with a different code path would silently change
gradients and break reward-curve regression vs the Manager version
(see ``observation_manager.py`` lines 391-426).
"""

from __future__ import annotations

import torch
import warp as wp
import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from isaaclab.envs import DirectRLEnv
from isaaclab.managers import (
    CommandManager,
    CurriculumManager,
    SceneEntityCfg,
)
from isaaclab.utils import noise as noise_utils
from isaaclab.utils import modifiers
from isaaclab.utils.buffers import CircularBuffer

if TYPE_CHECKING:
    from .velocity_base_env_cfg import UnitreeVelocityDirectEnvCfg


# =============================================================================
# Shims so existing reward / observation / curriculum mdp functions that
# touch ``env.reward_manager`` or ``env.action_manager`` keep working
# under the Direct env.  Without these:
#   * ``mdp.last_action`` raises AttributeError (reads ``env.action_manager.action``);
#   * ``mdp.action_rate_l2`` raises AttributeError (reads ``.action`` and ``.prev_action``);
#   * ``mdp.curriculums.lin_vel_cmd_levels`` raises AttributeError.
# =============================================================================


class _RewardManagerShim:
    """Minimal stand-in for :class:`~isaaclab.managers.RewardManager`.

    Only the attributes the bundled curriculum
    (:func:`unitree_rl_lab.tasks.locomotion.mdp.curriculums.lin_vel_cmd_levels`)
    touches are implemented; every other access surfaces a clean
    ``AttributeError`` so missing helpers are obvious.
    """

    def __init__(self, env: "UnitreeVelocityDirectEnv") -> None:
        self._env = env

    def get_term_cfg(self, name: str):
        for n, _func, _params, weight, term_cfg in self._env._reward_terms:
            if n == name:
                return term_cfg
        raise KeyError(f"reward term '{name}' not found in Direct env")

    @property
    def _episode_sums(self) -> dict[str, torch.Tensor]:
        return self._env._episode_sums


class _SingleActionTermShim:
    """Stand-in for :class:`~isaaclab.envs.mdp.actions.JointPositionAction`.

    ``mdp.last_action`` calls ``env.action_manager.get_term(name).raw_actions``
    when an explicit term name is passed.  In every cfg in this repo there is
    exactly one action term named ``"JointPositionAction"``; this shim returns
    the env's raw action buffer (matches Manager semantics exactly because
    ``JointAction.raw_actions`` is the same buffer ``ActionManager`` stores
    in ``self._action`` for the only registered term).
    """

    def __init__(self, env: "UnitreeVelocityDirectEnv") -> None:
        self._env = env

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._env._raw_actions


class _ActionManagerShim:
    """Stand-in for :class:`~isaaclab.managers.ActionManager`.

    Implements only the surface ``mdp.last_action`` / ``mdp.action_rate_l2`` /
    ``mdp.action_l2`` need.  Both ``action`` and ``prev_action`` mirror
    :class:`~isaaclab.managers.ActionManager`'s buffers (see
    ``action_manager.py:268,273,388-389``):

    * ``self.action``     - latest raw action passed to ``_pre_physics_step``;
    * ``self.prev_action`` - raw action of the previous environment step.

    The Direct env updates these in :meth:`_pre_physics_step` (mirroring
    :meth:`ActionManager.process_action`) and zeros them in :meth:`_reset_idx`
    (mirroring :meth:`ActionManager.reset`).
    """

    def __init__(self, env: "UnitreeVelocityDirectEnv") -> None:
        self._env = env
        # The repo only ever has one action term in cfg.  If a future cfg
        # registers more, ``KeyError`` is raised by ``get_term`` so the
        # mismatch is loud.
        self._term_shim = _SingleActionTermShim(env)
        self._term_name = "JointPositionAction"

    @property
    def action(self) -> torch.Tensor:
        return self._env._raw_actions

    @property
    def prev_action(self) -> torch.Tensor:
        return self._env._prev_action

    @property
    def total_action_dim(self) -> int:
        # ``RslRlVecEnvWrapper.__init__`` line 86-87 reads this when the env
        # exposes an ``action_manager`` attribute.  Without it the wrapper
        # raises AttributeError before training starts.
        return self._env.cfg.action_space

    def get_term(self, name: str):
        if name != self._term_name:
            raise KeyError(
                f"action term '{name}' not registered on Direct env "
                f"(only '{self._term_name}' exists)"
            )
        return self._term_shim


class _TerminationManagerShim:
    """Stand-in for :class:`~isaaclab.managers.TerminationManager`.

    Why: ``mdp.is_alive`` / ``mdp.is_terminated`` / ``mdp.is_terminated_term``
    in :mod:`isaaclab.envs.mdp.rewards` read ``env.termination_manager``
    properties.  We forward to the env's own ``reset_terminated`` /
    ``reset_time_outs`` buffers (which are populated by
    :meth:`UnitreeVelocityDirectEnv._get_dones`).
    """

    def __init__(self, env: "UnitreeVelocityDirectEnv") -> None:
        self._env = env
        self._term_name_to_idx = {
            name: i for i, (name, _func, _params, _is_time_out) in enumerate(env._termination_terms)
        }

    @property
    def terminated(self) -> torch.Tensor:
        # Same buffer DirectRLEnv writes after _get_dones.
        return self._env.reset_terminated

    @property
    def time_outs(self) -> torch.Tensor:
        return self._env.reset_time_outs

    @property
    def dones(self) -> torch.Tensor:
        return self._env.reset_terminated | self._env.reset_time_outs

    def find_terms(self, name_keys: list[str] | str) -> list[str]:
        # Mirrors ManagerBase.find_terms (regex-style match against term names).
        # Used by ``mdp.is_terminated_term`` reward.
        from isaaclab.utils.string import resolve_matching_names

        names = list(self._term_name_to_idx.keys())
        if isinstance(name_keys, str):
            name_keys = [name_keys]
        _, matching = resolve_matching_names(name_keys, names, preserve_order=True)
        return matching

    def get_term(self, name: str) -> torch.Tensor:
        if self._env._term_dones is None:
            raise RuntimeError("termination terms not built yet")
        return self._env._term_dones[:, self._term_name_to_idx[name]]


# =============================================================================
# Helpers shared by reward / obs / termination dispatch
# =============================================================================


def _resolve_scene_entities(params: dict | None, scene) -> dict:
    """Resolve any :class:`~isaaclab.managers.SceneEntityCfg` in ``params``.

    We mutate ``params[k]`` in place (the cfg lives for the lifetime of the
    env) so this only runs once at startup.  Returns the dict for chaining.
    """
    if params is None:
        return {}
    for v in params.values():
        if isinstance(v, SceneEntityCfg):
            v.resolve(scene)
    return params


def _to_tensor_scale(scale, device: str) -> torch.Tensor | None:
    """Match :meth:`ObservationManager._prepare_terms` casting of ``scale``.

    ``term_cfg.scale`` may be a Python ``float`` / ``int`` / ``tuple``; we cast
    it to a ``torch.Tensor`` so element-wise ``mul_`` with the obs tensor is
    cheap.  ``None`` stays ``None`` so the dispatch can skip the multiply.
    """
    if scale is None:
        return None
    if isinstance(scale, torch.Tensor):
        return scale
    if isinstance(scale, (int, float)):
        return torch.tensor(float(scale), device=device)
    return torch.tensor(scale, dtype=torch.float, device=device)


# =============================================================================
# Main env
# =============================================================================


class UnitreeVelocityDirectEnv(DirectRLEnv):
    """Direct-workflow velocity tracker shared by G1 / Go2 / H1.

    Lifecycle:

    * ``__init__`` calls :meth:`DirectRLEnv.__init__` which creates the
      :class:`~isaaclab.scene.InteractiveScene`, runs ``_setup_scene`` (we
      override it to register our :class:`~isaaclab.assets.Articulation` and
      ``ContactSensor``), instantiates :class:`~isaaclab.managers.EventManager`
      from ``cfg.events`` and invokes ``startup`` events.
    * After ``super().__init__`` returns we pre-resolve and cache:
      reward / observation / termination terms, the per-policy
      :class:`~isaaclab.utils.buffers.CircularBuffer` history, the
      :class:`~isaaclab.managers.CommandManager` and (optional)
      :class:`~isaaclab.managers.CurriculumManager`.

    Per-step contract:

    * Our :meth:`step` override is a near-copy of :meth:`DirectRLEnv.step` with
      one re-ordering: ``command_manager.compute`` is inserted between
      ``_reset_idx`` and ``event.apply(interval)`` so the per-step ordering
      mirrors :meth:`ManagerBasedRLEnv.step`
      (``manager_based_rl_env.py`` lines 239-246).
    * ``_pre_physics_step(actions)``  - mirrors
      :meth:`ActionManager.process_action` + :class:`JointPositionAction`:
      stash ``_prev_action``, store ``_raw_actions``, compute
      ``processed = raw * scale + default_joint_pos``, then clip the
      *processed* joint position target if :attr:`action_clip` is set.
    * ``_apply_action()`` (x decimation) - ``set_joint_position_target``.
    * ``_get_dones()`` - OR-merge of termination terms; also tracks
      per-term done state for ``Episode_Termination/<name>`` logging.
    * ``_get_rewards()`` - ``sum(func * weight * step_dt)``; also
      accumulates ``_episode_sums[name]`` for curriculum and per-episode log.
    * ``_get_observations()`` - computes obs term-by-term with the same
      pipeline as :meth:`ObservationManager.compute_group`.  Does **not**
      call ``command_manager.compute`` (that runs in :meth:`step`).

    Reset contract:

    * ``_reset_idx(env_ids)`` - matches the order of
      :meth:`ManagerBasedRLEnv._reset_idx` which is timing-sensitive
      (``curriculum.compute`` reads ``_episode_sums`` *before* we zero them;
      ``Episode_Reward`` / ``Episode_Termination`` extras drained the same
      way :class:`RewardManager` / :class:`TerminationManager` do it).
    """

    cfg: "UnitreeVelocityDirectEnvCfg"

    # =========================================================================
    # __init__ / scene setup
    # =========================================================================

    def __init__(self, cfg: "UnitreeVelocityDirectEnvCfg", render_mode: str | None = None, **kwargs):
        # super().__init__ runs:
        #   1. SimulationContext + _init_sim
        #      -> _setup_scene() (our override registers self.robot)
        #      -> EventManager(cfg.events)  + apply(startup)
        #      -> _configure_gym_env_spaces()  (uses cfg.observation_space etc.)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.robot = self.scene["robot"]

        # ----- Action buffers -----
        # Match ``ActionManager.process_action`` + ``JointPositionAction``:
        # both ``_raw_actions`` (= ActionManager.action) and ``_prev_action``
        # (= ActionManager.prev_action) are needed for action_rate_l2 reward
        # and last_action observation.  See action_manager.py lines 268, 273,
        # 388-389.
        self._raw_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_action = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # ``data.default_joint_pos`` is a wp.array even on PhysX backend in
        # IsaacLab 3.0 (unified storage); ``wp.to_torch`` is zero-copy.
        self._default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()
        # Resolve the per-joint action clip into a (1, num_joints, 2) tensor
        # so it broadcasts over envs (matches JointAction._clip layout in
        # joint_actions.py:155-163 conceptually; we collapse the per-joint
        # dim to 1 because every cfg in this repo uses the same (min, max)
        # for every joint).
        if self.cfg.action_clip is not None:
            self._action_clip_min = float(self.cfg.action_clip[0])
            self._action_clip_max = float(self.cfg.action_clip[1])
        else:
            self._action_clip_min = None
            self._action_clip_max = None

        # ``last_action`` observations are probed while building observation
        # terms, so the action shim must exist before `_build_obs_terms`.
        self.action_manager = _ActionManagerShim(self)

        # ----- Command + curriculum managers -----
        # CommandManager honours all our existing ``UniformLevelVelocityCommandCfg``
        # behaviour (resample timer, heading, standing envs).  Reset/compute
        # are driven from our hooks below.
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        if self.cfg.curriculum is not None:
            self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
            print("[INFO] Curriculum Manager: ", self.curriculum_manager)
        else:
            self.curriculum_manager = None

        # ----- Reward / observation / termination dispatch tables -----
        # All reads of ``cfg`` MUST come through these caches; the env class
        # may not poke into ``cfg`` on the hot path.
        self._reward_terms = self._build_reward_terms(self.cfg.rewards)
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name, *_ in self._reward_terms
        }
        self._reward_buf = torch.zeros(self.num_envs, device=self.device)

        self._policy_cfg = self.cfg.observations.policy
        self._critic_cfg = self.cfg.observations.critic
        self._obs_class_instances: list[modifiers.ModifierBase | noise_utils.NoiseModel] = []
        self._policy_obs_terms, self._policy_history_buf = self._build_obs_terms(self._policy_cfg)
        self._critic_obs_terms, self._critic_history_buf = self._build_obs_terms(self._critic_cfg)

        self._termination_terms = self._build_termination_terms(self.cfg.terminations)

        # ----- Termination per-term tracking (for Episode_Termination logs) -----
        # Manager-side ``TerminationManager`` keeps these to surface
        # ``Episode_Termination/<name>`` to extras.  We replicate the same
        # buffer shape so logged values are bit-for-bit comparable.
        if self._termination_terms:
            self._term_dones = torch.zeros(
                self.num_envs, len(self._termination_terms), dtype=torch.bool, device=self.device
            )
            self._last_episode_dones = torch.zeros_like(self._term_dones)
        else:
            self._term_dones = None
            self._last_episode_dones = None

        # ----- Compatibility shims for upstream mdp helpers -----
        # ``mdp.curriculums.lin_vel_cmd_levels`` reads ``env.reward_manager``;
        # ``mdp.last_action`` / ``mdp.action_rate_l2`` / ``mdp.action_l2`` read
        # ``env.action_manager``;
        # ``mdp.is_alive`` / ``mdp.is_terminated`` read ``env.termination_manager``.
        # Without these shims the env raises AttributeError on the first reset.
        # Termination shim must come AFTER ``_term_dones`` is allocated above.
        self.reward_manager = _RewardManagerShim(self)
        self.termination_manager = _TerminationManagerShim(self)

        # ----- Sanity print -----
        # We probe each obs term once by running it on the current state to
        # report the per-step dim; cfg-declared ``observation_space`` /
        # ``state_space`` should match these values * history_length.
        with torch.inference_mode():
            policy_per_step = sum(
                int(func(self, **params).shape[-1])
                for _name, func, params, _s, _n, _m, _t in self._policy_obs_terms
            )
            critic_per_step = sum(
                int(func(self, **params).shape[-1])
                for _name, func, params, _s, _n, _m, _t in self._critic_obs_terms
            )
        policy_history = self._policy_cfg.history_length or 1
        critic_history = self._critic_cfg.history_length or 1
        print(
            f"[INFO] {self.__class__.__name__} ready: "
            f"reward_terms={len(self._reward_terms)}, "
            f"policy_obs_dim={policy_per_step} x history={policy_history} "
            f"-> obs_space={policy_per_step * policy_history} (cfg={self.cfg.observation_space}), "
            f"critic_obs_dim={critic_per_step} x history={critic_history} "
            f"-> state_space={critic_per_step * critic_history} (cfg={self.cfg.state_space}), "
            f"termination_terms={len(self._termination_terms)}"
        )
        if policy_per_step * policy_history != self.cfg.observation_space:
            raise RuntimeError(
                f"cfg.observation_space ({self.cfg.observation_space}) does not match measured "
                f"policy obs dim ({policy_per_step}) x history ({policy_history}) = "
                f"{policy_per_step * policy_history}.  Update the cfg's __post_init__."
            )
        if critic_per_step * critic_history != self.cfg.state_space:
            raise RuntimeError(
                f"cfg.state_space ({self.cfg.state_space}) does not match measured "
                f"critic obs dim ({critic_per_step}) x history ({critic_history}) = "
                f"{critic_per_step * critic_history}.  Update the cfg's __post_init__."
            )

        # Optional NVTX instrumentation; zero-cost in normal training.
        self._maybe_apply_nvtx_instrumentation()

    def _maybe_apply_nvtx_instrumentation(self) -> None:
        """Hand off to ``nvtx_profiling.wrap_direct_env_terms`` if available.

        The NVTX helper lives in ``agent_tmp/nvtx/`` and is only on
        ``sys.path`` when running through ``agent_tmp/profile_scripts/profile.sh``.
        We tolerate ``ImportError`` so normal training has zero overhead and no
        new dependency.
        """
        try:
            import nvtx_profiling  # type: ignore[import-not-found]
        except ImportError:
            return
        fn = getattr(nvtx_profiling, "wrap_direct_env_terms", None)
        if fn is not None:
            try:
                fn(self)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[NVTX] Direct env wrapping failed: {exc}", flush=True)

    def _setup_scene(self) -> None:
        """Per-cfg robot/sensor/terrain instantiation.

        Called by :meth:`DirectRLEnv._init_sim` before any sim warm-up.  All
        the heavy lifting (terrain importer, sky light, contact sensor, robot
        articulation) is already declared in ``cfg.scene``; the
        :class:`~isaaclab.scene.InteractiveScene` constructor honours those
        cfgs automatically.  We do **nothing** in this hook so the cfg-only
        contract is preserved.

        If a subclass needs to mutate scene at runtime (e.g. add a
        :class:`~isaaclab.sensors.RayCaster` based on a flag), override this
        hook and remember to call ``self.scene.articulations[...] = ...``.
        """
        # Intentionally empty - InteractiveScene already populates
        # ``self.scene`` from ``cfg.scene`` before this hook runs.

    # =========================================================================
    # Action processing (replaces ActionManager)
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Replicate the **two** semantics ActionManager + JointPositionAction
        # wire together (action_manager.py:388-389 + joint_actions.py:170-179):
        #   1) ActionManager.process_action: prev_action <- action; action <- new
        #   2) JointPositionAction.process_actions:
        #        raw_actions <- action
        #        processed = raw * scale + default_joint_pos  (= ``offset``)
        #        if cfg.clip is not None: processed = clamp(processed, ...)
        # The clip is on the *processed* (joint-target-space) value, NOT on
        # the raw action.  Doing it on raw would shift the effective range
        # to ``[clip * scale + default, clip * scale + default]`` and break
        # behavioural parity with the Manager version.
        self._prev_action.copy_(self._raw_actions)
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(self._raw_actions * self.cfg.action_scale + self._default_joint_pos)
        if self._action_clip_min is not None:
            self._processed_actions.clamp_(min=self._action_clip_min, max=self._action_clip_max)

    def _apply_action(self) -> None:
        # Called once per decimation by DirectRLEnv.step (4x for the standard
        # ``decimation=4`` we inherit from Manager cfgs).
        # Match JointPositionAction.apply_actions exactly.  Even when all joints
        # are targeted, Manager goes through the indexed setter with
        # ``joint_ids=slice(None)``; keeping the same API path reduces one
        # possible source of Manager/Direct semantic drift.
        self.robot.set_joint_position_target_index(target=self._processed_actions, joint_ids=slice(None))

    # =========================================================================
    # Reward dispatch (replaces RewardManager)
    # =========================================================================

    def _get_rewards(self) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        step_dt = self.step_dt
        for name, func, params, weight, _term_cfg in self._reward_terms:
            if weight == 0.0:
                continue
            # Same formula as :meth:`RewardManager.compute`:
            # value_i = func_i(env, **params_i) * weight_i * dt
            value = func(self, **params) * weight * step_dt
            self._reward_buf += value
            self._episode_sums[name] += value
        return self._reward_buf

    # =========================================================================
    # Observation dispatch (replaces ObservationManager)
    # =========================================================================

    def _get_observations(self, update_history: bool = False) -> dict[str, torch.Tensor]:
        # ``command_manager.compute(step_dt)`` is driven by our :meth:`step`
        # override BEFORE the interval-event apply step, matching
        # :meth:`ManagerBasedRLEnv.step` (manager_based_rl_env.py:239-246).
        # Do NOT add a ``compute`` call here, otherwise the timer is
        # decremented twice per env step.
        return {
            "policy": self._compute_obs_group(
                self._policy_cfg, self._policy_obs_terms, self._policy_history_buf, update_history
            ),
            "critic": self._compute_obs_group(
                self._critic_cfg, self._critic_obs_terms, self._critic_history_buf, update_history
            ),
        }

    def _compute_obs_group(
        self, group_cfg, term_specs, history_bufs: dict[str, CircularBuffer], update_history: bool
    ) -> torch.Tensor:
        """Mirror of :meth:`ObservationManager.compute_group` (term pipeline)."""
        outputs: list[torch.Tensor] = []
        for name, func, params, scale, noise_cfg, modifier_cfgs, _term_cfg in term_specs:
            obs = func(self, **params).clone()
            # Pipeline order matches ObservationManager (line 393-407):
            #   1) func, 2) modifiers, 3) noise, 4) clip, 5) scale, then history append.
            # Don't reorder; some training runs are sensitive to the noise
            # being on the *raw* unscaled signal.
            if modifier_cfgs is not None:
                for modifier_cfg in modifier_cfgs:
                    obs = modifier_cfg.func(obs, **modifier_cfg.params)
            if noise_cfg is not None:
                if isinstance(noise_cfg, noise_utils.NoiseCfg):
                    obs = noise_cfg.func(obs, noise_cfg)
                else:
                    obs = noise_cfg.func(obs)
            if _term_cfg.clip is not None:
                obs = obs.clip_(min=_term_cfg.clip[0], max=_term_cfg.clip[1])
            if scale is not None:
                obs = obs.mul_(scale)
            buf = history_bufs.get(name)
            if buf is not None:
                if update_history or buf._buffer is None:
                    buf.append(obs)
                if _term_cfg.flatten_history_dim:
                    outputs.append(buf.buffer.reshape(self.num_envs, -1))
                else:
                    outputs.append(buf.buffer)
            else:
                outputs.append(obs)

        if group_cfg.concatenate_terms:
            return torch.cat(outputs, dim=-1)
        # Non-concatenated groups are not used in any current cfg, but support
        # them for symmetry with ObservationManager.compute_group.
        return outputs

    # =========================================================================
    # Termination dispatch (replaces TerminationManager)
    # =========================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = torch.zeros_like(terminated)
        if self._term_dones is not None:
            self._term_dones[:] = False
        for idx, (_name, func, params, is_time_out) in enumerate(self._termination_terms):
            value = func(self, **params)
            if is_time_out:
                time_out |= value
            else:
                terminated |= value
            if self._term_dones is not None:
                self._term_dones[:, idx] = value
        # Track which term fired in the last completed episode for any env that
        # did terminate this step.  Mirrors TerminationManager.compute lines
        # 178-180 (termination_manager.py) so Episode_Termination/<name> log
        # matches the Manager version exactly.
        if self._last_episode_dones is not None:
            rows = self._term_dones.any(dim=1).nonzero(as_tuple=True)[0]
            if rows.numel() > 0:
                self._last_episode_dones[rows] = self._term_dones[rows]
        return terminated, time_out

    # =========================================================================
    # Step override: insert ``command_manager.compute`` BEFORE the
    # interval-event apply so the per-step ordering matches
    # :meth:`ManagerBasedRLEnv.step` (manager_based_rl_env.py:239-246).
    # The interval event ``push_robot`` perturbs the robot state; running it
    # before ``command_manager.compute`` would let the command term observe
    # the perturbed velocity, which mismatches the Manager-version reward
    # / observation values for that step.
    # =========================================================================

    def step(self, action: torch.Tensor):  # noqa: D401 - matches DirectRLEnv API
        # The body below is a near-copy of :meth:`DirectRLEnv.step` (lines
        # 370-455) with one re-ordering: ``command_manager.compute`` is
        # inserted between ``_reset_idx`` and ``event.apply(interval)``.
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        self._pre_physics_step(action)

        is_rendering = self.sim.is_rendering
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # Manager-version ordering: command BEFORE interval event.
        self.command_manager.compute(dt=self.step_dt)

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations(update_history=True)

        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset all environments and fill observation history like ManagerBasedRLEnv.reset."""
        if seed is not None:
            self.seed(seed)

        indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self._reset_idx(indices)

        self.scene.write_data_to_sim()
        self.sim.forward()

        if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        if self.cfg.wait_for_textures and self.has_rtx_sensors:
            if hasattr(self.sim.physics_manager, "assets_loading"):
                while self.sim.physics_manager.assets_loading():
                    self.sim.render()

        self.obs_buf = self._get_observations(update_history=True)
        return self.obs_buf, self.extras

    # =========================================================================
    # Reset path
    # =========================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        # Order mirrors :meth:`ManagerBasedRLEnv._reset_idx` *step-by-step*
        # (manager_based_rl_env.py:331-376) so the relative ordering of
        # curriculum / event / reward / observation / command / termination
        # resets is preserved.  This matters when:
        #   - curriculum.compute reads ``_episode_sums`` BEFORE reward.reset
        #     zeros them (lin_vel_cmd_levels uses this);
        #   - reward.reset's class-term ``.reset`` runs AFTER scene+event
        #     reset (so state-dependent terms see the freshly-reset state);
        #   - command.reset must run AFTER event.reset(reset) so the new
        #     command is consistent with the freshly-randomised pose.
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # 1) Curriculum.compute - needs ``_episode_sums`` of the just-completed
        #    episode.  Output is intentionally NOT logged because Manager also
        #    discards it (manager_based_rl_env.py:337-338 vs 349).
        if self.curriculum_manager is not None:
            self.curriculum_manager.compute(env_ids=env_ids)

        # 2) Scene reset (clears scene's per-env buffers + calls
        #    ``robot.reset(env_ids)`` internally - see
        #    interactive_scene.py:434-444).
        self.scene.reset(env_ids)

        # 3) Event apply(reset) - randomise mass / pose / joint state.
        if self.cfg.events is not None and "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # 4) Initialise extras["log"] for this reset.  Manager does this here
        #    (manager_based_rl_env.py:349), AFTER curriculum.compute, which
        #    discards anything curriculum.compute wrote into the previous
        #    iteration's log.
        self.extras["log"] = dict()
        log_dict = self.extras["log"]

        # 5) ObservationManager.reset equivalent: clear history buffers.
        for buf in self._policy_history_buf.values():
            buf.reset(batch_ids=env_ids)
        for buf in self._critic_history_buf.values():
            buf.reset(batch_ids=env_ids)
        for class_instance in self._obs_class_instances:
            class_instance.reset(env_ids=env_ids)

        # 6) ActionManager.reset equivalent: clear raw / prev action buffers
        #    (action_manager.py:367-368).  ``processed_actions`` is reset to
        #    ``default_joint_pos`` so the next ``set_joint_position_target``
        #    after reset commands the home pose, not zero.
        self._raw_actions[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._processed_actions[env_ids] = self._default_joint_pos[env_ids]

        # 7) RewardManager.reset equivalent: drain Episode_Reward, zero
        #    _episode_sums, then call class-term .reset
        #    (reward_manager.py:115-127).
        for name, _func, _params, _weight, term_cfg in self._reward_terms:
            episodic_sum_avg = torch.mean(self._episode_sums[name][env_ids])
            log_dict[f"Episode_Reward/{name}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[name][env_ids] = 0.0
        for _name, _func, _params, _weight, term_cfg in self._reward_terms:
            func = term_cfg.func
            if hasattr(func, "reset") and callable(getattr(func, "reset")):
                func.reset(env_ids=env_ids)

        # 8) CurriculumManager.reset (logs Curriculum/<term>).
        if self.curriculum_manager is not None:
            info = self.curriculum_manager.reset(env_ids)
            if info:
                log_dict.update(info)

        # 9) CommandManager.reset (resamples + logs Metrics/<term>/<metric>).
        info = self.command_manager.reset(env_ids)
        if info:
            log_dict.update(info)

        # 10) EventManager.reset (resets interval timers).
        if self.cfg.events is not None:
            info = self.event_manager.reset(env_ids)
            if info:
                log_dict.update(info)

        # 11) TerminationManager.reset equivalent: log Episode_Termination/<name>
        #     and call class-term .reset (termination_manager.py:129-152).
        if self._last_episode_dones is not None:
            done_stats = self._last_episode_dones.float().mean(dim=0)
            for idx, (name, _func, _params, _is_time_out) in enumerate(self._termination_terms):
                log_dict[f"Episode_Termination/{name}"] = done_stats[idx].item()
                term_cfg = getattr(self.cfg.terminations, name, None)
                if term_cfg is not None and hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                    term_cfg.func.reset(env_ids=env_ids)

        # 12) Reset noise models (mirrors DirectRLEnv._reset_idx 626-630).
        if self.cfg.action_noise_model is not None:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model is not None:
            self._observation_noise_model.reset(env_ids)

        # 13) Final: zero episode-length buffer (matches Manager).
        self.episode_length_buf[env_ids] = 0

    # =========================================================================
    # Cfg parsing helpers (called once from __init__)
    # =========================================================================

    def _build_reward_terms(self, rewards_cfg) -> list[tuple]:
        """Return ``[(name, func, resolved_params, weight, term_cfg), ...]`` in declaration order."""
        terms = []
        for name in rewards_cfg.__dataclass_fields__:
            term_cfg = getattr(rewards_cfg, name, None)
            if term_cfg is None:
                continue
            params = _resolve_scene_entities(dict(term_cfg.params or {}), self.scene)
            terms.append((name, term_cfg.func, params, float(term_cfg.weight), term_cfg))
            print(f"[INFO]   reward.{name}: {term_cfg.func.__name__} weight={term_cfg.weight}")
        return terms

    def _build_obs_terms(self, group_cfg) -> tuple[list[tuple], dict[str, CircularBuffer]]:
        """Return ``([term_specs], {name: CircularBuffer})``.

        ``term_spec = (name, func, params, scale_tensor, noise_cfg, modifier_cfgs, term_cfg)``.

        Apply the same group-level overrides as
        :meth:`ObservationManager._prepare_terms`:

        * if ``group.enable_corruption is False`` -> drop ``noise``;
        * if ``group.history_length`` is set -> override ``term.history_length``
          and ``term.flatten_history_dim`` with the group value (so a group
          with ``history_length=5`` applies to every term, matching Manager).
        """
        terms: list[tuple] = []
        history_bufs: dict[str, CircularBuffer] = {}
        history_length_override = group_cfg.history_length
        flatten_override = group_cfg.flatten_history_dim
        enable_corruption = group_cfg.enable_corruption

        for name in group_cfg.__dataclass_fields__:
            if name in (
                "enable_corruption",
                "concatenate_terms",
                "history_length",
                "flatten_history_dim",
                "concatenate_dim",
            ):
                continue
            term_cfg = getattr(group_cfg, name, None)
            if term_cfg is None:
                continue

            # Apply group-level overrides (mirrors ObservationManager
            # ``_prepare_terms`` lines 548-553).
            if not enable_corruption:
                term_cfg.noise = None
            if history_length_override is not None and history_length_override > 0:
                term_cfg.history_length = history_length_override
                term_cfg.flatten_history_dim = flatten_override

            params = _resolve_scene_entities(dict(term_cfg.params or {}), self.scene)
            scale_t = _to_tensor_scale(term_cfg.scale, self.device)

            obs_dims = tuple(term_cfg.func(self, **params).shape)

            if term_cfg.modifiers is not None:
                for modifier_cfg in term_cfg.modifiers:
                    if not isinstance(modifier_cfg, modifiers.ModifierCfg):
                        raise TypeError(
                            f"Modifier configuration for observation term '{name}' is not a ModifierCfg."
                            f" Received: '{type(modifier_cfg)}'."
                        )
                    if inspect.isclass(modifier_cfg.func):
                        modifier_cfg.func = modifier_cfg.func(cfg=modifier_cfg, data_dim=obs_dims, device=self.device)
                        if not isinstance(modifier_cfg.func, modifiers.ModifierBase):
                            raise TypeError(
                                f"Modifier function '{modifier_cfg.func}' for observation term '{name}' is not a"
                                f" ModifierBase instance. Received: '{type(modifier_cfg.func)}'."
                            )
                        self._obs_class_instances.append(modifier_cfg.func)
                    if not callable(modifier_cfg.func):
                        raise AttributeError(
                            f"Modifier '{modifier_cfg}' of observation term '{name}' is not callable."
                            f" Received: {modifier_cfg.func}"
                        )

            noise_cfg = None
            if isinstance(term_cfg.noise, noise_utils.NoiseCfg):
                noise_cfg = term_cfg.noise
            elif isinstance(term_cfg.noise, noise_utils.NoiseModelCfg):
                term_cfg.noise.func = term_cfg.noise.class_type(
                    term_cfg.noise, num_envs=self.num_envs, device=self.device
                )
                if not isinstance(term_cfg.noise.func, noise_utils.NoiseModel):
                    raise TypeError(
                        f"Noise model for observation term '{name}' is not a NoiseModel instance."
                        f" Received: '{type(term_cfg.noise.func)}'."
                    )
                self._obs_class_instances.append(term_cfg.noise.func)
                noise_cfg = term_cfg.noise

            terms.append((name, term_cfg.func, params, scale_t, noise_cfg, term_cfg.modifiers, term_cfg))

            # Per-term CircularBuffer when history is enabled.  ``max_len`` of
            # the buffer is the history depth; the env writes the latest obs
            # at each call to ``_compute_obs_group`` and reads ``buf.buffer``
            # which has shape ``(num_envs, history_length, *obs_shape)``.
            if term_cfg.history_length and term_cfg.history_length > 0:
                history_bufs[name] = CircularBuffer(
                    max_len=term_cfg.history_length, batch_size=self.num_envs, device=self.device
                )

            print(
                f"[INFO]   obs.{name}: {term_cfg.func.__name__}"
                f"{' history=' + str(term_cfg.history_length) if term_cfg.history_length else ''}"
            )
        return terms, history_bufs

    def _build_termination_terms(self, term_cfg_obj) -> list[tuple]:
        terms = []
        for name in term_cfg_obj.__dataclass_fields__:
            term_cfg = getattr(term_cfg_obj, name, None)
            if term_cfg is None:
                continue
            params = _resolve_scene_entities(dict(term_cfg.params or {}), self.scene)
            terms.append((name, term_cfg.func, params, bool(term_cfg.time_out)))
            print(f"[INFO]   termination.{name}: {term_cfg.func.__name__} time_out={term_cfg.time_out}")
        return terms
