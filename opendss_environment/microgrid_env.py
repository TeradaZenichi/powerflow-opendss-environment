"""Gymnasium environment and configuration, with caller-owned scenario data."""
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import timedelta

import gymnasium as gym
from pu_normalizer import PUNormalizer

from opendss_environment.backend import OpenDSSDirectBackend
from opendss_environment.actions import prepare_action
from opendss_environment.controller_actions import ControllerActions, compose_actions
from opendss_environment.data import load_data, episode_data
from opendss_environment.states import (
    build_controller_observation, build_local_observation,
    build_named_observation, build_state,
)
from opendss_environment.rewards import Reward
from opendss_environment.diagnostics import electrical_diagnostics
from opendss_environment.step_state import StepCheckpoint
from opendss_environment.simulation import (
    _simulation_setup,
    _update_snapshot_powers,
    prepare_pre_action_observation,
    solve_power_flow,
)


@dataclass(frozen=True)
class EnvironmentConfig:
    case_path: object
    episode_steps: int | None = None
    state_functions: tuple = ()
    num_episodes: int = 1
    start_episode: int = 0
    reward_function: object = Reward.minimize_cost
    start_step: int = 0
    observe_evse: bool = True

    def __post_init__(self):
        if self.episode_steps is not None and (
            isinstance(self.episode_steps, bool) or not isinstance(self.episode_steps, int)
            or self.episode_steps < 1
        ):
            raise ValueError("episode_steps must be a positive integer or None for the complete case")
        for name, minimum in (("num_episodes", 1), ("start_episode", 0), ("start_step", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                kind = "positive" if minimum else "nonnegative"
                raise ValueError(f"{name} must be a {kind} integer")


class MicrogridEnv(gym.Env):

    def __init__(self, config, *, data=None):
        super().__init__()
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("MicrogridEnv requires an EnvironmentConfig")

        self.case_path = config.case_path
        self.config = config
        self.episode_steps = config.episode_steps
        self.start_episode = config.start_episode
        self.num_episodes = config.num_episodes
        self.state_functions = tuple(config.state_functions)
        self.reward_function = config.reward_function
        self.end_episode = config.start_episode + config.num_episodes

        self.data = load_data(config.case_path) if data is None else copy(data)
        if config.start_step >= self.data["steps"]:
            raise ValueError("start_step must identify an interval in the dataset")
        self.data["start_step"] = config.start_step
        self.case_path = self.data["case_path"]
        self.config_path = self.data["config_path"]
        self.network = self.data["network"]
        if self.episode_steps is None:
            self.episode_steps = self.data["steps"] - config.start_step
        self.normalizer = PUNormalizer(self.data["base_kva"], self.data["base_kv"])

        self.dss = OpenDSSDirectBackend()
        self._clear_episode_state()
        self._needs_reset = True

        self.load_episode_data(self.start_episode)
        _simulation_setup(self)

    def load_episode_data(self, episode_idx, *, start_step=None, episode_steps=None):
        steps = self.episode_steps if episode_steps is None else episode_steps
        episode = episode_data(self.data, episode_idx, steps, start_step=start_step)
        self.episode_steps = steps

        for name in (
            "dt", "steps", "timestamps", "grid", "bess_list", "pv_list",
            "evse_list", "load_list", "results",
        ):
            setattr(self, name, episode[name])

    def controller_actions(self, devices, *, seed=None):
        available = {"bess": self.bess_list, "pv": self.pv_list,
                     "evse": [device for device in self.evse_list if device.controlled]}
        if set(devices) - set(available):
            raise ValueError("Controller groups must be bess, pv or evse")
        selected = {group: [device for device in available[group] if device.id in ids]
                    for group, ids in devices.items()}
        if any(set(devices[group]) != {device.id for device in items} for group, items in selected.items()):
            raise ValueError("Unknown or noncontrollable device in controller scope")
        return ControllerActions(selected, self.data["base_kva"], phases=self.data["phases"], seed=seed)

    def compose_actions(self, *commands, observation=None):
        """Complete device commands using an optional current observation."""
        if observation is not None and (self._needs_reset or self.idx >= self.steps):
            raise RuntimeError("Action composition requires an active episode")
        return compose_actions(self, commands, observation=observation)

    def _clear_episode_state(self):
        self.idx = 0
        self.current_cost = 0.0
        self.episode_reward = 0.0
        self.reward_objectives = {}
        groups = {"bess": {}, "pv": {}}
        if getattr(self, "evse_list", []):
            groups["evse"] = {}
        self.current_device_measurements = deepcopy(groups)
        self.previous_device_measurements = deepcopy(groups)
        self.current_branch_measurements = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._needs_reset = True

        options = options or {}
        episode_idx = options.get("episode_idx", 0)

        if episode_idx >= self.num_episodes:
            episode_idx = 0

        default_steps = self.config.episode_steps or self.data["steps"] - self.data["start_step"]
        start = options.get("start_step", self.data["start_step"] +
                            (self.start_episode + episode_idx) * default_steps)
        steps = options.get("episode_steps", self.config.episode_steps)
        window = EnvironmentConfig(self.case_path, self.data["steps"] - start if steps is None else steps,
                                   start_step=start)
        self.load_episode_data(0, start_step=window.start_step, episode_steps=window.episode_steps)

        self._clear_episode_state()

        _simulation_setup(self)

        self.results.initialize_voltages(self.dss.circuit.buses_names, self.steps)

        prepare_pre_action_observation(self)
        self._needs_reset = False

        observation = self.observe()
        state = build_state(self, self.state_functions) if self.state_functions else observation

        return state, {"observation": deepcopy(observation), "start_step": start, "episode_steps": self.steps}

    def step(self, action=None, *, full_info=True):
        """Advance one interval. With full_info=False, info contains only rewards."""
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping, including after an execution failure")
        if self.idx >= self.steps:
            raise RuntimeError("The episode has terminated; call reset()")
        prepared = prepare_action(self, action)
        checkpoint = StepCheckpoint(self)
        pending = self._copy_for_step()
        try:
            result = pending._run_step(prepared, full_info=full_info)
        except BaseException:
            checkpoint.restore()
            # OpenDSS may have changed; reset rebuilds it before another attempt.
            self._needs_reset = True
            raise
        self._commit_step(pending)
        return result

    def _copy_for_step(self):
        """Stage measurements/index; a checkpoint guards device/history updates."""
        pending = copy(self)
        pending.reward_objectives = deepcopy(self.reward_objectives)
        return pending

    def _run_step(self, prepared, *, full_info):
        observation = self.observe() if full_info else None
        objectives = deepcopy(self.reward_objectives)
        applied_action = _update_snapshot_powers(self, prepared.physical)
        grid_kw, grid_kvar, cost = solve_power_flow(self, applied_action)

        self.current_cost = cost
        diagnostics = electrical_diagnostics(self)
        self.results.diagnostics.append({
            "within_limits": diagnostics["within_limits"],
            **{name: len(diagnostics[name]) for name in ("voltage_violations", "branch_overloads", "grid_excess_pu")},
            "unrated_branch_terminals": diagnostics["unrated_branch_terminals"],
        })
        evaluator = self.reward_function if isinstance(self.reward_function, Reward) else Reward()
        rewards = evaluator.evaluate(self, objectives)
        reward = rewards["global"] if isinstance(self.reward_function, Reward) else self.reward_function(self)
        rewards["global"] = reward
        self.episode_reward += reward
        self.idx += 1

        terminated = self.idx >= self.steps

        self.previous_device_measurements = self.current_device_measurements
        info = {"rewards": rewards}
        if full_info:
            info = {
                "units": "pu",
                "base": self.normalizer.as_dict()["base"],
                "network": self.network.as_dict(),
                "timestamp": self.timestamps.iloc[self.idx - 1],
                "interval_end": (self.timestamps.iloc[self.idx - 1] + timedelta(hours=self.dt)).isoformat(),
                "cost": cost,
                "costs": rewards["costs"],
                "reward": reward,
                "rewards": rewards,
                "grid_kw": grid_kw,
                "grid_kvar": grid_kvar,
                "grid_import_kw": max(-grid_kw, 0.0),
                "grid_export_kw": max(grid_kw, 0.0),
                "bus_voltages_pu": self.current_bus_voltages_pu,
                "bus_voltages_v": self.current_bus_snapshot.volts,
                "bus_angles_deg": self.current_bus_angles_deg,
                "branch_measurements": self.current_branch_measurements,
                "diagnostics": diagnostics,
                "requested_action": prepared.requested,
                "executed_action": applied_action,
                "device_measurements": self.current_device_measurements,
                "observation": observation,  # Compatibility key for the pre-action observation.
                "observation_before": deepcopy(observation),
            }

        if terminated:
            state = None
        else:
            prepare_pre_action_observation(self)
            state = build_state(self, self.state_functions) if self.state_functions else self.observe()

        info = self.normalizer.normalize(info)
        self.normalizer.require_pu(info)
        return state, reward, terminated, False, info

    def _commit_step(self, pending):
        for name in ("idx", "current_cost", "episode_reward", "current_grid_kw", "current_grid_kvar",
                     "current_bus_snapshot", "current_bus_voltages_pu", "current_bus_angles_deg", "current_voltages_pu",
                     "current_device_measurements", "current_branch_measurements", "previous_device_measurements",
                     "pre_action_branch_measurements"):
            setattr(self, name, getattr(pending, name))

    def observe(self):
        if self._needs_reset:
            raise RuntimeError("Call reset() before observing, including after an execution failure")
        return build_named_observation(self)

    def observe_local(self, kind, device_id, *, include=None, observation=None):
        """Return a selected device view without solving or advancing time."""
        if self._needs_reset:
            raise RuntimeError("Call reset() before observing, including after an execution failure")
        return build_local_observation(self, kind, device_id, observation, include)

    def observe_controller(
        self, bus, observed_components, *, extra_features=(), observation=None,
    ):
        """Return one nodal view with an explicit component allowlist."""
        if self._needs_reset:
            raise RuntimeError("Call reset() before observing, including after an execution failure")
        return build_controller_observation(
            self, bus, observed_components, observation, extra_features,
        )

    def get_episode_results(self):
        results = {
            "dt": self.dt,
            "steps": self.steps,
            "timestamps": self.timestamps,
            "grid": self.grid,
            "bess_list": self.bess_list,
            "pv_list": self.pv_list,
            "load_list": self.load_list,
            "results": self.results,
            "episode_reward": self.episode_reward,
        }
        if self.evse_list:
            results["evse_list"] = self.evse_list
        return results
