"""Gymnasium environment and configuration, with caller-owned scenario data."""
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import timedelta

import gymnasium as gym
import numpy as np
from pu_normalizer import PUNormalizer

from opendss_env.backend import OpenDSSDirectBackend
from opendss_env.actions import prepare_action
from opendss_env.data import load_data, episode_data
from opendss_env.states import build_named_observation, build_state
from opendss_env.rewards import minimize_cost
from opendss_env.simulation import (
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
    reward_function: object = minimize_cost

    def __post_init__(self):
        if self.episode_steps is not None and (
            isinstance(self.episode_steps, bool) or not isinstance(self.episode_steps, int)
            or self.episode_steps < 1
        ):
            raise ValueError("episode_steps must be a positive integer or None for the complete case")
        if isinstance(self.num_episodes, bool) or not isinstance(self.num_episodes, int) or self.num_episodes < 1:
            raise ValueError("num_episodes must be a positive integer")
        if isinstance(self.start_episode, bool) or not isinstance(self.start_episode, int) or self.start_episode < 0:
            raise ValueError("start_episode must be a nonnegative integer")


class MicrogridEnv(gym.Env):

    def __init__(self, config):
        super().__init__()
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("MicrogridEnv requires an EnvironmentConfig")

        self.case_path = config.case_path
        self.episode_steps = config.episode_steps
        self.start_episode = config.start_episode
        self.num_episodes = config.num_episodes
        self.state_functions = tuple(config.state_functions)
        self.reward_function = config.reward_function
        self.end_episode = config.start_episode + config.num_episodes

        self.data = load_data(config.case_path)
        self.case_path = self.data["case_path"]
        self.config_path = self.data["config_path"]
        self.network = self.data["network"]
        if self.episode_steps is None:
            self.episode_steps = self.data["steps"]
        self.normalizer = PUNormalizer(self.data["base_kva"], self.data["base_kv"])

        self.dss = OpenDSSDirectBackend()
        self.current_cost = 0.0
        self.current_device_measurements = {"bess": {}, "pv": {}}
        self.previous_device_measurements = {"bess": {}, "pv": {}}
        self.episode_reward = 0.0
        self._needs_reset = True

        self.load_episode_data(self.start_episode)
        _simulation_setup(self)

    def load_episode_data(self, episode_idx):
        episode = episode_data(self.data, episode_idx, self.episode_steps)

        self.dt = episode["dt"]
        self.steps = episode["steps"]
        self.timestamps = episode["timestamps"]
        self.grid = episode["grid"]
        self.bess_list = episode["bess_list"]
        self.pv_list = episode["pv_list"]
        self.load_list = episode["load_list"]
        self.results = episode["results"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._needs_reset = True

        episode_idx = options.get("episode_idx", 0) if options else 0

        if episode_idx >= self.num_episodes:
            episode_idx = 0

        self.load_episode_data(self.start_episode + episode_idx)

        self.idx = 0
        self.current_cost = 0.0
        self.current_device_measurements = {"bess": {}, "pv": {}}
        self.previous_device_measurements = {"bess": {}, "pv": {}}
        self.episode_reward = 0.0

        _simulation_setup(self)

        self.results.voltages = {bus: np.zeros(self.steps) for bus in self.dss.circuit.buses_names}
        self.results.voltages_pu = {bus: np.zeros(self.steps) for bus in self.dss.circuit.buses_names}
        self.results.phase_voltages = {bus: {} for bus in self.dss.circuit.buses_names}
        self.results.phase_voltages_pu = {bus: {} for bus in self.dss.circuit.buses_names}
        self.results.phase_angles_deg = {bus: {} for bus in self.dss.circuit.buses_names}

        prepare_pre_action_observation(self)
        self._needs_reset = False

        observation = self.observe()
        state = build_state(self, self.state_functions) if self.state_functions else observation

        return state, {"observation": deepcopy(observation)}

    def step(self, action=None):
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping, including after an execution failure")
        if self.idx >= self.steps:
            raise RuntimeError("The episode has terminated; call reset()")
        prepared = prepare_action(self, action)
        pending = self._copy_for_step()
        try:
            result = pending._run_step(prepared)
        except Exception:
            # OpenDSS may have changed; reset rebuilds it before another attempt.
            self._needs_reset = True
            raise
        self._commit_step(pending)
        return result

    def _copy_for_step(self):
        """Stage mutable simulation data while retaining the external DSS context."""
        pending = copy(self)
        pending.bess_list, pending.pv_list, pending.grid, pending.results = deepcopy(
            (self.bess_list, self.pv_list, self.grid, self.results)
        )
        return pending

    def _run_step(self, prepared):
        observation = self.observe()
        applied_action = _update_snapshot_powers(self, prepared.physical)
        grid_kw, grid_kvar, cost = solve_power_flow(self, applied_action)

        self.current_cost = cost
        reward = self.reward_function(self)
        self.episode_reward += reward
        self.idx += 1

        terminated = self.idx >= self.steps
        truncated = False

        self.previous_device_measurements = self.current_device_measurements
        info = {
            "units": "pu",
            "base": self.normalizer.as_dict()["base"],
            "network": self.network.as_dict(),
            "timestamp": self.timestamps.iloc[self.idx - 1],
            "interval_end": (self.timestamps.iloc[self.idx - 1] + timedelta(hours=self.dt)).isoformat(),
            "cost": cost,
            "reward": reward,
            "grid_kw": grid_kw,
            "grid_kvar": grid_kvar,
            "grid_import_kw": max(-grid_kw, 0.0),
            "grid_export_kw": max(grid_kw, 0.0),
            "bus_voltages_pu": self.current_bus_voltages_pu,
            "bus_angles_deg": self.current_bus_angles_deg,
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
        return state, reward, terminated, truncated, info

    def _commit_step(self, pending):
        for current, completed in zip(self.bess_list, pending.bess_list):
            current.commit(completed)
        for current, completed in zip(self.pv_list, pending.pv_list):
            current.commit(completed)
        self.grid.array_kw[:] = pending.grid.array_kw
        self.grid.array_kvar[:] = pending.grid.array_kvar
        self.results.commit(pending.results)
        self.idx = pending.idx
        self.current_cost = pending.current_cost
        self.episode_reward = pending.episode_reward
        self.current_grid_kw = pending.current_grid_kw
        self.current_grid_kvar = pending.current_grid_kvar
        self.current_bus_snapshot = pending.current_bus_snapshot
        self.current_bus_voltages_pu = pending.current_bus_voltages_pu
        self.current_bus_angles_deg = pending.current_bus_angles_deg
        self.current_voltages_pu = pending.current_voltages_pu
        self.current_device_measurements = pending.current_device_measurements
        self.previous_device_measurements = pending.previous_device_measurements

    def observe(self):
        if self._needs_reset:
            raise RuntimeError("Call reset() before observing, including after an execution failure")
        return build_named_observation(self)

    def get_episode_results(self):
        return {
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
