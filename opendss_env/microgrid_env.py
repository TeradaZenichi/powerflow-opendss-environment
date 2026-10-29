from copy import deepcopy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from pu_normalizer import PUNormalizer

from .backend import OpenDSSDirectBackend
from .data import load_data, episode_data
from .states import build_named_observation, build_state
from .rewards import minimize_cost
from .simulation import (
    _simulation_setup,
    _update_snapshot_powers,
    prepare_pre_action_observation,
    solve_power_flow,
)


@dataclass(frozen=True)
class EnvironmentConfig:
    case_path: object
    episode_steps: int
    state_functions: tuple = ()
    num_episodes: int = 1
    start_episode: int = 0
    reward_function: object = minimize_cost


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
        self.normalizer = PUNormalizer(self.data["base_kva"], self.data["base_kv"])

        self.dss = OpenDSSDirectBackend()
        self.current_cost = 0.0
        self.current_device_measurements = {"bess": {}, "pv": {}}
        self.previous_device_measurements = {"bess": {}, "pv": {}}
        self.episode_reward = 0.0

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

        observation = self.observe()
        state = build_state(self, self.state_functions) if self.state_functions else observation

        return state, {"observation": deepcopy(observation)}

    def step(self, action=None):
        observation = self.observe()
        physical_action = None
        if action is not None:
            self.normalizer.require_pu(action)
            physical_action = self.normalizer.denormalize(action)
        applied_action = _update_snapshot_powers(self, physical_action)
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
            "timestamp": self.timestamps.iloc[self.idx - 1],
            "cost": cost,
            "reward": reward,
            "grid_kw": grid_kw,
            "grid_kvar": grid_kvar,
            "grid_import_kw": max(-grid_kw, 0.0),
            "grid_export_kw": max(grid_kw, 0.0),
            "bus_voltages_pu": self.current_bus_voltages_pu,
            "bus_angles_deg": self.current_bus_angles_deg,
            "requested_action": deepcopy(action),
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

    def observe(self):
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
