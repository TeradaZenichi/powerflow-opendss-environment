import gymnasium as gym
import numpy as np

from .backend import OpenDSSDirectBackend
from .data import load_data, episode_data
from .states import build_state
from .rewards import minimize_cost
from .simulation import _simulation_setup, _update_snapshot_powers, solve_power_flow


class MicrogridEnv(gym.Env):

    def __init__(self, case_path, episode_steps, num_episodes, start_episode, state_functions, reward_function=minimize_cost):
        super().__init__()

        self.case_path = case_path
        self.episode_steps = episode_steps
        self.start_episode = start_episode
        self.num_episodes = num_episodes
        self.state_functions = state_functions
        self.reward_function = reward_function
        self.end_episode = start_episode + num_episodes

        self.data = load_data(case_path)

        self.dss = OpenDSSDirectBackend()
        self.current_cost = 0.0
        self.current_device_measurements = {"bess": {}, "pv": {}}
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
        self.episode_reward = 0.0

        _simulation_setup(self)

        self.results.voltages = {bus: np.zeros(self.steps) for bus in self.dss.circuit.buses_names}
        self.results.voltages_pu = {bus: np.zeros(self.steps) for bus in self.dss.circuit.buses_names}
        self.results.phase_voltages = {bus: {} for bus in self.dss.circuit.buses_names}
        self.results.phase_voltages_pu = {bus: {} for bus in self.dss.circuit.buses_names}
        self.results.phase_angles_deg = {bus: {} for bus in self.dss.circuit.buses_names}

        state = build_state(self, self.state_functions)

        return state, {}

    def step(self, action=None):
        applied_action = _update_snapshot_powers(self, action)
        grid_kw, grid_kvar, cost = solve_power_flow(self)

        self.current_cost = cost
        reward = self.reward_function(self)
        self.episode_reward += reward
        self.idx += 1

        terminated = self.idx >= self.steps
        truncated = False

        if terminated:
            state = None
        else:
            state = build_state(self, self.state_functions)

        info = {
            "timestamp": self.timestamps.iloc[self.idx - 1],
            "cost": cost,
            "reward": reward,
            "grid_kw": grid_kw,
            "grid_kvar": grid_kvar,
            "grid_import_kw": max(-grid_kw, 0.0),
            "grid_export_kw": max(grid_kw, 0.0),
            "bus_voltages_pu": self.current_bus_voltages_pu,
            "bus_angles_deg": self.current_bus_angles_deg,
            "requested_action": action,
            "executed_action": applied_action,
            "device_measurements": self.current_device_measurements,
        }

        return state, reward, terminated, truncated, info

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
