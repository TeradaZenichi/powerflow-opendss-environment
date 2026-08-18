import gymnasium as gym
import numpy as np
import py_dss_interface

from .data import load_data, split_episodes
from .states import build_state
from .rewards import minimize_cost
from .simulation import _simulation_setup, _update_snapshot_powers, solve_power_flow

class MicrogridEnv(gym.Env):

    def __init__(self, case_path, episode_steps, num_episodes, start_episode, state_functions, reward_function=minimize_cost):
        super().__init__() # __init__ of gym.Env

        self.case_path = case_path
        self.episode_steps = episode_steps
        self.state_functions = state_functions
        self.reward_function = reward_function

        # Load complete dataset
        self.data = load_data(case_path)

        # Create episode-specific data
        self.episodes = split_episodes(self.data, episode_steps)

        # Validate requested episodes
        total_episodes = len(self.episodes)
        end_episode = start_episode + num_episodes
        if end_episode > total_episodes:
            raise ValueError(f"Requested episodes {start_episode} to {end_episode - 1}, but the dataset only contains {total_episodes} episodes.")

        self.start_episode = start_episode
        self.num_episodes = num_episodes
        self.end_episode = end_episode

        self.episode_idx = 0
        self.idx = 0

        self.dss = py_dss_interface.DSS()
        self.current_cost = 0.0
        self.episode_reward = 0.0

        _simulation_setup(self)

    def _load_episode(self, episode_idx):
        """
        Loads the objects corresponding to one episode.
        """

        episode = self.episodes[episode_idx]

        self.dt = episode["dt"]
        self.steps = episode["steps"]
        self.timestamps = episode["timestamps"]

        self.grid = episode["grid"]
        self.bess_list = episode["bess_list"]
        self.pv_list = episode["pv_list"]
        self.load_list = episode["load_list"]
        self.results = episode["results"]

        self.idx = 0
        self.current_cost = 0.0
        self.episode_reward = 0.0

        # Initialize result arrays for this episode
        self.results.voltages = {
            bus: np.zeros(self.steps)
            for bus in self.dss.circuit.buses_names
        }

        self.results.voltages_pu = {
            bus: np.zeros(self.steps)
            for bus in self.dss.circuit.buses_names
        }

    def reset(self, *, seed=None, options=None):
        """
        Starts an episode. 
        """

        super().reset(seed=seed)

        if options is not None and "episode_idx" in options:
            episode_idx = options["episode_idx"]
        else:
            episode_idx = 0

        if episode_idx >= self.num_episodes:
            episode_idx = 0

        self._load_episode(self.start_episode + episode_idx)

        state = build_state(self, self.state_functions)

        return state, {}

    def step(self, action=None):
        """
        Executes one simulation step.

        The action is currently ignored because the BESS and PV
        controls are fixed in devices_control.py.
        """

        # here we would normally apply the action to the BESS and PV devices, but for now, we are ignoring it.

        _update_snapshot_powers(self)

        grid_kw, grid_kvar, cost = solve_power_flow(self)

        self.current_cost = cost

        reward = self.reward_function(self)
        self.episode_reward += reward

        self.idx += 1

        terminated = self.idx >= self.steps
        truncated = False

        if terminated:
            self.episode_idx += 1

        state = build_state(self, self.state_functions)

        info = {
            "cost": cost,
            "reward": reward,
            "grid_kw": grid_kw,
            "grid_kvar": grid_kvar,
        }

        return (
            state,
            reward,
            terminated,
            truncated,
            info,
        )

    def get_episode_results(self):
        """
        Returns the results of the current episode.
        """

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