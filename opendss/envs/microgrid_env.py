import gymnasium as gym
import numpy as np
import py_dss_interface

from ..data import load_data, split_episodes
from ..devices_control import (BESS_KW, BESS_KVAR, PV_KVAR, bess_control, pv_control)
from .states import build_state
from .rewards import minimize_cost

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

        self._simulation_setup()

    def _simulation_setup(self):
        """
        Creates the OpenDSS circuit using the devices defined in the case.
        """

        self.dss.text("Clear")
        self.dss.text(f'compile "{self.data["topology"]}"')
        self.dss.text("Vsource.source.model=Ideal")

        # PV generators
        for pv in self.episodes[0]["pv_list"]:
            self.dss.text(f"""New Generator.{pv.id} bus1={pv.bus} phases={self.data["phases"]} kv={self.data["base_kv"]} kw=0 kvar=0""")

        # BESS
        for bess in self.episodes[0]["bess_list"]:
            self.dss.text( f""" New Load.{bess.id} bus={bess.bus} phases={self.data["phases"]} kv={self.data["base_kv"]} kw=0 kvar=0 conn=y""")

        # Loads
        for load in self.episodes[0]["load_list"]:
            self.dss.text( f""" New Load.{load.id} bus1={load.bus} phases={self.data["phases"]} kv={self.data["base_kv"]} kw=0 kvar=0""")

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

    def _update_snapshot_powers(self):
        """
        Updates all loads, PVs and BESSs for the current time step.
        """

        # Loads
        for load in self.load_list:
            self.dss.text( f"Edit Load.{load.id} kw={load.array_kw[self.idx]} kvar={load.array_kvar[self.idx]}")

        # PV
        for pv_idx, pv in enumerate(self.pv_list):
            p_pv, q_pv_injection = pv_control(pv, self.idx, PV_KVAR)
            self.dss.text(f"Edit Generator.{pv.id} kw={p_pv} kvar={q_pv_injection}")

        # BESS
        for bess_idx, bess in enumerate(self.bess_list):
            p_bess, q_bess_injection = bess_control(bess, self.idx, self.dt, BESS_KW, BESS_KVAR)
            self.dss.text(f"Edit Load.{bess.id} kw={p_bess} kvar={-q_bess_injection}")

    def _solve_power_flow(self):
        """
        Solves the OpenDSS power flow and updates the results.
        """

        self.dss.text("Set Tolerance=1e-8")
        self.dss.solution.solve()

        # Bus voltages
        for bus in self.results.voltages:
            self.dss.circuit.set_active_bus(bus)
            voltage = self.dss.bus.vmag_angle[0]
            self.results.voltages[bus][self.idx] = voltage
            self.results.voltages_pu[bus][self.idx] = (voltage / (self.data["base_kv"] * 1000))

        # Grid power
        grid_kw = self.dss.circuit.total_power[0]
        grid_kvar = -self.dss.circuit.total_power[1]

        self.grid.array_kw.append(grid_kw)
        self.grid.array_kvar.append(grid_kvar)

        # Cost
        cost = (-grid_kw * self.grid.prices[self.idx]* self.dt)
        self.results.costs.append(cost)

        self.current_grid_kw = grid_kw
        self.current_grid_kvar = grid_kvar
        self.current_cost = cost
        self.current_voltages_pu = {bus: self.results.voltages_pu[bus][self.idx] for bus in self.results.voltages_pu}

        return grid_kw, grid_kvar, cost

    def step(self, action=None):
        """
        Executes one simulation step.

        The action is currently ignored because the BESS and PV
        controls are fixed in devices_control.py.
        """

        self._update_snapshot_powers()

        grid_kw, grid_kvar, cost = self._solve_power_flow()

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