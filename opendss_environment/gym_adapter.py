"""Flat Gymnasium interface over the named physical environment API."""
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import Action, PhaseAction, TotalAction


class GymAdapter(gym.Env):
    """Select numeric observation paths and controlled devices for an RL policy."""

    metadata = {"render_modes": []}

    def __init__(self, env, observation_paths=None, devices=None):
        self.env = env
        _, info = env.reset()
        observation = info["observation"]

        def paths(value, prefix):
            if isinstance(value, dict):
                return [path for key in sorted(value) for path in paths(value[key], prefix + (key,))]
            return [prefix]

        self.observation_paths = observation_paths if observation_paths is not None else [
            path for group in ("buses", "bess", "pv", "grid") for path in paths(observation[group], (group,))
        ]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.observation_paths),), dtype=np.float64)
        self.action_order, lower, upper = [], [], []
        available = {"bess": env.bess_list, "pv": env.pv_list}
        selected = {group: [device.id for device in items] for group, items in available.items()} if devices is None else devices
        if set(selected) - set(available):
            raise ValueError("Controlled device groups must be bess or pv")
        for group, items in available.items():
            if set(selected.get(group, [])) - {device.id for device in items}:
                raise ValueError(f"Unknown controlled {group} device")
            for device in items:
                if device.id not in selected.get(group, []):
                    continue
                phases = env.network.connected_phases(device.phases)
                slots = phases if device.dispatch_mode == "per_phase" else (None,)
                count = len(slots)
                for phase in slots:
                    p_min = -min(device.p_discharge_max_kw, device.s_max_kva / count) if group == "bess" else 0.0
                    p_max = min(device.p_charge_max_kw, device.s_max_kva / count) if group == "bess" else device.p_max_kw / count
                    q_max = device.s_max_kva / count if group == "pv" or device.reactive_control else 0.0
                    self.action_order.append((group, device.id, phase))
                    lower.extend((p_min, -q_max))
                    upper.extend((p_max, q_max))
        if not self.action_order:
            raise ValueError("Select at least one controlled device")
        base = env.data["base_kva"]
        self.action_space = spaces.Box(np.asarray(lower) / base, np.asarray(upper) / base, dtype=np.float64)
        self._encode(observation)

    def _encode(self, observation):
        values = []
        for path in self.observation_paths:
            value = observation
            for key in path:
                value = value[key]
            values.append(float(value))
        return np.asarray(values, dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        _, info = self.env.reset(seed=seed, options=options)
        return self._encode(info["observation"]), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action):
            raise ValueError("Action must be finite and inside action_space")
        observation = self.env.observe()
        requested = Action(
            bess={key: TotalAction(0.0) for key in observation["bess"]},
            pv={key: TotalAction(sum(value["available_pu"].values())) for key, value in observation["pv"].items()},
        )
        for (group, device, phase), (p, q) in zip(self.action_order, action.reshape(-1, 2)):
            commands = getattr(requested, group)
            if phase is None:
                commands[device] = TotalAction(float(p), float(q))
            else:
                if not isinstance(commands[device], PhaseAction):
                    commands[device] = PhaseAction({}, {})
                commands[device].p_pu[phase], commands[device].q_pu[phase] = float(p), float(q)
        _, reward, terminated, truncated, info = self.env.step(requested)
        state = self._terminal_observation(info) if terminated or truncated else self.env.observe()
        return self._encode(state), reward, terminated, truncated, info

    def _terminal_observation(self, info):
        """Encode the last executed state; it is not another decision interval."""
        state = deepcopy(info["observation_before"])
        state["timestamp"] = info["interval_end"]
        state["branches"] = info["branch_measurements"]
        for bus, values in state["buses"].items():
            values["v_before_pu"] = info["bus_voltages_pu"][bus]
            values["angle_before_deg"] = info["bus_angles_deg"][bus]
        for group, field in (("bess", "p_net_pu"), ("pv", "generation_pu")):
            for device, values in state[group].items():
                measured = info["device_measurements"][group][device]
                values["previous_p_pu"] = measured[field]
                values["previous_q_pu"] = measured["q_injection_pu"]
                if group == "bess":
                    values["soc_before_frac"] = info["executed_action"][group][device]["soc_after_frac"]
        return state

    def close(self):
        self.env.close()
