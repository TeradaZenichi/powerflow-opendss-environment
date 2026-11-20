"""Normalized local actions, independent of the learning algorithm."""
import numpy as np
from gymnasium import spaces

from .actions import Action, EVSEAction, PhaseAction, TotalAction
from .network import device_phases


class ControllerActions:
    """Concatenate all P coordinates, then all Q; return partial named commands."""

    def __init__(self, devices, base_kva, *, phases=3, seed=None):
        self.devices = []
        lower, upper, reactive = [], [], []
        for group, items in devices.items():
            for device in items:
                connected = device_phases(device, phases)
                per_phase = device.dispatch_mode == "per_phase" and group != "evse"
                count = len(connected) if per_phase else 1
                charge = device.p_charge_max_kw if group == "bess" else device.p_max_kw
                discharge = device.p_discharge_max_kw if group == "bess" or (
                    group == "evse" and device.v2g) else 0.0
                q_max = device.s_max_kva if group != "bess" or device.reactive_control else 0.0
                lower.extend([-min(discharge, device.s_max_kva) / count] * count)
                upper.extend([min(charge, device.s_max_kva) / count] * count)
                reactive.extend([q_max / count] * count)
                self.devices.append((group, device.id, connected, per_phase, count))
        if not self.devices:
            raise ValueError("Select at least one controllable device")
        self.low = np.asarray(lower + [-value for value in reactive]) / base_kva
        self.high = np.asarray(upper + reactive) / base_kva
        self.action_space = spaces.Box(-1.0, 1.0, shape=self.low.shape, dtype=np.float32, seed=seed)

    def command(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must match the controller's normalized action space")
        p, q = np.split(self.low + (np.asarray(action) + 1) * (self.high - self.low) / 2, 2)
        result, offset = Action(), 0
        for group, device, phases, per_phase, count in self.devices:
            active, reactive = p[offset:offset + count], q[offset:offset + count]
            if group == "evse":
                command = EVSEAction(float(active[0]), float(reactive[0]))
            elif per_phase:
                command = PhaseAction(dict(zip(phases, active.tolist())), dict(zip(phases, reactive.tolist())))
            else:
                command = TotalAction(float(active[0]), float(reactive[0]))
            getattr(result, group)[device] = command
            offset += count
        return result


def compose_actions(env, commands):
    observation = env.observe()
    result = Action(
        bess={device.id: TotalAction(0.0) for device in env.bess_list},
        pv={device.id: PhaseAction(observation["pv"][device.id]["available_pu"])
            if device.dispatch_mode == "per_phase" else
            TotalAction(sum(observation["pv"][device.id]["available_pu"].values())) for device in env.pv_list},
        evse={device.id: EVSEAction(0.0) for device in env.evse_list if device.controlled},
    )
    for group in ("bess", "pv", "evse"):
        seen = set()
        for command in commands:
            values = getattr(command, group)
            if seen.intersection(values):
                raise ValueError(f"Multiple controllers command the same {group} device")
            if set(values) - set(getattr(result, group)):
                raise ValueError(f"Unknown or noncontrollable {group} device")
            getattr(result, group).update(values)
            seen.update(values)
    return result
