"""Device actions in pu, independent of the controller that produced them."""
from copy import deepcopy
from dataclasses import dataclass, field
import math


@dataclass
class TotalAction:
    """P and Q for the whole device; P follows its BESS or PV convention."""

    p_pu: float
    q_pu: float = 0.0


@dataclass
class PhaseAction:
    """P and Q for each connected phase (winding leg for delta devices)."""

    p_pu: dict[str, float]
    q_pu: dict[str, float] = field(default_factory=dict)


@dataclass
class Action:
    bess: dict[str, TotalAction | PhaseAction] = field(default_factory=dict)
    pv: dict[str, TotalAction | PhaseAction] = field(default_factory=dict)

    def as_dict(self):
        values = {"bess": {}, "pv": {}}
        for kind, devices, active_field in (
            ("bess", self.bess, "p_net_pu"),
            ("pv", self.pv, "generation_pu"),
        ):
            for device_id, command in devices.items():
                if isinstance(command, TotalAction):
                    mode = "total"
                    reactive = command.q_pu
                elif isinstance(command, PhaseAction):
                    mode = "per_phase"
                    reactive = command.q_pu or dict.fromkeys(command.p_pu, 0.0)
                else:
                    raise TypeError("Device actions must be TotalAction or PhaseAction")
                values[kind][device_id] = {
                    "mode": mode, active_field: deepcopy(command.p_pu),
                    "q_injection_pu": deepcopy(reactive),
                }
        return values


@dataclass
class PreparedAction:
    requested: dict | None
    physical: dict | None


def _finite_power(value):
    if isinstance(value, bool):
        raise ValueError("Power values must be finite numbers")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Power values must be finite numbers")
    return value


def _phase_action(command, phases, active_field):
    """Turn a declared total/phase command into connected phase powers."""
    if not isinstance(command, dict) or active_field not in command:
        raise ValueError(f"Device action requires {active_field}")
    if set(command) - {"mode", active_field, "q_injection_pu"}:
        raise ValueError("Unknown device action fields")
    active = command[active_field]
    reactive = command.get("q_injection_pu", 0.0)
    mode = command.get("mode", "per_phase" if isinstance(active, dict) else "total")
    if mode == "total":
        if isinstance(active, dict) or isinstance(reactive, dict):
            raise ValueError("Total actions require scalar P and Q")
        p_total, q_total = _finite_power(active), _finite_power(reactive)
        return PhaseAction(
            dict.fromkeys(phases, p_total / len(phases)),
            dict.fromkeys(phases, q_total / len(phases)),
        )
    if mode != "per_phase":
        raise ValueError("Action mode must be total or per_phase")
    if not isinstance(active, dict) or set(active) != set(phases):
        raise ValueError(f"Phase action must use exactly {phases}")
    if not isinstance(reactive, dict):
        if _finite_power(reactive) != 0.0:
            raise ValueError("A nonzero phase Q command must name each phase")
        reactive = dict.fromkeys(phases, 0.0)
    if set(reactive) != set(phases):
        raise ValueError(f"Phase Q action must use exactly {phases}")
    return PhaseAction(
        {phase: _finite_power(active[phase]) for phase in phases},
        {phase: _finite_power(reactive[phase]) for phase in phases},
    )


def prepare_action(env, action):
    """Validate every device before touching OpenDSS or integrating energy.

    Named dictionaries remain the compatibility boundary for teacher labels.
    Their scalar/phase shape is checked here instead of silently aggregated.
    """
    if action is None:
        return PreparedAction(None, None)
    requested = action.as_dict() if isinstance(action, Action) else deepcopy(action)
    if not isinstance(requested, dict):
        raise TypeError("Action must be an Action object or a named dictionary")
    env.normalizer.require_pu(requested)
    if set(requested) - {"bess", "pv"}:
        raise ValueError("Action groups must be bess and pv")
    commands = {"bess": {}, "pv": {}}
    for kind, devices, active_field in (
        ("bess", env.bess_list, "p_net_pu"),
        ("pv", env.pv_list, "generation_pu"),
    ):
        group = requested.get(kind, {})
        if not isinstance(group, dict) or set(group) != {device.id for device in devices}:
            raise ValueError(f"Action {kind} IDs must exactly match the connected devices")
        for device in devices:
            command = group[device.id]
            phases = env.network.connected_phases(device.phases)
            phased = _phase_action(command, phases, active_field)
            p, q = phased.p_pu, phased.q_pu
            if device.dispatch_mode == "aggregate":
                for powers in (p, q):
                    first = powers[phases[0]]
                    if not all(math.isclose(value, first, rel_tol=1e-9, abs_tol=1e-12) for value in powers.values()):
                        raise ValueError(f"Unequal phase commands require per_phase dispatch for {device.id!r}")
                p, q = sum(p.values()), sum(q.values())
            commands[kind][device.id] = {active_field: p, "q_injection_pu": q}
    return PreparedAction(requested, env.normalizer.denormalize(commands))
