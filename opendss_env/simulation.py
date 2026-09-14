"""OpenDSS simulation setup and execution."""

from dataclasses import dataclass, replace
import numpy as np
from .devices_control import BESS_KVAR, BESS_KW, PV_KVAR, bess_control
from .elements import PVPoint, PVRequest


_PHASE_NAME = {1: "a", 2: "b", 3: "c"}
_PHASE_NODE = {value: key for key, value in _PHASE_NAME.items()}


def _aggregate(value):
    if isinstance(value, dict):
        return float(sum(value.values()))
    return float(value)


def _phase_result(value, phases, default_phase_count):
    names = tuple(
        _PHASE_NAME.get(int(phase), str(phase).lower())
        if isinstance(phase, int) or str(phase).isdigit()
        else str(phase).lower()
        for phase in (phases or tuple(range(1, default_phase_count + 1)))
    )
    share = float(value) / len(names)
    return {phase: share for phase in names}


def _device_phases(device, default_phase_count):
    return tuple(
        _PHASE_NAME.get(int(phase), str(phase).lower())
        if isinstance(phase, int) or str(phase).isdigit()
        else str(phase).lower()
        for phase in (
            getattr(device, "phases", ())
            or tuple(range(1, default_phase_count + 1))
        )
    )


def _element_name(device, phase=None):
    return device.id if phase is None else f"{device.id}_{phase}"


class PhaseAction:
    def __init__(self, values, phases):
        self.values = values
        self.phases = phases

    def get(self, field, default=0.0):
        value = self.values.get(field, {phase: default for phase in self.phases})
        if not isinstance(value, dict) or set(value) != set(self.phases):
            raise ValueError(f"{field} must define exactly the device phases {self.phases}")
        return {phase: float(value[phase]) for phase in self.phases}


@dataclass(frozen=True)
class BusSnapshot:
    volts: dict
    pu: dict
    angles: dict

    def device_pu(self, device):
        values = self.pu.get(device.bus, {})
        if not values:
            return {}
        phases = _device_phases(device, len(values))
        if device.connection == "wye":
            return {phase: float(values[phase]) for phase in phases}
        angles = self.angles[device.bus]
        phasors = {
            phase: float(values[phase]) * np.exp(1j * np.radians(float(angles[phase])))
            for phase in phases
        }
        return {
            phase: abs(phasors[phase] - phasors[phases[(index - 1) % len(phases)]]) / np.sqrt(3.0)
            for index, phase in enumerate(phases)
        }

    def device_average_pu(self, device):
        values = self.device_pu(device)
        return float(np.mean(tuple(values.values()))) if values else None


def _current_snapshot(env):
    volts = getattr(env, "current_bus_voltages_pu", {})
    angles = getattr(env, "current_bus_angles_deg", {})
    fallback = BusSnapshot({}, volts, angles)
    return getattr(env, "current_bus_snapshot", fallback)


def _bus_kv_ln(env, bus):
    if env.data["phases"] == 1:
        return float(env.data["base_kv"])
    env.dss.circuit.set_active_bus(bus)
    kv_ln = float(env.dss.bus.kv_base)
    if kv_ln <= 0.0:
        raise RuntimeError(f"OpenDSS bus {bus!r} has no voltage base")
    return kv_ln


class DssDevice:
    def __init__(self, env, device, element_class):
        self.env = env
        self.device = device
        self.element_class = element_class
        self.phases = _device_phases(device, getattr(env, "data", {}).get("phases", 3))
        self.connection = getattr(device, "connection", "wye")
        self.dispatch_mode = getattr(device, "dispatch_mode", "aggregate")

    def create(self):
        phase_items = [(None, None)] if self.dispatch_mode == "aggregate" else list(enumerate(self.phases))
        phase_count = len(self.phases) if self.dispatch_mode == "aggregate" else 1
        kv = _bus_kv_ln(self.env, self.device.bus)
        if self.connection == "delta" or phase_count > 1:
            kv *= np.sqrt(3.0)
        for index, phase in phase_items:
            name = _element_name(self.device, phase)
            bus = self._bus(index)
            command = (
                f"New {self.element_class}.{name} bus1={bus} phases={phase_count} kv={kv} "
                f"conn={self.connection} kw=0 kvar=0 model=1 vminpu=0 vmaxpu=2"
            )
            self.env.dss.text(command)

    def edit(self, p_kw, q_kvar, phase=None):
        name = _element_name(self.device, phase)
        self.env.dss.text(f"Edit {self.element_class}.{name} kw={p_kw} kvar={q_kvar}")

    def terminal_power(self):
        if self.dispatch_mode == "aggregate":
            terminal = _terminal_power_by_phase(self.env, f"{self.element_class}.{self.device.id}")
            if self.connection == "wye":
                return terminal
            p_total = sum(value["p_kw"] for value in terminal.values())
            q_total = sum(value["q_kvar"] for value in terminal.values())
            return {
                phase: {"p_kw": p_total / len(self.phases), "q_kvar": q_total / len(self.phases)}
                for phase in self.phases
            }
        result = {}
        for phase in self.phases:
            values = _terminal_power_by_phase(self.env, f"{self.element_class}.{_element_name(self.device, phase)}")
            result[phase] = {field: sum(value[field] for value in values.values()) for field in ("p_kw", "q_kvar")}
        return result

    def _bus(self, index):
        if index is None:
            return self.device.bus
        phase = self.phases[index]
        bus = f"{self.device.bus}.{_PHASE_NODE[phase]}"
        if self.connection == "delta":
            previous = self.phases[(index - 1) % len(self.phases)]
            bus += f".{_PHASE_NODE[previous]}"
        return bus


def _terminal_power_by_phase(env, element_name):
    if not env.dss.circuit.set_active_element(element_name):
        raise RuntimeError(f"OpenDSS element not found: {element_name}")

    conductor_count = env.dss.cktelement.num_conductors
    nodes = env.dss.cktelement.node_order[:conductor_count]
    powers = env.dss.cktelement.powers[:2 * conductor_count]

    terminal = {}
    for position, node in enumerate(nodes):
        if node not in _PHASE_NAME:
            continue
        terminal[_PHASE_NAME[node]] = {
            "p_kw": float(powers[2 * position]),
            "q_kvar": float(powers[2 * position + 1]),
        }
    return terminal


def collect_device_measurements(env):
    """Read solved terminal powers and convert them to the public convention."""

    measurements = {"bess": {}, "pv": {}}

    for bess in env.bess_list:
        terminal = DssDevice(env, bess, "Load").terminal_power()
        p_net = {phase: value["p_kw"] for phase, value in terminal.items()}
        q_injection = {
            phase: -value["q_kvar"] for phase, value in terminal.items()
        }
        measurements["bess"][bess.id] = {
            "p_net_kw": p_net,
            "q_injection_kvar": q_injection,
            "p_net_total_kw": sum(p_net.values()),
            "q_injection_total_kvar": sum(q_injection.values()),
        }

    for pv in env.pv_list:
        terminal = DssDevice(env, pv, "Generator").terminal_power()
        generation = {
            phase: -value["p_kw"] for phase, value in terminal.items()
        }
        q_injection = {
            phase: -value["q_kvar"] for phase, value in terminal.items()
        }
        measurements["pv"][pv.id] = {
            "generation_kw": generation,
            "q_injection_kvar": q_injection,
            "generation_total_kw": sum(generation.values()),
            "q_injection_total_kvar": sum(q_injection.values()),
        }

    return measurements


def collect_bus_measurements(env):
    voltages_pu = {}
    angles_deg = {}
    voltages = {}
    for bus in env.dss.circuit.buses_names:
        env.dss.circuit.set_active_bus(bus)
        nodes = env.dss.bus.nodes
        voltage_values = env.dss.bus.vmag_angle
        pu_values = env.dss.bus.pu_vmag_angle
        voltages[bus] = {}
        voltages_pu[bus] = {}
        angles_deg[bus] = {}
        for position, node in enumerate(nodes):
            if node not in _PHASE_NAME:
                continue
            phase = _PHASE_NAME[node]
            voltages[bus][phase] = float(voltage_values[2 * position])
            voltages_pu[bus][phase] = float(pu_values[2 * position])
            angles_deg[bus][phase] = float(pu_values[2 * position + 1])
    return BusSnapshot(voltages, voltages_pu, angles_deg)


def initialize_pre_action_observation(env):
    """Solve the reset snapshot with current loads and idle controllable devices."""

    for load in env.load_list:
        env.dss.text(
            f"Edit Load.{load.id} kw={load.array_kw[env.idx]} "
            f"kvar={load.array_kvar[env.idx]}"
        )
    env.dss.text("Set Tolerance=1e-8")
    env.dss.solution.solve()
    if not env.dss.solution.converged:
        raise RuntimeError("OpenDSS did not converge while initializing observation")
    snapshot = collect_bus_measurements(env)
    env.current_bus_snapshot = snapshot
    env.current_bus_voltages_pu = snapshot.pu
    env.current_bus_angles_deg = snapshot.angles
    env.current_device_measurements = collect_device_measurements(env)

def _simulation_setup(env):
    """Create the OpenDSS circuit and its devices."""

    env.dss.text("Clear")
    env.dss.text(f'compile "{env.data["topology"]}"')
    env.dss.text("Vsource.source.model=Ideal")
    env.dss.text("Set mode=snapshot controlmode=off")

    for pv in env.pv_list:
        DssDevice(env, pv, "Generator").create()

    for bess in env.bess_list:
        DssDevice(env, bess, "Load").create()

    for load in env.load_list:
        if load.phase_node is None:
            bus = load.bus
            phases = env.data["phases"]
        else:
            bus = f"{load.bus}.{load.phase_node}"
            phases = 1
        kv = _bus_kv_ln(env, load.bus)
        if phases > 1:
            kv *= np.sqrt(3.0)
        env.dss.text(
            f"New Load.{load.id} bus1={bus} phases={phases} kv={kv} "
            "kw=0 kvar=0 conn=wye model=1"
        )

def _update_snapshot_powers(env, action=None):
    """Apply load and device powers for the current step."""

    for load in env.load_list:
        command = f"Edit Load.{load.id} kw={load.array_kw[env.idx]} kvar={load.array_kvar[env.idx]}"
        env.dss.text(command)

    applied = {"bess": {}, "pv": {}}
    snapshot = _current_snapshot(env)

    for pv in env.pv_list:
        available_kw = float(pv.profile[env.idx])
        phases = _device_phases(pv, env.data["phases"])
        element = DssDevice(env, pv, "Generator")
        if pv.dispatch_mode == "aggregate":
            voltage_pu = snapshot.device_average_pu(pv)
            if action is None:
                requested_generation_kw = available_kw
                requested_q_kvar = float(PV_KVAR[env.idx])
            else:
                command = action.get("pv", {}).get(pv.id)
                if command is None:
                    raise ValueError(f"Missing action for PV {pv.id!r}")
                requested_generation_kw = _aggregate(command["generation_kw"])
                requested_q_kvar = _aggregate(command.get("q_injection_kvar", 0.0))
            request = PVRequest(requested_generation_kw, requested_q_kvar, available_kw, voltage_pu)
            p_pv, q_pv_injection = pv.apply(request)
            pv._control_request = request
            generation_by_phase = _phase_result(pv.array_kw[-1], phases, len(phases))
            q_by_phase = _phase_result(q_pv_injection, phases, len(phases))
            element.edit(p_pv, q_pv_injection)
        else:
            phase_available = {
                phase: float(pv.phase_profiles[phase][env.idx])
                for phase in phases
            }
            if action is None:
                requested_generation = dict(phase_available)
                requested_q = {
                    phase: float(PV_KVAR[env.idx]) / len(phases)
                    for phase in phases
                }
            else:
                command = action.get("pv", {}).get(pv.id)
                if command is None:
                    raise ValueError(f"Missing action for PV {pv.id!r}")
                command = PhaseAction(command, phases)
                requested_generation = command.get("generation_kw")
                requested_q = command.get("q_injection_kvar")
            phase_points = {}
            phase_requests = {}
            for phase in phases:
                voltage_pu = snapshot.device_pu(pv).get(phase)
                request = PVRequest(
                    requested_generation[phase], requested_q[phase], phase_available[phase],
                    voltage_pu, 1.0 / len(phases)
                )
                point = pv.operating_point(request)
                phase_points[phase] = point
                phase_requests[phase] = request
                element.edit(point.p_net_kw, point.q_injection_kvar, phase)
            pv._phase_control_requests = phase_requests
            pv._phase_points = phase_points
            generation_by_phase = {
                phase: point.generation_kw for phase, point in phase_points.items()
            }
            q_by_phase = {
                phase: point.q_injection_kvar for phase, point in phase_points.items()
            }
            p_pv = sum(point.p_net_kw for point in phase_points.values())
            q_pv_injection = sum(q_by_phase.values())
            pv.array_kw.append(sum(generation_by_phase.values()))
            pv.array_kvar.append(q_pv_injection)
            pv.array_p_net_kw.append(p_pv)
            pv.array_grid_consumption_kw.append(sum(point.grid_consumption_kw for point in phase_points.values()))
            pv.array_inverter_loss_kw.append(sum(point.inverter_loss_kw for point in phase_points.values()))
        generation_kw = pv.array_kw[-1]
        inverter_loss_kw = pv.array_inverter_loss_kw[-1]
        applied["pv"][pv.id] = {
            "generation_kw": generation_by_phase,
            "q_injection_kvar": q_by_phase,
            "generation_total_kw": generation_kw,
            "p_injection_total_kw": p_pv,
            "q_injection_total_kvar": q_pv_injection,
            "available_kw": available_kw,
            "curtailment_kw": max(available_kw - generation_kw - inverter_loss_kw, 0.0),
            "inverter_loss_kw": inverter_loss_kw,
        }

    for bess in env.bess_list:
        soc_before_frac = bess.soc
        phases = _device_phases(bess, env.data["phases"])
        element = DssDevice(env, bess, "Load")
        if bess.dispatch_mode == "aggregate":
            if action is None:
                p_bess, q_bess_injection = bess_control(bess, env.idx, env.dt, BESS_KW, BESS_KVAR)
            else:
                command = action.get("bess", {}).get(bess.id)
                if command is None:
                    raise ValueError(f"Missing action for BESS {bess.id!r}")
                p_bess, q_bess_injection = bess.operate(
                    _aggregate(command["p_net_kw"]),
                    _aggregate(command.get("q_injection_kvar", 0.0)),
                    env.dt,
                )
            p_by_phase = _phase_result(p_bess, phases, len(phases))
            q_by_phase = _phase_result(q_bess_injection, phases, len(phases))
            element.edit(p_bess, -q_bess_injection)
        else:
            if action is None:
                p_request = {
                    phase: float(BESS_KW[env.idx]) / len(phases)
                    for phase in phases
                }
                q_request = {
                    phase: float(BESS_KVAR[env.idx]) / len(phases)
                    for phase in phases
                }
            else:
                command = action.get("bess", {}).get(bess.id)
                if command is None:
                    raise ValueError(f"Missing action for BESS {bess.id!r}")
                command = PhaseAction(command, phases)
                p_request = command.get("p_net_kw")
                q_request = command.get("q_injection_kvar")
            p_by_phase, q_by_phase = bess.operate_phases(p_request, q_request, env.dt)
            p_bess = sum(p_by_phase.values())
            q_bess_injection = sum(q_by_phase.values())
            for phase in phases:
                element.edit(p_by_phase[phase], -q_by_phase[phase], phase)
        applied["bess"][bess.id] = {
            "p_net_kw": p_by_phase,
            "q_injection_kvar": q_by_phase,
            "p_net_total_kw": p_bess,
            "q_injection_total_kvar": q_bess_injection,
            "soc_before_frac": soc_before_frac,
            "soc_after_frac": bess.soc,
            "inverter_loss_kw": bess.array_inverter_loss_kw[-1],
        }

    return applied

def solve_power_flow(env, applied_action=None):
    """Solve the current snapshot and store its results."""

    env.dss.text("Set Tolerance=1e-8")
    for _ in range(50):
        env.dss.solution.solve()
        if not env.dss.solution.converged:
            raise RuntimeError(f"OpenDSS did not converge at timestep {env.idx}")
        trial = collect_bus_measurements(env)
        changed = False
        for pv in env.pv_list:
            if pv.control not in {"volt-var", "volt-watt", "volt-var-watt"}:
                continue
            phases = _device_phases(pv, env.data["phases"])
            element = DssDevice(env, pv, "Generator")
            if pv.dispatch_mode == "aggregate":
                voltage_pu = trial.device_average_pu(pv)
                request = pv._control_request
                control_available_kw = request.available_kw
                point = pv.operating_point(replace(request, voltage_pu=voltage_pu))
                previous_p = pv.array_p_net_kw[-1]
                previous_q = pv.array_kvar[-1]
                if max(
                    abs(point.p_net_kw - previous_p),
                    abs(point.q_injection_kvar - previous_q),
                ) <= 1e-7:
                    continue
                changed = True
                generation_by_phase = _phase_result(point.generation_kw, phases, len(phases))
                q_by_phase = _phase_result(point.q_injection_kvar, phases, len(phases))
                pv.array_kw[-1] = point.generation_kw
                pv.array_kvar[-1] = point.q_injection_kvar
                pv.array_p_net_kw[-1] = point.p_net_kw
                pv.array_grid_consumption_kw[-1] = point.grid_consumption_kw
                pv.array_inverter_loss_kw[-1] = point.inverter_loss_kw
                element.edit(point.p_net_kw, point.q_injection_kvar)
            else:
                phase_points = {}
                max_change = 0.0
                for phase in phases:
                    request = pv._phase_control_requests[phase]
                    command = replace(request, voltage_pu=trial.device_pu(pv)[phase])
                    phase_point = pv.operating_point(command)
                    previous = pv._phase_points[phase]
                    max_change = max(
                        max_change,
                        abs(phase_point.p_net_kw - previous.p_net_kw),
                        abs(phase_point.q_injection_kvar - previous.q_injection_kvar),
                    )
                    phase_points[phase] = phase_point
                if max_change <= 1e-7:
                    continue
                changed = True
                pv._phase_points = phase_points
                for phase, phase_point in phase_points.items():
                    element.edit(phase_point.p_net_kw, phase_point.q_injection_kvar, phase)
                generation_by_phase = {
                    phase: phase_point.generation_kw
                    for phase, phase_point in phase_points.items()
                }
                q_by_phase = {
                    phase: phase_point.q_injection_kvar
                    for phase, phase_point in phase_points.items()
                }
                point = PVPoint.total(phase_points.values())
                control_available_kw = sum(request.available_kw for request in pv._phase_control_requests.values())
                pv.array_kw[-1] = point.generation_kw
                pv.array_kvar[-1] = point.q_injection_kvar
                pv.array_p_net_kw[-1] = point.p_net_kw
                pv.array_grid_consumption_kw[-1] = point.grid_consumption_kw
                pv.array_inverter_loss_kw[-1] = point.inverter_loss_kw
            if applied_action is not None:
                applied_action["pv"][pv.id].update({
                    "generation_kw": generation_by_phase,
                    "q_injection_kvar": q_by_phase,
                    "generation_total_kw": point.generation_kw,
                    "p_injection_total_kw": point.p_net_kw,
                    "q_injection_total_kvar": point.q_injection_kvar,
                    "curtailment_kw": max(
                        control_available_kw
                        - point.generation_kw
                        - point.inverter_loss_kw,
                        0.0,
                    ),
                    "inverter_loss_kw": point.inverter_loss_kw,
                })
        if not changed:
            break
    else:
        raise RuntimeError(f"PV voltage controls did not converge at timestep {env.idx}")

    snapshot = collect_bus_measurements(env)
    for bus in env.results.voltages:
        for phase, value in snapshot.volts[bus].items():
            env.results.phase_voltages[bus].setdefault(phase, np.zeros(env.steps))[env.idx] = value
            env.results.phase_voltages_pu[bus].setdefault(phase, np.zeros(env.steps))[env.idx] = snapshot.pu[bus][phase]
            env.results.phase_angles_deg[bus].setdefault(phase, np.zeros(env.steps))[env.idx] = snapshot.angles[bus][phase]
        first_phase = next(iter(snapshot.pu[bus]))
        env.results.voltages[bus][env.idx] = snapshot.volts[bus][first_phase]
        env.results.voltages_pu[bus][env.idx] = snapshot.pu[bus][first_phase]

    grid_kw = env.dss.circuit.total_power[0]
    grid_kvar = -env.dss.circuit.total_power[1]

    env.grid.array_kw.append(grid_kw)
    env.grid.array_kvar.append(grid_kvar)

    cost = -grid_kw * env.grid.prices[env.idx] * env.dt
    env.results.costs.append(cost)

    env.current_grid_kw = grid_kw
    env.current_grid_kvar = grid_kvar
    env.current_cost = cost
    env.current_voltages_pu = {
        bus: env.results.voltages_pu[bus][env.idx]
        for bus in env.results.voltages_pu
    }
    env.current_bus_snapshot = snapshot
    env.current_bus_voltages_pu = snapshot.pu
    env.current_bus_angles_deg = snapshot.angles
    env.current_device_measurements = collect_device_measurements(env)

    return grid_kw, grid_kvar, cost
