'''
Functions to setup and run the OpenDSS simulation.
''' 

import numpy as np
from .devices_control import bess_control, pv_control, BESS_KW, BESS_KVAR, PV_KVAR


_PHASE_NAME = {1: "a", 2: "b", 3: "c"}


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

def _simulation_setup(env):
    """
    Creates the OpenDSS circuit using the devices defined in the case.
    """

    env.dss.text("Clear")
    env.dss.text(f'compile "{env.data["topology"]}"')
    env.dss.text("Vsource.source.model=Ideal")
    env.dss.text("Set mode=snapshot controlmode=off")

    # PV generators
    for pv in env.pv_list:
        phases = len(pv.phases) or env.data["phases"]
        env.dss.text(f"""New Generator.{pv.id} bus1={pv.bus} phases={phases} kv={env.data["base_kv"]} kw=0 kvar=0 model=1""")

    # BESS
    for bess in env.bess_list:
        phases = len(bess.phases) or env.data["phases"]
        env.dss.text(f"""New Load.{bess.id} bus1={bess.bus} phases={phases} kv={env.data["base_kv"]} kw=0 kvar=0 conn=y model=1""")

    # Loads
    for load in env.load_list:
        if load.phase_node is None:
            bus = load.bus
            phases = env.data["phases"]
            kv = env.data["base_kv"]
        else:
            bus = f"{load.bus}.{load.phase_node}"
            phases = 1
            kv = env.data["base_kv"] / np.sqrt(3.0)
        env.dss.text(
            f"New Load.{load.id} bus1={bus} phases={phases} kv={kv} "
            "kw=0 kvar=0 conn=wye model=1"
        )

def _update_snapshot_powers(env, action=None):
    """
    Updates all loads, PVs and BESSs for the current time step.
    """

    # Loads
    for load in env.load_list:
        env.dss.text( f"Edit Load.{load.id} kw={load.array_kw[env.idx]} kvar={load.array_kvar[env.idx]}")

    applied = {"bess": {}, "pv": {}}

    # PV
    for pv_idx, pv in enumerate(env.pv_list):
        if action is None:
            p_pv, q_pv_injection = pv_control(pv, env.idx, PV_KVAR)
        else:
            command = action.get("pv", {}).get(pv.id)
            if command is None:
                raise ValueError(f"Missing action for PV {pv.id!r}")
            p_pv, q_pv_injection = pv.operate(
                _aggregate(command["generation_kw"]),
                _aggregate(command.get("q_injection_kvar", 0.0)),
                available_kw=pv.profile[env.idx],
            )
        env.dss.text(f"Edit Generator.{pv.id} kw={p_pv} kvar={q_pv_injection}")
        applied["pv"][pv.id] = {
            "generation_kw": _phase_result(p_pv, pv.phases, env.data["phases"]),
            "q_injection_kvar": _phase_result(
                q_pv_injection, pv.phases, env.data["phases"]
            ),
        }

    # BESS
    for bess_idx, bess in enumerate(env.bess_list):
        if action is None:
            p_bess, q_bess_injection = bess_control(
                bess, env.idx, env.dt, BESS_KW, BESS_KVAR
            )
        else:
            command = action.get("bess", {}).get(bess.id)
            if command is None:
                raise ValueError(f"Missing action for BESS {bess.id!r}")
            p_bess, q_bess_injection = bess.operate(
                _aggregate(command["p_net_kw"]),
                _aggregate(command.get("q_injection_kvar", 0.0)),
                env.dt,
            )
        env.dss.text(f"Edit Load.{bess.id} kw={p_bess} kvar={-q_bess_injection}")
        applied["bess"][bess.id] = {
            "p_net_kw": _phase_result(p_bess, bess.phases, env.data["phases"]),
            "q_injection_kvar": _phase_result(
                q_bess_injection, bess.phases, env.data["phases"]
            ),
            "soc_after_frac": bess.soc,
        }

    return applied

def solve_power_flow(env):
    """
    Solves the OpenDSS power flow and updates the results.
    """

    env.dss.text("Set Tolerance=1e-8")
    env.dss.solution.solve()
    if not env.dss.solution.converged:
        raise RuntimeError(f"OpenDSS did not converge at timestep {env.idx}")

    # Bus voltages
    for bus in env.results.voltages:
        env.dss.circuit.set_active_bus(bus)
        nodes = env.dss.bus.nodes
        voltage_values = env.dss.bus.vmag_angle
        pu_values = env.dss.bus.pu_vmag_angle
        voltage_by_phase = {}
        pu_by_phase = {}
        angle_by_phase = {}
        for position, node in enumerate(nodes):
            if node not in _PHASE_NAME:
                continue
            phase = _PHASE_NAME[node]
            voltage_by_phase[phase] = voltage_values[2 * position]
            pu_by_phase[phase] = pu_values[2 * position]
            angle_by_phase[phase] = pu_values[2 * position + 1]
            env.results.phase_voltages[bus].setdefault(phase, np.zeros(env.steps))[env.idx] = voltage_by_phase[phase]
            env.results.phase_voltages_pu[bus].setdefault(phase, np.zeros(env.steps))[env.idx] = pu_by_phase[phase]
            env.results.phase_angles_deg[bus].setdefault(phase, np.zeros(env.steps))[env.idx] = angle_by_phase[phase]
        first_phase = next(iter(pu_by_phase))
        env.results.voltages[bus][env.idx] = voltage_by_phase[first_phase]
        env.results.voltages_pu[bus][env.idx] = pu_by_phase[first_phase]

    # Grid power
    grid_kw = env.dss.circuit.total_power[0]
    grid_kvar = -env.dss.circuit.total_power[1]

    env.grid.array_kw.append(grid_kw)
    env.grid.array_kvar.append(grid_kvar)

    # Cost
    cost = (-grid_kw * env.grid.prices[env.idx]* env.dt)
    env.results.costs.append(cost)

    env.current_grid_kw = grid_kw
    env.current_grid_kvar = grid_kvar
    env.current_cost = cost
    env.current_voltages_pu = {
        bus: env.results.voltages_pu[bus][env.idx]
        for bus in env.results.voltages_pu
    }
    env.current_bus_voltages_pu = {
        bus: {
            phase: values[env.idx]
            for phase, values in phases.items()
        }
        for bus, phases in env.results.phase_voltages_pu.items()
    }
    env.current_bus_angles_deg = {
        bus: {
            phase: values[env.idx]
            for phase, values in phases.items()
        }
        for bus, phases in env.results.phase_angles_deg.items()
    }

    return grid_kw, grid_kvar, cost
