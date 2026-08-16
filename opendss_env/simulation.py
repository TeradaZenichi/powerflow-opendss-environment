'''
Functions to setup and run the OpenDSS simulation.
''' 

import numpy as np
from .devices_control import bess_control, pv_control, BESS_KW, BESS_KVAR, PV_KVAR

def _simulation_setup(env):
    """
    Creates the OpenDSS circuit using the devices defined in the case.
    """

    env.dss.text("Clear")
    env.dss.text(f'compile "{env.data["topology"]}"')
    env.dss.text("Vsource.source.model=Ideal")

    # PV generators
    for pv in env.episodes[0]["pv_list"]:
        env.dss.text(f"""New Generator.{pv.id} bus1={pv.bus} phases={env.data["phases"]} kv={env.data["base_kv"]} kw=0 kvar=0""")

    # BESS
    for bess in env.episodes[0]["bess_list"]:
        env.dss.text( f""" New Load.{bess.id} bus={bess.bus} phases={env.data["phases"]} kv={env.data["base_kv"]} kw=0 kvar=0 conn=y""")

    # Loads
    for load in env.episodes[0]["load_list"]:
        env.dss.text( f""" New Load.{load.id} bus1={load.bus} phases={env.data["phases"]} kv={env.data["base_kv"]} kw=0 kvar=0""")

def _update_snapshot_powers(env):
    """
    Updates all loads, PVs and BESSs for the current time step.
    """

    # Loads
    for load in env.load_list:
        env.dss.text( f"Edit Load.{load.id} kw={load.array_kw[env.idx]} kvar={load.array_kvar[env.idx]}")

    # PV
    for pv_idx, pv in enumerate(env.pv_list):
        p_pv, q_pv_injection = pv_control(pv, env.idx, PV_KVAR)
        env.dss.text(f"Edit Generator.{pv.id} kw={p_pv} kvar={q_pv_injection}")

    # BESS
    for bess_idx, bess in enumerate(env.bess_list):
        p_bess, q_bess_injection = bess_control(bess, env.idx, env.dt, BESS_KW, BESS_KVAR)
        env.dss.text(f"Edit Load.{bess.id} kw={p_bess} kvar={-q_bess_injection}")

def solve_power_flow(env):
    """
    Solves the OpenDSS power flow and updates the results.
    """

    env.dss.text("Set Tolerance=1e-8")
    env.dss.solution.solve()

    # Bus voltages
    for bus in env.results.voltages:
        env.dss.circuit.set_active_bus(bus)
        voltage = env.dss.bus.vmag_angle[0]
        env.results.voltages[bus][env.idx] = voltage
        env.results.voltages_pu[bus][env.idx] = (voltage / (env.data["base_kv"] * 1000))

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
    env.current_voltages_pu = {bus: env.results.voltages_pu[bus][env.idx] for bus in env.results.voltages_pu}

    return grid_kw, grid_kvar, cost