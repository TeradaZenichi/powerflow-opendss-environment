'''
Runs a power flow simulation for a given data set and returns the results as a dictionary.
The simulation uses the OpenDSS engine to solve the power flow for each time step, updating the power values of loads, PV generators, and BESS based on the provided data.
The results include: updated elements with active and reactive power arrays, bus voltage magnitudes and hourly energy costs.
''' 

import py_dss_interface
import numpy as np
from .devices_control import bess_control, pv_control

def run_simulation(data):
    dss = py_dss_interface.DSS()
    dss.text("Clear")

    _simulation_setup(data, dss)

    data["results"].voltages = {bus: np.zeros(data["steps"]) for bus in dss.circuit.buses_names}
    data["results"].voltages_pu = {bus: np.zeros(data["steps"]) for bus in dss.circuit.buses_names}

    for idx in range(data["steps"]):
        _update_snapshot_powers(data, dss, idx)
        dss.text("Set Tolerance=1e-8")
        dss.solution.solve()
        for bus in data["results"].voltages:
            dss.circuit.set_active_bus(bus)
            data["results"].voltages[bus][idx] = dss.bus.vmag_angle[0]
            data["results"].voltages_pu[bus][idx] = dss.bus.vmag_angle[0]/(data["base_kv"] * 1000)
        data["grid"].array_kw.append(dss.circuit.total_power[0])
        data["grid"].array_kvar.append(dss.circuit.total_power[1])
        data["results"].costs.append(-data["grid"].array_kw[idx] * data["grid"].prices[idx] * data["dt"])

    return {
        "dt": data["dt"],
        "steps": data["steps"],
        "timestamps": data["timestamps"],
        "grid": data["grid"], 
        "bess_list": data["bess_list"], 
        "pv_list": data["pv_list"], 
        "load_list": data["load_list"],
        "results": data["results"]
    }

def _simulation_setup(data,dss):
    '''
    Creates the OpenDSS circuit and adds all the elements to it.
    '''
    # Topology
    dss.text(f'compile "{data["topology"]}"')

    dss.text("Vsource.source.model=Ideal")

    # PV generators
    for pv in data["pv_list"]:
        dss.text(f"""
        New Generator.{pv.id} bus1={pv.bus} phases={data["phases"]} kv={data["base_kv"]} kw=0 kvar=0
        """)

    # BESS
    for bess in data["bess_list"]:
        dss.text(f"""
        New Load.{bess.id} bus={bess.bus} phases={data["phases"]} kv={data["base_kv"]} kw=0 kvar=0 conn=y
        """)

    # Load demand
    for load in data["load_list"]:
        dss.text(f"""
        New Load.{load.id} bus1={load.bus} phases={data["phases"]} kv={data["base_kv"]} kw=0 kvar=0
        """)

def _update_snapshot_powers(data, dss, idx):
    '''
    Update the power values of the loads, PV generators and BESS for a given time step.
    '''
    for load in data["load_list"]:
        dss.text(
            f"Edit Load.{load.id} kw={load.array_kw[idx]} kvar={load.array_kvar[idx]}"
        )

    for pv in data["pv_list"]:
        p_pv, q_pv = pv_control(pv,idx)
        dss.text(
            f"Edit Generator.{pv.id} kw={p_pv} kvar={q_pv}"
        )

    for bess in data["bess_list"]:
        p_bess, q_bess = bess_control(bess,idx,data["dt"])
        dss.text(
            f"Edit Load.{bess.id} kw={p_bess} kvar={q_bess}"
        )