'''
Runs a power flow simulation for a given data set and returns the results as a dictionary.
The simulation uses the OpenDSS engine to solve the power flow for each time step, updating the power values of loads, PV generators, and BESS based on the provided data.
The results include: active power flow, bus voltage magnitudes, and hourly energy costs.
'''

import py_dss_interface
import numpy as np

def run_simulation(data):
    dss = py_dss_interface.DSS()
    dss.text("Clear")

    _simulation_setup(data, dss)

    bess_kw = np.array([ # BESS power profile for 24 hours (will change in the future for bess control)
        0,  0,  0,  0,  0,  0,
        8, 24, 48, 64, 76, 80,
        76, 68, 56, 40, 16,  0,
        0,  0,  0,  0,  0,  0
    ])

    grid_kw = np.array([])
    grid_kvar = np.array([])
    costs = np.array([])
    voltages = {bus: np.zeros(data["steps"]) for bus in dss.circuit.buses_names}
    voltages_pu = {bus: np.zeros(data["steps"]) for bus in dss.circuit.buses_names}

    for idx in range(data["steps"]):
        _update_snapshot_powers(data, dss, bess_kw, idx)
        dss.solution.solve()
        for bus in voltages:
            dss.circuit.set_active_bus(bus)
            voltages[bus][idx] = dss.bus.vmag_angle[0]
            voltages_pu[bus][idx] = dss.bus.vmag_angle[0]/(data["base_kv"] * 1000)
        grid_kw = np.append(grid_kw, dss.circuit.total_power[0])
        grid_kvar = np.append(grid_kvar, dss.circuit.total_power[1])
        costs = np.append(costs, _calculate_costs(data, grid_kw, idx))

    data["grid"].array_kw = grid_kw
    data["grid"].array_kvar = grid_kvar

    pv_kw = np.sum(
        [pv.profile for pv in data["pv_list"]],
        axis=0
    )
    
    load_kw = np.sum(
        [load.array_kw for load in data["load_list"]],
        axis=0
    )

    load_kvar = np.sum(
        [load.array_kvar for load in data["load_list"]],
        axis=0
    )

    return {
        "steps": data["steps"],
        "load_kw": load_kw,
        "load_kvar": load_kvar,
        "bess_kw": bess_kw,
        "pv_kw": pv_kw,
        "grid_kw": grid_kw,
        "grid_kvar": grid_kvar,
        "costs": costs,
        "voltages": voltages,
        "voltages_pu": voltages_pu
    }

def _simulation_setup(data,dss):
    '''
    Creates the OpenDSS circuit and adds all the elements to it.
    '''
    # Topology
    dss.text(f'compile "{data["topology"]}"')

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

def _update_snapshot_powers(data,dss,bess_kw,idx):
    '''
    Update the power values of the loads, PV generators and BESS for a given time step.
    '''
    for load in data["load_list"]:
        dss.text(
            f"Edit Load.{load.id} kw={load.array_kw[idx]:.2f} kvar={load.array_kvar[idx]:.2f}"
        )

    for pv in data["pv_list"]:
        dss.text(
            f"Edit Generator.{pv.id} kw={pv.profile[idx]:.2f} kvar=0"
        )

    for bess in data["bess_list"]:
        dss.text(
            f"Edit Load.{bess.id} kw={bess_kw[idx]:.2f} "
            f"kvar=0"
        )

def _calculate_costs(data, grid_kw, idx):
    '''
    Calculates the costs of the energy consumed from the grid at each time step.
    '''
    cost = -grid_kw[idx] * data["grid"].prices[idx] * (24/data["steps"])  # Prices are in $/kWh and grid_kw is in kW
    return cost