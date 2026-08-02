'''
Summarizes the simulation results and saves them to a JSON file and CSV files.
'''

import json
import numpy as np
from pathlib import Path
import pandas as pd

def save_summary(sim_results, output_dir):

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = sim_results["steps"]
    dt = sim_results["dt"]
    timestamps = sim_results["timestamps"]

    grid = sim_results["grid"]
    bess_list = sim_results["bess_list"]
    pv_list = sim_results["pv_list"]
    load_list = sim_results["load_list"]

    costs = np.asarray(sim_results["results"].costs)
    voltages_pu = sim_results["results"].voltages_pu

    pv_kw = np.sum([pv.array_kw for pv in pv_list], axis=0)
    load_kw = np.sum([load.array_kw for load in load_list], axis=0)
    load_kvar = np.sum([load.array_kvar for load in load_list], axis=0)
    grid_kw = -np.asarray(grid.array_kw)

    load_energy = np.sum(load_kw) * dt
    pv_energy = np.sum(pv_kw) * dt

    grid_import_energy = np.sum(np.abs(grid_kw[grid_kw < 0])) * dt
    grid_export_energy = np.sum(grid_kw[grid_kw > 0]) * dt

    bess_charge_energy = 0
    bess_discharge_energy = 0

    for bess in bess_list:
        p = np.asarray(bess.array_kw)
        bess_charge_energy += np.sum(p[p > 0]) * dt
        bess_discharge_energy += np.sum(-p[p < 0]) * dt

    total_cost = np.sum(costs)

    voltage_values = np.concatenate(list(voltages_pu.values()))

    summary = {
        "simulation": {
            "time_steps": steps,
            "dt": dt
        },
        "energy": {
            "total_load_kwh": load_energy,
            "total_pv_generation_kwh": pv_energy,
            "grid_import_kwh": grid_import_energy,
            "grid_export_kwh": grid_export_energy,
            "bess_charge_kwh": bess_charge_energy,
            "bess_discharge_kwh": bess_discharge_energy
        },
        "cost": {
            "total_energy_cost": total_cost
        },
        "voltage_pu": {
            "minimum_voltage": float(np.min(voltage_values)),
            "maximum_voltage": float(np.max(voltage_values)),
            "average_voltage": float(np.mean(voltage_values))
        }
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # CSV
    devices_data = {
        "timestep": np.arange(steps),
        "timestamp": timestamps,
        "load_total_kw": load_kw,
        "load_total_kvar": load_kvar,
    }

    for bess in bess_list:

        p = np.asarray(bess.array_kw)
        q = np.asarray(bess.array_kvar)
        soc_frac = np.asarray(bess.array_soc)
        soc_kwh = soc_frac * bess.e_cap_kwh

        devices_data[f"bess_{bess.id}_p_net_kw"] = p
        devices_data[f"bess_{bess.id}_q_injection_kvar"] = q
        devices_data[f"bess_{bess.id}_p_charge_kw"] = np.maximum(p, 0)
        devices_data[f"bess_{bess.id}_p_discharge_kw"] = np.maximum(-p, 0)
        devices_data[f"bess_{bess.id}_soc_kwh"] = soc_kwh
        devices_data[f"bess_{bess.id}_soc_frac"] = soc_frac

    for pv in pv_list:
        devices_data[f"pv_{pv.id}_available_kw"] = np.asarray(pv.profile)
        devices_data[f"pv_{pv.id}_generation_kw"] = np.asarray(pv.array_kw)


    pd.DataFrame(devices_data).to_csv(
        output_dir / "sim_devices_timeseries.csv",
        index=False
    )

    grid_p = -np.asarray(grid.array_kw)       # + import, - export
    grid_q = np.asarray(grid.array_kvar)

    data_buses = {
        "timestep": np.arange(steps),
        "timestamp": timestamps,
        "price_per_kwh": np.asarray(grid.prices),
        "grid_import_kw": np.maximum(grid_p, 0),
        "grid_export_kw": np.maximum(-grid_p, 0),
        "grid_q_kvar": grid_q,
        "grid_cost": costs,
    }

    # Group loads by bus
    loads_by_bus = {
        load.bus: (
            np.asarray(load.array_kw),
            np.asarray(load.array_kvar)
        )
        for load in load_list
    }

    # Add voltage and load columns bus by bus
    for bus, v_pu in voltages_pu.items():

        data_buses[f"{bus}_voltage_pu"] = np.asarray(v_pu)

        if bus in loads_by_bus:
            p_load, q_load = loads_by_bus[bus]
        else:
            p_load = np.zeros(steps)
            q_load = np.zeros(steps)

        data_buses[f"{bus}_p_load_kw"] = p_load
        data_buses[f"{bus}_q_load_kvar"] = q_load

    pd.DataFrame(data_buses).to_csv(
        output_dir / "sim_buses_timeseries.csv",
        index=False,
    )

    return summary