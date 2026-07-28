'''
Summarizes the simulation results and saves them to a JSON file.
'''

import json
import numpy as np
from pathlib import Path

def save_summary(sim_results, output_dir):

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = sim_results["steps"]
    dt = sim_results["dt"]
    grid = sim_results["grid"]
    bess_list = sim_results["bess_list"]
    pv_list = sim_results["pv_list"]
    load_list = sim_results["load_list"]
    costs = np.array(sim_results["results"].costs)

    voltages_pu = sim_results["results"].voltages_pu

    pv_kw = np.sum([pv.array_kw for pv in pv_list], axis=0)
    load_kw = np.sum([load.array_kw for load in load_list], axis=0)
    bess_kw = np.sum([bess.array_kw for bess in bess_list], axis = 0)
    grid_kw = -np.array(grid.array_kw)

    # Energy calculations [kWh]
    load_energy = np.sum(load_kw) * dt
    pv_energy = np.sum(pv_kw) * dt

    # Grid exchange
    grid_import_energy = np.sum(np.abs(grid_kw[grid_kw < 0])) * dt
    grid_export_energy = np.sum(grid_kw[grid_kw > 0]) * dt

    # Battery energy
    bess_charge_energy = 0
    bess_discharge_energy = 0

    for bess in bess_list:
        bess_kw = np.array(bess.array_kw)

        bess_charge_energy += np.sum(bess_kw[bess_kw > 0]) * dt
        bess_discharge_energy += np.sum(np.abs(bess_kw[bess_kw < 0])) * dt

    # Cost
    total_cost = np.sum(costs)

    # Voltage statistics
    voltage_values = np.concatenate(list(voltages_pu.values()))

    min_voltage = np.min(voltage_values)
    max_voltage = np.max(voltage_values)
    avg_voltage = np.mean(voltage_values)

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
            "minimum_voltage": min_voltage,
            "maximum_voltage": max_voltage,
            "average_voltage": avg_voltage
        }
    }

    summary_file = output_dir / "summary.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary