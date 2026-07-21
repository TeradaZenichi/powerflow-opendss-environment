'''
Summarizes the simulation results and saves them to a JSON file.
'''

import json
import numpy as np
from pathlib import Path


def save_summary(results, output_dir):

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = results["steps"]
    load_kw = np.array(results["load_kw"])
    pv_kw = np.array(results["pv_kw"])
    bess_kw = np.array(results["bess_kw"])
    grid_kw = np.array(results["grid_kw"])
    costs = np.array(results["costs"])

    voltages_pu = results.get("voltages_pu", {})

    # Time step duration [hours]
    dt = 24 / steps

    # Energy calculations [kWh]
    load_energy = np.sum(load_kw) * dt
    pv_energy = np.sum(pv_kw) * dt

    # Grid exchange
    grid_import_energy = np.sum(np.abs(grid_kw[grid_kw < 0])) * dt
    grid_export_energy = np.sum(grid_kw[grid_kw > 0]) * dt

    # Battery energy
    bess_charge_energy = np.sum(np.abs(bess_kw[bess_kw > 0])) * dt
    bess_discharge_energy = np.sum(np.abs(bess_kw[bess_kw < 0])) * dt

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