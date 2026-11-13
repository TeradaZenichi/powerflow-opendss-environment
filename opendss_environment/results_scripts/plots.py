"""Export electrical and energy plots for a complete simulation episode."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_plots(results, output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_power(results, output_dir, reactive=False)
    _plot_power(results, output_dir, reactive=True)
    _plot_voltages(results, output_dir, per_unit=False)
    _plot_voltages(results, output_dir, per_unit=True)
    _plot_costs(results, output_dir)
    _plot_battery(results, output_dir, energy=True)
    _plot_battery(results, output_dir, energy=False)
    _plot_price_grid_import(results, output_dir)


def _figure(results, title, ylabel):
    hours = np.arange(results["steps"]) * results["dt"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set(title=title, xlabel="Elapsed time [h]", ylabel=ylabel)
    ax.set_xlim(-results["dt"] / 2, hours[-1] + results["dt"] / 2)
    ax.grid(color="lightgrey")
    ax.set_axisbelow(True)
    return fig, ax, hours


def _save(fig, ax, path):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _total(devices, field, steps):
    return sum((np.asarray(getattr(device, field)) for device in devices), np.zeros(steps))


def _stack_bottoms(*series):
    """Stack positive and negative powers on their respective sides of zero."""
    data = np.vstack(series)
    positive = np.maximum(data, 0)
    negative = np.minimum(data, 0)
    return np.where(data >= 0, np.cumsum(positive, axis=0) - positive,
                    np.cumsum(negative, axis=0) - negative)


def _plot_power(results, output_dir, reactive):
    field = "array_kvar" if reactive else "array_kw"
    unit = "kvar" if reactive else "kW"
    title = "Reactive power" if reactive else "Active power"
    filename = "power_reactive_plot.png" if reactive else "power_flow_plot.png"
    fig, ax, hours = _figure(results, title, f"Power [{unit}]")
    load = _total(results["load_list"], field, results["steps"])
    pv = _total(results["pv_list"], field, results["steps"])
    bess = _total(results["bess_list"], field, results["steps"])
    grid = np.asarray(getattr(results["grid"], field)) * (1 if reactive else -1)
    ax.axhline(0, color="black", linestyle="-.")
    ax.plot(hours, load, color="black", label="Load")
    series = (pv, grid, bess)
    for values, bottom, color, label in zip(series, _stack_bottoms(*series),
                                            ("gold", "hotpink", "blueviolet"), ("PV", "Grid", "BESS")):
        ax.bar(hours, values, width=0.9 * results["dt"], bottom=bottom, color=color, label=label)
    _save(fig, ax, output_dir / filename)


def _plot_voltages(results, output_dir, per_unit):
    field = "phase_voltages_pu" if per_unit else "phase_voltages"
    unit = "pu" if per_unit else "V"
    filename = "bus_voltage_pu_plot.png" if per_unit else "bus_voltage_plot.png"
    fig, ax, hours = _figure(results, "Bus voltage magnitude", f"Voltage [{unit}]")
    for bus, phases in getattr(results["results"], field).items():
        for phase, values in phases.items():
            ax.plot(hours, values, linewidth=1, label=f"{bus}/{phase}")
    _save(fig, ax, output_dir / filename)


def _plot_costs(results, output_dir):
    fig, ax, hours = _figure(results, "Energy cost per interval", "Cost [currency units]")
    ax.bar(hours, results["results"].costs, width=0.9 * results["dt"], color="hotpink")
    _save(fig, ax, output_dir / "hourly_cost_plot.png")


def _plot_battery(results, output_dir, energy):
    title = "Battery energy" if energy else "Battery state of charge"
    unit = "kWh" if energy else "%"
    filename = "bess_energy_plot.png" if energy else "bess_soc_plot.png"
    fig, ax, hours = _figure(results, title, f"{title} [{unit}]")
    for bess in results["bess_list"]:
        values = np.asarray(bess.array_soc) * (bess.e_cap_kwh if energy else 100)
        ax.plot(hours, values, label=f"BESS {bess.id}")
    if not results["bess_list"]:
        ax.text(0.5, 0.5, "No BESS in this scenario", ha="center", transform=ax.transAxes)
    _save(fig, ax, output_dir / filename)


def _plot_price_grid_import(results, output_dir):
    fig, ax, hours = _figure(results, "Electricity price and net grid import", "Price [currency/kWh]")
    energy_axis = ax.twinx()
    energy = -np.asarray(results["grid"].array_kw) * results["dt"]
    price_line, = ax.step(hours, results["grid"].prices, where="post", color="dodgerblue", label="Price")
    energy_line, = energy_axis.step(hours, energy, where="post", color="hotpink", label="Net grid import")
    energy_axis.set_ylabel("Net imported energy per interval [kWh]")
    energy_axis.legend(handles=[price_line, energy_line], loc="upper right")
    price_line.set_label("_nolegend_")
    _save(fig, ax, output_dir / "price_grid_import_plot.png")
