'''
Plots results from the simulation and saves them to the specified output directory.
The plots include:
- Active power flow (load, PV generation, BESS, and grid)
- Bus voltage magnitudes
- Bus voltage magnitudes in per unit (pu)
- Hourly energy costs
'''

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_plots(results, output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_power_flow(results, output_dir)
    _plot_bus_voltages(results, output_dir)
    _plot_bus_voltages_pu(results, output_dir)
    _plot_hourly_costs(results, output_dir)

def _plot_power_flow(results, output_dir):

       steps = results["steps"]
       load_kw = results["load_kw"]
       pv_kw = results["pv_kw"]
       bess_kw = results["bess_kw"]
       grid_kw = results["grid_kw"]

       hours = np.linspace(0, 24, steps, endpoint=False)

       fig, ax = plt.subplots(figsize=(10,5))

       ax.axhline(0, color='k', linestyle='-.')
       ax.grid(color='lightgrey')

       ax.plot(hours, load_kw,
              color='k',
              marker='o',
              label='Load')

       ax.bar(hours,
              -grid_kw,
              color='hotpink',
              label='Grid')

       ax.bar(hours,
              bess_kw,
              bottom=-grid_kw,
              color='blueviolet',
              label='BESS')

       ax.bar(hours,
              -pv_kw,
              bottom=(grid_kw/2)+np.abs(grid_kw)/2,
              color='gold',
              label='PV Generation')

       ax.set_title("Active power")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("Power [kW]")

       ax.set_xticks(np.arange(0, 24))
       ax.set_xlim(0, 24)

       ax.legend(loc='upper right')
       ax.set_axisbelow(True)

       plot_file = output_dir / "power_flow_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)

def _plot_bus_voltages(results, output_dir):

       steps = results["steps"]
       voltages = results["voltages"]

       hours = np.linspace(0, 24, steps, endpoint=False)

       fig, ax = plt.subplots(figsize=(10, 5))

       ax.grid(color="lightgrey")

       for bus, voltage in voltages.items():
              ax.plot(
              hours,
              voltage,
              marker="o",
              linewidth=2,
              label=bus
              )

       ax.set_title("Bus voltage magnitude")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("Voltage [V]")

       ax.set_xticks(np.arange(0, 24))
       ax.set_xlim(0, 24)

       ax.legend(loc="best")
       ax.set_axisbelow(True)

       plot_file = output_dir / "bus_voltage_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)

def _plot_bus_voltages_pu(results, output_dir):

       steps = results["steps"]
       voltages = results["voltages_pu"]

       hours = np.linspace(0, 24, steps, endpoint=False)

       fig, ax = plt.subplots(figsize=(10, 5))

       ax.grid(color="lightgrey")

       for bus, voltage in voltages.items():
              ax.plot(
              hours,
              voltage,
              marker="o",
              linewidth=2,
              label=bus
              )

       ax.set_title("Bus voltage magnitude")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("Voltage [pu]")

       ax.set_xticks(np.arange(0, 24))
       ax.set_xlim(0, 24)

       ax.legend(loc="best")
       ax.set_axisbelow(True)

       plot_file = output_dir / "bus_voltage_pu_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)

def _plot_hourly_costs(results, output_dir):

       steps = results["steps"]
       costs = results["costs"]

       hours = np.linspace(0, 24, steps, endpoint=False)

       fig, ax = plt.subplots(figsize=(10, 5))

       ax.grid(color="lightgrey")

       ax.bar(
              hours,
              costs,
              color="hotpink",
              edgecolor="black"
       )

       ax.set_title("Hourly energy cost")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("Cost [$]")

       ax.set_xticks(np.arange(0, 24, 1))
       ax.set_xlim(0, 24)

       ax.set_axisbelow(True)

       plot_file = output_dir / "hourly_cost_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)