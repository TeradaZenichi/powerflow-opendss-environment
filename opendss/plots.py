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
    _plot_bess_energy_level(results, output_dir)

import numpy as np
import matplotlib.pyplot as plt

def _stack_bottoms(*series):
       '''
       Calculates the bottom positions for stacked bar plots, 
       ensuring that positive and negative values are stacked separately.
       '''
       data = np.vstack(series)

       pos = np.maximum(data, 0) # Takes the positive values of the data
       neg = np.minimum(data, 0) # Takes the negative values of the data

       # Calculate the cumulative sum for positive and negative values separately
       pos_bottom = np.vstack([np.zeros(data.shape[1]), np.cumsum(pos, axis=0)[:-1]])
       neg_bottom = np.vstack([np.zeros(data.shape[1]), np.cumsum(neg, axis=0)[:-1]])

       # If the original data is positive, use the positive bottom; otherwise, use the negative bottom
       return np.where(data >= 0, pos_bottom, neg_bottom)


def _plot_power_flow(results, output_dir):

       steps = results["steps"]
       load_kw = results["load_kw"]
       pv_kw = results["pv_kw"]      
       bess_kw = results["bess_kw"]
       grid_kw = -results["grid_kw"]  
       dt = results["dt"]

       hours = np.arange(steps) * dt

       fig, ax = plt.subplots(figsize=(10, 5))

       ax.axhline(0, color="k", linestyle="-.")
       ax.grid(color="lightgrey")

       ax.plot(hours, load_kw,
              color="k",
              marker="o",
              label="Load")

       series = [pv_kw, grid_kw, bess_kw]
       bottoms = _stack_bottoms(*series)

       colors = ["gold", "hotpink", "blueviolet"]
       labels = ["PV Generation", "Grid", "BESS"]

       for y, bottom, color, label in zip(series, bottoms, colors, labels):
              ax.bar(hours, y, width=0.9*dt, bottom=bottom, color=color, label=label)

       ax.set_title("Active power")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("Power [kW]")

       ax.set_xticks(hours[::int(1/dt)])
       ax.set_xlim(hours[0] - dt/2, hours[-1] + dt/2)       
       ax.legend(loc="upper right")
       ax.set_axisbelow(True)

       plt.tight_layout()
       plt.savefig(output_dir / "power_flow_plot.png")

def _plot_bus_voltages(results, output_dir):

       steps = results["steps"]
       voltages = results["voltages"]
       dt = results["dt"]

       hours = np.arange(steps) * dt

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

       ax.set_xticks(hours[::int(1/dt)])
       ax.set_xlim(hours[0] - dt/2, hours[-1] + dt/2)    

       ax.legend(loc="best")
       ax.set_axisbelow(True)

       plot_file = output_dir / "bus_voltage_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)

def _plot_bus_voltages_pu(results, output_dir):

       steps = results["steps"]
       voltages = results["voltages_pu"]
       dt = results["dt"]

       hours = np.arange(steps) * dt

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

       ax.set_xticks(hours[::int(1/dt)])
       ax.set_xlim(hours[0] - dt/2, hours[-1] + dt/2)    

       ax.legend(loc="best")
       ax.set_axisbelow(True)

       plot_file = output_dir / "bus_voltage_pu_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)

def _plot_hourly_costs(results, output_dir):

       steps = results["steps"]
       costs = results["costs"]
       dt = results["dt"]

       hours = np.arange(steps) * dt

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

       ax.set_xticks(hours[::int(1/dt)])
       ax.set_xlim(hours[0] - dt/2, hours[-1] + dt/2)   

       ax.set_axisbelow(True)

       plot_file = output_dir / "hourly_cost_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)

def _plot_bess_energy_level(results, output_dir):

       steps = results["steps"]
       bess_energy = results["bess_energy"]
       dt = results["dt"]

       hours = np.arange(steps) * dt

       fig, ax = plt.subplots(figsize=(10, 5))

       ax.grid(color="lightgrey")

       ax.plot(
              hours,
              bess_energy,
              marker="o",
              linewidth=2,
              color="deeppink",
              label="BESS Energy"
       )

       ax.set_title("Battery Energy Level")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("Energy [kWh]")

       ax.set_xticks(hours[::int(1/dt)])
       ax.set_xlim(hours[0] - dt/2, hours[-1] + dt/2)    

       ax.legend(loc="best")
       ax.set_axisbelow(True)

       plot_file = output_dir / "bess_energy_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)