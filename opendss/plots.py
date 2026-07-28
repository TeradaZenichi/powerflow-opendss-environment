'''
Plots sim_results from the simulation and saves them to the specified output directory.
The plots include:
- Active power flow (load, PV generation, BESS, and grid)
- Bus voltage magnitudes
- Bus voltage magnitudes in per unit (pu)
- Hourly energy costs
'''

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_plots(sim_results, output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_power_flow(sim_results, output_dir)
    _plot_bus_voltages(sim_results, output_dir)
    _plot_bus_voltages_pu(sim_results, output_dir)
    _plot_hourly_costs(sim_results, output_dir)
    _plot_bess_energy_level(sim_results, output_dir)
    _plot_bess_soc(sim_results, output_dir)
    _plot_price_grid_import(sim_results, output_dir)

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


def _plot_power_flow(sim_results, output_dir):

       steps = sim_results["steps"]
       dt = sim_results["dt"]
       grid = sim_results["grid"]
       bess_list = sim_results["bess_list"]
       pv_list = sim_results["pv_list"]
       load_list = sim_results["load_list"]

       pv_kw = np.sum([pv.array_kw for pv in pv_list], axis=0)
       load_kw = np.sum([load.array_kw for load in load_list], axis=0)
       bess_kw = np.sum([bess.array_kw for bess in bess_list], axis = 0)
       grid_kw = -np.array(grid.array_kw)

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

def _plot_bus_voltages(sim_results, output_dir):

       steps = sim_results["steps"]
       voltages = sim_results["results"].voltages
       dt = sim_results["dt"]

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

def _plot_bus_voltages_pu(sim_results, output_dir):

       steps = sim_results["steps"]
       voltages = sim_results["results"].voltages_pu
       dt = sim_results["dt"]

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

def _plot_hourly_costs(sim_results, output_dir):

       steps = sim_results["steps"]
       costs = sim_results["results"].costs
       dt = sim_results["dt"]

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

def _plot_bess_energy_level(sim_results, output_dir):

       steps = sim_results["steps"]
       bess_list = sim_results["bess_list"]
       dt = sim_results["dt"]

       hours = np.arange(steps) * dt

       fig, ax = plt.subplots(figsize=(10, 5))

       ax.grid(color="lightgrey")

       for bess in bess_list:
              energy = np.array(bess.array_soc) * bess.e_cap_kwh

              ax.plot(
                     hours,
                     energy,
                     marker="o",
                     linewidth=2,
                     label=f"BESS {bess.id}"
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

def _plot_bess_soc(sim_results, output_dir):

       steps = sim_results["steps"]
       bess_list = sim_results["bess_list"]
       dt = sim_results["dt"]

       hours = np.arange(steps) * dt

       fig, ax = plt.subplots(figsize=(10, 5))

       ax.grid(color="lightgrey")

       for bess in bess_list:
              soc = np.array(bess.array_soc) * 100

              ax.plot(
                     hours,
                     soc,
                     marker="o",
                     linewidth=2,
                     label=f"BESS {bess.id}"
        )

       ax.set_title("Battery SoC Level")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("SoC [%]")

       ax.set_xticks(hours[::int(1/dt)])
       ax.set_xlim(hours[0] - dt/2, hours[-1] + dt/2)    

       ax.legend(loc="best")
       ax.set_axisbelow(True)

       plot_file = output_dir / "bess_soc_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)

def _plot_price_grid_import(sim_results, output_dir):

       prices = sim_results["grid"].prices
       grid_kw = -np.array(sim_results["grid"].array_kw)
       steps = sim_results["steps"]
       dt = sim_results["dt"]

       hours = np.arange(steps) * dt
       grid_kwh = grid_kw * dt

       fig, ax1 = plt.subplots(figsize=(10, 5))
       ax2 = ax1.twinx()

       ax1.grid(color="lightgrey")

       ax1.step(hours, prices, where="post", color="dodgerblue", linewidth=2, label="Electricity Price")
       ax2.step(hours, grid_kwh, where="post", color="hotpink", linewidth=2, label="Grid Import")

       ax1.set_title("Electricity Price and Grid Import")
       ax1.set_xlabel("Time [h]")
       ax1.set_ylabel("Price [$]")
       ax2.set_ylabel("Imported Energy [kWh]")

       ax1.set_xticks(hours[::int(1/dt)])
       ax1.set_xlim(hours[0] - dt/2, hours[-1] + dt/2)

       lines1, labels1 = ax1.get_legend_handles_labels()
       lines2, labels2 = ax2.get_legend_handles_labels()
       ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

       ax1.set_axisbelow(True)

       plot_file = output_dir / "price_grid_import_plot.png"
       plt.tight_layout()
       plt.savefig(plot_file)
       plt.close(fig)
