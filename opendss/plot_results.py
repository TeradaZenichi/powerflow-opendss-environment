'''
Plots power flow results from the simulation.
'''

import matplotlib.pyplot as plt
import numpy as np
from .simulation import run_simulation
from .data import load_data

def plot_power_flow(data):
       results = run_simulation(data)

       hours = results["hours"]
       load_kw = results["load_kw"]
       pv_kw = results["pv_kw"]
       bess_kw = results["bess_kw"]
       grid_power = results["grid_power"]

       fig, ax = plt.subplots(figsize=(10,5))

       ax.axhline(0, color='k', linestyle='-.')
       ax.grid(color='lightgrey')

       ax.plot(hours, load_kw,
              color='k',
              marker='o',
              label='Load')

       ax.bar(hours,
              -grid_power,
              color='lightcoral',
              label='Grid')

       ax.bar(hours,
              bess_kw,
              bottom=-grid_power,
              color='dodgerblue',
              label='BESS')

       ax.bar(hours,
              -pv_kw,
              bottom=(grid_power/2)+np.abs(grid_power)/2,
              color='gold',
              label='PV Generation')

       ax.set_title("Active power")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("Power [kW]")

       ax.set_xticks(hours)
       ax.set_xlim(1,24)

       ax.legend(loc='upper right')
       ax.set_axisbelow(True)

       #plt.tight_layout()
       plt.show()