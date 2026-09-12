# Powerflow OpenDSS Environment

Gymnasium environment for simulating microgrids and radial distribution networks using [OpenDSS](https://sourceforge.net/projects/electricdss/) as the power-flow solver.

The environment provides a simulation framework in which loads, photovoltaic (PV) systems, and battery energy storage systems (BESS) are simulated over discrete time steps. At each step, device operating conditions are updated, an OpenDSS power flow is solved, electrical quantities are collected, and a reward is calculated.

The environment is designed to support future integration with optimization and machine learning methods, in which BESS and PV operating commands can be selected by an agent.

For use as a module from a separate training repository:

```powershell
python -m pip install -e path\to\powerflow-opendss-environment
```

## Running the example

The main simulation is configured in `main_env.py`. The main runtime options are defined at the top of the file:

```Python
CASE_PATH = project_dir / "examples" / "case5"

EPISODE_STEPS = 24
NUM_EPISODES = 1
START_EPISODE = 0

env = MicrogridEnv(
    case_path=CASE_PATH,
    episode_steps=EPISODE_STEPS,
    num_episodes=NUM_EPISODES,
    start_episode=START_EPISODE,
    state_functions=[
        get_hour,
        get_price,
        get_previous_pv_kw,
        get_bess_soc,
        get_previous_load_kw,
        get_current_load_kw
    ],
    reward_function=minimize_cost,
)
```

The main parameters are:

* `CASE_PATH`: path to the simulation case;
* `EPISODE_STEPS`: number of time steps in each episode;
* `NUM_EPISODES`: number of episodes to simulate;
* `START_EPISODE`: first episode to run;
* `state_functions`: variables included in the environment state;
* `reward_function`: reward used by the environment.

Run:

```
.\.venv\Scripts\python.exe main_env.py
```

After the simulation, the results are written to:

```
outputs/case5/
├── plots/
└── summary/
```

The `outputs` directory contains the generated plots, summary JSON, and bus and device time-series CSV files.

## Input files

The `case5` directory is the reference example:

```text
examples/case5/
├── config.json
├── devices.json
├── demand.csv
├── price.csv
├── pv.csv
└── dss/
    └── Master.dss
```

The exact case files can be adapted to represent different microgrids or distribution networks.

| File               | Contents                                                                    |
| ------------------ | --------------------------------------------------------------------------- |
| `config.json`    | Simulation parameters, network information, system bases, and configuration |
| `devices.json`   | BESS and PV parameters and device locations                                 |
| `demand.csv`     | Active and reactive load profiles                                           |
| `price.csv`      | Electricity price profile                                                   |
| `pv.csv`         | PV availability profile                                                     |
| `dss/Master.dss` | OpenDSS network topology and electrical parameters                          |

`MicrogridEnv` accepts either the case directory or the JSON configuration
file. Paths inside the configuration are relative to that file, so an external
training repository can own one scenario and pass the same path to both this
environment and `opf-teacher`. The optional `files` object selects the demand,
price, and device files; conventional names are used when it is absent. Both
modules currently use `"schema_version": 1`.

```python
case_source = "scenarios/network_01/config.json"  # directory also accepted
env = MicrogridEnv(case_path=case_source, ...)
```

The case data is loaded by `data.py`. The data loader also divides the available time series into episodes according to the parameters provided to the environment.

## Simulation with OpenDSS

OpenDSS is used as the electrical power-flow solver.

When the environment is initialized, `_simulation_setup()` in `simulation.py` creates the OpenDSS circuit from the topology specified in the case. Loads, PV generators, and BESS devices are then created in the OpenDSS circuit.

At each simulation step, `_update_snapshot_powers()` updates the operating point of the devices and loads for the current time index.

After the solution, the environment records:

* bus voltage magnitudes;
* bus voltage magnitudes in per unit;
* grid active power;
* grid reactive power;
* energy cost;
* device operating quantities.

---

# BESS and PV control

Device operation is currently defined in:

```text
opendss_env/devices_control.py
```

This file contains the control functions that determine the operating commands applied to the BESS and PV systems.

When `step(None)` is used, the environment falls back to predefined time-series
commands. `step(action)` accepts named commands supplied by an external
controller.

## BESS control

The current BESS controller uses two predefined arrays:

```python
BESS_KW
BESS_KVAR
```

At each time step, the corresponding active and reactive power commands are passed to:

```python
bess_control()
```

which calls:

```python
bess.operate(
    bess_kw,
    bess_kvar,
    dt,
)
```

The `operate()` method receives the requested powers and applies the device operational constraints, including power and energy limits.

Therefore, the command requested by the controller may be adjusted by the BESS model to keep the device within its feasible operating range.

The current implementation supports active and reactive BESS commands from
either the baseline profiles or `step(action)`.

## PV control

The current PV controller uses a predefined reactive-power profile:

```python
PV_KVAR
```

The active power comes from the PV availability profile:

```python
pv.profile[idx]
```

and the command is applied through:

```python
pv_control()
```

which calls:

```python
pv.operate(
    pv.profile[idx],
    pv_kvar[idx],
)
```

The `operate()` method applies the PV operating constraints before the resulting operating point is sent to OpenDSS.

## External device control

The predefined control remains a baseline. An external controller or machine
learning agent can provide named device actions at each time step.

The intended action space includes:

```text
BESS:
    P_BESS
    Q_BESS

PV:
    P_PV
    Q_PV
```

The action passed to `step()` uses the following structure:

```python
{
    "bess": {
        "b1": {"p_net_kw": ..., "q_injection_kvar": ...},
    },
    "pv": {
        "pv1": {"generation_kw": ..., "q_injection_kvar": ...},
    },
}
```

Power values may be aggregate scalars or dictionaries indexed by phase. The
`info` returned by `step()` includes both `requested_action` and
`executed_action` after device limits, plus `bus_voltages_pu`,
`bus_angles_deg`, and grid exchange for the solved snapshot.

`info["device_measurements"]` contains the terminal powers read back from the
solved OpenDSS circuit, by phase and in aggregate. BESS uses `p_net_kw > 0` for
charging and `q_injection_kvar > 0` for reactive injection. PV uses positive
`generation_kw` and positive `q_injection_kvar` for injection. This makes the
requested, device-limited, and electrically measured operation independently
comparable.

`env.observe()` returns a named pre-action observation with
`observation_schema_version`, `timestamp`,
`dt_h`, phase order, bus voltage/angle/load by phase, grid prices, BESS SoC and
previous terminal powers, and PV availability and previous terminal powers.
The same observation is returned in `reset()` info and as `info["observation"]`
for the action applied by each `step()`. The original numeric state remains
available through `state_functions` for backward compatibility. Temporal
history is intentionally assembled by the training application.

Each environment owns an independent OpenDSS context, so multiple environment
instances can coexist in one process without sharing the active circuit.

The control flow will then become:

```text
Environment state
       │
       ▼
      Agent
       │
       ▼
     Action
       │
       ├── BESS P
       ├── BESS Q
       └── PV Q
       │
       ▼
Device operational limits
       │
       ▼
OpenDSS power flow
       │
       ▼
Next state + reward
```

---

# Gymnasium environment

The main environment is implemented in:

```text
opendss_env/microgrid_env.py
```

through the:

```python
MicrogridEnv
```

class.

The environment follows the standard Gymnasium interaction:

```python
state, info = env.reset()

state, reward, terminated, truncated, info = env.step(action)
```

## Episode

An episode is a sequence of consecutive simulation time steps. Its length is defined by `EPISODE_STEPS`.

When `reset()` is called, the environment:

1. loads the selected episode data;
2. resets the simulation index and episode reward;
3. initializes the result arrays;
4. builds the initial state.

## Step

At each `step(action)`, the environment:

1. updates the loads and device operating points;
2. solves the OpenDSS power flow;
3. stores the electrical results;
4. calculates the reward;
5. advances the simulation index;
6. builds the next state.

An episode terminates when the number of simulated steps reaches `EPISODE_STEPS`.

When `action` is provided, it determines BESS and PV operation. With
`action=None`, the predefined profiles in `devices_control.py` are used as the
baseline.

## States

States are defined in:

```text
opendss_env/states.py
```

The environment allows the user to select which state variables are provided to the controller.

For example:

```python
state_functions=[
    get_hour,
    get_price,
    get_previous_pv_kw,
    get_bess_soc,
    get_previous_load_kw,
    get_current_load_kw,
]
```

The available state functions currently include:

| Function                 | Description                |
| ------------------------ | -------------------------- |
| `get_hour`             | Current hour               |
| `get_price`            | Current electricity price  |
| `get_bess_soc`         | Total BESS state of charge |
| `get_previous_pv_kw`   | Previous PV active power   |
| `get_previous_load_kw` | Previous load active power |
| `get_current_load_kw`  | Current load active power  |

The state is assembled by:

```python
build_state()
```

which evaluates the selected state functions and returns a NumPy array.

For example:

```python
state = np.array([
    current_hour,
    current_price,
    previous_pv_power,
    bess_soc,
    previous_load,
    current_load,
])
```

The state definition is configurable, allowing different observation spaces to be tested without changing the main environment.

## Rewards

Reward functions are defined in:

```text
opendss_env/rewards.py
```

The current implementation includes a cost-based reward:

```python
def minimize_cost(env):
    return -env.results.costs[env.idx]
```

Since the objective is to minimize the operating cost, the negative cost is used as the reward.

A voltage-deviation reward is also available:

```python
def minimize_voltage_deviation(env):
    deviation = 0.0

    for bus in env.results.voltages_pu:
        voltage = env.results.voltages_pu[bus][env.idx]
        deviation += abs(voltage - 1.0)

    return -deviation
```

This reward penalizes deviations of the bus voltage magnitudes from 1.0 pu.

The reward function is passed directly to the environment:

```python
env = MicrogridEnv(
    ...,
    reward_function=minimize_cost,
)
```

This structure allows additional objectives or reward functions to be added independently of the environment implementation.

---

# Results and outputs

Simulation results are generated by the scripts in:

```text
opendss_env/results_scripts/
├── plots.py
├── results.py
└── summary.py
```

The main results interface is:

```python
simulation_results(results, CASE_PATH)
```

called after an episode has finished.

The generated files are stored under:

```text
outputs/
└── case5/
    ├── plots/
    └── summary/
```

## Plots

The `plots.py` script generates plots for the main electrical and operational quantities, including:

* BESS energy;
* bus voltage magnitudes;
* energy cost by time step;
* active power flow;
* reactive power flow;
* electricity price versus grid import.

These plots provide a quick visual assessment of the network and device operation during the simulation.

## Summary

The `summary.py` script generates a JSON summary containing the main simulation indicators.

Example:

```json
{
    "simulation": {
        "time_steps": 24,
        "dt": 1.0
    },
    "energy": {
        "total_load_kwh": 11879.4,
        "total_pv_generation_kwh": 1769.4104962053088,
        "grid_import_kwh": 10214.645558885826,
        "grid_export_kwh": 0.0,
        "bess_charge_kwh": 210.1364216418943,
        "bess_discharge_kwh": 188.657592,
        "bess_reactive_loss_kwh": 1.042584706675009,
        "pv_reactive_loss_kwh": 0.5895037946912434
    },
    "cost": {
        "total_energy_cost": 6163.920018093198
    },
    "voltage_pu": {
        "minimum_voltage": 0.9783265308704857,
        "maximum_voltage": 0.999999196869168,
        "average_voltage": 0.9924105129847486
    }
}
```

The summary includes:

* number of simulation steps;
* simulation time step;
* total load energy;
* total PV generation;
* grid import and export;
* BESS charging and discharging energy;
* reactive-power-related losses;
* total energy cost;
* minimum, maximum, and average voltage magnitude.

## Time-series CSV files

The simulation also generates time-series data for buses and devices:

```text
outputs/case5/summary/
├── summary.json
├── sim_buses_timeseries.csv
└── sim_devices_timeseries.csv
```

`sim_buses_timeseries.csv` contains time-series information for the network buses.

`sim_devices_timeseries.csv` contains time-series information for the simulated devices, including BESS and PV operating quantities.

These files can be used for further analysis, visualization, validation, or dataset generation for learning-based controllers.

---

# Repository structure

The main repository structure is:

```text
powerflow-opendss-environment/
│
├── examples/                     Contains example network cases and their input data.
│   └── case5/
│       ├── config.json
│       ├── devices.json
│       ├── demand.csv
│       ├── price.csv
│       ├── pv.csv
│       └── dss/
│           └── Master.dss
│
├── opendss_env/
│   ├── __init__.py
│   ├── data.py                   Loads the case data and prepares the data used by each simulation episode.
│   ├── devices_control.py        Defines the current BESS and PV control strategies.
│   ├── elements.py               Contains the classes representing the main simulation elements, such as BESS, PV, loads, and grid-related objects.
│   ├── microgrid_env.py          Implements the Gymnasium environment and controls the simulation lifecycle
│   ├── rewards.py                Contains reward functions that can be selected by the user.
│   ├── simulation.py             Contains the OpenDSS simulation routines, including circuit initialization, device and load updates, power-flow execution, and result collection.
│   ├── states.py                 Contains the functions used to construct the observation/state provided to the controller.
│   │ 
│   └── results_scripts/          Contains the post-processing routines used to generate plots, CSV files, and simulation summaries.
│       ├── plots.py
│       ├── results.py
│       └── summary.py
│
├── outputs/
│   └── case5/
│       ├── plots/
│       └── summary/
│
├── main_env.py                   Provides an example of how to instantiate and run the environment.
└── README.md
```
