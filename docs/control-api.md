# Profiles, reporting and RL integration

## Device profiles

Each PV's `profile` in `devices.json` is an explicit `file.csv:column` reference.
Use `pv.csv:pv1` and `pv.csv:pv2` for separate columns in one file, or separate
files with the same column name. Profiles are stored by device ID, so neither
case overwrites another device's data. Three-phase `phase_profiles` continue to
select each phase explicitly; their sum must match the total profile.

## Independent export tariffs

The price CSV accepts `timestamp,price_per_kwh,sell_price_per_kwh`.
The optional sell column takes precedence over `grid.feed_in_tariff_ratio`.
When absent, the ratio retains its existing behavior. Both tariffs must be
finite, and price timestamps must match demand timestamps exactly.
Billing uses total net exchange at the grid connection, summed across phases.
The teacher uses the same convention, including horizon slicing and pair replay.
CSV loads use constant P/Q within 0–2 pu, matching the teacher's load model;
the OpenDSS fallback to impedance outside its default voltage band is disabled
over that range. Electrical limit violations are still reported separately.

## Executed diagnostics and reports

`step()` returns `info["diagnostics"]` with:

- `voltage_violations`: bus, phase, voltage and excess outside the configured
  voltage band (default 0.95–1.05 pu).
- `branch_overloads`: line/transformer terminal phases above their OpenDSS
  normal current rating. Each branch measurement includes `normal_current_a`
  and `loading_pct`; transformer limits account for winding voltage levels.
- `grid_excess_pu`: excess above configured active import/export and reactive
  exchange limits, on the total network power base.
- `within_limits`: whether the checks above found any violation.
- `unrated_branch_terminals`: phase-terminal measurements without a usable
  current rating. Such measurements are excluded from overload checks.

These diagnostics do not clip network flows, change reward, or stop the episode.
They cover the configured limits and available ratings, not a safety certificate.
Details stay in `info`; `save_summary()` exports per-step counts to
`sim_diagnostics_timeseries.csv`.

The bus CSV retains aggregate columns and adds voltage, angle and load by phase.
Bus load totals sum every connected phase. Voltage extrema and plots include
every phase. Device CSVs add BESS/PV powers by phase. BESS charge/discharge
columns and energy totals sum positive/negative phase powers separately, so
opposite phase powers are visible even when total net power is zero.

## Optional Gymnasium adapter

The named API remains unchanged. Use `GymAdapter` for flat numeric `Box` spaces:

```python
from opendss_environment import EnvironmentConfig, GymAdapter, MicrogridEnv

physical = MicrogridEnv(EnvironmentConfig("data/case5", episode_steps=96))
paths = [
    ("buses", "bus_004", "v_before_pu", "a"),
    ("bess", "b1", "soc_before_frac"),
]
env = GymAdapter(physical, observation_paths=paths, devices={"bess": ["b1"]})
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`observation_paths` defines the exact observation order. When omitted, the
adapter includes numeric bus, BESS, PV and grid observations from the whole
network; use explicit local paths for a local controller. Paths may select
phase voltages, local branch measurements, angles, loads and device states.

`devices` selects controlled IDs. Other BESS receive zero P/Q and other PV
receive available generation with Q=0, subject to their configured controls.
When omitted, every device is controlled. `action_order` lists
`(group, device_id, phase)` slots, each containing P followed by Q in pu;
`phase=None` means an aggregate command. Box bounds cover individual power
limits; the physical environment still applies SoC and coupled P/Q limits.

`reset(seed=...)` seeds both the physical environment and action sampling.
The last step returns a numeric terminal state with executed voltage, SoC and
measured powers. Exogenous data is held at the last interval; this state is
not another decision interval. The underlying named environment retains its
existing terminal `None` observation. Observation bounds are unbounded because
out-of-limit electrical observations must remain representable.
