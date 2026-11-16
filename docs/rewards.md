# Global and controller rewards

`Reward` evaluates one executed interval after OpenDSS and before advancing the
time index. The environment returns a scalar as before, plus `info["rewards"]`:

```python
{
    "global": ...,                 # Selected scalar, or the legacy callable result.
    "cost": ...,                   # Negative actual grid energy cost.
    "teacher": ...,                # Negative selected teacher objective.
    "teacher_terms": {
        "energy_cost": ..., "tie_break_cost": ...,
        "profile": {"formulation": ..., "feed_in_ratio": ..., "tie_break_penalty": ...},
    },
    "controllers": {"battery": ..., "solar": ...},
    "terms": {"battery": {"voltage": {
        "raw": ..., "scale": ..., "normalized": ..., "weight": ..., "contribution": ...,
    }}},
}
```

Reward values retain their physical/economic units; the observation's `units="pu"`
does not turn currency, kWh, VUF or reward terms into per-unit power.

## Configuration and extraction

```python
from opendss_environment import EnvironmentConfig, MicrogridEnv, Reward

CONTROLLERS = {
    "battery": {
        "devices": {"bess": ["b1"]},
        "buses": ["bus_004"],
        "weights": {"voltage": 1.0, "energy_cost": 1.0, "throughput": 0.1},
        "scales": {"voltage": 0.01, "energy_cost": 1.0, "throughput": 10.0},
        "parameters": {"voltage": {"v_min": 0.95, "v_max": 1.05}},
    },
    "solar": {
        "devices": {"pv": ["pv1"]},
        "buses": ["bus_005"],
        "weights": {"voltage": 1.0, "curtailment": 0.1},
        "scales": {"voltage": 0.01, "curtailment": 10.0},
    },
}
REWARD = Reward(
    controllers=CONTROLLERS,
    global_objective="teacher",  # Or "cost", the default.
    teacher_profile={"formulation": "single_phase_active"},
)
env = MicrogridEnv(EnvironmentConfig("examples/case5", episode_steps=2, reward_function=REWARD))
observation, _ = env.reset()
observation, global_reward, terminated, truncated, info = env.step(None)

cost = REWARD.get_cost(info["rewards"])
teacher = REWARD.get_teacher(info["rewards"])
battery = REWARD.get_local(info["rewards"], "battery")
terms = REWARD.get_terms(info["rewards"], "battery")
env.close()
```

`get_global()`, `get_cost()`, `get_teacher()`, `get_local()` and `get_terms()`
only extract an existing result. `get_terms()` returns a private copy. The
environment calls `evaluate(env, objectives)` once; users do not call it again
after `step()`, when the environment already points to the next interval.

`devices`, `buses` and `branches` are explicit scopes, not control authority.
Several controllers can monitor the same electrical quantities. Branch IDs
use the measured OpenDSS names, such as `Line.feeder`. Coordinator rewards
are evaluated over their own scope, not summed automatically from children.
No controller hierarchy or messaging protocol is required by this evaluator.

For local power terms, a nonempty device selection defines the net device-group
boundary. Otherwise, selected buses define the boundary, using all their
connected terminals. Both P and Q references use **positive net import**.
Device actions still use **positive Q injection**, the opposite Q sign.

## Objectives from a superior controller

Set objectives after `reset()` and before the corresponding `step()`:

```python
env.reward_objectives = {
    "battery": {
        "weights": {"tracking": 2.0},
        "parameters": {"tracking": {"p_reference_kw": -10.0, "q_reference_kvar": -5.0}},
    },
}
```

These overrides merge with configured weights and per-term parameters. Scopes
and normalization scales remain fixed. Objectives are copied before actions
execute, so later changes affect later decisions. `reset()` clears overrides.
The caller must provide the same objectives to its policy; reward configuration
does not automatically add them to observations.

For a controller, the result is `sum(weight * raw / scale)`. Scales must be
finite and positive; weights must be finite and nonnegative. Terms with zero
weight are disabled and recorded with raw/contribution zero. Configuration
errors reject the reward evaluation; failed steps do not commit device state.

## Global and teacher formulas

Let `[x]+ = max(x, 0)`, `dt` be hours, powers be kW/kvar and energies be kWh.
The actual grid cost is

`C = dt * (buy_price * P_import - sell_price * P_export)`, with `r_cost = -C`.

Teacher energy cost uses `sell_price = feed_in_ratio * buy_price`, even when
the environment has an independent export tariff. The profile defaults to the
case's `feed_in_tariff_ratio`. Record both rewards to expose tariff differences.

`r_teacher = -(C_teacher + D)`:

| Formulation | Tie-break cost D | Default coefficient |
|---|---|---|
| `single_phase_socp` | 0 | 0 |
| `single_phase_active` | `epsilon * dt * (P_import + P_export + sum(BESS_P_charge + BESS_P_discharge))` | 1e-6 |
| `three_phase_ivr` | `epsilon * dt * sum(BESS_P_charge + BESS_P_discharge)` | 1e-8 |

`teacher_profile` accepts `formulation`, `feed_in_ratio` and `tie_break_penalty`.
The SOCP profile rejects a nonzero tie-break. These expressions follow the
inspected `opf-teacher` formulations, not an online dependency on its solver.
EV throughput is not added to the teacher tie-break.

Matching an objective does not match network models or feasibility. The teacher
active model excludes network losses, and terminal SoC/network constraints are
not automatically enforced by a reward. Compare undiscounted returns over the
same horizon and initial/terminal conditions. The evaluator does not change
physical constraints or add a terminal penalty to the teacher score.

## Local terms

Each term maps to the public method below. Method keyword arguments belong in
`parameters[term]`; powers there are physical kW/kvar. Electrical terms sum
selected buses/phases unless stated otherwise.

| Term / method | Raw reward and parameters |
|---|---|
| `voltage` / `minimize_voltage_violation` | `-sum([v_min - V]+ + [V - v_max]+)`; defaults 0.95/1.05 |
| `voltage_deviation` / `minimize_voltage_deviation` | `-sum(abs(V - reference))`; default reference 1 |
| `unbalance` / `minimize_voltage_unbalance` | `-sum([100*abs(V_negative)/abs(V_positive) - maximum_pct]+)`; default threshold 0; requires a/b/c |
| `branch_overload` / `minimize_branch_overload` | `-sum([I/I_normal - 1]+)`; worst terminal per branch/phase; unrated selected terminals raise an error |
| `power_violation` / `minimize_power_violation` | `-[P-import_limit_kw]+ - [-P-export_limit_kw]+`; both limits required |
| `tracking` / `track_power` | `-hypot(P-p_reference_kw, Q-q_reference_kvar)`; references default to 0 |
| `energy_cost` / `minimize_energy_cost` | `-dt*(buy_price*[P]+ - sell_price*[-P]+)` for the net selected device group; prices default to current grid tariffs |
| `curtailment` / `minimize_curtailment` | `-dt*sum([PV_available - PV_generation - PV_inverter_loss]+)` |
| `self_consumption` / `maximize_self_consumption` | `-solar_export_kwh + r_curtailment`; requires explicit solar attribution |
| `losses` / `minimize_losses` | Negative conversion and charge/discharge efficiency losses of selected devices, in kWh |
| `throughput` / `minimize_throughput` | Negative cell-side charge + discharge energy, including converter losses drawn from cells |
| `soc_reserve` / `minimize_soc_violation` | `-sum([minimum-SoC]+ + [SoC-maximum]+)`; default range 0..1 |
| `soc_target` / `track_soc_target` | `-sum(abs(SoC-target))` only at `at_step`, default episode end; target default 0.5 |
| `ev_unmet` / `minimize_ev_unmet_energy` | Negative AC-equivalent remaining kWh of sessions departing in this interval |
| `ev_infeasibility` / `minimize_ev_infeasibility` | Negative cell-side deficit above estimated recoverable energy before departure |
| `action_variation` / `minimize_action_variation` | `-sum_devices,phases(hypot(delta_P,delta_Q)/S_device)`; first interval compares with zero |
| `service` / `service_incentive` | Capped requested-direction P/Q remuneration during validity, described below |
| `import` / `minimize_import` | `-[P]+*dt` |
| `reactive` / `minimize_reactive_power` | `-abs(Q)` |

Power, voltage, VUF and reserve penalties are instantaneous values per decision,
preserving the existing methods. Energy terms integrate exact interval duration.
Changing time resolution therefore requires reviewing weights/scales.

## EV/V2G, incentives and accounting

Use an explicit EVSE group, for example `devices={"evse": ["c1", "a1"]}` in the
EVSE fixture. Session service is assessed individually before aggregation.
Departure deficit is counted once, not as an accumulated historical penalty.
`soc_reserve` and `soc_target` assess BESS and EVs still connected at interval
end; use `ev_unmet` for vehicles that depart during the interval.

`ev_infeasibility` estimates, at interval end:

`E_needed = [target_soc - measured_soc]+ * capacity_kwh`

`E_recoverable = [eta*P_available - converter_loss(P_available, Q=0)]+ * hours_until_departure`.

It sums `-[E_needed-E_recoverable]+` over connected vehicles. The optional
`available_charge_kw` can be a constant or a device-ID mapping and is capped
by charger, apparent-power and vehicle limits. Without it, nominal availability
is optimistic. It is not a forecast of future network feasibility. Sessions
continuing beyond episode end are not treated as departures.

V2G remuneration uses executed net AC energy in `energy_cost`. Throughput and
losses use cell-side measurements with each session's efficiency; opposite
phase powers cannot cancel throughput. BESS/PV losses and EV converter losses
remain part of the physical energy balance. Penalizing them separately gives
efficiency additional weight; it is not another energy bill.

`service` takes `p_reference_kw`, `q_reference_kvar`, nonnegative `p_rate` and
`q_rate`, plus optional `valid_until`. For each channel:

`service_power = min([measured_power * sign(reference)]+, abs(reference))`.

Payment is rate times service power times the interval's overlap with validity;
zero references, opposite-direction service and expired requests earn nothing.
The Q sign is net import, so a negative Q reference requests injection.
Energy settlement and service payment are distinct contracts: do not pay for
the same service twice. Local settlements are not automatically added to
global cost, since they may be internal transfers.

Self-consumption deliberately has no net-export fallback. Supply measured or
accounted `solar_export_kwh` for the selected boundary; grid export can come
from PV, BESS or EVs. This module does not implement solar provenance tracking.
Combine this objective with losses to avoid rewarding dissipation. Combining
`self_consumption` and `curtailment` gives curtailment additional weight.
