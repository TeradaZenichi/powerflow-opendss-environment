# EVSE and V2G model

This document specifies the quasi-static EVSE model implemented by the
environment. It covers the public sign convention, independent P/Q control,
converter capability, internal losses, session energy and SoC integration.

## Sign convention

EVSE and BESS use the same net-active-power convention at the AC terminal:

- `p_net > 0`: charging, power absorbed from the grid;
- `p_net < 0`: V2G discharging, power injected into the grid;
- `q_injection > 0`: reactive power injected into the grid;
- `q_injection < 0`: reactive power absorbed from the grid.

Actions and observations use pu on the configured total network power base.
Executed physical results use kW and kvar before public normalization.

## Device configuration

An EVSE may define:

| Field | Meaning | Default |
|---|---|---|
| `p_max_kw` | Maximum charging power | required |
| `p_discharge_max_kw` | Maximum V2G active-power magnitude | `p_max_kw` |
| `s_max_kva` | Converter apparent-power rating | `p_max_kw / power_factor` |
| `control_mode` | `asap` or `controlled` | required |
| `v2g` | Allow negative active power | `false` |
| `soc_min_frac` | Minimum SoC available to V2G and losses | `0` |
| `power_factor` | ASAP charging power-factor magnitude | `1` |
| `standby_loss_kw` | Constant connected converter loss, P0 | `0` |
| `loss_linear_kw` | Linear loss coefficient, a | `0` |
| `loss_quadratic_kw` | Quadratic loss coefficient, b | `0` |

V2G is allowed only for controlled EVSEs. `s_max_kva` must cover both maximum
charge and discharge power. Loss coefficients are nonnegative and must be
calibrated for the charger technology being represented.

## ASAP operation

An ASAP EVSE is not part of the action. While a session is connected and still
requires energy, it requests the lower of device and session rated power. Its
reactive injection is derived from its fixed lagging power factor:

```text
q_injection = -p_net * tan(acos(power_factor))
```

ASAP charging is unidirectional. The inferred `s_max_kva` ensures that the
configured active power and power factor lie on or inside the converter
capability circle.

## Controlled P/Q operation

A controlled EVSE receives one aggregate action:

```python
EVSEAction(p_pu=-0.010, q_pu=0.015)
```

P and Q are independent. The active request is limited first by charge or
discharge power, session rated power, V2G permission and available battery
energy. Q is then clipped to the remaining converter capability:

```text
p^2 + q^2 <= s_max_kva^2
q_limit = sqrt(s_max_kva^2 - p^2)
```

This is an active-power-priority policy. Q may be nonzero when P is zero, so a
connected controlled charger can provide reactive support without scheduled
active exchange. The Gymnasium adapter exposes rectangular P/Q bounds; the
physical model applies the circular constraint during execution.

## Session energy and SoC

For a session with capacity `C`, efficiency `eta`, grid-side charged energy
`E_charge`, grid-side discharged energy `E_discharge`, and converter loss
`E_loss`, the implemented incremental relations are:

```text
delta_soc_charge    =  eta * E_charge / C
delta_soc_discharge = -E_discharge / (eta * C)
delta_soc_loss      = -E_loss / C
```

The remaining grid energy required to recover the target SoC evolves as:

```text
remaining -= E_charge
remaining += E_discharge / eta^2
remaining += E_loss / eta
```

Charging stops at target SoC. V2G and converter losses cannot take SoC below
`soc_min_frac`. Energy still required at departure is reported as unmet energy.
The session `eta` represents the base directional energy efficiency; the loss
curve represents additional operating-point-dependent converter loss.

## Converter losses

For an executed operating point, define:

```text
S = sqrt(p_net^2 + q_injection^2)
loading = min(S / s_max_kva, 1)
P_loss = P0 + a * loading + b * loading^2
```

`P0`, `a`, and `b` correspond to `standby_loss_kw`, `loss_linear_kw`, and
`loss_quadratic_kw`. The constant term applies during the connected portion of
an interval, including idle P/Q operation. Linear and quadratic terms
approximate switching/current-dependent and conduction losses. Q therefore
does not directly change SoC, but it increases S, converter current and loss.

OpenDSS separately calculates feeder copper and network losses caused by the
resulting P/Q flow. `P_loss` represents internal EVSE conversion loss and must
not be interpreted as feeder loss.

## Events inside one environment interval

Arrival and departure are processed at their exact timestamps. Energy, loss
and reactive support are integrated only over the connected overlap. OpenDSS
receives energy-equivalent interval averages:

```text
p_average = (E_charge - E_discharge) / dt
q_average = integral(q_injection dt) / dt
```

Several non-overlapping sessions may be processed inside one environment step.
Controlled carry-in sessions are rejected because their pre-episode P/Q
commands are unknown. ASAP carry-in is reconstructed deterministically.

## Observations and results

Observation schema version 4 exposes, per EVSE:

- connection, session, SoC, target and deadline information;
- signed `previous_p_pu` and `previous_q_pu`;
- `power_bounds.p_net_pu` and `power_bounds.q_injection_pu`;
- V2G enablement and configured power factor.

Executed actions report signed P/Q, charged/discharged energy, converter loss,
connected fraction and session events. Summary exports include EV charge,
discharge, net energy, internal conversion loss and unmet energy.

## Scope and limitations

- The model is quasi-static over each environment interval; it does not model
  switching waveforms, harmonics, DC-link dynamics or electromagnetic
  transients.
- Efficiency and loss coefficients are scalar approximations. Temperature,
  voltage-dependent efficiency, aging and charger-specific efficiency maps are
  not yet represented.
- The loss coefficients in example cases are synthetic assumptions, not values
  identified from GridCharge measurements.
- P/Q capability uses a circular apparent-power limit and active-power priority;
  other converter control priorities may require a different projection.

## References

1. M. C. Kisacikoglu, M. Kesler, and L. M. Tolbert, "Single-Phase On-Board
   Bidirectional PEV Charger for V2G Reactive Power Operation," *IEEE
   Transactions on Smart Grid*, 6(2), 767-775, 2015.
   [doi:10.1109/TSG.2014.2360685](https://doi.org/10.1109/TSG.2014.2360685)
   - Experimental independent P/Q command tracking and reactive support.
2. K. M. Tan, S. Padmanaban, J. Y. Yong, and V. K. Ramachandaramurthy,
   "A multi-control vehicle-to-grid charger with bi-directional active and
   reactive power capabilities for power grid support," *Energy*, 171,
   1150-1163, 2019.
   [doi:10.1016/j.energy.2019.01.053](https://doi.org/10.1016/j.energy.2019.01.053)
   - P/Q, power-factor and voltage-control modes, including the efficiency
     versus voltage-support tradeoff.
3. Y. Iwafune and T. Kawai, "Data analysis and estimation of the conversion
   efficiency of bidirectional EV chargers using home energy management systems
   data," *Smart Energy*, 15, 100145, 2024.
   [doi:10.1016/j.segy.2024.100145](https://doi.org/10.1016/j.segy.2024.100145)
   - Measured direction-dependent, partial-load and standby behavior of a
     commercial bidirectional charger.
4. M. Restrepo, J. Morris, M. Kazerani, and C. A. Canizares, "Modeling and
   Testing of a Bidirectional Smart Charger for Distribution System EV
   Integration," *IEEE Transactions on Smart Grid*, 9(1), 152-162, 2018.
   [doi:10.1109/TSG.2016.2547178](https://doi.org/10.1109/TSG.2016.2547178)
   - Distribution-system model and experimental testing of a bidirectional
     smart charger.
