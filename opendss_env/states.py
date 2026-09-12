import numpy as np


_PHASE_NAME = {1: "a", 2: "b", 3: "c"}


def _phase_names(device, default_phase_count):
    phases = device.phases or tuple(range(1, default_phase_count + 1))
    return tuple(
        _PHASE_NAME.get(int(phase), str(phase).lower())
        if isinstance(phase, int) or str(phase).isdigit()
        else str(phase).lower()
        for phase in phases
    )


def _phase_values(values, phases):
    return {phase: float(values.get(phase, 0.0)) for phase in phases}

def get_hour(env):
    return env.timestamps[env.idx].hour

def get_price(env):
    return env.grid.prices[env.idx]

def get_bess_soc(env):
    return sum(bess.soc for bess in env.bess_list)

def get_previous_pv_kw(env):
    return sum(pv.array_kw[env.idx - 1] for pv in env.pv_list) if env.idx > 0 else 0.0

def get_previous_load_kw(env):
    return sum(load.array_kw[env.idx - 1] for load in env.load_list) if env.idx > 0 else 0.0

def get_current_load_kw(env):
    return sum(load.array_kw[env.idx] for load in env.load_list)

def build_state(env, state_functions):
    return np.array([function(env) for function in state_functions], dtype=np.float32)


def build_named_observation(env):
    """Return the causal pre-action observation using named physical fields."""

    if env.idx >= env.steps:
        raise RuntimeError("No observation is available after the episode terminates")

    bus_names = set(env.current_bus_voltages_pu)
    bus_names.update(load.bus for load in env.load_list)
    buses = {}
    for bus in sorted(bus_names):
        voltages = env.current_bus_voltages_pu.get(bus, {})
        angles = env.current_bus_angles_deg.get(bus, {})
        phases = tuple(voltages) or tuple(
            _PHASE_NAME[index] for index in range(1, env.data["phases"] + 1)
        )
        buses[bus] = {
            "v_before_pu": _phase_values(voltages, phases),
            "angle_before_deg": _phase_values(angles, phases),
            "p_load_kw": {phase: 0.0 for phase in phases},
            "q_load_kvar": {phase: 0.0 for phase in phases},
        }

    for load in env.load_list:
        phases = tuple(buses[load.bus]["p_load_kw"])
        if load.phase_node is None:
            selected = phases
        else:
            selected = (_PHASE_NAME[load.phase_node],)
        p_share = float(load.array_kw[env.idx]) / len(selected)
        q_share = float(load.array_kvar[env.idx]) / len(selected)
        for phase in selected:
            buses[load.bus]["p_load_kw"][phase] += p_share
            buses[load.bus]["q_load_kvar"][phase] += q_share

    bess = {}
    for device in env.bess_list:
        phases = _phase_names(device, env.data["phases"])
        measured = env.current_device_measurements["bess"].get(device.id, {})
        bess[device.id] = {
            "soc_before_frac": float(device.soc),
            "previous_p_kw": _phase_values(measured.get("p_net_kw", {}), phases),
            "previous_q_kvar": _phase_values(
                measured.get("q_injection_kvar", {}), phases
            ),
        }

    pv = {}
    for device in env.pv_list:
        phases = _phase_names(device, env.data["phases"])
        if device.phase_profiles:
            available = {
                phase: float(device.phase_profiles[phase][env.idx])
                for phase in phases
            }
        else:
            share = float(device.profile[env.idx]) / len(phases)
            available = {phase: share for phase in phases}
        measured = env.current_device_measurements["pv"].get(device.id, {})
        pv[device.id] = {
            "available_kw": available,
            "previous_p_kw": _phase_values(
                measured.get("generation_kw", {}), phases
            ),
            "previous_q_kvar": _phase_values(
                measured.get("q_injection_kvar", {}), phases
            ),
        }

    price = float(env.grid.prices[env.idx])
    return {
        "timestamp": env.timestamps.iloc[env.idx].isoformat(),
        "dt_h": float(env.dt),
        "phase_order": list(_PHASE_NAME.values())[:env.data["phases"]],
        "buses": buses,
        "grid": {
            "buy_price_per_kwh": price,
            "sell_price_per_kwh": price * env.data["feed_in_tariff_ratio"],
        },
        "bess": bess,
        "pv": pv,
    }
