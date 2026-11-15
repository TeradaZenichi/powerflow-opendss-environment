from copy import deepcopy
import numpy as np

from .evse import utc_timestamp
from .network import PHASE_NAME as _PHASE_NAME, device_phases as _phase_names


OBSERVATION_SCHEMA_VERSION = 2
LOCAL_OBSERVATION_SCHEMA_VERSION = 1
CONTROLLER_OBSERVATION_SCHEMA_VERSION = 1
LOCAL_OBSERVATION_SECTIONS = ("device", "bus", "branches", "grid")
LOCAL_DEFAULT_SECTIONS = ("device", "bus")
DEVICE_KINDS = ("bess", "pv", "evse")
OBSERVATION_METADATA = (
    "units", "base", "network", "phase_power_base", "timestamp", "dt_h", "phase_order",
)


def _phase_values(values, phases):
    return {phase: float(values.get(phase, 0.0)) for phase in phases}


def _device_groups(env):
    return {kind: getattr(env, f"{kind}_list") for kind in DEVICE_KINDS}


def _validated_observation(env, observation):
    source = env.observe() if observation is None else observation
    if source.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("Global observation schema version does not match the environment")
    env.normalizer.check_base(source.get("base", {}))
    return source


def _view_metadata(source, schema_name, schema_version):
    metadata = {
        schema_name: schema_version,
        "observation_schema_version": source["observation_schema_version"],
    }
    metadata.update({name: deepcopy(source[name]) for name in OBSERVATION_METADATA})
    return metadata


def _neighbor_branches(source, bus):
    result = []
    for item in source["branches"]:
        if item["bus"].lower() != bus.lower():
            continue
        flow = deepcopy(item)
        flow["neighbor_bus"] = next((
            other["bus"] for other in source["branches"]
            if other["element"] == item["element"]
            and other["terminal"] != item["terminal"]
        ), None)
        result.append(flow)
    return result

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
    """Return the causal pre-action observation in pu on the configured bases."""

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
        measured = env.previous_device_measurements["bess"].get(device.id, {})
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
        measured = env.previous_device_measurements["pv"].get(device.id, {})
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
    observation = {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "units": "pu",
        "base": env.normalizer.as_dict()["base"],
        "network": env.network.as_dict(),
        "phase_power_base": "total_network",
        "timestamp": env.timestamps.iloc[env.idx].isoformat(),
        "dt_h": float(env.dt),
        "phase_order": list(_PHASE_NAME.values())[:env.data["phases"]],
        "buses": buses,
        "branches": env.pre_action_branch_measurements,
        "grid": {
            "buy_price_per_kwh": price,
            "sell_price_per_kwh": float(env.grid.sell_prices[env.idx]),
        },
        "bess": bess,
        "pv": pv,
    }
    if env.evse_list:
        timestamp = utc_timestamp(env.timestamps.iloc[env.idx])
        information = env.data["ev_session_information"]
        evse = {}
        for device in env.evse_list:
            phases = _phase_names(device, env.data["phases"])
            session = device.current_session(timestamp)
            measured = env.current_device_measurements["evse"].get(device.id, {})
            available_kw = 0.0 if session is None or session.remaining_kwh <= 0.0 else min(
                device.p_max_kw, session.config.rated_power_kw
            )
            values = {
                "controlled": device.controlled,
                "connected": session is not None,
                "carry_in": session is not None and session.carry_in,
                "session_id": 0 if session is None else session.config.id,
                "soc_before_frac": 0.0 if session is None else float(session.soc),
                "available_kw": dict.fromkeys(
                    phases,
                    available_kw / len(phases),
                ),
                "previous_p_kw": _phase_values(
                    measured.get("p_net_kw", {}), phases
                ),
            }
            if information in {"target", "target_deadline"}:
                values.update({
                    "target_soc_frac": 0.0 if session is None else float(session.config.target_soc),
                    "remaining_energy_kwh": 0.0 if session is None else float(session.remaining_kwh),
                })
            if information == "target_deadline":
                values["time_to_departure_h"] = 0.0 if session is None else max(
                    (session.config.departure - timestamp).total_seconds() / 3600.0, 0.0
                )
            evse[device.id] = values
        observation["evse"] = evse
    return env.normalizer.normalize(observation)


def build_local_observation(env, kind, device_id, observation=None, include=None):
    """Select one device's causal view from a named global observation."""
    devices = _device_groups(env)
    if kind not in devices:
        raise ValueError("Device kind must be bess, pv or evse")
    matches = [device for device in devices[kind] if device.id == device_id]
    if not matches:
        raise ValueError(f"Unknown {kind} device {device_id!r}")
    device = matches[0]
    sections = LOCAL_DEFAULT_SECTIONS if include is None else tuple(include)
    if not sections or set(sections) - set(LOCAL_OBSERVATION_SECTIONS):
        raise ValueError(f"Local observation sections must be {LOCAL_OBSERVATION_SECTIONS}")

    source = _validated_observation(env, observation)
    if kind not in source or device_id not in source[kind]:
        raise ValueError("Global observation does not contain the selected device")

    phases = list(_phase_names(device, env.data["phases"]))
    local = _view_metadata(
        source, "local_observation_schema_version", LOCAL_OBSERVATION_SCHEMA_VERSION,
    )
    local["controller"] = {
        "kind": kind, "id": device.id, "bus": device.bus, "phases": phases,
    }
    if "device" in sections:
        local["device"] = deepcopy(source[kind][device_id])
    if "bus" in sections:
        local["bus"] = deepcopy(source["buses"][device.bus])
    if "branches" in sections:
        local["branches"] = _neighbor_branches(source, device.bus)
    if "grid" in sections:
        local["grid"] = deepcopy(source["grid"])
    return local


def build_controller_observation(
    env, bus, observed_components, observation=None, extra_features=(),
):
    """Build one nodal view with only explicitly selected components."""
    source = _validated_observation(env, observation)
    if bus not in source["buses"]:
        raise ValueError(f"Unknown observation bus {bus!r}")
    if not isinstance(observed_components, dict) or not observed_components:
        raise ValueError("observed_components must name at least one device")
    if set(observed_components) - set(DEVICE_KINDS):
        raise ValueError("Observed component groups must be bess, pv or evse")

    components = {}
    references = {}
    device_groups = _device_groups(env)
    for kind, requested_ids in observed_components.items():
        ids = tuple(requested_ids)
        if len(ids) != len(set(ids)):
            raise ValueError(f"Observed {kind} IDs must be unique")
        available = {device.id: device for device in device_groups[kind]}
        unknown = set(ids) - set(available)
        if unknown:
            raise ValueError(f"Unknown observed {kind} devices {sorted(unknown)}")
        components[kind] = {}
        references[kind] = []
        for device_id in ids:
            device = available[device_id]
            values = deepcopy(source[kind][device_id])
            values["bus"] = device.bus
            values["phases"] = list(_phase_names(device, env.data["phases"]))
            components[kind][device_id] = values
            references[kind].append(device_id)
    if not any(references.values()):
        raise ValueError("observed_components must name at least one device")

    extras = tuple(extra_features)
    if set(extras) - {"branches", "grid"}:
        raise ValueError("Controller extra_features must be branches or grid")
    result = _view_metadata(
        source,
        "controller_observation_schema_version",
        CONTROLLER_OBSERVATION_SCHEMA_VERSION,
    )
    result.update({
        "controller": {"bus": bus, "observed_components": references},
        "bus": deepcopy(source["buses"][bus]),
        "components": components,
    })
    if "branches" in extras:
        result["branches"] = _neighbor_branches(source, bus)
    if "grid" in extras:
        result["grid"] = deepcopy(source["grid"])
    return result
