import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from .case_source import resolve_case_source, validate_case_config
from .elements import BESS, BESSConfig, Grid, Load, PV, PVConfig, Results


_PHASE_DEMAND = re.compile(r"^P(.+)_([abcABC123])$")
_PHASE_NODE = {"a": 1, "b": 2, "c": 3, "1": 1, "2": 2, "3": 3}
_PHASE_NAME = {"1": "a", "2": "b", "3": "c"}


def _normalize_phase(value):
    text = str(value).lower()
    return _PHASE_NAME.get(text, text)


def load_data(path):
    """
    Loads the complete dataset from the specified path.
    """
    path, config_path = resolve_case_source(path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_case_config(cfg)
    files = cfg.get("files", {})
    demand_path = path / files.get("demand", "demand.csv")
    price_path = path / files.get("prices", "price.csv")
    devices_path = path / files.get("devices", "devices.json")

    # prices.csv
    price_frame = pd.read_csv(price_path, parse_dates=["timestamp"])
    price_frame = price_frame.sort_values("timestamp").reset_index(drop=True)
    prices = price_frame["price_per_kwh"].to_numpy()

    base_kv = cfg["base"]["v_base_kv"]

    # circuit and line data
    topology = path / cfg["network"]["master"]

    with open(topology, "r", encoding="utf-8") as f: 
        # searches for "phases = x" in topology file and returns x
        phases = int(re.search(r"phases\s*=\s*(\d+)",f.read(),re.IGNORECASE).group(1))

    # devices.json
    if devices_path.exists():
        with open(devices_path, "r", encoding="utf-8") as f:
            devices = json.load(f)
    else:
        devices = {}
    _validate_device_modes(devices)

    # PV profiles
    pv_profiles = {}
    pv_phase_profiles = {}
    profile_timestamps = []

    for pv_data in devices.get("pv", []):
        profile_file, profile_col = pv_data["profile"].split(":")

        profile_frame = pd.read_csv(
            path / profile_file, parse_dates=["timestamp"]
        ).sort_values("timestamp").reset_index(drop=True)
        pv_profiles[profile_col] = profile_frame[profile_col].to_numpy()
        profile_timestamps.append(
            (f"PV profile {profile_file}:{profile_col}", profile_frame["timestamp"])
        )
        phase_profiles = {}
        for phase, reference in pv_data.get("phase_profiles", {}).items():
            phase_file, phase_col = reference.split(":")
            phase_frame = pd.read_csv(
                path / phase_file, parse_dates=["timestamp"]
            ).sort_values("timestamp").reset_index(drop=True)
            phase_profiles[_normalize_phase(phase)] = phase_frame[phase_col].to_numpy()
            profile_timestamps.append(
                (f"PV phase profile {phase_file}:{phase_col}", phase_frame["timestamp"])
            )
        if phase_profiles:
            phase_sum = sum(phase_profiles.values())
            if not np.allclose(
                phase_sum, pv_profiles[profile_col], rtol=0.0, atol=1e-9
            ):
                raise ValueError(
                    f"PV {pv_data['id']!r} phase profiles must sum to its total profile"
                )
        pv_phase_profiles[str(pv_data["id"])] = phase_profiles

    # demand.csv
    demand = pd.read_csv(demand_path, parse_dates=["timestamp"])
    demand = demand.sort_values("timestamp").reset_index(drop=True)

    timestamps = demand["timestamp"]
    _require_timestamps(price_frame["timestamp"], timestamps, "price.csv")
    for label, profile_index in profile_timestamps:
        _require_timestamps(profile_index, timestamps, label)

    dt = get_dt_hours(demand)

    steps = len(demand)

    return {
        "dt": dt,
        "case_path": path,
        "config_path": config_path,
        "name": cfg.get("name", path.name),
        "formulation": cfg.get("formulation"),
        "schema_version": cfg.get("schema_version", 1),
        "timestamps": timestamps,
        "steps": steps,
        "phases": phases,
        "base_kv": base_kv,
        "topology": topology,
        "devices": devices,
        "pv_profiles": pv_profiles,
        "pv_phase_profiles": pv_phase_profiles,
        "demand": demand,
        "prices": prices,
        "feed_in_tariff_ratio": float(
            cfg.get("grid", {}).get("feed_in_tariff_ratio", 1.0)
        ),
    }

def episode_data(data, episode_idx, episode_steps):
    """ 
    Returns the data corresponding to one episode.
    """

    episode_start = episode_idx * episode_steps
    episode_end = episode_idx * episode_steps + episode_steps

    if episode_end > data["steps"]:
        raise ValueError(
            f"Episode {episode_idx+1} with {episode_steps} steps exceeds total steps {data['steps']}"
        )
    
    # bess
    bess_list = [BESS(BESSConfig(**values)) for values in data["devices"].get("bess", [])]

    # grid
    grid = Grid(data["prices"][episode_start:episode_end])

    # pv
    pv_list = []
    for pv_data in data["devices"].get("pv", []):
        profile_id = pv_data["profile"].split(":",1)[1]
        profile = data["pv_profiles"][profile_id]
        episode_profile = profile[episode_start:episode_end]

        phase_profiles = {
            phase: values[episode_start:episode_end]
            for phase, values in data["pv_phase_profiles"].get(str(pv_data["id"]), {}).items()
        }
        config = {**pv_data, "profile": episode_profile, "phase_profiles": phase_profiles}
        pv = PV(PVConfig(**config))
        pv_list.append(pv)

    # load
    demand = data["demand"][episode_start:episode_end]
    load_list = []

    for col in data["demand"].columns:
        if not col.startswith("P"):
            continue
        phase_match = _PHASE_DEMAND.fullmatch(col)
        if phase_match:
            bus, phase = phase_match.groups()
            phase_node = _PHASE_NODE[phase.lower()]
            q_col = f"Q{bus}_{phase}"
        else:
            bus = col[1:]
            phase, phase_node = None, None
            q_col = f"Q{bus}"
        if q_col not in data["demand"]:
            raise ValueError(f"demand.csv is missing reactive column {q_col!r}")
        load_list.append(
            Load(
                id=f"Load_{bus}" + (f"_{phase.lower()}" if phase else ""),
                bus=bus,
                array_kw=data["demand"][col].to_numpy()[episode_start:episode_end],
                array_kvar=data["demand"][q_col].to_numpy()[episode_start:episode_end],
                phase_node=phase_node,
            )
        )

    return {
        "dt": data["dt"],
        "timestamps": data["timestamps"].iloc[episode_start:episode_end].reset_index(drop=True),
        "steps": episode_steps,
        "phases": data["phases"],
        "base_kv": data["base_kv"],
        "topology": data["topology"],
        "grid": grid,
        "bess_list": bess_list,
        "pv_list": pv_list,
        "load_list": load_list,
        "results": Results()
    }

def get_dt_hours(df):

    timestamps = pd.to_datetime(df["timestamp"])
    if len(timestamps) < 2:
        return 1.0
    differences = timestamps.diff().dropna().dt.total_seconds() / 3600.0
    if (differences <= 0.0).any() or not np.allclose(
        differences, differences.iloc[0], rtol=0.0, atol=1e-12
    ):
        raise ValueError("demand.csv timestamps must be increasing and equally spaced")
    return float(differences.iloc[0])


def _require_timestamps(actual, expected, label):
    if not pd.DatetimeIndex(actual).equals(pd.DatetimeIndex(expected)):
        raise ValueError(f"{label} timestamps must exactly match demand.csv")


def _validate_device_modes(devices):
    for kind in ("bess", "pv"):
        for device in devices.get(kind, []):
            connection = str(device.get("connection", "wye")).lower()
            if connection not in {"wye", "delta"}:
                raise ValueError("connection must be 'wye' or 'delta'")
            if connection == "delta" and {
                _normalize_phase(phase) for phase in device.get("phases", ())
            } != {"a", "b", "c"}:
                raise ValueError("delta devices must define phases a, b and c")
            if str(device.get("dispatch_mode", "aggregate")).lower() not in {
                "aggregate", "per_phase"
            }:
                raise ValueError("dispatch_mode must be 'aggregate' or 'per_phase'")
            if (
                kind == "pv"
                and str(device.get("dispatch_mode", "aggregate")).lower()
                == "per_phase"
            ):
                phases = {
                    _normalize_phase(phase)
                    for phase in device.get("phases", ("a", "b", "c"))
                }
                profiles = {
                    _normalize_phase(phase)
                    for phase in device.get("phase_profiles", {})
                }
                if profiles != phases:
                    raise ValueError(
                        "per_phase PV phase_profiles must define exactly its phases"
                    )
