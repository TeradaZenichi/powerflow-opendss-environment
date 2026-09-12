import json
import re
from pathlib import Path
import pandas as pd
import copy
from .elements import BESS, PV, Load, Grid, Results


_PHASE_DEMAND = re.compile(r"^P(.+)_([abcABC123])$")
_PHASE_NODE = {"a": 1, "b": 2, "c": 3, "1": 1, "2": 2, "3": 3}


def load_data(path):
    """
    Loads the complete dataset from the specified path.
    """
    path = Path(path).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"File not found: {path}")

    # prices.csv
    prices = pd.read_csv(path / "price.csv")["price_per_kwh"].to_numpy()

    # config.json
    with open(path / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_kv = cfg["base"]["v_base_kv"]

    # circuit and line data
    topology = path / cfg["network"]["master"]

    with open(topology, "r", encoding="utf-8") as f: 
        # searches for "phases = x" in topology file and returns x
        phases = int(re.search(r"phases\s*=\s*(\d+)",f.read(),re.IGNORECASE).group(1))

    # devices.json
    with open(path / "devices.json", "r", encoding="utf-8") as f:
        devices = json.load(f)

    # PV profiles
    pv_profiles = {}

    for pv_data in devices.get("pv", []):
        profile_file, profile_col = pv_data["profile"].split(":")

        pv_profiles[profile_col] = pd.read_csv(
            path / profile_file
        )[profile_col].to_numpy()

    # demand.csv
    demand = pd.read_csv(path / "demand.csv")

    timestamps = pd.to_datetime(demand["timestamp"])

    dt = get_dt_hours(demand)

    steps = len(demand)

    return {
        "dt": dt,
        "timestamps": timestamps,
        "steps": steps,
        "phases": phases,
        "base_kv": base_kv,
        "topology": topology,
        "devices": devices,
        "pv_profiles": pv_profiles,
        "demand": demand,
        "prices": prices,
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
    bess_list = [BESS(**bess_data) for bess_data in data["devices"].get("bess", [])]

    # grid
    grid = Grid(data["prices"][episode_start:episode_end])

    # pv
    pv_list = []
    for pv_data in data["devices"].get("pv", []):
        profile_id = pv_data["profile"].split(":",1)[1]
        profile = data["pv_profiles"][profile_id]
        episode_profile = profile[episode_start:episode_end]

        pv = PV(**pv_data)
        pv.profile = episode_profile
        pv_list.append(pv)

    # load
    demand = data["demand"][episode_start:episode_end]
    load_list = []

    for col in data["demand"].columns:
        if col.startswith("Pbus_"):
            phase_match = _PHASE_DEMAND.fullmatch(col)
            if phase_match:
                bus, phase = phase_match.groups()
                phase_node = _PHASE_NODE[phase.lower()]
                q_col = f"Q{bus}_{phase}"
            else:
                bus = col[1:]
                phase, phase_node = None, None
                q_col = f"Q{bus}"
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

    return (
        timestamps.iloc[1] - timestamps.iloc[0]
    ).total_seconds() / 3600
