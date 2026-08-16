import json
import re
from pathlib import Path
import pandas as pd
import copy
from .elements import BESS, PV, Load, Grid, Results


def load_data(path):
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
        phases = int(
            re.search(
                r"phases\s*=\s*(\d+)",
                f.read(),
                re.IGNORECASE
            ).group(1)
        )

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

    dt = _get_dt_hours(demand)

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


def create_episode_data(data, start, end):

    bess_list = [
        BESS(**bess_data)
        for bess_data in data["devices"].get("bess", [])
    ]

    pv_list = []

    for pv_data in data["devices"].get("pv", []):

        profile_file, profile_col = pv_data["profile"].split(":")

        pv_list.append(
            PV(
                id=pv_data["id"],
                bus=pv_data["bus"],
                p_max_kw=pv_data["p_max_kw"],
                s_max_kva=pv_data["s_max_kva"],
                q_loss_rated_kw=pv_data["q_loss_rated_kw"],
                night_var=pv_data["night_var"],
                profile=data["pv_profiles"][profile_col][start:end],
                control=pv_data["control"],
                curtailable=pv_data["curtailable"],
                power_factor=pv_data["power_factor"],
            )
        )

    load_list = []

    for col in data["demand"].columns:

        if col.startswith("Pbus_"):

            bus = col[1:]
            q_col = f"Q{bus}"

            load_list.append(
                Load(
                    id=f"Load_{bus}",
                    bus=bus,
                    array_kw=data["demand"][col].to_numpy()[start:end],
                    array_kvar=data["demand"][q_col].to_numpy()[start:end],
                )
            )

    grid = Grid(
        data["prices"][start:end]
    )

    return {
        "dt": data["dt"],
        "steps": end - start,
        "timestamps": data["timestamps"].iloc[start:end].reset_index(drop=True),
        "phases": data["phases"],
        "base_kv": data["base_kv"],
        "topology": data["topology"],
        "grid": grid,
        "bess_list": bess_list,
        "pv_list": pv_list,
        "load_list": load_list,
        "results": Results(),
    }


def split_episodes(data, episode_steps):

    if episode_steps <= 0:
        raise ValueError("episode_steps must be positive")

    if data["steps"] % episode_steps != 0:
        raise ValueError(
            "The number of data steps must be divisible by episode_steps."
        )

    return [
        create_episode_data(
            data,
            start,
            start + episode_steps,
        )
        for start in range(
            0,
            data["steps"],
            episode_steps,
        )
    ]

def _get_dt_hours(df):

    timestamps = pd.to_datetime(df["timestamp"])

    return (
        timestamps.iloc[1] - timestamps.iloc[0]
    ).total_seconds() / 3600

# only for simulation.py, not used in the environment
def simulation_data(path):
    path = Path(path).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"File not found: {path}")
    
    # prices.csv
    prices = pd.read_csv(path / "price.csv")["price_per_kwh"].to_numpy()
    grid = Grid(prices)
    steps = len(prices)

    # config.json
    with open(path / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_kv = cfg["base"]["v_base_kv"]

    # circuit and line data
    topology = path / cfg["network"]["master"]
    with open(topology, "r", encoding="utf-8") as f:
        phases = int(re.search(r"phases\s*=\s*(\d+)", f.read(), re.IGNORECASE).group(1))
    
    # devices.json
    with open(path / "devices.json", "r", encoding="utf-8") as f:
        devices = json.load(f)
    bess_list = [BESS(**bess_data) for bess_data in devices.get("bess", [])]
    pv_list = []
    for pv_data in devices.get("pv", []):
        profile_file, profile_col = pv_data["profile"].split(":")
        pv_data["profile"] = pd.read_csv(path / profile_file)[profile_col].to_numpy()
        pv_list.append(PV(**pv_data))
        
    # demand.csv
    demand = pd.read_csv(path / "demand.csv")
    load_list = []
    for col in demand.columns:
        if col.startswith("Pbus_"):
            bus = col[1:]              # "Pbus_005" -> "bus_005"
            q_col = f"Q{bus}"          # "Qbus_005"
    
            load_list.append(
                Load(
                    id=f"Load_{bus}",
                    bus=bus,
                    array_kw=demand[col].to_numpy(),
                    array_kvar=demand[q_col].to_numpy()
                )
            )
    steps = len(demand)
    dt = _get_dt_hours(demand)
    timestamps = pd.to_datetime(demand["timestamp"])

    results = Results()

    return {"dt": dt, "timestamps": timestamps, "steps": steps, "phases": phases, "base_kv": base_kv, "grid": grid, "bess_list": bess_list, "pv_list": pv_list, "load_list": load_list, "topology": topology, "results": results}
