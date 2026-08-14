"""
Loads data from a given path and returns a dictionary with information
for the simulation.
"""

import json
import copy
from pathlib import Path
import pandas as pd
import re

from .elements import BESS, PV, Load, Grid, Results


def load_data(path):
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

    bess_list = [
        BESS(**bess_data)
        for bess_data in devices.get("bess", [])
    ]

    pv_list = []

    for pv_data in devices.get("pv", []):
        profile_file, profile_col = pv_data["profile"].split(":")

        pv_data["profile"] = pd.read_csv(
            path / profile_file
        )[profile_col].to_numpy()

        pv_list.append(PV(**pv_data))

    # demand.csv
    demand = pd.read_csv(path / "demand.csv")

    load_list = []

    for col in demand.columns:

        if col.startswith("Pbus_"):

            bus = col[1:]
            q_col = f"Q{bus}"

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

    timestamps = pd.to_datetime(
        demand["timestamp"]
    )

    results = Results()

    return {
        "dt": dt,
        "timestamps": timestamps,
        "steps": steps,
        "phases": phases,
        "base_kv": base_kv,
        "grid": grid,
        "bess_list": bess_list,
        "pv_list": pv_list,
        "load_list": load_list,
        "topology": topology,
        "results": results,
    }


def create_episode(data, start, end):
    episode = copy.deepcopy(data)

    episode["steps"] = end - start

    episode["timestamps"] = (
        data["timestamps"]
        .iloc[start:end]
        .reset_index(drop=True)
    )

    # Grid
    episode["grid"].prices = data["grid"].prices[start:end]
    episode["grid"].array_kw = []
    episode["grid"].array_kvar = []

    # Loads
    for load in episode["load_list"]:

        load.array_kw = load.array_kw[start:end]
        load.array_kvar = load.array_kvar[start:end]

    # PV
    for pv in episode["pv_list"]:

        pv.profile = pv.profile[start:end]

        pv.array_kw = []
        pv.array_kvar = []
        pv.array_p_net_kw = []
        pv.array_grid_consumption_kw = []
        pv.array_inverter_loss_kw = []

    # BESS
    for bess in episode["bess_list"]:

        bess.soc = bess.soc_init_frac

        bess.array_soc = []
        bess.array_kw = []
        bess.array_kvar = []
        bess.array_inverter_loss_kw = []

    # Results
    episode["results"] = Results()

    return episode


def _get_dt_hours(df):
    timestamps = pd.to_datetime(df["timestamp"])

    return (
        timestamps.iloc[1] - timestamps.iloc[0]
    ).total_seconds() / 3600