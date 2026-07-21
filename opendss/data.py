"""
Loads data from a given path and returns a dictionary with information
for the simulation.
"""

import json
from pathlib import Path
import pandas as pd
import re

from .elements import BESS, PV, Load, Grid

def load_data(path):
    path = Path(path).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"File not found: {path}")
    
    # prices.csv
    prices = pd.read_csv(path / "price.csv")["price_per_kwh"].to_numpy()
    grid = Grid(prices)

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
        steps = len(pv_data["profile"])
        
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
    return {"steps": steps, "phases": phases, "base_kv": base_kv, "grid": grid, "bess_list": bess_list, "pv_list": pv_list, "load_list": load_list, "topology": topology}