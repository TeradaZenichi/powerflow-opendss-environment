"""
Loads data from a given path and returns a dictionary with information
for the simulation.
"""

import json
from pathlib import Path
import pandas as pd

from elements import BESS, PV, Load

def load_data(path):
    path = Path(path).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"File not found: {path}")

    # config.json
    with open(path / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_kv = cfg["base"]["v_base_kv"]
    topology = path / cfg["network"]["master"]

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
    return {"steps": steps, "base_kv": base_kv, "bess_list": bess_list, "pv_list": pv_list, "demand": demand, "topology": topology}

# TESTES, APAGAR DEPOIS
data = load_data("../examples/case5")
print(data["pv_list"][0].profile)