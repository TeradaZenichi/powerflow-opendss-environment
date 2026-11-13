from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, PhaseAction, TotalAction
from opendss_environment.data import episode_data, load_data
from opendss_environment.results_scripts.summary import save_summary


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/three_phase_volt_control"


class ProfilesReportsTest(unittest.TestCase):
    def test_csv_load_power_is_preserved_above_default_dss_voltage_band(self):
        env = MicrogridEnv(EnvironmentConfig(FIXTURE / "config_per_phase.json"))
        env.reset(seed=42)
        _, _, _, _, info = env.step(Action(bess={"b1": TotalAction(0.0)}, pv={"pv1": TotalAction(0.0)}))
        if min(info["bus_voltages_pu"]["load"].values()) <= 1.05:
            raise ValueError("The test must exercise the high-voltage operating region")
        for load in env.load_list:
            env.dss.circuit.set_active_element(f"Load.{load.id}")
            measured = sum(env.dss.cktelement.powers[::2])
            if not math.isclose(measured, float(load.array_kw[0]), abs_tol=1e-7):
                raise ValueError("CSV load power must remain constant across the teacher voltage range")

    def test_same_column_in_different_files_keeps_each_device_profile(self):
        with TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            shutil.copytree(ROOT / "examples/case5", case)
            path = case / "devices.json"
            devices = json.loads(path.read_text(encoding="utf-8"))
            other = deepcopy(devices["pv"][0])
            filename, column = other["profile"].split(":")
            frame = pd.read_csv(case / filename)
            original = frame[column].to_numpy().copy()
            frame[column] += 10.0
            frame.to_csv(case / "other.csv", index=False)
            other.update(id="pv2", profile=f"other.csv:{column}")
            devices["pv"].append(other)
            path.write_text(json.dumps(devices), encoding="utf-8")
            episode = episode_data(load_data(case), 0, len(frame))
            for device, expected in zip(episode["pv_list"], (original, original + 10.0)):
                if not np.array_equal(device.profile, expected):
                    raise ValueError("PV profile was overwritten by another device")

    def test_reports_sum_phases_and_preserve_opposite_battery_powers(self):
        env = MicrogridEnv(EnvironmentConfig(FIXTURE / "config_per_phase.json"))
        env.reset(seed=42)
        action = Action(bess={"b1": PhaseAction({"a": 0.03, "b": -0.03, "c": 0.0})},
                        pv={"pv1": TotalAction(0.0)})
        for _ in range(env.steps):
            env.step(action)
        # A low voltage on another phase must affect the summary minimum.
        env.results.phase_voltages_pu["load"]["b"][-1] = 0.8
        with TemporaryDirectory() as directory:
            summary = save_summary(env.get_episode_results(), directory)
            buses = pd.read_csv(Path(directory) / "sim_buses_timeseries.csv")
            devices = pd.read_csv(Path(directory) / "sim_devices_timeseries.csv")
            if not np.allclose(buses["load_p_load_kw"], 6.0):
                raise ValueError("Bus load must sum all three phases")
            if not np.allclose(buses["load_a_p_load_kw"], 2.0):
                raise ValueError("Phase loads must remain available")
            if not np.allclose(devices["bess_b1_p_charge_kw"], 3.0) or not np.allclose(devices["bess_b1_p_discharge_kw"], 3.0):
                raise ValueError("Opposite phase powers must not cancel in charge/discharge reports")
            if not np.allclose(devices["bess_b1_a_p_net_kw"], 3.0):
                raise ValueError("Battery phase histories are missing")
            if summary["voltage_pu"]["minimum_voltage"] != 0.8:
                raise ValueError("Voltage summary must include every phase")
            if not math.isclose(summary["energy"]["bess_charge_kwh"], 3.0 * env.dt * env.steps):
                raise ValueError("Charge energy must account for phase dispatch")
            if len(pd.read_csv(Path(directory) / "sim_diagnostics_timeseries.csv")) != env.steps:
                raise ValueError("Report timestamps must cover the whole episode")

    def test_independent_sell_price_overrides_ratio(self):
        with TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            shutil.copytree(ROOT / "examples/case5", case)
            path = case / "price.csv"
            demand = pd.read_csv(case / "demand.csv")
            demand[demand.columns[1:]] *= 0.1
            demand.to_csv(case / "demand.csv", index=False)
            frame = pd.read_csv(path)
            frame["sell_price_per_kwh"] = np.linspace(0.01, 0.08, len(frame))
            frame.to_csv(path, index=False)
            env = MicrogridEnv(EnvironmentConfig(case))
            observation, _ = env.reset(seed=42)
            exports = 0
            for index in range(env.steps):
                action = Action(bess={"b1": TotalAction(0.0)}, pv={"pv1": TotalAction(0.26)})
                if observation["grid"]["sell_price_per_kwh"] != frame["sell_price_per_kwh"].iloc[index]:
                    raise ValueError("Observation must expose the selected sell tariff")
                observation, reward, _, _, info = env.step(action)
                buy, sell = frame.iloc[index][["price_per_kwh", "sell_price_per_kwh"]]
                imported = info["grid_import_pu"] * env.data["base_kva"]
                exported = info["grid_export_pu"] * env.data["base_kva"]
                expected = (buy * imported - sell * exported) * env.dt
                if not math.isclose(reward, -expected, abs_tol=1e-9):
                    raise ValueError("Executed cost does not use independent tariffs")
                exports += exported > 0
            if not exports:
                raise ValueError("The test must exercise export")
