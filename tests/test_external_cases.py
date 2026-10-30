import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, TotalAction


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "case5"


def external_case(directory, representation):
    case = Path(directory) / "data" / representation
    shutil.copytree(SOURCE, case)
    config_path = case / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["network"]["representation"] = representation
    master = case / "dss" / "Master.dss"
    circuit = master.read_text(encoding="utf-8")
    if representation != "three_phase":
        circuit = circuit.replace("phases=3", "phases=1")
    if representation == "single_phase":
        config["base"]["v_base_kv"] = 0.23
        circuit = circuit.replace("basekV=11", "basekV=0.23")
        circuit = circuit.replace("VoltageBases=[11]", "VoltageBases=[0.3983716857408418]")
    # An externally owned master may redirect its actual network definition.
    (master.parent / "physical.dss").write_text(circuit, encoding="utf-8")
    master.write_text("Redirect physical.dss\n", encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    for csv in case.glob("*.csv"):
        frame = pd.read_csv(csv).iloc[:3].copy()
        frame["timestamp"] = pd.date_range("2026-01-01", periods=3, freq="15min")
        if representation == "single_phase" and csv.name != "price.csv":
            for column in frame.columns:
                if column != "timestamp":
                    frame[column] *= 0.001
        frame.to_csv(csv, index=False)
    return case


class ExternalCasesTest(unittest.TestCase):
    def test_single_phase_device_on_three_phase_network_uses_its_declared_phase(self):
        with TemporaryDirectory() as directory:
            case = external_case(directory, "three_phase")
            path = case / "devices.json"
            devices = json.loads(path.read_text(encoding="utf-8"))
            devices["bess"][0]["phases"] = ["b"]
            path.write_text(json.dumps(devices), encoding="utf-8")
            env = MicrogridEnv(EnvironmentConfig(case))
            env.reset()
            action = Action(bess={"b1": TotalAction(0.001)}, pv={"pv1": TotalAction(0.0)})
            observation, _, _, _, info = env.step(action)
            measured = info["device_measurements"]["bess"]["b1"]["p_net_pu"]
            self.assertEqual(set(measured), {"b"})
            self.assertAlmostEqual(measured["b"], 0.001, places=7)
            self.assertEqual(set(observation["bess"]["b1"]["previous_p_pu"]), {"b"})

    def test_external_data_supports_all_three_representations(self):
        with TemporaryDirectory() as directory:
            for representation in ("single_phase", "single_phase_equivalent", "three_phase"):
                with self.subTest(representation=representation):
                    case = external_case(directory, representation)
                    for source in (case, case / "config.json"):
                        env = MicrogridEnv(EnvironmentConfig(source))
                        observation, _ = env.reset()
                        self.assertEqual(observation["network"]["representation"], representation)
                        self.assertEqual(observation["dt_h"], 0.25)
                        for value in observation["buses"]["bus_001"]["v_before_pu"].values():
                            self.assertAlmostEqual(value, 1.0, places=5)
                        action = Action(bess={"b1": TotalAction(0.0001)}, pv={"pv1": TotalAction(0.0)})
                        for step in range(3):
                            observation, _, terminated, _, info = env.step(action)
                            expected_soc = 0.5 + (step + 1) * 0.95 * 0.1 * 0.25 / 100.0
                            self.assertAlmostEqual(env.bess_list[0].soc, expected_soc)
                            expected_end = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * (step + 1))
                            self.assertEqual(info["interval_end"], expected_end.isoformat())
                            self.assertEqual(info["network"], env.network.as_dict())
                            env.normalizer.require_pu(info)
                        self.assertTrue(terminated)
                        self.assertIsNone(observation)

    def test_declared_representation_must_match_the_circuit(self):
        with TemporaryDirectory() as directory:
            case = external_case(directory, "single_phase")
            path = case / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["network"]["representation"] = "three_phase"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "representation"):
                MicrogridEnv(EnvironmentConfig(case, 3))


if __name__ == "__main__":
    unittest.main()
