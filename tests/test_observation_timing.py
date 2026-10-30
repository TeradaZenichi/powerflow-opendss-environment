from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from opendss_environment import EnvironmentConfig, MicrogridEnv
from opendss_environment.elements import PVRequest
from opendss_environment.simulation import _update_snapshot_powers, solve_power_flow


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "case5"
FIXTURE = ROOT / "tests" / "fixtures" / "three_phase_volt_control"


def bus_voltage(env):
    return env.current_bus_voltages_pu["bus_004"]["a"]


class ObservationTimingTest(unittest.TestCase):
    def test_voltage_uses_current_loads_without_an_extra_energy_step(self):
        with TemporaryDirectory() as directory:
            mono = Path(directory) / "single_phase"
            shutil.copytree(CASE, mono)
            topology = mono / "dss" / "Master.dss"
            topology.write_text(topology.read_text(encoding="utf-8").replace("phases=3", "phases=1"), encoding="utf-8")
            for case in (mono, CASE):
                with self.subTest(case=case.name):
                    env = MicrogridEnv(EnvironmentConfig(case, 2, (bus_voltage,)))
                    _, reset_info = env.reset()
                    for voltage in reset_info["observation"]["buses"]["bus_001"]["v_before_pu"].values():
                        self.assertAlmostEqual(voltage, 1.0, places=5)
                    action = {
                        "bess": {"b1": {"p_net_kw": -10.0, "q_injection_kvar": 5.0}},
                        "pv": {"pv1": {"generation_kw": 0.0}},
                    }
                    state, _, terminated, _, info = env.step(env.normalizer.normalize(action))
                    info = env.normalizer.denormalize(info)
                    observation = env.observe()
                    reference = MicrogridEnv(EnvironmentConfig(case, 2, ()))
                    reference.reset()
                    reference.idx = 1
                    applied = _update_snapshot_powers(reference, action)
                    solve_power_flow(reference, applied)

                    self.assertFalse(terminated)
                    self.assertEqual(observation["timestamp"], "2026-01-01T01:00:00")
                    self.assertAlmostEqual(sum(observation["buses"]["bus_002"]["p_load_pu"].values()), 0.09)
                    for bus, values in observation["buses"].items():
                        for phase, voltage in values["v_before_pu"].items():
                            self.assertAlmostEqual(voltage, reference.current_bus_voltages_pu[bus][phase], places=7)
                            self.assertAlmostEqual(values["angle_before_deg"][phase], reference.current_bus_angles_deg[bus][phase], places=6)
                    before_voltage = observation["buses"]["bus_004"]["v_before_pu"]["a"]
                    self.assertAlmostEqual(float(state[0]), before_voltage, places=7)
                    self.assertGreater(abs(before_voltage - info["bus_voltages_pu"]["bus_004"]["a"]), 1e-5)
                    self.assertAlmostEqual(info["bus_voltages_pu"]["bus_004"]["a"], env.results.voltages_pu["bus_004"][0])
                    bess = env.bess_list[0]
                    self.assertEqual(bess.soc, info["executed_action"]["bess"]["b1"]["soc_after_frac"])
                    self.assertEqual(bess.array_soc, [bess.soc])
                    self.assertEqual(len(bess.array_kw), 1)
                    self.assertEqual(len(env.pv_list[0].array_kw), 1)
                    self.assertEqual(len(env.results.costs), 1)
                    self.assertEqual(len(env.grid.array_kw), 1)
                    self.assertEqual(observation, env.observe())
                    _, _, terminated, _, final_info = env.step(env.normalizer.normalize(action))
                    self.assertTrue(terminated)
                    self.assertEqual(len(bess.array_soc), 2)
                    self.assertEqual(len(env.results.costs), 2)
                    self.assertEqual(final_info["timestamp"], env.timestamps.iloc[1])
                    env.reset()
                    self.assertEqual(env.bess_list[0].array_soc, [])
                    self.assertEqual(env.results.costs, [])
                    self.assertTrue(all(value == 0.0 for value in env.observe()["bess"]["b1"]["previous_p_pu"].values()))

    def test_pv_availability_changes_without_rewriting_previous_powers(self):
        for config in ("config_per_phase.json", "config_delta.json"):
            with self.subTest(config=config):
                env = MicrogridEnv(EnvironmentConfig(FIXTURE / config, 2, ()))
                env.reset()
                pv = env.pv_list[0]
                pv.profile = pv.profile.copy()
                pv.profile[1] = 6.0
                for phase, available in zip(("a", "b", "c"), (1.0, 2.0, 3.0)):
                    pv.phase_profiles[phase] = pv.phase_profiles[phase].copy()
                    pv.phase_profiles[phase][1] = available
                action = {
                    "bess": {"b1": {
                        "p_net_kw": {"a": 1.0, "b": 2.0, "c": 3.0},
                        "q_injection_kvar": {"a": 0.0, "b": 1.0, "c": -1.0},
                    }},
                    "pv": {"pv1": {"generation_kw": {"a": 4.0, "b": 5.0, "c": 6.0}}},
                }
                _, _, _, _, info = env.step(env.normalizer.normalize(action))
                observation = env.observe()
                measured = env.current_device_measurements["pv"]["pv1"]["generation_kw"]
                for phase, expected in zip(("a", "b", "c"), (1.0, 2.0, 3.0)):
                    self.assertAlmostEqual(measured[phase], expected, places=5)
                    self.assertEqual(observation["pv"]["pv1"]["previous_p_pu"][phase], info["device_measurements"]["pv"]["pv1"]["generation_pu"][phase])
                self.assertEqual(pv.array_kw, [15.0])
                self.assertEqual(len(env.bess_list[0].array_soc), 1)
                self.assertEqual(observation, env.observe())

    def test_night_var_limits_only_the_new_snapshot(self):
        for night_var in (False, True):
            with self.subTest(night_var=night_var):
                env = MicrogridEnv(EnvironmentConfig(FIXTURE, 2, ()))
                env.reset()
                pv = env.pv_list[0]
                pv.control = "optimal"
                pv.night_var = night_var
                pv.q_loss_rated_kw = 0.5
                pv.profile = pv.profile.copy()
                pv.profile[1] = 0.0
                for phase in pv.phase_profiles:
                    pv.phase_profiles[phase] = pv.phase_profiles[phase].copy()
                    pv.phase_profiles[phase][1] = 0.0
                action = {"bess": {}, "pv": {"pv1": {"generation_kw": 20.0, "q_injection_kvar": 5.0}}}
                _, _, _, _, info = env.step(env.normalizer.normalize(action))
                measured = env.current_device_measurements["pv"]["pv1"]
                self.assertAlmostEqual(measured["generation_total_kw"], -0.005 if night_var else 0.0, places=5)
                self.assertAlmostEqual(measured["q_injection_total_kvar"], 5.0 if night_var else 0.0, places=5)
                self.assertEqual(pv.array_kw, [20.0])
                self.assertEqual(env.observe()["pv"]["pv1"]["previous_p_pu"], info["device_measurements"]["pv"]["pv1"]["generation_pu"])

    def test_voltage_controls_converge_before_the_next_action(self):
        for config in ("config.json", "config_delta_volt.json"):
            with self.subTest(config=config):
                env = MicrogridEnv(EnvironmentConfig(FIXTURE / config, 2, ()))
                env.reset()
                for load in env.load_list:
                    load.array_kw = load.array_kw.copy()
                    load.array_kvar = load.array_kvar.copy()
                    load.array_kw[1] *= 5.0
                    load.array_kvar[1] *= 5.0
                action = {"bess": {}, "pv": {"pv1": {"generation_kw": {"a": 10.0, "b": 10.0, "c": 10.0}}}}
                env.step(env.normalizer.normalize(action))
                pv = env.pv_list[0]
                snapshot = env.current_bus_snapshot
                measured = env.current_device_measurements["pv"]["pv1"]
                if pv.dispatch_mode == "aggregate":
                    request = PVRequest(pv.array_kw[-1], pv.array_kvar[-1], float(pv.profile[1]), snapshot.device_average_pu(pv))
                    point = pv.operating_point(request)
                    self.assertAlmostEqual(measured["q_injection_total_kvar"], point.q_injection_kvar, places=5)
                    self.assertAlmostEqual(measured["generation_total_kw"], point.generation_kw, places=5)
                else:
                    for phase, voltage in snapshot.device_pu(pv).items():
                        previous = pv._phase_points[phase]
                        request = PVRequest(previous.generation_kw, previous.q_injection_kvar, float(pv.phase_profiles[phase][1]), voltage, 1.0 / 3.0)
                        point = pv.operating_point(request)
                        self.assertAlmostEqual(measured["q_injection_kvar"][phase], point.q_injection_kvar, places=5)
                        self.assertAlmostEqual(measured["generation_kw"][phase], point.generation_kw, places=5)
                self.assertEqual(len(pv.array_kw), 1)
                self.assertEqual(len(env.results.costs), 1)
