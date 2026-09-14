from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd

from opendss_env.data import (
    _require_timestamps,
    _validate_device_modes,
    episode_data,
    get_dt_hours,
    load_data,
)
from opendss_env.case_source import validate_case_config
from opendss_env.devices_control import (
    BESS_KVAR,
    BESS_KW,
    PV_KVAR,
    bess_control,
    pv_control,
)
from opendss_env.elements import BESS, PV, BESSConfig, PVConfig, PVRequest
from opendss_env.microgrid_env import EnvironmentConfig, MicrogridEnv
from opendss_env.simulation import (
    _update_snapshot_powers,
    collect_device_measurements,
)
from opendss_env.states import get_bess_soc


CASE_PATH = Path(__file__).resolve().parents[1] / "examples" / "case5"
VOLT_CONTROL_CASE = Path(__file__).resolve().parent / "fixtures" / "three_phase_volt_control"
PER_PHASE_CASE = VOLT_CONTROL_CASE / "config_per_phase.json"
DELTA_CASE = VOLT_CONTROL_CASE / "config_delta.json"
DELTA_VOLT_CASE = VOLT_CONTROL_CASE / "config_delta_volt.json"


class _FakeDSS:
    def __init__(self):
        self.commands = []

    def text(self, command):
        self.commands.append(command)


class _FakeTerminalDSS:
    def __init__(self):
        self.circuit = self
        self.cktelement = self
        self.active_element = None
        self.elements = {
            "Load.b1": [-6.0, -2.0, -6.0, -2.0, -6.0, -2.0, 0.0, 0.0],
            "Generator.pv1": [
                -30.0, -5.0, -30.0, -5.0, -30.0, -5.0, 0.0, 0.0
            ],
        }

    def set_active_element(self, name):
        self.active_element = name
        return int(name in self.elements)

    @property
    def num_conductors(self):
        return 4

    @property
    def node_order(self):
        return [1, 2, 3, 0]

    @property
    def powers(self):
        return self.elements[self.active_element]


class ReactiveOperationTest(unittest.TestCase):
    def episode(self):
        data = load_data(CASE_PATH)
        return data, episode_data(data, 0, data["steps"])

    def fake_env(self):
        data, episode = self.episode()
        return SimpleNamespace(data=data, dss=_FakeDSS(), idx=0, **episode)

    def environment(self, case=CASE_PATH, steps=24):
        config = EnvironmentConfig(case, steps, (get_bess_soc,))
        return MicrogridEnv(config)

    def test_bess_applies_capability_and_reactive_loss(self):
        bess = BESS(BESSConfig(
            id="b1", bus="bus_004", e_cap_kwh=100.0,
            p_charge_max_kw=40.0, p_discharge_max_kw=40.0,
            s_max_kva=50.0, reactive_control=True,
            q_loss_rated_kw=0.5, eta_charge=0.95,
            eta_discharge=0.95, soc_init_frac=0.5,
            soc_min_frac=0.1, soc_max_frac=1.0, cyclic_soc=True,
        ))

        p_bess, q_bess = bess.operate(40.0, 40.0, 1.0)

        self.assertAlmostEqual(p_bess, 30.0)
        self.assertAlmostEqual(q_bess, 40.0)
        self.assertAlmostEqual(bess.array_inverter_loss_kw[0], 0.32)
        self.assertAlmostEqual(bess.soc, 0.7818)
        self.assertLessEqual(math.hypot(p_bess, q_bess), bess.s_max_kva)

    def test_reference_controls_reproduce_teacher_reactive_profiles(self):
        data, episode = self.episode()
        bess = episode["bess_list"][0]
        pv = episode["pv_list"][0]

        for idx in range(data["steps"]):
            bess_control(bess, idx, data["dt"], BESS_KW, BESS_KVAR)
            pv_control(pv, idx, PV_KVAR)

        self.assertAlmostEqual(bess.soc, bess.soc_init_frac, places=5)
        self.assertGreater(max(bess.array_kvar), 25.0)
        self.assertGreater(max(pv.array_kvar), 49.0)
        self.assertTrue(all(
            math.hypot(p, q) <= bess.s_max_kva + 1e-9
            for p, q in zip(bess.array_kw, bess.array_kvar)
        ))
        self.assertTrue(all(
            math.hypot(p, q) <= pv.s_max_kva + 1e-9
            for p, q in zip(pv.array_p_net_kw, pv.array_kvar)
        ))
        self.assertTrue(all(
            q == 0.0
            for available, q in zip(pv.profile, pv.array_kvar)
            if available <= 0.0
        ))
        self.assertTrue(all(
            generation + loss <= available + 1e-9
            for generation, loss, available in zip(
                pv.array_kw,
                pv.array_inverter_loss_kw,
                pv.profile,
            )
        ))

    def test_pv_applies_volt_var_and_volt_watt_curves(self):
        common = {
            "id": "pv1",
            "bus": "bus_005",
            "p_max_kw": 100.0,
            "s_max_kva": 100.0,
            "q_loss_rated_kw": 0.0,
            "night_var": False,
            "profile": [100.0],
            "curtailable": True,
            "power_factor": 1.0,
        }
        volt_var = PV(PVConfig(control="volt-var", **common))
        volt_watt = PV(PVConfig(control="volt-watt", **common))

        _, q_injection = volt_var.apply(PVRequest(100.0, available_kw=100.0, voltage_pu=0.95))
        generation, _ = volt_watt.apply(PVRequest(100.0, available_kw=100.0, voltage_pu=1.10))

        self.assertGreater(q_injection, 0.0)
        self.assertAlmostEqual(generation, 0.0)

    def test_three_phase_voltage_control_converges_with_opendss(self):
        env = self.environment(VOLT_CONTROL_CASE, 2)
        env.reset()
        action = {
            "bess": {},
            "pv": {
                "pv1": {
                    "generation_kw": {"a": 10.0, "b": 10.0, "c": 10.0},
                    "q_injection_kvar": {"a": 0.0, "b": 0.0, "c": 0.0},
                }
            },
        }

        _, _, _, _, info = env.step(action)
        executed = info["executed_action"]["pv"]["pv1"]
        measured = info["device_measurements"]["pv"]["pv1"]

        self.assertLess(executed["generation_total_kw"], 30.0)
        self.assertLess(executed["q_injection_total_kvar"], 0.0)
        self.assertAlmostEqual(
            measured["generation_total_kw"],
            executed["generation_total_kw"],
            places=4,
        )
        self.assertAlmostEqual(
            measured["q_injection_total_kvar"],
            executed["q_injection_total_kvar"],
            places=4,
        )

    def test_bess_q_sign_is_converted_only_at_opendss_boundary(self):
        env = self.fake_env()
        _update_snapshot_powers(env)

        bess_command = next(
            command for command in env.dss.commands if "Edit Load.b1" in command
        )
        self.assertIn("kvar=-9.364064", bess_command)
        self.assertAlmostEqual(env.bess_list[0].array_kvar[0], 9.364064)

    def test_pv_q_keeps_injection_sign_at_opendss_boundary(self):
        env = self.fake_env()

        for idx in range(7):
            env.idx = idx
            _update_snapshot_powers(env)

        pv_commands = [
            command for command in env.dss.commands if "Edit Generator.pv1" in command
        ]
        self.assertIn("kvar=27.09902", pv_commands[-1])
        self.assertAlmostEqual(env.pv_list[0].array_kvar[-1], 27.099020)

    def test_named_action_reports_executed_values(self):
        env = self.fake_env()
        action = {
            "bess": {
                "b1": {"p_net_kw": -20.0, "q_injection_kvar": 10.0}
            },
            "pv": {
                "pv1": {"generation_kw": 0.0, "q_injection_kvar": 0.0}
            },
        }

        executed = _update_snapshot_powers(env, action)

        self.assertEqual(set(executed["bess"]["b1"]["p_net_kw"]), {"a"})
        self.assertAlmostEqual(
            sum(executed["bess"]["b1"]["p_net_kw"].values()), -20.0
        )
        self.assertAlmostEqual(
            sum(executed["bess"]["b1"]["q_injection_kvar"].values()), 10.0
        )
        self.assertAlmostEqual(executed["bess"]["b1"]["soc_before_frac"], 0.5)
        self.assertAlmostEqual(
            executed["bess"]["b1"]["soc_after_frac"], 0.2892736842
        )

    def test_step_reports_actual_terminal_powers(self):
        env = self.environment()
        env.reset()
        action = {
            "bess": {
                "b1": {"p_net_kw": -20.0, "q_injection_kvar": 10.0}
            },
            "pv": {
                "pv1": {"generation_kw": 0.0, "q_injection_kvar": 0.0}
            },
        }

        _, _, _, _, info = env.step(action)

        bess = info["device_measurements"]["bess"]["b1"]
        pv = info["device_measurements"]["pv"]["pv1"]
        self.assertEqual(set(bess["p_net_kw"]), {"a"})
        self.assertAlmostEqual(bess["p_net_total_kw"], -20.0, places=5)
        self.assertAlmostEqual(bess["q_injection_total_kvar"], 10.0, places=5)
        self.assertAlmostEqual(pv["generation_total_kw"], 0.0, places=5)
        self.assertAlmostEqual(pv["q_injection_total_kvar"], 0.0, places=5)

    def test_named_observation_is_pre_action_and_advances_causally(self):
        working_directory = Path.cwd()
        env = self.environment()
        _, reset_info = env.reset()
        self.assertEqual(Path.cwd(), working_directory)
        before = reset_info["observation"]
        action = {
            "bess": {
                "b1": {"p_net_kw": -20.0, "q_injection_kvar": 10.0}
            },
            "pv": {
                "pv1": {"generation_kw": 0.0, "q_injection_kvar": 0.0}
            },
        }

        _, _, _, _, info = env.step(action)
        after = env.observe()

        self.assertEqual(info["observation"], before)
        self.assertEqual(before["timestamp"], "2026-01-01T00:00:00")
        self.assertEqual(after["timestamp"], "2026-01-01T01:00:00")
        self.assertEqual(before["phase_order"], ["a"])
        self.assertAlmostEqual(before["bess"]["b1"]["soc_before_frac"], 0.5)
        self.assertAlmostEqual(
            after["bess"]["b1"]["soc_before_frac"],
            info["executed_action"]["bess"]["b1"]["soc_after_frac"],
        )
        self.assertAlmostEqual(
            after["bess"]["b1"]["previous_p_kw"]["a"], -20.0, places=5
        )
        self.assertAlmostEqual(
            after["bess"]["b1"]["previous_q_kvar"]["a"], 10.0, places=5
        )
        self.assertAlmostEqual(
            before["buses"]["bus_002"]["p_load_kw"]["a"], 100.0
        )
        self.assertGreater(
            before["buses"]["bus_002"]["v_before_pu"]["a"], 0.0
        )

    def test_time_contract_rejects_misaligned_or_irregular_series(self):
        expected = pd.Series(pd.date_range("2026-01-01", periods=3, freq="h"))
        shifted = expected + pd.Timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            _require_timestamps(shifted, expected, "pv.csv")

        irregular = pd.DataFrame({
            "timestamp": [expected.iloc[0], expected.iloc[1], expected.iloc[2] + pd.Timedelta(minutes=5)]
        })
        with self.assertRaisesRegex(ValueError, "equally spaced"):
            get_dt_hours(irregular)

    def test_configuration_file_and_directory_resolve_to_same_case(self):
        directory_data = load_data(CASE_PATH)
        file_data = load_data(CASE_PATH / "config.json")

        self.assertEqual(directory_data["case_path"], file_data["case_path"])
        self.assertEqual(directory_data["config_path"], file_data["config_path"])
        self.assertEqual(directory_data["steps"], file_data["steps"])

    def test_unknown_case_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_case_config({"schema_version": 2})

    def test_per_phase_and_delta_are_explicitly_validated(self):
        _validate_device_modes({
            "bess": [{"id": "b1", "dispatch_mode": "per_phase"}]
        })
        _validate_device_modes({
            "pv": [{
                "id": "pv1",
                "connection": "delta",
                "phases": ["a", "b", "c"],
            }]
        })
        with self.assertRaisesRegex(ValueError, "connection"):
            _validate_device_modes({
                "pv": [{"id": "pv1", "connection": "zigzag"}]
            })

    def test_per_phase_pv_and_bess_are_applied_to_separate_elements(self):
        env = self.environment(PER_PHASE_CASE, 2)
        env.reset()
        action = {
            "bess": {
                "b1": {
                    "p_net_kw": {"a": 1.0, "b": 2.0, "c": 3.0},
                    "q_injection_kvar": {"a": 1.0, "b": -1.0, "c": 0.0},
                }
            },
            "pv": {
                "pv1": {
                    "generation_kw": {"a": 1.0, "b": 2.0, "c": 3.0},
                    "q_injection_kvar": {"a": 0.5, "b": -0.5, "c": 0.0},
                }
            },
        }

        _, _, _, _, info = env.step(action)
        executed_bess = info["executed_action"]["bess"]["b1"]
        executed_pv = info["executed_action"]["pv"]["pv1"]
        measured_bess = info["device_measurements"]["bess"]["b1"]
        measured_pv = info["device_measurements"]["pv"]["pv1"]

        self.assertEqual(executed_bess["p_net_kw"], action["bess"]["b1"]["p_net_kw"])
        self.assertEqual(executed_pv["generation_kw"], action["pv"]["pv1"]["generation_kw"])
        for phase in ("a", "b", "c"):
            self.assertAlmostEqual(
                measured_bess["p_net_kw"][phase],
                executed_bess["p_net_kw"][phase],
                places=5,
            )
            self.assertAlmostEqual(
                measured_pv["generation_kw"][phase],
                executed_pv["generation_kw"][phase],
                places=5,
            )
        self.assertAlmostEqual(executed_bess["soc_after_frac"], 0.56)

    def test_delta_per_phase_devices_use_line_to_line_elements(self):
        env = self.environment(DELTA_CASE, 2)
        env.reset()
        action = {
            "bess": {
                "b1": {
                    "p_net_kw": {"a": 1.0, "b": 2.0, "c": 3.0},
                    "q_injection_kvar": {"a": 1.0, "b": -1.0, "c": 0.0},
                }
            },
            "pv": {
                "pv1": {
                    "generation_kw": {"a": 1.0, "b": 2.0, "c": 3.0},
                    "q_injection_kvar": {"a": 0.5, "b": -0.5, "c": 0.0},
                }
            },
        }

        _, _, _, _, info = env.step(action)
        for kind, device_id, field in (
            ("bess", "b1", "p_net_kw"),
            ("pv", "pv1", "generation_kw"),
        ):
            requested = action[kind][device_id][field]
            measured = info["device_measurements"][kind][device_id][field]
            for phase in ("a", "b", "c"):
                self.assertAlmostEqual(
                    measured[phase], requested[phase], places=5
                )

    def test_delta_volt_var_watt_uses_each_line_to_line_voltage(self):
        env = self.environment(DELTA_VOLT_CASE, 2)
        env.reset()
        action = {
            "bess": {},
            "pv": {
                "pv1": {
                    "generation_kw": {"a": 10.0, "b": 10.0, "c": 10.0},
                    "q_injection_kvar": {"a": 0.0, "b": 0.0, "c": 0.0},
                }
            },
        }

        _, _, _, _, info = env.step(action)
        executed = info["executed_action"]["pv"]["pv1"]
        measured = info["device_measurements"]["pv"]["pv1"]

        self.assertLess(executed["generation_total_kw"], 30.0)
        self.assertLess(executed["q_injection_total_kvar"], 0.0)
        for phase in ("a", "b", "c"):
            self.assertAlmostEqual(
                measured["generation_kw"][phase],
                executed["generation_kw"][phase],
                places=4,
            )
            self.assertAlmostEqual(
                measured["q_injection_kvar"][phase],
                executed["q_injection_kvar"][phase],
                places=4,
            )

    def test_actual_measurements_cover_charge_absorption_and_pv_curtailment(self):
        env = self.environment()
        env.reset()
        idle = {
            "bess": {
                "b1": {"p_net_kw": 0.0, "q_injection_kvar": 0.0}
            },
            "pv": {
                "pv1": {"generation_kw": 0.0, "q_injection_kvar": 0.0}
            },
        }
        for _ in range(6):
            env.step(idle)

        action = {
            "bess": {
                "b1": {"p_net_kw": 10.0, "q_injection_kvar": -5.0}
            },
            "pv": {
                "pv1": {"generation_kw": 5.0, "q_injection_kvar": -2.0}
            },
        }
        _, _, _, _, info = env.step(action)

        executed_bess = info["executed_action"]["bess"]["b1"]
        executed_pv = info["executed_action"]["pv"]["pv1"]
        measured_bess = info["device_measurements"]["bess"]["b1"]
        measured_pv = info["device_measurements"]["pv"]["pv1"]
        self.assertAlmostEqual(measured_bess["p_net_total_kw"], 10.0, places=5)
        self.assertAlmostEqual(
            measured_bess["q_injection_total_kvar"], -5.0, places=5
        )
        self.assertAlmostEqual(measured_pv["generation_total_kw"], 5.0, places=5)
        self.assertAlmostEqual(
            measured_pv["q_injection_total_kvar"], -2.0, places=5
        )
        self.assertAlmostEqual(executed_bess["p_net_total_kw"], 10.0)
        self.assertAlmostEqual(executed_pv["available_kw"], 10.0)
        self.assertGreater(executed_pv["curtailment_kw"], 4.9)

    def test_three_phase_terminal_measurements_follow_public_signs(self):
        env = SimpleNamespace(
            dss=_FakeTerminalDSS(),
            bess_list=[SimpleNamespace(id="b1")],
            pv_list=[SimpleNamespace(id="pv1")],
        )

        measurements = collect_device_measurements(env)

        bess = measurements["bess"]["b1"]
        pv = measurements["pv"]["pv1"]
        self.assertEqual(set(bess["p_net_kw"]), {"a", "b", "c"})
        self.assertEqual(bess["p_net_kw"], {"a": -6.0, "b": -6.0, "c": -6.0})
        self.assertEqual(bess["q_injection_total_kvar"], 6.0)
        self.assertEqual(pv["generation_total_kw"], 90.0)
        self.assertEqual(pv["q_injection_total_kvar"], 15.0)


if __name__ == "__main__":
    unittest.main()
