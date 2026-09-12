from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from opendss_env.data import episode_data, load_data
from opendss_env.devices_control import (
    BESS_KVAR,
    BESS_KW,
    PV_KVAR,
    bess_control,
    pv_control,
)
from opendss_env.elements import BESS
from opendss_env.microgrid_env import MicrogridEnv
from opendss_env.simulation import (
    _update_snapshot_powers,
    collect_device_measurements,
)
from opendss_env.states import get_bess_soc


CASE_PATH = Path(__file__).resolve().parents[1] / "examples" / "case5"


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

    def test_bess_applies_capability_and_reactive_loss(self):
        bess = BESS(
            id="b1", bus="bus_004", e_cap_kwh=100.0,
            p_charge_max_kw=40.0, p_discharge_max_kw=40.0,
            s_max_kva=50.0, reactive_control=True,
            q_loss_rated_kw=0.5, eta_charge=0.95,
            eta_discharge=0.95, soc_init_frac=0.5,
            soc_min_frac=0.1, soc_max_frac=1.0, cyclic_soc=True,
        )

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
        env = MicrogridEnv(
            CASE_PATH,
            episode_steps=24,
            num_episodes=1,
            start_episode=0,
            state_functions=[get_bess_soc],
        )
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

    def test_actual_measurements_cover_charge_absorption_and_pv_curtailment(self):
        env = MicrogridEnv(
            CASE_PATH,
            episode_steps=24,
            num_episodes=1,
            start_episode=0,
            state_functions=[get_bess_soc],
        )
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
