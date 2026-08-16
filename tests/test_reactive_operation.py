from __future__ import annotations

import math
from pathlib import Path
import unittest

from opendss_env.data import load_data
from opendss_env.devices_control import bess_control, pv_control
from opendss_env.elements import BESS
from opendss_env.envs.simulation import _update_snapshot_powers


CASE_PATH = Path(__file__).resolve().parents[1] / "examples" / "case5"


class _FakeDSS:
    def __init__(self):
        self.commands = []

    def text(self, command):
        self.commands.append(command)


class ReactiveOperationTest(unittest.TestCase):
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
        data = load_data(CASE_PATH)
        bess = data["bess_list"][0]
        pv = data["pv_list"][0]

        for idx in range(data["steps"]):
            bess_control(bess, idx, data["dt"])
            pv_control(pv, idx)

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
        data = load_data(CASE_PATH)
        dss = _FakeDSS()

        _update_snapshot_powers(data, dss, 0)

        bess_command = next(
            command for command in dss.commands if "Edit Load.b1" in command
        )
        self.assertIn("kvar=-9.364064", bess_command)
        self.assertAlmostEqual(data["bess_list"][0].array_kvar[0], 9.364064)

    def test_pv_q_keeps_injection_sign_at_opendss_boundary(self):
        data = load_data(CASE_PATH)
        dss = _FakeDSS()

        for idx in range(7):
            _update_snapshot_powers(data, dss, idx)

        pv_commands = [
            command for command in dss.commands if "Edit Generator.pv1" in command
        ]
        self.assertIn("kvar=27.09902", pv_commands[-1])
        self.assertAlmostEqual(data["pv_list"][0].array_kvar[-1], 27.099020)


if __name__ == "__main__":
    unittest.main()
