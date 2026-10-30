import math
import unittest

from opendss_environment.elements import BESS, BESSConfig


def battery(soc):
    return BESS(BESSConfig(
        id="b1", bus="bus", e_cap_kwh=10.0,
        p_charge_max_kw=15.0, p_discharge_max_kw=15.0,
        s_max_kva=30.0, reactive_control=True, q_loss_rated_kw=0.5,
        eta_charge=0.95, eta_discharge=0.9, soc_init_frac=soc,
        soc_min_frac=0.1, soc_max_frac=0.9, cyclic_soc=False,
        phases=("a", "b", "c"), dispatch_mode="per_phase",
    ))


class BessEnergyTest(unittest.TestCase):
    def test_phase_dispatch_conserves_energy_at_both_soc_limits(self):
        requests = (
            {"a": 0.0, "b": 0.0, "c": 0.0},
            {"a": -15.0, "b": -15.0, "c": -15.0},
            {"a": 15.0, "b": 15.0, "c": 15.0},
            {"a": 10.0, "b": -10.0, "c": 0.0},
        )
        for soc in (0.1, 0.5, 0.9):
            for dt in (0.25, 1.0):
                for request in requests:
                    with self.subTest(soc=soc, dt=dt, request=request):
                        bess = battery(soc)
                        before = dict(request)
                        q_request = {"a": 3.0, "b": -2.0, "c": 1.0}
                        p, q = bess.operate_phases(request, q_request, dt)
                        charge = sum(max(value, 0.0) for value in p.values())
                        discharge = sum(max(-value, 0.0) for value in p.values())
                        loss = bess.array_inverter_loss_kw[-1]
                        energy = (bess.eta_charge * charge - discharge / bess.eta_discharge - loss) * dt
                        self.assertAlmostEqual((bess.soc - soc) * bess.e_cap_kwh, energy, places=10)
                        self.assertGreaterEqual(bess.soc, bess.soc_min_frac)
                        self.assertLessEqual(bess.soc, bess.soc_max_frac)
                        self.assertLessEqual(charge, bess.p_charge_max_kw + 1e-10)
                        self.assertLessEqual(discharge, bess.p_discharge_max_kw + 1e-10)
                        for phase in p:
                            self.assertLessEqual(math.hypot(p[phase], q[phase]), bess.s_max_kva / 3 + 1e-10)
                        self.assertEqual(request, before)
                        self.assertEqual(len(bess.array_soc), 1)

    def test_empty_battery_cannot_supply_unfunded_reactive_losses(self):
        bess = battery(0.1)
        p, q = bess.operate_phases(dict.fromkeys("abc", 0.0), dict.fromkeys("abc", 10.0), 1.0)
        self.assertEqual(sum(p.values()), 0.0)
        self.assertEqual(sum(q.values()), 0.0)
        self.assertEqual(bess.array_inverter_loss_kw[-1], 0.0)
        self.assertEqual(bess.soc, 0.1)

    def test_balanced_phase_dispatch_matches_total_dispatch(self):
        for soc in (0.1, 0.5, 0.9):
            for dt in (0.25, 1.0):
                for active in (-9.0, 0.0, 9.0):
                    with self.subTest(soc=soc, dt=dt, active=active):
                        total = battery(soc)
                        phased = battery(soc)
                        p_total, q_total = total.operate(active, 6.0, dt)
                        p, q = phased.operate_phases(dict.fromkeys("abc", active / 3), dict.fromkeys("abc", 2.0), dt)
                        self.assertAlmostEqual(sum(p.values()), p_total)
                        self.assertAlmostEqual(sum(q.values()), q_total)
                        self.assertAlmostEqual(phased.soc, total.soc)


if __name__ == "__main__":
    unittest.main()
