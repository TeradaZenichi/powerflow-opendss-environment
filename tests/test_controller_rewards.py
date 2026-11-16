from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opendss_environment import Action, EnvironmentConfig, EVSEAction, MicrogridEnv, Reward, TotalAction
from opendss_environment.evse import EVSE, EVSEConfig, EVSessionConfig, utc_timestamp


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/case5"
EV_CASE = ROOT / "tests/fixtures/evse_case"


class ControllerRewardsTest(unittest.TestCase):
    def make_env(self, reward, case=CASE, steps=2):
        env = MicrogridEnv(EnvironmentConfig(case, episode_steps=steps, reward_function=reward))
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def test_scoped_terms_are_post_action_and_do_not_change_global_cost(self):
        reward = Reward(controllers={
            "battery": {"devices": {"bess": ["b1"]}, "buses": ["bus_004"],
                        "weights": {"voltage": 2, "energy_cost": 1, "throughput": 0.1},
                        "scales": {"voltage": 0.01},
                        "parameters": {"voltage": {"v_min": 0.999, "v_max": 1.001}}},
            "solar": {"devices": {"pv": ["pv1"]}, "weights": {"curtailment": 1}},
        })
        env = self.make_env(reward)
        with patch.object(reward, "evaluate", wraps=reward.evaluate) as evaluate:
            _, value, _, _, info = env.step(Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0)}))
            self.assertEqual(evaluate.call_count, 1)
        rewards = info["rewards"]
        self.assertEqual(value, -info["cost"])
        self.assertEqual(value, reward.get_cost(rewards))
        terms = reward.get_terms(rewards, "battery")
        voltages = info["bus_voltages_pu"]["bus_004"].values()
        expected = -sum(max(0.999 - v, 0) + max(v - 1.001, 0) for v in voltages)
        self.assertAlmostEqual(terms["voltage"]["raw"], expected)
        self.assertAlmostEqual(terms["voltage"]["contribution"], 2 * expected / 0.01)
        battery = env.bess_list[0]
        self.assertAlmostEqual(terms["throughput"]["raw"], -battery.eta_charge * battery.array_charge_kw[-1] * env.dt)
        self.assertAlmostEqual(reward.get_local(rewards, "battery"), sum(t["contribution"] for t in terms.values()))
        terms["voltage"]["raw"] = 99
        self.assertNotEqual(reward.get_terms(rewards, "battery")["voltage"]["raw"], 99)

    def test_teacher_profiles_reproduce_each_objective_and_tariff(self):
        for formulation, epsilon in (("single_phase_socp", 0), ("single_phase_active", 1e-6), ("three_phase_ivr", 1e-8)):
            with self.subTest(formulation=formulation):
                reward = Reward(global_objective="teacher", teacher_profile={"formulation": formulation, "feed_in_ratio": 0.3})
                env = self.make_env(reward)
                for _ in range(2):
                    _, value, _, _, info = env.step(Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0)}))
                    imported = info["grid_import_pu"] * env.data["base_kva"]
                    exported = info["grid_export_pu"] * env.data["base_kva"]
                    battery = env.bess_list[0]
                    throughput = battery.array_charge_kw[-1] + battery.array_discharge_kw[-1]
                    if formulation == "single_phase_active":
                        throughput += imported + exported
                    expected = -env.dt * (env.grid.prices[env.idx - 1] * (imported - 0.3 * exported) + epsilon * throughput)
                    self.assertAlmostEqual(value, expected)
                    self.assertEqual(value, reward.get_teacher(info["rewards"]))
                    self.assertEqual(reward.get_cost(info["rewards"]), -info["cost"])
                    profile = info["rewards"]["teacher_terms"]["profile"]
                    self.assertEqual(profile["formulation"], formulation)
                    self.assertEqual(profile["tie_break_penalty"], epsilon)

    def test_teacher_export_tariff_is_independent_of_actual_export_tariff(self):
        env = SimpleNamespace(current_cost=-0.25, current_grid_kw=10, dt=0.25, idx=0, bess_list=[],
                              data={"feed_in_tariff_ratio": 0.3}, grid=SimpleNamespace(prices=[0.2]))
        reward = Reward()
        self.assertAlmostEqual(reward.minimize_teacher_objective(env), 0.15)
        self.assertAlmostEqual(reward.minimize_cost(env), 0.25)

    def test_objectives_are_snapshotted_before_execution_and_do_not_mutate_configuration(self):
        controllers = {"battery": {"devices": {"bess": ["b1"]}, "weights": {"tracking": 1},
                                   "parameters": {"tracking": {"p_reference_kw": 3}}}}
        original = deepcopy(controllers)
        reward = Reward(controllers=controllers)
        env = self.make_env(reward)
        env.reward_objectives = {"battery": {"weights": {"tracking": 2},
                                            "parameters": {"tracking": {"p_reference_kw": 10}}}}
        from opendss_environment.simulation import solve_power_flow

        def solve(pending, action):
            pending.reward_objectives["battery"]["parameters"]["tracking"]["p_reference_kw"] = 99
            return solve_power_flow(pending, action)

        with patch("opendss_environment.microgrid_env.solve_power_flow", side_effect=solve):
            _, _, _, _, info = env.step(Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0)}))
        terms = reward.get_terms(info["rewards"], "battery")
        self.assertAlmostEqual(terms["tracking"]["raw"], 0, places=7)
        self.assertEqual(terms["tracking"]["weight"], 2)
        self.assertEqual(reward.controllers, original)
        self.assertEqual(env.reward_objectives["battery"]["parameters"]["tracking"]["p_reference_kw"], 10)

    def test_ev_v2g_has_settlement_throughput_and_recovery_penalties(self):
        reward = Reward(controllers={"ev": {"devices": {"evse": ["c1"]}, "weights": {
            "energy_cost": 1, "throughput": 1, "ev_infeasibility": 1, "ev_unmet": 1}}})
        env = self.make_env(reward, EV_CASE, 4)
        _, _, _, _, info = env.step(Action(evse={"c1": EVSEAction(-0.08)}))
        terms = reward.get_terms(info["rewards"], "ev")
        self.assertAlmostEqual(terms["throughput"]["raw"], -4 / 3)
        self.assertGreater(terms["energy_cost"]["raw"], 0)
        self.assertAlmostEqual(terms["ev_infeasibility"]["raw"], -7 / 3)
        self.assertEqual(terms["ev_unmet"]["raw"], 0)
        _, _, _, _, info = env.step(None)
        terms = reward.get_terms(info["rewards"], "ev")
        self.assertAlmostEqual(terms["ev_unmet"]["raw"], -10 / 3)
        self.assertEqual(terms["ev_infeasibility"]["raw"], 0)
        _, _, _, _, info = env.step(None)
        self.assertEqual(reward.get_terms(info["rewards"], "ev")["ev_unmet"]["raw"], 0)

    def test_ev_losses_and_throughput_include_each_session_efficiency(self):
        start = utc_timestamp("2026-01-01")
        cfg = EVSessionConfig(1, start, start + timedelta(hours=1), 0.5, 1, 20, 0.8, 12.5, 4)
        device = EVSE(EVSEConfig("e1", "load", 4, "controlled", v2g=True, standby_loss_kw=0.1), [cfg])
        device.operate(start, start + timedelta(hours=0.25), 4)
        self.assertAlmostEqual(device.array_total_loss_kwh[-1], 0.225)
        self.assertAlmostEqual(device.array_battery_throughput_kwh[-1], 0.825)
        device.operate(start + timedelta(hours=0.25), start + timedelta(hours=0.5), -4)
        self.assertAlmostEqual(device.array_total_loss_kwh[-1], 0.275)
        self.assertAlmostEqual(device.array_battery_throughput_kwh[-1], 1.275)

    def test_opposite_phase_powers_do_not_hide_battery_throughput(self):
        reward = Reward(controllers={"b": {"devices": {"bess": ["b1"]}, "weights": {"throughput": 1, "losses": 1}}})
        case = ROOT / "tests/fixtures/three_phase_volt_control/config_per_phase.json"
        env = self.make_env(reward, case)
        env.bess_list[0].eta_charge = 0.9
        env.bess_list[0].eta_discharge = 0.8
        _, _, _, _, info = env.step({"bess": {"b1": {"p_net_pu": {"a": 0.01, "b": -0.01, "c": 0.0}}},
                                    "pv": {"pv1": {"generation_pu": 0.0}}})
        terms = reward.get_terms(info["rewards"], "b")
        self.assertLess(terms["throughput"]["raw"], 0)
        self.assertLess(terms["losses"]["raw"], 0)

    def test_terminal_soc_and_previous_action_use_the_correct_interval(self):
        reward = Reward(controllers={"b": {"devices": {"bess": ["b1"]},
                                            "weights": {"soc_target": 1, "action_variation": 1},
                                            "parameters": {"soc_target": {"target": 0.5}}}})
        env = self.make_env(reward)
        action = Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0)})
        _, _, _, _, info = env.step(action)
        first = reward.get_terms(info["rewards"], "b")
        self.assertEqual(first["soc_target"]["raw"], 0)
        self.assertLess(first["action_variation"]["raw"], 0)
        _, _, _, _, info = env.step(action)
        last = reward.get_terms(info["rewards"], "b")
        self.assertAlmostEqual(last["soc_target"]["raw"], -0.19)
        self.assertAlmostEqual(last["action_variation"]["raw"], 0, places=7)

    def test_branch_overload_counts_worst_terminal_without_cancellation(self):
        flows = [{"element": "Line.x", "bus": bus, "phase": "a", "loading_pct": loading}
                 for bus, loading in (("upstream", 120), ("local", 130))]
        env = SimpleNamespace(current_branch_measurements=flows)
        reward = Reward(scope={"branches": ["Line.x"]})
        self.assertAlmostEqual(reward.minimize_branch_overload(env), -0.3)
        flows[1]["loading_pct"] = None
        with self.assertRaisesRegex(ValueError, "rating"):
            reward.minimize_branch_overload(env)

    def test_service_pays_only_requested_direction_caps_power_and_expires(self):
        env = SimpleNamespace(current_bus_voltages_pu={"local": {"a": 1}}, data={"base_kva": 100},
                              current_branch_measurements=[{"bus": "local", "p_flow_pu": -0.1, "q_flow_pu": 0.02}],
                              dt=0.25, idx=0, timestamps=SimpleNamespace(iloc=[utc_timestamp("2026-01-01")]))
        reward = Reward("local")
        value = reward.service_incentive(env, p_reference_kw=5, q_reference_kvar=-1, p_rate=2, q_rate=1,
                                         valid_until="2026-01-01T00:06:00Z")
        self.assertAlmostEqual(value, 1.1)
        self.assertEqual(reward.service_incentive(env, p_reference_kw=-5, p_rate=2), 0)
        self.assertEqual(reward.service_incentive(env, p_reference_kw=5, p_rate=2, valid_until="2025-12-31"), 0)

    def test_self_consumption_requires_provenance_and_does_not_count_losses_as_curtailment(self):
        pv = SimpleNamespace(id="pv", profile=[10], array_kw=[7], array_inverter_loss_kw=[1])
        env = SimpleNamespace(pv_list=[pv], idx=0, dt=0.25)
        reward = Reward(scope={"devices": {"pv": ["pv"]}})
        self.assertAlmostEqual(reward.minimize_curtailment(env), -0.5)
        with self.assertRaisesRegex(ValueError, "solar_export"):
            reward.maximize_self_consumption(env)
        self.assertAlmostEqual(reward.maximize_self_consumption(env, solar_export_kwh=0.3), -0.8)

    def test_invalid_configuration_rolls_back_the_step(self):
        bad_specs = [
            {"devices": {"bess": ["missing"]}, "weights": {"throughput": 1}},
            {"devices": {"bess": ["b1"]}, "weights": {"unknown": 1}},
            {"devices": {"bess": ["b1"]}, "weights": {"throughput": 1}, "scales": {"throughput": 0}},
            {"devices": {"bess": ["b1"]}, "weights": {"throughput": float("nan")}},
        ]
        for spec in bad_specs:
            with self.subTest(spec=spec):
                env = self.make_env(Reward(controllers={"b": spec}))
                with self.assertRaises(ValueError):
                    env.step(Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0)}))
                self.assertEqual(env.idx, 0)
                self.assertEqual(env.bess_list[0].array_kw, [])
                self.assertEqual(env.results.costs, [])

    def test_legacy_callable_keeps_its_scalar_and_exposes_cost(self):
        env = self.make_env(lambda env: -123.0)
        _, value, _, _, info = env.step(None)
        self.assertEqual(value, -123)
        self.assertEqual(info["rewards"]["global"], value)
        self.assertEqual(info["rewards"]["cost"], -info["cost"])

    def test_reward_failure_does_not_commit_new_ev_energy_measurements(self):
        def fail(env):
            raise ValueError("reward failed")

        env = self.make_env(fail, EV_CASE)
        with self.assertRaisesRegex(ValueError, "reward failed"):
            env.step(Action(evse={"c1": EVSEAction(-0.08)}))
        for device in env.evse_list:
            self.assertEqual(device.array_total_loss_kwh, [])
            self.assertEqual(device.array_battery_throughput_kwh, [])
            self.assertEqual(device.events, [])


if __name__ == "__main__":
    unittest.main()
