from pathlib import Path
from types import SimpleNamespace
import unittest

from opendss_environment import Action, EnvironmentConfig, EVSEAction, MicrogridEnv, Reward, TotalAction


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/case5"


class BoundaryCostsTest(unittest.TestCase):
    def make_snapshot(self):
        battery = SimpleNamespace(id="b1", bus="local", eta_charge=0.9, eta_discharge=0.8,
                                  array_charge_kw=[0], array_discharge_kw=[4], array_inverter_loss_kw=[0.2])
        return SimpleNamespace(
            dt=0.25, idx=0, current_cost=-0.1, current_grid_kw=10,
            grid=SimpleNamespace(prices=[0.2], sell_prices=[0.04]),
            data={"base_kva": 100, "grid_limits": {"bus": "source"}},
            current_bus_voltages_pu={bus: {"a": 1} for bus in ("source", "local", "solar")},
            current_branch_measurements=[
                {"bus": "source", "p_flow_pu": -0.1},
                {"bus": "local", "p_flow_pu": -0.3},
                {"bus": "local", "p_flow_pu": 0.1},
                {"bus": "solar", "p_flow_pu": 0.3},
            ],
            bess_list=[battery], pv_list=[SimpleNamespace(id="pv1", bus="local")], evse_list=[],
            current_device_measurements={"bess": {"b1": {"p_net_total_kw": -4, "q_injection_total_kvar": 0}},
                                         "pv": {"pv1": {"generation_total_kw": 8, "q_injection_total_kvar": 0}}},
        )

    def test_feeder_export_node_import_and_bess_discharge_are_distinct(self):
        env = self.make_snapshot()
        reward = Reward(scope={"buses": ["local"], "devices": {"bess": ["b1"]}})
        self.assertAlmostEqual(reward.feeder_cost(env), -0.1)
        self.assertAlmostEqual(reward.node_cost(env), 1.0)
        self.assertAlmostEqual(reward.device_cost(env), -0.04)
        self.assertEqual(reward.minimize_feeder_cost(env), reward.minimize_cost(env))
        self.assertAlmostEqual(reward.minimize_node_cost(env), -1.0)
        self.assertAlmostEqual(reward.minimize_device_cost(env), 0.04)
        self.assertAlmostEqual(reward.node_cost(env, buy_price=0.3), 1.5)
        self.assertAlmostEqual(reward.device_cost(env, sell_price=0.08), -0.08)

    def test_nodes_and_devices_settle_separately_but_phases_net_at_one_meter(self):
        env = self.make_snapshot()
        reward = Reward(scope={"buses": ["local", "solar"],
                              "devices": {"bess": ["b1"], "pv": ["pv1"]}})
        # Two node meters: 20 kW import and 30 kW export, not one 10 kW export.
        self.assertAlmostEqual(reward.node_cost(env), 0.7)
        env.current_device_measurements["bess"]["b1"]["p_net_total_kw"] = 8
        self.assertAlmostEqual(reward.device_cost(env), 0.32)
        self.assertAlmostEqual(reward.minimize_energy_cost(env), 0.0)
        env.current_branch_measurements = [
            {"bus": "local", "p_flow_pu": -0.1, "phase": "a"},
            {"bus": "local", "p_flow_pu": 0.05, "phase": "b"},
        ]
        self.assertAlmostEqual(Reward("local").node_cost(env), 0.25)

    def test_source_node_excludes_external_source_and_keeps_its_own_consumption(self):
        env = self.make_snapshot()
        reward = Reward("source")
        self.assertAlmostEqual(reward.node_cost(env), 0)
        env.current_grid_kw = 5  # 10 kW received from branches, 5 exported externally.
        self.assertAlmostEqual(reward.node_cost(env), 0.25)

    def test_device_wear_and_service_are_explicit_and_do_not_reprice_losses(self):
        env = self.make_snapshot()
        reward = Reward(scope={"devices": {"bess": ["b1"]}})
        # 1.3 cell-side kWh, including reactive converter loss; energy uses net AC.
        self.assertAlmostEqual(reward.device_cost(env, throughput_price=0.1, service_payment=0.03), 0.06)
        self.assertAlmostEqual(reward.device_cost(env, buy_price=0, sell_price=0, throughput_price=0.1), 0.13)
        self.assertAlmostEqual(reward.device_cost(env), -0.04)
        self.assertEqual(env.current_cost, -0.1)

    def test_invalid_parameters_and_missing_boundaries_fail_explicitly(self):
        env = self.make_snapshot()
        reward = Reward("local")
        for parameters in ({"buy_price": float("nan")}, {"sell_price": float("inf")},
                           {"throughput_price": -1}, {"service_payment": -1}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                reward.device_cost(env, **parameters)
        with self.assertRaisesRegex(ValueError, "buses"):
            Reward().node_cost(env)
        with self.assertRaisesRegex(ValueError, "devices"):
            Reward(scope={"buses": ["local"]}).device_cost(env)
        # Finite negative market prices remain supported.
        self.assertAlmostEqual(reward.node_cost(env, buy_price=-0.1), -0.5)

    def test_live_costs_are_snapshotted_before_time_advances_and_leave_grid_bill_unchanged(self):
        reward = Reward(controllers={
            "feeder": {"weights": {"feeder_cost": 2}, "scales": {"feeder_cost": 10}},
            "node": {"devices": {"bess": ["b1"]}, "buses": ["bus_004"],
                     "weights": {"node_cost": 1}, "parameters": {"node_cost": {"buy_price": 0.3}}},
            "device": {"devices": {"bess": ["b1"]}, "weights": {"device_cost": 1},
                       "parameters": {"device_cost": {"buy_price": 0.4, "throughput_price": 0.1}}},
            "source": {"buses": ["bus_001"], "weights": {"node_cost": 1}},
        })
        env = MicrogridEnv(EnvironmentConfig(CASE, 2, reward_function=reward))
        self.addCleanup(env.close)
        env.reset(seed=42)
        action = Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0)})
        _, value, _, _, first = env.step(action)
        costs = reward.get_costs(first["rewards"])
        self.assertEqual(costs, first["costs"])
        self.assertEqual(costs["feeder"], first["cost"])
        self.assertEqual(costs["controllers"]["feeder"]["feeder"], first["cost"])
        self.assertEqual(value, -first["cost"])
        self.assertEqual(reward.get_cost(first["rewards"]), value)
        self.assertAlmostEqual(reward.get_local(first["rewards"], "feeder"), -2 * first["cost"] / 10)
        self.assertAlmostEqual(costs["controllers"]["source"]["node"], 0, places=6)
        # bus_004 has a 72 kW load as well as the 10 kW BESS charge.
        self.assertAlmostEqual(costs["controllers"]["node"]["node"], 24.6, places=5)
        self.assertAlmostEqual(costs["controllers"]["device"]["device"], 4.95, places=5)
        self.assertEqual(env.normalizer.denormalize(first)["costs"], first["costs"])
        copy = reward.get_costs(first["rewards"], "device")
        copy["device"] = 999
        env.reward_objectives = {"device": {"parameters": {"device_cost": {"buy_price": 0.5}}}}
        _, _, _, _, second = env.step(action)
        self.assertAlmostEqual(reward.get_costs(second["rewards"], "device")["device"], 5.95, places=5)
        self.assertAlmostEqual(reward.get_costs(first["rewards"], "device")["device"], 4.95, places=5)
        self.assertEqual(env.results.costs[0], first["cost"])

    def test_v2g_device_cost_uses_executed_energy_and_cell_throughput(self):
        reward = Reward(controllers={"ev": {"devices": {"evse": ["c1"]},
                                            "weights": {"device_cost": 1, "throughput": 1},
                                            "parameters": {"device_cost": {"sell_price": 0.1,
                                                                             "throughput_price": 0.02}}}})
        env = MicrogridEnv(EnvironmentConfig(ROOT / "tests/fixtures/evse_case", 2, reward_function=reward))
        self.addCleanup(env.close)
        env.reset(seed=42)
        _, _, _, _, info = env.step(Action(evse={"c1": EVSEAction(-0.08)}))
        terms = reward.get_terms(info["rewards"], "ev")
        # V2G runs for 10 minutes at 8 kW: 4/3 kWh at both AC and cell sides.
        self.assertAlmostEqual(reward.get_costs(info["rewards"], "ev")["device"], (0.02 - 0.1) * 4 / 3)
        self.assertAlmostEqual(terms["device_cost"]["raw"], (0.1 - 0.02) * 4 / 3)

    def test_invalid_cost_configuration_rolls_back_execution(self):
        for spec in ({"devices": {"bess": ["b1"]}, "weights": {"node_cost": 1}},
                     {"devices": {"bess": ["b1"]}, "weights": {"device_cost": 1},
                      "parameters": {"device_cost": {"throughput_price": -1}}}):
            with self.subTest(spec=spec):
                env = MicrogridEnv(EnvironmentConfig(CASE, 1, reward_function=Reward(controllers={"b": spec})))
                self.addCleanup(env.close)
                env.reset(seed=42)
                with self.assertRaises(ValueError):
                    env.step(None)
                self.assertEqual(env.idx, 0)
                self.assertEqual(env.results.costs, [])
                self.assertEqual(env.bess_list[0].array_kw, [])


if __name__ == "__main__":
    unittest.main()
