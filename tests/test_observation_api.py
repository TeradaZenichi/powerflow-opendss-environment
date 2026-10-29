from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from opendss_env import EnvironmentConfig, MicrogridEnv
from opendss_env.states import get_bess_soc, get_hour


CASE = Path(__file__).resolve().parents[1] / "examples" / "case5"


class ObservationApiTest(unittest.TestCase):
    def test_step_rejects_physical_or_nonfinite_actions_before_execution(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 2))
        before, _ = env.reset()
        invalid = (
            {"bess": {"b1": {"p_net_kw": -10.0}}},
            {"bess": {"b1": {"p_net_pu": float("nan")}}},
        )
        for action in invalid:
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.observe(), before)
            self.assertEqual(env.bess_list[0].array_soc, [])
            self.assertEqual(env.results.costs, [])

    def test_pu_results_can_be_converted_back_to_physical_units(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 2))
        observation, _ = env.reset()
        self.assertEqual(observation["units"], "pu")
        self.assertEqual(observation["base"], {"s_base_kva": 1000.0, "v_base_kv": 11.0})
        action = {
            "bess": {"b1": {"p_net_pu": -0.01, "q_injection_pu": 0.005}},
            "pv": {"pv1": {"generation_pu": 0.0}},
        }
        _, _, _, _, info = env.step(action)
        env.normalizer.require_pu(info)
        self.assertEqual(info["units"], "pu")
        self.assertAlmostEqual(info["device_measurements"]["bess"]["b1"]["p_net_total_pu"], -0.01)
        physical = env.normalizer.denormalize(info)
        self.assertEqual(physical["units"], "physical")
        self.assertAlmostEqual(physical["device_measurements"]["bess"]["b1"]["p_net_total_kw"], -10.0, places=5)
        self.assertEqual(physical["bus_voltages_pu"], info["bus_voltages_pu"])
        self.assertEqual(physical["cost"], info["cost"])

    def test_reset_returns_named_pu_observations_by_default(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 2))
        observation, info = env.reset()
        self.assertIsInstance(observation, dict)
        self.assertEqual(observation, env.observe())
        self.assertEqual(observation, info["observation"])
        self.assertEqual(observation["observation_schema_version"], 2)
        self.assertEqual(observation["timestamp"], "2026-01-01T00:00:00")
        self.assertAlmostEqual(sum(observation["buses"]["bus_002"]["p_load_pu"].values()), 0.1)
        self.assertAlmostEqual(observation["bess"]["b1"]["soc_before_frac"], 0.5)
        self.assertEqual(json.loads(json.dumps(observation)), observation)
        observation["bess"]["b1"]["soc_before_frac"] = 0.1
        observation["buses"]["bus_002"]["p_load_pu"]["a"] = 999.0
        self.assertEqual(info["observation"]["bess"]["b1"]["soc_before_frac"], 0.5)
        self.assertEqual(env.observe(), info["observation"])

    def test_step_returns_the_next_observation_and_records_the_previous_one(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 2))
        before, _ = env.reset()
        action = {
            "bess": {"b1": {"p_net_pu": -0.01, "q_injection_pu": 0.005}},
            "pv": {"pv1": {"generation_pu": 0.0}},
        }
        requested = deepcopy(action)
        after, _, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(after, env.observe())
        self.assertEqual(after["timestamp"], "2026-01-01T01:00:00")
        self.assertEqual(info["observation_before"], before)
        self.assertEqual(info["observation"], before)
        self.assertEqual(info["timestamp"].isoformat(), before["timestamp"])
        self.assertAlmostEqual(sum(after["buses"]["bus_002"]["p_load_pu"].values()), 0.09)
        self.assertAlmostEqual(sum(after["bess"]["b1"]["previous_p_pu"].values()), -0.01, places=5)
        self.assertEqual(after["bess"]["b1"]["soc_before_frac"], info["executed_action"]["bess"]["b1"]["soc_after_frac"])
        self.assertEqual(len(env.bess_list[0].array_soc), 1)
        self.assertEqual(len(env.results.costs), 1)
        action["bess"]["b1"]["p_net_pu"] = 999.0
        self.assertEqual(info["requested_action"], requested)
        info["observation_before"]["bess"]["b1"]["soc_before_frac"] = 0.1
        self.assertEqual(info["observation"], before)
        after["bess"]["b1"]["soc_before_frac"] = 0.9
        self.assertEqual(env.observe()["bess"]["b1"]["soc_before_frac"], env.bess_list[0].soc)

    def test_final_step_has_no_next_decision_observation(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 1))
        before, _ = env.reset()
        observation, _, terminated, truncated, info = env.step()
        self.assertIsNone(observation)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["observation_before"], before)
        self.assertIn("bus_voltages_pu", info)
        self.assertEqual(len(env.bess_list[0].array_soc), 1)
        with self.assertRaises(RuntimeError):
            env.step()
        self.assertEqual(len(env.bess_list[0].array_soc), 1)
        reset_observation, _ = env.reset()
        self.assertEqual(reset_observation["timestamp"], before["timestamp"])
        self.assertEqual(reset_observation["bess"]["b1"]["soc_before_frac"], 0.5)
        self.assertEqual(env.bess_list[0].array_soc, [])

    def test_explicit_state_functions_preserve_legacy_numeric_states(self):
        config = EnvironmentConfig(CASE, 2, (get_hour, get_bess_soc))
        env = MicrogridEnv(config)
        state, reset_info = env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.tolist(), [0.0, 0.5])
        state, _, _, _, info = env.step()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (2,))
        self.assertEqual(float(state[0]), 1.0)
        self.assertEqual(info["observation_before"], reset_info["observation"])
        self.assertIsInstance(env.observe(), dict)


if __name__ == "__main__":
    unittest.main()
