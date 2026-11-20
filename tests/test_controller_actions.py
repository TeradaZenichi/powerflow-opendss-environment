from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from opendss_environment import EnvironmentConfig, MicrogridEnv
from opendss_environment.data import load_data


CASE = Path(__file__).resolve().parents[1] / "examples/case5"
EV_CASE = Path(__file__).parent / "fixtures/evse_case"


class ControllerActionsTest(unittest.TestCase):
    def env(self, case=CASE):
        env = MicrogridEnv(EnvironmentConfig(case, 2))
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_local_sampling_is_seeded_and_does_not_change_physics(self):
        env = self.env()
        before = env.observe()
        first = env.controller_actions({"bess": ["b1"]}, seed=42)
        second = env.controller_actions({"bess": ["b1"]}, seed=42)
        self.assertEqual(first.action_space.shape, (2,))
        for _ in range(5):
            action = first.action_space.sample()
            np.testing.assert_array_equal(action, second.action_space.sample())
            self.assertTrue(first.action_space.contains(action))
        self.assertEqual(env.observe(), before)
        command = first.command([-1, 1])
        self.assertAlmostEqual(command.bess["b1"].p_pu, -0.04)
        self.assertAlmostEqual(command.bess["b1"].q_pu, 0.05)
        self.assertEqual(command.pv, {})

    def test_two_controllers_compose_and_execute_without_overwriting_each_other(self):
        env = self.env()
        bess = env.controller_actions({"bess": ["b1"]})
        pv = env.controller_actions({"pv": ["pv1"]})
        first, second = bess.command([0.5, 0]), pv.command([-1, 0])
        observation = env.observe()
        combined = env.compose_actions(first, second)
        with patch.object(env, "observe", side_effect=AssertionError("Observation was already supplied")):
            self.assertEqual(env.compose_actions(first, second, observation=observation), combined)
        self.assertEqual(combined.bess, first.bess)
        self.assertEqual(combined.pv, second.pv)
        env.step(combined)
        self.assertEqual(env.idx, 1)
        with self.assertRaisesRegex(ValueError, "Multiple controllers"):
            env.compose_actions(first, first)

    def test_lean_info_matches_default_step_with_teacher_action(self):
        action = {
            "bess": {"b1": {"mode": "total", "p_net_pu": -0.02, "q_injection_pu": 0.01}},
            "pv": {"pv1": {"mode": "total", "generation_pu": 0.02, "q_injection_pu": 0.0}},
        }
        full_env, lean_env = self.env(), self.env()
        for _ in range(2):
            *full, info = full_env.step(action)
            *lean, lean_info = lean_env.step(action, full_info=False)
            self.assertEqual(lean, full)
            self.assertEqual(lean_info, {"rewards": info["rewards"]})
            self.assertEqual(info["requested_action"]["bess"], action["bess"])
            self.assertIn("executed_action", info)
            self.assertIn("observation_before", info)

    def test_one_controller_can_select_multiple_device_types(self):
        env = self.env()
        actions = env.controller_actions({"bess": ["b1"], "pv": ["pv1"]})
        command = actions.command([-1, 1, 0, 0])
        self.assertAlmostEqual(command.bess["b1"].p_pu, -0.04)
        self.assertAlmostEqual(command.pv["pv1"].p_pu, 0.26)
        env.step(env.compose_actions(command))

    def test_two_local_bess_controllers_can_act_in_the_same_step(self):
        data = load_data(CASE)
        data["devices"]["bess"].append(data["devices"]["bess"][0] | {"id": "b2", "bus": "bus_003"})
        env = MicrogridEnv(EnvironmentConfig(CASE, 2), data=data)
        self.addCleanup(env.close)
        env.reset()
        first = env.controller_actions({"bess": ["b1"]})
        second = env.controller_actions({"bess": ["b2"]})
        command = env.compose_actions(first.command([1, 0]), second.command([-1, 0]))
        _, _, _, _, info = env.step(command)
        self.assertGreater(info["executed_action"]["bess"]["b1"]["soc_after_frac"], 0.5)
        self.assertLess(info["executed_action"]["bess"]["b2"]["soc_after_frac"], 0.5)

    def test_defaults_keep_asap_and_only_controlled_evse_can_be_selected(self):
        env = self.env(EV_CASE)
        actions = env.controller_actions({"evse": ["c1"]})
        command = actions.command([-1, 0])
        self.assertAlmostEqual(command.evse["c1"].p_pu, -0.08)
        combined = env.compose_actions(command)
        self.assertEqual(set(combined.evse), {"c1"})
        _, _, _, _, info = env.step(combined)
        self.assertGreater(info["executed_action"]["evse"]["a1"]["energy_delivered_kwh"], 0)
        with self.assertRaisesRegex(ValueError, "noncontrollable"):
            env.controller_actions({"evse": ["a1"]})

    def test_invalid_vectors_and_scopes_are_rejected(self):
        env = self.env()
        for selection in ({}, {"bess": ["missing"]}, {"loads": ["load"]}):
            with self.assertRaises(ValueError):
                env.controller_actions(selection)
        actions = env.controller_actions({"bess": ["b1"]})
        for action in ([0], [0, 0, 0], [2, 0], [float("nan"), 0]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                actions.command(action)


if __name__ == "__main__":
    unittest.main()
