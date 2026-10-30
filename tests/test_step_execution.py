from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, TotalAction


CASE = Path(__file__).resolve().parents[1] / "examples" / "case5"


class StepExecutionTest(unittest.TestCase):
    def setUp(self):
        self.env = MicrogridEnv(EnvironmentConfig(CASE, 2))
        self.before, _ = self.env.reset()
        self.action = Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0.0)})

    def assert_no_step_recorded(self):
        env = self.env
        self.assertEqual(env.idx, 0)
        self.assertEqual(env.bess_list[0].soc, 0.5)
        self.assertEqual(env.bess_list[0].array_soc, [])
        self.assertEqual(env.pv_list[0].array_kw, [])
        self.assertEqual(env.grid.array_kw, [])
        self.assertEqual(env.results.costs, [])
        self.assertEqual(env.episode_reward, 0.0)

    def test_all_device_commands_are_validated_before_execution(self):
        second = deepcopy(self.env.bess_list[0])
        second.id = "b2"
        self.env.bess_list.append(second)
        with patch.object(self.env.dss, "text", side_effect=AssertionError("DSS must not change")):
            with self.assertRaisesRegex(ValueError, "IDs"):
                self.env.step(self.action)
        self.assert_no_step_recorded()
        self.assertFalse(self.env._needs_reset)
        self.assertEqual(second.array_soc, [])

    def test_power_flow_failure_keeps_live_state_and_requires_reset(self):
        original_battery = self.env.bess_list[0]
        with patch("opendss_environment.microgrid_env.solve_power_flow", side_effect=RuntimeError("power flow failed")):
            with self.assertRaisesRegex(RuntimeError, "power flow failed"):
                self.env.step(self.action)
        self.assert_no_step_recorded()
        self.assertIs(self.env.bess_list[0], original_battery)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.env.step(self.action)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.env.observe()
        observation, _ = self.env.reset()
        self.assertEqual(observation, self.before)
        self.env.step(self.action)
        self.assertAlmostEqual(self.env.bess_list[0].soc, 0.595)
        self.assertEqual(len(self.env.bess_list[0].array_soc), 1)

    def test_failure_preparing_the_next_snapshot_does_not_commit(self):
        with patch("opendss_environment.microgrid_env.prepare_pre_action_observation", side_effect=RuntimeError("next snapshot failed")):
            with self.assertRaisesRegex(RuntimeError, "next snapshot failed"):
                self.env.step(self.action)
        self.assert_no_step_recorded()

    def test_failed_second_step_preserves_the_first_successful_step(self):
        self.env.step(self.action)
        bess = self.env.bess_list[0]
        soc = bess.soc
        costs = list(self.env.results.costs)
        powers = list(bess.array_kw)
        with patch("opendss_environment.microgrid_env.solve_power_flow", side_effect=RuntimeError("power flow failed")):
            with self.assertRaisesRegex(RuntimeError, "power flow failed"):
                self.env.step(self.action)
        self.assertEqual(self.env.idx, 1)
        self.assertEqual(bess.soc, soc)
        self.assertEqual(bess.array_kw, powers)
        self.assertEqual(len(bess.array_soc), 1)
        self.assertEqual(self.env.results.costs, costs)

    def test_reward_failure_does_not_commit(self):
        self.env.reward_function = Mock(side_effect=ValueError("reward failed"))
        with self.assertRaisesRegex(ValueError, "reward failed"):
            self.env.step(self.action)
        self.assert_no_step_recorded()

    def test_success_preserves_device_identity_and_records_once(self):
        bess, pv = self.env.bess_list[0], self.env.pv_list[0]
        results = self.env.results
        for step in range(2):
            _, _, _, _, info = self.env.step(self.action)
            self.assertIs(self.env.bess_list[0], bess)
            self.assertIs(self.env.pv_list[0], pv)
            self.assertIs(self.env.results, results)
            self.assertEqual(len(bess.array_soc), step + 1)
            self.assertEqual(len(results.costs), step + 1)
            self.assertEqual(info["interval_end"], f"2026-01-01T0{step + 1}:00:00")


if __name__ == "__main__":
    unittest.main()
