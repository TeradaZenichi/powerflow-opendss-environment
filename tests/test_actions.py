from pathlib import Path
import unittest

from opendss_env import Action, EnvironmentConfig, MicrogridEnv, PhaseAction, TotalAction


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "case5"
PHASED = ROOT / "tests" / "fixtures" / "three_phase_volt_control" / "config_per_phase.json"


class ActionsTest(unittest.TestCase):
    def test_total_and_balanced_phase_commands_have_the_same_effect(self):
        actions = (
            Action(bess={"b1": TotalAction(-0.018, 0.003)}, pv={"pv1": TotalAction(0.0)}),
            Action(bess={"b1": PhaseAction(dict.fromkeys("abc", -0.006), dict.fromkeys("abc", 0.001))}, pv={"pv1": TotalAction(0.0)}),
        )
        results = []
        for action in actions:
            env = MicrogridEnv(EnvironmentConfig(CASE, 2))
            env.reset()
            _, _, _, _, info = env.step(action)
            self.assertEqual(info["requested_action"], action.as_dict())
            executed = info["executed_action"]["bess"]["b1"]
            for value in executed["p_net_pu"].values():
                self.assertAlmostEqual(value, -0.006)
            results.append(executed)
        for name in ("p_net_total_pu", "q_injection_total_pu", "soc_after_frac", "inverter_loss_pu"):
            self.assertAlmostEqual(results[0][name], results[1][name])

    def test_total_command_is_distributed_to_per_phase_device(self):
        env = MicrogridEnv(EnvironmentConfig(PHASED, 2))
        env.reset()
        action = Action(bess={"b1": TotalAction(-0.009, 0.003)}, pv={"pv1": TotalAction(0.006)})
        _, _, _, _, info = env.step(action)
        executed = info["executed_action"]["bess"]["b1"]
        for phase in "abc":
            self.assertAlmostEqual(executed["p_net_pu"][phase], -0.003)
            self.assertAlmostEqual(executed["q_injection_pu"][phase], 0.001)

    def test_unequal_phase_command_is_never_silently_redistributed(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 2))
        before, _ = env.reset()
        action = Action(bess={"b1": PhaseAction({"a": -0.001, "b": -0.002, "c": -0.003})}, pv={"pv1": TotalAction(0.0)})
        with self.assertRaisesRegex(ValueError, "Unequal phase"):
            env.step(action)
        self.assertEqual(env.observe(), before)
        self.assertEqual(env.bess_list[0].array_soc, [])

    def test_wrong_phase_keys_are_rejected_before_execution(self):
        env = MicrogridEnv(EnvironmentConfig(PHASED, 2))
        before, _ = env.reset()
        action = Action(bess={"b1": PhaseAction({"a": 0.001})}, pv={"pv1": TotalAction(0.0)})
        with self.assertRaisesRegex(ValueError, "exactly"):
            env.step(action)
        self.assertEqual(env.observe(), before)


if __name__ == "__main__":
    unittest.main()
