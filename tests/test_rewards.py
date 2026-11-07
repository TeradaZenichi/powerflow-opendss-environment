import cmath
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from opendss_environment import EnvironmentConfig, MicrogridEnv, Reward
from opendss_environment.rewards import minimize_cost
from test_external_cases import external_case


ROOT = Path(__file__).resolve().parents[1]


class RewardTest(unittest.TestCase):
    def test_cost_remains_the_default_and_local_rewards_use_executed_voltages(self):
        with TemporaryDirectory() as directory:
            cases = (external_case(directory, "single_phase_equivalent"),
                     ROOT / "tests/fixtures/three_phase_volt_control/config_per_phase.json")
            for case in cases:
                for local in (False, True):
                    with self.subTest(case=str(case), local=local):
                        env = MicrogridEnv(EnvironmentConfig(case, episode_steps=2))
                        observation, _ = env.reset(seed=42)
                        bus = next(device.bus for device in env.bess_list)
                        reward = Reward(bus, v_min=0.999, v_max=1.001)
                        env.reward_function = reward.minimize_voltage_violation if local else reward
                        action = {
                            "bess": {key: {"p_net_pu": 0.01, "q_injection_pu": 0.0} for key in observation["bess"]},
                            "pv": {key: {"generation_pu": sum(value["available_pu"].values()), "q_injection_pu": 0.0}
                                   for key, value in observation["pv"].items()},
                        }
                        _, value, _, _, info = env.step(action)
                        voltages = info["bus_voltages_pu"][bus]
                        expected = -sum(max(0.999 - v, 0) + max(v - 1.001, 0) for v in voltages.values()) if local else -info["cost"]
                        if not math.isclose(value, expected, abs_tol=1e-12):
                            self.fail("Reward must evaluate executed measurements, including every phase")
                        if minimize_cost(env) != -info["cost"]:
                            self.fail("The public cost function must remain compatible")
                        env.close()

    def test_voltage_penalties_include_all_phases_at_only_the_selected_bus(self):
        env = SimpleNamespace(current_bus_voltages_pu={"local": {"a": 1.0, "b": 1.1, "c": 0.9},
                                                     "remote": {"a": 2.0}})
        reward = Reward("local")
        if not math.isclose(reward.minimize_voltage_violation(env), -0.1, abs_tol=1e-12):
            self.fail("Both high and low phases must contribute to the band penalty")
        if not math.isclose(reward.minimize_voltage_deviation(env), -0.2, abs_tol=1e-12):
            self.fail("The local deviation must exclude other buses")
        env.current_bus_voltages_pu["local"] = {"a": 1.01}
        if reward.minimize_voltage_violation(env) != 0:
            self.fail("An in-band single-phase voltage must have zero penalty")

    def test_vuf_uses_positive_and_negative_sequence_phasors(self):
        a = cmath.rect(1, 2 * math.pi / 3)
        reward = Reward("local")
        for negative in (0.0, 0.02):
            phases = {"c": a + negative * a**2, "a": 1 + negative, "b": a**2 + negative * a}
            phases = {key: value * cmath.rect(1, 0.4) for key, value in phases.items()}
            env = SimpleNamespace(current_bus_voltages_pu={"local": {k: abs(v) for k, v in phases.items()}},
                                  current_bus_angles_deg={"local": {k: math.degrees(cmath.phase(v)) for k, v in phases.items()}})
            if not math.isclose(reward.minimize_voltage_unbalance(env), -100 * negative, abs_tol=1e-10):
                self.fail("VUF must be 100 times the negative/positive sequence ratio")
            if not math.isclose(reward.minimize_voltage_unbalance(env, maximum_pct=1.0), -max(100 * negative - 1, 0), abs_tol=1e-10):
                self.fail("The VUF target must define a zero-penalty band")

    def test_single_phase_unbalance_is_explicitly_unsupported(self):
        env = SimpleNamespace(current_bus_voltages_pu={"local": {"a": 1.0}})
        with self.assertRaisesRegex(ValueError, "phases a, b and c"):
            Reward("local").minimize_voltage_unbalance(env)

    def test_live_vuf_matches_opendss_sequence_voltages(self):
        reward = Reward("load")
        checked = []

        def evaluate(env):
            value = -reward.minimize_voltage_unbalance(env)
            env.dss.circuit.set_active_bus("load")
            sequence = env.dss._dss.Bus.SeqVoltages()
            expected = 100 * sequence[2] / sequence[1]
            # OpenDSS angle conversion introduces tiny reconstruction differences.
            if not math.isclose(value, expected, abs_tol=1e-7):
                self.fail("Local VUF must agree with OpenDSS sequence voltages")
            checked.append(value)
            return reward.minimize_cost(env)

        case = ROOT / "tests/fixtures/three_phase_volt_control/config_per_phase.json"
        env = MicrogridEnv(EnvironmentConfig(case, episode_steps=1, reward_function=evaluate))
        env.reset(seed=42)
        action = {"bess": {"b1": {"p_net_pu": {"a": 0.01, "b": 0.02, "c": 0.03}}},
                  "pv": {"pv1": {"generation_pu": {"a": 0.01, "b": 0.0, "c": 0.02}}}}
        _, value, _, _, info = env.step(action)
        env.close()
        if not checked or value != -info["cost"]:
            self.fail("The VUF check must preserve the executed-cost reward")

    def test_power_objectives_use_net_local_import_across_branches_and_phases(self):
        for phases in (("a",), ("a", "b", "c")):
            count = len(phases)
            flows = [{"bus": "local", "phase": phase, "p_flow_pu": -0.08 / count, "q_flow_pu": -0.008 / count}
                     for phase in phases]
            flows += [{"bus": "local", "phase": phase, "p_flow_pu": 0.02 / count, "q_flow_pu": 0.002 / count}
                      for phase in phases]
            flows += [{"bus": "remote", "p_flow_pu": 9.0, "q_flow_pu": 9.0}]
            env = SimpleNamespace(current_bus_voltages_pu={"local": dict.fromkeys(phases, 1.0)},
                                  current_branch_measurements=flows, data={"base_kva": 1000}, dt=0.25)
            reward = Reward("local")
            values = (reward.minimize_import(env), reward.minimize_reactive_power(env),
                      reward.minimize_power_violation(env, 50, 20), reward.track_power(env, 60, 6))
            if any(not math.isclose(actual, expected, abs_tol=1e-10) for actual, expected in zip(values, (-15, -6, -10, 0))):
                self.fail("Import, Q, limits and tracking must use signed local totals")
            for flow in flows:
                flow["p_flow_pu"] *= -1
            if reward.minimize_import(env) != 0 or not math.isclose(reward.minimize_power_violation(env, 50, 20), -40, abs_tol=1e-10):
                self.fail("Export must not be counted as imported energy")


if __name__ == "__main__":
    unittest.main()
