from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from opendss_environment import (
    Action, EnvironmentConfig, EVSEAction, GymAdapter, MicrogridEnv,
)
from opendss_environment.evse import EVSE, EVSEConfig, EVSessionConfig


CASE = Path(__file__).resolve().parent / "fixtures" / "evse_case"


class EVSETest(unittest.TestCase):
    def make_env(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 4))
        observation, _ = env.reset()
        return env, observation

    def test_controlled_and_asap_sessions_use_exact_connection_time(self):
        env, observation = self.make_env()
        self.assertFalse(observation["evse"]["c1"]["connected"])
        self.assertFalse(observation["evse"]["a1"]["connected"])

        action = Action(evse={"c1": EVSEAction(0.08)})
        state, _, _, _, info = env.step(action)
        controlled = info["executed_action"]["evse"]["c1"]
        immediate = info["executed_action"]["evse"]["a1"]
        self.assertAlmostEqual(controlled["energy_delivered_kwh"], 4.0 / 3.0)
        self.assertAlmostEqual(controlled["p_net_total_pu"], 0.05333333333333334)
        self.assertAlmostEqual(immediate["energy_delivered_kwh"], 1.0)
        self.assertAlmostEqual(immediate["p_net_total_pu"], 0.04)
        self.assertTrue(state["evse"]["c1"]["connected"])
        self.assertAlmostEqual(state["evse"]["c1"]["remaining_energy_kwh"], 2.0 / 3.0)

        _, _, _, _, info = env.step(action)
        controlled = info["executed_action"]["evse"]["c1"]
        immediate = info["executed_action"]["evse"]["a1"]
        self.assertAlmostEqual(controlled["energy_delivered_kwh"], 2.0 / 3.0)
        self.assertAlmostEqual(immediate["energy_delivered_kwh"], 0.3)
        self.assertEqual(controlled["events"][-1]["event"], "departure")
        self.assertAlmostEqual(controlled["events"][-1]["unmet_kwh"], 0.0)
        self.assertEqual(immediate["events"][-1]["session_id"], 3)

    def test_none_action_stops_controlled_and_runs_asap(self):
        env, _ = self.make_env()
        _, _, _, _, info = env.step(None)
        self.assertEqual(info["executed_action"]["evse"]["c1"]["p_net_total_pu"], 0.0)
        self.assertAlmostEqual(
            info["executed_action"]["evse"]["a1"]["energy_delivered_kwh"], 1.0
        )

    def test_only_controlled_ids_are_accepted(self):
        env, _ = self.make_env()
        with self.assertRaisesRegex(ValueError, "controlled"):
            env.step(Action(evse={"a1": EVSEAction(0.01)}))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            env.step(Action(evse={"c1": EVSEAction(-0.01)}))

    def test_power_flow_failure_rolls_back_sessions(self):
        env, before = self.make_env()
        action = Action(evse={"c1": EVSEAction(0.08)})
        with patch(
            "opendss_environment.microgrid_env.solve_power_flow",
            side_effect=RuntimeError("power flow failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "power flow failed"):
                env.step(action)
        self.assertEqual(env.idx, 0)
        self.assertEqual(env.evse_list[0].array_energy_kwh, [])
        self.assertEqual(env.evse_list[1].array_energy_kwh, [])
        self.assertEqual(env.evse_list[1].sessions[0].remaining_kwh, 1.0)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.observe()
        observation, _ = env.reset()
        self.assertEqual(observation, before)

    def test_gym_adapter_controls_only_the_controlled_evse(self):
        env, _ = self.make_env()
        adapter = GymAdapter(env, devices={"evse": ["c1"]})
        observation, _ = adapter.reset()
        self.assertEqual(observation.shape, adapter.observation_space.shape)
        _, _, _, _, info = adapter.step(np.array([0.08, 0.0]))
        self.assertAlmostEqual(
            info["executed_action"]["evse"]["c1"]["energy_delivered_kwh"],
            4.0 / 3.0,
        )
        self.assertAlmostEqual(
            info["executed_action"]["evse"]["a1"]["energy_delivered_kwh"],
            1.0,
        )

    def test_asap_carry_in_is_reconstructed_but_controlled_is_rejected(self):
        session = EVSessionConfig(
            10, pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 01:00:00", tz="UTC"),
            0.5, 0.9, 10.0, 1.0, 4.0, 4.0,
        )
        start = pd.Timestamp("2026-01-01 00:15:00", tz="UTC")
        immediate = EVSE(EVSEConfig("a", "load", 4.0, "asap"), [session])
        immediate.initialize(start)
        self.assertTrue(immediate.sessions[0].carry_in)
        self.assertAlmostEqual(immediate.sessions[0].remaining_kwh, 3.0)
        self.assertAlmostEqual(immediate.energy_before_episode_kwh, 1.0)

        controlled = EVSE(EVSEConfig("c", "load", 4.0, "controlled"), [session])
        with self.assertRaisesRegex(ValueError, "previous actions"):
            controlled.initialize(start)

    def test_evse_local_observation_keeps_only_its_local_view(self):
        env, global_observation = self.make_env()
        local = env.observe_local(
            "evse", "c1", include=("device", "bus", "branches"),
            observation=global_observation,
        )
        self.assertEqual(local["controller"]["bus"], "load")
        self.assertEqual(local["controller"]["phases"], ["a"])
        self.assertEqual(local["device"], global_observation["evse"]["c1"])
        self.assertNotIn("a1", local)
        self.assertTrue(all(item["bus"] == "load" for item in local["branches"]))
        self.assertTrue(all(item["neighbor_bus"] == "source" for item in local["branches"]))

    def test_controller_observation_uses_an_explicit_component_allowlist(self):
        env, global_observation = self.make_env()
        local = env.observe_controller(
            "load", {"evse": ("c1",)},
            extra_features=("branches",), observation=global_observation,
        )
        self.assertEqual(local["controller"], {
            "bus": "load", "observed_components": {"evse": ["c1"]},
        })
        self.assertEqual(set(local["components"]["evse"]), {"c1"})
        self.assertNotIn("a1", local["components"]["evse"])
        self.assertNotIn("grid", local)
        self.assertTrue(all(item["neighbor_bus"] == "source" for item in local["branches"]))
        self.assertEqual(local["components"]["evse"]["c1"]["bus"], "load")
        self.assertEqual(local["components"]["evse"]["c1"]["phases"], ["a"])

        with self.assertRaisesRegex(ValueError, "at least one"):
            env.observe_controller("load", {}, observation=global_observation)
        with self.assertRaisesRegex(ValueError, "Unknown observed"):
            env.observe_controller(
                "load", {"evse": ("missing",)}, observation=global_observation
            )


if __name__ == "__main__":
    unittest.main()
