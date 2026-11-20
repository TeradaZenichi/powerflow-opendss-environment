from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, TotalAction
from opendss_environment.data import load_data


CASE = Path(__file__).resolve().parents[1] / "examples" / "case5"


class StartStepTest(unittest.TestCase):
    def test_preloaded_data_shares_sources_but_not_offsets_or_physical_state(self):
        data = load_data(CASE)
        with patch("opendss_environment.microgrid_env.load_data", side_effect=AssertionError("Reloaded case")):
            train = MicrogridEnv(EnvironmentConfig(CASE, 2), data=data)
            evaluate = MicrogridEnv(EnvironmentConfig(CASE, 2, start_step=5), data=data)
        self.addCleanup(train.close)
        self.addCleanup(evaluate.close)
        self.assertIs(train.data["demand"], data["demand"])
        self.assertIs(evaluate.data["prices"], data["prices"])
        self.assertIsNot(train.data, evaluate.data)
        self.assertNotIn("start_step", data)
        self.assertEqual((train.data["start_step"], evaluate.data["start_step"]), (0, 5))
        train.reset()
        before = train.observe()
        evaluate.reset()
        evaluate.step()
        self.assertEqual(train.observe(), before)
        self.assertIsNot(train.bess_list[0], evaluate.bess_list[0])
        self.assertIsNot(train.results, evaluate.results)
        self.assertIsNot(train.dss, evaluate.dss)

    def test_reset_selects_absolute_window_without_reloading_and_restores_defaults(self):
        with patch("opendss_environment.microgrid_env.load_data", wraps=load_data) as load:
            env = MicrogridEnv(EnvironmentConfig(CASE, 2, start_step=3, num_episodes=2))
            self.addCleanup(env.close)
            data, dss = env.data, env.dss
            observation, info = env.reset(options={"start_step": 7, "episode_steps": 3})
            self.assertEqual(info["start_step"], 7)
            self.assertEqual(info["episode_steps"], 3)
            self.assertEqual(pd.Timestamp(observation["timestamp"]), data["timestamps"].iloc[7])
            self.assertEqual(observation["grid"]["buy_price_per_kwh"], float(data["prices"][7]))
            original_bess = env.bess_list[0]
            env.step(Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0.0)}))
            observation, info = env.reset(options={"episode_idx": 1})
            self.assertEqual((info["start_step"], env.steps, env.idx), (5, 2, 0))
            self.assertIsNot(env.bess_list[0], original_bess)
            self.assertEqual(env.bess_list[0].soc, 0.5)
            self.assertEqual(env.episode_reward, 0)
            self.assertEqual(pd.Timestamp(observation["timestamp"]), data["timestamps"].iloc[5])
            self.assertIs(env.data, data)
            self.assertIs(env.dss, dss)
            load.assert_called_once()

    def test_explicit_none_uses_remaining_rows_for_this_reset(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 2))
        self.addCleanup(env.close)
        _, info = env.reset(options={"start_step": 3, "episode_steps": None})
        self.assertEqual(info["episode_steps"], env.data["steps"] - 3)
        _, info = env.reset()
        self.assertEqual((info["start_step"], info["episode_steps"]), (0, 2))

    def test_invalid_reset_windows_are_rejected(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 2))
        self.addCleanup(env.close)
        for options in ({"start_step": -1}, {"episode_steps": 0},
                        {"start_step": env.data["steps"]},
                        {"start_step": env.data["steps"] - 1, "episode_steps": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                env.reset(options=options)
        env.reset()
        env.step()

    def test_evaluation_context_does_not_change_training_state(self):
        train = MicrogridEnv(EnvironmentConfig(CASE, 3))
        evaluate = MicrogridEnv(EnvironmentConfig(CASE, 2))
        self.addCleanup(train.close)
        self.addCleanup(evaluate.close)
        train.reset()
        train.step(Action(bess={"b1": TotalAction(0.01)}, pv={"pv1": TotalAction(0.0)}))
        before = train.observe()
        evaluate.reset(options={"start_step": 5})
        evaluate.step()
        self.assertEqual(train.idx, 1)
        self.assertEqual(train.observe(), before)
        train.step()

    def test_ev_session_state_is_rebuilt_for_selected_window(self):
        case = Path(__file__).parent / "fixtures/evse_case"
        reused = MicrogridEnv(EnvironmentConfig(case, 2))
        self.addCleanup(reused.close)
        reused.reset()
        reused.step()
        with self.assertRaisesRegex(ValueError, "controlled session"):
            reused.reset(options={"start_step": 1, "episode_steps": 1})
        reused.data["devices"]["evse"][0]["control_mode"] = "asap"
        actual, _ = reused.reset(options={"start_step": 1, "episode_steps": 1})
        self.assertTrue(actual["evse"]["c1"]["connected"])
        self.assertAlmostEqual(actual["evse"]["c1"]["remaining_energy_kwh"], 0.0)
        original, _ = reused.reset()
        self.assertFalse(original["evse"]["c1"]["connected"])

    def test_offset_precedes_episode_indexing_and_keeps_profiles_aligned(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, episode_steps=2, num_episodes=2, start_step=3))
        for episode in range(2):
            observation, _ = env.reset(options={"episode_idx": episode})
            row = 3 + episode * 2
            if pd.Timestamp(observation["timestamp"]) != env.data["timestamps"].iloc[row]:
                raise ValueError("The episode did not start at the selected dataset row")
            load = sum(observation["buses"]["bus_004"]["p_load_pu"].values()) * env.data["base_kva"]
            if abs(load - env.data["demand"]["Pbus_004"].iloc[row]) > 1e-8:
                raise ValueError("Load profile is not aligned with the offset")
            for device in env.pv_list:
                available = sum(observation["pv"][device.id]["available_pu"].values()) * env.data["base_kva"]
                if abs(available - float(device.profile[0])) > 1e-8:
                    raise ValueError("PV profile is not aligned with the offset")
            if observation["grid"]["buy_price_per_kwh"] != float(env.data["prices"][row]):
                raise ValueError("Price profile is not aligned with the offset")
        env.close()

    def test_complete_episode_uses_only_remaining_rows(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, start_step=3))
        observation, _ = env.reset()
        if env.steps != env.data["steps"] - 3:
            raise ValueError("The full episode includes rows before the offset")
        if pd.Timestamp(observation["timestamp"]) != env.data["timestamps"].iloc[3]:
            raise ValueError("The full episode starts at the wrong timestamp")
        env.close()


if __name__ == "__main__":
    unittest.main()
