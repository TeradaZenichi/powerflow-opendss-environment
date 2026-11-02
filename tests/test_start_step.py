from pathlib import Path
import unittest

import pandas as pd

from opendss_environment import EnvironmentConfig, MicrogridEnv


CASE = Path(__file__).resolve().parents[1] / "examples" / "case5"


class StartStepTest(unittest.TestCase):
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
