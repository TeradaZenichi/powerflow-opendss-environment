import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from opendss_environment import EnvironmentConfig, MicrogridEnv


class CostRewardTest(unittest.TestCase):
    def run_case(self, ratio):
        source = Path(__file__).resolve().parents[1] / "examples" / "case5"
        with TemporaryDirectory() as folder:
            case = Path(folder) / "case"
            shutil.copytree(source, case)
            demand = pd.read_csv(case / "demand.csv")
            demand[demand.columns[1:]] *= 0.1
            demand.to_csv(case / "demand.csv", index=False)
            config_path = case / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["grid"]["feed_in_tariff_ratio"] = ratio
            config_path.write_text(json.dumps(config), encoding="utf-8")
            env = MicrogridEnv(EnvironmentConfig(case))
            observation, _ = env.reset()
            costs = []
            for step in range(env.steps):
                action = {
                    "bess": {did: {"p_net_pu": 0.0, "q_injection_pu": 0.0} for did in observation["bess"]},
                    "pv": {did: {"generation_pu": sum(values["available_pu"].values()), "q_injection_pu": 0.0}
                           for did, values in observation["pv"].items()},
                }
                observation, reward, terminated, truncated, info = env.step(action)
                if reward != -info["cost"]:
                    raise ValueError("Reward does not match executed cost")
                costs.append(info["cost"])
                if terminated or truncated:
                    break
            env.close()
            if not math.isclose(env.episode_reward, -sum(costs), abs_tol=1e-9):
                raise ValueError("Cumulative reward does not match episode cost")
            return costs

    def test_export_tariff_changes_export_revenue_only(self):
        reference = self.run_case(1.0)
        discounted = self.run_case(0.2)
        if not any(cost < 0 for cost in reference) or not any(cost > 0 for cost in reference):
            raise ValueError("The fixture must exercise both import and export")
        for full, partial in zip(reference, discounted):
            expected = full if full >= 0 else 0.2 * full
            if not math.isclose(partial, expected, abs_tol=1e-9):
                raise ValueError("The export tariff changed the wrong energy cost")


if __name__ == "__main__":
    unittest.main()
