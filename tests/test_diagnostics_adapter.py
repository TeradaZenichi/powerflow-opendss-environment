import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from opendss_environment import Action, EnvironmentConfig, GymAdapter, MicrogridEnv, TotalAction


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/three_phase_volt_control"


class DiagnosticsAdapterTest(unittest.TestCase):
    def test_violations_are_reported_without_changing_reward_or_command(self):
        with TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            shutil.copytree(FIXTURE, case)
            path = case / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["voltage_limits"] = {"v_min_pu": 1.2, "v_max_pu": 1.3}
            config["grid"]["p_import_max_kw"] = 1.0
            path.write_text(json.dumps(config), encoding="utf-8")
            master = case / "dss/Master.dss"
            master.write_text(master.read_text(encoding="utf-8").replace("NormAmps=300", "NormAmps=1"), encoding="utf-8")
            env = MicrogridEnv(EnvironmentConfig(case))
            env.reset(seed=42)
            _, reward, _, _, info = env.step(Action(pv={"pv1": TotalAction(0.0)}))
            diagnostics = info["diagnostics"]
            if diagnostics["within_limits"] or not diagnostics["voltage_violations"] or not diagnostics["branch_overloads"]:
                raise ValueError("Executed voltage and current violations must be reported")
            if diagnostics["grid_excess_pu"].get("p_import_max_pu", 0) <= 0:
                raise ValueError("Import limit violation is missing")
            if reward != -info["cost"] or info["executed_action"]["pv"]["pv1"]["generation_total_pu"] != 0:
                raise ValueError("Diagnostics must not change the reward or action")

    def test_transformer_current_limits_follow_winding_voltage(self):
        env = MicrogridEnv(EnvironmentConfig(FIXTURE))
        env.dss.text("New Transformer.audit phases=3 windings=2 buses=[load auditbus] conns=[wye wye] kvs=[0.4 0.23] kvas=[30 30] xhl=1")
        env.dss.circuit.set_active_element("Transformer.audit")
        limits = env.dss.cktelement.normal_current_limits
        if not math.isclose(limits[1] / limits[0], 0.4 / 0.23):
            raise ValueError("Transformer terminal limits must use their winding voltages")

    def test_adapter_passes_gym_checker_and_returns_terminal_state(self):
        for source in (ROOT / "examples/case5", FIXTURE / "config_per_phase.json"):
            with self.subTest(case=str(source)):
                env = GymAdapter(MicrogridEnv(EnvironmentConfig(source, episode_steps=2)), devices={"bess": ["b1"]})
                check_env(env, skip_render_check=True)
                first, _ = env.reset(seed=43)
                command = env.action_space.sample()
                first_next, first_reward, _, _, _ = env.step(command)
                repeated, _ = env.reset(seed=43)
                repeated_command = env.action_space.sample()
                repeated_next, repeated_reward, _, _, _ = env.step(repeated_command)
                if not np.array_equal(first, repeated) or not np.array_equal(command, repeated_command):
                    raise ValueError("Adapter reset must propagate the seed")
                if not np.allclose(first_next, repeated_next) or first_reward != repeated_reward:
                    raise ValueError("Seeded execution must be reproducible")
                final, _, terminated, _, info = env.step(command)
                if not terminated or not env.observation_space.contains(final):
                    raise ValueError("Terminal state must belong to observation_space")
                if info["executed_action"]["bess"]["b1"]["soc_after_frac"] != env.env.bess_list[0].soc:
                    raise ValueError("Terminal SoC must be the executed SoC")

    def test_local_adapter_observation_and_phase_action_range(self):
        paths = [("buses", "load", "v_before_pu", "b"), ("bess", "b1", "soc_before_frac")]
        env = GymAdapter(MicrogridEnv(EnvironmentConfig(FIXTURE / "config_per_phase.json")), paths, {"bess": ["b1"]})
        observation, _ = env.reset(seed=42)
        if observation.shape != (2,) or len(env.action_order) != 3:
            raise ValueError("Local paths and per-phase actions must define the spaces")
        if not np.allclose(env.action_space.high[::2], 0.1):
            raise ValueError("Phase capability must not be restricted to one third of the active total limit")
