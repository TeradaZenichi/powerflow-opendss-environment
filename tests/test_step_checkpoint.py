from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from opendss_environment import Action, EnvironmentConfig, EVSEAction, MicrogridEnv
from opendss_environment.evse import utc_timestamp
from opendss_environment.simulation import prepare_pre_action_observation
from opendss_environment.step_state import StepCheckpoint


ROOT = Path(__file__).resolve().parents[1]
CASES = (ROOT / "examples/case5",
         ROOT / "tests/fixtures/three_phase_volt_control/config_per_phase.json",
         ROOT / "tests/fixtures/evse_case")


def snapshot(env):
    devices = [*env.bess_list, *env.pv_list, *env.evse_list, env.grid, env.results]
    sessions = [session for device in env.evse_list for session in device.sessions]
    return deepcopy({
        "objects": [{key: value for key, value in vars(obj).items() if key != "sessions"}
                    for obj in (*devices, *sessions)],
        "environment": {name: getattr(env, name) for name in (
            "idx", "current_cost", "episode_reward", "current_bus_voltages_pu",
            "current_device_measurements", "previous_device_measurements",
            "current_branch_measurements", "pre_action_branch_measurements",
        )},
    })


class StepCheckpointTest(unittest.TestCase):
    def test_failure_restores_devices_sessions_histories_and_voltage_arrays(self):
        for case in CASES:
            for completed in (0, 1):
                for stage in ("reward", "next_observation"):
                    steps = 2 if case == CASES[1] else 3
                    if stage == "next_observation" and completed + 1 == steps:
                        continue
                    with self.subTest(case=case, completed=completed, stage=stage):
                        env = MicrogridEnv(EnvironmentConfig(case, steps))
                        self.addCleanup(env.close)
                        env.reset(seed=42)
                        action = Action(evse={"c1": EVSEAction(-0.08, 0.06)}) if env.evse_list else None
                        for _ in range(completed):
                            env.step(action)
                        before = snapshot(env)
                        sessions = [s for device in env.evse_list for s in device.sessions]

                        def fail_next(pending):
                            prepare_pre_action_observation(pending)
                            raise RuntimeError("failed after next observation")

                        failure = patch.object(env, "reward_function", Mock(side_effect=RuntimeError("failed reward")))
                        if stage == "next_observation":
                            failure = patch("opendss_environment.microgrid_env.prepare_pre_action_observation", fail_next)
                        with failure, self.assertRaisesRegex(RuntimeError, "failed"):
                            env.step(action)
                        np.testing.assert_equal(snapshot(env), before)
                        for actual, original in zip(
                            (s for device in env.evse_list for s in device.sessions), sessions,
                        ):
                            self.assertIs(actual, original)
                        self.assertTrue(env._needs_reset)

    def test_only_overlapping_ev_sessions_are_checkpointed(self):
        env = MicrogridEnv(EnvironmentConfig(CASES[-1], 4))
        self.addCleanup(env.close)
        env.reset()
        env.step(None)
        checkpoint = StepCheckpoint(env)
        saved = {id(obj) for obj, _ in checkpoint.objects}
        start = utc_timestamp(env.timestamps.iloc[env.idx])
        end = start + timedelta(hours=env.dt)
        for device in env.evse_list:
            for session in device.sessions:
                overlaps = session.config.arrival < end and session.config.departure > start
                self.assertEqual(id(session) in saved, overlaps)
        self.assertTrue(all(history is not values for history, _, _ in checkpoint.histories
                            for values in (device.sessions for device in env.evse_list)))

    def test_interruption_also_restores_state(self):
        env = MicrogridEnv(EnvironmentConfig(CASES[0], 2))
        self.addCleanup(env.close)
        env.reset()
        before = snapshot(env)
        env.reward_function = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            env.step(None)
        np.testing.assert_equal(snapshot(env), before)
        self.assertTrue(env._needs_reset)


if __name__ == "__main__":
    unittest.main()
