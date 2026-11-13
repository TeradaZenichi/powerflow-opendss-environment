"""Run a configured case and export the last episode."""
from pathlib import Path

from opendss_environment import EnvironmentConfig, MicrogridEnv
from opendss_environment.results_scripts.plots import save_plots
from opendss_environment.results_scripts.summary import save_summary


CASE_PATH = Path(__file__).resolve().parents[2] / "examples" / "case123"
EPISODE_STEPS = 24
NUM_EPISODES = 1
START_EPISODE = 0
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / CASE_PATH.name


def main():
    env = MicrogridEnv(EnvironmentConfig(
        CASE_PATH, episode_steps=EPISODE_STEPS,
        num_episodes=NUM_EPISODES, start_episode=START_EPISODE,
    ))
    episode_rewards = []
    for episode_idx in range(NUM_EPISODES):
        env.reset(options={"episode_idx": episode_idx})
        for _ in range(env.steps):
            _, _, terminated, truncated, _ = env.step(None)
            if terminated or truncated:
                break
        results = env.get_episode_results()
        episode_rewards.append(results["episode_reward"])
        print(f"Episode {episode_idx + 1} reward: {results['episode_reward']}")
    save_plots(results, OUTPUT / "plots")
    save_summary(results, OUTPUT / "summary")
    print(f"Results: {OUTPUT.resolve()}")
    return episode_rewards


if __name__ == "__main__":
    main()
