from pathlib import Path
from opendss_env import MicrogridEnv
from opendss_env.states import get_hour,get_price,get_previous_pv_kw,get_bess_soc,get_previous_load_kw
from opendss_env.rewards import minimize_cost, minimize_voltage_deviation
from opendss_env.results_scripts.results import simulation_results

if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parent
    CASE_PATH = project_dir / "examples" / "case5"

    EPISODE_STEPS = 24
    NUM_EPISODES = 1
    START_EPISODE = 0

    env = MicrogridEnv(
        case_path=CASE_PATH,
        episode_steps=EPISODE_STEPS,
        num_episodes=NUM_EPISODES,
        start_episode=START_EPISODE,
        state_functions=[ # What the agent observes
            get_hour,
            get_price,
            get_previous_pv_kw,
            get_bess_soc,
            get_previous_load_kw,
        ],
        reward_function=minimize_cost,
    )

    episode_rewards = []

    for episode_idx in range(env.num_episodes):

        state, info = env.reset(options={"episode_idx": episode_idx})

        terminated = False
        while not terminated:
            # Fixed control for now
            action = None
            state, reward, terminated, truncated, info = env.step(action)

        results = env.get_episode_results()

        episode_reward = results["episode_reward"]
        episode_rewards.append(episode_reward)

        print(f"Episode {episode_idx + 1} total reward ({env.reward_function.__name__}): {episode_reward}")

        if episode_idx == env.num_episodes - 1:
            simulation_results(results, CASE_PATH)