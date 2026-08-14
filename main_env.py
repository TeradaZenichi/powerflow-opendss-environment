from pathlib import Path
from opendss.envs import MicrogridEnv
from opendss.envs.states import (
    get_cost,
    get_previous_pv_kw,
    get_bess_soc,
    get_previous_load_kw,
)
from opendss.envs.rewards import minimize_cost, minimize_voltage_deviation

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent

    case_path = (project_dir/"examples"/"case5")

    episode_steps = 24
    num_episodes = 1
    start_episode = 0

    reward_function = minimize_cost  # choose the reward function

    env = MicrogridEnv(
        case_path=case_path,
        episode_steps=episode_steps,
        num_episodes=num_episodes,
        start_episode=start_episode,
        state_functions=[ # choose what the agent observes
            get_cost,
            get_previous_pv_kw,
            get_bess_soc,
            get_previous_load_kw],
        reward_function=reward_function,
    )

    episode_rewards = []

    for episode_idx in range(env.num_episodes):
        state, info = env.reset(options={"episode_idx": episode_idx})
        terminated = False
        while not terminated:
            action = None # Action is ignored for now because BESS/PV operation is fixed.
            (state, reward, terminated, truncated, info) = env.step(action)

        # Get complete results of this episode
        results = env.get_episode_results()

        episode_reward = results["episode_reward"]
        episode_rewards.append(episode_reward)

        print(f"Episode {episode_idx + 1} total reward \"{reward_function.__name__}\": {episode_reward:.6f}")