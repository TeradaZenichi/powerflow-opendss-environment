from pathlib import Path
from opendss.results import simulation_results

if __name__ == "__main__":

    case = "case5"

    episodes = 1
    episode_steps = 24

    simulation_results(
        case,
        episodes,
        episode_steps,
    )