'''
Given a case name, this function runs the simulation and saves 
the plots and summary in the appropriate output directories.
'''

import json
from pathlib import Path

from .plots import save_plots
from .summary import save_summary
from .data import load_data, create_episode
from .simulation import run_simulation
from .devices_control import split_control_profiles


def simulation_results(case, episodes, episode_steps):

    project_dir = Path(__file__).resolve().parent.parent

    case_path = project_dir / "examples" / case

    data = load_data(case_path)

    total_steps = episodes * episode_steps

    if total_steps > data["steps"]:
        raise ValueError(
            f"Requested {total_steps} steps "
            f"({episodes} episodes x {episode_steps} steps), "
            f"but the data only contains {data['steps']} steps."
        )

    (
        bess_kw_episodes,
        bess_kvar_episodes,
        pv_kvar_episodes,
    ) = split_control_profiles(
        episodes,
        episode_steps,
    )

    output_dir = (
        project_dir
        / "outputs"
        / case_path.name
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_episode_results = {}

    for episode_idx in range(episodes):

        start = episode_idx * episode_steps
        end = start + episode_steps

        print(
            f"Running episode "
            f"{episode_idx + 1}/{episodes}..."
        )

        episode_data = create_episode(
            data,
            start,
            end,
        )

        result = run_simulation(
            episode_data,
            bess_kw_episodes[episode_idx],
            bess_kvar_episodes[episode_idx],
            pv_kvar_episodes[episode_idx],
        )

        all_episode_results[
            f"episode_{episode_idx + 1}"
        ] = _extract_episode_results(result)

        # Only save plots and summary for the last episode
        if episode_idx == episodes - 1:

            save_plots(
                result,
                plots_dir,
            )

            save_summary(
                result,
                summary_dir,
            )

    _save_episode_results(
        all_episode_results,
        episodes_dir / "episodes.json",
    )

    print(
        f"Simulation completed for case '{case}'.\n"
        f"Results and plots saved in '{output_dir}'."
    )


def _extract_episode_results(result):

    episode = {
        "cost_array": result["results"].costs,
        "total_cost": sum(result["results"].costs),
        "bess": {},
        "pv": {},
    }

    for bess in result["bess_list"]:

        episode["bess"][bess.id] = {
            "kw_array": bess.array_kw,
            "kvar_array": bess.array_kvar,
            "soc_array": bess.array_soc,
        }

    for pv in result["pv_list"]:

        episode["pv"][pv.id] = {
            "kw_array": pv.array_kw,
            "kvar_array": pv.array_kvar,
        }

    return episode


def _save_episode_results(
    episode_results,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            episode_results,
            f,
            indent=4,
        )