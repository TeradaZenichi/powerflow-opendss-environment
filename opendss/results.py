'''
Given a case name, this function runs the simulation and saves 
the plots and summary in the appropriate output directories.
'''

from .plots import save_plots
from .summary import save_summary
from pathlib import Path
from .data import load_data
from .simulation import run_simulation

def simulation_results(case):

    project_dir = Path(__file__).resolve().parent.parent
    case_path = project_dir / "examples" / case

    results = run_simulation(load_data(case_path))

    output_dir = project_dir / "outputs" / case_path.name

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    save_plots(results, plots_dir)
    save_summary(results, summary_dir)

    print(f"Simulation completed for case '{case}'. \n Results and plots saved in '{output_dir}'.")