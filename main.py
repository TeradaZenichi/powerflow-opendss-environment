from pathlib import Path
from opendss.plot_results import save_plots
from opendss.summary import save_summary
from opendss.data import load_data
from opendss.simulation import run_simulation

if __name__ == "__main__":

    # Select the case to run the simulation
    case = "case5"

    project_dir = Path(__file__).resolve().parent
    case_path = project_dir / "examples" / case

    data = load_data(case_path)
    results = run_simulation(data)

    output_dir = project_dir / "outputs" / case_path.name

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    save_plots(results, plots_dir)
    save_summary(results, summary_dir)

    print(f"Simulation completed for case '{case}'. Results and plots saved in '{output_dir}'.")