from pathlib import Path
from opendss.results import simulation_results

if __name__ == "__main__":

    # Select the case to run the simulation
    case = "case5"

    # Run the simulation and save the results
    simulation_results(case)