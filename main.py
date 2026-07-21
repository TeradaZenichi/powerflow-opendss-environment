from opendss.plot_results import plot_power_flow
from opendss.data import load_data

if __name__ == "__main__":
    data = load_data("./examples/case5")
    plot_power_flow(data)