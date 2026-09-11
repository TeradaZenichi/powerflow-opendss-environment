def minimize_cost(env):
    return -env.results.costs[env.idx]

def minimize_voltage_deviation(env):
    deviation = 0.0

    for bus in env.results.voltages_pu:
        voltage = env.results.voltages_pu[bus][env.idx]
        deviation += abs(voltage - 1.0)

    return -deviation