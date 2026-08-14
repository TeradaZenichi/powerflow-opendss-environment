def minimize_cost(env):
    return -env.current_cost

def minimize_voltage_deviation(env):

    deviation = sum(
        abs(voltage - 1.0)
        for voltage in env.current_voltages_pu
    )

    return -deviation