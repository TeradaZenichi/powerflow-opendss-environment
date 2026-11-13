"""Report executed electrical violations without changing actions or rewards."""


def electrical_diagnostics(env):
    limits = env.data["voltage_limits"]
    minimum, maximum = limits.get("v_min_pu", 0.95), limits.get("v_max_pu", 1.05)
    voltage = [
        {"bus": bus, "phase": phase, "voltage_pu": value,
         "excess_pu": max(minimum - value, value - maximum, 0.0)}
        for bus, phases in env.current_bus_voltages_pu.items() for phase, value in phases.items()
        if value < minimum - 1e-8 or value > maximum + 1e-8
    ]
    overloaded = [dict(flow) for flow in env.current_branch_measurements
                  if flow["loading_pct"] is not None and flow["loading_pct"] > 100 + 1e-6]
    grid = {}
    for name, value in (("p_import_max_kw", max(-env.current_grid_kw, 0.0)),
                        ("p_export_max_kw", max(env.current_grid_kw, 0.0)),
                        ("q_max_kvar", abs(env.current_grid_kvar))):
        limit = env.data["grid_limits"].get(name)
        if limit is not None and value > limit + 1e-6:
            grid[name.replace("_kw", "_pu").replace("_kvar", "_pu")] = (value - limit) / env.data["base_kva"]
    return {"within_limits": not (voltage or overloaded or grid),
            "voltage_violations": voltage, "branch_overloads": overloaded, "grid_excess_pu": grid,
            "unrated_branch_terminals": sum(flow["normal_current_a"] is None for flow in env.current_branch_measurements)}
