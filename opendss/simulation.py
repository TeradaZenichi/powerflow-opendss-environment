import py_dss_interface
import numpy as np

def run_simulation(data):
    # =============================================================================
    # OpenDSS simulation setup
    # =============================================================================

    hours = np.arange(1, data["steps"] + 1)

    dss = py_dss_interface.DSS()
    dss.text("Clear")

    # Topology
    dss.text(f'compile "{data["topology"]}"')

    # PV generators
    for pv in data["pv_list"]:
        dss.text(f"""
        New Generator.{pv.id} bus1={pv.bus} phases={data["phases"]} kv={data["base_kv"]} kw=0 kvar=0
        """)

    # BESS
    for bess in data["bess_list"]:
        dss.text(f"""
        New Load.{bess.id} bus={bess.bus} phases={data["phases"]} kv={data["base_kv"]} kw=0 kvar=0 conn=y
        """)

    # Load demand
    for load in data["load_list"]:
        dss.text(f"""
        New Load.{load.id} bus1={load.bus} phases={data["phases"]} kv={data["base_kv"]} kw=0 kvar=0
        """)

    # =============================================================================
    # 24-hour profiles (kW / kvar)
    # =============================================================================

    bess_kw = np.array([
        0,  0,  0,  0,  0,  0,
        8, 24, 48, 64, 76, 80,
        76, 68, 56, 40, 16,  0,
        0,  0,  0,  0,  0,  0
    ])

    # =============================================================================
    # Time-series simulation
    # =============================================================================
    grid_power = []

    idx = 0
    while idx < data["steps"]:
        
        for load in data["load_list"]:
            dss.text(
                f"Edit Load.{load.id} kw={load.array_kw[idx]:.2f} kvar={load.array_kvar[idx]:.2f}"
            )

        for pv in data["pv_list"]:
            dss.text(
                f"Edit Generator.{pv.id} kw={pv.profile[idx]:.2f} kvar=0"
            )

        for bess in data["bess_list"]:
            dss.text(
                f"Edit Load.{bess.id} kw={bess_kw[idx]:.2f} "
                f"kvar=0"
            )

        dss.solution.solve()

        grid_power.append(dss.circuit.total_power[0])

        idx += 1

    pv_kw = np.sum(
        [pv.profile for pv in data["pv_list"]],
        axis=0
    )
    
    load_kw = np.sum(
        [load.array_kw for load in data["load_list"]],
        axis=0
    )

    load_kvar = np.sum(
        [load.array_kvar for load in data["load_list"]],
        axis=0
    )

    return {
        "hours": hours,
        "load_kw": load_kw,
        "load_kvar": load_kvar,
        "bess_kw": bess_kw,
        "pv_kw": pv_kw,
        "grid_power": np.array(grid_power),
    }
