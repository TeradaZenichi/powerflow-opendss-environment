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
        New Generator.{pv.id} bus1={pv.bus} phases=1 kv={data["base_kv"]} kw=0 kvar=0
        """)

    # BESS
    for bess in data["bess_list"]:
        dss.text(f"""
        New Load.{bess.id} bus={bess.bus} phases=1 kv={data["base_kv"]} kw=0 kvar=0 conn=y
        """)

    # Load demand
    dss.text(f"""
    New Load.Load1 bus1=bus_005 phases=1 kv={data["base_kv"]} kw=0 kvar=0
    """)

    # =============================================================================
    # 24-hour profiles (kW / kvar)
    # =============================================================================

    load_kw = 2.5*np.array([
        60, 55, 50, 50, 55, 65,
        80, 90,100,105,110,115,
        120,115,110,105,100,110,
        120,115,100, 90, 80, 70
    ])

    load_kvar = np.array([
        12, 11, 10, 10, 11, 13,
        16, 18, 20, 21, 22, 23,
        24, 23, 22, 21, 20, 22,
        24, 23, 20, 18, 16, 14
    ])

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

        dss.text(
            f"Edit Load.Load1 kw={load_kw[idx]:.2f} "
            f"kvar={load_kvar[idx]:.2f}"
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

    return {
        "hours": hours,
        "load_kw": load_kw,
        "load_kvar": load_kvar,
        "bess_kw": bess_kw,
        "pv_kw": pv_kw,
        "grid_power": np.array(grid_power),
    }
