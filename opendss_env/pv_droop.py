"""Local inverter curves shared by the environment control path."""
import numpy as np


VV_V = (0.90, 0.92, 0.98, 1.02, 1.08, 1.15)
VV_Q = (1.0, 1.0, 0.0, 0.0, -1.0, -1.0)
VW_V = (0.90, 1.06, 1.10, 1.15)
VW_P = (1.0, 1.0, 0.0, 0.0)
Q_MAX_FRACTION = 0.44


def volt_var_kvar(voltage_pu, s_max_kva):
    fraction = np.interp(float(voltage_pu), VV_V, VV_Q)
    return float(Q_MAX_FRACTION * s_max_kva * fraction)


def volt_watt_limit_kw(voltage_pu, available_kw):
    fraction = np.interp(float(voltage_pu), VW_V, VW_P)
    return float(max(available_kw, 0.0) * fraction)
