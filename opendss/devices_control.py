import numpy as np

'''
The following functions can be changed by the user to determine BESS and PV operation
'''

BESS_KW = np.array([
    -18.173081, 0.000124, 34.035833, 38.786080, 0.012121, 0.017326,
    0.025920, -5.301635, -39.999932, -39.999864, 0.001183, 0.037441,
    39.999676, 39.999834, 14.843866, 0.042374, 0.059174, -5.183183,
    -39.999976, -39.999921, 0.075356, 0.050025, 2.150388, 39.999954,
])

BESS_KVAR = np.array([
    9.364064, 7.981201, 7.398722, 7.188212, 7.511163, 9.051157,
    11.092626, 15.459909, 18.817502, 16.327392, 13.772692, 13.298422,
    13.050813, 12.612928, 13.368814, 14.178134, 16.764826, 21.337111,
    25.135366, 24.090899, 18.926604, 15.413595, 11.996410, 8.376173,
])

PV_KVAR = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    27.099020, 34.656132, 38.165036, 35.143874, 33.732499, 32.797249,
    32.689956, 32.204547, 32.842744, 34.669396, 40.993161, 47.801269,
    49.927218, 0.0, 0.0, 0.0, 0.0, 0.0,
])


def bess_control(bess, idx, dt):
    # BESS power profiles for 24 hours (will change in the future for bess control)
    # To respect energy and power limits, use with bess.operate
    return bess.operate(BESS_KW[idx], BESS_KVAR[idx], dt)


def pv_control(pv, idx):
    # PV active and reactive power profiles (will change in the future)
    return pv.operate(pv.profile[idx], PV_KVAR[idx], idx)
