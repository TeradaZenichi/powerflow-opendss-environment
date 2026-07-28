import numpy as np

def bess_control(bess, idx, dt):
    # BESS power profile for 24 hours (will change in the future for bess control)
    # To respect energy and power limits, use with bess.operate
    bess_kw = np.array([ 
            0,  0,  0,  0,  0,  0,
            0, 0, 0, 25, 50, 50, 
            50, 50, 25, 0, 0, -50,
            -200, -100,  -100,  0,  0,  0
        ])
    bess_kvar = -abs(0.05*bess_kw)

    p_bess = bess.operate(bess_kw[idx], dt)
    q_bess = bess_kvar[idx]

    return p_bess, q_bess

def pv_control(pv, idx):
    # PV active power used and reactive power profile (will change in the future)
    
    # Fow now, active power used from PV is equal to the profile
    pv.array_kw = pv.profile

    # Fow, reactive power from PV is 0
    pv.array_kvar = np.zeros(len(pv.array_kw))

    p_pv = pv.array_kw[idx]
    q_pv = pv.array_kvar[idx]

    return p_pv, q_pv
