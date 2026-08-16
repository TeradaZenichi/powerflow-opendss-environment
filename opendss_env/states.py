import numpy as np


def get_hour(env):
    #print(env.timestamps[env.idx].hour if env.idx < len(env.timestamps) else 0)
    return env.timestamps[env.idx].hour if env.idx < len(env.timestamps) else 0

def get_price(env):
    # print(env.grid.prices[env.idx] if env.idx < len(env.grid.prices) else 0.0)
    return env.grid.prices[env.idx] if env.idx < len(env.grid.prices) else 0.0

def get_bess_soc(env):
    # print(sum(bess.soc for bess in env.bess_list))
    return sum(bess.soc for bess in env.bess_list)

def get_previous_pv_kw(env):
    return sum(pv.array_kw[env.idx - 1] for pv in env.pv_list) if env.idx > 0 else 0.0

def get_previous_load_kw(env):
    return sum(load.array_kw[env.idx - 1] for load in env.load_list) if env.idx > 0 else 0.0

def get_current_pv_kw(env):
    return sum(pv.array_kw[env.idx] for pv in env.pv_list) if env.idx < len(env.pv_list[0].array_kw) else 0.0

def get_current_load_kw(env):
    return sum(load.array_kw[env.idx] for load in env.load_list) if env.idx < len(env.load_list[0].array_kw) else 0.0

def build_state(env, state_functions):
    return np.array([function(env) for function in state_functions], dtype=np.float32)