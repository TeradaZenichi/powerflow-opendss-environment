import numpy as np


def get_cost(env):
    return env.current_cost


def get_previous_pv_kw(env):

    if env.idx == 0:
        return 0.0

    return sum(
        pv.array_kw[env.idx - 1]
        for pv in env.pv_list
    )


def get_bess_soc(env):

    return sum(
        bess.soc
        for bess in env.bess_list
    )


def get_previous_load_kw(env):

    if env.idx == 0:
        return 0.0

    return sum(
        load.array_kw[env.idx - 1]
        for load in env.load_list
    )


def build_state(env, state_functions):

    return np.array(
        [
            function(env)
            for function in state_functions
        ],
        dtype=np.float32,
    )