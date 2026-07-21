class Load:
    def __init__(self, id, bus, array_kw, array_kvar):
        self.id = id
        self.bus = bus
        self.array_kw = array_kw
        self.array_kvar = array_kvar

class PV:
    def __init__(self, id, bus, p_max_kw, s_max_kva,
                 profile, control, curtailable, power_factor):
        self.id, self.bus = id, bus
        self.p_max_kw, self.s_max_kva = p_max_kw, s_max_kva
        self.profile, self.control = profile, control
        self.curtailable, self.power_factor = curtailable, power_factor

class BESS:
    def __init__(self, id, bus, e_cap_kwh, p_charge_max_kw, p_discharge_max_kw,
                 eta_charge, eta_discharge, soc_init_frac, soc_min_frac, soc_max_frac, cyclic_soc):
        self.id, self.bus, self.e_cap_kwh = id, bus, e_cap_kwh
        self.p_charge_max_kw, self.p_discharge_max_kw = p_charge_max_kw, p_discharge_max_kw
        self.eta_charge, self.eta_discharge = eta_charge, eta_discharge
        self.soc_init_frac, self.soc_min_frac, self.soc_max_frac = soc_init_frac, soc_min_frac, soc_max_frac
        self.cyclic_soc = cyclic_soc