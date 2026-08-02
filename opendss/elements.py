class Load:
    def __init__(self, id, bus, array_kw, array_kvar):
        self.id = id
        self.bus = bus
        self.array_kw = array_kw
        self.array_kvar = array_kvar

class PV:
    def __init__(self, id, bus, p_max_kw, s_max_kva, q_loss_rated_kw, night_var,
                 profile, control, curtailable, power_factor):
        self.id, self.bus = id, bus
        self.p_max_kw, self.s_max_kva = p_max_kw, s_max_kva
        self.q_loss_rated_kw, self.night_var = q_loss_rated_kw, night_var
        self.profile, self.control = profile, control
        self.curtailable, self.power_factor = curtailable, power_factor
        self.array_kw = None
        self.array_kvar = []

class BESS:
    def __init__(self, id, bus, e_cap_kwh, p_charge_max_kw, p_discharge_max_kw, s_max_kva, reactive_control, q_loss_rated_kw,
                 eta_charge, eta_discharge, soc_init_frac, soc_min_frac, soc_max_frac, cyclic_soc):
        self.id, self.bus, self.e_cap_kwh = id, bus, e_cap_kwh
        self.p_charge_max_kw, self.p_discharge_max_kw = p_charge_max_kw, p_discharge_max_kw
        self.s_max_kva, self.reactive_control, self.q_loss_rated_kw = s_max_kva, reactive_control, q_loss_rated_kw
        self.eta_charge, self.eta_discharge = eta_charge, eta_discharge
        self.soc_init_frac, self.soc_min_frac, self.soc_max_frac = soc_init_frac, soc_min_frac, soc_max_frac
        self.cyclic_soc = cyclic_soc
        self.soc = soc_init_frac 
        self.array_soc = []        
        self.array_kw = []              
        self.array_kvar = []             

    def operate(self, p_bess, dt):
        if p_bess >= 0:  # charging
            p_bess = min(p_bess, self.p_charge_max_kw) # limits to max charging power
            new_soc = self.soc + (self.eta_charge * p_bess) / self.e_cap_kwh * dt
            if new_soc > self.soc_max_frac: 
                # limits soc to max and reduces power accordingly
                p_bess = (self.soc_max_frac - self.soc) * self.e_cap_kwh / (self.eta_charge * dt)
                self.soc = self.soc_max_frac
            else:
                self.soc = new_soc

        else:  # p_bess < 0 -> discharging
            p_bess = max(p_bess, -self.p_discharge_max_kw) # limits to max discharging power
            new_soc = self.soc + (p_bess / self.eta_discharge) / self.e_cap_kwh * dt
            if new_soc < self.soc_min_frac: 
                # limits soc to min and reduce power accordingly
                p_bess = (self.soc_min_frac - self.soc) * self.e_cap_kwh * self.eta_discharge / dt
                self.soc = self.soc_min_frac
            else:
                self.soc = new_soc
            
        self.array_kw.append(p_bess)
        self.array_soc.append(self.soc)

        return p_bess

class Grid:
    def __init__(self, prices):
        self.prices = prices
        self.array_kw = []
        self.array_kvar = []

class Results:
    def __init__(self):
        self.costs = []
        voltages = None
        voltages_pu = None        