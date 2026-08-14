import math


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
        self.array_kw = []
        self.array_kvar = []
        self.array_p_net_kw = []
        self.array_grid_consumption_kw = []
        self.array_inverter_loss_kw = []

    def operate(self, p_pv, q_pv):
        available_kw = max(0.0, min(p_pv, self.p_max_kw))

        if available_kw <= 0.0 and not self.night_var:
            q_pv = 0.0
        q_pv = max(-self.s_max_kva, min(q_pv, self.s_max_kva))

        inverter_loss_kw = self._reactive_loss(q_pv)
        if available_kw > 0.0 and inverter_loss_kw > available_kw:
            q_pv = self._q_for_loss(available_kw, q_pv)
            inverter_loss_kw = self._reactive_loss(q_pv)

        p_capability_kw = math.sqrt(max(self.s_max_kva ** 2 - q_pv ** 2, 0.0))
        generation_limit_kw = min(
            self.p_max_kw,
            p_capability_kw,
            max(available_kw - inverter_loss_kw, 0.0),
        )
        if self.curtailable:
            generation_kw = max(0.0, min(p_pv, generation_limit_kw))
        else:
            generation_kw = generation_limit_kw

        grid_consumption_kw = (
            inverter_loss_kw if available_kw <= 0.0 and self.night_var else 0.0
        )
        p_net_kw = generation_kw - grid_consumption_kw

        self.array_kw.append(generation_kw)
        self.array_kvar.append(q_pv)
        self.array_p_net_kw.append(p_net_kw)
        self.array_grid_consumption_kw.append(grid_consumption_kw)
        self.array_inverter_loss_kw.append(inverter_loss_kw)

        return p_net_kw, q_pv

    def _reactive_loss(self, q_pv):
        if self.q_loss_rated_kw <= 0.0:
            return 0.0
        return self.q_loss_rated_kw * (q_pv / self.s_max_kva) ** 2

    def _q_for_loss(self, loss_kw, requested_q):
        if self.q_loss_rated_kw <= 0.0:
            return requested_q
        q_limit = self.s_max_kva * math.sqrt(
            max(loss_kw, 0.0) / self.q_loss_rated_kw
        )
        return math.copysign(min(abs(requested_q), q_limit), requested_q)

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
        self.array_inverter_loss_kw = []

    def operate(self, p_bess, q_bess, dt):
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if not self.reactive_control:
            q_bess = 0.0
        q_bess = max(-self.s_max_kva, min(q_bess, self.s_max_kva))

        p_capability_kw = math.sqrt(max(self.s_max_kva ** 2 - q_bess ** 2, 0.0))
        p_bess = max(
            -min(self.p_discharge_max_kw, p_capability_kw),
            min(p_bess, min(self.p_charge_max_kw, p_capability_kw)),
        )

        inverter_loss_kw = self._reactive_loss(q_bess)
        available_energy_rate_kw = (
            (self.soc - self.soc_min_frac) * self.e_cap_kwh / dt
        )

        if p_bess < 0.0:
            if inverter_loss_kw > available_energy_rate_kw:
                q_bess = self._q_for_loss(available_energy_rate_kw, q_bess)
                inverter_loss_kw = self._reactive_loss(q_bess)
            max_discharge_kw = max(
                available_energy_rate_kw - inverter_loss_kw,
                0.0,
            ) * self.eta_discharge
            p_bess = max(p_bess, -max_discharge_kw)
        else:
            available_loss_kw = available_energy_rate_kw + self.eta_charge * p_bess
            if inverter_loss_kw > available_loss_kw:
                q_bess = self._q_for_loss(available_loss_kw, q_bess)
                inverter_loss_kw = self._reactive_loss(q_bess)
            max_charge_kw = (
                (self.soc_max_frac - self.soc) * self.e_cap_kwh / dt
                + inverter_loss_kw
            ) / self.eta_charge
            p_bess = min(p_bess, max(max_charge_kw, 0.0))

        if p_bess >= 0:  # charging
            energy_change_kw = self.eta_charge * p_bess - inverter_loss_kw

        else:  # p_bess < 0 -> discharging
            energy_change_kw = p_bess / self.eta_discharge - inverter_loss_kw

        self.soc += energy_change_kw / self.e_cap_kwh * dt
        self.soc = min(self.soc_max_frac, max(self.soc_min_frac, self.soc))

        self.array_kw.append(p_bess)
        self.array_kvar.append(q_bess)
        self.array_soc.append(self.soc)
        self.array_inverter_loss_kw.append(inverter_loss_kw)

        return p_bess, q_bess

    def _reactive_loss(self, q_bess):
        if self.q_loss_rated_kw <= 0.0:
            return 0.0
        return self.q_loss_rated_kw * (q_bess / self.s_max_kva) ** 2

    def _q_for_loss(self, loss_kw, requested_q):
        if self.q_loss_rated_kw <= 0.0:
            return requested_q
        q_limit = self.s_max_kva * math.sqrt(
            max(loss_kw, 0.0) / self.q_loss_rated_kw
        )
        return math.copysign(min(abs(requested_q), q_limit), requested_q)

class Grid:
    def __init__(self, prices):
        self.prices = prices
        self.array_kw = []
        self.array_kvar = []

class Results:
    def __init__(self):
        self.costs = []
        self.voltages = None
        self.voltages_pu = None
