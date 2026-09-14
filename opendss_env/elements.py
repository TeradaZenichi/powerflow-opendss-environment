import math
from dataclasses import dataclass

from .pv_droop import volt_var_kvar, volt_watt_limit_kw


@dataclass(frozen=True)
class PVRequest:
    generation_kw: float
    q_injection_kvar: float = 0.0
    available_kw: float | None = None
    voltage_pu: float | None = None
    capacity_fraction: float = 1.0


@dataclass(frozen=True)
class PVPoint:
    generation_kw: float
    q_injection_kvar: float
    p_net_kw: float
    grid_consumption_kw: float
    inverter_loss_kw: float

    @classmethod
    def total(cls, points):
        points = tuple(points)
        totals = {field: sum(getattr(point, field) for point in points) for field in cls.__dataclass_fields__}
        return cls(**totals)


@dataclass(frozen=True)
class PVConfig:
    id: str
    bus: str
    p_max_kw: float
    s_max_kva: float
    q_loss_rated_kw: float
    night_var: bool
    profile: object
    control: str
    curtailable: bool
    power_factor: float
    phases: tuple = ()
    connection: str = "wye"
    dispatch_mode: str = "aggregate"
    phase_profiles: dict | None = None


@dataclass(frozen=True)
class BESSConfig:
    id: str
    bus: str
    e_cap_kwh: float
    p_charge_max_kw: float
    p_discharge_max_kw: float
    s_max_kva: float
    reactive_control: bool
    q_loss_rated_kw: float
    eta_charge: float
    eta_discharge: float
    soc_init_frac: float
    soc_min_frac: float
    soc_max_frac: float
    cyclic_soc: bool
    phases: tuple = ()
    connection: str = "wye"
    dispatch_mode: str = "aggregate"
    soc_terminal_frac: float | None = None


class Load:
    def __init__(self, id, bus, array_kw, array_kvar, phase_node=None):
        self.id = id
        self.bus = bus
        self.array_kw = array_kw
        self.array_kvar = array_kvar
        self.phase_node = phase_node


class PV:
    def __init__(self, config):
        self.id, self.bus = config.id, config.bus
        self.p_max_kw, self.s_max_kva = config.p_max_kw, config.s_max_kva
        self.q_loss_rated_kw, self.night_var = config.q_loss_rated_kw, config.night_var
        self.profile, self.control = config.profile, config.control
        self.curtailable, self.power_factor = config.curtailable, config.power_factor
        self.phases = tuple(config.phases)
        self.connection, self.dispatch_mode = config.connection, config.dispatch_mode
        self.phase_profiles = config.phase_profiles or {}
        self.array_kw = []
        self.array_kvar = []
        self.array_p_net_kw = []
        self.array_grid_consumption_kw = []
        self.array_inverter_loss_kw = []

    def operate(self, p_pv, q_pv):
        return self.apply(PVRequest(p_pv, q_pv))

    def apply(self, request):
        point = self.operating_point(request)
        self.array_kw.append(point.generation_kw)
        self.array_kvar.append(point.q_injection_kvar)
        self.array_p_net_kw.append(point.p_net_kw)
        self.array_grid_consumption_kw.append(point.grid_consumption_kw)
        self.array_inverter_loss_kw.append(point.inverter_loss_kw)
        return point.p_net_kw, point.q_injection_kvar

    def operating_point(self, request):
        p_pv = request.generation_kw
        q_pv = request.q_injection_kvar
        available_kw = request.available_kw
        voltage_pu = request.voltage_pu
        capacity_fraction = request.capacity_fraction
        requested_kw = p_pv
        available_kw = p_pv if available_kw is None else available_kw
        p_max_kw = self.p_max_kw * capacity_fraction
        s_max_kva = self.s_max_kva * capacity_fraction
        q_loss_rated_kw = self.q_loss_rated_kw * capacity_fraction
        available_kw = max(0.0, min(available_kw, p_max_kw))

        if voltage_pu is not None:
            if "watt" in self.control:
                requested_kw = min(requested_kw, volt_watt_limit_kw(voltage_pu, available_kw))
            if "var" in self.control:
                q_pv = volt_var_kvar(voltage_pu, s_max_kva)
            elif self.control == "volt-watt":
                q_pv = 0.0
        if self.control == "fixed_pf":
            q_pv = math.tan(math.acos(self.power_factor)) * requested_kw
        if available_kw <= 0.0 and not self.night_var:
            q_pv = 0.0
        q_pv = max(-s_max_kva, min(q_pv, s_max_kva))

        inverter_loss_kw = q_loss_rated_kw * (q_pv / s_max_kva) ** 2 if q_loss_rated_kw > 0.0 else 0.0
        if available_kw > 0.0 and inverter_loss_kw > available_kw:
            q_limit = s_max_kva * math.sqrt(available_kw / q_loss_rated_kw)
            q_pv = math.copysign(min(abs(q_pv), q_limit), q_pv)
            inverter_loss_kw = q_loss_rated_kw * (q_pv / s_max_kva) ** 2

        p_capability_kw = math.sqrt(max(s_max_kva ** 2 - q_pv ** 2, 0.0))
        generation_limit_kw = min(p_max_kw, p_capability_kw, max(available_kw - inverter_loss_kw, 0.0))
        if self.curtailable:
            generation_kw = max(0.0, min(requested_kw, generation_limit_kw))
        else:
            generation_kw = generation_limit_kw

        grid_consumption_kw = inverter_loss_kw if available_kw <= 0.0 and self.night_var else 0.0
        p_net_kw = generation_kw - grid_consumption_kw

        return PVPoint(generation_kw, q_pv, p_net_kw, grid_consumption_kw, inverter_loss_kw)


class BESS:
    def __init__(self, config):
        self.id, self.bus, self.e_cap_kwh = config.id, config.bus, config.e_cap_kwh
        self.p_charge_max_kw, self.p_discharge_max_kw = config.p_charge_max_kw, config.p_discharge_max_kw
        self.s_max_kva, self.reactive_control = config.s_max_kva, config.reactive_control
        self.q_loss_rated_kw = config.q_loss_rated_kw
        self.eta_charge, self.eta_discharge = config.eta_charge, config.eta_discharge
        self.soc_init_frac, self.soc_min_frac = config.soc_init_frac, config.soc_min_frac
        self.soc_max_frac, self.cyclic_soc = config.soc_max_frac, config.cyclic_soc
        self.phases = tuple(config.phases)
        self.connection, self.dispatch_mode = config.connection, config.dispatch_mode
        self.soc_terminal_frac = config.soc_terminal_frac
        self.soc = config.soc_init_frac
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
        p_bess = max(-min(self.p_discharge_max_kw, p_capability_kw), min(p_bess, min(self.p_charge_max_kw, p_capability_kw)))

        inverter_loss_kw = self._reactive_loss(q_bess)
        available_energy_rate_kw = (self.soc - self.soc_min_frac) * self.e_cap_kwh / dt

        if p_bess < 0.0:
            if inverter_loss_kw > available_energy_rate_kw:
                q_bess = self._q_for_loss(available_energy_rate_kw, q_bess)
                inverter_loss_kw = self._reactive_loss(q_bess)
            max_discharge_kw = max(available_energy_rate_kw - inverter_loss_kw, 0.0) * self.eta_discharge
            p_bess = max(p_bess, -max_discharge_kw)
        else:
            available_loss_kw = available_energy_rate_kw + self.eta_charge * p_bess
            if inverter_loss_kw > available_loss_kw:
                q_bess = self._q_for_loss(available_loss_kw, q_bess)
                inverter_loss_kw = self._reactive_loss(q_bess)
            max_charge_kw = ((self.soc_max_frac - self.soc) * self.e_cap_kwh / dt + inverter_loss_kw) / self.eta_charge
            p_bess = min(p_bess, max(max_charge_kw, 0.0))

        if p_bess >= 0:
            energy_change_kw = self.eta_charge * p_bess - inverter_loss_kw
        else:
            energy_change_kw = p_bess / self.eta_discharge - inverter_loss_kw

        self.soc += energy_change_kw / self.e_cap_kwh * dt
        self.soc = min(self.soc_max_frac, max(self.soc_min_frac, self.soc))

        self.array_kw.append(p_bess)
        self.array_kvar.append(q_bess)
        self.array_soc.append(self.soc)
        self.array_inverter_loss_kw.append(inverter_loss_kw)

        return p_bess, q_bess

    def operate_phases(self, p_by_phase, q_by_phase, dt):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        phases = tuple(p_by_phase)
        if set(q_by_phase) != set(phases) or not phases:
            raise ValueError("BESS phase commands must use the same nonempty phases")
        count = len(phases)
        phase_rating = self.s_max_kva / count
        p = {phase: float(p_by_phase[phase]) for phase in phases}
        q = {phase: float(q_by_phase[phase]) if self.reactive_control else 0.0 for phase in phases}
        for phase in phases:
            q[phase] = max(-phase_rating, min(q[phase], phase_rating))
            p[phase] = max(-self.p_discharge_max_kw, min(p[phase], self.p_charge_max_kw))
            capability = math.sqrt(max(phase_rating ** 2 - q[phase] ** 2, 0.0))
            p[phase] = max(-capability, min(p[phase], capability))

        charge = sum(max(value, 0.0) for value in p.values())
        discharge = sum(max(-value, 0.0) for value in p.values())
        if charge > self.p_charge_max_kw:
            scale = self.p_charge_max_kw / charge
            p = {phase: value * scale if value > 0.0 else value for phase, value in p.items()}
            charge = self.p_charge_max_kw
        if discharge > self.p_discharge_max_kw:
            scale = self.p_discharge_max_kw / discharge
            p = {phase: value * scale if value < 0.0 else value for phase, value in p.items()}
            discharge = self.p_discharge_max_kw

        inverter_loss_kw = sum(
            self.q_loss_rated_kw / count
            * (q[phase] / phase_rating) ** 2
            for phase in phases
        ) if self.q_loss_rated_kw > 0.0 else 0.0
        available_discharge_kw = max(
            (
                (self.soc - self.soc_min_frac) * self.e_cap_kwh / dt
                - inverter_loss_kw
            ) * self.eta_discharge,
            0.0,
        )
        if discharge > available_discharge_kw and discharge > 0.0:
            scale = available_discharge_kw / discharge
            p = {phase: value * scale if value < 0.0 else value for phase, value in p.items()}
            discharge = available_discharge_kw
        max_charge_kw = max(
            (
                (self.soc_max_frac - self.soc) * self.e_cap_kwh / dt
                + inverter_loss_kw
            ) / self.eta_charge,
            0.0,
        )
        if charge > max_charge_kw and charge > 0.0:
            scale = max_charge_kw / charge
            p = {phase: value * scale if value > 0.0 else value for phase, value in p.items()}
            charge = max_charge_kw

        energy_change_kw = self.eta_charge * charge - discharge / self.eta_discharge - inverter_loss_kw
        self.soc += energy_change_kw * dt / self.e_cap_kwh
        self.soc = min(self.soc_max_frac, max(self.soc_min_frac, self.soc))
        self.array_kw.append(sum(p.values()))
        self.array_kvar.append(sum(q.values()))
        self.array_soc.append(self.soc)
        self.array_inverter_loss_kw.append(inverter_loss_kw)
        return p, q

    def _reactive_loss(self, q_bess):
        if self.q_loss_rated_kw <= 0.0:
            return 0.0
        return self.q_loss_rated_kw * (q_bess / self.s_max_kva) ** 2

    def _q_for_loss(self, loss_kw, requested_q):
        if self.q_loss_rated_kw <= 0.0:
            return requested_q
        q_limit = self.s_max_kva * math.sqrt(max(loss_kw, 0.0) / self.q_loss_rated_kw)
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
        self.phase_voltages = None
        self.phase_voltages_pu = None
        self.phase_angles_deg = None
