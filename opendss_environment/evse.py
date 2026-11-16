from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EVSessionConfig:
    id: int
    arrival: object
    departure: object
    arrival_soc: float
    target_soc: float
    capacity_kwh: float
    efficiency: float
    target_energy_kwh: float
    rated_power_kw: float


@dataclass(frozen=True)
class EVSEConfig:
    id: str
    bus: str
    p_max_kw: float
    control_mode: str
    phases: tuple = ()
    connection: str = "wye"
    dispatch_mode: str = "aggregate"
    v2g: bool = False
    p_discharge_max_kw: float | None = None
    soc_min_frac: float = 0.0
    power_factor: float = 1.0
    s_max_kva: float | None = None
    standby_loss_kw: float = 0.0
    loss_linear_kw: float = 0.0
    loss_quadratic_kw: float = 0.0


class EVSession:
    def __init__(self, config):
        self.config = config
        self.remaining_kwh = config.target_energy_kwh
        self.delivered_kwh = 0.0
        self.discharged_kwh = 0.0
        self.conversion_loss_kwh = 0.0
        self.soc = config.arrival_soc
        self.carry_in = False
        self.energy_before_episode_kwh = 0.0

    def deliver(self, energy_kwh):
        soc_limit = max(
            (self.config.target_soc - self.soc)
            * self.config.capacity_kwh / self.config.efficiency,
            0.0,
        )
        energy = min(max(float(energy_kwh), 0.0), self.remaining_kwh, soc_limit)
        self.remaining_kwh -= energy
        if self.remaining_kwh < 1e-12:
            self.remaining_kwh = 0.0
        self.delivered_kwh += energy
        self.soc = min(self.config.target_soc, self.soc + (
            self.config.efficiency * energy / self.config.capacity_kwh
        ))
        return energy

    def discharge(self, energy_kwh, soc_min_frac):
        available = max(
            (self.soc - soc_min_frac)
            * self.config.capacity_kwh * self.config.efficiency,
            0.0,
        )
        energy = min(max(float(energy_kwh), 0.0), available)
        self.discharged_kwh += energy
        self.remaining_kwh += energy / self.config.efficiency ** 2
        self.soc = max(
            soc_min_frac,
            self.soc - energy / (
                self.config.efficiency * self.config.capacity_kwh
            ),
        )
        return energy

    def apply_loss(self, energy_kwh, soc_min_frac):
        available = max(
            (self.soc - soc_min_frac) * self.config.capacity_kwh,
            0.0,
        )
        energy = min(max(float(energy_kwh), 0.0), available)
        self.conversion_loss_kwh += energy
        self.remaining_kwh += energy / self.config.efficiency
        self.soc = max(
            soc_min_frac,
            self.soc - energy / self.config.capacity_kwh,
        )
        return energy


class EVSE:
    def __init__(self, config, sessions):
        self.id, self.bus = config.id, config.bus
        self.p_max_kw, self.control_mode = config.p_max_kw, config.control_mode
        self.phases = tuple(config.phases)
        self.connection, self.dispatch_mode = config.connection, config.dispatch_mode
        self.v2g, self.soc_min_frac = config.v2g, config.soc_min_frac
        self.power_factor = config.power_factor
        self.s_max_kva = (
            config.p_max_kw / config.power_factor
            if config.s_max_kva is None else config.s_max_kva
        )
        self.standby_loss_kw = config.standby_loss_kw
        self.loss_linear_kw = config.loss_linear_kw
        self.loss_quadratic_kw = config.loss_quadratic_kw
        self.p_discharge_max_kw = (
            config.p_max_kw
            if config.p_discharge_max_kw is None
            else config.p_discharge_max_kw
        )
        self.sessions = [EVSession(session) for session in sessions]
        self.next_session = 0
        self.last_command_kw = 0.0
        self.last_command_kvar = 0.0
        self.delivered_kwh = 0.0
        self.discharged_kwh = 0.0
        self.conversion_loss_kwh = 0.0
        self.energy_before_episode_kwh = 0.0
        self.unmet_kwh = 0.0
        self.completed_sessions = 0
        self.array_kw = []
        self.array_kvar = []
        self.array_energy_kwh = []
        self.array_charge_kwh = []
        self.array_discharge_kwh = []
        self.array_loss_kw = []
        self.array_total_loss_kwh = []
        self.array_battery_throughput_kwh = []
        self.array_connected_fraction = []
        self.array_session_id = []
        self.array_soc = []
        self.events = []

    @property
    def controlled(self):
        return self.control_mode == "controlled"

    def current_session(self, timestamp):
        for session in self.sessions[self.next_session:]:
            if session.config.arrival <= timestamp < session.config.departure:
                return session
            if session.config.arrival > timestamp:
                break
        return None

    def preview_power(self, timestamp):
        session = self.current_session(timestamp)
        if session is None:
            return 0.0
        command = self.last_command_kw if self.controlled else self.p_max_kw
        if command >= 0.0:
            return 0.0 if session.remaining_kwh <= 0.0 else min(
                command, self.p_max_kw, session.config.rated_power_kw
            )
        if not self.v2g or session.soc <= self.soc_min_frac:
            return 0.0
        return -min(-command, self.p_discharge_max_kw, session.config.rated_power_kw)

    def preview_setpoint(self, timestamp):
        session = self.current_session(timestamp)
        if session is None:
            return 0.0, 0.0
        power = self.preview_power(timestamp)
        reactive = (
            self.last_command_kvar if self.controlled
            else self.reactive_injection(power)
        )
        return self.limit_setpoint(power, reactive, session.config.rated_power_kw)

    def reactive_injection(self, p_net_kw):
        return -float(p_net_kw) * math.tan(math.acos(self.power_factor))

    def limit_setpoint(self, p_net_kw, q_injection_kvar, rated_power_kw):
        p = min(max(float(p_net_kw), -self.p_discharge_max_kw), self.p_max_kw)
        p = min(max(p, -rated_power_kw), rated_power_kw)
        if not self.v2g:
            p = max(p, 0.0)
        q_limit = math.sqrt(max(self.s_max_kva ** 2 - p ** 2, 0.0))
        q = min(max(float(q_injection_kvar), -q_limit), q_limit)
        return p, q

    def conversion_loss(self, p_net_kw, q_injection_kvar):
        loading = min(
            math.hypot(p_net_kw, q_injection_kvar) / self.s_max_kva,
            1.0,
        )
        return (
            self.standby_loss_kw
            + self.loss_linear_kw * loading
            + self.loss_quadratic_kw * loading ** 2
        )

    def initialize(self, timestamp):
        session = self.current_session(timestamp)
        if session is None or session.config.arrival >= timestamp:
            return
        if self.controlled:
            raise ValueError(
                f"EV episode starts during controlled session {session.config.id}; "
                "previous actions are unknown"
            )
        power = min(self.p_max_kw, session.config.rated_power_kw)
        elapsed_h = (timestamp - session.config.arrival).total_seconds() / 3600.0
        delivered = session.deliver(power * elapsed_h)
        actual_power = delivered / elapsed_h
        reactive = self.reactive_injection(actual_power)
        session.apply_loss(
            self.conversion_loss(actual_power, reactive) * elapsed_h,
            self.soc_min_frac,
        )
        session.carry_in = True
        session.energy_before_episode_kwh = delivered
        self.energy_before_episode_kwh = delivered

    def operate(self, start, end, command_kw=0.0, q_command_kvar=0.0):
        duration_h = (end - start).total_seconds() / 3600.0
        if duration_h <= 0.0:
            raise ValueError("EVSE interval must be positive")
        command = float(command_kw)
        q_command = float(q_command_kvar)
        self.last_command_kw, self.last_command_kvar = command, q_command
        charged_kwh = 0.0
        discharged_kwh = 0.0
        conversion_loss_kwh = 0.0
        efficiency_loss_kwh = 0.0
        battery_throughput_kwh = 0.0
        reactive_energy_kvarh = 0.0
        connected_h = 0.0
        events = []
        start_session = self.current_session(start)
        index = self.next_session

        while index < len(self.sessions):
            session = self.sessions[index]
            cfg = session.config
            if cfg.departure <= start:
                self.next_session = index + 1
                index += 1
                continue
            if cfg.arrival >= end:
                break

            overlap_start = max(start, cfg.arrival)
            overlap_end = min(end, cfg.departure)
            overlap_h = max((overlap_end - overlap_start).total_seconds() / 3600.0, 0.0)
            connected_h += overlap_h
            power = self.p_max_kw if not self.controlled else command
            reactive = self.reactive_injection(power) if not self.controlled else q_command
            power, reactive = self.limit_setpoint(power, reactive, cfg.rated_power_kw)
            if power >= 0.0:
                energy = session.deliver(power * overlap_h)
                charged_kwh += energy
                battery_throughput_kwh += cfg.efficiency * energy
                efficiency_loss_kwh += (1 - cfg.efficiency) * energy
                actual_power = energy / overlap_h if overlap_h > 0.0 else 0.0
            else:
                energy = session.discharge(-power * overlap_h, self.soc_min_frac)
                discharged_kwh += energy
                battery_throughput_kwh += energy / cfg.efficiency
                efficiency_loss_kwh += (1 / cfg.efficiency - 1) * energy
                actual_power = -energy / overlap_h if overlap_h > 0.0 else 0.0
            if not self.controlled:
                reactive = self.reactive_injection(actual_power)
            loss = self.conversion_loss(actual_power, reactive) * overlap_h
            actual_loss = session.apply_loss(loss, self.soc_min_frac)
            conversion_loss_kwh += actual_loss
            battery_throughput_kwh += actual_loss
            reactive_energy_kvarh += reactive * overlap_h

            if start <= cfg.arrival < end:
                events.append({"event": "arrival", "session_id": cfg.id})
            if start < cfg.departure <= end:
                unmet = session.remaining_kwh
                self.unmet_kwh += unmet
                self.completed_sessions += 1
                events.append({
                    "event": "departure", "session_id": cfg.id,
                    "delivered_kwh": session.delivered_kwh,
                    "discharged_kwh": session.discharged_kwh,
                    "conversion_loss_kwh": session.conversion_loss_kwh,
                    "energy_before_episode_kwh": session.energy_before_episode_kwh,
                    "unmet_kwh": unmet, "soc_frac": session.soc,
                })
                self.next_session = index + 1
                index += 1
                continue
            break

        net_energy_kwh = charged_kwh - discharged_kwh
        average_kw = net_energy_kwh / duration_h
        self.delivered_kwh += charged_kwh
        self.discharged_kwh += discharged_kwh
        self.conversion_loss_kwh += conversion_loss_kwh
        self.array_kw.append(average_kw)
        q_injection_kvar = reactive_energy_kvarh / duration_h
        self.array_kvar.append(q_injection_kvar)
        self.array_energy_kwh.append(net_energy_kwh)
        self.array_charge_kwh.append(charged_kwh)
        self.array_discharge_kwh.append(discharged_kwh)
        self.array_loss_kw.append(conversion_loss_kwh / duration_h)
        self.array_total_loss_kwh.append(conversion_loss_kwh + efficiency_loss_kwh)
        self.array_battery_throughput_kwh.append(battery_throughput_kwh)
        self.array_connected_fraction.append(min(connected_h / duration_h, 1.0))
        self.array_session_id.append(0 if start_session is None else start_session.config.id)
        end_session = self.current_session(end)
        self.array_soc.append(math.nan if end_session is None else end_session.soc)
        self.events.append(events)
        return {
            "p_net_total_kw": average_kw,
            "q_injection_total_kvar": q_injection_kvar,
            "energy_delivered_kwh": charged_kwh,
            "energy_discharged_kwh": discharged_kwh,
            "net_energy_kwh": net_energy_kwh,
            "conversion_loss_kwh": conversion_loss_kwh,
            "connected_fraction": min(connected_h / duration_h, 1.0),
            "session_id": 0 if start_session is None else start_session.config.id,
            "events": events,
        }

    def commit(self, pending):
        for name in (
            "sessions", "next_session", "last_command_kw", "last_command_kvar",
            "delivered_kwh", "discharged_kwh", "conversion_loss_kwh",
            "energy_before_episode_kwh", "unmet_kwh", "completed_sessions",
        ):
            setattr(self, name, getattr(pending, name))
        for name in (
            "array_kw", "array_kvar", "array_energy_kwh", "array_charge_kwh",
            "array_discharge_kwh", "array_loss_kw", "array_connected_fraction",
            "array_total_loss_kwh", "array_battery_throughput_kwh",
            "array_session_id", "array_soc", "events",
        ):
            getattr(self, name)[:] = getattr(pending, name)


def utc_timestamp(value):
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def load_ev_sessions(path, devices):
    if not devices:
        return {}
    if path is None or not path.is_file():
        raise FileNotFoundError("EVSE devices require files.ev_sessions")

    columns = {
        "id", "evse", "arr", "dep", "soc0", "soct",
        "cap_kwh", "eta", "e_kwh", "p_kw",
    }
    frame = pd.read_csv(path, usecols=lambda name: name in columns)
    missing = columns - set(frame)
    if missing:
        raise ValueError(f"EV session CSV is missing columns {sorted(missing)}")
    selected_ids = {str(device["id"]) for device in devices}
    frame = frame[frame["evse"].astype(str).isin(selected_ids)].copy()
    frame["arr"] = pd.to_datetime(frame["arr"], utc=True, errors="raise")
    frame["dep"] = pd.to_datetime(frame["dep"], utc=True, errors="raise")
    numeric = ["soc0", "soct", "cap_kwh", "eta", "e_kwh", "p_kw"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if frame["id"].duplicated().any():
        raise ValueError("EV session IDs must be unique")
    valid = (
        (frame["dep"] > frame["arr"])
        & frame[numeric].notna().all(axis=1)
        & np.isfinite(frame[numeric]).all(axis=1)
        & frame["soc0"].between(0.0, 1.0)
        & frame["soct"].between(0.0, 1.0)
        & (frame["soct"] >= frame["soc0"])
        & (frame["cap_kwh"] > 0.0)
        & frame["eta"].between(0.0, 1.0, inclusive="right")
        & (frame["e_kwh"] >= 0.0)
        & (frame["p_kw"] > 0.0)
    )
    if not valid.all():
        raise ValueError("EV session CSV contains invalid times, SoC, energy or power")

    result = {}
    for evse_id in selected_ids:
        group = frame[frame["evse"].astype(str) == evse_id].sort_values(["arr", "dep"])
        if group.empty:
            raise ValueError(f"EVSE {evse_id!r} has no sessions")
        if (group["arr"].iloc[1:].reset_index(drop=True)
                < group["dep"].iloc[:-1].reset_index(drop=True)).any():
            raise ValueError(f"EVSE {evse_id!r} has overlapping sessions")
        result[evse_id] = tuple(
            EVSessionConfig(
                id=int(row.id), arrival=row.arr, departure=row.dep,
                arrival_soc=float(row.soc0), target_soc=float(row.soct),
                capacity_kwh=float(row.cap_kwh), efficiency=float(row.eta),
                target_energy_kwh=float(row.e_kwh), rated_power_kw=float(row.p_kw),
            )
            for row in group.itertuples(index=False)
        )
    return result
