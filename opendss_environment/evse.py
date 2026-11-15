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


class EVSession:
    def __init__(self, config):
        self.config = config
        self.remaining_kwh = config.target_energy_kwh
        self.delivered_kwh = 0.0
        self.soc = config.arrival_soc
        self.carry_in = False
        self.energy_before_episode_kwh = 0.0

    def deliver(self, energy_kwh):
        energy = min(max(float(energy_kwh), 0.0), self.remaining_kwh)
        self.remaining_kwh -= energy
        if self.remaining_kwh < 1e-12:
            self.remaining_kwh = 0.0
        self.delivered_kwh += energy
        self.soc = min(
            self.config.target_soc,
            self.config.arrival_soc
            + self.config.efficiency * self.delivered_kwh / self.config.capacity_kwh,
        )
        return energy


class EVSE:
    def __init__(self, config, sessions):
        self.id, self.bus = config.id, config.bus
        self.p_max_kw, self.control_mode = config.p_max_kw, config.control_mode
        self.phases = tuple(config.phases)
        self.connection, self.dispatch_mode = config.connection, config.dispatch_mode
        self.sessions = [EVSession(session) for session in sessions]
        self.next_session = 0
        self.last_command_kw = 0.0
        self.delivered_kwh = 0.0
        self.energy_before_episode_kwh = 0.0
        self.unmet_kwh = 0.0
        self.completed_sessions = 0
        self.array_kw = []
        self.array_energy_kwh = []
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
        if session is None or session.remaining_kwh <= 0.0:
            return 0.0
        command = self.last_command_kw if self.controlled else self.p_max_kw
        return min(max(command, 0.0), self.p_max_kw, session.config.rated_power_kw)

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
        session.carry_in = True
        session.energy_before_episode_kwh = delivered
        self.energy_before_episode_kwh = delivered

    def operate(self, start, end, command_kw=0.0):
        duration_h = (end - start).total_seconds() / 3600.0
        if duration_h <= 0.0:
            raise ValueError("EVSE interval must be positive")
        command = min(max(float(command_kw), 0.0), self.p_max_kw)
        self.last_command_kw = command
        interval_energy = 0.0
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
            power = min(power, self.p_max_kw, cfg.rated_power_kw)
            interval_energy += session.deliver(power * overlap_h)

            if start <= cfg.arrival < end:
                events.append({"event": "arrival", "session_id": cfg.id})
            if start < cfg.departure <= end:
                unmet = session.remaining_kwh
                self.unmet_kwh += unmet
                self.completed_sessions += 1
                events.append({
                    "event": "departure", "session_id": cfg.id,
                    "delivered_kwh": session.delivered_kwh,
                    "energy_before_episode_kwh": session.energy_before_episode_kwh,
                    "unmet_kwh": unmet, "soc_frac": session.soc,
                })
                self.next_session = index + 1
                index += 1
                continue
            break

        average_kw = interval_energy / duration_h
        self.delivered_kwh += interval_energy
        self.array_kw.append(average_kw)
        self.array_energy_kwh.append(interval_energy)
        self.array_connected_fraction.append(min(connected_h / duration_h, 1.0))
        self.array_session_id.append(0 if start_session is None else start_session.config.id)
        end_session = self.current_session(end)
        self.array_soc.append(math.nan if end_session is None else end_session.soc)
        self.events.append(events)
        return {
            "p_net_total_kw": average_kw,
            "energy_delivered_kwh": interval_energy,
            "connected_fraction": min(connected_h / duration_h, 1.0),
            "session_id": 0 if start_session is None else start_session.config.id,
            "events": events,
        }

    def commit(self, pending):
        for name in (
            "sessions", "next_session", "last_command_kw", "delivered_kwh",
            "energy_before_episode_kwh", "unmet_kwh", "completed_sessions",
        ):
            setattr(self, name, getattr(pending, name))
        for name in (
            "array_kw", "array_energy_kwh", "array_connected_fraction",
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
