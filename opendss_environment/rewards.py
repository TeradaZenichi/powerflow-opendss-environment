"""Executed global and controller rewards, with explicit scopes and scales."""
from copy import deepcopy
from datetime import timedelta
import cmath
import math

from .evse import utc_timestamp


TERMS = {
    "feeder_cost": "minimize_feeder_cost", "node_cost": "minimize_node_cost",
    "device_cost": "minimize_device_cost",
    "energy_cost": "minimize_energy_cost", "voltage": "minimize_voltage_violation",
    "voltage_deviation": "minimize_voltage_deviation", "unbalance": "minimize_voltage_unbalance",
    "branch_overload": "minimize_branch_overload", "power_violation": "minimize_power_violation",
    "tracking": "track_power", "self_consumption": "maximize_self_consumption",
    "curtailment": "minimize_curtailment", "losses": "minimize_losses",
    "throughput": "minimize_throughput", "soc_reserve": "minimize_soc_violation",
    "soc_target": "track_soc_target", "ev_unmet": "minimize_ev_unmet_energy",
    "ev_infeasibility": "minimize_ev_infeasibility", "action_variation": "minimize_action_variation",
    "service": "service_incentive", "import": "minimize_import", "reactive": "minimize_reactive_power",
}


def _number(value, name, minimum=None):
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return value


def _names(values):
    if isinstance(values, str) or len(values) != len(set(values)):
        raise ValueError("Reward scopes require lists of unique IDs")
    return tuple(values)


class Reward:
    """Compose named rewards; calling an instance returns its global scalar."""

    def __init__(self, bus=None, v_min=0.95, v_max=1.05, *, controllers=None,
                 global_objective="cost", teacher_profile=None, scope=None):
        v_min, v_max = _number(v_min, "v_min"), _number(v_max, "v_max")
        if not 0 < v_min <= v_max:
            raise ValueError("Voltage limits must satisfy 0 < v_min <= v_max")
        if global_objective not in {"cost", "teacher"}:
            raise ValueError("global_objective must be cost or teacher")
        self.bus = bus.lower() if bus is not None else None
        self.v_min, self.v_max = v_min, v_max
        self.controllers = deepcopy(controllers or {})
        self.global_objective = global_objective
        self.teacher_profile = deepcopy(teacher_profile or {})
        self.scope = deepcopy(scope) if scope is not None else None

    def __call__(self, env):
        return self.minimize_cost(env) if self.global_objective == "cost" else self.minimize_teacher_objective(env)

    def evaluate(self, env, objectives=None):
        """Evaluate one executed interval; objectives override weights/parameters only."""
        objectives = deepcopy(objectives or {})
        if set(objectives) - set(self.controllers):
            raise ValueError("Objectives contain unknown controller IDs")
        result = {"cost": self.minimize_cost(env), "controllers": {}, "terms": {},
                  "costs": {"feeder": self.feeder_cost(env), "controllers": {}}}
        teacher = self._teacher_components(env)
        result["teacher"] = -(teacher["energy_cost"] + teacher["tie_break_cost"])
        result["teacher_terms"] = teacher
        result["global"] = result[self.global_objective]
        for controller_id, specification in self.controllers.items():
            spec = deepcopy(specification)
            if set(spec) - {"devices", "buses", "branches", "weights", "scales", "parameters"}:
                raise ValueError(f"Unknown reward configuration for {controller_id}")
            override = objectives.get(controller_id, {})
            if set(override) - {"weights", "parameters"}:
                raise ValueError("Objectives may override only weights and parameters")
            weights = spec.get("weights", {}) | override.get("weights", {})
            parameters = deepcopy(spec.get("parameters", {}))
            for term, values in override.get("parameters", {}).items():
                parameters[term] = parameters.get(term, {}) | values
            scales = spec.get("scales", {})
            if (set(weights) | set(scales) | set(parameters)) - TERMS.keys():
                raise ValueError(f"Unknown reward terms for {controller_id}")
            local = Reward(v_min=self.v_min, v_max=self.v_max,
                           scope={key: spec.get(key, {} if key == "devices" else [])
                                  for key in ("devices", "buses", "branches")})
            local._validate_scope(env)
            terms = {}
            for term, weight in weights.items():
                weight = _number(weight, "weight", 0.0)
                scale = _number(scales.get(term, 1.0), "scale", 0.0)
                if scale == 0.0:
                    raise ValueError("Reward scales must be positive")
                raw = _number(getattr(local, TERMS[term])(env, **parameters.get(term, {})), term) if weight else 0.0
                contribution = _number(weight * raw / scale, "weighted reward")
                terms[term] = {"raw": raw, "scale": scale, "normalized": raw / scale,
                               "weight": weight, "contribution": contribution}
            result["terms"][controller_id] = terms
            result["controllers"][controller_id] = _number(sum(value["contribution"] for value in terms.values()), "controller reward")
            result["costs"]["controllers"][controller_id] = {
                term.removesuffix("_cost"): -value["raw"] for term, value in terms.items()
                if term in {"feeder_cost", "node_cost", "device_cost"} and value["weight"]
            }
        return result

    @staticmethod
    def get_global(rewards):
        return rewards["global"]

    @staticmethod
    def get_cost(rewards):
        return rewards["cost"]

    @staticmethod
    def get_costs(rewards, controller_id=None):
        """Return unweighted currency costs; negative values are net revenue."""
        costs = rewards["costs"]
        return deepcopy(costs if controller_id is None else costs["controllers"][controller_id])

    @staticmethod
    def get_teacher(rewards):
        return rewards["teacher"]

    @staticmethod
    def get_local(rewards, controller_id):
        return rewards["controllers"][controller_id]

    @staticmethod
    def get_terms(rewards, controller_id):
        return deepcopy(rewards["terms"][controller_id])

    def _validate_scope(self, env):
        if set(self.scope) - {"devices", "buses", "branches"}:
            raise ValueError("Unknown reward scope fields")
        devices = self.scope.get("devices", {})
        if set(devices) - {"bess", "pv", "evse"}:
            raise ValueError("Reward devices must be bess, pv or evse")
        for kind, ids in devices.items():
            available = {device.id for device in getattr(env, f"{kind}_list", [])}
            if set(_names(ids)) - available:
                raise ValueError(f"Unknown {kind} device in reward scope")
        if set(_names(self.scope.get("buses", []))) - env.current_bus_voltages_pu.keys():
            raise ValueError("Unknown bus in reward scope")
        branches = {flow["element"] for flow in env.current_branch_measurements}
        if set(_names(self.scope.get("branches", []))) - branches:
            raise ValueError("Unknown branch in reward scope; use Line.name or Transformer.name")

    def _buses(self, env):
        buses = self.scope.get("buses", []) if self.scope is not None else (
            [self.bus] if self.bus is not None else list(env.current_bus_voltages_pu))
        if not buses or set(buses) - env.current_bus_voltages_pu.keys():
            raise ValueError("Select existing buses for an electrical reward")
        return buses

    def _devices(self, env, kinds=("bess", "pv", "evse")):
        selected = self.scope.get("devices", {}) if self.scope is not None else None
        if selected is not None and not any(selected.values()):
            raise ValueError("Select devices for a device reward")
        return [(kind, device) for kind in kinds for device in getattr(env, f"{kind}_list", [])
                if (device.id in selected.get(kind, []) if selected is not None
                    else self.bus is None or device.bus == self.bus)]

    def _power(self, env):
        """Net import P/Q at selected devices, or at the selected buses."""
        if self.scope is None or not any(self.scope.get("devices", {}).values()):
            return self.bus_power(env)
        p, q = 0.0, 0.0
        for kind, device in self._devices(env):
            measured = env.current_device_measurements[kind][device.id]
            p += -measured["generation_total_kw"] if kind == "pv" else measured["p_net_total_kw"]
            q -= measured["q_injection_total_kvar"]
        return {"p_kw": p, "q_kvar": q}

    @staticmethod
    def minimize_cost(env):
        """Negative executed grid energy cost, in case currency."""
        return -env.current_cost

    @staticmethod
    def feeder_cost(env):
        """Actual settlement at the external grid connection, in case currency."""
        return float(env.current_cost)

    def node_cost(self, env, buy_price=None, sell_price=None):
        """Settle each selected node separately, including all its loads/devices."""
        if self.scope is None and self.bus is None:
            raise ValueError("Select buses for node cost")
        powers = dict.fromkeys(self._buses(env), 0.0)
        for flow in self._flows(env):
            powers[flow["bus"]] -= flow["p_flow_pu"] * env.data["base_kva"]
        source = env.data.get("grid_limits", {}).get("bus", "").lower()
        if source in powers:
            # Remove the external source contribution from the local node balance.
            powers[source] -= env.current_grid_kw
        return sum(self._settlement(env, power, buy_price, sell_price) for power in powers.values())

    def device_cost(self, env, buy_price=None, sell_price=None, throughput_price=0.0, service_payment=0.0):
        """Sum individual AC settlements plus storage wear proxy minus service payment."""
        devices = self._devices(env)
        if not devices:
            raise ValueError("Select devices for device cost")
        wear = _number(throughput_price, "throughput_price", 0.0)
        payment = _number(service_payment, "service_payment", 0.0)
        cost = 0.0
        for kind, device in devices:
            measured = env.current_device_measurements[kind][device.id]
            power = -measured["generation_total_kw"] if kind == "pv" else measured["p_net_total_kw"]
            cost += self._settlement(env, power, buy_price, sell_price)
        if wear:
            cost -= wear * self.minimize_throughput(env)
        return cost - payment

    @staticmethod
    def _settlement(env, power, buy_price, sell_price):
        buy = _number(env.grid.prices[env.idx] if buy_price is None else buy_price, "buy_price")
        sell = _number(env.grid.sell_prices[env.idx] if sell_price is None else sell_price, "sell_price")
        return (buy * max(power, 0.0) - sell * max(-power, 0.0)) * env.dt

    def minimize_feeder_cost(self, env):
        return -self.feeder_cost(env)

    def minimize_node_cost(self, env, **parameters):
        return -self.node_cost(env, **parameters)

    def minimize_device_cost(self, env, **parameters):
        return -self.device_cost(env, **parameters)

    def minimize_teacher_objective(self, env):
        """Negative per-interval teacher objective; powers are converted from pu to kW."""
        terms = self._teacher_components(env)
        return -(terms["energy_cost"] + terms["tie_break_cost"])

    def _teacher_components(self, env):
        profile = self.teacher_profile
        if set(profile) - {"formulation", "feed_in_ratio", "tie_break_penalty"}:
            raise ValueError("Unknown teacher profile fields")
        formulation = profile.get("formulation", "single_phase_socp")
        penalties = {"single_phase_socp": 0.0, "single_phase_active": 1e-6, "three_phase_ivr": 1e-8}
        if formulation not in penalties:
            raise ValueError("Unknown teacher formulation")
        ratio = _number(profile.get("feed_in_ratio", env.data.get("feed_in_tariff_ratio", 1.0)), "feed_in_ratio", 0.0)
        epsilon = _number(profile.get("tie_break_penalty", penalties[formulation]), "tie_break_penalty", 0.0)
        if formulation == "single_phase_socp" and epsilon:
            raise ValueError("single_phase_socp has no tie-break penalty")
        imported, exported = max(-env.current_grid_kw, 0.0), max(env.current_grid_kw, 0.0)
        cost = env.dt * float(env.grid.prices[env.idx]) * (imported - ratio * exported)
        throughput = sum(device.array_charge_kw[-1] + device.array_discharge_kw[-1] for device in env.bess_list)
        if formulation == "single_phase_active":
            throughput += imported + exported
        return {"energy_cost": cost, "tie_break_cost": epsilon * env.dt * throughput,
                "profile": {"formulation": formulation, "feed_in_ratio": ratio, "tie_break_penalty": epsilon}}

    def _flows(self, env):
        buses = self._buses(env)
        return [flow for flow in env.current_branch_measurements if flow["bus"] in buses]

    def bus_power(self, env):
        """Net bus exchange in kW/kvar, positive importing from connected branches."""
        flows = self._flows(env)
        base = env.data["base_kva"]
        return {"p_kw": -sum(flow["p_flow_pu"] for flow in flows) * base,
                "q_kvar": -sum(flow["q_flow_pu"] for flow in flows) * base}

    def minimize_voltage_deviation(self, env, reference=1.0):
        """Negative sum of phase voltage deviations; bus=None covers all buses."""
        reference = _number(reference, "reference", 0.0)
        return -sum(abs(value - reference) for bus in self._buses(env)
                    for value in env.current_bus_voltages_pu[bus].values())

    def minimize_voltage_violation(self, env, v_min=None, v_max=None):
        """Negative sum of phase deviations outside the desired pu band."""
        minimum = self.v_min if v_min is None else _number(v_min, "v_min")
        maximum = self.v_max if v_max is None else _number(v_max, "v_max")
        if not 0 < minimum <= maximum:
            raise ValueError("Voltage limits must satisfy 0 < v_min <= v_max")
        return -sum(max(minimum - value, 0.0) + max(value - maximum, 0.0)
                    for bus in self._buses(env) for value in env.current_bus_voltages_pu[bus].values())

    def minimize_voltage_unbalance(self, env, maximum_pct=0.0):
        """Negative excess VUF (%), using negative/positive sequence voltages."""
        maximum_pct = _number(maximum_pct, "maximum_pct", 0.0)
        return -sum(max(self._vuf(env, bus) - maximum_pct, 0.0) for bus in self._buses(env))

    @staticmethod
    def _vuf(env, bus):
        voltages = env.current_bus_voltages_pu[bus]
        if set(voltages) != {"a", "b", "c"}:
            raise ValueError("Voltage unbalance requires phases a, b and c at the selected bus")
        angles = env.current_bus_angles_deg[bus]
        va, vb, vc = (cmath.rect(voltages[phase], math.radians(angles[phase])) for phase in ("a", "b", "c"))
        a = cmath.rect(1.0, 2 * math.pi / 3)
        positive = (va + a * vb + a**2 * vc) / 3
        negative = (va + a**2 * vb + a * vc) / 3
        if abs(positive) < 1e-12:
            raise ValueError("VUF is undefined when positive-sequence voltage is zero")
        return 100 * abs(negative) / abs(positive)

    def minimize_import(self, env):
        """Negative imported energy at the selected bus, in kWh per interval."""
        return -max(self._power(env)["p_kw"], 0.0) * env.dt

    def minimize_reactive_power(self, env):
        """Negative magnitude of the bus's net reactive exchange, in kvar."""
        return -abs(self._power(env)["q_kvar"])

    def minimize_power_violation(self, env, import_limit_kw, export_limit_kw):
        """Negative power exceeding the local import/export envelope, in kW."""
        import_limit_kw = _number(import_limit_kw, "import_limit_kw", 0.0)
        export_limit_kw = _number(export_limit_kw, "export_limit_kw", 0.0)
        power = self._power(env)["p_kw"]
        return -max(power - import_limit_kw, 0.0) - max(-power - export_limit_kw, 0.0)

    def track_power(self, env, p_reference_kw=0.0, q_reference_kvar=0.0):
        """Negative distance to the requested net import P/Q, in kVA."""
        p_reference_kw = _number(p_reference_kw, "p_reference_kw")
        q_reference_kvar = _number(q_reference_kvar, "q_reference_kvar")
        power = self._power(env)
        return -math.hypot(power["p_kw"] - p_reference_kw, power["q_kvar"] - q_reference_kvar)

    def minimize_branch_overload(self, env):
        """Negative loading above 1; use the worst terminal per branch/phase."""
        branches = self.scope.get("branches", []) if self.scope is not None else []
        buses = self.scope.get("buses", []) if self.scope is not None else ([self.bus] if self.bus else [])
        if self.scope is not None and not (branches or buses):
            raise ValueError("Select branches or buses for overload rewards")
        loading = {}
        for flow in env.current_branch_measurements:
            if (branches and flow["element"] not in branches) or (not branches and buses and flow["bus"] not in buses):
                continue
            if flow["loading_pct"] is None:
                raise ValueError("Selected branch has no current rating")
            key = flow["element"], flow["phase"]
            loading[key] = max(loading.get(key, 0.0), flow["loading_pct"] / 100)
        return -sum(max(value - 1, 0.0) for value in loading.values())

    def minimize_energy_cost(self, env, buy_price=None, sell_price=None):
        """Device-group net AC settlement; this is not an allocated grid bill."""
        buy = float(env.grid.prices[env.idx]) if buy_price is None else _number(buy_price, "buy_price")
        sell = float(env.grid.sell_prices[env.idx]) if sell_price is None else _number(sell_price, "sell_price")
        self._devices(env)
        power = self._power(env)["p_kw"]
        return -(buy * max(power, 0.0) - sell * max(-power, 0.0)) * env.dt

    def minimize_curtailment(self, env):
        """Exclude inverter losses from available solar energy curtailed."""
        return -sum(max(float(device.profile[env.idx]) - max(device.array_kw[-1], 0.0)
                        - device.array_inverter_loss_kw[-1], 0.0) * env.dt
                    for _, device in self._devices(env, ("pv",)))

    def maximize_self_consumption(self, env, solar_export_kwh=None):
        """Solar provenance must be supplied by an explicit energy accounting model."""
        if solar_export_kwh is None:
            raise ValueError("Self-consumption requires measured/accounted solar_export_kwh")
        return -_number(solar_export_kwh, "solar_export_kwh", 0.0) + self.minimize_curtailment(env)

    def minimize_losses(self, env):
        losses = 0.0
        for kind, device in self._devices(env):
            if kind == "evse":
                losses += device.array_total_loss_kwh[-1]
            else:
                losses += device.array_inverter_loss_kw[-1] * env.dt
                if kind == "bess":
                    losses += ((1 - device.eta_charge) * device.array_charge_kw[-1]
                               + (1 / device.eta_discharge - 1) * device.array_discharge_kw[-1]) * env.dt
        return -losses

    def minimize_throughput(self, env):
        """Cell-side energy throughput, including converter losses drawn from the cell."""
        throughput = 0.0
        for kind, device in self._devices(env, ("bess", "evse")):
            throughput += device.array_battery_throughput_kwh[-1] if kind == "evse" else env.dt * (
                device.eta_charge * device.array_charge_kw[-1]
                + device.array_discharge_kw[-1] / device.eta_discharge + device.array_inverter_loss_kw[-1])
        return -throughput

    def _storage_states(self, env):
        end = utc_timestamp(env.timestamps.iloc[env.idx]) + timedelta(hours=env.dt)
        for kind, device in self._devices(env, ("bess", "evse")):
            if kind == "bess":
                yield device, device.soc
            else:
                session = device.current_session(end)
                if session is not None:
                    yield device, session.soc

    def minimize_soc_violation(self, env, minimum=0.0, maximum=1.0):
        minimum, maximum = _number(minimum, "minimum", 0.0), _number(maximum, "maximum", 0.0)
        if not minimum <= maximum <= 1.0:
            raise ValueError("SoC reserve must satisfy 0 <= minimum <= maximum <= 1")
        return -sum(max(minimum - soc, 0.0) + max(soc - maximum, 0.0) for _, soc in self._storage_states(env))

    def track_soc_target(self, env, target=0.5, at_step=None):
        """By default apply only at episode end; at_step is a 1-based interval count."""
        target = _number(target, "target", 0.0)
        if target > 1 or (at_step is not None and (int(at_step) != at_step or at_step < 1)):
            raise ValueError("Invalid SoC target or at_step")
        if env.idx + 1 != (env.steps if at_step is None else at_step):
            return 0.0
        return -sum(abs(soc - target) for _, soc in self._storage_states(env))

    def minimize_ev_unmet_energy(self, env):
        """AC-equivalent deficit; only departure events in this interval count."""
        return -sum(event["unmet_kwh"] for _, device in self._devices(env, ("evse",))
                    for event in device.events[-1] if event["event"] == "departure")

    def minimize_ev_infeasibility(self, env, available_charge_kw=None):
        """Cell-side deficit under a constant future P limit and Q=0; nominal is optimistic."""
        end = utc_timestamp(env.timestamps.iloc[env.idx]) + timedelta(hours=env.dt)
        penalty = 0.0
        for _, device in self._devices(env, ("evse",)):
            session = device.current_session(end)
            if session is None:
                continue
            cfg = session.config
            power = min(device.p_max_kw, device.s_max_kva, cfg.rated_power_kw)
            if available_charge_kw is not None:
                limit = available_charge_kw[device.id] if isinstance(available_charge_kw, dict) else available_charge_kw
                power = min(power, _number(limit, "available_charge_kw", 0.0))
            hours = (cfg.departure - end).total_seconds() / 3600
            recoverable = max(cfg.efficiency * power - device.conversion_loss(power, 0.0), 0.0) * hours
            needed = max((cfg.target_soc - session.soc) * cfg.capacity_kwh, 0.0)
            penalty += max(needed - recoverable, 0.0)
        return -penalty

    def minimize_action_variation(self, env):
        variation = 0.0
        for kind, device in self._devices(env):
            current = env.current_device_measurements[kind][device.id]
            previous = env.previous_device_measurements[kind].get(device.id, {})
            field = "generation_kw" if kind == "pv" else "p_net_kw"
            for phase, power in current[field].items():
                dp = power - previous.get(field, {}).get(phase, 0.0)
                dq = current["q_injection_kvar"][phase] - previous.get("q_injection_kvar", {}).get(phase, 0.0)
                variation += math.hypot(dp, dq) / device.s_max_kva
        return -variation

    def service_incentive(self, env, p_reference_kw=0.0, q_reference_kvar=0.0,
                          p_rate=0.0, q_rate=0.0, valid_until=None):
        """Pay capped requested-direction P/Q service; references use net-import signs."""
        start = utc_timestamp(env.timestamps.iloc[env.idx])
        hours = env.dt if valid_until is None else max(min(
            (utc_timestamp(valid_until) - start).total_seconds() / 3600, env.dt), 0.0)
        power = self._power(env)
        payment = 0.0
        for actual, reference, rate in ((power["p_kw"], p_reference_kw, p_rate),
                                         (power["q_kvar"], q_reference_kvar, q_rate)):
            reference, rate = _number(reference, "reference"), _number(rate, "rate", 0.0)
            served = min(max(actual * (1 if reference > 0 else -1), 0.0), abs(reference))
            payment += rate * served * hours
        return payment


# Public compatibility names for existing callers.
minimize_cost = Reward.minimize_cost
minimize_voltage_deviation = Reward().minimize_voltage_deviation
