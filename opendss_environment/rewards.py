"""Executed-cost reward and optional local electrical objectives."""
import cmath
import math


class Reward:
    """Evaluate post-action measurements; local methods require a bus name."""

    def __init__(self, bus=None, v_min=0.95, v_max=1.05):
        if not 0 < v_min <= v_max:
            raise ValueError("Voltage limits must satisfy 0 < v_min <= v_max")
        self.bus = bus.lower() if bus is not None else None
        self.v_min, self.v_max = v_min, v_max

    def __call__(self, env):
        return self.minimize_cost(env)

    @staticmethod
    def minimize_cost(env):
        """Negative executed grid energy cost, in case currency."""
        return -env.current_cost

    def _voltages(self, env):
        if self.bus not in env.current_bus_voltages_pu:
            raise ValueError("Select an existing bus for a local reward")
        return env.current_bus_voltages_pu[self.bus]

    def _flows(self, env):
        self._voltages(env)
        return [flow for flow in env.current_branch_measurements if flow["bus"] == self.bus]

    def bus_power(self, env):
        """Net bus exchange in kW/kvar, positive importing from connected branches."""
        flows = self._flows(env)
        base = env.data["base_kva"]
        return {"p_kw": -sum(flow["p_flow_pu"] for flow in flows) * base,
                "q_kvar": -sum(flow["q_flow_pu"] for flow in flows) * base}

    def minimize_voltage_deviation(self, env, reference=1.0):
        """Negative sum of phase voltage deviations; bus=None covers all buses."""
        buses = [self._voltages(env)] if self.bus is not None else env.current_bus_voltages_pu.values()
        return -sum(abs(value - reference) for phases in buses for value in phases.values())

    def minimize_voltage_violation(self, env):
        """Negative sum of phase deviations outside the desired pu band."""
        return -sum(max(self.v_min - value, 0.0) + max(value - self.v_max, 0.0)
                    for value in self._voltages(env).values())

    def minimize_voltage_unbalance(self, env, maximum_pct=0.0):
        """Negative excess VUF (%), using negative/positive sequence voltages."""
        voltages = self._voltages(env)
        if set(voltages) != {"a", "b", "c"}:
            raise ValueError("Voltage unbalance requires phases a, b and c at the selected bus")
        angles = env.current_bus_angles_deg[self.bus]
        va, vb, vc = (cmath.rect(voltages[phase], math.radians(angles[phase])) for phase in ("a", "b", "c"))
        a = cmath.rect(1.0, 2 * math.pi / 3)
        positive = (va + a * vb + a**2 * vc) / 3
        negative = (va + a**2 * vb + a * vc) / 3
        if abs(positive) < 1e-12:
            raise ValueError("VUF is undefined when positive-sequence voltage is zero")
        return -max(100 * abs(negative) / abs(positive) - maximum_pct, 0.0)

    def minimize_import(self, env):
        """Negative imported energy at the selected bus, in kWh per interval."""
        return -max(self.bus_power(env)["p_kw"], 0.0) * env.dt

    def minimize_reactive_power(self, env):
        """Negative magnitude of the bus's net reactive exchange, in kvar."""
        return -abs(self.bus_power(env)["q_kvar"])

    def minimize_power_violation(self, env, import_limit_kw, export_limit_kw):
        """Negative power exceeding the local import/export envelope, in kW."""
        if min(import_limit_kw, export_limit_kw) < 0:
            raise ValueError("Import/export limits must be nonnegative")
        power = self.bus_power(env)["p_kw"]
        return -max(power - import_limit_kw, 0.0) - max(-power - export_limit_kw, 0.0)

    def track_power(self, env, p_reference_kw=0.0, q_reference_kvar=0.0):
        """Negative distance to the requested net import P/Q, in kVA."""
        power = self.bus_power(env)
        return -math.hypot(power["p_kw"] - p_reference_kw, power["q_kvar"] - q_reference_kvar)


# Public compatibility names for existing callers.
minimize_cost = Reward.minimize_cost
minimize_voltage_deviation = Reward().minimize_voltage_deviation
