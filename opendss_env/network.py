"""Physical meaning of the network, separate from the controller formulation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Network:
    representation: str
    phases: tuple[str, ...]

    def __post_init__(self):
        expected = {
            "single_phase": ("a",),
            "single_phase_equivalent": ("a",),
            "three_phase": ("a", "b", "c"),
        }
        if self.representation not in expected:
            raise ValueError("Network representation must be single_phase, single_phase_equivalent or three_phase")
        if self.phases != expected[self.representation]:
            raise ValueError("Network phases must match its representation")

    @property
    def voltage_basis(self):
        return "line_to_neutral" if self.representation == "single_phase" else "line_to_line"

    def connected_phases(self, phases):
        names = {"1": "a", "2": "b", "3": "c"}
        connected = tuple(names.get(str(phase).lower(), str(phase).lower()) for phase in phases) or self.phases
        if len(set(connected)) != len(connected) or not set(connected) <= set(self.phases):
            raise ValueError("Device phases must belong to the network without duplicates")
        return connected

    def as_dict(self):
        return {
            "representation": self.representation,
            "phases": list(self.phases),
            "voltage_basis": self.voltage_basis,
        }
