"""Physical meaning of the network, separate from the controller formulation."""
from dataclasses import dataclass


PHASE_NAME = {1: "a", 2: "b", 3: "c"}


def device_phases(device, default_phase_count):
    """Return connected phase names, including legacy numeric node labels."""
    phases = getattr(device, "phases", ()) or tuple(range(1, default_phase_count + 1))
    return tuple(PHASE_NAME.get(int(phase), str(phase).lower())
                 if isinstance(phase, int) or str(phase).isdigit() else str(phase).lower()
                 for phase in phases)


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
