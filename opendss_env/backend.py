"""Small OpenDSSDirect adapter used by the Gym environment."""
from __future__ import annotations

import opendssdirect as _dss


class _Circuit:
    @property
    def buses_names(self):
        return _dss.Circuit.AllBusNames()

    @property
    def total_power(self):
        return _dss.Circuit.TotalPower()

    def set_active_bus(self, name):
        return _dss.Circuit.SetActiveBus(name)

    def set_active_element(self, name):
        return _dss.Circuit.SetActiveElement(name)


class _Bus:
    @property
    def nodes(self):
        return _dss.Bus.Nodes()

    @property
    def kv_base(self):
        return _dss.Bus.kVBase()

    @property
    def vmag_angle(self):
        return _dss.Bus.VMagAngle()

    @property
    def pu_vmag_angle(self):
        return _dss.Bus.puVmagAngle()


class _Solution:
    def solve(self):
        return _dss.Solution.Solve()

    @property
    def converged(self):
        return bool(_dss.Solution.Converged())


class _CktElement:
    @property
    def node_order(self):
        return _dss.CktElement.NodeOrder()

    @property
    def num_conductors(self):
        return _dss.CktElement.NumConductors()

    @property
    def powers(self):
        return _dss.CktElement.Powers()


class OpenDSSDirectBackend:
    """Expose the subset of OpenDSS used by :class:`MicrogridEnv`."""

    def __init__(self):
        self.circuit = _Circuit()
        self.bus = _Bus()
        self.cktelement = _CktElement()
        self.solution = _Solution()

    def text(self, command):
        return _dss.Text.Command(command)
