"""Small OpenDSSDirect adapter used by the Gym environment."""
from __future__ import annotations

import opendssdirect as _dss


class _Circuit:
    def __init__(self, dss):
        self._dss = dss

    @property
    def buses_names(self):
        return self._dss.Circuit.AllBusNames()

    @property
    def total_power(self):
        return self._dss.Circuit.TotalPower()

    def set_active_bus(self, name):
        return self._dss.Circuit.SetActiveBus(name)

    def set_active_element(self, name):
        return self._dss.Circuit.SetActiveElement(name)


class _Bus:
    def __init__(self, dss):
        self._dss = dss

    @property
    def nodes(self):
        return self._dss.Bus.Nodes()

    @property
    def kv_base(self):
        return self._dss.Bus.kVBase()

    @property
    def vmag_angle(self):
        return self._dss.Bus.VMagAngle()

    @property
    def pu_vmag_angle(self):
        return self._dss.Bus.puVmagAngle()


class _Solution:
    def __init__(self, dss):
        self._dss = dss

    def solve(self):
        return self._dss.Solution.Solve()

    @property
    def converged(self):
        return bool(self._dss.Solution.Converged())


class _CktElement:
    def __init__(self, dss):
        self._dss = dss

    @property
    def node_order(self):
        return self._dss.CktElement.NodeOrder()

    @property
    def num_conductors(self):
        return self._dss.CktElement.NumConductors()

    @property
    def powers(self):
        return self._dss.CktElement.Powers()


class OpenDSSDirectBackend:
    """Expose the subset of OpenDSS used by :class:`MicrogridEnv`."""

    def __init__(self):
        self._dss = _dss.NewContext()
        self._dss.Basic.AllowChangeDir(False)
        self.circuit = _Circuit(self._dss)
        self.bus = _Bus(self._dss)
        self.cktelement = _CktElement(self._dss)
        self.solution = _Solution(self._dss)

    def text(self, command):
        return self._dss.Text.Command(command)
