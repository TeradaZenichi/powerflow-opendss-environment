"""Small rollback checkpoints for built-in device updates and append-only histories."""
from datetime import timedelta

import numpy as np

from .evse import utc_timestamp


class StepCheckpoint:
    def __init__(self, env):
        self.index = env.idx
        self.objects, self.mappings, self.histories, self.arrays = [], [], [], []
        for device in (*env.bess_list, *env.pv_list, *env.evse_list, env.grid):
            self._object(device)
        self._object(env.results, track_arrays=True)
        start = utc_timestamp(env.timestamps.iloc[env.idx])
        end = start + timedelta(hours=env.dt)
        for device in env.evse_list:
            for index in range(device.next_session, len(device.sessions)):
                session = device.sessions[index]
                if session.config.arrival >= end:
                    break
                if session.config.departure > start:
                    self._object(session)

    def _object(self, obj, track_arrays=False):
        attributes = vars(obj).copy()
        self.objects.append((obj, attributes))
        for name, value in attributes.items():
            if name != "sessions":
                self._value(value, track_arrays)

    def _value(self, value, track_arrays):
        if isinstance(value, list):
            self.histories.append((value, len(value), value[-1] if value else None))
        elif isinstance(value, dict):
            self.mappings.append((value, value.copy()))
            for item in value.values():
                self._value(item, track_arrays)
        elif track_arrays and isinstance(value, np.ndarray):
            self.arrays.append((value, value[self.index]))

    def restore(self):
        for array, value in self.arrays:
            array[self.index] = value
        for history, length, last in self.histories:
            del history[length:]
            if length:
                history[-1] = last
        for mapping, previous in reversed(self.mappings):
            mapping.clear()
            mapping.update(previous)
        for obj, attributes in reversed(self.objects):
            vars(obj).clear()
            vars(obj).update(attributes)
