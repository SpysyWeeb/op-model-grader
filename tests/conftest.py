import numpy as np
import pytest

from opgrader.extract import Channel, Drive


DT = 0.01  # 100 Hz


def make_drive(
    duration: float = 60.0,
    name: str = "synth",
    **signals,
) -> Drive:
    """Build a synthetic Drive. signals maps channel name -> array or scalar.

    vEgo defaults to a constant 15 m/s; aEgo to zeros; booleans to False.
    All channels share one 100 Hz timebase.
    """
    n = int(round(duration / DT))
    t = np.arange(n) * DT

    def arr(name, default, dtype):
        v = signals.get(name, default)
        if np.isscalar(v):
            v = np.full(n, v)
        v = np.asarray(v)
        assert len(v) == n, f"{name}: expected {n} samples, got {len(v)}"
        return v.astype(dtype)

    d = Drive(name=name)
    floats = {
        "vEgo": 15.0,
        "aEgo": 0.0,
        "steeringAngleDeg": 0.0,
        "steeringRateDeg": 0.0,
        "leadDRel": 0.0,
        "leadVRel": 0.0,
        "leadVLead": 0.0,
        "leadALeadK": 0.0,
        "yawRate": 0.0,
        "ccCurvature": 0.0,
        "ccSteeringAngleDeg": 0.0,
        "ccAccel": 0.0,
        "aTarget": 0.0,
        "desiredCurvature": 0.0,
        "planVisA0": 0.0,
        "planVisDA": 0.0,
        "planVisV4": 0.0,
    }
    bools = {
        "gasPressed": False,
        "brakePressed": False,
        "standstill": False,
        "steeringPressed": False,
        "leftBlinker": False,
        "rightBlinker": False,
        "enabled": False,
        "active": False,
        "experimentalMode": False,
        "ccEnabled": False,
        "latActive": False,
        "longActive": False,
        "leadStatus": False,
    }
    for k, dv in floats.items():
        d.channels[k] = Channel(t.copy(), arr(k, dv, np.float64))
    for k, dv in bools.items():
        d.channels[k] = Channel(t.copy(), arr(k, dv, np.bool_))
    ints = {"personality": 1, "planSource": 1}  # standard / lead0 unless overridden
    for k, dv in ints.items():
        d.channels[k] = Channel(t.copy(), arr(k, dv, np.int16))
    d.meta.vm_params = {
        # roughly a midsize SUV; any self-consistent set works for tests
        "mass": 2200.0,
        "wheelbase": 2.9,
        "centerToFront": 1.3,
        "steerRatio": 16.0,
        "steerRatioRear": 0.0,
        "tireStiffnessFront": 190000.0,
        "tireStiffnessRear": 250000.0,
    }
    return d


@pytest.fixture
def timebase():
    def f(duration):
        n = int(round(duration / DT))
        return np.arange(n) * DT

    return f
