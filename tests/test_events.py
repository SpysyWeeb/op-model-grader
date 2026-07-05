import numpy as np
import pytest

from opgrader.events import build_arrays, detect_events
from opgrader.segments import segment_drive
from tests.conftest import DT, make_drive


def _analyze(drive):
    seg = segment_drive(drive)
    assert seg is not None
    da = build_arrays(drive, seg)
    return seg, da, detect_events(drive, seg, da)


def test_stop_and_launch_detected_engaged():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.empty(n)
    a = np.zeros(n)
    # cruise 15 m/s, brake to 0 over 10..20, stand still, launch at 25 (1 m/s^2)
    v[t < 10] = 15.0
    dec = (t >= 10) & (t < 20)
    v[dec] = 15.0 - 1.5 * (t[dec] - 10)
    a[dec] = -1.5
    stand = (t >= 20) & (t < 25)
    v[stand] = 0.0
    go = t >= 25
    v[go] = np.minimum(1.0 * (t[go] - 25), 10.0)
    a[go] = np.where(v[go] < 10.0, 1.0, 0.0)

    d = make_drive(dur, vEgo=v, aEgo=a, enabled=True, latActive=True, longActive=True,
                   standstill=v < 0.1)
    _seg, _da, events = _analyze(d)

    stops = [e for e in events if e.kind == "stop"]
    launches = [e for e in events if e.kind == "launch"]
    assert len(stops) == 1
    assert stops[0].engaged is True
    # window starts at the last moment v >= 8 (t ~ 14.67)
    assert stops[0].t0 == pytest.approx(10 + 7 / 1.5, abs=0.1)
    assert len(launches) == 1
    lm = launches[0]
    assert lm.engaged is True
    # first motion when v > 0.15, i.e. t ~ 25.15
    assert lm.values["t_first_motion"] == pytest.approx(25.15, abs=0.05)


def test_lead_decel_response_latency():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    alk = np.zeros(n)
    alk[(t >= 30) & (t < 31)] = -2.0  # stimulus onset at t=30
    a = np.zeros(n)
    a[t >= 31.2] = -0.5  # our response starts 1.2 s later

    d = make_drive(dur, aEgo=a, leadStatus=True, leadDRel=30.0, leadVLead=15.0,
                   leadALeadK=alk, enabled=True, latActive=True, longActive=True)
    _seg, _da, events = _analyze(d)

    follows = [e for e in events if e.kind == "follow"]
    decels = [e for e in events if e.kind == "lead_decel"]
    assert len(follows) == 1
    assert len(decels) == 1
    ev = decels[0]
    assert ev.values["censored"] is False
    # smoothed accel needs ~0.18 s beyond the step to fall 0.3 below baseline
    assert ev.values["latency"] == pytest.approx(1.3, abs=0.2)


def test_lead_decel_censored_when_no_response():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    alk = np.zeros(n)
    alk[(t >= 30) & (t < 31)] = -2.0

    d = make_drive(dur, leadStatus=True, leadDRel=30.0, leadVLead=15.0,
                   leadALeadK=alk, enabled=True, latActive=True, longActive=True)
    _seg, _da, events = _analyze(d)
    decels = [e for e in events if e.kind == "lead_decel"]
    assert len(decels) == 1
    assert decels[0].values["censored"] is True
    assert decels[0].values["latency"] == 4.0


def test_pullaway_latency():
    dur = 40.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.zeros(n)
    a = np.zeros(n)
    go = t >= 11.5
    v[go] = np.minimum(1.0 * (t[go] - 11.5), 8.0)
    a[go] = np.where(v[go] < 8.0, 1.0, 0.0)
    v_lead = np.where(t >= 10.0, 1.0, 0.0)

    d = make_drive(dur, vEgo=v, aEgo=a, leadStatus=True, leadDRel=8.0,
                   leadVLead=v_lead, standstill=v < 0.1,
                   enabled=True, latActive=True, longActive=True)
    _seg, _da, events = _analyze(d)
    pulls = [e for e in events if e.kind == "pullaway"]
    assert len(pulls) == 1
    assert pulls[0].values["latency"] == pytest.approx(1.5, abs=0.1)


def test_manual_events_tagged_manual():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.empty(n)
    a = np.zeros(n)
    v[t < 10] = 15.0
    dec = (t >= 10) & (t < 20)
    v[dec] = 15.0 - 1.5 * (t[dec] - 10)
    a[dec] = -1.5
    v[t >= 20] = 0.0

    d = make_drive(dur, vEgo=v, aEgo=a, brakePressed=dec, standstill=v < 0.1)
    _seg, _da, events = _analyze(d)
    stops = [e for e in events if e.kind == "stop"]
    assert len(stops) == 1
    assert stops[0].engaged is False
    assert stops[0].has_override is False  # braking manually is not an override
