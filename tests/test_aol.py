"""Always-On-Lateral attribution: model steers, human works the pedals."""

import numpy as np

from opgrader.events import build_arrays, detect_events
from opgrader.grading import collect_samples
from opgrader.lateral import detect_turn_episodes
from opgrader.segments import segment_drive
from tests.conftest import DT, make_drive


def _aol_drive():
    """latActive true, longActive false; driver brakes to a stop while the
    model steers through a 100 deg turn."""
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

    act = np.zeros(n)
    up = (t >= 5) & (t < 8)
    act[up] = 100.0 * (t[up] - 5) / 3
    act[(t >= 8) & (t < 10)] = 100.0
    down = (t >= 10) & (t < 14)
    act[down] = 100.0 * (1 - (t[down] - 10) / 4)

    return make_drive(
        dur,
        vEgo=v,
        aEgo=a,
        brakePressed=dec,
        standstill=v < 0.1,
        steeringAngleDeg=act,
        enabled=False,
        latActive=True,
        longActive=False,
    )


def test_aol_bucket_and_span_axes():
    d = _aol_drive()
    seg = segment_drive(d)
    assert seg.per_axis is True
    b = seg.bucket_times()
    assert b["lat_only"] > 50.0
    assert b["both"] == 0.0
    # long axis sees a manual span; lat axis sees an engaged span
    assert seg.spans_of("manual", axis="long")
    assert not seg.spans_of("engaged", axis="long")
    assert seg.spans_of("engaged", axis="lat")
    assert not seg.spans_of("manual", axis="lat")


def test_aol_stop_is_manual_long_and_turn_is_model_lat():
    d = _aol_drive()
    seg = segment_drive(d)
    da = build_arrays(d, seg)

    stops = [e for e in detect_events(d, seg, da) if e.kind == "stop"]
    assert len(stops) == 1
    assert stops[0].engaged is False  # human was braking

    eps = detect_turn_episodes("synth", seg, da)
    assert len(eps) == 1
    assert eps[0].engaged is True  # model was steering


def test_aol_samples_land_in_correct_pools():
    d = _aol_drive()
    seg = segment_drive(d)
    da = build_arrays(d, seg)
    events = detect_events(d, seg, da)
    samples = collect_samples([(d, seg, da, events)])

    # longitudinal smoothness: human pool only
    assert len(samples["rms_jerk"]["driver"]) >= 1
    assert len(samples["rms_jerk"]["model"]) == 0
    # lateral smoothness: model pool only
    assert len(samples["rms_lat_jerk"]["model"]) >= 1
    assert len(samples["rms_lat_jerk"]["driver"]) == 0
    # the stop's metrics feed the human stopping pool
    assert len(samples["stop_lurch"]["driver"]) == 1
    assert len(samples["stop_lurch"]["model"]) == 0
