"""Plan-vs-You counterfactuals on synthetic traces with known lags."""

import numpy as np
import pytest

from opgrader.counterfactual import analyze_counterfactual
from opgrader.events import build_arrays, detect_events
from opgrader.lateral import detect_intent_windows, detect_turn_episodes
from opgrader.segments import segment_drive
from opgrader.vehicle_model import vehicle_model_from_params
from tests.conftest import DT, make_drive


def _prep(d):
    seg = segment_drive(d)
    da = build_arrays(d, seg)
    return seg, da, detect_events(d, seg, da)


def _run(d, with_intents=False, with_turns=False, targets=None):
    seg, da, events = _prep(d)
    vm = vehicle_model_from_params(d.meta.vm_params)
    vms = {d.name: vm}
    intents = detect_intent_windows(d.name, seg, da, vm) if with_intents else []
    turns = detect_turn_episodes(d.name, seg, da) if with_turns else []
    cf, cf_events = analyze_counterfactual(
        [(d, seg, da, events)], turns, intents,
        targets or {"aggressive": 1.0, "standard": 1.45, "relaxed": 1.75}, vms,
    )
    return cf, cf_events, da


def _act_curv(vm, steer_deg, v):
    return vm.calc_curvature(np.radians(steer_deg), np.maximum(v, 0.1))


def _turn_drive(plan_delay_s: float | None):
    """Manual left intersection turn; planned curvature = actual delayed by
    plan_delay_s (None = the model never plans the turn)."""
    dur = 40.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    blink = (t >= 2) & (t < 15)
    yaw = np.where((t >= 4) & (t < 12), -np.radians(90) / 8, 0.0)  # left turn
    act = np.zeros(n)
    up = (t >= 4) & (t < 6)
    act[up] = 60.0 * (t[up] - 4) / 2
    act[(t >= 6) & (t < 12)] = 60.0
    down = (t >= 12) & (t < 14)
    act[down] = 60.0 * (1 - (t[down] - 12) / 2)

    d = make_drive(dur, vEgo=5.0, steeringAngleDeg=act, leftBlinker=blink, yawRate=yaw)
    vm = vehicle_model_from_params(d.meta.vm_params)
    act_curv = _act_curv(vm, act, np.full(n, 5.0))
    if plan_delay_s is None:
        desired = np.zeros(n)
    else:
        # raw modelV2 sign is inverted (left-negative), delayed copy of actual
        desired = -np.interp(t - plan_delay_s, t, act_curv, left=0.0)
    d.channels["desiredCurvature"].v[:] = desired
    return d


def test_turn_in_lag_known_delay():
    cf, cf_events, _da = _run(_turn_drive(0.5), with_intents=True)
    assert len(cf.turn_in) == 1
    row = cf.turn_in[0]
    assert row["side"] == "left"
    assert row["never_planned"] is False
    assert row["lag"] == pytest.approx(0.5, abs=0.06)
    s = cf.turn_in_summary()
    assert s["median_lag"] == pytest.approx(0.5, abs=0.06)
    assert s["never"] == 0
    assert any(e.kind == "cf_turnin" for e in cf_events)


def test_turn_never_planned():
    cf, _ev, _da = _run(_turn_drive(None), with_intents=True)
    assert len(cf.turn_in) == 1
    assert cf.turn_in[0]["never_planned"] is True
    assert cf.turn_in[0]["lag"] is None
    assert cf.turn_in_summary()["never"] == 1


def test_unwind_lag_plan_earlier():
    """Manual sharp turn; planned curvature unwinds 1 s before the driver."""
    dur = 40.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    act = np.zeros(n)
    up = (t >= 5) & (t < 8)
    act[up] = 120.0 * (t[up] - 5) / 3
    act[(t >= 8) & (t < 12)] = 120.0
    down = (t >= 12) & (t < 18)
    act[down] = 120.0 * (1 - (t[down] - 12) / 6)

    d = make_drive(dur, vEgo=3.0, steeringAngleDeg=act)
    vm = vehicle_model_from_params(d.meta.vm_params)
    act_curv = _act_curv(vm, act, np.full(n, 3.0))
    desired = -np.interp(t + 1.0, t, act_curv, right=0.0)  # plan LEADS by 1 s
    d.channels["desiredCurvature"].v[:] = desired

    cf, _ev, _da = _run(d, with_turns=True)
    assert len(cf.unwind) == 1
    assert cf.unwind[0]["side"] == "left"
    assert cf.unwind[0]["lag"] == pytest.approx(-1.0, abs=0.1)


def _stop_drive(plan_lag_s: float | None, with_lead=True):
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
    # stop window starts at last v>=8: t0 = 10 + 7/1.5
    t_win = 10 + 7 / 1.5
    at = np.zeros(n)
    if plan_lag_s is not None:
        at[t >= t_win + plan_lag_s] = -2.0
    return make_drive(dur, vEgo=v, aEgo=a, brakePressed=dec, standstill=v < 0.1,
                      leadStatus=with_lead, leadDRel=30.0 if with_lead else 0.0,
                      leadVLead=np.maximum(v - 1.0, 0.0), aTarget=at)


def test_braking_onset_lag():
    cf, cf_events, _da = _run(_stop_drive(0.3))
    assert cf.braking_summary()["n"] == 1
    assert cf.braking[0]["lag"] == pytest.approx(0.3, abs=0.06)
    assert cf.braking_never == 0
    assert any(e.kind == "cf_brake" for e in cf_events)


def test_braking_plan_never_brakes():
    cf, _ev, _da = _run(_stop_drive(None))
    assert cf.braking_never == 1
    assert cf.braking_summary()["n"] == 0


def test_braking_requires_lead():
    cf, cf_events, _da = _run(_stop_drive(0.3, with_lead=False))
    assert cf.braking_summary()["n"] == 0
    assert cf.braking_skipped_no_lead == 1
    assert not any(e.kind == "cf_brake" for e in cf_events)


def test_launch_onset_plan_earlier():
    """Manual pull-away: driver responds 1.5 s after lead moves, plan +0.3
    crossing at 0.8 s -> lag -0.7."""
    dur = 40.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.zeros(n)
    a = np.zeros(n)
    go = t >= 11.5
    v[go] = np.minimum(1.0 * (t[go] - 11.5), 8.0)
    a[go] = np.where(v[go] < 8.0, 1.0, 0.0)
    v_lead = np.where(t >= 10.0, 1.0, 0.0)
    at = np.where(t >= 10.8, 1.0, 0.0)

    d = make_drive(dur, vEgo=v, aEgo=a, leadStatus=True, leadDRel=8.0,
                   leadVLead=v_lead, standstill=v < 0.1, aTarget=at)
    cf, cf_events, _da = _run(d)
    assert cf.launch_summary()["n"] == 1
    assert cf.launch[0]["lag"] == pytest.approx(-0.7, abs=0.1)
    assert any(e.kind == "cf_launch" for e in cf_events)


def test_path_agreement_bins():
    """Constant curvature disagreement c at speed v -> RMS = c*v^2."""
    d = make_drive(60.0, vEgo=10.0)  # 22.4 mph -> the 20-35 bin
    d.channels["desiredCurvature"].v[:] = -0.005  # plan-left = +0.005, actual 0
    cf, _ev, _da = _run(d)
    assert cf.path_overall == pytest.approx(0.005 * 100.0, rel=0.02)
    assert len(cf.path_bins) == 1
    b = cf.path_bins[0]
    assert (b["lo_mph"], b["hi_mph"]) == (20, 35)
    assert b["rms"] == pytest.approx(0.5, rel=0.02)


def test_path_agreement_excludes_model_steering():
    d = make_drive(60.0, vEgo=10.0, enabled=True, latActive=True, longActive=True)
    d.channels["desiredCurvature"].v[:] = -0.005
    cf, _ev, _da = _run(d)
    assert cf.path_overall is None  # no manual-lat samples at all


def test_follow_opinion_gated_on_lead_source():
    n = int(60.0 / DT)
    common = dict(vEgo=15.0, leadStatus=True, leadVLead=15.0,
                  leadDRel=1.2 * 15.0 + 6.0,  # driver's effective gap 1.2 s
                  personality=np.zeros(n, np.int16))  # aggressive
    d = make_drive(60.0, planSource=np.full(n, 1, np.int16), **common)  # lead0
    cf, _ev, _da = _run(d, targets={"aggressive": 1.0, "standard": 1.45, "relaxed": 1.75})
    assert "aggressive" in cf.follow_opinion
    fo = cf.follow_opinion["aggressive"]
    assert fo["driver_median"] == pytest.approx(1.2, abs=0.02)
    assert fo["target"] == pytest.approx(1.0)
    assert fo["seconds"] > 30

    # same drive but e2e plan source -> excluded, with the explanatory note
    d2 = make_drive(60.0, planSource=np.full(n, 4, np.int16), **common)
    cf2, _ev2, _da2 = _run(d2)
    assert cf2.follow_opinion == {}
    assert "lead constraint" in cf2.follow_opinion_note


def test_unavailable_without_plan_channels():
    d = make_drive(30.0, vEgo=10.0)
    from opgrader.extract import Channel

    d.channels["desiredCurvature"] = Channel(np.array([]), np.array([]))
    d.channels["aTarget"] = Channel(np.array([]), np.array([]))
    d.channels["planSource"] = Channel(np.array([]), np.array([], dtype=np.int16))
    cf, _ev, _da = _run(d)
    assert cf.available is False
    assert "neither" in cf.why_unavailable
