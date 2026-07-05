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
    """Manual stop: driver decel (-1.5) starts at t=10; the vision plan's
    accel drops below -0.5 at (driver onset + plan_lag_s)."""
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
    # centered 0.31 s MA over the step: crossing when 1/3 of the window is
    # past the step -> t = 10 - h + 2h/3 = 10 - h/3 with h = 0.155
    t_driver = 10.0 - 0.155 / 3
    vis = np.zeros(n)
    if plan_lag_s is not None:
        vis[t >= t_driver + plan_lag_s] = -2.0
    return make_drive(dur, vEgo=v, aEgo=a, brakePressed=dec, standstill=v < 0.1,
                      leadStatus=with_lead, leadDRel=30.0 if with_lead else 0.0,
                      leadVLead=np.maximum(v - 1.0, 0.0), planVisA0=vis)


def test_braking_onset_lag_with_lead():
    cf, cf_events, _da = _run(_stop_drive(0.3))
    bs = cf.braking_summary()
    assert bs["lead"]["n"] == 1
    assert bs["lead"]["median"] == pytest.approx(0.3, abs=0.08)
    assert bs["nolead"]["n"] == 0
    assert any(e.kind == "cf_brake" and e.values["lead"] for e in cf_events)


def test_braking_onset_no_lead_red_light():
    """Leadless stop (red light): the vision plan brakes 0.4 s EARLIER."""
    cf, cf_events, _da = _run(_stop_drive(-0.4, with_lead=False))
    bs = cf.braking_summary()
    assert bs["nolead"]["n"] == 1
    assert bs["nolead"]["median"] == pytest.approx(-0.4, abs=0.08)
    assert bs["lead"]["n"] == 0
    ev = next(e for e in cf_events if e.kind == "cf_brake")
    assert ev.values["lead"] is False


def test_braking_plan_never_brakes_counts_per_bucket():
    cf, _ev, _da = _run(_stop_drive(None, with_lead=False))
    bs = cf.braking_summary()
    assert bs["nolead"]["never"] == 1 and bs["nolead"]["n"] == 0
    cf2, _ev2, _da2 = _run(_stop_drive(None, with_lead=True))
    assert cf2.braking_summary()["lead"]["never"] == 1


def test_mpc_diagnostic_lag():
    d = _stop_drive(0.0)
    t = np.arange(int(60.0 / DT)) * DT
    at = np.zeros(len(t))
    at[t >= 11.0] = -2.0  # MPC brakes ~0.9 s after the driver
    d.channels["aTarget"].v[:] = at
    cf, _ev, _da = _run(d)
    bs = cf.braking_summary()
    assert bs["mpc_n"] == 1
    # MPC at 11.0 vs driver crossing at ~9.95
    assert bs["mpc_median"] == pytest.approx(1.05, abs=0.1)


def test_launch_onset_plan_earlier():
    """Manual pull-away: driver responds 1.5 s after lead moves, vision plan
    +0.3 crossing at 0.8 s -> lag -0.7."""
    dur = 40.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.zeros(n)
    a = np.zeros(n)
    go = t >= 11.5
    v[go] = np.minimum(1.0 * (t[go] - 11.5), 8.0)
    a[go] = np.where(v[go] < 8.0, 1.0, 0.0)
    v_lead = np.where(t >= 10.0, 1.0, 0.0)
    vis = np.where(t >= 10.8, 1.0, 0.0)

    d = make_drive(dur, vEgo=v, aEgo=a, leadStatus=True, leadDRel=8.0,
                   leadVLead=v_lead, standstill=v < 0.1, planVisA0=vis)
    cf, cf_events, _da = _run(d)
    ls = cf.launch_summary()
    assert ls["lead"]["n"] == 1
    assert ls["lead"]["median"] == pytest.approx(-0.7, abs=0.1)
    assert any(e.kind == "cf_launch" for e in cf_events)


def test_launch_onset_no_lead_green_light():
    """No lead: plan rises at 19.5; first motion (v > 0.15) at 20.15 -> lag -0.65."""
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.zeros(n)
    a = np.zeros(n)
    go = t >= 20.0  # first motion at t=20
    v[go] = np.minimum(1.0 * (t[go] - 20.0), 10.0)
    a[go] = np.where(v[go] < 10.0, 1.0, 0.0)
    vis = np.where(t >= 19.5, 1.0, 0.0)

    d = make_drive(dur, vEgo=v, aEgo=a, standstill=v < 0.1, planVisA0=vis)
    cf, _ev, _da = _run(d)
    ls = cf.launch_summary()
    assert ls["nolead"]["n"] == 1
    assert ls["nolead"]["median"] == pytest.approx(-0.65, abs=0.06)


def test_speed_opinion_and_accel_rms():
    """Model plans 90% of the driver's speed -> 'wants 10% slower'."""
    n = int(60.0 / DT)
    d = make_drive(60.0, vEgo=10.0, planVisV4=9.0, planVisA0=0.3)
    cf, _ev, _da = _run(d)
    assert "free" in cf.speed_opinion
    so = cf.speed_opinion["free"]
    assert so["median_ratio"] == pytest.approx(0.9, abs=0.01)
    assert so["pct"] == pytest.approx(-10.0, abs=1.0)
    assert "lead" not in cf.speed_opinion
    assert cf.accel_rms == pytest.approx(0.3, abs=0.02)  # planned 0.3 vs actual 0


def test_speed_opinion_excludes_stop_runup():
    """The 8 s before a standstill are excluded from X2."""
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.full(n, 10.0)
    dec = (t >= 30) & (t < 35)
    v[dec] = 10.0 - 2.0 * (t[dec] - 30)
    v[t >= 35] = 0.0
    d = make_drive(dur, vEgo=v, planVisV4=9.0, standstill=v < 0.1)
    cf, _ev, _da = _run(d)
    so = cf.speed_opinion.get("free")
    assert so is not None
    # v>5 lasts until t=32.5 (32.5 s of samples); the exclusion removes
    # [26.85, 34.85] (8 s before v<0.3 at ~34.85), leaving ~26.9 s
    assert 20.0 < so["seconds"] < 28.5


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


def test_unavailable_without_plan_channels():
    d = make_drive(30.0, vEgo=10.0)
    from opgrader.extract import Channel

    for ch in ("desiredCurvature", "aTarget", "planSource",
               "planVisA0", "planVisDA", "planVisV4"):
        d.channels[ch] = Channel(np.array([]), np.array([]))
    cf, _ev, _da = _run(d)
    assert cf.available is False
    assert "neither" in cf.why_unavailable
