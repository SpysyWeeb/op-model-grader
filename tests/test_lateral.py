import numpy as np
import pytest

from opgrader.events import build_arrays
from opgrader.grading import score_ratio
from opgrader.lateral import (
    analyze_pingpong,
    detect_intent_windows,
    detect_turn_episodes,
    highpass_angle,
    swing_reversal_count,
)
from opgrader.segments import segment_drive
from opgrader.vehicle_model import vehicle_model_from_params
from tests.conftest import DT, make_drive


def _prep(drive):
    seg = segment_drive(drive)
    assert seg is not None
    return seg, build_arrays(drive, seg)


def _t(dur):
    return np.arange(int(dur / DT)) * DT


def test_swing_reversal_count_sine():
    t = _t(10)
    x = 5.0 * np.sin(2 * np.pi * 1.0 * t)  # 10 periods -> ~20 direction changes
    c = swing_reversal_count(x, thresh=3.0)
    assert 18 <= c <= 20


def test_swing_reversals_ignore_small_wobble():
    t = _t(10)
    x = 1.0 * np.sin(2 * np.pi * 1.0 * t)  # swings of 2 < 3 deg threshold
    assert swing_reversal_count(x, thresh=3.0) == 0


def test_highpass_keeps_wobble_drops_ramp():
    t = _t(30)
    ramp = 2.0 * t  # slow maneuver trend
    wob = 4.0 * np.sin(2 * np.pi * 1.5 * t)
    resid = highpass_angle(t, ramp + wob)
    # trend removed, oscillation mostly kept
    assert abs(np.mean(resid[500:-500])) < 0.3
    assert np.std(resid[500:-500]) == pytest.approx(np.std(wob), rel=0.2)


def _turn_profile(dur=30.0, peak=120.0, overshoot=-24.0):
    """0 -> peak (t 2..5) -> 0 (t 5..9, 30 deg/s) -> overshoot lobe -> 0."""
    t = _t(dur)
    act = np.zeros(len(t))
    up = (t >= 2) & (t < 5)
    act[up] = peak * (t[up] - 2) / 3
    down = (t >= 5) & (t < 9)
    act[down] = peak * (1 - (t[down] - 5) / 4)
    lobe = (t >= 9) & (t < 11)
    act[lobe] = overshoot * np.sin(np.pi * (t[lobe] - 9) / 2)
    return t, act


def test_turn_episode_overshoot_and_unwind():
    _t_, act = _turn_profile()
    d = make_drive(30.0, vEgo=5.0, steeringAngleDeg=act)
    seg, da = _prep(d)
    eps = detect_turn_episodes("synth", seg, da)
    assert len(eps) == 1
    ep = eps[0]
    assert ep.engaged is False
    assert ep.side == "left"
    assert ep.peak_act == pytest.approx(120.0, abs=1.0)
    assert ep.sharp is True  # peak >= 90 and onset speed 5 m/s < 6.7
    assert ep.band == "90-150"
    assert ep.overshoot_pct == pytest.approx(20.0, abs=1.0)
    assert ep.overshoot_deg == pytest.approx(24.0, abs=1.0)
    # spec: peak / (time from peak to 20 deg) = 120 / (100/30) = 36 deg/s
    assert ep.unwind_rate == pytest.approx(36.0, rel=0.05)
    assert ep.wobbles == 0


def test_turn_recovery_wobble_counted():
    t = _t(30)
    act = np.zeros(len(t))
    up = (t >= 2) & (t < 5)
    act[up] = 120.0 * (t[up] - 2) / 3
    down = (t >= 5) & (t < 9)
    act[down] = 120.0 * (1 - (t[down] - 5) / 4)
    lobe1 = (t >= 9) & (t < 10.5)  # overshoot: -24 deg
    act[lobe1] = -24.0 * np.sin(np.pi * (t[lobe1] - 9) / 1.5)
    lobe2 = (t >= 10.5) & (t < 12)  # recovery wobble back past zero: +15 deg
    act[lobe2] = 15.0 * np.sin(np.pi * (t[lobe2] - 10.5) / 1.5)

    d = make_drive(30.0, vEgo=5.0, steeringAngleDeg=act)
    seg, da = _prep(d)
    eps = detect_turn_episodes("synth", seg, da)
    assert len(eps) == 1
    assert eps[0].wobbles == 1


def test_turn_episode_not_sharp_at_speed():
    _t_, act = _turn_profile()
    d = make_drive(30.0, vEgo=10.0, steeringAngleDeg=act)  # > 6.7 m/s at onset
    seg, da = _prep(d)
    eps = detect_turn_episodes("synth", seg, da)
    assert len(eps) == 1
    assert eps[0].sharp is False


def test_turn_rescue_detected():
    t, act = _turn_profile()
    pressed = (t >= 6) & (t < 7)  # driver grabs the wheel during unwind
    d = make_drive(30.0, vEgo=5.0, steeringAngleDeg=act, steeringPressed=pressed,
                   enabled=True, latActive=True, longActive=True)
    seg, da = _prep(d)
    eps = detect_turn_episodes("synth", seg, da)
    assert len(eps) == 1
    assert eps[0].engaged is True
    assert eps[0].rescued is True
    assert eps[0].contaminated is False  # pressed only after the peak


def test_cmd_onset_lead_computed():
    """cmd ramps the same shape as act but starting 0.4 s later -> the
    onset-phase lead (mirrors cmd_unwind_lead, just at the 20 deg crossing
    instead of the 50%-of-peak fall) should read ~+0.4 s."""
    t, act = _turn_profile()  # act: 0->120 over [2,5) -> crosses 20 deg at t=2.5
    cmd = np.zeros(len(t))
    up = (t >= 2.4) & (t < 5.4)
    cmd[up] = 120.0 * (t[up] - 2.4) / 3.0
    down = (t >= 5.4) & (t < 9.4)
    cmd[down] = 120.0 * (1 - (t[down] - 5.4) / 4.0)

    d = make_drive(30.0, vEgo=5.0, steeringAngleDeg=act, ccSteeringAngleDeg=cmd,
                   enabled=True, latActive=True, longActive=True)
    seg, da = _prep(d)
    eps = detect_turn_episodes("synth", seg, da)
    assert len(eps) == 1
    ep = eps[0]
    assert ep.engaged is True
    assert ep.peak_act == pytest.approx(120.0, abs=1.0)
    assert ep.never_commanded is False  # cmd reached 120, well past the 20 deg bar
    assert ep.cmd_onset_lead is not None
    assert ep.cmd_onset_lead == pytest.approx(0.4, abs=0.05)


def test_never_commanded_when_cmd_stays_low():
    """act peaks at 120 deg (a sharp turn physically happened) but cmd never
    exceeds 15 deg -- the model's own plan never called for this turn."""
    t, act = _turn_profile()
    cmd = np.zeros(len(t))
    cmd[(t >= 2.0) & (t < 9.0)] = 15.0  # constant, mild, never sharp

    d = make_drive(30.0, vEgo=5.0, steeringAngleDeg=act, ccSteeringAngleDeg=cmd,
                   enabled=True, latActive=True, longActive=True)
    seg, da = _prep(d)
    eps = detect_turn_episodes("synth", seg, da)
    assert len(eps) == 1
    ep = eps[0]
    assert ep.peak_act == pytest.approx(120.0, abs=1.0)
    assert ep.peak_cmd == pytest.approx(15.0, abs=0.5)
    assert ep.never_commanded is True
    assert ep.cmd_onset_lead is None  # never computed: peak_cmd never reached 20 deg


def test_never_commanded_false_when_not_engaged():
    """A manual (not engaged) episode never sets never_commanded -- there's
    no model command to compare against."""
    t, act = _turn_profile()
    d = make_drive(30.0, vEgo=5.0, steeringAngleDeg=act)  # no enabled/latActive
    seg, da = _prep(d)
    eps = detect_turn_episodes("synth", seg, da)
    assert len(eps) == 1
    assert eps[0].engaged is False
    assert eps[0].never_commanded is False
    assert eps[0].cmd_onset_lead is None


def test_intent_window_turn_delay():
    dur = 40.0
    t = _t(dur)
    blink = (t >= 2) & (t < 15)
    # device yaw z is right-positive; a LEFT turn integrates to -90 deg raw
    yaw = np.where((t >= 4) & (t < 12), -np.radians(90) / 8, 0.0)
    act = np.zeros(len(t))
    up = (t >= 4) & (t < 6)
    act[up] = 60.0 * (t[up] - 4) / 2  # crosses 20 deg at t = 4 + 2/3
    act[(t >= 6) & (t < 12)] = 60.0
    down = (t >= 12) & (t < 14)
    act[down] = 60.0 * (1 - (t[down] - 12) / 2)

    d = make_drive(dur, vEgo=5.0, steeringAngleDeg=act, leftBlinker=blink, yawRate=yaw)
    seg, da = _prep(d)
    ws = detect_intent_windows("synth", seg, da)
    assert len(ws) == 1
    w = ws[0]
    assert w.engaged is False
    assert w.side == "left"
    assert w.outcome == "turn"
    assert w.heading_deg == pytest.approx(90.0, abs=2.0)
    assert w.delay == pytest.approx(2.0 + 2.0 / 3.0, abs=0.1)
    assert w.missed is False


def test_intent_missed_when_driver_intervenes():
    dur = 40.0
    t = _t(dur)
    blink = (t >= 2) & (t < 15)
    yaw = np.where((t >= 4) & (t < 12), -np.radians(90) / 8, 0.0)
    act = np.zeros(len(t))
    up = (t >= 4) & (t < 6)
    act[up] = 60.0 * (t[up] - 4) / 2
    act[(t >= 6) & (t < 12)] = 60.0
    pressed = t >= 3.5  # driver forces the turn before any 20 deg crossing

    d = make_drive(dur, vEgo=5.0, steeringAngleDeg=act, leftBlinker=blink,
                   yawRate=yaw, steeringPressed=pressed,
                   enabled=True, latActive=True, longActive=True)
    seg, da = _prep(d)
    ws = detect_intent_windows("synth", seg, da)
    assert len(ws) == 1
    assert ws[0].engaged is True
    assert ws[0].missed is True
    assert ws[0].delay is None


def test_intent_lane_change_classified():
    dur = 40.0
    t = _t(dur)
    blink = (t >= 2) & (t < 8)
    yaw = np.where((t >= 3) & (t < 4), np.radians(8), 0.0)  # 8 deg net
    d = make_drive(dur, vEgo=5.0, leftBlinker=blink, yawRate=yaw)
    seg, da = _prep(d)
    ws = detect_intent_windows("synth", seg, da)
    assert len(ws) == 1
    assert ws[0].outcome == "lane_change"


def test_pingpong_bins_and_worst():
    dur = 140.0
    t = _t(dur)
    engaged = t < 70
    angle = np.where(
        engaged,
        8.0 * np.sin(2 * np.pi * 0.8 * t),
        4.0 * np.sin(2 * np.pi * 0.8 * t),
    )
    d = make_drive(dur, vEgo=2.2, steeringAngleDeg=angle,
                   enabled=engaged, latActive=engaged, longActive=engaged)
    seg, da = _prep(d)
    pp = analyze_pingpong([("synth", seg, da)], lambda m, dd: score_ratio(m, dd, "lower", 0.05))
    assert pp is not None
    b0 = pp.bins[0]  # 0-5 mph (2.2 m/s = 4.9 mph)
    assert b0.engaged_s > 30 and b0.manual_s > 30
    assert b0.engaged_rms == pytest.approx(2 * b0.manual_rms, rel=0.1)
    # same frequency both sides -> similar reversal rates (~2*0.8*60 /min)
    assert b0.engaged_rev == pytest.approx(96.0, rel=0.15)
    assert b0.manual_rev == pytest.approx(96.0, rel=0.15)
    assert b0.score is not None
    assert pp.score == pytest.approx(b0.score)
    assert pp.worst_bin is b0
    assert len(pp.worst_windows) == 3
    # 1 mph sub-bins: 60 s of engaged time at 4.9 mph
    assert any(sb.lo_mph == 4 for sb in pp.sub_bins)


def test_commanded_angle_roundtrip():
    vm = vehicle_model_from_params(
        {
            "mass": 2200.0,
            "wheelbase": 2.9,
            "centerToFront": 1.3,
            "steerRatio": 16.0,
            "steerRatioRear": 0.0,
            "tireStiffnessFront": 190000.0,
            "tireStiffnessRear": 250000.0,
        }
    )
    assert vm is not None
    sa = np.radians(45.0)
    curv = vm.calc_curvature(sa, 10.0)
    back = vm.get_steer_from_curvature(curv, 10.0)
    assert back == pytest.approx(sa, rel=1e-9)
    assert vehicle_model_from_params({}) is None
    assert vehicle_model_from_params({"mass": 0.0}) is None
