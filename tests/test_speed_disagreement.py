"""Speed Disagreement: gas-override episodes, contexts, brake disengagements."""

import numpy as np
import pytest

from opgrader.events import build_arrays, detect_events
from opgrader.grading import CATEGORY_GROUPS, METRICS, grade
from opgrader.segments import segment_drive
from opgrader.speed_disagreement import (
    analyze_speed_disagreement,
    detect_brake_disengagements,
    detect_gas_overrides,
)
from tests.conftest import DT, make_drive


def _da(d):
    seg = segment_drive(d)
    assert seg is not None
    return seg, build_arrays(d, seg)


def _full(d):
    """Pipeline-style prep: events + speed-disagreement events appended."""
    seg, da = _da(d)
    events = detect_events(d, seg, da)
    sd = analyze_speed_disagreement([(d, seg, da, events)])
    events.extend(sd.events)
    return seg, da, events, sd


def _gas_drive(dur=60.0, gas_windows=(), **kw):
    n = int(dur / DT)
    t = np.arange(n) * DT
    gas = np.zeros(n, bool)
    for a, b in gas_windows:
        gas |= (t >= a) & (t < b)
    kw.setdefault("enabled", True)
    kw.setdefault("latActive", True)
    kw.setdefault("longActive", True)
    return make_drive(dur, gasPressed=gas, **kw), t


# ------------------------------------------------------------ episode rules


def test_episodes_merge_short_gaps():
    d, _t = _gas_drive(gas_windows=[(10, 12), (12.5, 14)])  # 0.5 s gap
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert len(eps) == 1
    assert eps[0].t0 == pytest.approx(10.0, abs=0.05)
    assert eps[0].t1 == pytest.approx(14.0, abs=0.05)


def test_episodes_not_merged_across_long_gap():
    d, _t = _gas_drive(gas_windows=[(10, 12), (14, 16)])  # 2 s gap
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert len(eps) == 2


def test_min_episode_duration():
    d, _t = _gas_drive(gas_windows=[(10, 10.2), (20, 20.4)])
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert len(eps) == 1  # 0.2 s tap dropped, 0.4 s press kept
    assert eps[0].t0 == pytest.approx(20.0, abs=0.05)


def test_episode_breaks_at_segment_time_gap():
    d, t = _gas_drive(gas_windows=[(10, 20)])
    for ch in d.channels.values():  # 5 s recording gap at t=15
        ch.t[ch.t >= 15.0] += 5.0
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert len(eps) == 2


def test_no_episode_when_gas_is_manual():
    # gas pressed while fully manual (not enabled) -> no override episodes
    d, _t = _gas_drive(gas_windows=[(10, 14)], longActive=False, enabled=False)
    _seg, da = _da(d)
    assert detect_gas_overrides(d.name, da) == []


def test_episode_detected_when_longactive_drops_during_press():
    """Real openpilot drops longActive during a gas override while staying
    ENABLED (verified on the owner's 0.11.2 Palisade logs) -- the episode
    must still be detected via the enabled&gasPressed union."""
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    gas = (t >= 20) & (t < 24)
    d = make_drive(dur, gasPressed=gas, enabled=True, latActive=True,
                   longActive=~gas)  # longActive clears for the press
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert len(eps) == 1
    assert eps[0].t0 == pytest.approx(20.0, abs=0.05)
    assert eps[0].t1 == pytest.approx(24.0, abs=0.05)
    # and the longActive False->True bounce is NOT a brake disengagement
    assert detect_brake_disengagements(d.name, da) == []


# ------------------------------------------------------------- context tags


def test_context_launch():
    dur = 40.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.where(t < 10, 0.0, np.minimum(0.5 * (t - 10), 12.0))
    d, _ = _gas_drive(dur, gas_windows=[(11, 13)], vEgo=v, standstill=v < 0.05)
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert len(eps) == 1
    assert eps[0].context == "launch"


def test_context_exp_slowdown():
    d, _t = _gas_drive(gas_windows=[(20, 22)], planVisA0=-0.8)
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert eps[0].context == "exp-slowdown"


def test_context_lead_pullaway():
    d, _t = _gas_drive(gas_windows=[(20, 22)], leadStatus=True,
                       leadDRel=20.0, leadVLead=16.0)  # vEgo 15 -> +1 m/s
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert eps[0].context == "lead-pullaway"


def test_context_free_road():
    d, _t = _gas_drive(gas_windows=[(20, 22)])  # leadStatus defaults False
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert eps[0].context == "free-road"


def test_context_other_steady_lead():
    d, _t = _gas_drive(gas_windows=[(20, 22)], leadStatus=True,
                       leadDRel=20.0, leadVLead=15.0)  # same speed, close
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert eps[0].context == "other"


# ------------------------------------------------- magnitude and follow-ups


def test_magnitude_aego_minus_plan():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    a = np.where((t >= 20) & (t < 24), 1.5, 0.0)
    d, _ = _gas_drive(dur, gas_windows=[(20, 24)], aEgo=a, planVisA0=0.5)
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert eps[0].magnitude == pytest.approx(1.0, abs=0.01)


def test_magnitude_none_without_vision_plan():
    d, _t = _gas_drive(gas_windows=[(20, 22)])
    del d.channels["planVisA0"]
    del d.channels["planVisDA"]
    _seg, da = _da(d)
    assert da.vis_accel is None
    eps = detect_gas_overrides(d.name, da)
    assert len(eps) == 1  # detection still works
    assert eps[0].magnitude is None  # insufficient, not a crash


def test_speed_taken_back_after_release():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.full(n, 15.0)
    up = (t >= 20) & (t < 25)
    v[up] = 15.0 + 1.0 * (t[up] - 20)  # override accelerates to 20
    down = (t >= 25) & (t < 30)
    v[down] = 20.0 - 1.0 * (t[down] - 25)  # model sheds it back to 15
    d, _ = _gas_drive(dur, gas_windows=[(20, 25)], vEgo=v)
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert eps[0].speed_taken_back == pytest.approx(5.0, abs=0.1)


def test_reoverride_pairing():
    d, _t = _gas_drive(gas_windows=[(10, 12), (17, 19), (50, 52)])
    _seg, da = _da(d)
    eps = detect_gas_overrides(d.name, da)
    assert len(eps) == 3
    assert eps[0].reoverride is True  # next press 5 s after release
    assert eps[1].reoverride is False  # next press 31 s later
    assert eps[2].reoverride is False  # nothing after


# ------------------------------------------------------ brake disengagement


def test_brake_disengagement_at_transition():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    d = make_drive(dur, enabled=t < 30, latActive=t < 30, longActive=t < 30,
                   brakePressed=t >= 29.8)
    _seg, da = _da(d)
    brakes = detect_brake_disengagements(d.name, da)
    assert len(brakes) == 1
    assert brakes[0].t == pytest.approx(30.0, abs=0.05)
    assert brakes[0].context == "free_road"  # constant 15 m/s, no lead


def test_brake_disengagement_lead_or_stop_context():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.full(n, 15.0)
    dec = (t >= 29) & (t < 34)
    v[dec] = 15.0 - 3.0 * (t[dec] - 29)
    v[t >= 34] = 0.0
    d = make_drive(dur, vEgo=v, enabled=t < 30, latActive=t < 30,
                   longActive=t < 30, brakePressed=(t >= 29) & (t < 35))
    _seg, da = _da(d)
    brakes = detect_brake_disengagements(d.name, da)
    assert len(brakes) == 1
    assert brakes[0].context == "lead_or_stop"  # a stop follows within 8 s


def test_no_brake_disengagement_without_brake():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    d = make_drive(dur, enabled=t < 30, latActive=t < 30, longActive=t < 30)
    _seg, da = _da(d)
    assert detect_brake_disengagements(d.name, da) == []


# --------------------------------------------------------------- aggregates


def test_analyze_rates_and_context_table():
    d, _t = _gas_drive(dur=600.0, gas_windows=[(20, 22), (100, 103)])
    seg, da = _da(d)
    events = detect_events(d, seg, da)
    sd = analyze_speed_disagreement([(d, seg, da, events)])
    assert sd.model_long_seconds == pytest.approx(600.0, abs=1.0)
    assert sd.overall_rate == pytest.approx(2.0, abs=0.05)  # 2 eps / 10 min
    assert sd.overall_pct == pytest.approx(100.0 * 5.0 / 600.0, abs=0.1)
    ctx = {r["context"]: r["n"] for r in sd.context_table}
    assert ctx["free-road"] == 2
    assert sd.biggest_context == "free-road"
    assert sd.reoverride_pct == pytest.approx(0.0)


# ----------------------------------------------------------------- grading


def _sd_stub(rate=None, pct=None, mag=None, secs=600.0, n_eps=0, n_mag_eps=0,
             have_gas=True, have_vis=True):
    from opgrader.speed_disagreement import (
        GasOverrideEpisode,
        SpeedDisagreementResult,
    )

    sd = SpeedDisagreementResult()
    sd.model_long_seconds = secs
    sd.have_gas = have_gas
    sd.have_vis = have_vis
    sd.overall_rate = rate
    sd.overall_pct = pct
    sd.overall_magnitude = mag
    sd.n_mag_episodes = n_mag_eps
    sd.n_mag_samples = n_mag_eps * 30
    sd.episodes = [
        GasOverrideEpisode("d", 0, 1, 0.0, 1.0, "other", mag, None, False,
                           "chill", "standard")
        for _ in range(n_eps)
    ]
    return sd


def _grade_with_sd(sd):
    samples = {m.key: {"model": [], "driver": []} for m in METRICS}
    return grade(samples, speed_disagreement_extra={"result": sd})


def test_anchor_scores():
    sd = _sd_stub(rate=2.0, pct=12.5, mag=1.1, n_eps=3, n_mag_eps=3)
    rep = _grade_with_sd(sd)
    res = {m.definition.key: m for c in rep.categories for m in c.metrics}
    assert res["gas_override_rate"].score == pytest.approx(75.0)  # (0,4,8)
    assert res["gas_override_pct"].score == pytest.approx(50 - 50 * 2.5 / 15)  # (0,10,25)
    assert res["gas_override_magnitude"].score == pytest.approx(45.0)  # (0.2,1,2)
    assert res["speed_taken_back"].score is None  # display-only
    assert res["reoverride_pct"].score is None
    cat = next(c for c in rep.categories if c.name == "Speed Disagreement")
    scored = [m.score for m in cat.metrics if m.score is not None]
    assert cat.score == pytest.approx(np.mean(scored))


def test_zero_override_rate_scores_100_with_enough_time():
    sd = _sd_stub(rate=0.0, pct=0.0, secs=600.0)
    rep = _grade_with_sd(sd)
    res = {m.definition.key: m for c in rep.categories for m in c.metrics}
    assert res["gas_override_rate"].score == pytest.approx(100.0)
    assert res["gas_override_pct"].score == pytest.approx(100.0)


def test_rate_gated_below_minimum_long_time():
    sd = _sd_stub(rate=0.0, pct=0.0, secs=60.0)  # < 120 s of model-long time
    rep = _grade_with_sd(sd)
    m = next(m for c in rep.categories for m in c.metrics
             if m.definition.key == "gas_override_rate")
    assert m.score is None


def test_magnitude_insufficient_below_three_episodes():
    sd = _sd_stub(rate=1.0, pct=1.0, mag=1.0, n_eps=2, n_mag_eps=2)
    rep = _grade_with_sd(sd)
    m = next(m for c in rep.categories for m in c.metrics
             if m.definition.key == "gas_override_magnitude")
    assert m.score is None
    assert m.model_agg == pytest.approx(1.0)  # still displayed


def test_magnitude_marked_unavailable_without_vision_plan():
    sd = _sd_stub(rate=1.0, pct=1.0, have_vis=False)
    rep = _grade_with_sd(sd)
    m = next(m for c in rep.categories for m in c.metrics
             if m.definition.key == "gas_override_magnitude")
    assert m.score is None
    assert "not in these logs" in m.definition.note


def test_long_weights_sum_to_one_with_new_category():
    weights = CATEGORY_GROUPS["Longitudinal"]
    assert "Speed Disagreement" in weights
    assert weights["Speed Disagreement"] == pytest.approx(0.15)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_bucket_stats_personality_attribution():
    dur = 300.0
    n = int(dur / DT)
    pers = np.full(n, 0, np.int16)  # aggressive throughout
    d, _t = _gas_drive(dur, gas_windows=[(20, 22)],
                       personality=pers, experimentalMode=True)
    _seg, _da2, _events, sd = _full(d)
    assert len(sd.episodes) == 1
    assert sd.episodes[0].personality == "aggressive"
    agg = sd.bucket_stats["aggressive"]
    assert agg.n_eps == 1
    assert agg.seconds == pytest.approx(300.0, abs=1.0)
    assert agg.rate == pytest.approx(2.0, abs=0.1)  # 1 episode / 5 min
    assert agg.pct == pytest.approx(100 * 2 / 300, abs=0.2)
    assert sd.bucket_stats["standard"].n_eps == 0
    assert sd.bucket_stats["experimental"].n_eps == 1
    assert sd.bucket_stats["chill"].n_eps == 0


def test_pipeline_grades_and_breakdowns_include_sd():
    from opgrader.pipeline import analyze

    dur = 300.0
    n = int(dur / DT)
    pers = np.full(n, 0, np.int16)
    d, _t = _gas_drive(dur, gas_windows=[(20, 22)], personality=pers)
    seg, da = _da(d)
    events = detect_events(d, seg, da)
    an = analyze([(d, seg, da, events)])
    cat = next(c for c in an.grades.categories if c.name == "Speed Disagreement")
    assert cat.extra.get("result") is an.speed_disagreement
    rate = next(m for m in cat.metrics if m.definition.key == "gas_override_rate")
    assert rate.model_agg == pytest.approx(2.0, abs=0.1)
    assert rate.score is not None
    # personality breakdown carries the episode
    agg = an.grades.breakdowns["personality"]["aggressive"]
    sd_cat = next(c for c in agg.categories if c.name == "Speed Disagreement")
    b_rate = next(m for m in sd_cat.metrics if m.definition.key == "gas_override_rate")
    assert b_rate.model_agg == pytest.approx(2.0, abs=0.1)
    assert b_rate.score is not None
