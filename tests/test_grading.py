import numpy as np
import pytest

from opgrader.grading import (
    CATEGORY_GROUPS,
    METRIC_BY_KEY,
    METRICS,
    add_turn_samples,
    grade,
    letter,
    score_absolute,
    score_ratio,
)
from opgrader.lateral import TurnEpisode


def empty_samples():
    return {m.key: {"model": [], "driver": []} for m in METRICS}


def _ep(engaged=True, sharp=True, contaminated=False, rescued=False,
        never_commanded=False, cmd_onset_lead=None, cmd_unwind_lead=None,
        side="left", i=(0, 1)):
    """Minimal TurnEpisode for add_turn_samples wiring tests."""
    return TurnEpisode(
        engaged=engaged, drive="synth", side=side, sharp=sharp,
        band="90-150" if sharp else "20-90", i0=i[0], i1=i[1],
        t_onset=0.0, v_onset=5.0, peak_act=120.0 if side == "left" else -120.0,
        peak_cmd=None, t_peak_act=1.0, contaminated=contaminated, rescued=rescued,
        never_commanded=never_commanded, cmd_onset_lead=cmd_onset_lead,
        cmd_unwind_lead=cmd_unwind_lead,
    )


def test_score_ratio_anchor_points():
    assert score_ratio(1.0, 1.0) == 100.0
    assert score_ratio(0.5, 1.0) == 100.0  # better than you caps at 100
    assert score_ratio(1.5, 1.0) == 75.0
    assert score_ratio(2.0, 1.0) == 50.0
    assert score_ratio(3.0, 1.0) == 25.0
    assert score_ratio(4.0, 1.0) == 0.0
    assert score_ratio(10.0, 1.0) == 0.0


def test_score_ratio_higher_better():
    assert score_ratio(2.0, 1.0, better="higher") == 100.0
    assert score_ratio(0.5, 1.0, better="higher") == 50.0


def test_score_ratio_eps_floor():
    # driver at 0 with a sane eps floor: model tiny value still scores well
    assert score_ratio(0.05, 0.0, eps=0.1) == 100.0
    assert score_ratio(0.2, 0.0, eps=0.1) == 50.0


def test_score_absolute():
    anchors = (0.0, 25.0, 50.0)
    assert score_absolute(0.0, anchors) == 100.0
    assert score_absolute(12.5, anchors) == 75.0
    assert score_absolute(25.0, anchors) == 50.0
    assert score_absolute(50.0, anchors) == 0.0
    assert score_absolute(80.0, anchors) == 0.0


def test_letters():
    assert letter(100.0) == "S"
    assert letter(95) == "A"
    assert letter(85) == "A-"
    assert letter(78) == "B+"
    assert letter(70) == "B"
    assert letter(60) == "C"
    assert letter(50) == "D"
    assert letter(49.9) == "F"


def test_s_grade_survives_float_summation_noise_but_not_real_near_perfect():
    # a weighted mean of several exact-100.0 scores can land a hair under
    # 100.0 in double precision -- S must still trigger. A genuinely
    # non-perfect score close to 100 (a real, slightly-off ratio) must not.
    almost_by_fp_noise = sum(100.0 * w for w in (0.30, 0.20, 0.20, 0.17, 0.13))
    assert letter(almost_by_fp_noise) == "S"
    assert letter(99.97) != "S"  # e.g. score_ratio(1.0006, 1.0) -- really not perfect


def test_insufficient_data_needs_three_per_side():
    s = empty_samples()
    s["rms_jerk"]["model"] = [1.0, 1.0, 1.0]
    s["rms_jerk"]["driver"] = [1.0, 1.0]  # only 2
    rep = grade(s)
    m = next(
        m for c in rep.categories for m in c.metrics if m.definition.key == "rms_jerk"
    )
    assert m.score is None


def test_weights_renormalize_when_categories_empty():
    # only Smoothness has data -> Longitudinal score == Smoothness score,
    # Lateral group has no data -> overall == Longitudinal score
    s = empty_samples()
    for key in ("rms_jerk", "p95_jerk", "accel_reversals", "pct_hard_accel"):
        s[key]["model"] = [2.0, 2.0, 2.0]
        s[key]["driver"] = [1.0, 1.0, 1.0]  # model 2x worse -> 50 each
    rep = grade(s)
    lon = next(g for g in rep.groups if g.name == "Longitudinal")
    lat = next(g for g in rep.groups if g.name == "Lateral")
    assert lon.score == pytest.approx(50.0)
    assert lat.score is None
    assert rep.overall_score == pytest.approx(50.0)
    assert rep.overall_letter == "D"


def test_two_groups_average():
    s = empty_samples()
    for key in ("rms_jerk", "p95_jerk", "accel_reversals", "pct_hard_accel"):
        s[key]["model"] = [1.0] * 3
        s[key]["driver"] = [1.0] * 3  # 100
    rep = grade(s, pingpong_score=50.0)
    lon = next(g for g in rep.groups if g.name == "Longitudinal")
    lat = next(g for g in rep.groups if g.name == "Lateral")
    assert lon.score == pytest.approx(100.0)
    assert lat.score == pytest.approx(50.0)  # only Ping-Pong, renormalized
    assert rep.overall_score == pytest.approx(75.0)


def test_rescue_rate_absolute_no_driver_needed():
    s = empty_samples()
    s["rescue_rate"]["model"] = [100.0, 0.0, 0.0, 0.0]  # 25% of turns rescued
    rep = grade(s)
    m = next(
        m for c in rep.categories for m in c.metrics if m.definition.key == "rescue_rate"
    )
    assert m.model_agg == pytest.approx(25.0)
    assert m.score == pytest.approx(50.0)


def test_overshoot_absolute_fallback_when_driver_is_clean():
    s = empty_samples()
    s["s_overshoot"]["model"] = [20.0, 20.0, 20.0]
    s["s_overshoot"]["driver"] = [0.5, 0.5, 0.5]  # human ~0 -> absolute scale
    rep = grade(s)
    m = next(
        m for c in rep.categories for m in c.metrics if m.definition.key == "s_overshoot"
    )
    assert m.score == pytest.approx(50.0)  # 20% overshoot on the absolute scale


def test_diagnostic_metrics_never_scored():
    s = empty_samples()
    s["cmd_unwind_lead_left"]["model"] = [0.5] * 10
    rep = grade(s)
    m = next(
        m
        for c in rep.categories
        for m in c.metrics
        if m.definition.key == "cmd_unwind_lead_left"
    )
    assert m.score is None
    assert m.model_agg == pytest.approx(0.5)


def test_category_weights_sum_to_one_per_group():
    for grp, cats in CATEGORY_GROUPS.items():
        assert sum(cats.values()) == pytest.approx(1.0)


def test_score_ratio_match_penalizes_both_directions():
    # style metrics: model must not be rewarded for e.g. tailgating harder
    assert score_ratio(1.0, 1.0, better="match") == 100.0
    assert score_ratio(2.0, 1.0, better="match") == 50.0
    assert score_ratio(0.5, 1.0, better="match") == 50.0  # half your gap = same penalty as double
    assert score_ratio(0.25, 1.0, better="match") == 0.0


# ------------------------------------------------------ Turn-In Timing (blinker-free)


def test_turn_in_delay_retired():
    assert "turn_in_delay" not in METRIC_BY_KEY
    assert not any(m.key == "turn_in_delay" for m in METRICS)


def test_cmd_onset_lead_scored_metric_defs():
    for key in ("cmd_onset_lead_left", "cmd_onset_lead_right"):
        d = METRIC_BY_KEY[key]
        assert d.category == "Turn-In Timing"
        assert d.scorer == "abs"
        assert d.needs_driver is False
        assert d.abs_anchors == (0.0, 0.5, 1.5)


def test_missed_turn_in_counts_only_engaged_sharp_never_commanded():
    turns = [
        _ep(engaged=True, sharp=True, never_commanded=True),   # counts: 100
        _ep(engaged=True, sharp=True, never_commanded=False),  # counts: 0
        _ep(engaged=True, sharp=False, never_commanded=True),  # curve turn: excluded
        _ep(engaged=False, sharp=True, never_commanded=True),  # driver side: excluded
    ]
    samples = empty_samples()
    add_turn_samples(samples, turns)
    assert samples["missed_turn_in"]["model"] == [100.0, 0.0]


def test_missed_turn_in_not_gated_by_contaminated():
    """Contamination (the driver forcing the wheel) is exactly the mechanism
    by which "the model never commanded this turn" shows up in practice --
    gating missed_turn_in on it would exclude the cases it exists to catch."""
    turns = [
        _ep(engaged=True, sharp=True, never_commanded=True, contaminated=True),
        _ep(engaged=True, sharp=True, never_commanded=True, rescued=True),
    ]
    samples = empty_samples()
    add_turn_samples(samples, turns)
    assert samples["missed_turn_in"]["model"] == [100.0, 100.0]


def test_cmd_onset_lead_excluded_when_contaminated_or_rescued():
    """cmd_onset_lead mirrors cmd_unwind_lead's existing exclusion: a
    contaminated/rescued episode's cmd-vs-act timing comparison is noisy
    (the driver's own torque, not the model's onset behavior)."""
    turns = [
        _ep(engaged=True, cmd_onset_lead=0.4, contaminated=False, rescued=False, side="left"),
        _ep(engaged=True, cmd_onset_lead=0.9, contaminated=True, side="left"),
        _ep(engaged=True, cmd_onset_lead=0.7, rescued=True, side="left"),
    ]
    samples = empty_samples()
    add_turn_samples(samples, turns)
    assert samples["cmd_onset_lead_left"]["model"] == [0.4]


def test_cmd_onset_lead_anchor_scores():
    samples = empty_samples()
    samples["cmd_onset_lead_left"]["model"] = [-0.3, -0.3, -0.3]  # negative -> clipped to 100
    samples["cmd_onset_lead_right"]["model"] = [0.25, 0.25, 0.25]  # halfway to the 50-point anchor
    rep = grade(samples)
    res = {m.definition.key: m for c in rep.categories for m in c.metrics}
    assert res["cmd_onset_lead_left"].score == pytest.approx(100.0)
    assert res["cmd_onset_lead_right"].score == pytest.approx(75.0)


def test_missed_turn_in_anchor_scoring_unchanged_shape():
    samples = empty_samples()
    samples["missed_turn_in"]["model"] = [100.0, 0.0, 0.0, 0.0]  # 25% missed
    rep = grade(samples)
    m = next(m for c in rep.categories for m in c.metrics if m.definition.key == "missed_turn_in")
    assert m.model_agg == pytest.approx(25.0)
    assert m.score == pytest.approx(50.0)
