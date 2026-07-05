import numpy as np
import pytest

from opgrader.grading import (
    CATEGORY_GROUPS,
    METRICS,
    grade,
    letter,
    score_absolute,
    score_ratio,
)


def empty_samples():
    return {m.key: {"model": [], "driver": []} for m in METRICS}


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
    assert letter(95) == "A"
    assert letter(85) == "A-"
    assert letter(78) == "B+"
    assert letter(70) == "B"
    assert letter(60) == "C"
    assert letter(50) == "D"
    assert letter(49.9) == "F"


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
