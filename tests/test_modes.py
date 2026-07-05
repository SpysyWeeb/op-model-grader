"""Mode/personality tracking, breakdown buckets, follow adherence, config."""

import json

import numpy as np
import pytest

from opgrader.config import (
    DEFAULT_T_FOLLOW,
    parse_t_follow_flag,
    resolve_t_follow,
)
from opgrader.events import build_arrays, detect_events, mode_tag, personality_tag
from opgrader.grading import collect_samples, grade
from opgrader.metrics import effective_t_follow
from opgrader.pipeline import _bucket_times, _follow_adherence
from opgrader.segments import segment_drive
from tests.conftest import DT, make_drive


def _prep(d):
    seg = segment_drive(d)
    da = build_arrays(d, seg)
    return seg, da, detect_events(d, seg, da)


# ------------------------------------------------------------- tagging rules


def test_mode_and_personality_tags_constancy_rule():
    exp = np.zeros(1000, bool)
    exp[:50] = True  # 5% disagreement -> still "chill"
    assert mode_tag(exp, 0, 1000) == "chill"
    exp[:200] = True  # 20% -> mixed
    assert mode_tag(exp, 0, 1000) == "mixed"
    assert mode_tag(~np.zeros(1000, bool), 0, 1000) == "experimental"
    assert mode_tag(None, 0, 1000) == "unknown"

    pers = np.full(1000, 2, np.int16)
    pers[:80] = 0  # 8% -> still relaxed
    assert personality_tag(pers, 0, 1000) == "relaxed"
    pers[:300] = 0  # 30% -> mixed
    assert personality_tag(pers, 0, 1000) == "mixed"
    assert personality_tag(np.full(10, -1, np.int16), 0, 10) == "unknown"


def test_midspan_personality_flip_buckets_span_samples():
    """One engaged span, personality flips halfway -> both buckets get a
    smoothness sample; overall model pool gets one per span."""
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    pers = np.where(t < 30, 0, 2).astype(np.int16)  # aggressive -> relaxed
    a = 0.3 * np.sin(2 * np.pi * 0.2 * t)

    d = make_drive(dur, aEgo=a, enabled=True, latActive=True, longActive=True,
                   personality=pers, experimentalMode=True)
    seg, da, events = _prep(d)
    samples, buckets = collect_samples([(d, seg, da, events)])

    assert len(samples["rms_jerk"]["model"]) == 1  # one span overall
    assert len(buckets["rms_jerk"]["aggressive"]) == 1
    assert len(buckets["rms_jerk"]["relaxed"]) == 1
    assert len(buckets["rms_jerk"]["standard"]) == 0
    assert len(buckets["rms_jerk"]["experimental"]) == 1
    assert len(buckets["rms_jerk"]["chill"]) == 0


def test_event_with_flip_tagged_mixed_and_excluded_from_buckets():
    """A stop whose window straddles a mode flip is 'mixed': kept overall,
    excluded from every bucket."""
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
    exp = t >= 17.0  # flips experimental inside the stop window (~14.7..20.5)

    d = make_drive(dur, vEgo=v, aEgo=a, standstill=v < 0.1,
                   enabled=True, latActive=True, longActive=True,
                   experimentalMode=exp)
    seg, da, events = _prep(d)
    stops = [e for e in events if e.kind == "stop"]
    assert len(stops) == 1
    assert stops[0].values["mode"] == "mixed"
    assert stops[0].values["personality"] == "standard"  # constant default

    samples, buckets = collect_samples([(d, seg, da, events)])
    assert len(samples["stop_lurch"]["model"]) == 1  # kept in overall
    assert len(buckets["stop_lurch"]["chill"]) == 0
    assert len(buckets["stop_lurch"]["experimental"]) == 0
    assert len(buckets["stop_lurch"]["standard"]) == 1  # personality was constant


def test_bucket_grades_against_shared_baseline():
    from opgrader.grading import METRICS, grade_breakdowns

    # Two Smoothness metrics with identical shapes: a category needs
    # >= MIN_SCORED_FOR_CATEGORY scored metrics before it gets a grade (see
    # grading.MIN_SCORED_FOR_CATEGORY), so a single populated metric must not
    # by itself produce a bucket score.
    samples = {m.key: {"model": [], "driver": []} for m in METRICS}
    buckets = {m.key: {b: [] for b in ("chill", "experimental",
                                       "aggressive", "standard", "relaxed")} for m in METRICS}
    for key in ("rms_jerk", "p95_jerk"):
        samples[key]["driver"] = [1.0, 1.0, 1.0]
        buckets[key]["chill"] = [1.0, 1.0, 1.0]  # matches you -> 100
        buckets[key]["experimental"] = [2.0, 2.0, 2.0]  # 2x -> 50
        buckets[key]["aggressive"] = [1.0, 1.0]  # n=2 -> gated

    bd = grade_breakdowns(samples, buckets)
    assert bd["mode"]["chill"].score == pytest.approx(100.0)
    assert bd["mode"]["experimental"].score == pytest.approx(50.0)
    agg = bd["personality"]["aggressive"]
    assert agg.score is None  # only gated metrics -> no score
    m = next(m for c in agg.categories for m in c.metrics if m.definition.key == "rms_jerk")
    assert m.n_model == 2 and m.score is None


def test_bucket_single_scored_metric_insufficient():
    """A bucket with only ONE scored metric must not get a category grade."""
    from opgrader.grading import METRICS, grade_breakdowns

    samples = {m.key: {"model": [], "driver": []} for m in METRICS}
    buckets = {m.key: {b: [] for b in ("chill", "experimental",
                                       "aggressive", "standard", "relaxed")} for m in METRICS}
    samples["rms_jerk"]["driver"] = [1.0, 1.0, 1.0]
    buckets["rms_jerk"]["chill"] = [1.0, 1.0, 1.0]  # would be 100 alone

    bd = grade_breakdowns(samples, buckets)
    chill = bd["mode"]["chill"]
    m = next(mr for c in chill.categories for mr in c.metrics if mr.definition.key == "rms_jerk")
    assert m.score == pytest.approx(100.0)  # the metric itself still scores
    smoothness = next(c for c in chill.categories if c.name == "Smoothness")
    assert smoothness.score is None  # but the category doesn't, with only 1
    assert chill.score is None  # and nothing rolls up to the bucket grade


# --------------------------------------------------------- follow adherence


def _steady_follow_drive(t_follow: float, personality: int, v: float = 20.0, dur=60.0):
    """Lead held at exactly the MPC distance for t_follow at speed v."""
    n = int(dur / DT)
    d_rel = t_follow * v + 6.0  # equal speeds: dynamic term = 0
    return make_drive(
        dur, vEgo=v, leadStatus=True, leadDRel=d_rel, leadVLead=v,
        enabled=True, latActive=True, longActive=True,
        personality=np.full(n, personality, np.int16), experimentalMode=True,
    )


def test_adherence_inversion_exact_target():
    d = _steady_follow_drive(1.45, personality=1)
    seg, da, events = _prep(d)
    adherence = _follow_adherence([(d, seg, da, events)])
    assert "standard" in adherence
    info = adherence["standard"]
    assert info["median_eff"] == pytest.approx(1.45, abs=0.01)
    assert info["seconds"] > 30
    # error vs target ~0 -> scores 100
    from opgrader.grading import METRICS

    rep = grade({m.key: {"model": [], "driver": []} for m in METRICS},
                adherence=adherence, t_follow_targets=DEFAULT_T_FOLLOW)
    row = next(m for c in rep.categories for m in c.metrics
               if m.definition.key == "follow_adherence_standard")
    assert row.score == pytest.approx(100.0)
    assert row.driver_agg == pytest.approx(1.45)  # the target in the You column


def test_adherence_thirty_percent_off():
    d = _steady_follow_drive(1.3, personality=0)  # aggressive target 1.0
    seg, da, events = _prep(d)
    adherence = _follow_adherence([(d, seg, da, events)])
    info = adherence["aggressive"]
    assert info["median_eff"] == pytest.approx(1.3, abs=0.01)
    targets = {"aggressive": 1.0, "standard": 1.45, "relaxed": 2.0}
    pct = abs(info["median_eff"] - targets["aggressive"]) / 1.0 * 100
    assert pct == pytest.approx(30.0, abs=1.5)
    from opgrader.grading import METRICS
    rep = grade({m.key: {"model": [], "driver": []} for m in METRICS},
                adherence=adherence, t_follow_targets=targets)
    row = next(m for c in rep.categories for m in c.metrics
               if m.definition.key == "follow_adherence_aggressive")
    # 30% error on (5, 25, 50) anchors -> between 50 and 0: 50 - 50*(30-25)/25 = 40
    assert row.score == pytest.approx(40.0, abs=3.0)


def test_adherence_transients_are_filtered():
    """Approach transients (big vRel) must not pollute the median."""
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    v = np.full(n, 20.0)
    vl = np.where(t < 30, 25.0, 20.0)  # lead pulling away for 30 s, then steady
    d_rel = np.where(t < 30, 70.0, 1.45 * 20.0 + 6.0)
    d = make_drive(dur, vEgo=v, leadStatus=True, leadDRel=d_rel, leadVLead=vl,
                   enabled=True, latActive=True, longActive=True,
                   personality=np.full(n, 1, np.int16))
    seg, da, events = _prep(d)
    adherence = _follow_adherence([(d, seg, da, events)])
    assert adherence["standard"]["median_eff"] == pytest.approx(1.45, abs=0.02)


def test_bucket_times_split():
    dur = 60.0
    n = int(dur / DT)
    t = np.arange(n) * DT
    pers = np.where(t < 20, 0, 2).astype(np.int16)
    exp = t >= 40
    d = make_drive(dur, enabled=True, latActive=True, longActive=True,
                   personality=pers, experimentalMode=exp)
    seg, da, events = _prep(d)
    bt = _bucket_times([(d, seg, da, events)])
    assert bt["aggressive"] == pytest.approx(20.0, abs=0.5)
    assert bt["relaxed"] == pytest.approx(40.0, abs=0.5)
    assert bt["chill"] == pytest.approx(40.0, abs=0.5)
    assert bt["experimental"] == pytest.approx(20.0, abs=0.5)


# ------------------------------------------------------------------- config


def test_parse_t_follow_flag_lenient():
    assert parse_t_follow_flag("aggressive=1.0,standard=1.45,relaxed=2.0") == {
        "aggressive": 1.0, "standard": 1.45, "relaxed": 2.0}
    assert parse_t_follow_flag(" agg = 1.1 ; rel : 1.9 ") == {
        "aggressive": 1.1, "relaxed": 1.9}
    with pytest.raises(ValueError):
        parse_t_follow_flag("bogus=1.0")
    with pytest.raises(ValueError):
        parse_t_follow_flag("aggressive=99")
    with pytest.raises(ValueError):
        parse_t_follow_flag("aggressive")


def test_config_file_and_flag_precedence(tmp_path, monkeypatch):
    import opgrader.config as cfg

    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    # defaults when nothing stored
    assert resolve_t_follow(None) == DEFAULT_T_FOLLOW
    # file overrides defaults
    cfg.set_t_follow({"aggressive": 1.0})
    got = resolve_t_follow(None)
    assert got["aggressive"] == 1.0 and got["standard"] == 1.45
    # flag overrides file
    got = resolve_t_follow("aggressive=1.2")
    assert got["aggressive"] == 1.2
    # garbage in the file is ignored
    (tmp_path / "config.json").write_text(json.dumps({"t_follow": {"aggressive": "lol", "relaxed": 99}}))
    got = resolve_t_follow(None)
    assert got == DEFAULT_T_FOLLOW


def test_effective_t_follow_formula():
    v = np.array([20.0, 20.0, 0.1])
    vl = np.array([20.0, 15.0, 0.1])
    d = np.array([1.45 * 20 + 6.0, 1.0 * 20 + 6.0 + (400 - 225) / 5.0, 10.0])
    eff = effective_t_follow(v, vl, d)
    assert eff[0] == pytest.approx(1.45)
    assert eff[1] == pytest.approx(1.0)
    assert np.isnan(eff[2])  # v too low
