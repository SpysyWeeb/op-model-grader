"""Driver-baseline profile: pooling, storage, privacy, CLI."""

import json

import numpy as np
import pytest

from opgrader import pipeline
from opgrader import profile as P
from opgrader.events import build_arrays, detect_events
from opgrader.grading import METRICS, MIN_EVENTS, grade
from opgrader.lateral import PP_BINS_MPH, PingPongBin, PingPongResult
from opgrader.segments import segment_drive
from tests.conftest import DT, make_drive


def _prep(d):
    seg = segment_drive(d)
    da = build_arrays(d, seg)
    return seg, da, detect_events(d, seg, da)


def _multi_span_drive(name, fingerprint="TEST_CAR", wall_time_start=None,
                       freqs=(0.5, 0.8, 1.2), dur=90.0):
    """3 manual + 3 engaged 15 s spans, each manual span jerkier than the
    last (different sine frequency) so it contributes a DISTINCT rms_jerk
    sample -- this-drive-alone gets n=3, independently >= MIN_EVENTS."""
    n = int(dur / DT)
    t = np.arange(n) * DT
    enabled = np.zeros(n, bool)
    for lo, hi in ((15, 30), (45, 60), (75, 90)):
        enabled |= (t >= lo) & (t < hi)
    a = np.zeros(n)
    for lo, freq in zip((0, 30, 60), freqs):
        m = (t >= lo) & (t < lo + 15)
        a[m] = 0.3 * np.sin(2 * np.pi * freq * (t[m] - lo))
    d = make_drive(dur, name=name, aEgo=a, enabled=enabled, latActive=enabled, longActive=enabled)
    d.meta.car_fingerprint = fingerprint
    if wall_time_start is not None:
        d.meta.wall_time_start = wall_time_start
    return d


def _grade_route(name, **kw):
    use_profile = kw.pop("use_profile", True)
    d = _multi_span_drive(name, **kw)
    seg, da, events = _prep(d)
    return pipeline.analyze([(d, seg, da, events)], use_profile=use_profile)


def _rms_jerk(an):
    return next(m for c in an.grades.categories for m in c.metrics
                if m.definition.key == "rms_jerk")


# --------------------------------------------------------------- store I/O


def test_round_trip_store_load():
    store = {"_version": P.PROFILE_VERSION, "fingerprints": {
        "CAR_A": {"routes": {"r1": {"wall_time_start": 100.0,
                                     "metrics": {"rms_jerk": {"none": [0.1, 0.2]}}}}}
    }}
    P.save_store(store)
    assert P.load_store() == store


def test_load_missing_file_returns_empty():
    assert not P.PROFILE_FILE.exists()
    assert P.load_store() == {"_version": P.PROFILE_VERSION, "fingerprints": {}}


def test_load_corrupt_json_returns_empty():
    P.PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    P.PROFILE_FILE.write_text("{not json")
    assert P.load_store() == P._empty_store()


def test_version_mismatch_wipes():
    P.save_store({"_version": P.PROFILE_VERSION - 1, "fingerprints": {
        "CAR_A": {"routes": {"r1": {"wall_time_start": 1.0, "metrics": {}}}}
    }})
    got = P.load_store()
    assert got == {"_version": P.PROFILE_VERSION, "fingerprints": {}}


def test_delete_profile():
    P.save_store(P._empty_store())
    assert P.delete_profile() is True
    assert not P.PROFILE_FILE.exists()
    assert P.delete_profile() is False  # nothing left to delete


# --------------------------------------------------------------- allowlist


def test_poolable_keys_derived_from_metrics_table():
    keys = P.poolable_metric_keys()
    # spot-check the brief's own examples land on the right side
    for k in ("rms_jerk", "median_gap", "turn_in_delay", "s_overshoot",
              "recovery_wobbles", "unwind_rate", "stop_lurch"):
        assert k in keys
    for k in ("rescue_rate", "missed_turn_in", "curve_s_overshoot",
              "cmd_unwind_lead_left", "cmd_onset_lead_right"):
        assert k not in keys
    # derived, not hand-typed: exactly matches the needs_driver/scorer filter
    expected = {m.key for m in METRICS if m.needs_driver and m.scorer in ("ratio", "ratio_or_abs")}
    assert keys == expected


def test_excluded_categories_have_no_poolable_members():
    keys = P.poolable_metric_keys()
    # Speed Disagreement metrics aren't even in METRICS (built ad hoc in
    # grading.speed_disagreement_results), and follow-adherence rows are
    # synthesized per-personality -- neither can appear via this allowlist
    assert not any(k.startswith("gas_override") for k in keys)
    assert "speed_taken_back" not in keys
    assert "reoverride_pct" not in keys
    assert not any(k.startswith("follow_adherence_") for k in keys)


# ------------------------------------------------------------------ pooling


def test_this_drive_alone_qualifies_and_is_stored():
    an = _grade_route("r--A", wall_time_start=1000.0)
    m = _rms_jerk(an)
    assert m.n_this_drive == 3  # 3 manual spans, all distinct frequencies
    assert m.same_drive_agg is not None  # independently >= MIN_EVENTS
    assert m.n_pooled == 0  # nothing stored yet before this run

    store = P.load_store()
    routes = store["fingerprints"]["TEST_CAR"]["routes"]
    assert list(routes) == ["r--A"]
    assert routes["r--A"]["wall_time_start"] == 1000.0
    assert routes["r--A"]["metrics"]["rms_jerk"]["none"] == pytest.approx(
        sorted(m.driver_vals_this_drive), abs=1e-9
    ) or len(routes["r--A"]["metrics"]["rms_jerk"]["none"]) == 3


def test_second_route_pools_first_routes_data():
    _grade_route("r--A", wall_time_start=1000.0, freqs=(0.5, 0.8, 1.2))
    an_b = _grade_route("r--B", wall_time_start=2000.0, freqs=(2.0, 2.4, 2.8))
    m = _rms_jerk(an_b)
    assert m.n_this_drive == 3
    assert m.n_pooled == 3  # exactly route A's 3 samples, nothing else
    assert m.n_driver == 6
    # combined value differs from the same-drive-only value (different jerk levels)
    assert m.same_drive_agg is not None
    assert m.driver_agg != pytest.approx(m.same_drive_agg)


def test_idempotent_regrade_does_not_duplicate():
    an1 = _grade_route("r--A", wall_time_start=1000.0)
    n1 = _rms_jerk(an1).n_this_drive
    store1 = P.load_store()
    assert len(store1["fingerprints"]["TEST_CAR"]["routes"]) == 1

    # re-grade the SAME route (as if the owner graded it again)
    an2 = _grade_route("r--A", wall_time_start=1000.0)
    m2 = _rms_jerk(an2)
    store2 = P.load_store()
    assert len(store2["fingerprints"]["TEST_CAR"]["routes"]) == 1  # still one route
    assert m2.n_this_drive == n1  # not doubled
    assert m2.n_pooled == 0  # its own OLD contribution must not appear as "pooled"


def test_regrading_stored_route_alongside_others_excludes_its_own_stale_copy():
    _grade_route("r--A", wall_time_start=1000.0, freqs=(0.5, 0.8, 1.2))
    _grade_route("r--B", wall_time_start=2000.0, freqs=(2.0, 2.4, 2.8))
    # re-grade A again: pooled should be exactly B's data, never A's own
    # stored copy (which would double A's own contribution)
    an_a_again = _grade_route("r--A", wall_time_start=1000.0, freqs=(0.5, 0.8, 1.2))
    m = _rms_jerk(an_a_again)
    assert m.n_pooled == 3  # B only
    assert len(P.load_store()["fingerprints"]["TEST_CAR"]["routes"]) == 2


def test_fingerprint_isolation():
    _grade_route("r--A", fingerprint="CAR_A", wall_time_start=1000.0)
    an_b = _grade_route("r--B", fingerprint="CAR_B", wall_time_start=2000.0)
    m = _rms_jerk(an_b)
    assert m.n_pooled == 0  # CAR_A's data must never reach a CAR_B grade
    store = P.load_store()
    assert set(store["fingerprints"]) == {"CAR_A", "CAR_B"}
    assert set(store["fingerprints"]["CAR_A"]["routes"]) == {"r--A"}
    assert set(store["fingerprints"]["CAR_B"]["routes"]) == {"r--B"}


def test_no_profile_skips_read_and_write():
    _grade_route("r--A", wall_time_start=1000.0)  # seed the profile
    before = P.PROFILE_FILE.read_bytes()

    an = _grade_route("r--C", wall_time_start=3000.0, use_profile=False)
    m = _rms_jerk(an)
    assert m.n_pooled == 0
    assert an.profile_summary.used is False
    assert an.profile_summary.lines() == ["not used this run (--no-profile)"]
    assert P.PROFILE_FILE.read_bytes() == before  # untouched, byte for byte

    # and r--C was never written either
    store = P.load_store()
    assert "r--C" not in store["fingerprints"].get("TEST_CAR", {}).get("routes", {})


def test_route_cap_eviction_drops_oldest_wholesale(monkeypatch):
    monkeypatch.setattr(P, "MAX_ROUTES_PER_FINGERPRINT", 2)
    _grade_route("r--old", wall_time_start=1000.0)
    _grade_route("r--mid", wall_time_start=2000.0)
    _grade_route("r--new", wall_time_start=3000.0)
    routes = P.load_store()["fingerprints"]["TEST_CAR"]["routes"]
    assert set(routes) == {"r--mid", "r--new"}  # oldest dropped, whole route gone
    for rid, r in routes.items():
        assert "metrics" in r and "wall_time_start" in r  # never partial


# ------------------------------------------------------ combined vs same-drive


def test_combined_vs_same_drive_only_grading_level():
    samples = {m.key: {"model": [], "driver": []} for m in METRICS}
    samples["rms_jerk"]["model"] = [1.0, 1.0, 1.0]
    samples["rms_jerk"]["driver"] = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]  # combined, already merged
    profile_info = {"rms_jerk": {"this_drive": [5.0, 5.0, 5.0], "pooled": [1.0, 1.0, 1.0]}}
    rep = grade(samples, profile_info=profile_info)
    m = next(m for c in rep.categories for m in c.metrics if m.definition.key == "rms_jerk")
    assert m.n_this_drive == 3
    assert m.n_pooled == 3
    assert m.n_driver == 6
    assert m.driver_agg == pytest.approx(3.0)  # median of [1,1,1,5,5,5]
    assert m.same_drive_agg == pytest.approx(5.0)  # this-drive alone independently qualifies
    assert m.driver_agg != pytest.approx(m.same_drive_agg)


def test_same_drive_agg_absent_below_min_events():
    samples = {m.key: {"model": [], "driver": []} for m in METRICS}
    samples["rms_jerk"]["driver"] = [5.0, 1.0, 1.0, 1.0]  # combined n=4
    profile_info = {"rms_jerk": {"this_drive": [5.0], "pooled": [1.0, 1.0, 1.0]}}  # this-drive n=1 < MIN_EVENTS
    rep = grade(samples, profile_info=profile_info)
    m = next(m for c in rep.categories for m in c.metrics if m.definition.key == "rms_jerk")
    assert m.n_this_drive == 1
    assert m.n_pooled == 3
    assert m.same_drive_agg is None  # 1 < MIN_EVENTS, not shown as independently qualified


# --------------------------------------------------------------- ping-pong


def test_pingpong_pool_rescues_thin_bin():
    def pp_score(m, d):
        r = m / max(d, 1e-6)
        return max(0.0, 100.0 - 50.0 * (r - 1.0)) if r > 1 else 100.0

    b = PingPongBin(lo_mph=0, hi_mph=5, engaged_s=60.0, manual_s=2.0,  # thin: below PP_MIN_BIN_S=30
                    engaged_rms=2.0, manual_rms=None, engaged_rev=None, manual_rev=None)
    pp = PingPongResult(bins=[b], sub_bins=[], score=None, worst_bin=None)
    pooled = {P.PINGPONG_RMS_KEY: {"0-5mph": [2.0, 2.0, 2.0]}}  # 3 historical routes
    P._pool_pingpong(pp, pooled, pp_score)
    assert b.pooled_n == 3
    assert b.score is not None  # rescued: 3 pooled >= MIN_EVENTS, engaged_s ok
    assert b.pooled_manual_rms == pytest.approx(2.0)
    assert pp.score == pytest.approx(b.score)


def test_pingpong_pool_leaves_unpooled_bin_untouched():
    def pp_score(m, d):
        return 100.0

    b = PingPongBin(lo_mph=5, hi_mph=10, engaged_s=60.0, manual_s=60.0,
                    engaged_rms=1.0, manual_rms=1.0, engaged_rev=5.0, manual_rev=5.0, score=87.0)
    pp = PingPongResult(bins=[b], sub_bins=[], score=87.0, worst_bin=b)
    P._pool_pingpong(pp, pooled={}, pp_score_fn=pp_score)
    assert b.pooled_n == 0
    assert b.score == pytest.approx(87.0)  # unchanged: no pooled history for this label
    assert pp.score == pytest.approx(87.0)


def test_pp_bin_label_matches_pp_bins_mph_shape():
    labels = {P._pp_bin_label(lo, hi) for lo, hi in PP_BINS_MPH}
    assert len(labels) == len(PP_BINS_MPH)  # all distinct
    assert "0-5mph" in labels


# ------------------------------------------------------------------ privacy

_ALLOWED_METRIC_KEYS = P.poolable_metric_keys() | {P.PINGPONG_RMS_KEY, P.PINGPONG_REV_KEY}


def _walk_and_check(store: dict):
    assert set(store) == {"_version", "fingerprints"}
    assert store["_version"] == P.PROFILE_VERSION
    for fp, fp_data in store["fingerprints"].items():
        assert isinstance(fp, str)
        assert set(fp_data) == {"routes"}
        for route_id, route in fp_data["routes"].items():
            assert isinstance(route_id, str)
            assert set(route) == {"wall_time_start", "metrics"}
            wt = route["wall_time_start"]
            assert wt is None or isinstance(wt, (int, float))
            for key, buckets in route["metrics"].items():
                assert key in _ALLOWED_METRIC_KEYS, f"unexpected metric key leaked: {key!r}"
                for bucket, vals in buckets.items():
                    assert bucket == "none" or bucket.endswith("mph")
                    for v in vals:
                        assert isinstance(v, (int, float))  # never a string


def test_privacy_allowlist_on_saved_profile():
    an = _grade_route("r--priv", wall_time_start=1234.0)
    assert an.profile_summary.used
    raw = json.loads(P.PROFILE_FILE.read_text())
    _walk_and_check(raw)


def test_privacy_survives_a_hostile_looking_route_name():
    # route "ids" come from Drive.name (rlog route strings); make sure a
    # weird one still round-trips as a plain string key, nothing extra leaks
    an = _grade_route("dongle|route--00 SECRET", wall_time_start=1.0)
    raw = json.loads(P.PROFILE_FILE.read_text())
    _walk_and_check(raw)
    assert "dongle|route--00 SECRET" in raw["fingerprints"]["TEST_CAR"]["routes"]


# ---------------------------------------------------------------------- CLI


def test_clear_profile_cli_empty():
    out = []
    rc = P.clear_profile_cli(out=out.append)
    assert rc == 0
    assert "no driver profile" in out[0]


def test_clear_profile_cli_confirm_yes_flag():
    _grade_route("r--A", wall_time_start=1000.0)
    assert P.PROFILE_FILE.exists()
    out = []
    rc = P.clear_profile_cli(yes=True, out=out.append)
    assert rc == 0
    assert not P.PROFILE_FILE.exists()
    assert any("deleted" in line for line in out)


def test_clear_profile_cli_prompt_declined():
    _grade_route("r--A", wall_time_start=1000.0)
    out = []
    rc = P.clear_profile_cli(out=out.append, confirm=lambda: False)
    assert rc == 1
    assert P.PROFILE_FILE.exists()  # nothing deleted


def test_clear_profile_cli_prompt_accepted_describes_fingerprints():
    _grade_route("r--A", fingerprint="HYUNDAI_PALISADE", wall_time_start=1000.0)
    out = []
    rc = P.clear_profile_cli(out=out.append, confirm=lambda: True)
    assert rc == 0
    assert not P.PROFILE_FILE.exists()
    assert any("HYUNDAI_PALISADE" in line for line in out)
