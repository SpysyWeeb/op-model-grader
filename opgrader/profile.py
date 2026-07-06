"""Persistent local driver-baseline profile (~/.local/share/opgrader/profile.json).

"How you drive" doesn't reset between grading runs -- it's a property of you
and your car, not any one route. This module accumulates your MANUAL-driving
samples (the driver side of every ratio metric) across every route you've
ever graded, keyed by car fingerprint, and lets future grades borrow that
history to fill out thin per-route baselines (Launch, Turns, and low-speed
Ping-Pong bins are the ones that run dry on a single drive).

Storage is route-atomic: each stored route is a self-contained JSON blob
keyed by its route id, so re-grading the same route replaces its entry
(idempotent) and eviction drops whole routes, never partial ones.

What gets pooled (see poolable_metric_keys()): only MetricDefs that are a
genuine model-vs-driver comparison with a real human baseline --
needs_driver=True and scorer in ("ratio", "ratio_or_abs"). The allowlist is
DERIVED from the METRICS table (not hand-typed) so it stays correct as
metrics are added. Explicitly EXCLUDED, and why:
  - needs_driver=False / scorer in ("abs", "none"): no human baseline exists
    (rescue_rate, missed_turn_in) or it's a diagnostic, not a comparison --
    pooling a number with no human counterpart would be meaningless.
  - Follow adherence: scored against the PERSONALITY TARGET, not you --
    pooling it would blend unrelated targets across personalities.
  - Speed Disagreement: measures your OVERRIDE behavior itself (a property
    of how tolerant you are of the model, not a baseline quantity to
    compare the model against) -- pooling it would be circular.
  - Plan-vs-You counterfactuals: paired same-moment plan-vs-actual data, a
    different shape entirely (counterfactual.py owns that, unscored anyway).
Ping-Pong's per-speed-bin manual oscillation RMS / reversal rate ARE pooled
too, via a separate path (PINGPONG_RMS_KEY / PINGPONG_REV_KEY below) because
Ping-Pong is scored outside the METRICS table (time-weighted bin comparison,
not a per-event ratio metric).

Never skews: pooled data only ever extends the DRIVER side. The model side
is always exactly what you engaged THIS run -- pooling cannot make the
model look better or worse than it actually drove, only make the human
baseline it's compared against less noisy. A bin/metric with no pooled
history behaves EXACTLY as if profiling didn't exist.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grading import METRICS, MIN_EVENTS, add_turn_samples, collect_samples
from .lateral import PP_MIN_BIN_S, PingPongResult, analyze_pingpong

DATA_DIR = Path(os.environ.get("OPGRADER_DATA", "~/.local/share/opgrader")).expanduser()
PROFILE_FILE = DATA_DIR / "profile.json"

PROFILE_VERSION = 1
MAX_ROUTES_PER_FINGERPRINT = 60  # size/staleness bound, not a correctness requirement

# A route contributes a Ping-Pong bin value with even a little manual data in
# it (the whole point is thin per-route bins accumulating across routes);
# whether the COMBINED pool is enough to actually score against is a
# separate, later gate (MIN_EVENTS combined samples + real engaged data).
PP_POOL_MIN_SECONDS = 3.0

PINGPONG_RMS_KEY = "pingpong_osc_rms"
PINGPONG_REV_KEY = "pingpong_reversal_rate"


def poolable_metric_keys() -> frozenset[str]:
    """Genuine model-vs-driver ratio metrics, derived from METRICS."""
    return frozenset(
        m.key for m in METRICS if m.needs_driver and m.scorer in ("ratio", "ratio_or_abs")
    )


# --------------------------------------------------------------- store I/O


def _empty_store() -> dict:
    return {"_version": PROFILE_VERSION, "fingerprints": {}}


def load_store() -> dict:
    """Raw on-disk profile, version-guarded. Never raises."""
    try:
        raw = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_store()
    if not isinstance(raw, dict) or raw.get("_version") != PROFILE_VERSION:
        # Shape or field semantics may have changed since this was written --
        # blunt wipe rather than risk silently misinterpreting old data. Same
        # pattern as this fork's on-device drive_statsd.py ANALYZER_VERSION
        # guard (a version bump there clears ALL routes for full reanalysis).
        return _empty_store()
    raw.setdefault("fingerprints", {})
    return raw


def save_store(store: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROFILE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
    os.replace(tmp, PROFILE_FILE)


def delete_profile() -> bool:
    """True if a file was actually removed."""
    if PROFILE_FILE.exists():
        PROFILE_FILE.unlink()
        return True
    return False


def _prune(fp_entry: dict) -> None:
    """Drop the OLDEST whole routes once a fingerprint exceeds the cap.

    Routes with no wall_time_start (old logs without initData timing) can't
    be judged for staleness, so they're evicted first -- a defensible
    default since the cap is a staleness bound, not a correctness one.
    """
    routes = fp_entry.get("routes", {})
    if len(routes) <= MAX_ROUTES_PER_FINGERPRINT:
        return

    def key(item):
        _rid, r = item
        wt = r.get("wall_time_start")
        return wt if isinstance(wt, (int, float)) else -1.0

    ordered = sorted(routes.items(), key=key)
    drop = len(ordered) - MAX_ROUTES_PER_FINGERPRINT
    for rid, _r in ordered[:drop]:
        del routes[rid]


# ------------------------------------------------------------ pool building


def _finite(vals) -> list[float]:
    return [float(v) for v in vals if v is not None and np.isfinite(v)]


def _pp_bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}mph"


def _route_metrics_blob(drive, seg, da, events, turns, pp_score_fn) -> dict:
    """This one route's own poolable driver-side samples (route-atomic)."""
    rsamples, _rbuckets = collect_samples([(drive, seg, da, events)])
    add_turn_samples(rsamples, turns)
    blob: dict[str, dict[str, list[float]]] = {}
    for key in poolable_metric_keys():
        vals = _finite(rsamples.get(key, {}).get("driver", []))
        if vals:
            blob[key] = {"none": vals}

    rp = analyze_pingpong([(drive.name, seg, da)], pp_score_fn)
    if rp:
        for b in rp.bins:
            if b.manual_s < PP_POOL_MIN_SECONDS:
                continue
            label = _pp_bin_label(b.lo_mph, b.hi_mph)
            if b.manual_rms is not None:
                blob.setdefault(PINGPONG_RMS_KEY, {})[label] = [float(b.manual_rms)]
            if b.manual_rev is not None:
                blob.setdefault(PINGPONG_REV_KEY, {})[label] = [float(b.manual_rev)]
    return blob


@dataclass
class ProfileSummary:
    used: bool  # False when --no-profile / profiling disabled for this run
    fingerprints: dict[str, dict] = field(default_factory=dict)
    # fingerprints[fp] = {"n_routes": int, "date_range": (t0, t1) | None}

    @property
    def empty(self) -> bool:
        return not self.fingerprints

    def line_for(self, fp: str) -> str:
        info = self.fingerprints.get(fp)
        if not info:
            return f"{fp}: no profile yet"
        n = info["n_routes"]
        rng = info.get("date_range")
        if rng:
            t0, t1 = rng
            d0 = time.strftime("%Y-%m-%d", time.gmtime(t0))
            d1 = time.strftime("%Y-%m-%d", time.gmtime(t1))
            when = d0 if d0 == d1 else f"{d0} – {d1}"
            return f"{fp}: {n} route(s) pooled, {when}"
        return f"{fp}: {n} route(s) pooled"

    def lines(self) -> list[str]:
        if not self.used:
            return ["not used this run (--no-profile)"]
        if self.empty:
            return ["empty (no prior routes stored yet)"]
        return [self.line_for(fp) for fp in sorted(self.fingerprints)]


def _summarize(store: dict, fingerprints: list[str]) -> ProfileSummary:
    out = {}
    for fp in fingerprints:
        routes = store.get("fingerprints", {}).get(fp, {}).get("routes", {})
        if not routes:
            continue
        times = [r["wall_time_start"] for r in routes.values() if r.get("wall_time_start")]
        out[fp] = {
            "n_routes": len(routes),
            "date_range": (min(times), max(times)) if times else None,
        }
    return ProfileSummary(used=True, fingerprints=out)


def current_summary() -> ProfileSummary:
    """Snapshot of the whole stored profile (for the GUI panel, before any
    grading happens -- every fingerprint on disk, not just this run's)."""
    store = load_store()
    return _summarize(store, sorted(store.get("fingerprints", {})))


def _pool_pingpong(pp: PingPongResult, pooled: dict[str, dict[str, list[float]]], pp_score_fn) -> None:
    """Extend each bin's manual baseline with pooled history, IN PLACE.

    Only ever ADDS capability: a bin with no pooled history for its label is
    left exactly as analyze_pingpong scored it. The model/engaged side is
    never touched.
    """
    for b in pp.bins:
        label = _pp_bin_label(b.lo_mph, b.hi_mph)
        rms_pool = pooled.get(PINGPONG_RMS_KEY, {}).get(label, [])
        rev_pool = pooled.get(PINGPONG_REV_KEY, {}).get(label, [])
        b.pooled_n = len(rms_pool)
        if not rms_pool and not rev_pool:
            continue
        rms_vals = list(rms_pool)
        if b.manual_s >= PP_POOL_MIN_SECONDS and b.manual_rms is not None:
            rms_vals.append(float(b.manual_rms))
        rev_vals = list(rev_pool)
        if b.manual_s >= PP_POOL_MIN_SECONDS and b.manual_rev is not None:
            rev_vals.append(float(b.manual_rev))
        if len(rms_vals) < MIN_EVENTS or b.engaged_rms is None or b.engaged_s < PP_MIN_BIN_S:
            continue  # still not enough combined data, or no live model data to grade
        combined_rms = float(np.median(rms_vals))
        combined_rev = float(np.median(rev_vals)) if rev_vals else None
        s_rms = pp_score_fn(b.engaged_rms, combined_rms)
        s_rev = (
            pp_score_fn(b.engaged_rev, combined_rev)
            if (b.engaged_rev is not None and combined_rev is not None)
            else None
        )
        b.score = float(np.mean([s for s in (s_rms, s_rev) if s is not None]))
        b.pooled_manual_rms = combined_rms
        b.pooled_manual_rev = combined_rev

    scored = [b for b in pp.bins if b.score is not None]
    if scored:
        w = np.array([b.engaged_s for b in scored])
        s = np.array([b.score for b in scored])
        pp.score = float(np.sum(w * s) / np.sum(w))
        pp.worst_bin = min(scored, key=lambda b: b.score)


def pool_for_grading(an, per_drive, pp_score_fn, save: bool = True) -> tuple[ProfileSummary, dict]:
    """Enrich an.samples[<poolable key>]["driver"] and an.pingpong's bins
    with pooled profile history, IN PLACE, then upsert this run's own
    per-route contributions and persist. Returns (summary, profile_info)
    where profile_info feeds MetricResult provenance in grading.grade().

    Combined = this run's own driver_vals (an.samples, already the union of
    every route in THIS invocation) + every OTHER stored route's pooled
    values for the matching fingerprint(s). Routes already part of THIS run
    are excluded from the "other" pool, so re-grading a stored route can
    never double-count it -- combined counts don't inflate on re-grade.
    """
    store = load_store()  # pre-update state
    current_route_ids = {d.name for d, _s, _a, _e in per_drive}
    fingerprints = sorted({d.meta.car_fingerprint for d, _s, _a, _e in per_drive})

    turns_by_drive: dict[str, list] = {}
    for t in an.turns:
        turns_by_drive.setdefault(t.drive, []).append(t)

    # 1. this run's own per-route contributions (storage + provenance)
    route_updates: dict[str, dict[str, dict]] = {}
    for drive, seg, da, events in per_drive:
        fp = drive.meta.car_fingerprint
        blob = _route_metrics_blob(
            drive, seg, da, events, turns_by_drive.get(drive.name, []), pp_score_fn,
        )
        route_updates.setdefault(fp, {})[drive.name] = {
            "wall_time_start": drive.meta.wall_time_start,
            "metrics": blob,
        }

    # 2. pooled "other routes" values per fingerprint, key, bucket
    pooled: dict[str, dict[str, list[float]]] = {}
    for fp in fingerprints:
        fp_entry = store.get("fingerprints", {}).get(fp, {})
        for route_id, route_data in fp_entry.get("routes", {}).items():
            if route_id in current_route_ids:
                continue  # this run supersedes it; use the fresh data only
            for key, buckets in route_data.get("metrics", {}).items():
                for bucket, vals in buckets.items():
                    pooled.setdefault(key, {}).setdefault(bucket, []).extend(_finite(vals))

    # 3. combine into an.samples -- profile_info carries provenance for
    # grading.grade() to populate MetricResult.driver_vals_this_drive/etc.
    profile_info: dict[str, dict] = {}
    for key in poolable_metric_keys():
        this_drive = _finite(an.samples.get(key, {}).get("driver", []))
        pooled_vals = pooled.get(key, {}).get("none", [])
        if not this_drive and not pooled_vals:
            continue
        an.samples.setdefault(key, {"model": [], "driver": []})
        an.samples[key]["driver"] = this_drive + pooled_vals
        profile_info[key] = {"this_drive": this_drive, "pooled": pooled_vals}

    # 4. Ping-Pong bins (separate code path, not in METRICS)
    if an.pingpong is not None:
        _pool_pingpong(an.pingpong, pooled, pp_score_fn)

    # 5. upsert + prune + persist (route-atomic; never partial)
    if save:
        for fp, routes in route_updates.items():
            entry = store.setdefault("fingerprints", {}).setdefault(fp, {"routes": {}})
            entry.setdefault("routes", {}).update(routes)
            _prune(entry)
        store["_version"] = PROFILE_VERSION
        save_store(store)

    return _summarize(store, fingerprints), profile_info


# --------------------------------------------------------------------- CLI


def _fmt_date(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def describe_store(store: dict) -> list[str]:
    lines = []
    for fp, data in sorted(store.get("fingerprints", {}).items()):
        routes = data.get("routes", {})
        times = [r["wall_time_start"] for r in routes.values() if r.get("wall_time_start")]
        rng = f"{_fmt_date(min(times))} to {_fmt_date(max(times))}" if times else "dates unknown"
        lines.append(f"  {fp}: {len(routes)} route(s), {rng}")
    return lines


def clear_profile_cli(yes: bool = False, out=print, confirm=None) -> int:
    """Standalone action: describe what's stored, confirm, delete.

    confirm (if given) is called with no args and must return bool; used by
    tests to avoid real input(). Returns a process exit code.
    """
    store = load_store()
    fps = store.get("fingerprints", {})
    if not fps:
        out("no driver profile data to clear.")
        return 0
    out(f"This will permanently delete the local driver profile ({PROFILE_FILE}):")
    for line in describe_store(store):
        out(line)
    if not yes:
        ask = confirm or (lambda: input("Delete it? [y/N] ").strip().lower() in ("y", "yes"))
        if not ask():
            out("cancelled.")
            return 1
    delete_profile()
    out("driver profile deleted.")
    return 0
