"""Score the model against the driver and assign letter grades.

For each metric we aggregate (median) the per-event/per-span values on each
side. With m = model aggregate and d = driver aggregate, and lower-is-better:

    r = m / max(d, eps)
    score = 100                  if r <= 1
    score = 100 - 50*(r-1)       if 1 < r <= 2     (2x worse than you = 50)
    score = max(0, 50 - 25*(r-2))if r > 2

Higher-is-better metrics invert the ratio (none currently, but supported).
Each metric needs n >= MIN_EVENTS samples on both sides, else it is marked
"insufficient data" and excluded. eps is a per-metric floor chosen at a
scale where differences stop being meaningful (so a driver value of ~0
doesn't turn a negligible model value into an F).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .events import DriveArrays, Event
from .extract import Drive
from .segments import Segmentation

MIN_EVENTS = 3

CATEGORY_WEIGHTS = {
    "Smoothness": 0.25,
    "Following": 0.20,
    "Stopping": 0.20,
    "Launch": 0.15,
    "Responsiveness": 0.10,
    "Lateral": 0.10,
}

LETTER_BINS = [
    (93, "A"),
    (85, "A-"),
    (78, "B+"),
    (70, "B"),
    (60, "C"),
    (50, "D"),
]


def letter(score: float) -> str:
    for cutoff, lt in LETTER_BINS:
        if score >= cutoff:
            return lt
    return "F"


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    category: str
    unit: str
    better: str = "lower"  # "lower" | "higher"
    eps: float = 1e-6


METRICS: list[MetricDef] = [
    # Smoothness (per span)
    MetricDef("rms_jerk", "RMS jerk", "Smoothness", "m/s³", eps=0.02),
    MetricDef("p95_jerk", "P95 |jerk|", "Smoothness", "m/s³", eps=0.05),
    MetricDef("accel_reversals", "Accel reversals", "Smoothness", "/min", eps=0.2),
    MetricDef("pct_hard_accel", "Time |accel| > 2", "Smoothness", "%", eps=0.1),
    # Following (per follow window)
    MetricDef("median_gap", "Median time gap", "Following", "s", eps=0.2),
    MetricDef("gap_hunting", "Gap hunting (detrended std)", "Following", "s", eps=0.05),
    MetricDef("follow_reversals", "Accel reversals while following", "Following", "/min", eps=0.2),
    # Stopping (per stop approach)
    MetricDef("peak_decel", "Peak decel", "Stopping", "m/s²", eps=0.1),
    MetricDef("peak_decel_frac", "Peak-decel timing (fraction of approach)", "Stopping", "", eps=0.05),
    MetricDef("stop_lurch", "Stop lurch (max |jerk|, last 2 s)", "Stopping", "m/s³", eps=0.05),
    MetricDef("accel_at_crawl", "|Accel| at 0.2 m/s", "Stopping", "m/s²", eps=0.05),
    # Launch (per launch)
    MetricDef("time_to_5", "Time to 5 m/s", "Launch", "s", eps=0.2),
    MetricDef("launch_peak_jerk", "Peak |jerk| in launch", "Launch", "m/s³", eps=0.05),
    # Responsiveness (per stimulus)
    MetricDef("lead_decel_latency", "Lead-decel response latency", "Responsiveness", "s", eps=0.1),
    MetricDef("pullaway_latency", "Lead pull-away latency", "Responsiveness", "s", eps=0.1),
    # Lateral (per span)
    MetricDef("rms_lat_jerk", "RMS lateral jerk", "Lateral", "m/s³", eps=0.02),
    MetricDef("steer_rate_rms", "Steering rate RMS (>10 m/s)", "Lateral", "deg/s", eps=0.5),
    MetricDef("steer_reversals", "Steering reversals (>10 m/s)", "Lateral", "/min", eps=0.5),
    MetricDef("pct_high_lat", "Time |lat accel| > 3", "Lateral", "%", eps=0.1),
]

METRIC_BY_KEY = {m.key: m for m in METRICS}


def score_ratio(m: float, d: float, better: str = "lower", eps: float = 1e-6) -> float:
    """Score 0..100 from model aggregate m vs driver aggregate d."""
    m = max(float(m), 0.0)
    d = max(float(d), 0.0)
    if better == "higher":
        m, d = d, m
    r = m / max(d, eps)
    if r <= 1.0:
        return 100.0
    if r <= 2.0:
        return 100.0 - 50.0 * (r - 1.0)
    return max(0.0, 50.0 - 25.0 * (r - 2.0))


@dataclass
class MetricResult:
    definition: MetricDef
    model_vals: list[float]
    driver_vals: list[float]
    model_agg: float | None = None
    driver_agg: float | None = None
    score: float | None = None  # None => insufficient data

    @property
    def n_model(self) -> int:
        return len(self.model_vals)

    @property
    def n_driver(self) -> int:
        return len(self.driver_vals)


@dataclass
class CategoryResult:
    name: str
    weight: float
    metrics: list[MetricResult] = field(default_factory=list)
    score: float | None = None

    @property
    def letter(self) -> str | None:
        return letter(self.score) if self.score is not None else None


@dataclass
class GradeReport:
    categories: list[CategoryResult]
    overall_score: float | None
    overall_letter: str | None


# --------------------------------------------------------------- collection


def _clean(vals) -> list[float]:
    return [float(v) for v in vals if v is not None and np.isfinite(v)]


def collect_samples(
    per_drive: list[tuple[Drive, Segmentation, DriveArrays, list[Event]]],
) -> dict[str, dict[str, list[float]]]:
    """Per-metric sample lists: {metric_key: {"model": [...], "driver": [...]}}.

    Model samples come from engaged spans/events; events contaminated by a
    driver override (gas/brake while engaged) are excluded from the model
    side. Lateral model samples exclude steeringPressed moments.
    """
    samples: dict[str, dict[str, list[float]]] = {
        m.key: {"model": [], "driver": []} for m in METRICS
    }

    def add(key: str, side: str, value: float) -> None:
        if value is not None and np.isfinite(value):
            samples[key][side].append(float(value))

    for drive, seg, da, events in per_drive:
        # ---- span-based: smoothness + lateral
        for span in seg.spans:
            side = "model" if span.kind == "engaged" else "driver"
            sl = slice(span.i0, span.i1)
            t = da.t[sl]
            keep = ~da.override[sl] if side == "model" else np.ones(span.i1 - span.i0, bool)
            if keep.sum() < 10:
                continue

            a_s = da.a_smooth[sl]
            j = M.derivative(t, a_s)
            add("rms_jerk", side, M.rms(j[keep]))
            add("p95_jerk", side, M.p95_abs(j[keep]))
            add("accel_reversals", side, M.sign_reversals_per_min(t, a_s))
            add("pct_hard_accel", side, M.pct_time_above(t[keep], np.abs(da.a[sl][keep]), 2.0))

            # lateral: engaged side additionally excludes steeringPressed
            lat_keep = keep.copy()
            if side == "model" and da.steering_pressed is not None:
                lat_keep &= ~da.steering_pressed[sl]
            if lat_keep.sum() >= 10 and da.steering_rate is not None:
                lm = M.lateral_metrics(
                    t[lat_keep],
                    da.v[sl][lat_keep],
                    da.steering_rate[sl][lat_keep],
                    da.lat_accel[sl][lat_keep] if da.lat_accel is not None else None,
                )
                for k in ("rms_lat_jerk", "steer_rate_rms", "steer_reversals", "pct_high_lat"):
                    if k in lm:
                        add(k, side, lm[k])

        # ---- event-based
        for ev in events:
            side = "model" if ev.engaged else "driver"
            if ev.engaged and ev.has_override:
                continue  # driver interfered; not the model's behavior
            sl = slice(ev.i0, ev.i1)
            t, v, a = da.t[sl], da.v[sl], da.a[sl]
            if ev.kind == "stop":
                sm = M.stop_metrics(t, v, a, ev.values.get("t_standstill"))
                add("peak_decel", side, sm["peak_decel"])
                add("peak_decel_frac", side, sm["peak_decel_frac"])
                add("stop_lurch", side, sm["stop_lurch"])
                add("accel_at_crawl", side, sm["accel_at_crawl"])
                ev.values.update(sm)
            elif ev.kind == "launch":
                lm = M.launch_metrics(t, v, a, ev.values["t_first_motion"])
                add("time_to_5", side, lm["time_to_5"])
                add("launch_peak_jerk", side, lm["peak_jerk"])
                ev.values.update(lm)
            elif ev.kind == "follow":
                fm = M.follow_metrics(t, v, a, da.d_rel[sl])
                add("median_gap", side, fm["median_gap"])
                add("gap_hunting", side, fm["gap_hunting"])
                add("follow_reversals", side, fm["accel_reversals"])
                ev.values.update(fm)
            elif ev.kind == "lead_decel":
                add("lead_decel_latency", side, ev.values["latency"])
            elif ev.kind == "pullaway":
                add("pullaway_latency", side, ev.values["latency"])

    return samples


# ------------------------------------------------------------------ grading


def grade(samples: dict[str, dict[str, list[float]]]) -> GradeReport:
    categories: dict[str, CategoryResult] = {
        name: CategoryResult(name, w) for name, w in CATEGORY_WEIGHTS.items()
    }

    for mdef in METRICS:
        mv = _clean(samples.get(mdef.key, {}).get("model", []))
        dv = _clean(samples.get(mdef.key, {}).get("driver", []))
        res = MetricResult(mdef, mv, dv)
        if len(mv) >= MIN_EVENTS and len(dv) >= MIN_EVENTS:
            res.model_agg = float(np.median(mv))
            res.driver_agg = float(np.median(dv))
            res.score = score_ratio(res.model_agg, res.driver_agg, mdef.better, mdef.eps)
        else:
            res.model_agg = float(np.median(mv)) if mv else None
            res.driver_agg = float(np.median(dv)) if dv else None
        categories[mdef.category].metrics.append(res)

    for cat in categories.values():
        scored = [m.score for m in cat.metrics if m.score is not None]
        cat.score = float(np.mean(scored)) if scored else None

    valid = [c for c in categories.values() if c.score is not None]
    if valid:
        total_w = sum(c.weight for c in valid)
        overall = sum(c.score * c.weight for c in valid) / total_w
        return GradeReport(list(categories.values()), overall, letter(overall))
    return GradeReport(list(categories.values()), None, None)
