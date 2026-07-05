"""Score the model against the driver and assign letter grades.

Two top-level groups, each with its own headline grade:
- Longitudinal: Smoothness 0.30, Following 0.20, Stopping 0.20,
  Launch 0.17, Responsiveness 0.13
- Lateral: Ping-Pong 0.40, Turn Execution 0.30, Turn-In Timing 0.20,
  General Smoothness 0.10
Overall = 0.5*Longitudinal + 0.5*Lateral (renormalized if a group has no data).

Relative (ratio) scoring, lower-is-better unless stated: with m = model
aggregate, d = driver aggregate:

    r = m / max(d, eps)
    score = 100                   if r <= 1
    score = 100 - 50*(r-1)        if 1 < r <= 2     (2x worse than you = 50)
    score = max(0, 50 - 25*(r-2)) if r > 2

Higher-is-better metrics invert the ratio. eps is a per-metric floor at the
scale where differences stop being meaningful. Metrics without a human
counterpart (driver-rescue rate, missed turn-ins) or whose human baseline is
~zero (S-curve overshoot) use an absolute anchor scale instead, documented
per metric. Aggregation is the median of per-event values (mean for
rate-of-events metrics). A ratio metric needs >= MIN_EVENTS samples per side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .events import DriveArrays, Event
from .extract import Drive
from .segments import Segmentation

MIN_EVENTS = 3

GROUP_WEIGHTS = {"Longitudinal": 0.5, "Lateral": 0.5}

CATEGORY_GROUPS = {
    "Longitudinal": {
        "Smoothness": 0.30,
        "Following": 0.20,
        "Stopping": 0.20,
        "Launch": 0.17,
        "Responsiveness": 0.13,
    },
    "Lateral": {
        "Ping-Pong": 0.40,
        "Turn Execution": 0.30,
        "Turn-In Timing": 0.20,
        "General Smoothness": 0.10,
    },
}

LETTER_BINS = [(93, "A"), (85, "A-"), (78, "B+"), (70, "B"), (60, "C"), (50, "D")]


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
    agg: str = "median"  # "median" | "mean"
    scorer: str = "ratio"  # "ratio" | "abs" | "ratio_or_abs" | "none"
    abs_anchors: tuple[float, float, float] | None = None  # value at 100/50/0
    abs_when_driver_below: float | None = None  # ratio_or_abs switch point
    needs_driver: bool = True
    note: str = ""


METRICS: list[MetricDef] = [
    # ---- Longitudinal / Smoothness (per span)
    MetricDef("rms_jerk", "RMS jerk", "Smoothness", "m/s³", eps=0.02),
    MetricDef("p95_jerk", "P95 |jerk|", "Smoothness", "m/s³", eps=0.05),
    MetricDef("accel_reversals", "Accel reversals", "Smoothness", "/min", eps=0.2),
    MetricDef("pct_hard_accel", "Time |accel| > 2", "Smoothness", "%", eps=0.1),
    # ---- Longitudinal / Following (per follow window)
    MetricDef("median_gap", "Median time gap", "Following", "s", eps=0.2),
    MetricDef("gap_hunting", "Gap hunting (detrended std)", "Following", "s", eps=0.05),
    MetricDef("follow_reversals", "Accel reversals while following", "Following", "/min", eps=0.2),
    # ---- Longitudinal / Stopping (per stop approach)
    MetricDef("peak_decel", "Peak decel", "Stopping", "m/s²", eps=0.1),
    MetricDef("peak_decel_frac", "Peak-decel timing (fraction of approach)", "Stopping", "", eps=0.05),
    MetricDef("stop_lurch", "Stop lurch (max |jerk|, last 2 s)", "Stopping", "m/s³", eps=0.05),
    MetricDef("accel_at_crawl", "|Accel| at 0.2 m/s", "Stopping", "m/s²", eps=0.05),
    # ---- Longitudinal / Launch (per launch)
    MetricDef("time_to_5", "Time to 5 m/s", "Launch", "s", eps=0.2),
    MetricDef("launch_peak_jerk", "Peak |jerk| in launch", "Launch", "m/s³", eps=0.05),
    # ---- Longitudinal / Responsiveness (per stimulus)
    MetricDef("lead_decel_latency", "Lead-decel response latency", "Responsiveness", "s", eps=0.1),
    MetricDef("pullaway_latency", "Lead pull-away latency", "Responsiveness", "s", eps=0.1),
    # ---- Lateral / Turn Execution (per turn episode)
    MetricDef(
        "s_overshoot", "S-curve overshoot after unwind", "Turn Execution", "% of peak",
        eps=1.0, scorer="ratio_or_abs", abs_anchors=(5.0, 20.0, 40.0), abs_when_driver_below=5.0,
        note="absolute scale (100 at ≤5%, 50 at 20%, 0 at ≥40%) when your own overshoot is ~0",
    ),
    MetricDef(
        "recovery_wobbles", "Recovery wobbles (>10° re-crossings)", "Turn Execution", "/turn",
        eps=0.1, scorer="ratio_or_abs", abs_anchors=(0.0, 1.0, 2.0), abs_when_driver_below=0.5,
        note="absolute scale (100 at 0, 50 at 1, 0 at ≥2 per turn) when your baseline is ~0",
    ),
    MetricDef("unwind_rate", "Unwind rate after peak", "Turn Execution", "deg/s", better="higher", eps=1.0),
    MetricDef(
        "rescue_rate", "Driver-rescue rate in unwind", "Turn Execution", "%",
        agg="mean", scorer="abs", abs_anchors=(0.0, 25.0, 50.0), needs_driver=False,
        note="absolute scale: 100 at 0%, 50 at 25%, 0 at ≥50% of engaged turns",
    ),
    MetricDef("cmd_unwind_lead_left", "Cmd-vs-actual unwind lead (left)", "Turn Execution", "s",
              agg="mean", scorer="none", needs_driver=False, note="diagnostic, not scored"),
    MetricDef("cmd_unwind_lead_right", "Cmd-vs-actual unwind lead (right)", "Turn Execution", "s",
              agg="mean", scorer="none", needs_driver=False, note="diagnostic, not scored"),
    # ---- Lateral / Turn-In Timing (per intersection-turn intent)
    MetricDef("turn_in_delay", "Turn-in delay after blinker", "Turn-In Timing", "s", eps=0.2),
    MetricDef(
        "missed_turn_in", "Missed turn-ins", "Turn-In Timing", "%",
        agg="mean", scorer="abs", abs_anchors=(0.0, 25.0, 50.0), needs_driver=False,
        note="absolute scale: 100 at 0%, 50 at 25%, 0 at ≥50% of engaged turn intents",
    ),
    MetricDef("cmd_onset_lead_left", "Cmd-vs-actual onset lead (left)", "Turn-In Timing", "s",
              agg="mean", scorer="none", needs_driver=False, note="diagnostic, not scored"),
    MetricDef("cmd_onset_lead_right", "Cmd-vs-actual onset lead (right)", "Turn-In Timing", "s",
              agg="mean", scorer="none", needs_driver=False, note="diagnostic, not scored"),
    # ---- Lateral / General Smoothness (per span)
    MetricDef("rms_lat_jerk", "RMS lateral jerk", "General Smoothness", "m/s³", eps=0.02),
    MetricDef("steer_rate_rms", "Steering rate RMS (>10 m/s)", "General Smoothness", "deg/s", eps=0.5),
    MetricDef("steer_reversals", "Steering reversals (>10 m/s)", "General Smoothness", "/min", eps=0.5),
    MetricDef("pct_high_lat", "Time |lat accel| > 3", "General Smoothness", "%", eps=0.1),
]

METRIC_BY_KEY = {m.key: m for m in METRICS}
CATEGORY_TO_GROUP = {
    cat: grp for grp, cats in CATEGORY_GROUPS.items() for cat in cats
}


def score_ratio(m: float, d: float, better: str = "lower", eps: float = 1e-6) -> float:
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


def score_absolute(m: float, anchors: tuple[float, float, float]) -> float:
    """Piecewise-linear score through (a100 -> 100, a50 -> 50, a0 -> 0)."""
    a100, a50, a0 = anchors
    if m <= a100:
        return 100.0
    if m <= a50:
        return 100.0 - 50.0 * (m - a100) / (a50 - a100)
    if m <= a0:
        return 50.0 - 50.0 * (m - a50) / (a0 - a50)
    return 0.0


@dataclass
class MetricResult:
    definition: MetricDef
    model_vals: list[float]
    driver_vals: list[float]
    model_agg: float | None = None
    driver_agg: float | None = None
    score: float | None = None

    @property
    def n_model(self) -> int:
        return len(self.model_vals)

    @property
    def n_driver(self) -> int:
        return len(self.driver_vals)


@dataclass
class CategoryResult:
    name: str
    weight: float  # within its group
    metrics: list[MetricResult] = field(default_factory=list)
    score: float | None = None
    extra: dict = field(default_factory=dict)  # category-specific report data

    @property
    def letter(self) -> str | None:
        return letter(self.score) if self.score is not None else None


@dataclass
class GroupResult:
    name: str
    weight: float
    categories: list[CategoryResult] = field(default_factory=list)
    score: float | None = None

    @property
    def letter(self) -> str | None:
        return letter(self.score) if self.score is not None else None


@dataclass
class GradeReport:
    groups: list[GroupResult]
    overall_score: float | None
    overall_letter: str | None

    @property
    def categories(self) -> list[CategoryResult]:
        return [c for g in self.groups for c in g.categories]


# --------------------------------------------------------------- collection


def collect_samples(
    per_drive: list[tuple[Drive, Segmentation, DriveArrays, list[Event]]],
) -> dict[str, dict[str, list[float]]]:
    """Per-metric sample lists for span- and longitudinal-event metrics.

    Model samples come from engaged spans/events; events contaminated by a
    driver override (gas/brake while engaged) are excluded from the model
    side. Lateral model samples exclude steeringPressed moments.
    """
    samples: dict[str, dict[str, list[float]]] = {
        m.key: {"model": [], "driver": []} for m in METRICS
    }

    def add(key: str, side: str, value) -> None:
        if value is not None and np.isfinite(value):
            samples[key][side].append(float(value))

    for drive, seg, da, events in per_drive:
        # ---- longitudinal span-based (attributed by who controls gas/brake)
        for span in seg.long_spans:
            side = "model" if span.kind == "engaged" else "driver"
            sl = slice(span.i0, span.i1)
            t = da.t[sl]
            keep = ~da.long_override[sl] if side == "model" else np.ones(span.i1 - span.i0, bool)
            if keep.sum() < 10:
                continue

            a_s = da.a_smooth[sl]
            j = M.derivative(t, a_s)
            add("rms_jerk", side, M.rms(j[keep]))
            add("p95_jerk", side, M.p95_abs(j[keep]))
            add("accel_reversals", side, M.sign_reversals_per_min(t, a_s))
            add("pct_hard_accel", side, M.pct_time_above(t[keep], np.abs(da.a[sl][keep]), 2.0))

        # ---- lateral span-based (attributed by who steers; AOL time = model)
        for span in seg.lat_spans:
            side = "model" if span.kind == "engaged" else "driver"
            sl = slice(span.i0, span.i1)
            t = da.t[sl]
            lat_keep = np.ones(span.i1 - span.i0, bool)
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

        # ---- longitudinal event-based
        for ev in events:
            side = "model" if ev.engaged else "driver"
            if ev.engaged and ev.has_override:
                continue
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


def add_turn_samples(samples: dict, turns, intents) -> None:
    """Fold turn episodes and intent windows into the sample lists.

    Turn-execution behavior metrics skip engaged episodes where the driver
    interfered before the peak (contaminated) or during the unwind (rescued):
    those are not the model's unwind. Rescue itself is the rescue_rate signal.
    """

    def add(key, side, value):
        if value is not None and np.isfinite(value):
            samples[key][side].append(float(value))

    for ep in turns:
        side = "model" if ep.engaged else "driver"
        if ep.engaged:
            add("rescue_rate", "model", 100.0 if ep.rescued else 0.0)
        if ep.engaged and (ep.contaminated or ep.rescued):
            continue
        add("s_overshoot", side, ep.overshoot_pct)
        if ep.wobbles is not None:
            add("recovery_wobbles", side, float(ep.wobbles))
        add("unwind_rate", side, ep.unwind_rate)
        if ep.engaged and ep.cmd_unwind_lead is not None:
            add(f"cmd_unwind_lead_{ep.side}", "model", ep.cmd_unwind_lead)

    for w in intents:
        if w.outcome != "turn":
            continue
        if not w.engaged:
            if w.delay is not None:
                samples["turn_in_delay"]["driver"].append(float(w.delay))
        else:
            samples["missed_turn_in"]["model"].append(100.0 if w.missed else 0.0)
            if w.delay is not None:
                samples["turn_in_delay"]["model"].append(float(w.delay))
            if w.cmd_onset_lead is not None:
                samples[f"cmd_onset_lead_{w.side}"]["model"].append(float(w.cmd_onset_lead))


# ------------------------------------------------------------------ grading


def _aggregate(vals: list[float], how: str) -> float | None:
    if not vals:
        return None
    return float(np.mean(vals) if how == "mean" else np.median(vals))


def _score_metric(res: MetricResult) -> float | None:
    d = res.definition
    if d.scorer == "none":
        return None
    n_m, n_d = res.n_model, res.n_driver
    if n_m < MIN_EVENTS:
        return None
    if d.scorer == "abs" or not d.needs_driver:
        return score_absolute(res.model_agg, d.abs_anchors)
    if d.scorer == "ratio_or_abs":
        if n_d >= MIN_EVENTS and res.driver_agg is not None and res.driver_agg >= d.abs_when_driver_below:
            return score_ratio(res.model_agg, res.driver_agg, d.better, d.eps)
        return score_absolute(res.model_agg, d.abs_anchors)
    # plain ratio
    if n_d < MIN_EVENTS:
        return None
    return score_ratio(res.model_agg, res.driver_agg, d.better, d.eps)


def grade(
    samples: dict[str, dict[str, list[float]]],
    pingpong_score: float | None = None,
    pingpong_extra: dict | None = None,
) -> GradeReport:
    cats: dict[str, CategoryResult] = {}
    for grp, weights in CATEGORY_GROUPS.items():
        for cat, w in weights.items():
            cats[cat] = CategoryResult(cat, w)

    for mdef in METRICS:
        mv = [v for v in samples.get(mdef.key, {}).get("model", []) if np.isfinite(v)]
        dv = [v for v in samples.get(mdef.key, {}).get("driver", []) if np.isfinite(v)]
        res = MetricResult(mdef, mv, dv)
        res.model_agg = _aggregate(mv, mdef.agg)
        res.driver_agg = _aggregate(dv, mdef.agg)
        res.score = _score_metric(res)
        cats[mdef.category].metrics.append(res)

    for cat in cats.values():
        scored = [m.score for m in cat.metrics if m.score is not None]
        cat.score = float(np.mean(scored)) if scored else None

    # Ping-Pong is scored specially (time-weighted speed-bin comparison)
    cats["Ping-Pong"].score = pingpong_score
    if pingpong_extra:
        cats["Ping-Pong"].extra = pingpong_extra

    groups = []
    for grp_name, weights in CATEGORY_GROUPS.items():
        grp = GroupResult(grp_name, GROUP_WEIGHTS[grp_name])
        grp.categories = [cats[c] for c in weights]
        valid = [(c, weights[c.name]) for c in grp.categories if c.score is not None]
        if valid:
            tw = sum(w for _c, w in valid)
            grp.score = sum(c.score * w for c, w in valid) / tw
        groups.append(grp)

    valid_g = [g for g in groups if g.score is not None]
    if valid_g:
        tw = sum(g.weight for g in valid_g)
        overall = sum(g.score * g.weight for g in valid_g) / tw
        return GradeReport(groups, overall, letter(overall))
    return GradeReport(groups, None, None)
