"""Score the model against the driver and assign letter grades.

Two top-level groups, each with its own headline grade:
- Longitudinal: Smoothness 0.25, Following 0.18, Stopping 0.17,
  Launch 0.15, Responsiveness 0.10, Speed Disagreement 0.15
- Lateral: Ping-Pong 0.40, Turns 0.50, General Smoothness 0.10
Overall = 0.5*Longitudinal + 0.5*Lateral (renormalized if a group has no data).

Relative (ratio) scoring, lower-is-better unless stated: with m = model
aggregate, d = driver aggregate:

    r = m / max(d, eps)
    score = 100                   if r <= 1
    score = 100 - 50*(r-1)        if 1 < r <= 2     (2x worse than you = 50)
    score = max(0, 50 - 25*(r-2)) if r > 2

Higher-is-better metrics invert the ratio. "match" metrics (follow gap,
peak decel and its timing, launch time-to-speed, unwind rate) are style
targets: deviation from the driver in either direction is penalized, so
r = max(m,d)/min(m,d). eps is a per-metric floor at the
scale where differences stop being meaningful. Metrics without a human
counterpart (driver-rescue rate, missed turn-ins) or whose human baseline is
~zero (S-curve overshoot) use an absolute anchor scale instead, documented
per metric. Aggregation is the median of per-event values (mean for
rate-of-events metrics). A ratio metric needs >= MIN_EVENTS samples per side.

A category's own score is the mean of its scored metrics -- but only once
>= MIN_SCORED_FOR_CATEGORY of them actually have a score; one lone scored
metric (everything else gated by MIN_EVENTS) would otherwise hand its single
value straight through as the whole category's grade. Same rule for
Ping-Pong's bin average (a category unto itself, scored outside METRICS) and
every mode/personality breakdown bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .events import DriveArrays, Event
from .extract import Drive
from .segments import Segmentation, _contiguous_runs

MIN_EVENTS = 3
# A category graded off a single scored metric is one metric's quirks away
# from a misleading 100 (or 0) with nothing else to check it against -- need
# at least two independently-scored things before a category gets a grade.
MIN_SCORED_FOR_CATEGORY = 2

GROUP_WEIGHTS = {"Lateral": 0.5, "Longitudinal": 0.5}

# longitudinal breakdown buckets (mode and personality tracked separately);
# events tagged "mixed"/"unknown" stay in the overall grade but are excluded
# from every bucket
MODE_BUCKETS = ("chill", "experimental")
PERSONALITY_BUCKETS = ("aggressive", "standard", "relaxed")
ALL_BUCKETS = MODE_BUCKETS + PERSONALITY_BUCKETS

CATEGORY_GROUPS = {
    "Lateral": {
        # Card order = dict order (see grade()'s "grp.categories = [cats[c]
        # for c in weights]"), which also drives report.py's multi-column
        # card packing. Turns is by far the tallest card in this group (it
        # absorbed two former categories), so it's placed LAST -- CSS
        # multi-column fills columns in strict content order, so the two
        # short cards need to come first to end up packed together in one
        # column instead of Turns splitting them across columns.
        "Ping-Pong": 0.40,
        "General Smoothness": 0.10,
        "Turns": 0.50,
    },
    "Longitudinal": {
        "Smoothness": 0.25,
        "Following": 0.18,
        "Stopping": 0.17,
        "Launch": 0.15,
        "Responsiveness": 0.10,
        "Speed Disagreement": 0.15,
    },
}

# Scores are mathematically capped at 100.0 (score_ratio/score_absolute never
# exceed it), so S_CUTOFF only needs to absorb float summation noise from
# weighted averages -- not distinguish real near-100 scores, which differ
# from a true 100 by far more than 1e-6.
S_CUTOFF = 100.0 - 1e-6

LETTER_BINS = [(S_CUTOFF, "S"), (93, "A"), (85, "A-"), (78, "B+"), (70, "B"), (60, "C"), (50, "D")]


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
    better: str = "lower"  # "lower" | "higher" | "match" (deviation either way is worse)
    eps: float = 1e-6
    agg: str = "median"  # "median" | "mean"
    scorer: str = "ratio"  # "ratio" | "abs" | "ratio_or_abs" | "none"
    abs_anchors: tuple[float, float, float] | None = None  # value at 100/50/0
    abs_when_driver_below: float | None = None  # ratio_or_abs switch point
    needs_driver: bool = True
    note: str = ""
    # A scorer="none" row is hidden from the report table by default (see
    # report._metric_rows) -- most unscored rows are low-value diagnostics
    # not worth the clutter. Set True for the rare case that's genuinely
    # worth showing despite not (yet) having a defensible scoring formula.
    show_unscored: bool = False
    # Plain-English, 1-3 sentence explanation of this ONE row, shown only
    # when the user clicks it open (see report._metric_rows). Distinct from
    # `note`, which is either short score-cell filler (scorer="none") or
    # scoring-anchor detail -- neither is meant for a casual reader.
    desc: str = ""


METRICS: list[MetricDef] = [
    # ---- Longitudinal / Smoothness (per span)
    MetricDef(
        "rms_jerk", "RMS jerk", "Smoothness", "m/s³", eps=0.02,
        desc="How fast acceleration typically changes — the head-toss feeling passengers get. "
        "Higher means jerkier driving overall.",
    ),
    MetricDef(
        "p95_jerk", "P95 |jerk|", "Smoothness", "m/s³", eps=0.05,
        desc="How bad the worst 5% of acceleration-change moments get, rather than the typical case "
        "RMS jerk shows — catches occasional harsh moments that an average would hide.",
    ),
    MetricDef(
        "accel_reversals", "Accel reversals", "Smoothness", "/min", eps=0.2,
        desc="How many times per minute the car flips between throttle↔brake. Higher means more "
        "speed-hunting, less settled cruising.",
    ),
    MetricDef(
        "pct_hard_accel", "Time |accel| > 2", "Smoothness", "%", eps=0.1,
        desc="The share of time spent accelerating or braking harder than 2 m/s² — genuinely hard, "
        "not routine driving.",
    ),
    # ---- Longitudinal / Following (per follow window)
    MetricDef(
        "median_gap", "Median time gap", "Following", "s", better="match", eps=0.2,
        desc="Typical following distance behind the lead car, in seconds. Graded either direction — "
        "too close or too far back both count against it.",
    ),
    MetricDef(
        "gap_hunting", "Gap hunting (detrended std)", "Following", "s", eps=0.05,
        desc="How much the following gap oscillates once the overall trend is removed — creeping "
        "up then falling back repeatedly, rather than holding a steady distance.",
    ),
    MetricDef(
        "follow_reversals", "Accel reversals while following", "Following", "/min", eps=0.2,
        desc="How many times per minute the car flips between accelerating and braking specifically "
        "while following a lead car.",
    ),
    # ---- Longitudinal / Stopping (per stop approach)
    MetricDef(
        "peak_decel", "Peak decel", "Stopping", "m/s²", better="match", eps=0.1,
        desc="The hardest braking moment during a stop approach. Graded either direction — braking "
        "too gently or too hard both count against it.",
    ),
    MetricDef(
        "peak_decel_frac", "Peak-decel timing (fraction of approach)", "Stopping", "", better="match", eps=0.05,
        desc="Where in the stop approach the hardest braking happens: earlier means front-loaded "
        "braking, later means a last-second squeeze. Expressed as a fraction of the whole approach.",
    ),
    MetricDef(
        "stop_lurch", "Stop lurch (max |jerk|, last 2 s)", "Stopping", "m/s³", eps=0.05,
        desc="The jolt in the final 2 seconds before coming to a full stop — the classic head-nod "
        "right at the end.",
    ),
    MetricDef(
        "accel_at_crawl", "|Accel| at 0.2 m/s", "Stopping", "m/s²", eps=0.05,
        desc="How hard the brakes are still biting at walking pace, right before the car actually "
        "comes to rest.",
    ),
    # ---- Longitudinal / Launch (per launch)
    MetricDef(
        "time_to_5", "Time to 5 m/s", "Launch", "s", better="match", eps=0.2,
        desc="How long it takes to reach 5 m/s (about 11 mph) from a complete stop. Graded either "
        "direction — too slow feels sluggish, too fast feels abrupt.",
    ),
    MetricDef(
        "launch_peak_jerk", "Peak |jerk| in launch", "Launch", "m/s³", eps=0.05,
        desc="How abrupt the initial getaway from a stop feels — the peak rate of acceleration "
        "change right as the car starts moving.",
    ),
    # ---- Longitudinal / Responsiveness (per stimulus)
    MetricDef(
        "lead_decel_latency", "Lead-decel response latency", "Responsiveness", "s", eps=0.1,
        desc="Seconds between the radar seeing the lead car brake meaningfully and this car actually "
        "backing off. Your own number is often measured slow because you react to cues radar can't "
        "see yet (brake lights, a changing light) — a slow human number here doesn't mean you're slow.",
    ),
    MetricDef(
        "pullaway_latency", "Lead pull-away latency", "Responsiveness", "s", eps=0.1,
        desc="Seconds between the lead car starting to move and this car starting to move. Same "
        "anticipation caveat as lead-decel latency applies.",
    ),
    # (Speed Disagreement rows are duration-weighted global aggregates built
    #  in speed_disagreement_results, adherence-style, not METRICS entries)
    # ---- Lateral / Turns (per turn episode; unwind quality, turn-in
    # commitment, and contested-execution divergence all live in one
    # category -- see add_turn_samples)
    MetricDef(
        "s_overshoot", "S-curve overshoot after unwind (sharp turns)", "Turns", "% of peak",
        eps=1.0, scorer="ratio_or_abs", abs_anchors=(5.0, 20.0, 40.0), abs_when_driver_below=5.0,
        note="absolute scale (100 at ≤5%, 50 at 20%, 0 at ≥40%) when your own overshoot is ~0",
        desc="After a sharp turn straightens out, how far the wheel swings past center the other "
        "way before settling, as a % of how far it turned into the turn. Higher means more overcorrection.",
    ),
    MetricDef(
        "recovery_wobbles", "Recovery wobbles, >10° re-crossings (sharp turns)", "Turns", "/turn",
        eps=0.1, scorer="ratio_or_abs", abs_anchors=(0.0, 1.0, 2.0), abs_when_driver_below=0.5,
        note="absolute scale (100 at 0, 50 at 1, 0 at ≥2 per turn) when your baseline is ~0",
        desc="How many extra back-and-forth swings past 10° happen while settling out of a sharp "
        "turn, on top of the main overshoot. Higher means a twitchier finish.",
    ),
    MetricDef(
        "unwind_rate", "Unwind rate after peak (sharp turns)", "Turns", "deg/s", better="match", eps=1.0,
        desc="How fast the wheel comes back toward straight after a sharp turn's peak angle, in "
        "degrees per second. Graded against your own unwind speed in either direction — too fast or "
        "too slow both count against it.",
    ),
    MetricDef(
        "rescue_rate", "Driver-rescue rate in unwind", "Turns", "%",
        agg="mean", scorer="abs", abs_anchors=(0.0, 25.0, 50.0), needs_driver=False,
        note="absolute scale: 100 at 0%, 50 at 25%, 0 at ≥50% of model-executed turns",
        desc="Of the sharp turns the model was executing, the % where you had to grab the wheel "
        "(or it disengaged) before it finished straightening out on its own.",
    ),
    MetricDef("curve_s_overshoot", "Overshoot, curve episodes 20–90°", "Turns", "% of peak",
              scorer="none", needs_driver=False, note="reported separately, not scored"),
    MetricDef("curve_recovery_wobbles", "Wobbles, curve episodes 20–90°", "Turns", "/turn",
              scorer="none", needs_driver=False, note="reported separately, not scored"),
    MetricDef("curve_unwind_rate", "Unwind rate, curve episodes 20–90°", "Turns", "deg/s",
              scorer="none", needs_driver=False, note="reported separately, not scored"),
    MetricDef("cmd_unwind_lead_left", "Cmd-vs-actual unwind lead (left)", "Turns", "s",
              agg="mean", scorer="none", needs_driver=False, note="diagnostic, not scored"),
    MetricDef("cmd_unwind_lead_right", "Cmd-vs-actual unwind lead (right)", "Turns", "s",
              agg="mean", scorer="none", needs_driver=False, note="diagnostic, not scored"),
    MetricDef(
        "turn_effort_left", "Peak turn-in effort before any override (left)", "Turns", "%",
        scorer="none", needs_driver=False, show_unscored=True,
        note="context only, not scored -- see help text below",
        desc="The highest % of available steering torque the model reached from the start of the "
        "turn up to the moment you first touched the wheel (or across the whole turn if you never "
        "did) — a rough read on whether it's actually committing. A low number isn't necessarily "
        "wrong, since some turns just don't need much torque.",
    ),
    MetricDef(
        "turn_effort_right", "Peak turn-in effort before any override (right)", "Turns", "%",
        scorer="none", needs_driver=False, show_unscored=True,
        note="context only, not scored -- see help text below",
        desc="The highest % of available steering torque the model reached from the start of the "
        "turn up to the moment you first touched the wheel (or across the whole turn if you never "
        "did) — a rough read on whether it's actually committing. A low number isn't necessarily "
        "wrong, since some turns just don't need much torque.",
    ),
    MetricDef(
        "resisted_divergence_left", "Cmd-vs-actual divergence while you resisted (left)", "Turns", "deg",
        scorer="abs", abs_anchors=(15.0, 75.0, 300.0), needs_driver=False,
        note="absolute scale: 100 at <=15°, 50 at 75°, 0 at >=300° peak |actual - commanded| angle "
        "during a sustained (>=0.3s) window where you genuinely resisted the model's own steering "
        "torque -- only scored on episodes with real, sustained disagreement. Split left/right by "
        "which way the turn itself went (same convention as cmd_onset_lead_left/right), same anchors "
        "both sides. Anchors set from the real-data divergence distribution across two Palisade "
        "routes (35 conflict episodes: median ~79°, p90 ~337°, max 823° on one genuine multi-second "
        "tug-of-war) -- see report footer",
        desc="Only scores turns where you genuinely fought the model's steering. Model/You show the "
        "angle each side actually held at the peak of the disagreement, so you can see directly "
        "whether the model wanted a sharper or softer turn than you were willing to give it.",
    ),
    MetricDef(
        "resisted_divergence_right", "Cmd-vs-actual divergence while you resisted (right)", "Turns", "deg",
        scorer="abs", abs_anchors=(15.0, 75.0, 300.0), needs_driver=False,
        note="same metric and anchors as resisted_divergence_left, scoped to turns that went right",
        desc="Only scores turns where you genuinely fought the model's steering. Model/You show the "
        "angle each side actually held at the peak of the disagreement, so you can see directly "
        "whether the model wanted a sharper or softer turn than you were willing to give it.",
    ),
    MetricDef(
        "cmd_onset_lead_left", "Cmd-vs-actual onset timing (left)", "Turns", "s",
        agg="mean", scorer="abs", abs_anchors=(0.0, 0.5, 1.5), needs_driver=False,
        note="how much later (+) or sooner (-) the model's own commanded path called for the turn, "
        "vs. when the wheel actually turned in (that instant is the zero reference, shown as 'You: "
        "0.00'). Absolute scale: 100 at <=0s (model committed at/before the wheel moved), 50 at 0.5s "
        "late, 0 at >=1.5s late",
        desc="How much later (+) or sooner (−) the model's own commanded path called for the turn, "
        "vs. when the wheel actually turned in. 'You' is always 0.00 — that's the instant being "
        "measured against.",
    ),
    MetricDef(
        "cmd_onset_lead_right", "Cmd-vs-actual onset timing (right)", "Turns", "s",
        agg="mean", scorer="abs", abs_anchors=(0.0, 0.5, 1.5), needs_driver=False,
        note="how much later (+) or sooner (-) the model's own commanded path called for the turn, "
        "vs. when the wheel actually turned in (that instant is the zero reference, shown as 'You: "
        "0.00'). Absolute scale: 100 at <=0s (model committed at/before the wheel moved), 50 at 0.5s "
        "late, 0 at >=1.5s late",
        desc="How much later (+) or sooner (−) the model's own commanded path called for the turn, "
        "vs. when the wheel actually turned in. 'You' is always 0.00 — that's the instant being "
        "measured against.",
    ),
    # ---- Lateral / General Smoothness (per span)
    MetricDef(
        "rms_lat_jerk", "RMS lateral jerk", "General Smoothness", "m/s³", eps=0.02,
        desc="Side-to-side smoothness felt by passengers — how fast lateral acceleration changes "
        "during ordinary driving at speed.",
    ),
    MetricDef(
        "steer_rate_rms", "Steering rate RMS (>10 m/s)", "General Smoothness", "deg/s", eps=0.5,
        desc="How fast the steering wheel is typically being moved, above about 22 mph. Higher means "
        "more constant small corrections rather than a settled line.",
    ),
    MetricDef(
        "steer_reversals", "Steering reversals (>10 m/s)", "General Smoothness", "/min", eps=0.5,
        desc="How many times per minute the wheel saws back and forth, above about 22 mph.",
    ),
    MetricDef(
        "pct_high_lat", "Time |lat accel| > 3", "General Smoothness", "%", eps=0.1,
        desc="The share of time spent in genuinely hard cornering (lateral acceleration over 3 m/s²) "
        "at speed.",
    ),
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
    if better == "match":
        # style metrics: deviating from the driver in EITHER direction is worse
        # (e.g. a shorter follow gap than the driver's must not score 100)
        m, d = max(m, d), min(m, d)
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
    driver_vals: list[float]  # COMBINED (this-run + pooled profile) driver samples
    model_agg: float | None = None
    driver_agg: float | None = None
    score: float | None = None
    # driver-profile pooling provenance (see profile.py). Empty/None when
    # profiling is off or this metric isn't poolable. driver_vals above is
    # what actually feeds driver_agg/score; these fields exist purely so the
    # report/CLI can show where the combined number came from.
    driver_vals_this_drive: list[float] = field(default_factory=list)
    same_drive_agg: float | None = None  # set only when this drive ALONE has >= MIN_EVENTS

    @property
    def n_model(self) -> int:
        return len(self.model_vals)

    @property
    def n_driver(self) -> int:
        return len(self.driver_vals)

    @property
    def n_this_drive(self) -> int:
        return len(self.driver_vals_this_drive)

    @property
    def n_pooled(self) -> int:
        return max(0, self.n_driver - self.n_this_drive)


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
class BucketGrade:
    """Longitudinal grades for one mode/personality breakdown bucket."""

    bucket: str
    categories: list[CategoryResult]
    score: float | None

    @property
    def letter(self) -> str | None:
        return letter(self.score) if self.score is not None else None


@dataclass
class GradeReport:
    groups: list[GroupResult]
    overall_score: float | None
    overall_letter: str | None
    # {"mode": {bucket: BucketGrade}, "personality": {...}} for Longitudinal
    breakdowns: dict = field(default_factory=dict)

    @property
    def categories(self) -> list[CategoryResult]:
        return [c for g in self.groups for c in g.categories]


# --------------------------------------------------------------- collection


def _event_buckets(ev: Event) -> tuple[str, ...]:
    """Breakdown buckets an engaged-long event contributes to."""
    out = []
    if ev.values.get("mode") in MODE_BUCKETS:
        out.append(ev.values["mode"])
    if ev.values.get("personality") in PERSONALITY_BUCKETS:
        out.append(ev.values["personality"])
    return tuple(out)


def collect_samples(
    per_drive: list[tuple[Drive, Segmentation, DriveArrays, list[Event]]],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """Per-metric sample lists for span- and longitudinal-event metrics.

    Model samples come from engaged spans/events; events contaminated by a
    driver override (gas/brake while engaged) are excluded from the model
    side. Lateral model samples exclude steeringPressed moments.
    """
    samples: dict[str, dict[str, list[float]]] = {
        m.key: {"model": [], "driver": []} for m in METRICS
    }
    bucket_samples: dict[str, dict[str, list[float]]] = {
        m.key: {b: [] for b in ALL_BUCKETS} for m in METRICS
    }

    def add(key: str, side: str, value, buckets: tuple[str, ...] = ()) -> None:
        if value is not None and np.isfinite(value):
            samples[key][side].append(float(value))
            if side == "model":
                for b in buckets:
                    bucket_samples[key][b].append(float(value))

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

            # per-bucket smoothness samples (model side only): mask the span
            # by the mode / personality active at each sample, so mid-span
            # hot-swaps attribute samples to the right bucket
            if side == "model":
                span_buckets = []
                if da.exp_mode is not None:
                    exp = da.exp_mode[sl]
                    span_buckets += [("experimental", exp), ("chill", ~exp)]
                if da.personality is not None:
                    pers = da.personality[sl]
                    span_buckets += [
                        (name, pers == idx)
                        for idx, name in ((0, "aggressive"), (1, "standard"), (2, "relaxed"))
                    ]
                for bname, bmask in span_buckets:
                    m2 = keep & bmask
                    if m2.sum() < 500:  # ~5 s at 100 Hz
                        continue
                    bucket_samples["rms_jerk"][bname].append(float(M.rms(j[m2])))
                    bucket_samples["p95_jerk"][bname].append(float(M.p95_abs(j[m2])))
                    # reversal rate per contiguous chunk of the bucket mask so
                    # the time between chunks doesn't dilute the rate
                    count, dur = 0.0, 0.0
                    for a2, b2 in _contiguous_runs(m2):
                        if b2 - a2 < 100:  # ignore sub-second slivers
                            continue
                        d2 = float(t[b2 - 1] - t[a2])
                        rate = M.sign_reversals_per_min(t[a2:b2], a_s[a2:b2])
                        if np.isfinite(rate) and d2 > 0:
                            count += rate * d2 / 60.0
                            dur += d2
                    if dur > 0:
                        bucket_samples["accel_reversals"][bname].append(60.0 * count / dur)
                    pct = M.pct_time_above(t[m2], np.abs(da.a[sl][m2]), 2.0)
                    if np.isfinite(pct):
                        bucket_samples["pct_hard_accel"][bname].append(float(pct))

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
            if ev.kind == "gas_override":
                continue  # sampled in speed_disagreement.py, not here
            side = "model" if ev.engaged else "driver"
            if ev.engaged and ev.has_override:
                continue
            bk = _event_buckets(ev) if ev.engaged else ()
            sl = slice(ev.i0, ev.i1)
            t, v, a = da.t[sl], da.v[sl], da.a[sl]
            if ev.kind == "stop":
                sm = M.stop_metrics(t, v, a, ev.values.get("t_standstill"))
                add("peak_decel", side, sm["peak_decel"], bk)
                add("peak_decel_frac", side, sm["peak_decel_frac"], bk)
                add("stop_lurch", side, sm["stop_lurch"], bk)
                add("accel_at_crawl", side, sm["accel_at_crawl"], bk)
                ev.values.update(sm)
            elif ev.kind == "launch":
                lm = M.launch_metrics(t, v, a, ev.values["t_first_motion"])
                add("time_to_5", side, lm["time_to_5"], bk)
                add("launch_peak_jerk", side, lm["peak_jerk"], bk)
                ev.values.update(lm)
            elif ev.kind == "follow":
                fm = M.follow_metrics(t, v, a, da.d_rel[sl])
                add("median_gap", side, fm["median_gap"], bk)
                add("gap_hunting", side, fm["gap_hunting"], bk)
                add("follow_reversals", side, fm["accel_reversals"], bk)
                ev.values.update(fm)
            elif ev.kind == "lead_decel":
                add("lead_decel_latency", side, ev.values["latency"], bk)
            elif ev.kind == "pullaway":
                add("pullaway_latency", side, ev.values["latency"], bk)

    return samples, bucket_samples


def add_turn_samples(samples: dict, turns) -> None:
    """Fold turn episodes into the sample lists.

    The unwind-phase behavior metrics (s_overshoot, recovery_wobbles,
    unwind_rate -- and the onset-timing lag, cmd_onset_lead) skip engaged
    episodes where the driver interfered before the peak (contaminated) or
    during the unwind (rescued): those aren't the model's own unwind/onset,
    they're a human-forced one. Rescue itself is the rescue_rate signal.

    resisted_divergence_{left,right} and turn_effort_{left,right} are the
    exceptions: neither is gated on contaminated/rescued. resisted_divergence
    exists BECAUSE sustained driver resistance against the model's own
    steering is the contamination mechanism -- gating it the same way as the
    others would exclude precisely the cases it exists to catch. It only
    scores episodes where lateral.detect_turn_episodes found a genuine,
    sustained (>=0.3s) directional conflict (steering torque opposing the
    model's own commanded torque); an episode with no such conflict
    contributes nothing (not a zero), same as cmd_onset_lead. turn_effort is
    peak_effort_frac, which is ALREADY scoped (in lateral.py) to stop at the
    driver's first touch, so it needs no further gating here -- a
    contaminated episode's pre-contamination effort is exactly what it's
    for. Split left/right by ep.side, same convention as
    cmd_onset_lead_left/right. This category is blinker-free and
    band-agnostic: every engaged turn counts, sharp or curve, signaled or
    not (see lateral.detect_turn_episodes).
    """

    def add(key, side, value):
        if value is not None and np.isfinite(value):
            samples[key][side].append(float(value))

    for ep in turns:
        side = "model" if ep.engaged else "driver"
        if ep.engaged and not ep.contaminated:
            # rescue rate counts only turns the model was actually executing
            # up to the peak (driver forcing the wheel earlier is a different
            # failure, measured by turn-in metrics)
            add("rescue_rate", "model", 100.0 if ep.rescued else 0.0)
        if ep.engaged:
            add(f"resisted_divergence_{ep.side}", "model", ep.divergence_deg)
        if ep.engaged and ep.peak_effort_frac is not None:
            add(f"turn_effort_{ep.side}", "model", ep.peak_effort_frac * 100.0)
        if ep.engaged and (ep.contaminated or ep.rescued):
            continue
        # sharp turns feed the scored metrics; 20-90 deg curve episodes are
        # reported separately (diagnostic rows)
        prefix = "" if ep.sharp else "curve_"
        add(prefix + "s_overshoot", side, ep.overshoot_pct)
        if ep.wobbles is not None:
            add(prefix + "recovery_wobbles", side, float(ep.wobbles))
        add(prefix + "unwind_rate", side, ep.unwind_rate)
        if ep.engaged and ep.cmd_unwind_lead is not None:
            add(f"cmd_unwind_lead_{ep.side}", "model", ep.cmd_unwind_lead)
        if ep.engaged and ep.cmd_onset_lead is not None:
            add(f"cmd_onset_lead_{ep.side}", "model", ep.cmd_onset_lead)


INITIATOR_BUCKETS = ("model", "driver", "lag", "unknown")


def turn_in_breakdown(turns) -> dict[str, dict]:
    """Unscored Turns texture: every engaged episode by initiator
    (who moved first -- descriptive only, never gates scoring), with the
    conflict/divergence data resisted_divergence actually scores from, and
    torque-ceiling status as pure diagnostic context (was the model already
    at its output limit -- never a scoring gate, see lateral.py)."""
    buckets = {
        b: {"n": 0, "n_conflict": 0, "divergences": [],
            "ceiling_true": 0, "ceiling_false": 0, "ceiling_unknown": 0}
        for b in INITIATOR_BUCKETS
    }
    for ep in turns:
        if not ep.engaged:
            continue
        bucket = buckets[ep.initiator if ep.initiator in INITIATOR_BUCKETS else "unknown"]
        bucket["n"] += 1
        if ep.divergence_deg is None:
            continue
        bucket["n_conflict"] += 1
        bucket["divergences"].append(ep.divergence_deg)
        if ep.conflict_ceiling is True:
            bucket["ceiling_true"] += 1
        elif ep.conflict_ceiling is False:
            bucket["ceiling_false"] += 1
        else:
            bucket["ceiling_unknown"] += 1
    for bucket in buckets.values():
        divs = bucket.pop("divergences")
        bucket["median_divergence"] = float(np.median(divs)) if divs else None
    return buckets


def resisted_angle_context(turns) -> dict[str, dict]:
    """Per-side (left/right) median |actual| and |commanded| angle at the
    peak-disagreement moment, across the SAME episodes resisted_divergence_
    {side} scores from -- turns a single hard-to-interpret divergence number
    into "you held ~Xdeg, the model wanted ~Ydeg", which is what the
    divergence score is actually built from. Report-display use only (see
    report.py's row_overrides); never affects scoring."""
    out: dict[str, dict] = {}
    for side in ("left", "right"):
        you_vals = [
            ep.conflict_you_deg for ep in turns
            if ep.engaged and ep.side == side and ep.conflict_you_deg is not None
        ]
        model_vals = [
            ep.conflict_model_deg for ep in turns
            if ep.engaged and ep.side == side and ep.conflict_model_deg is not None
        ]
        if you_vals and model_vals:
            out[side] = {
                "you_deg": float(np.median(you_vals)),
                "model_deg": float(np.median(model_vals)),
                "n": len(you_vals),
            }
    return out


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


ADHERENCE_MIN_SECONDS = 30.0
ADHERENCE_ANCHORS = (5.0, 25.0, 50.0)  # % error at score 100 / 50 / 0


def adherence_results(
    adherence: dict[str, dict], targets: dict[str, float]
) -> list[MetricResult]:
    """One Following row per personality with steady-follow data.

    Model column = median effective t_follow the model actually held;
    "You" column = that personality's TARGET (fork-dependent), not the human.
    Scored absolutely on % error: 100 at <=5%, 50 at 25%, 0 at >=50%.
    """
    out: list[MetricResult] = []
    for p in PERSONALITY_BUCKETS:
        info = adherence.get(p)
        if not info or not np.isfinite(info.get("median_eff", float("nan"))):
            continue
        target = float(targets.get(p, 0.0)) or 1.0
        med = float(info["median_eff"])
        secs = float(info.get("seconds", 0.0))
        pct = abs(med - target) / target * 100.0
        enough = secs >= ADHERENCE_MIN_SECONDS
        d = MetricDef(
            key=f"follow_adherence_{p}",
            label=f"Follow adherence ({p})",
            category="Following",
            unit="s",
            scorer="abs" if enough else "none",
            abs_anchors=ADHERENCE_ANCHORS,
            needs_driver=False,
            # show_unscored: without it, an insufficient-data row here would
            # be silently hidden entirely by _metric_rows' hide-unscored-rows
            # rule instead of showing why (the note below already explains
            # it) -- unlike the curve_*/cmd_unwind_lead_* diagnostics that
            # rule was written for, this one's worth seeing even when it
            # can't yet score.
            show_unscored=True,
            note=(
                f"holds {med:.2f} s vs {target:.2f} s target, {pct:.0f}% off "
                f"({secs:.0f} s of steady follow)"
                + ("" if enough else f" — need {ADHERENCE_MIN_SECONDS:.0f} s to score")
            ),
            desc="How closely the model holds the ACTIVE personality's target following gap. The "
            "You column shows that personality's target, not a human baseline — there's no 'your' "
            "following distance to compare against here.",
        )
        r = MetricResult(d, [med], [])
        r.model_agg = med
        r.driver_agg = target  # displayed in the "You" column as the target
        if enough:
            r.score = score_absolute(pct, ADHERENCE_ANCHORS)
        out.append(r)
    return out


# ------------------------------------------------- speed disagreement rows

SD_MIN_SECONDS = 120.0  # model-long time needed to score rate / %-time
SD_MIN_MAG_EPISODES = 3  # episodes with plan data needed to score magnitude
SD_RATE_ANCHORS = (0.0, 4.0, 8.0)  # episodes / 10 min at score 100 / 50 / 0
SD_PCT_ANCHORS = (0.0, 10.0, 25.0)  # % of model-long time
SD_MAG_ANCHORS = (0.2, 1.0, 2.0)  # m/s^2 of extra demanded accel


_SD_DESC = {
    "gas_override_rate": "How often you press the gas to override the model's speed while it's "
    "controlling longitudinal, per 10 minutes of model-controlled driving. Every press is a clean "
    "signal you wanted more speed than it was giving you.",
    "gas_override_pct": "The share of model-controlled driving time you spend actively overriding "
    "its speed with the gas pedal.",
    "gas_override_magnitude": "How much more acceleration you demanded than the model's own plan "
    "called for, averaged across your overrides. A bigger number means you're pushing noticeably "
    "harder than it wanted to.",
    "speed_taken_back": "How much speed the model sheds in the 10 seconds after you let go of the "
    "gas — the model actively fighting back against the cruise speed you just set.",
    "reoverride_pct": "The share of overrides where you had to press the gas again within 15 "
    "seconds — a sign the model settled back to a speed you still didn't want.",
}


def _sd_row(
    key: str, label: str, unit: str, value: float | None,
    anchors: tuple[float, float, float] | None, enough: bool, note: str,
    n_vals: int = 1,
) -> MetricResult:
    d = MetricDef(
        key=key, label=label, category="Speed Disagreement", unit=unit,
        scorer="abs" if (enough and anchors is not None) else "none",
        abs_anchors=anchors, needs_driver=False, note=note,
        # show_unscored: speed_taken_back/reoverride_pct have no anchors at
        # all and are ALWAYS built with enough=False (context by design, see
        # speed_disagreement_results) -- without this they'd be permanently
        # invisible in the main card despite CATEGORY_HELP explicitly
        # describing them, and the rate/pct/magnitude rows would vanish too
        # whenever data is thin rather than showing why.
        show_unscored=True,
        desc=_SD_DESC.get(key, ""),
    )
    r = MetricResult(d, [float(value)] * n_vals if value is not None else [], [])
    r.model_agg = value
    if enough and anchors is not None and value is not None:
        r.score = score_absolute(value, anchors)
    return r


def speed_disagreement_results(sd) -> list[MetricResult]:
    """Speed Disagreement rows from duration-weighted global aggregates.

    There is no human baseline for overriding yourself, so rate/%/magnitude
    are scored on documented absolute anchors; instead of the usual n>=3
    events per side, rate and %-time gate on SD_MIN_SECONDS of model-long
    time (a rate of zero over plenty of time is a meaningful 100)."""
    secs = sd.model_long_seconds
    enough_time = sd.have_gas and sd.overall_rate is not None and secs >= SD_MIN_SECONDS
    time_note = f"{len(sd.episodes)} episodes over {secs / 60:.1f} min of model-long time"
    if not sd.have_gas:
        time_note = "gasPressed channel missing in these logs"
    elif not enough_time:
        time_note += f" — need {SD_MIN_SECONDS:.0f} s of model-long time to score"
    rows = [
        _sd_row("gas_override_rate", "Gas-override episodes", "/10min",
                sd.overall_rate, SD_RATE_ANCHORS, enough_time, time_note),
        _sd_row("gas_override_pct", "Gas-override time", "%",
                sd.overall_pct, SD_PCT_ANCHORS, enough_time, time_note),
    ]
    enough_mag = sd.n_mag_episodes >= SD_MIN_MAG_EPISODES
    if not sd.have_vis:
        mag_note = "vision-plan accel not in these logs (older openpilot)"
    else:
        mag_note = (f"median (aEgo − planned accel) over {sd.n_mag_samples} override samples, "
                    f"{sd.n_mag_episodes} episodes")
        if not enough_mag:
            mag_note += f" — need {SD_MIN_MAG_EPISODES} episodes to score"
    rows.append(_sd_row("gas_override_magnitude", "Gas-override magnitude", "m/s²",
                        sd.overall_magnitude, SD_MAG_ANCHORS, enough_mag, mag_note,
                        n_vals=sd.n_mag_episodes))
    n_tb = sum(1 for e in sd.episodes if e.speed_taken_back is not None)
    rows.append(_sd_row("speed_taken_back", "Speed taken back after release", "m/s",
                        sd.speed_taken_back_median, None, False,
                        f"median vEgo the model sheds in the 10 s after you lift off (n={n_tb})"))
    rows.append(_sd_row("reoverride_pct", "Re-override within 15 s", "%",
                        sd.reoverride_pct, None, False,
                        "share of episodes followed by another within 15 s",
                        n_vals=len(sd.episodes)))
    return rows


def _sd_bucket_results(sd, bucket: str) -> list[MetricResult]:
    """Same rows for one mode/personality bucket (uniform across buckets so
    the breakdown table can align them by index)."""
    from .speed_disagreement import BucketStats

    st = sd.bucket_stats.get(bucket) or BucketStats()
    enough_time = sd.have_gas and st.seconds >= SD_MIN_SECONDS
    note = f"{st.n_eps} episodes over {st.seconds / 60:.1f} min"
    tb = float(np.median(st.taken_back)) if st.taken_back else None
    reov = float(np.mean(st.reoverride)) if st.reoverride else None
    return [
        _sd_row("gas_override_rate", "Gas-override episodes", "/10min",
                st.rate, SD_RATE_ANCHORS, enough_time, note),
        _sd_row("gas_override_pct", "Gas-override time", "%",
                st.pct, SD_PCT_ANCHORS, enough_time, note),
        _sd_row("gas_override_magnitude", "Gas-override magnitude", "m/s²",
                st.magnitude, SD_MAG_ANCHORS,
                st.n_mag_eps >= SD_MIN_MAG_EPISODES, note, n_vals=st.n_mag_eps),
        _sd_row("speed_taken_back", "Speed taken back after release", "m/s",
                tb, None, False, note, n_vals=len(st.taken_back)),
        _sd_row("reoverride_pct", "Re-override within 15 s", "%",
                reov, None, False, note, n_vals=len(st.reoverride)),
    ]


def grade_breakdowns(
    samples: dict, bucket_samples: dict, speed_disagreement=None
) -> dict[str, dict[str, BucketGrade]]:
    """Longitudinal grades per mode and per personality bucket.

    Bucket model samples are scored against the SAME overall human baseline;
    metrics gate at n>=3 per bucket like everywhere else.
    """
    weights = CATEGORY_GROUPS["Longitudinal"]
    long_defs = [m for m in METRICS if m.category in weights]
    out: dict[str, dict[str, BucketGrade]] = {"mode": {}, "personality": {}}
    for group_name, buckets in (("mode", MODE_BUCKETS), ("personality", PERSONALITY_BUCKETS)):
        for b in buckets:
            cats = {c: CategoryResult(c, w) for c, w in weights.items()}
            any_data = False
            for mdef in long_defs:
                mv = [v for v in bucket_samples.get(mdef.key, {}).get(b, []) if np.isfinite(v)]
                dv = [v for v in samples.get(mdef.key, {}).get("driver", []) if np.isfinite(v)]
                res = MetricResult(mdef, mv, dv)
                res.model_agg = _aggregate(mv, mdef.agg)
                res.driver_agg = _aggregate(dv, mdef.agg)
                res.score = _score_metric(res)
                if mv:
                    any_data = True
                cats[mdef.category].metrics.append(res)
            if speed_disagreement is not None:
                sd_rows = _sd_bucket_results(speed_disagreement, b)
                cats["Speed Disagreement"].metrics.extend(sd_rows)
                st = speed_disagreement.bucket_stats.get(b)
                if any(r.score is not None for r in sd_rows) or (st and st.n_eps):
                    any_data = True
            if not any_data:
                continue
            for cat in cats.values():
                scored = [m.score for m in cat.metrics if m.score is not None]
                cat.score = float(np.mean(scored)) if len(scored) >= MIN_SCORED_FOR_CATEGORY else None
            valid = [(c, weights[c.name]) for c in cats.values() if c.score is not None]
            score = (
                sum(c.score * w for c, w in valid) / sum(w for _c, w in valid)
                if valid else None
            )
            out[group_name][b] = BucketGrade(b, list(cats.values()), score)
    return out


def grade(
    samples: dict[str, dict[str, list[float]]],
    pingpong_score: float | None = None,
    pingpong_extra: dict | None = None,
    bucket_samples: dict | None = None,
    adherence: dict | None = None,
    t_follow_targets: dict | None = None,
    speed_disagreement_extra: dict | None = None,
    profile_info: dict | None = None,
    turn_in_extra: dict | None = None,
) -> GradeReport:
    cats: dict[str, CategoryResult] = {}
    for grp, weights in CATEGORY_GROUPS.items():
        for cat, w in weights.items():
            cats[cat] = CategoryResult(cat, w)

    for mdef in METRICS:
        mv = [v for v in samples.get(mdef.key, {}).get("model", []) if np.isfinite(v)]
        # dv is already the COMBINED (this-run + pooled-profile) list when a
        # driver profile is in play -- profile.py mutates samples[...]
        # ["driver"] in place before grade() runs, so MIN_EVENTS gating below
        # naturally benefits from pooling with no changes needed here.
        dv = [v for v in samples.get(mdef.key, {}).get("driver", []) if np.isfinite(v)]
        res = MetricResult(mdef, mv, dv)
        res.model_agg = _aggregate(mv, mdef.agg)
        res.driver_agg = _aggregate(dv, mdef.agg)
        res.score = _score_metric(res)
        info = (profile_info or {}).get(mdef.key)
        if info is not None:
            res.driver_vals_this_drive = list(info["this_drive"])
            if len(info["this_drive"]) >= MIN_EVENTS:
                res.same_drive_agg = _aggregate(info["this_drive"], mdef.agg)
        else:
            # profiling off, or this metric had no pooled contribution at
            # all: "this drive" IS the whole (unpooled) combined list, so
            # n_pooled correctly reads 0 rather than mistaking dv for pooled
            res.driver_vals_this_drive = list(dv)
        cats[mdef.category].metrics.append(res)

    if adherence and t_follow_targets:
        cats["Following"].metrics.extend(adherence_results(adherence, t_follow_targets))
        cats["Following"].extra["t_follow_targets"] = dict(t_follow_targets)

    sd = (speed_disagreement_extra or {}).get("result")
    if sd is not None:
        cats["Speed Disagreement"].metrics.extend(speed_disagreement_results(sd))
        cats["Speed Disagreement"].extra.update(speed_disagreement_extra)

    if turn_in_extra:
        cats["Turns"].extra.update(turn_in_extra)

    for cat in cats.values():
        scored = [m.score for m in cat.metrics if m.score is not None]
        cat.score = float(np.mean(scored)) if len(scored) >= MIN_SCORED_FOR_CATEGORY else None

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

    breakdowns = (
        grade_breakdowns(samples, bucket_samples, speed_disagreement=sd)
        if bucket_samples else {}
    )

    valid_g = [g for g in groups if g.score is not None]
    if valid_g:
        tw = sum(g.weight for g in valid_g)
        overall = sum(g.score * g.weight for g in valid_g) / tw
        return GradeReport(groups, overall, letter(overall), breakdowns)
    return GradeReport(groups, None, None, breakdowns)
