"""Speed Disagreement: the moments you and the model wanted different speeds.

openpilot's pedal asymmetry makes this measurable: pressing GAS while the
model controls longitudinal overrides it WITHOUT disengaging (you wanted to
go faster than the plan), while pressing BRAKE forces a disengagement (you
wanted to go slower, right now). Every instance is a labeled data point
about your tolerance for the model's pace.

IMPORTANT masking detail (verified on the owner's 0.11.2 logs): during a gas
override openpilot DROPS carControl.longActive while the car stays ENABLED
(controlsd: longActive = active and not overriding). So
"gasPressed & longActive" is ~empty in real logs; the model-in-long-control
mask must be

    long_ctl = longActive | (enabled & gasPressed)

which recovers the override moments (and degrades gracefully on forks or
old logs where longActive stays true through the press, and on the
per-axis-fallback path where long_model == enabled).

Detection:
- Gas-override episode: contiguous gasPressed & long_ctl samples, gaps
  < 1.0 s merged, minimum episode 0.3 s, split at inter-segment time gaps
  (the same project-wide rule segments.py uses for spans).
- Brake disengagement: a long_ctl True -> False transition with brakePressed
  True within +/- 0.5 s of the transition.

Each gas-override episode gets one context tag, priority order (first match
wins): launch (<=4 s after a standstill launch) > exp-slowdown (the vision
plan was already braking hard at onset -- pure judgment disagreement) >
lead-pullaway (a lead just ahead is accelerating away) > free-road (no lead)
> other.

Scoring note: rate and %-time are duration-weighted GLOBAL aggregates
(episodes / total model-long time), not medians of per-span rates --
overrides cluster heavily, and a median across engagement spans would swing
between 0 and huge depending on how the spans happen to be cut. The scored
rows are therefore built adherence-style in grading.speed_disagreement_results
from this module's totals, gated on minimum model-long time instead of the
usual n>=3 events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import (
    DriveArrays,
    Event,
    LAUNCH_MOTION_V,
    LAUNCH_STANDSTILL_HOLD_S,
    LAUNCH_STANDSTILL_V,
    STOP_STANDSTILL_V,
    _fill_short_false_gaps,
    _idx_after,
    _runs_min_dur,
    mode_tag,
    personality_tag,
)
from .segments import _contiguous_runs, split_runs_at_gaps

GAP_MERGE_S = 1.0  # merge gas-pressed gaps shorter than this into one episode
MIN_EPISODE_S = 0.3

LAUNCH_MAX_LAG_S = 4.0  # episode start within this long of a standstill launch
EXP_SLOWDOWN_ACCEL = -0.5  # m/s^2: vision plan already braking hard at onset
PULLAWAY_DREL = 40.0  # m
PULLAWAY_VREL = 0.5  # m/s: vLead - vEgo above this = pulling away
FREE_ROAD_DREL = 80.0  # m

REOVERRIDE_WITHIN_S = 15.0
TAKEN_BACK_HORIZON_S = 10.0

BRAKE_WINDOW_S = 0.5  # +/- around a long-ctl drop to call it brake-forced
STOP_FOLLOWS_WITHIN_S = 8.0
BRAKE_LEAD_DREL = 40.0  # m: lead within this at the transition = lead context

CONTEXTS = ("launch", "exp-slowdown", "lead-pullaway", "free-road", "other")
BUCKETS = ("chill", "experimental", "aggressive", "standard", "relaxed")


@dataclass
class GasOverrideEpisode:
    drive: str
    i0: int
    i1: int
    t0: float
    t1: float
    context: str
    magnitude: float | None  # median (aEgo - vis_accel) over override samples
    speed_taken_back: float | None
    reoverride: bool  # another episode follows within REOVERRIDE_WITHIN_S
    mode: str
    personality: str

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class BrakeDisengagement:
    drive: str
    idx: int
    t: float
    context: str  # "lead_or_stop" | "free_road"
    mode: str
    personality: str


@dataclass
class BucketStats:
    """Speed-disagreement totals inside one mode/personality bucket."""

    seconds: float = 0.0  # long-ctl seconds with this bucket active
    override_seconds: float = 0.0
    n_eps: int = 0
    mag_vals: list[float] = field(default_factory=list)  # per-sample aEgo - vis
    n_mag_eps: int = 0  # episodes with a magnitude value
    taken_back: list[float] = field(default_factory=list)
    reoverride: list[float] = field(default_factory=list)  # 0/100 per episode

    @property
    def rate(self) -> float | None:
        return self.n_eps / (self.seconds / 600.0) if self.seconds > 0 else None

    @property
    def pct(self) -> float | None:
        return 100.0 * self.override_seconds / self.seconds if self.seconds > 0 else None

    @property
    def magnitude(self) -> float | None:
        return float(np.median(self.mag_vals)) if self.mag_vals else None


@dataclass
class SpeedDisagreementResult:
    events: list[Event] = field(default_factory=list)  # kind="gas_override" drill-downs
    episodes: list[GasOverrideEpisode] = field(default_factory=list)
    brake_events: list[BrakeDisengagement] = field(default_factory=list)
    model_long_seconds: float = 0.0  # override-aware long-ctl seconds
    have_gas: bool = False  # gasPressed channel present in at least one drive
    have_vis: bool = False  # vision-plan accel present in at least one drive

    overall_rate: float | None = None  # episodes / 10 min long-ctl
    overall_pct: float | None = None  # % of long-ctl time with gas pressed
    overall_magnitude: float | None = None  # median (aEgo - vis) over override samples
    n_mag_samples: int = 0
    n_mag_episodes: int = 0
    speed_taken_back_median: float | None = None
    reoverride_pct: float | None = None
    brake_rate: float | None = None  # brake disengagements / 10 min long-ctl
    brake_context: dict[str, int] = field(default_factory=dict)  # context -> count
    context_table: list[dict] = field(default_factory=list)  # per CONTEXTS bucket
    bucket_stats: dict[str, BucketStats] = field(default_factory=dict)

    @property
    def biggest_context(self) -> str | None:
        if not self.context_table:
            return None
        top = max(self.context_table, key=lambda r: r["n"])
        return top["context"] if top["n"] > 0 else None


# ------------------------------------------------------------------ helpers


def long_ctl_mask(da: DriveArrays) -> np.ndarray:
    """Model-in-longitudinal-control, override-aware (see module docstring)."""
    if da.gas_pressed is None:
        return da.long_model
    return da.long_model | (da.enabled & da.gas_pressed)


def _launch_lag(t: np.ndarray, v: np.ndarray, i0: int) -> float | None:
    """Seconds since the closest preceding standstill->launch crossing, or
    None if i0 isn't within LAUNCH_MAX_LAG_S of one."""
    t0 = t[i0]
    lo = max(0, _idx_after(t, t0 - LAUNCH_MAX_LAG_S - LAUNCH_STANDSTILL_HOLD_S - 1.0))
    if i0 <= lo:
        return None
    wt, wv = t[lo : i0 + 1], v[lo : i0 + 1]
    best = None
    for a, b in _runs_min_dur(wt, wv < LAUNCH_STANDSTILL_V, LAUNCH_STANDSTILL_HOLD_S):
        moving = np.flatnonzero(wv[b:] > LAUNCH_MOTION_V)
        if len(moving) == 0:
            continue
        t_fm = wt[b + int(moving[0])]
        if t_fm > t0:
            continue
        lag = t0 - t_fm
        if lag <= LAUNCH_MAX_LAG_S and (best is None or lag < best):
            best = lag
    return best


def _classify_context(da: DriveArrays, i0: int) -> str:
    if _launch_lag(da.t, da.v, i0) is not None:
        return "launch"
    if da.vis_accel is not None and np.isfinite(da.vis_accel[i0]) and da.vis_accel[i0] < EXP_SLOWDOWN_ACCEL:
        return "exp-slowdown"
    has_lead = bool(da.lead_status is not None and da.lead_status[i0])
    d_rel = float(da.d_rel[i0]) if (has_lead and da.d_rel is not None) else None
    if has_lead and d_rel is not None and d_rel < PULLAWAY_DREL and da.v_lead is not None:
        v_lead = float(da.v_lead[i0])
        if (v_lead - float(da.v[i0])) > PULLAWAY_VREL:
            return "lead-pullaway"
    if da.lead_status is None:
        return "other"  # no radar data: can't claim a free road
    if not has_lead or (d_rel is not None and d_rel > FREE_ROAD_DREL):
        return "free-road"
    return "other"


def _mag_samples(da: DriveArrays, i0: int, i1: int) -> np.ndarray:
    """Finite (aEgo - vis_accel) samples over one episode window."""
    if da.vis_accel is None:
        return np.empty(0)
    vis = da.vis_accel[i0:i1]
    a = da.a[i0:i1]
    ok = np.isfinite(vis) & np.isfinite(a)
    return a[ok] - vis[ok]


def _magnitude(da: DriveArrays, i0: int, i1: int) -> float | None:
    vals = _mag_samples(da, i0, i1)
    return float(np.median(vals)) if len(vals) else None


def _speed_taken_back(da: DriveArrays, i1: int) -> float | None:
    """vEgo at release minus the lowest vEgo in the next 10 s, truncated at
    the first new gas press, loss of long control, inter-segment gap, or
    data end. "Still in control" is enabled|longActive: longActive takes a
    few frames to come back after the pedal is released."""
    t, v = da.t, da.v
    if i1 <= 0 or i1 >= len(t):
        return None
    still_ctl = da.long_model | da.enabled
    end_t = t[i1 - 1] + TAKEN_BACK_HORIZON_S
    limit = min(len(t), _idx_after(t, end_t))
    idx = i1
    while idx < limit:
        if not still_ctl[idx]:
            break
        if da.gas_pressed is not None and da.gas_pressed[idx]:
            break
        if t[idx] - t[idx - 1] > 1.0:
            break
        idx += 1
    if idx - i1 < 2:
        return None
    v_release = float(v[i1 - 1])
    v_min = float(np.min(v[i1:idx]))
    return v_release - v_min


# ------------------------------------------------------------------ detect


def detect_gas_overrides(drive_name: str, da: DriveArrays) -> list[GasOverrideEpisode]:
    if da.gas_pressed is None:
        return []
    t = da.t
    mask = long_ctl_mask(da) & da.gas_pressed
    filled = _fill_short_false_gaps(t, mask, GAP_MERGE_S)
    runs = [
        (a, b)
        for a, b in split_runs_at_gaps(t, _contiguous_runs(filled))
        if t[b - 1] - t[a] >= MIN_EPISODE_S
    ]
    episodes: list[GasOverrideEpisode] = []
    for a, b in runs:
        episodes.append(
            GasOverrideEpisode(
                drive=drive_name,
                i0=a,
                i1=b,
                t0=float(t[a]),
                t1=float(t[b - 1]),
                context=_classify_context(da, a),
                magnitude=_magnitude(da, a, b),
                speed_taken_back=_speed_taken_back(da, b),
                reoverride=False,
                mode=mode_tag(da.exp_mode, a, b),
                personality=personality_tag(da.personality, a, b),
            )
        )
    episodes.sort(key=lambda e: e.t0)
    for i, ep in enumerate(episodes):
        nxt = episodes[i + 1] if i + 1 < len(episodes) else None
        ep.reoverride = bool(nxt is not None and (nxt.t0 - ep.t1) <= REOVERRIDE_WITHIN_S)
    return episodes


def detect_brake_disengagements(drive_name: str, da: DriveArrays) -> list[BrakeDisengagement]:
    if da.brake_pressed is None:
        return []
    lm = long_ctl_mask(da)
    t = da.t
    if len(lm) < 2:
        return []
    falls = np.flatnonzero(lm[:-1] & ~lm[1:]) + 1  # first False sample after a True run
    out: list[BrakeDisengagement] = []
    for idx in falls:
        idx = int(idx)
        t_trans = t[idx]
        lo = max(0, _idx_after(t, t_trans - BRAKE_WINDOW_S))
        hi = min(len(t), _idx_after(t, t_trans + BRAKE_WINDOW_S) + 1)
        if not da.brake_pressed[lo:hi].any():
            continue
        stop_hi = min(len(t), _idx_after(t, t_trans + STOP_FOLLOWS_WITHIN_S))
        stop_follows = bool(np.any(da.v[idx:stop_hi] < STOP_STANDSTILL_V)) if stop_hi > idx else False
        has_lead_now = bool(
            da.lead_status is not None
            and da.lead_status[idx]
            and da.d_rel is not None
            and float(da.d_rel[idx]) < BRAKE_LEAD_DREL
        )
        context = "lead_or_stop" if (has_lead_now or stop_follows) else "free_road"
        out.append(
            BrakeDisengagement(
                drive=drive_name,
                idx=idx,
                t=float(t_trans),
                context=context,
                mode=mode_tag(da.exp_mode, idx, idx + 1),
                personality=personality_tag(da.personality, idx, idx + 1),
            )
        )
    return out


def _mask_seconds(t: np.ndarray, mask: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    dt = np.clip(np.diff(t, append=t[-1]), 0.0, 0.05)
    return float(np.sum(dt[mask]))


EVENT_PAD_BEFORE_S = 2.0
EVENT_PAD_AFTER_S = 10.0  # covers the speed-taken-back horizon in the trace


def _episode_event(ep: GasOverrideEpisode, da: DriveArrays) -> Event:
    # pad the drill-down window so the trace shows the run-up and the model
    # taking the speed back; the episode itself is values["i_onset".."i_end"]
    i0 = max(0, _idx_after(da.t, ep.t0 - EVENT_PAD_BEFORE_S))
    i1 = min(len(da.t), _idx_after(da.t, ep.t1 + EVENT_PAD_AFTER_S) + 1)
    return Event(
        kind="gas_override",
        engaged=True,
        drive=ep.drive,
        t0=float(da.t[i0]),
        t1=float(da.t[i1 - 1]),
        i0=i0,
        i1=i1,
        has_override=True,
        values={
            "context": ep.context,
            "magnitude": ep.magnitude,
            "speed_taken_back": ep.speed_taken_back,
            "reoverride": 100.0 if ep.reoverride else 0.0,
            "mode": ep.mode,
            "personality": ep.personality,
            "i_onset": ep.i0,  # true episode bounds (padding excluded)
            "i_end": ep.i1,
        },
    )


def _bucket_masks(da: DriveArrays) -> list[tuple[str, np.ndarray]]:
    out = []
    if da.exp_mode is not None:
        out += [("experimental", da.exp_mode), ("chill", ~da.exp_mode)]
    if da.personality is not None:
        out += [
            (name, da.personality == idx)
            for idx, name in ((0, "aggressive"), (1, "standard"), (2, "relaxed"))
        ]
    return out


def _accumulate_bucket_stats(
    sd: SpeedDisagreementResult,
    da: DriveArrays,
    episodes: list[GasOverrideEpisode],
) -> None:
    """Per mode/personality bucket: long-ctl time, override time, episode
    counts and per-episode values. Time masks attribute per sample; episodes
    attribute by their (90%-constancy) tag, mixed/unknown excluded — the
    same rules as the other longitudinal breakdowns."""
    long_ctl = long_ctl_mask(da)
    override = long_ctl & da.gas_pressed if da.gas_pressed is not None else None
    for bname, bmask in _bucket_masks(da):
        st = sd.bucket_stats.setdefault(bname, BucketStats())
        st.seconds += _mask_seconds(da.t, long_ctl & bmask)
        if override is not None:
            st.override_seconds += _mask_seconds(da.t, override & bmask)
    for ep in episodes:
        for bname in (ep.mode, ep.personality):
            if bname not in BUCKETS:
                continue
            st = sd.bucket_stats.setdefault(bname, BucketStats())
            st.n_eps += 1
            vals = _mag_samples(da, ep.i0, ep.i1)
            if len(vals):
                st.mag_vals.extend(float(v) for v in vals)
                st.n_mag_eps += 1
            if ep.speed_taken_back is not None:
                st.taken_back.append(float(ep.speed_taken_back))
            st.reoverride.append(100.0 if ep.reoverride else 0.0)


def analyze_speed_disagreement(per_drive) -> SpeedDisagreementResult:
    """per_drive: list[(Drive, Segmentation, DriveArrays, list[Event])]."""
    sd = SpeedDisagreementResult()
    all_episodes: list[GasOverrideEpisode] = []
    all_brakes: list[BrakeDisengagement] = []
    override_seconds = 0.0
    mag_all: list[np.ndarray] = []

    for drive, _seg, da, _events in per_drive:
        long_ctl = long_ctl_mask(da)
        sd.model_long_seconds += _mask_seconds(da.t, long_ctl)
        sd.have_gas = sd.have_gas or da.gas_pressed is not None
        sd.have_vis = sd.have_vis or da.vis_accel is not None
        eps = detect_gas_overrides(drive.name, da)
        all_episodes.extend(eps)
        sd.events.extend(_episode_event(ep, da) for ep in eps)
        all_brakes.extend(detect_brake_disengagements(drive.name, da))
        for ep in eps:
            vals = _mag_samples(da, ep.i0, ep.i1)
            if len(vals):
                mag_all.append(vals)
                sd.n_mag_episodes += 1
        if da.gas_pressed is not None:
            override_seconds += _mask_seconds(da.t, long_ctl & da.gas_pressed)
        _accumulate_bucket_stats(sd, da, eps)

    sd.episodes = all_episodes
    sd.brake_events = all_brakes

    min10 = sd.model_long_seconds / 600.0
    if min10 > 0 and sd.have_gas:
        sd.overall_rate = len(all_episodes) / min10
        sd.overall_pct = 100.0 * override_seconds / sd.model_long_seconds
        sd.brake_rate = len(all_brakes) / min10

    if mag_all:
        allv = np.concatenate(mag_all)
        sd.overall_magnitude = float(np.median(allv))
        sd.n_mag_samples = int(len(allv))

    taken_back = [e.speed_taken_back for e in all_episodes if e.speed_taken_back is not None]
    if taken_back:
        sd.speed_taken_back_median = float(np.median(taken_back))

    if all_episodes:
        sd.reoverride_pct = 100.0 * sum(1 for e in all_episodes if e.reoverride) / len(all_episodes)

    for ctx in ("lead_or_stop", "free_road"):
        sd.brake_context[ctx] = sum(1 for b in all_brakes if b.context == ctx)

    for ctx in CONTEXTS:
        ctx_eps = [e for e in all_episodes if e.context == ctx]
        ctx_mags = [e.magnitude for e in ctx_eps if e.magnitude is not None]
        sd.context_table.append(
            {
                "context": ctx,
                "n": len(ctx_eps),
                "median_magnitude": float(np.median(ctx_mags)) if ctx_mags else None,
            }
        )

    return sd
