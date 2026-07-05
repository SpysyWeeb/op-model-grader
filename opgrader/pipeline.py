"""Glue: run lateral analysis on decoded drives and produce the final grades."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import DriveArrays, Event
from .extract import Drive
from .grading import (
    GradeReport,
    METRIC_BY_KEY,
    add_turn_samples,
    collect_samples,
    grade,
    score_ratio,
)
from .lateral import (
    IntentWindow,
    PingPongResult,
    TurnEpisode,
    analyze_pingpong,
    detect_intent_windows,
    detect_turn_episodes,
)
from .segments import Segmentation
from .vehicle_model import vehicle_model_from_params

PerDrive = tuple[Drive, Segmentation, DriveArrays, list[Event]]


@dataclass
class Analysis:
    per_drive: list[PerDrive]
    turns: list[TurnEpisode] = field(default_factory=list)
    intents: list[IntentWindow] = field(default_factory=list)
    pingpong: PingPongResult | None = None
    samples: dict = field(default_factory=dict)
    bucket_samples: dict = field(default_factory=dict)
    adherence: dict = field(default_factory=dict)  # personality -> stats
    t_follow_targets: dict = field(default_factory=dict)
    bucket_times: dict = field(default_factory=dict)  # bucket -> model-long seconds
    model_id: dict | None = None
    counterfactual: object | None = None  # counterfactual.Counterfactual
    speed_disagreement: object | None = None  # speed_disagreement.SpeedDisagreementResult
    profile_summary: object | None = None  # profile.ProfileSummary
    grades: GradeReport | None = None


def _follow_adherence(per_drive: list[PerDrive]) -> dict[str, dict]:
    """Median effective t_follow per ACTIVE personality over steady-follow
    samples (model-long active, vEgo > 8, lead present) inside detected
    follow windows."""
    from .metrics import effective_t_follow

    by_p: dict[str, list] = {p: [] for p in ("aggressive", "standard", "relaxed")}
    dt_by_p: dict[str, float] = {p: 0.0 for p in by_p}
    for _drive, _seg, da, events in per_drive:
        if da.d_rel is None or da.v_lead is None or da.personality is None:
            continue
        dt = float(np.median(np.diff(da.t))) if len(da.t) > 1 else 0.01
        for ev in events:
            if ev.kind != "follow" or not ev.engaged:
                continue
            sl = slice(ev.i0, ev.i1)
            v = da.v[sl]
            vl = da.v_lead[sl]
            # steady-follow only: the MPC distance inversion assumes steady
            # state, so drop approach/pull-away transients
            mask = (
                da.long_model[sl]
                & (v > 8.0)
                & (da.lead_status[sl] if da.lead_status is not None else True)
                & (np.abs(vl - v) < 1.5)
                & (np.abs(da.a[sl]) < 0.5)
            )
            if not mask.any():
                continue
            eff = effective_t_follow(v, vl, da.d_rel[sl])
            pers = da.personality[sl]
            for idx, name in ((0, "aggressive"), (1, "standard"), (2, "relaxed")):
                m = mask & (pers == idx) & np.isfinite(eff)
                if m.any():
                    by_p[name].append(eff[m])
                    dt_by_p[name] += float(m.sum()) * dt
    out = {}
    for name, chunks in by_p.items():
        if chunks:
            allv = np.concatenate(chunks)
            out[name] = {
                "median_eff": float(np.median(allv)),
                "seconds": dt_by_p[name],
                "n_samples": int(len(allv)),
            }
    return out


def _bucket_times(per_drive: list[PerDrive]) -> dict[str, float]:
    """Model-longitudinal seconds per mode and personality bucket."""
    out = {b: 0.0 for b in ("chill", "experimental",
                            "aggressive", "standard", "relaxed")}
    for _drive, _seg, da, _events in per_drive:
        if len(da.t) < 2:
            continue
        dt = np.clip(np.diff(da.t, append=da.t[-1]), 0.0, 0.05)
        lm = da.long_model
        if da.exp_mode is not None:
            out["experimental"] += float(np.sum(dt[lm & da.exp_mode]))
            out["chill"] += float(np.sum(dt[lm & ~da.exp_mode]))
        if da.personality is not None:
            for idx, name in ((0, "aggressive"), (1, "standard"), (2, "relaxed")):
                out[name] += float(np.sum(dt[lm & (da.personality == idx)]))
    return {k: v for k, v in out.items() if v >= 1.0}  # drop sub-second slivers


def _turn_event(ep: TurnEpisode, da: DriveArrays) -> Event:
    # include ~6 s past the run end so the unwind/overshoot is in the trace
    end_t = da.t[min(ep.i1, len(da.t) - 1)] + 6.0
    i1 = min(len(da.t), int(np.searchsorted(da.t, end_t)))
    return Event(
        kind="turn",
        engaged=ep.engaged,
        drive=ep.drive,
        t0=float(da.t[ep.i0]),
        t1=float(da.t[i1 - 1]),
        i0=ep.i0,
        i1=i1,
        has_override=ep.contaminated,
        values={
            "side": ep.side,
            "sharp": ep.sharp,
            "band": ep.band,
            "peak_deg": round(abs(ep.peak_act), 1),
            "overshoot_pct": ep.overshoot_pct,
            "unwind_rate": ep.unwind_rate,
            "rescued": ep.rescued,
            "wobbles": ep.wobbles,
        },
    )


def _intent_event(w: IntentWindow) -> Event:
    return Event(
        kind="intent",
        engaged=w.engaged,
        drive=w.drive,
        t0=w.t_on,
        t1=w.t_end,
        i0=w.i0,
        i1=w.i1,
        has_override=False,
        values={
            "side": w.side,
            "outcome": w.outcome,
            "heading_deg": w.heading_deg,
            "delay": w.delay,
            "missed": w.missed,
        },
    )


def analyze(
    per_drive: list[PerDrive],
    t_follow_targets: dict | None = None,
    use_profile: bool = True,
) -> Analysis:
    from .config import DEFAULT_T_FOLLOW

    an = Analysis(per_drive=per_drive)
    an.t_follow_targets = dict(t_follow_targets or DEFAULT_T_FOLLOW)
    by_name = {}
    vms: dict[str, object] = {}
    for drive, seg, da, events in per_drive:
        by_name[drive.name] = events
        vm = vehicle_model_from_params(drive.meta.vm_params)
        vms[drive.name] = vm
        turns = detect_turn_episodes(drive.name, seg, da)
        intents = detect_intent_windows(drive.name, seg, da, vm)
        an.turns += turns
        an.intents += intents
        events.extend(_turn_event(ep, da) for ep in turns)
        events.extend(_intent_event(w) for w in intents)

    def pp_score(m, d):
        return score_ratio(m, d, "lower", METRIC_BY_KEY["steer_rate_rms"].eps)

    an.pingpong = analyze_pingpong(
        [(d.name, s, a) for d, s, a, _e in per_drive], pp_score
    )
    if an.pingpong:
        for ev in an.pingpong.worst_windows:
            if ev.drive in by_name:
                by_name[ev.drive].append(ev)

    # counterfactual "Plan vs You" (manual time only; unscored)
    from .counterfactual import analyze_counterfactual

    an.counterfactual, cf_events = analyze_counterfactual(
        per_drive, an.turns, an.intents, an.t_follow_targets, vms
    )
    for ev in cf_events:
        if ev.drive in by_name:
            by_name[ev.drive].append(ev)

    # Speed Disagreement: gas overrides (without disengaging) and brake-forced
    # disengagements -- both directions of "I wanted a different speed than
    # the model".
    from .speed_disagreement import analyze_speed_disagreement

    an.speed_disagreement = analyze_speed_disagreement(per_drive)
    for ev in an.speed_disagreement.events:
        if ev.drive in by_name:
            by_name[ev.drive].append(ev)

    an.samples, an.bucket_samples = collect_samples(per_drive)
    add_turn_samples(an.samples, an.turns, an.intents)
    an.adherence = _follow_adherence(per_drive)
    an.bucket_times = _bucket_times(per_drive)

    # Driver profile: pool this fingerprint's accumulated manual-driving
    # baseline into an.samples (driver side only, see profile.py) and
    # upsert/persist this run's own per-route contribution. --no-profile
    # skips both the read and the write.
    profile_info: dict = {}
    if use_profile:
        from . import profile as P

        an.profile_summary, profile_info = P.pool_for_grading(an, per_drive, pp_score)
    else:
        from .profile import ProfileSummary

        an.profile_summary = ProfileSummary(used=False)

    # best-effort driving-model identification (cached; never blocks grading)
    try:
        from . import modelid

        seen: dict[tuple, dict] = {}
        for drive, _s, _a, _e in per_drive:
            key = (drive.meta.git_remote, drive.meta.git_commit,
                   tuple(sorted(drive.meta.model_params.items())))
            if key not in seen:
                seen[key] = modelid.resolve(drive.meta)
        results = list(seen.values())
        an.model_id = results[0] if len(results) == 1 else {
            "label": " / ".join(sorted({r["label"] for r in results})),
            "provenance": "multiple builds",
            "sha256": None,
        }
    except Exception:
        an.model_id = None

    pp = an.pingpong
    an.grades = grade(
        an.samples,
        pingpong_score=pp.score if pp else None,
        pingpong_extra={"bins": pp.bins, "sub_bins": pp.sub_bins, "worst": pp.worst_bin} if pp else None,
        bucket_samples=an.bucket_samples,
        adherence=an.adherence,
        t_follow_targets=an.t_follow_targets,
        speed_disagreement_extra={"result": an.speed_disagreement} if an.speed_disagreement else None,
        profile_info=profile_info,
    )
    return an
