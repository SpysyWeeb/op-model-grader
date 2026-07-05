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
    grades: GradeReport | None = None


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


def analyze(per_drive: list[PerDrive]) -> Analysis:
    an = Analysis(per_drive=per_drive)
    by_name = {}
    for drive, seg, da, events in per_drive:
        by_name[drive.name] = events
        vm = vehicle_model_from_params(drive.meta.vm_params)
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

    an.samples = collect_samples(per_drive)
    add_turn_samples(an.samples, an.turns, an.intents)

    pp = an.pingpong
    an.grades = grade(
        an.samples,
        pingpong_score=pp.score if pp else None,
        pingpong_extra={"bins": pp.bins, "sub_bins": pp.sub_bins, "worst": pp.worst_bin} if pp else None,
    )
    return an
