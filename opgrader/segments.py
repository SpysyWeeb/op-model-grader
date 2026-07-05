"""Per-axis control segmentation on the carState timebase.

Always-On-Lateral (AOL) / MADS forks let the model steer while the human
works gas and brake. A single engaged flag would mis-attribute both axes
there, so control is tracked per axis:

- lat_model  = carControl.latActive  (model is steering right now)
- long_model = carControl.longActive (model is doing gas/brake)

Old logs without latActive/longActive fall back to the single enabled flag
for both axes (per_axis=False, noted in the report). Longitudinal metrics
attribute by long_model, lateral metrics by lat_model, so AOL time feeds the
HUMAN longitudinal baseline and the MODEL lateral side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import Drive, hold_align

MAX_GAP_S = 1.0  # break spans at inter-segment gaps longer than this
MIN_SPAN_S = 5.0  # ignore spans shorter than this
MANUAL_MIN_VEGO = 0.5  # a manual span must actually be driven


@dataclass
class Span:
    kind: str  # "engaged" | "manual"  (w.r.t. one axis)
    i0: int  # start index into carState-aligned arrays (inclusive)
    i1: int  # end index (exclusive)
    t0: float
    t1: float

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class Segmentation:
    t: np.ndarray  # carState timebase
    enabled: np.ndarray  # bool, single engaged flag (for reference/UI)
    lat_model: np.ndarray  # bool: model controls steering
    long_model: np.ndarray  # bool: model controls gas/brake
    long_override: np.ndarray  # gas/brake pressed while long_model
    lat_override: np.ndarray  # steeringPressed while lat_model
    per_axis: bool  # True if real latActive/longActive were available
    long_spans: list[Span]
    lat_spans: list[Span]

    def spans_of(self, kind: str, axis: str = "long") -> list[Span]:
        spans = self.long_spans if axis == "long" else self.lat_spans
        return [s for s in spans if s.kind == kind]

    def time_of(self, kind: str, axis: str = "long") -> float:
        return sum(s.duration for s in self.spans_of(kind, axis))

    def bucket_times(self) -> dict[str, float]:
        """Seconds in each control bucket (while the car is being driven)."""
        if len(self.t) == 0:
            return {"both": 0.0, "lat_only": 0.0, "long_only": 0.0, "manual": 0.0}
        dt = np.diff(self.t, append=self.t[-1])
        dt = np.clip(dt, 0.0, 0.05)  # cap so inter-segment gaps don't inflate
        lat, lng = self.lat_model, self.long_model
        return {
            "both": float(np.sum(dt[lat & lng])),
            "lat_only": float(np.sum(dt[lat & ~lng])),
            "long_only": float(np.sum(dt[~lat & lng])),
            "manual": float(np.sum(dt[~lat & ~lng])),
        }


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if len(mask) == 0:
        return []
    m = mask.astype(np.int8)
    d = np.diff(m)
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def split_runs_at_gaps(
    t: np.ndarray, runs: list[tuple[int, int]], max_gap: float = MAX_GAP_S
) -> list[tuple[int, int]]:
    """Break index runs wherever the timebase jumps by more than max_gap."""
    if len(t) < 2:
        return runs
    gap_after = np.flatnonzero(np.diff(t) > max_gap)  # gap between i and i+1
    out = []
    for a, b in runs:
        cuts = gap_after[(gap_after >= a) & (gap_after < b - 1)]
        prev = a
        for c in cuts:
            out.append((prev, int(c) + 1))
            prev = int(c) + 1
        out.append((prev, b))
    return [(a, b) for a, b in out if b > a]


def _build_spans(t: np.ndarray, v: np.ndarray, flag: np.ndarray) -> list[Span]:
    spans: list[Span] = []
    for kind, mask in (("engaged", flag), ("manual", ~flag)):
        for a, b in split_runs_at_gaps(t, _contiguous_runs(mask)):
            if t[b - 1] - t[a] < MIN_SPAN_S:
                continue
            if kind == "manual" and v[a:b].max(initial=0.0) <= MANUAL_MIN_VEGO:
                continue  # parked / ignition-on idle, not driving
            spans.append(Span(kind, a, b, float(t[a]), float(t[b - 1])))
    spans.sort(key=lambda s: s.t0)
    return spans


def segment_drive(drive: Drive) -> Segmentation | None:
    cs = drive.ch("vEgo")
    if cs is None:
        return None
    t = cs.t
    v = cs.v

    def aligned_bool(name):
        ch = drive.ch(name)
        if ch is None:
            return None
        return hold_align(ch, t, default=False).astype(bool)

    enabled = aligned_bool("enabled")
    if enabled is None:
        enabled = np.zeros(len(t), dtype=bool)

    lat_model = aligned_bool("latActive")
    long_model = aligned_bool("longActive")
    per_axis = lat_model is not None and long_model is not None and (
        lat_model.any() or long_model.any() or not enabled.any()
    )
    if not per_axis:
        # old logs: no per-axis actuator flags -> single-flag behavior
        lat_model = enabled.copy()
        long_model = enabled.copy()
        if "per-axis control flags unavailable (old log); using enabled for both axes" not in drive.meta.notes:
            drive.meta.notes.append(
                "per-axis control flags unavailable (old log); using enabled for both axes"
            )

    def aligned_pressed(name):
        ch = drive.ch(name)
        return (
            ch.v.astype(bool)
            if ch is not None and len(ch) == len(t)
            else (hold_align(ch, t, default=False).astype(bool) if ch is not None else np.zeros(len(t), bool))
        )

    gas = aligned_pressed("gasPressed")
    brk = aligned_pressed("brakePressed")
    sp = aligned_pressed("steeringPressed")
    long_override = long_model & (gas | brk)
    lat_override = lat_model & sp

    return Segmentation(
        t=t,
        enabled=enabled,
        lat_model=lat_model,
        long_model=long_model,
        long_override=long_override,
        lat_override=lat_override,
        per_axis=per_axis,
        long_spans=_build_spans(t, v, long_model),
        lat_spans=_build_spans(t, v, lat_model),
    )
