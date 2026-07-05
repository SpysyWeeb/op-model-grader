"""Split a drive into engaged / manual spans on the carState timebase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import Drive, hold_align

MAX_GAP_S = 1.0  # break spans at inter-segment gaps longer than this
MIN_SPAN_S = 5.0  # ignore spans shorter than this
MANUAL_MIN_VEGO = 0.5  # a manual span must actually be driven


@dataclass
class Span:
    kind: str  # "engaged" | "manual"
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
    enabled: np.ndarray  # bool, aligned to t
    override: np.ndarray  # bool: gas/brake pressed while enabled
    spans: list[Span]

    def spans_of(self, kind: str) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]

    def time_of(self, kind: str) -> float:
        return sum(s.duration for s in self.spans_of(kind))


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


def segment_drive(drive: Drive) -> Segmentation | None:
    cs = drive.ch("vEgo")
    if cs is None:
        return None
    t = cs.t
    v = cs.v

    en_ch = drive.ch("enabled")
    if en_ch is not None:
        enabled = hold_align(en_ch, t, default=False).astype(bool)
    else:
        enabled = np.zeros(len(t), dtype=bool)

    gas = drive.ch("gasPressed")
    brk = drive.ch("brakePressed")
    gas_v = gas.v.astype(bool) if gas is not None and len(gas) == len(t) else np.zeros(len(t), bool)
    brk_v = brk.v.astype(bool) if brk is not None and len(brk) == len(t) else np.zeros(len(t), bool)
    override = enabled & (gas_v | brk_v)

    spans: list[Span] = []
    for kind, mask in (("engaged", enabled), ("manual", ~enabled)):
        runs = split_runs_at_gaps(t, _contiguous_runs(mask))
        for a, b in runs:
            if t[b - 1] - t[a] < MIN_SPAN_S:
                continue
            if kind == "manual" and v[a:b].max(initial=0.0) <= MANUAL_MIN_VEGO:
                continue  # parked / ignition-on idle, not driving
            spans.append(Span(kind, a, b, float(t[a]), float(t[b - 1])))

    spans.sort(key=lambda s: s.t0)
    return Segmentation(t=t, enabled=enabled, override=override, spans=spans)
