"""Lateral analysis: ping-pong oscillation, turn episodes, turn-in timing.

Definitions mirror the owner's on-device analyzer so numbers are comparable:
- turn onset when |steeringAngleDeg| (or |commanded angle|) crosses 20 deg
- unwind point = first time after a signal's peak it falls to <= 50% of peak
- sharp turn = peak >= 90 deg with speed at onset < 15 mph (6.7 m/s)
- positive steeringAngleDeg = LEFT (ISO sign convention)

All detection happens inside engaged/manual spans (which never cross
inter-segment time gaps > 1 s), except blinker turn-intent windows, which are
detected on the whole timeline (a disengagement inside the window is part of
the signal) and discarded if they cross a gap.

One of Turns' scored metrics is cmd-vs-actual DIVERGENCE during genuine
driver resistance, not torque-ceiling status: a continuously-replanning
vision model rarely "refuses" a turn outright, so what actually matters is
how far the realized path ended up from what the model wanted while the
driver was actively fighting it. "Genuine resistance" = steeringPressed
True AND the driver's torque sign opposes the model's own commanded torque
(torqueState.output), sustained >= CONFLICT_MIN_S -- not merely a hand
resting on the wheel. torque_output/initiator/torque_ceiling_* are kept as
DESCRIPTIVE/diagnostic context (who moved first, was the model already at
its output limit) but never gate whether an episode counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import DriveArrays, Event, _contiguous_runs, _idx_after
from .grading import MIN_SCORED_FOR_CATEGORY, score_absolute
from .metrics import rms, smooth
from .segments import Segmentation

MPH = 0.44704
TURN_ONSET_DEG = 20.0
UNWIND_FRACTION = 0.5
TURN_PEAK_MIN_DEG = 90.0  # sharp-turn peak threshold
SHARP_MAX_ONSET_V = 15 * MPH  # 6.7 m/s
OVERSHOOT_WINDOW_S = 4.0
WOBBLE_MIN_DEG = 5.0

# initiator / torque-ceiling classification (who steered a sharp turn, and if
# the driver did, was the model already commanding max torque beforehand).
# Constants verified against real Palisade data -- see detect_turn_episodes.
INITIATOR_TOL_S = 0.15  # "at/before onset" tolerance for driver- or model-led
CEILING_FRAC = 0.92  # |torque_output| at/above this counts as "at the ceiling"
CEILING_MIN_S = 0.3  # ceiling must be sustained this long to count as real
MIN_PRE_OVERRIDE_S = 0.3  # need at least this much pre-override data to judge at all
CONFLICT_MIN_S = 0.3  # sustained opposition needed to count as genuine resistance,
                      # not an incidental hand-on-wheel moment

PP_BINS_MPH: list[tuple[float, float]] = [
    (0, 5), (5, 10), (10, 20), (20, 35), (35, 55), (55, 200),
]
# Ping-pong is FAST: a real back-and-forth completes in ~2 s; anything slower
# is the car holding a curving road, not hunting (owner's own definition,
# confirmed on real data -- the slow ~5-9 s weave is road-following, present on
# good and bad models alike). So the detrend stays tight: band = smooth(SHORT)
# - smooth(LONG) keeps oscillation with period ~PP_BAND_SHORT..2xPP_BAND_LONG
# s. smooth(LONG=2s) removes the slow intended path (turns, road-following);
# smooth(SHORT) drops sensor noise. The number that actually distinguishes
# ping-pong is the REVERSAL RATE on this residual ("how quickly the wheel
# saws"): genuine fast ping-pong runs 40-60 reversals/min (period ~1-2 s);
# road-following weave is few/slow. This is essentially the original 2 s
# high-pass -- the measurement was never the bug.
PP_BAND_SHORT_S = 0.3
PP_BAND_LONG_S = 2.0
PP_SWING_DEG = 3.0  # min swing between extrema to count a reversal
PP_MIN_BIN_S = 30.0  # per side, to score a speed bin
PP_SUB_BIN_MIN_S = 60.0  # engaged seconds needed for a 1 mph sub-bin row
PP_WORST_WINDOW_S = 10.0
# Category score = PP_WORST_BLEND * worst bin + (1-PP_WORST_BLEND) * the
# engaged-time-weighted mean. Ping-pong is felt as episodes: a route that saws
# badly in one speed range but is calm elsewhere should not average that away.
PP_WORST_BLEND = 0.4

# The REAL fix: bins with no manual baseline (you rarely hand-steer above
# ~20 mph) could never be scored, so ping-pong at speed was invisible in the
# grade even though the metric measured it. These absolute anchors (osc RMS
# deg, reversal rate /min at 100/50/0) score such bins directly. Applied only
# to 10-20 mph and up: below that, tight low-speed maneuvering (parking) looks
# identical to ping-pong on any hands-off metric and genuinely needs the
# human ratio baseline. Calibrated from 3 real Palisade routes / 73 segments
# (tight-window pooled): a steady model runs ~1-2 reversals/min on the highway
# and 5-35/min in town; genuine fast ping-pong pushes reversal rate well above
# that (route 27: 50/min at 10-20, individual episodes 60+). Reversal rate is
# weighted as the primary signal; amplitude is secondary (it double-counts
# legitimate maneuvering at low speed). Keyed by the (lo, hi) mph tuple.
PP_ABS_REV_ANCHORS: dict[tuple[float, float], tuple[float, float, float]] = {
    (10, 20): (20.0, 45.0, 75.0),
    (20, 35): (5.0, 20.0, 40.0),
    (35, 55): (3.0, 15.0, 30.0),
}
PP_ABS_RMS_ANCHORS: dict[tuple[float, float], tuple[float, float, float]] = {
    (10, 20): (2.5, 5.5, 9.0),
    (20, 35): (1.0, 2.8, 5.0),
    (35, 55): (0.6, 1.6, 3.0),
}

INTENT_MAX_V = 20 * MPH  # 8.9 m/s
INTENT_WINDOW_S = 20.0
INTENT_OFF_PAD_S = 5.0
INTENT_TURN_DEG = 45.0  # |net heading| >= this => intersection turn
INTENT_LANE_DEG = 20.0  # < this => lane change


# ------------------------------------------------------------------ helpers


def pingpong_band(t: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Tight band-pass isolating fast ping-pong: smooth(SHORT) - smooth(LONG).
    smooth(LONG=2s) is the intended path (turns, road-following) and is
    subtracted off; smooth(SHORT) strips sensor noise. Keeps only the fast
    back-and-forth (period up to ~2xPP_BAND_LONG s) -- a slower weave is the
    car holding a curving road, not hunting, and is deliberately excluded."""
    return smooth(t, angle, PP_BAND_SHORT_S) - smooth(t, angle, PP_BAND_LONG_S)


def swing_reversal_count(x: np.ndarray, thresh: float = PP_SWING_DEG) -> int:
    """Direction reversals where the swing between successive extrema > thresh.

    Zig-zag walk: commit to a direction once the signal moves > thresh away
    from the running extreme; each committed direction change after the first
    counts as one reversal.
    """
    if len(x) < 2:
        return 0
    count = 0
    direction = 0  # +1 rising, -1 falling, 0 uncommitted
    ext = x[0]  # running extreme of the current direction
    for v in x[1:]:
        if direction == 0:
            if v - ext > thresh:
                direction = 1
                ext = v
            elif ext - v > thresh:
                direction = -1
                ext = v
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext - v > thresh:
                count += 1
                direction = -1
                ext = v
        else:
            if v < ext:
                ext = v
            elif v - ext > thresh:
                count += 1
                direction = 1
                ext = v
    return count


def _first_idx(mask: np.ndarray, start: int = 0) -> int | None:
    idx = np.flatnonzero(mask[start:])
    return int(start + idx[0]) if len(idx) else None


# ------------------------------------------------------------- turn episodes


@dataclass
class TurnEpisode:
    engaged: bool
    drive: str
    side: str  # "left" | "right"
    sharp: bool  # peak >= 90 deg and onset speed < 6.7 m/s
    band: str  # "20-90" | "90-150" | "150+"
    i0: int
    i1: int
    t_onset: float
    v_onset: float
    peak_act: float  # signed deg
    peak_cmd: float | None
    t_peak_act: float
    contaminated: bool  # steeringPressed/override before the actual peak
    rescued: bool = False  # driver grabbed wheel / disengaged during unwind
    unwind_rate: float | None = None  # deg/s, peak to |act|<=20
    overshoot_pct: float | None = None
    overshoot_deg: float | None = None
    wobbles: int | None = None
    cmd_unwind_lead: float | None = None  # cmd 50% crossing minus act 50% crossing
    cmd_onset_lead: float | None = None  # cmd's first |>=20deg| crossing minus t_onset
    never_commanded: bool = False  # engaged: wheel turned sharply but cmd never called for it
    # who steered this turn in, and (if the driver did) was there a torque-
    # ceiling defense for the model -- engaged episodes only, see
    # detect_turn_episodes for the exact classification rules
    override_onset_t: float | None = None  # first steeringPressed within the episode window
    initiator: str = "unknown"  # "model" | "driver" | "lag" | "unknown" -- DESCRIPTIVE, not scored
    torque_ceiling_pre_override: bool | None = None  # diagnostic only, see module docstring
    torque_ceiling_direction_agrees: bool | None = None  # only meaningful when the above is True
    # Peak |torque_output| (0..1) reached BEFORE override_onset_t (or across
    # the whole episode if never overridden) -- "is the model actually
    # committing", not just "did it eventually cross some ceiling". Shown,
    # not scored: a low value could mean the model isn't trying, but it can
    # just as easily mean this particular turn never needed much torque --
    # torque alone can't distinguish those, so this is context, not a grade.
    peak_effort_frac: float | None = None
    # SCORED quantity: peak |actual - commanded| angle during a sustained
    # window of genuine driver resistance (steering torque opposing the
    # model's own commanded torque, sustained >= CONFLICT_MIN_S). None if no
    # such window exists in this episode -- nothing scored against the model
    # when there was no real disagreement to diverge from.
    divergence_deg: float | None = None
    conflict_duration_s: float | None = None  # duration of the window divergence_deg came from
    conflict_ceiling: bool | None = None  # diagnostic: was torque_output at the ceiling during it
    # |actual| and |commanded| angle at the SAME instant divergence_deg was
    # measured (the peak-disagreement moment within the window) -- so
    # divergence_deg == abs(conflict_model_deg - conflict_you_deg) exactly,
    # by construction. Lets the report show "you held ~Xdeg, model wanted
    # ~Ydeg" instead of a single hard-to-interpret difference number.
    conflict_you_deg: float | None = None
    conflict_model_deg: float | None = None


def _band(peak_abs: float) -> str:
    if peak_abs >= 150:
        return "150+"
    if peak_abs >= TURN_PEAK_MIN_DEG:
        return "90-150"
    return "20-90"


def detect_turn_episodes(
    drive_name: str, seg: Segmentation, da: DriveArrays
) -> list[TurnEpisode]:
    if da.steering_angle is None:
        return []
    episodes: list[TurnEpisode] = []
    for span in seg.lat_spans:
        sl = slice(span.i0, span.i1)
        t = da.t[sl]
        act = da.steering_angle[sl]
        engaged = span.kind == "engaged"
        cmd = da.cmd_angle[sl] if (engaged and da.cmd_angle is not None) else None
        combined = np.abs(act) if cmd is None else np.maximum(np.abs(act), np.abs(cmd))
        pressed = (
            da.steering_pressed[sl] if da.steering_pressed is not None else np.zeros(len(t), bool)
        )
        lat_flag = da.lat_model[sl]
        override = da.lat_override[sl]
        torque = da.torque_output[sl] if da.torque_output is not None else None

        for a, b in _contiguous_runs(combined >= TURN_ONSET_DEG):
            if b - a < 3:
                continue
            ip_act = a + int(np.argmax(np.abs(act[a:b])))
            peak_act = float(act[ip_act])
            if abs(peak_act) < TURN_ONSET_DEG:
                continue  # command-only excursion; the wheel never turned
            side = "left" if peak_act > 0 else "right"
            v_onset = float(da.v[span.i0 + a])
            sharp = abs(peak_act) >= TURN_PEAK_MIN_DEG and v_onset < SHARP_MAX_ONSET_V

            ep = TurnEpisode(
                engaged=engaged,
                drive=drive_name,
                side=side,
                sharp=sharp,
                band=_band(abs(peak_act)),
                i0=span.i0 + a,
                i1=span.i0 + b,
                t_onset=float(t[a]),
                v_onset=v_onset,
                peak_act=peak_act,
                peak_cmd=None,
                t_peak_act=float(t[ip_act]),
                contaminated=bool(engaged and (pressed[a:ip_act + 1].any() or override[a:ip_act + 1].any())),
            )

            # search region for unwind/overshoot: to end of span, capped 15 s past run
            end = min(len(t), _idx_after(t, t[min(b, len(t) - 1)] + 15.0))

            # unwind point (50% of peak) for the actual angle
            sgn = np.sign(peak_act)
            below_half = np.abs(act) <= UNWIND_FRACTION * abs(peak_act)
            iu_act = _first_idx(below_half[:end], ip_act + 1)

            # unwind rate: peak to |act| <= 20
            i20 = _first_idx((np.abs(act) <= TURN_ONSET_DEG)[:end], ip_act + 1)
            if i20 is not None and t[i20] > t[ip_act]:
                ep.unwind_rate = abs(peak_act) / float(t[i20] - t[ip_act])

            # driver rescue during unwind phase (peak -> |act| < 20)
            if engaged:
                ru_end = i20 if i20 is not None else end
                ph = slice(ip_act, ru_end)
                ep.rescued = bool(pressed[ph].any() or (~lat_flag[ph]).any())

            # commanded-signal peak, onset lead, and unwind lead
            t_onset_cmd = None  # used below for initiator classification
            if cmd is not None:
                ip_cmd = a + int(np.argmax(np.abs(cmd[a:b])))
                ep.peak_cmd = float(cmd[ip_cmd])
                # blinker-free "missed turn-in": the wheel turned sharply but
                # the model's own commanded path never called for a turn in
                # the SAME direction. Magnitude alone isn't enough here -- a
                # large commanded excursion the OTHER way (e.g. the model
                # wanted to go right while the driver forced a left turn) is
                # not "commanding this turn", it's disagreeing with it, so
                # use the signed peak in the actual turn's own direction.
                same_dir_peak_cmd = float(np.max(sgn * cmd[a:b]))
                ep.never_commanded = bool(engaged and same_dir_peak_cmd < TURN_ONSET_DEG)
                if same_dir_peak_cmd >= TURN_ONSET_DEG:
                    # onset-phase lead: mirrors the unwind-phase lead below,
                    # just at the first same-direction |cmd| >= 20 deg
                    # crossing (within this episode's own window) instead of
                    # the fall to 50% of peak; t_onset is already the first
                    # |act|-or-|cmd| crossing that opened this episode, so
                    # this is simply cmd's own crossing time minus that.
                    ion_cmd = _first_idx((sgn * cmd >= TURN_ONSET_DEG)[:b], a)
                    if ion_cmd is not None:
                        ep.cmd_onset_lead = float(t[ion_cmd] - t[a])
                        t_onset_cmd = float(t[ion_cmd])
                if abs(ep.peak_cmd) >= TURN_ONSET_DEG:
                    if iu_act is not None:
                        below_half_cmd = np.abs(cmd) <= UNWIND_FRACTION * abs(ep.peak_cmd)
                        iu_cmd = _first_idx(below_half_cmd[:end], ip_cmd + 1)
                        if iu_cmd is not None:
                            ep.cmd_unwind_lead = float(t[iu_cmd] - t[iu_act])

            # who steered this turn in: did the driver's hand get there
            # first (override_onset_t at/before t_onset), the model's own
            # commanded path lead the physical result (cmd led), or did the
            # physical angle simply outrun a model that hadn't caught up yet
            # (a control-loop lag, not clearly either party's doing)?
            if engaged:
                op_idx = _first_idx(pressed[a:b])
                if op_idx is not None:
                    ep.override_onset_t = float(t[a + op_idx])

                if ep.override_onset_t is not None and ep.override_onset_t <= ep.t_onset + INITIATOR_TOL_S:
                    ep.initiator = "driver"
                elif cmd is None:
                    ep.initiator = "unknown"  # old log / no commanded-angle source at all
                elif t_onset_cmd is not None and t_onset_cmd <= ep.t_onset + INITIATOR_TOL_S:
                    ep.initiator = "model"
                else:
                    ep.initiator = "lag"  # cmd caught up late, or never did -- not clearly the driver's doing

                # torque-ceiling defense: was the model already sustained at
                # its output ceiling BEFORE the override started? The window
                # searched ends strictly at override_onset_t (never overlaps
                # or extends past it) -- torque pegging "at the same time as"
                # the override is not exonerating evidence, it could just be
                # the controller fighting the driver's own input in that
                # instant (confirmed on real data: steeringPressed can go
                # True essentially at the same sample the ceiling is hit).
                if ep.override_onset_t is not None and torque is not None:
                    ov_idx = a + op_idx
                    pre_t = t[a:ov_idx]
                    if len(pre_t) >= 2 and (pre_t[-1] - pre_t[0]) >= MIN_PRE_OVERRIDE_S:
                        pre_torque = torque[a:ov_idx]
                        ceiling_mask = np.abs(pre_torque) >= CEILING_FRAC
                        found_run = None
                        for ca, cb in _contiguous_runs(ceiling_mask):
                            if pre_t[cb - 1] - pre_t[ca] >= CEILING_MIN_S:
                                found_run = (ca, cb)
                                break
                        ep.torque_ceiling_pre_override = found_run is not None
                        if found_run is not None:
                            ca, cb = found_run
                            # direction agreement: maxed-out torque pushing
                            # the SAME way as the turn ended up going is a
                            # real capability ceiling (exonerating); maxed
                            # out the OTHER way means the model's plan
                            # disagreed with the turn -- capability is beside
                            # the point, and that must not be conflated in
                            # with the exonerating bucket.
                            ep.torque_ceiling_direction_agrees = bool(
                                np.sign(np.median(pre_torque[ca:cb])) == sgn
                            )
                    # else: <0.3s of pre-override data exists at all -- can't
                    # judge, torque_ceiling_pre_override stays None

                # Quantified effort (unlike the binary ceiling check above,
                # this applies whether or not the episode was ever
                # overridden): peak |torque_output| from episode onset to
                # override_onset_t, or to the episode's own end if the
                # driver never touched the wheel at all.
                if torque is not None:
                    pre_end = a + op_idx if op_idx is not None else b
                    if pre_end > a:
                        ep.peak_effort_frac = float(np.max(np.abs(torque[a:pre_end])))

            # SCORED: cmd-vs-actual divergence during genuine driver
            # resistance. "Resistance" = steeringPressed AND the driver's
            # torque opposes the model's own commanded torque -- the sign
            # check validated by hand against a real episode where the
            # model wanted left, the driver's hand physically overpowered
            # it to the right, and torqueState.output/steeringTorque had
            # opposite signs throughout. Falls back to opposing ACT-vs-CMD
            # angle signs when either torque channel is unavailable (angle-
            # control cars, old logs) so the metric still degrades
            # gracefully instead of going dark. Every engaged episode is
            # eligible -- sharp or curve-band, unlike the old sharp-only
            # missed_turn_in -- because the owner wants every real tug-of-
            # war counted, not just the ones that happen to cross 90 deg.
            if engaged and cmd is not None:
                driver_t = da.driver_torque[sl] if da.driver_torque is not None else None
                if driver_t is not None and torque is not None:
                    opposing = (np.sign(driver_t) != np.sign(torque)) & (driver_t != 0) & (torque != 0)
                else:
                    opposing = (np.sign(act) != np.sign(cmd)) & (act != 0) & (cmd != 0)
                conflict_mask = pressed & opposing

                best = None  # (divergence, duration, ra, rb, peak_idx) -- span-local indices
                for ca, cb in _contiguous_runs(conflict_mask[a:b]):
                    ra, rb = a + ca, a + cb
                    dur = float(t[rb - 1] - t[ra]) if rb > ra else 0.0
                    if dur < CONFLICT_MIN_S:
                        continue
                    local_diff = np.abs(act[ra:rb] - cmd[ra:rb])
                    peak_idx = ra + int(np.argmax(local_diff))
                    div = float(local_diff[peak_idx - ra])
                    if best is None or div > best[0]:
                        best = (div, dur, ra, rb, peak_idx)

                if best is not None:
                    div, dur, ra, rb, peak_idx = best
                    ep.divergence_deg = div
                    ep.conflict_duration_s = dur
                    ep.conflict_you_deg = float(abs(act[peak_idx]))
                    ep.conflict_model_deg = float(abs(cmd[peak_idx]))
                    if torque is not None:
                        win_t = t[ra:rb]
                        ceil_mask = np.abs(torque[ra:rb]) >= CEILING_FRAC
                        ep.conflict_ceiling = any(
                            win_t[zb - 1] - win_t[za] >= CEILING_MIN_S
                            for za, zb in _contiguous_runs(ceil_mask)
                        )

            # S-curve overshoot after the actual angle unwinds through zero
            izero = _first_idx((sgn * act <= 0)[:end], ip_act + 1)
            if izero is not None:
                iw_end = min(end, _idx_after(t, t[izero] + OVERSHOOT_WINDOW_S))
                w = act[izero:iw_end]
                if len(w):
                    opp = -sgn * w  # positive = opposite-sign excursion
                    ep.overshoot_deg = float(max(0.0, np.max(opp)))
                    ep.overshoot_pct = 100.0 * ep.overshoot_deg / abs(peak_act)
                    # recovery wobbles: zero re-crossings after the overshoot
                    # excursion whose lobe exceeds WOBBLE_MIN_DEG (0.5 deg deadband)
                    lobes: list[float] = []
                    cur_sign, cur_max = 0, 0.0
                    for val in w:
                        sg = 1 if val > 0.5 else (-1 if val < -0.5 else 0)
                        if sg == 0:
                            continue
                        if sg != cur_sign and cur_sign != 0:
                            lobes.append(cur_max)
                            cur_max = 0.0
                        cur_sign = sg
                        cur_max = max(cur_max, abs(val))
                    if cur_sign != 0:
                        lobes.append(cur_max)
                    ep.wobbles = sum(1 for m in lobes[1:] if m > WOBBLE_MIN_DEG)

            # suppress recovery excursions: an opposite-side "turn" starting
            # right after a bigger episode's unwind is its S-curve overshoot,
            # not a new maneuver (it is already measured as overshoot_pct)
            prev = episodes[-1] if episodes and episodes[-1].drive == drive_name else None
            if (
                prev is not None
                and prev.i1 >= span.i0  # same span region
                and ep.t_onset - da.t[min(prev.i1, len(da.t) - 1)] < OVERSHOOT_WINDOW_S + 2.0
                and ep.side != prev.side
                and abs(ep.peak_act) < 0.5 * abs(prev.peak_act)
            ):
                continue
            episodes.append(ep)
    return episodes


# ---------------------------------------------------------------- ping-pong


@dataclass
class PingPongBin:
    lo_mph: float
    hi_mph: float
    engaged_s: float = 0.0
    manual_s: float = 0.0
    engaged_rms: float | None = None
    manual_rms: float | None = None
    engaged_rev: float | None = None  # reversals/min
    manual_rev: float | None = None
    score: float | None = None
    # True when scored on ABSOLUTE anchors (no manual baseline in this bin)
    # rather than the ratio vs your own driving -- the report flags this so a
    # reader knows the number isn't relative to their manual steering.
    abs_scored: bool = False
    # driver-profile pooling (see profile.py): set only when pooled history
    # for this bin's speed label exists; None/0 means "not pooled", i.e.
    # this bin is scored exactly as if profiling didn't exist
    pooled_n: int = 0  # count of OTHER routes' pooled point-estimates
    pooled_manual_rms: float | None = None  # combined (this-route + pooled) median
    pooled_manual_rev: float | None = None


@dataclass
class PingPongResult:
    bins: list[PingPongBin]
    sub_bins: list[PingPongBin]  # 1 mph resolution inside 0-10 mph
    score: float | None  # time-weighted category score
    worst_bin: PingPongBin | None
    worst_windows: list[Event] = field(default_factory=list)


def pp_category_score(bins) -> tuple[float | None, "PingPongBin | None"]:
    """Ping-pong category score from the scored bins: the engaged-time-
    weighted mean blended toward the WORST bin (PP_WORST_BLEND), so a genuine
    ping-pong hotspot in one speed range isn't diluted away by calm driving
    elsewhere. None until >= MIN_SCORED_FOR_CATEGORY bins are scored. Shared by
    analyze_pingpong and the driver-profile re-aggregation so both agree."""
    scored = [b for b in bins if b.score is not None]
    if len(scored) < MIN_SCORED_FOR_CATEGORY:
        return None, None
    w = np.array([b.engaged_s for b in scored], float)
    s = np.array([b.score for b in scored], float)
    twm = float(np.sum(w * s) / np.sum(w))
    worst = min(scored, key=lambda b: b.score)
    score = PP_WORST_BLEND * worst.score + (1.0 - PP_WORST_BLEND) * twm
    return score, worst


def _pp_accumulate(t, resid, v, base_mask, lo, hi, acc):
    """Accumulate sum-of-squares, duration and reversals for one speed bin."""
    m = base_mask & (v >= lo * MPH) & (v < hi * MPH)
    if not m.any():
        return
    x = resid[m]
    acc["ss"] += float(np.sum(np.square(x)))
    acc["n"] += int(len(x))
    for a, b in _contiguous_runs(m):
        if b - a < 5:
            continue
        acc["dur"] += float(t[b - 1] - t[a])
        acc["rev"] += swing_reversal_count(resid[a:b])


def analyze_pingpong(
    per_drive: list[tuple[str, Segmentation, DriveArrays]], score_fn
) -> PingPongResult | None:
    """score_fn(model_value, driver_value) -> 0..100 (lower-better ratio score)."""
    have_angle = any(da.steering_angle is not None for _n, _s, da in per_drive)
    if not have_angle:
        return None

    def new_acc():
        return {"ss": 0.0, "n": 0, "dur": 0.0, "rev": 0}

    bins_acc = {
        (side, i): new_acc() for side in ("engaged", "manual") for i in range(len(PP_BINS_MPH))
    }
    sub_edges = [(m, m + 1) for m in range(10)]
    sub_acc = {
        (side, i): new_acc() for side in ("engaged", "manual") for i in range(len(sub_edges))
    }
    windows: list[tuple[float, Event]] = []

    for name, seg, da in per_drive:
        if da.steering_angle is None:
            continue
        for span in seg.lat_spans:
            sl = slice(span.i0, span.i1)
            t = da.t[sl]
            v = da.v[sl]
            angle = da.steering_angle[sl]
            resid = pingpong_band(t, angle)
            base = np.ones(len(t), bool)
            if da.standstill is not None:
                base &= ~da.standstill[sl]
            else:
                base &= v > 0.1
            if span.kind == "engaged":
                if da.steering_pressed is not None:
                    base &= ~da.steering_pressed[sl]
            side = "engaged" if span.kind == "engaged" else "manual"

            for i, (lo, hi) in enumerate(PP_BINS_MPH):
                _pp_accumulate(t, resid, v, base, lo, hi, bins_acc[(side, i)])
            for i, (lo, hi) in enumerate(sub_edges):
                _pp_accumulate(t, resid, v, base, lo, hi, sub_acc[(side, i)])

            # candidate worst 10 s windows (engaged only)
            if side == "engaged":
                for a, b in _contiguous_runs(base):
                    step = max(1, int(round(PP_WORST_WINDOW_S / max(float(np.median(np.diff(t))) if len(t) > 1 else 0.01, 1e-3))))
                    for w0 in range(a, b - step, step):
                        w1 = w0 + step
                        wr = rms(resid[w0:w1])
                        if np.isfinite(wr):
                            windows.append(
                                (wr, Event(
                                    kind="pingpong", engaged=True, drive=name,
                                    t0=float(t[w0]), t1=float(t[w1 - 1]),
                                    i0=span.i0 + w0, i1=span.i0 + w1,
                                    has_override=False,
                                    values={"osc_rms": float(wr)},
                                ))
                            )

    def finish(acc):
        rms_v = float(np.sqrt(acc["ss"] / acc["n"])) if acc["n"] else None
        rev = 60.0 * acc["rev"] / acc["dur"] if acc["dur"] > 0 else None
        return rms_v, rev, acc["dur"]

    def build_bins(edges, accs, min_s):
        out = []
        for i, (lo, hi) in enumerate(edges):
            b = PingPongBin(lo, hi)
            b.engaged_rms, b.engaged_rev, b.engaged_s = finish(accs[("engaged", i)])
            b.manual_rms, b.manual_rev, b.manual_s = finish(accs[("manual", i)])
            if b.engaged_s < min_s or b.engaged_rms is None:
                out.append(b)
                continue
            if b.manual_s >= min_s and b.manual_rms is not None:
                # ratio vs your own steering in this bin -- rms and reversal
                # rate weighted equally
                s_rms = score_fn(b.engaged_rms, b.manual_rms)
                s_rev = (score_fn(b.engaged_rev, b.manual_rev)
                         if b.engaged_rev is not None and b.manual_rev is not None else None)
                parts = [(s, 1.0) for s in (s_rms, s_rev) if s is not None]
            elif (lo, hi) in PP_ABS_REV_ANCHORS:
                # no manual baseline -> absolute anchors. Reversal RATE is the
                # primary "how fast the wheel saws" ping-pong signal (weight 2);
                # amplitude is secondary (weight 1 -- it double-counts the big
                # but legitimate steering of low-speed maneuvering).
                s_rev = (score_absolute(b.engaged_rev, PP_ABS_REV_ANCHORS[(lo, hi)])
                         if b.engaged_rev is not None else None)
                s_rms = (score_absolute(b.engaged_rms, PP_ABS_RMS_ANCHORS[(lo, hi)])
                         if b.engaged_rms is not None else None)
                parts = [(s, w) for s, w in ((s_rev, 2.0), (s_rms, 1.0)) if s is not None]
                b.abs_scored = True
            else:
                parts = []
            if parts:
                b.score = float(sum(s * w for s, w in parts) / sum(w for _, w in parts))
            out.append(b)
        return out

    bins = build_bins(PP_BINS_MPH, bins_acc, PP_MIN_BIN_S)
    sub_bins = [
        b for b in build_bins(sub_edges, sub_acc, PP_SUB_BIN_MIN_S)
        if b.engaged_s >= PP_SUB_BIN_MIN_S
    ]

    score, worst = pp_category_score(bins)
    windows.sort(key=lambda p: -p[0])
    return PingPongResult(bins, sub_bins, score, worst, [e for _r, e in windows[:3]])


# -------------------------------------------------------------- turn intent


@dataclass
class IntentWindow:
    engaged: bool
    drive: str
    side: str
    t_on: float
    t_end: float
    i0: int
    i1: int
    outcome: str  # "turn" | "lane_change" | "ambiguous"
    heading_deg: float | None
    delay: float | None = None  # blinker-on -> |act| crosses 20 (executed turns)
    missed: bool = False  # engaged turn the driver had to take over
    cmd_onset_lead: float | None = None


def detect_intent_windows(
    drive_name: str, seg: Segmentation, da: DriveArrays, vm=None
) -> list[IntentWindow]:
    if da.steering_angle is None or (da.left_blinker is None and da.right_blinker is None):
        return []
    t = da.t
    v = da.v
    act = da.steering_angle
    lb = da.left_blinker if da.left_blinker is not None else np.zeros(len(t), bool)
    rb = da.right_blinker if da.right_blinker is not None else np.zeros(len(t), bool)
    pressed = da.steering_pressed if da.steering_pressed is not None else np.zeros(len(t), bool)

    # heading rate, LEFT-positive to match the ISO steering sign.
    # livePose/liveLocationKalman angularVelocityDevice.z is right-positive
    # (device frame, z down) -- verified empirically against blinker sides --
    # so measured yaw is negated; the vehicle-model fallback derives from the
    # steering angle and is already left-positive.
    if da.yaw_rate is not None:
        hr = -da.yaw_rate
    elif vm is not None:
        with np.errstate(all="ignore"):
            hr = vm.calc_curvature(np.radians(act) / 1.0, np.maximum(v, 0.1)) * v
        hr = np.where(np.isfinite(hr), hr, 0.0)
    else:
        hr = None

    out: list[IntentWindow] = []
    blink = lb | rb
    for a, b in _contiguous_runs(blink):
        low = np.flatnonzero(v[a:b] < INTENT_MAX_V)
        if len(low) == 0:
            continue
        i_on = a + int(low[0])
        side = "left" if lb[i_on] else "right"
        t_on = float(t[i_on])
        t_end = max(t_on + INTENT_WINDOW_S, float(t[b - 1]) + INTENT_OFF_PAD_S)
        i_end = min(len(t), _idx_after(t, t_end))
        if i_end - i_on < 10:
            continue
        if np.any(np.diff(t[i_on:i_end]) > 1.0):
            continue  # crosses an inter-segment gap
        engaged = bool(da.lat_model[i_on])

        if hr is None:
            outcome, heading = "ambiguous", None
        else:
            _trapz = getattr(np, "trapezoid", None) or np.trapz
            heading = float(np.degrees(_trapz(hr[i_on:i_end], t[i_on:i_end])))
            h = heading if side == "left" else -heading
            if h >= INTENT_TURN_DEG:
                outcome = "turn"
            elif abs(heading) < INTENT_LANE_DEG:
                outcome = "lane_change"
            else:
                outcome = "ambiguous"

        w = IntentWindow(
            engaged=engaged, drive=drive_name, side=side,
            t_on=t_on, t_end=float(t[i_end - 1]), i0=i_on, i1=i_end,
            outcome=outcome, heading_deg=heading,
        )

        if outcome == "turn":
            sgn = 1.0 if side == "left" else -1.0
            crossed = sgn * act[i_on:i_end] >= TURN_ONSET_DEG
            ic = _first_idx(crossed)
            press_slice = pressed[i_on:i_end]
            # a lat disengagement only counts against the model if it happens
            # before the turn-in crossing (the turn was then taken manually)
            lat_slice = da.lat_model[i_on:i_end]
            diseng = engaged and (
                (~lat_slice[:ic]).any() if ic is not None else (~lat_slice).any()
            )
            if not engaged:
                if ic is not None:
                    w.delay = float(t[i_on + ic] - t_on)
            else:
                pressed_before = (
                    press_slice[:ic].any() if ic is not None else press_slice.any()
                )
                if ic is not None and not pressed_before and not diseng:
                    w.delay = float(t[i_on + ic] - t_on)
                    if da.cmd_angle is not None:
                        cmdw = sgn * da.cmd_angle[i_on:i_end] >= TURN_ONSET_DEG
                        icc = _first_idx(cmdw)
                        if icc is not None:
                            w.cmd_onset_lead = float(t[i_on + icc] - t[i_on + ic])
                else:
                    w.missed = True
        out.append(w)
    return out
