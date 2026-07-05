"""Maneuver detection inside engaged/manual spans.

All thresholds follow the tool spec; speeds are m/s, accels m/s^2.
Every event is tagged with the span kind it was found in (engaged/manual),
so the same detector measures the model and the human symmetrically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .extract import Drive, hold_align, interp_align
from .metrics import smooth
from .segments import Segmentation, Span, _contiguous_runs, split_runs_at_gaps

# thresholds
STOP_FROM_V = 8.0
STOP_STANDSTILL_V = 0.3
STOP_STANDSTILL_HOLD_S = 0.5
STOP_MAX_DUR_S = 45.0

LAUNCH_STANDSTILL_V = 0.1
LAUNCH_STANDSTILL_HOLD_S = 1.0
LAUNCH_MOTION_V = 0.15
LAUNCH_CROSS_V = 3.0
LAUNCH_CROSS_WITHIN_S = 10.0
LAUNCH_END_V = 5.0
LAUNCH_CAP_S = 8.0

FOLLOW_MAX_DREL = 80.0
FOLLOW_MIN_V = 8.0
FOLLOW_MIN_DUR_S = 15.0
FOLLOW_DROPOUT_S = 1.0

LEAD_DECEL_THRESH = -1.2
LEAD_DECEL_HOLD_S = 0.4
LEAD_DECEL_RESPONSE_DROP = 0.3
LEAD_DECEL_MAX_LATENCY_S = 4.0

PULLAWAY_DREL = 12.0
PULLAWAY_VLEAD = 0.5
PULLAWAY_VLEAD_HOLD_S = 0.3
PULLAWAY_GO_V = 0.15
PULLAWAY_GO_A = 0.2
PULLAWAY_MAX_LATENCY_S = 6.0

CRUISE_MIN_V = 15.0
CRUISE_NO_LEAD_DREL = 120.0
CRUISE_MIN_DUR_S = 15.0


@dataclass
class DriveArrays:
    """Channels aligned onto the carState timebase for one drive."""

    t: np.ndarray
    v: np.ndarray
    a: np.ndarray
    a_smooth: np.ndarray
    long_override: np.ndarray  # gas/brake pressed while model controls long
    lat_override: np.ndarray  # steeringPressed while model controls lat
    gas_pressed: np.ndarray | None  # bool: raw carState.gasPressed (any control state)
    brake_pressed: np.ndarray | None  # bool: raw carState.brakePressed (any control state)
    enabled: np.ndarray  # bool (single flag, reference only)
    lat_model: np.ndarray  # bool: model steering (AOL/MADS aware)
    long_model: np.ndarray  # bool: model gas/brake
    exp_mode: np.ndarray | None  # bool: experimental mode active (None if unknown)
    personality: np.ndarray | None  # int: 0 aggressive / 1 standard / 2 relaxed
    standstill: np.ndarray | None
    steering_angle: np.ndarray | None  # deg, positive = LEFT (ISO)
    cmd_angle: np.ndarray | None  # commanded steering angle via VehicleModel, deg
    steering_rate: np.ndarray | None
    steering_pressed: np.ndarray | None
    left_blinker: np.ndarray | None
    right_blinker: np.ndarray | None
    yaw_rate: np.ndarray | None  # rad/s
    lat_accel: np.ndarray | None  # vEgo * yawRate, None if no yaw source
    lead_status: np.ndarray | None  # bool
    d_rel: np.ndarray | None
    v_lead: np.ndarray | None
    a_lead_k: np.ndarray | None
    a_target: np.ndarray | None  # planned accel (longitudinalPlan.aTarget)
    plan_source: np.ndarray | None  # longitudinalPlanSource raw enum
    desired_curv: np.ndarray | None  # modelV2.action.desiredCurvature (raw sign)
    vis_accel: np.ndarray | None  # vision plan accel: acceleration.x[0] (fallback action.desiredAcceleration)
    vis_v4: np.ndarray | None  # vision planned speed ~4 s ahead
    torque_output: np.ndarray | None  # controlsState.lateralControlState.torqueState.output, -1..1
    torque_saturated: np.ndarray | None  # .saturated -- UNRELIABLE, diagnostic only (see lateral.py)
    desired_lat_accel: np.ndarray | None  # .desiredLateralAccel, m/s^2
    actual_lat_accel: np.ndarray | None  # .actualLateralAccel, m/s^2
    driver_torque: np.ndarray | None  # carState.steeringTorque, driver's raw torque (native CAN units)

    @property
    def has_lead_data(self) -> bool:
        return self.lead_status is not None


@dataclass
class Event:
    kind: str
    engaged: bool
    drive: str
    t0: float
    t1: float
    i0: int  # slice into DriveArrays
    i1: int
    has_override: bool
    values: dict = field(default_factory=dict)  # per-event metric inputs/outputs


def build_arrays(drive: Drive, seg: Segmentation) -> DriveArrays:
    t = seg.t
    v = drive.ch("vEgo").v.astype(np.float64)
    a_ch = drive.ch("aEgo")
    a = a_ch.v.astype(np.float64) if a_ch is not None and len(a_ch) == len(t) else np.zeros(len(t))

    def f(name):
        ch = drive.ch(name)
        return interp_align(ch, t) if ch is not None else None

    def b(name):
        ch = drive.ch(name)
        return hold_align(ch, t, default=False).astype(bool) if ch is not None else None

    sr_ch = drive.ch("steeringRateDeg")
    steering_rate = (
        sr_ch.v.astype(np.float64) if sr_ch is not None and len(sr_ch) == len(t) else f("steeringRateDeg")
    )

    yaw = f("yawRate")
    lat_accel = v * yaw if yaw is not None else None

    # commanded steering angle: actuators.steeringAngleDeg if populated (angle
    # cars), else VehicleModel conversion of the NEGATED actuators.curvature
    # (torque cars) -- matching openpilot's controlsd convention.
    cmd_angle = None
    cmd_direct = f("ccSteeringAngleDeg")
    if cmd_direct is not None and np.nanmax(np.abs(cmd_direct)) > 1e-3:
        cmd_angle = cmd_direct
    else:
        from .vehicle_model import vehicle_model_from_params

        vm = vehicle_model_from_params(drive.meta.vm_params)
        curv = f("ccCurvature")
        if vm is not None and curv is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                cf = vm.curvature_factor(v)
                cmd_angle = np.degrees(-curv * vm.sR / cf)
            cmd_angle = np.where(np.isfinite(cmd_angle), cmd_angle, 0.0)

    return DriveArrays(
        t=t,
        v=v,
        a=a,
        a_smooth=smooth(t, a),
        long_override=seg.long_override,
        lat_override=seg.lat_override,
        gas_pressed=b("gasPressed"),
        brake_pressed=b("brakePressed"),
        enabled=seg.enabled,
        lat_model=seg.lat_model,
        long_model=seg.long_model,
        exp_mode=b("experimentalMode"),
        personality=(
            hold_align(drive.ch("personality"), t, default=-1).astype(np.int16)
            if drive.ch("personality") is not None
            else None
        ),
        standstill=b("standstill"),
        steering_angle=f("steeringAngleDeg"),
        cmd_angle=cmd_angle,
        steering_rate=steering_rate,
        steering_pressed=b("steeringPressed"),
        left_blinker=b("leftBlinker"),
        right_blinker=b("rightBlinker"),
        yaw_rate=yaw,
        lat_accel=lat_accel,
        lead_status=b("leadStatus"),
        d_rel=f("leadDRel"),
        v_lead=f("leadVLead"),
        a_lead_k=f("leadALeadK"),
        a_target=f("aTarget"),
        plan_source=(
            hold_align(drive.ch("planSource"), t, default=-1).astype(np.int16)
            if drive.ch("planSource") is not None
            else None
        ),
        desired_curv=f("desiredCurvature"),
        vis_accel=(f("planVisA0") if drive.ch("planVisA0") is not None else f("planVisDA")),
        vis_v4=f("planVisV4"),
        torque_output=f("torqueOutput"),
        torque_saturated=b("torqueSaturated"),
        desired_lat_accel=f("desiredLatAccel"),
        actual_lat_accel=f("actualLatAccel"),
        driver_torque=f("steeringTorque"),
    )


def _fill_short_false_gaps(t: np.ndarray, mask: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Set short False runs (<= max_gap_s) between True runs to True."""
    out = mask.copy()
    for a, b in _contiguous_runs(~mask):
        if a == 0 or b == len(mask):
            continue  # only fill interior gaps
        if t[b - 1] - t[a] <= max_gap_s:
            out[a:b] = True
    return out


def _runs_min_dur(t, mask, min_dur):
    return [
        (a, b)
        for a, b in split_runs_at_gaps(t, _contiguous_runs(mask))
        if t[b - 1] - t[a] >= min_dur
    ]


MIXED_CONTROL_TOLERANCE = 0.10  # <10% of samples may disagree with the tag

PERSONALITY_NAMES = {0: "aggressive", 1: "standard", 2: "relaxed"}


def mode_tag(exp_mode: np.ndarray | None, i0: int, i1: int) -> str:
    """"experimental"/"chill" with the 90% constancy rule, else "mixed"."""
    if exp_mode is None or i1 <= i0:
        return "unknown"
    frac = float(np.mean(exp_mode[i0:i1]))
    if frac >= 1.0 - MIXED_CONTROL_TOLERANCE:
        return "experimental"
    if frac <= MIXED_CONTROL_TOLERANCE:
        return "chill"
    return "mixed"


def personality_tag(personality: np.ndarray | None, i0: int, i1: int) -> str:
    """Dominant (>=90%) personality name, "mixed" on a mid-window flip."""
    if personality is None or i1 <= i0:
        return "unknown"
    window = personality[i0:i1]
    vals, counts = np.unique(window, return_counts=True)
    top = int(vals[np.argmax(counts)])
    if counts.max() / len(window) < 1.0 - MIXED_CONTROL_TOLERANCE:
        return "mixed"
    return PERSONALITY_NAMES.get(top, "unknown")


def axis_tag(flag: np.ndarray, i0: int, i1: int) -> bool | None:
    """Engaged/manual tag for a window, or None if control is mixed."""
    frac = float(np.mean(flag[i0:i1])) if i1 > i0 else 0.0
    if frac >= 1.0 - MIXED_CONTROL_TOLERANCE:
        return True
    if frac <= MIXED_CONTROL_TOLERANCE:
        return False
    return None  # mixed control: discard rather than mis-attribute


def _mk_event(
    kind: str, span: Span, da: DriveArrays, drive: str, i0: int, i1: int, **values
) -> Event | None:
    i0 = max(i0, 0)
    i1 = min(i1, len(da.t))
    # longitudinal events are attributed by who controlled gas/brake over the
    # actual event window (AOL-aware), not by the enclosing span
    engaged = axis_tag(da.long_model, i0, i1)
    if engaged is None:
        return None
    values = dict(values)
    values["mode"] = mode_tag(da.exp_mode, i0, i1)
    values["personality"] = personality_tag(da.personality, i0, i1)
    return Event(
        kind=kind,
        engaged=engaged,
        drive=drive,
        t0=float(da.t[i0]),
        t1=float(da.t[i1 - 1]),
        i0=i0,
        i1=i1,
        has_override=bool(da.long_override[i0:i1].any()),
        values=values,
    )


def _idx_after(t: np.ndarray, t_target: float) -> int:
    return int(np.searchsorted(t, t_target, side="left"))


def detect_events(drive: Drive, seg: Segmentation, da: DriveArrays) -> list[Event]:
    events: list[Event] = []
    for span in seg.long_spans:
        sl = slice(span.i0, span.i1)
        t = da.t[sl]
        v = da.v[sl]
        if len(t) < 10:
            continue
        off = span.i0

        events += _detect_stops(span, da, drive.name, t, v, off)
        events += _detect_launches(span, da, drive.name, t, v, off)

        if da.has_lead_data:
            status = da.lead_status[sl]
            d_rel = da.d_rel[sl]
            follows = _detect_follows(span, da, drive.name, t, v, status, d_rel, off)
            events += follows
            events += _detect_lead_decels(span, da, drive.name, follows, off)
            events += _detect_pullaways(span, da, drive.name, t, v, status, d_rel, off)
            events += _detect_cruise(span, da, drive.name, t, v, status, d_rel, off)
    events.sort(key=lambda e: e.t0)
    return events


def _detect_stops(span, da, name, t, v, off):
    out = []
    # standstill onsets: v < 0.3 held >= 0.5 s
    for a, b in _runs_min_dur(t, v < STOP_STANDSTILL_V, STOP_STANDSTILL_HOLD_S):
        before = np.flatnonzero(v[:a] >= STOP_FROM_V)
        if len(before) == 0:
            continue
        k = int(before[-1])
        if t[a] - t[k] > STOP_MAX_DUR_S or t[a] <= t[k]:
            continue
        # extend slightly into the standstill so the v<0.2 crossing is captured
        e = min(b, _idx_after(t, t[a] + 0.5))
        # window must not cross a time gap
        if np.any(np.diff(t[k:e]) > 1.0):
            continue
        ev = _mk_event("stop", span, da, name, off + k, off + e,
                       t_standstill=float(t[a]))
        if ev is not None:
            out.append(ev)
    return out


def _detect_launches(span, da, name, t, v, off):
    out = []
    for a, b in _runs_min_dur(t, v < LAUNCH_STANDSTILL_V, LAUNCH_STANDSTILL_HOLD_S):
        moving = np.flatnonzero(v[b:] > LAUNCH_MOTION_V)
        if len(moving) == 0:
            continue
        fm = b + int(moving[0])  # first motion
        within = (t >= t[fm]) & (t <= t[fm] + LAUNCH_CROSS_WITHIN_S)
        if not np.any(within & (v >= LAUNCH_CROSS_V)):
            continue
        hit5 = np.flatnonzero((np.arange(len(v)) >= fm) & (v >= LAUNCH_END_V)
                              & (t <= t[fm] + LAUNCH_CAP_S))
        i_end = int(hit5[0]) if len(hit5) else _idx_after(t, t[fm] + LAUNCH_CAP_S) - 1
        i_start = _idx_after(t, t[fm] - 1.0)
        if i_end <= i_start:
            continue
        if np.any(np.diff(t[i_start : i_end + 1]) > 1.0):
            continue
        ev = _mk_event("launch", span, da, name, off + i_start, off + i_end + 1,
                       t_first_motion=float(t[fm]))
        if ev is not None:
            out.append(ev)
    return out


def _detect_follows(span, da, name, t, v, status, d_rel, off):
    st = _fill_short_false_gaps(t, status, FOLLOW_DROPOUT_S)
    mask = st & (d_rel < FOLLOW_MAX_DREL) & (v > FOLLOW_MIN_V)
    evs = [
        _mk_event("follow", span, da, name, off + a, off + b)
        for a, b in _runs_min_dur(t, mask, FOLLOW_MIN_DUR_S)
    ]
    return [e for e in evs if e is not None]


def _detect_lead_decels(span, da, name, follows, off):
    out = []
    for fw in follows:
        sl = slice(fw.i0, fw.i1)
        t = da.t[sl]
        alk = da.a_lead_k[sl]
        a_s = da.a_smooth[sl]
        for a, b in _runs_min_dur(t, alk < LEAD_DECEL_THRESH, LEAD_DECEL_HOLD_S):
            a0 = a_s[a]
            horizon = (np.arange(len(t)) > a) & (t <= t[a] + LEAD_DECEL_MAX_LATENCY_S)
            resp = np.flatnonzero(horizon & (a_s < a0 - LEAD_DECEL_RESPONSE_DROP))
            if len(resp):
                latency = float(t[resp[0]] - t[a])
                censored = False
            else:
                latency = LEAD_DECEL_MAX_LATENCY_S
                censored = True
            i0 = fw.i0 + _idx_after(t, t[a] - 2.0)
            i1 = fw.i0 + min(len(t), _idx_after(t, t[a] + 6.0) + 1)
            ev = _mk_event("lead_decel", span, da, name, i0, i1,
                           t_onset=float(t[a]), latency=latency,
                           censored=censored)
            if ev is not None:
                out.append(ev)
    return out


def _detect_pullaways(span, da, name, t, v, status, d_rel, off):
    sl = slice(off, off + len(t))
    v_lead = da.v_lead[sl]
    a = da.a[sl]
    out = []
    for ra, rb in _runs_min_dur(t, v_lead > PULLAWAY_VLEAD, PULLAWAY_VLEAD_HOLD_S):
        # preconditions just before onset: ego standstill, close lead present
        pre0 = _idx_after(t, t[ra] - 0.5)
        pre = slice(max(pre0, 0), max(ra, pre0 + 1))
        if not (
            np.all(v[pre] < LAUNCH_STANDSTILL_V)
            and status[pre].mean() > 0.5
            and np.nanmedian(d_rel[pre]) < PULLAWAY_DREL
        ):
            continue
        horizon = (np.arange(len(t)) >= ra) & (t <= t[ra] + PULLAWAY_MAX_LATENCY_S)
        go = np.flatnonzero(horizon & ((v > PULLAWAY_GO_V) | (a > PULLAWAY_GO_A)))
        if len(go):
            latency = float(t[go[0]] - t[ra])
            censored = False
        else:
            latency = PULLAWAY_MAX_LATENCY_S
            censored = True
        i0 = _idx_after(t, t[ra] - 2.0)
        i1 = _idx_after(t, t[ra] + latency + 2.0)
        ev = _mk_event("pullaway", span, da, name, off + i0, off + i1,
                       t_onset=float(t[ra]), latency=latency,
                       censored=censored)
        if ev is not None:
            out.append(ev)
    return out


def _detect_cruise(span, da, name, t, v, status, d_rel, off):
    mask = (~status | (d_rel > CRUISE_NO_LEAD_DREL)) & (v > CRUISE_MIN_V)
    evs = [
        _mk_event("cruise", span, da, name, off + a, off + b)
        for a, b in _runs_min_dur(t, mask, CRUISE_MIN_DUR_S)
    ]
    return [e for e in evs if e is not None]
