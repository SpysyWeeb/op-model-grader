"""Turn rlog events into numpy time series (a Drive per route).

Every field access is tolerant: a message type or field missing from a
given openpilot era degrades to "channel absent" (recorded in
Drive.missing) instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .logreader import read_events


@dataclass
class Channel:
    """One signal at its native rate."""

    t: np.ndarray  # float64 seconds (logMonoTime-based)
    v: np.ndarray

    def __len__(self) -> int:
        return len(self.t)


def hold_align(src: Channel, dst_t: np.ndarray, default=0.0) -> np.ndarray:
    """Previous-value hold of src onto dst_t (for booleans / discrete)."""
    if len(src) == 0:
        return np.full(len(dst_t), default, dtype=src.v.dtype if len(src.v) else float)
    idx = np.searchsorted(src.t, dst_t, side="right") - 1
    out = src.v[np.clip(idx, 0, len(src.v) - 1)]
    if (idx < 0).any():
        out = out.copy()
        out[idx < 0] = default
    return out


def interp_align(src: Channel, dst_t: np.ndarray, default=np.nan) -> np.ndarray:
    """Linear interpolation of src onto dst_t (for floats)."""
    if len(src) == 0:
        return np.full(len(dst_t), default)
    return np.interp(dst_t, src.t, src.v.astype(np.float64))


@dataclass
class Meta:
    car_fingerprint: str = "unknown"
    openpilot_long: bool | None = None
    version: str = ""
    git_branch: str = ""
    personality: str | None = None
    experimental_mode: bool = False
    wall_time_start: float | None = None  # unix seconds, if initData had it
    routes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    vm_params: dict[str, float] = field(default_factory=dict)  # for VehicleModel


@dataclass
class Drive:
    """All extracted channels for one route (segments concatenated)."""

    name: str
    meta: Meta = field(default_factory=Meta)
    channels: dict[str, Channel] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)  # channel/field names absent
    n_segments: int = 0

    def ch(self, name: str) -> Channel | None:
        c = self.channels.get(name)
        return c if c is not None and len(c) > 0 else None


# (channel_key, message_which, field_path, dtype, converter)
# field_path entries are tried in order; the first that resolves wins.
_FLOAT = np.float64
_BOOL = np.bool_

_SPECS = {
    # carState @ ~100 Hz
    "vEgo": ("carState", ["vEgo"], _FLOAT),
    "aEgo": ("carState", ["aEgo"], _FLOAT),
    "gasPressed": ("carState", ["gasPressed"], _BOOL),
    "brakePressed": ("carState", ["brakePressed"], _BOOL),
    "standstill": ("carState", ["standstill"], _BOOL),
    "steeringAngleDeg": ("carState", ["steeringAngleDeg"], _FLOAT),
    "steeringRateDeg": ("carState", ["steeringRateDeg"], _FLOAT),
    "steeringPressed": ("carState", ["steeringPressed"], _BOOL),
    "leftBlinker": ("carState", ["leftBlinker"], _BOOL),
    "rightBlinker": ("carState", ["rightBlinker"], _BOOL),
    # enable state: selfdriveState (modern) with controlsState fallback below
    "enabled": ("selfdriveState", ["enabled"], _BOOL),
    "active": ("selfdriveState", ["active"], _BOOL),
    "experimentalMode": ("selfdriveState", ["experimentalMode"], _BOOL),
    # carControl @ ~100 Hz
    "ccEnabled": ("carControl", ["enabled"], _BOOL),
    "longActive": ("carControl", ["longActive"], _BOOL),
    "latActive": ("carControl", ["latActive"], _BOOL),
    "ccAccel": ("carControl", ["actuators.accel"], _FLOAT),
    "ccCurvature": ("carControl", ["actuators.curvature"], _FLOAT),
    "ccSteeringAngleDeg": ("carControl", ["actuators.steeringAngleDeg"], _FLOAT),
    # radarState @ ~20 Hz
    "leadStatus": ("radarState", ["leadOne.status"], _BOOL),
    "leadDRel": ("radarState", ["leadOne.dRel"], _FLOAT),
    "leadVRel": ("radarState", ["leadOne.vRel"], _FLOAT),
    "leadVLead": ("radarState", ["leadOne.vLead"], _FLOAT),
    "leadALeadK": ("radarState", ["leadOne.aLeadK"], _FLOAT),
    # longitudinalPlan @ ~20 Hz (nice-to-have)
    "aTarget": ("longitudinalPlan", ["aTarget"], _FLOAT),
    # yaw rate: livePose (modern) / liveLocationKalman (old)
    "yawRate": ("livePose", ["angularVelocityDevice.z"], _FLOAT),
    "yawRateLLK": (
        "liveLocationKalmanDEPRECATED",
        ["angularVelocityDevice.value[2]"],
        _FLOAT,
    ),
}

# old-log fallbacks for the enable state (0.9.x controlsState; the vendored
# 0.11-era schema keeps those fields inside a `deprecated` group)
_CS_FALLBACKS = {
    "enabled": ["enabled", "deprecated.enabled"],
    "active": ["active", "deprecated.active"],
    "experimentalMode": ["experimentalMode", "deprecated.experimentalMode"],
}


def _resolve(obj, path: str):
    """Follow a dotted path with optional [i] list indexing."""
    for part in path.split("."):
        if part.endswith("]"):
            name, _, idx = part.partition("[")
            obj = getattr(obj, name)[int(idx[:-1])]
        else:
            obj = getattr(obj, part)
    return obj


class _FieldGrabber:
    """Resolves a field path once, then reuses it; dead paths are skipped."""

    def __init__(self, paths: list[str]):
        self.paths = list(paths)
        self.resolved: str | None = None
        self.dead = False

    def get(self, msg):
        if self.dead:
            return None
        if self.resolved is not None:
            try:
                return _resolve(msg, self.resolved)
            except Exception:
                return None
        for p in self.paths:
            try:
                val = _resolve(msg, p)
                self.resolved = p
                return val
            except Exception:
                continue
        self.dead = True
        return None


_PERSONALITIES = {0: "aggressive", 1: "standard", 2: "relaxed"}


def extract_drive(name: str, segment_paths: list[str | Path], progress=None) -> Drive:
    """Decode segments (in order) and build a Drive of numpy channels."""
    drive = Drive(name=name)
    drive.meta.routes.append(name)

    # per-channel accumulators: key -> (t_list, v_list)
    acc: dict[str, tuple[list, list]] = {k: ([], []) for k in _SPECS}
    grabbers: dict[str, _FieldGrabber] = {}
    for key, (which, paths, _dt) in _SPECS.items():
        if which == "selfdriveState" and key in _CS_FALLBACKS:
            grabbers[key] = _FieldGrabber(paths)
            grabbers["cs_" + key] = _FieldGrabber(_CS_FALLBACKS[key])
            acc["cs_" + key] = ([], [])
        else:
            grabbers[key] = _FieldGrabber(paths)

    # which message types feed which channel keys
    by_which: dict[str, list[str]] = {}
    for key, (which, _p, _dt) in _SPECS.items():
        by_which.setdefault(which, []).append(key)
    by_which.setdefault("controlsState", []).extend(
        "cs_" + k for k in _CS_FALLBACKS
    )

    personality_raw: int | None = None
    saw_selfdrive = False

    for seg_i, path in enumerate(segment_paths):
        n_events = 0
        for t, which, e in read_events(path):
            n_events += 1
            keys = by_which.get(which)
            if keys is not None:
                msg = getattr(e, which)
                for key in keys:
                    val = grabbers[key].get(msg)
                    if val is not None:
                        ts, vs = acc[key]
                        ts.append(t)
                        vs.append(val)
                if which == "selfdriveState":
                    saw_selfdrive = True
                    try:
                        personality_raw = e.selfdriveState.personality.raw
                    except Exception:
                        pass
            elif which == "carParams":
                cp = e.carParams
                for attr, dest in (
                    ("carFingerprint", "car_fingerprint"),
                    ("carName", "car_fingerprint"),
                ):
                    try:
                        v = getattr(cp, attr)
                        if v:
                            drive.meta.car_fingerprint = v
                            break
                    except Exception:
                        continue
                try:
                    drive.meta.openpilot_long = bool(cp.openpilotLongitudinalControl)
                except Exception:
                    pass
                for f_name in (
                    "mass",
                    "wheelbase",
                    "centerToFront",
                    "steerRatio",
                    "steerRatioRear",
                    "tireStiffnessFront",
                    "tireStiffnessRear",
                ):
                    try:
                        drive.meta.vm_params[f_name] = float(getattr(cp, f_name))
                    except Exception:
                        pass
            elif which == "initData":
                init = e.initData
                for attr, dest in (
                    ("version", "version"),
                    ("gitBranch", "git_branch"),
                ):
                    try:
                        v = getattr(init, attr)
                        if v and not getattr(drive.meta, dest):
                            setattr(drive.meta, dest, v)
                    except Exception:
                        pass
                try:
                    wt = init.wallTimeNanos
                    if wt and drive.meta.wall_time_start is None:
                        drive.meta.wall_time_start = wt * 1e-9
                except Exception:
                    pass
        drive.n_segments += 1
        if progress:
            progress(seg_i, len(segment_paths), n_events)

    # convert accumulators -> Channels
    def to_channel(key: str, dtype) -> Channel:
        ts, vs = acc[key]
        return Channel(np.asarray(ts, dtype=np.float64), np.asarray(vs, dtype=dtype))

    for key, (which, _p, dtype) in _SPECS.items():
        drive.channels[key] = to_channel(key, dtype)

    # enable-state fallback: prefer selfdriveState, else controlsState
    if not saw_selfdrive:
        for key in _CS_FALLBACKS:
            ch = to_channel("cs_" + key, _BOOL)
            if len(ch) > 0:
                drive.channels[key] = ch
            else:
                drive.missing.append(key)
        drive.meta.notes.append(
            "enable state read from controlsState (pre-selfdriveState log)"
        )

    # yaw rate fallback: liveLocationKalman for old logs
    if len(drive.channels["yawRate"]) == 0:
        llk = drive.channels.pop("yawRateLLK")
        if len(llk) > 0:
            drive.channels["yawRate"] = llk
            drive.meta.notes.append("yaw rate from liveLocationKalman (old log)")
    else:
        drive.channels.pop("yawRateLLK", None)
    drive.channels.pop("yawRateLLK", None)

    # experimental mode: any-true over the drive
    exp = drive.ch("experimentalMode")
    drive.meta.experimental_mode = bool(exp.v.any()) if exp is not None else False

    # personality is only trustworthy from selfdriveState (in old controlsState
    # logs the field may simply not exist on the wire and reads as 0)
    if saw_selfdrive and personality_raw is not None:
        drive.meta.personality = _PERSONALITIES.get(personality_raw, str(personality_raw))

    for key in list(_SPECS):
        if key == "yawRateLLK":
            continue  # internal fallback source, not a user-facing channel
        c = drive.channels.get(key)
        if (c is None or len(c) == 0) and key not in drive.missing:
            drive.missing.append(key)

    return drive
