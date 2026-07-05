"""Plan-vs-You counterfactuals: the model's live plan during HUMAN driving.

plannerd/modeld keep running while disengaged, so manual driving carries the
model's intent at every sample. These analyses compare that intent to what
the driver actually did. Timing comparisons are robust; magnitudes are
indicative only — the plan is conditioned on the situation the driver
created. Results are reported unscored, never folded into the group grades.

Empirical grounding (verified on the owner's 0.11.2 logs, see tests):
- modelV2.action.desiredCurvature is live and accurate while lat is
  disengaged (corr -0.93, magnitude ratio 0.98 against steering-derived
  curvature) but SIGN-INVERTED vs the ISO left-positive convention, matching
  openpilot's actuators.curvature. We use plan_left = -desiredCurvature.
- Longitudinal counterfactuals use the VISION (end-to-end) plan -- the
  modelV2 velocity/acceleration trajectory that Experimental mode executes.
  It needs no lead and no cruise setpoint and reacts to lights/signs.
  Verified live while disengaged: 33-point arrays with velocity.t carrying
  T_IDXS (0..10 s), velocity.x[0] tracking vEgo (corr 0.997),
  acceleration.x[0] matching action.desiredAcceleration (corr 0.876);
  during leadless driver braking the planned accel p25 is -1.02 m/s2 (it
  does plan stops for lights, measured per event as onset lag). The chill
  MPC longitudinalPlan is deliberately NOT used for grading ("accelerate to
  set speed, match the lead" -- meaningless without a cruise target); its
  braking onset is shown as an unscored diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import DriveArrays, Event
from .lateral import MPH, PP_BINS_MPH, TurnEpisode, IntentWindow
from .metrics import smooth
from .segments import _contiguous_runs

# thresholds
BRAKE_ACCEL = -0.5  # m/s^2: "braking has begun" for both plan and driver
LAUNCH_ACCEL = 0.3  # m/s^2: "launch has begun" for the plan
TURN_IN_FRACTION = 0.30  # plan onset = crossing 30% of driver's peak curvature
UNWIND_FRACTION = 0.5
MIN_SPEED_PATH = 3.0  # m/s: path agreement needs the car to be moving
BRAKE_LEAD_DREL = 60.0  # lead-vs-no-lead split threshold at approach start
ONSET_CAP_S = 6.0
SPEED_OPINION_HORIZON_S = 4.0
STANDSTILL_EXCLUDE_S = 8.0  # drop the run-up to every stop from X2


@dataclass
class Counterfactual:
    available: bool = False
    why_unavailable: str = ""
    # L1: per-speed-bin RMS of (plan - actual) curvature * v^2  [m/s^2]
    path_bins: list[dict] = field(default_factory=list)
    path_overall: float | None = None
    path_seconds: float = 0.0
    # L2: manual intersection-turn intents
    turn_in: list[dict] = field(default_factory=list)  # per-window rows
    # L3: manual sharp-turn unwind lags per side
    unwind: list[dict] = field(default_factory=list)
    # X1 stop-approach onset, split by lead presence at approach start
    braking: list[dict] = field(default_factory=list)  # rows carry "lead": bool
    braking_never_lead: int = 0
    braking_never_nolead: int = 0
    mpc_brake_lags: list[float] = field(default_factory=list)  # diagnostic only
    # X2 desired-speed agreement over manual cruising samples
    accel_rms: float | None = None  # planned-vs-realized accel RMS (m/s^2)
    accel_rms_seconds: float = 0.0
    speed_opinion: dict = field(default_factory=dict)  # {"free"/"lead": {...}}
    # X3 launch onset
    launch: list[dict] = field(default_factory=list)  # rows carry "lead": bool

    # ---- aggregates used by report/CLI
    def turn_in_summary(self) -> dict:
        rows = [r for r in self.turn_in]
        lags = [r["lag"] for r in rows if r["lag"] is not None]
        never = sum(1 for r in rows if r["never_planned"])
        by_side = {}
        for side in ("left", "right"):
            sl = [r["lag"] for r in rows if r["side"] == side and r["lag"] is not None]
            if sl:
                by_side[side] = {"median": float(np.median(sl)), "n": len(sl)}
        return {
            "n": len(rows),
            "never": never,
            "median_lag": float(np.median(lags)) if lags else None,
            "n_lag": len(lags),
            "by_side": by_side,
        }

    def unwind_summary(self) -> dict:
        out = {}
        for side in ("left", "right"):
            ls = [r["lag"] for r in self.unwind if r["side"] == side]
            if ls:
                out[side] = {"mean": float(np.mean(ls)), "n": len(ls)}
        return out

    def braking_summary(self) -> dict:
        out = {}
        for name, has_lead, never in (
            ("nolead", False, self.braking_never_nolead),
            ("lead", True, self.braking_never_lead),
        ):
            lags = [r["lag"] for r in self.braking if r["lead"] == has_lead]
            out[name] = {
                "n": len(lags),
                "median": float(np.median(lags)) if lags else None,
                "never": never,
                "lags": lags,
            }
        out["mpc_median"] = (
            float(np.median(self.mpc_brake_lags)) if self.mpc_brake_lags else None
        )
        out["mpc_n"] = len(self.mpc_brake_lags)
        return out

    def launch_summary(self) -> dict:
        out = {}
        for name, has_lead in (("lead", True), ("nolead", False)):
            lags = [r["lag"] for r in self.launch if r.get("lead", True) == has_lead]
            out[name] = {"n": len(lags),
                         "median": float(np.median(lags)) if lags else None,
                         "lags": lags}
        return out


def _actual_curvature(da: DriveArrays, vm) -> np.ndarray | None:
    """Left-positive actual curvature: steering-derived via the vehicle
    model, else measured yaw/v (yaw z is right-positive, hence the minus)."""
    if vm is not None and da.steering_angle is not None:
        with np.errstate(all="ignore"):
            c = vm.calc_curvature(np.radians(da.steering_angle), np.maximum(da.v, 0.1))
        return np.where(np.isfinite(c), c, 0.0)
    if da.yaw_rate is not None:
        with np.errstate(all="ignore"):
            c = -da.yaw_rate / np.maximum(da.v, 0.5)
        return np.where(np.isfinite(c) & (da.v > MIN_SPEED_PATH), c, 0.0)
    return None


def _first_cross(t, mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return int(start + idx[0]) if len(idx) else None


def analyze_counterfactual(
    per_drive,
    turns: list[TurnEpisode],
    intents: list[IntentWindow],
    t_follow_targets: dict[str, float],
    vms: dict[str, object],
) -> tuple[Counterfactual, list[Event]]:
    """Returns (results, drill-down events). Never raises on missing channels."""
    cf = Counterfactual()
    events: list[Event] = []

    have_curv = any(da.desired_curv is not None for _d, _s, da, _e in per_drive)
    have_vis = any(da.vis_accel is not None for _d, _s, da, _e in per_drive)
    if not (have_curv or have_vis):
        cf.why_unavailable = "logs carry neither modelV2 plan trajectories nor its action"
        return cf, events
    cf.available = True

    # ---------------- L1: path agreement per speed bin (manual-lat samples)
    bin_acc = [{"ss": 0.0, "n": 0, "s": 0.0} for _ in PP_BINS_MPH]
    for drive, _seg, da, _e in per_drive:
        if da.desired_curv is None:
            continue
        act = _actual_curvature(da, vms.get(drive.name))
        if act is None:
            continue
        plan = -da.desired_curv  # left-positive (empirically verified)
        dt = float(np.median(np.diff(da.t))) if len(da.t) > 1 else 0.01
        base = ~da.lat_model & (da.v > MIN_SPEED_PATH)
        err = (plan - act) * np.square(da.v)
        for i, (lo, hi) in enumerate(PP_BINS_MPH):
            m = base & (da.v >= lo * MPH) & (da.v < hi * MPH) & np.isfinite(err)
            if m.any():
                bin_acc[i]["ss"] += float(np.sum(np.square(err[m])))
                bin_acc[i]["n"] += int(m.sum())
                bin_acc[i]["s"] += float(m.sum()) * dt
    tot_ss = sum(b["ss"] for b in bin_acc)
    tot_n = sum(b["n"] for b in bin_acc)
    cf.path_seconds = sum(b["s"] for b in bin_acc)
    if tot_n > 100:
        cf.path_overall = float(np.sqrt(tot_ss / tot_n))
        for (lo, hi), b in zip(PP_BINS_MPH, bin_acc):
            if b["s"] >= 10.0:
                cf.path_bins.append({
                    "lo_mph": lo, "hi_mph": hi,
                    "rms": float(np.sqrt(b["ss"] / b["n"])), "seconds": b["s"],
                })

    da_by_name = {d.name: da for d, _s, da, _e in per_drive}

    # ---------------- L2: counterfactual turn-in on DRIVER-EXECUTED
    # intersection turns: truly-manual windows, plus AOL windows where the
    # model missed the turn-in and the driver forced the wheel (with
    # always-on-lateral, most of the owner's turns are the latter -- the
    # plan-vs-driver comparison is equally valid there)
    for w in intents:
        if w.outcome != "turn" or (w.engaged and not w.missed):
            continue
        da = da_by_name.get(w.drive)
        if da is None or da.desired_curv is None:
            continue
        act = _actual_curvature(da, vms.get(w.drive))
        if act is None:
            continue
        sl = slice(w.i0, w.i1)
        t = da.t[sl]
        sgn = 1.0 if w.side == "left" else -1.0
        act_s = sgn * act[sl]
        plan_s = sgn * (-da.desired_curv[sl])
        peak = float(np.max(act_s))
        if peak <= 1e-4:
            continue
        thresh = TURN_IN_FRACTION * peak
        i_driver = _first_cross(t, act_s >= thresh)
        i_plan = _first_cross(t, plan_s >= thresh)
        never = i_plan is None
        lag = None
        if not never and i_driver is not None:
            lag = float(t[i_plan] - t[i_driver])
        aol = bool(w.engaged)  # driver-executed under always-on-lateral
        cf.turn_in.append({
            "side": w.side, "drive": w.drive, "t0": w.t_on,
            "lag": lag, "never_planned": never, "peak_curv": peak, "aol": aol,
        })
        events.append(Event(
            kind="cf_turnin", engaged=False, drive=w.drive,
            t0=float(t[0]), t1=float(t[-1]), i0=w.i0, i1=w.i1,
            has_override=False,
            values={"side": w.side, "lag": lag, "never_planned": never,
                    "aol": aol},
        ))

    # ---------------- L3: counterfactual unwind on MANUAL sharp turns
    for ep in turns:
        if not ep.sharp or (ep.engaged and not ep.contaminated):
            continue  # driver-executed sharp turns only
        da = da_by_name.get(ep.drive)
        if da is None or da.desired_curv is None:
            continue
        act = _actual_curvature(da, vms.get(ep.drive))
        if act is None:
            continue
        i1 = min(len(da.t), int(np.searchsorted(da.t, da.t[min(ep.i1, len(da.t) - 1)] + 15.0)))
        sl = slice(ep.i0, i1)
        t = da.t[sl]
        act_a = np.abs(act[sl])
        plan_a = np.abs(da.desired_curv[sl])
        ia = int(np.argmax(act_a))
        ip = int(np.argmax(plan_a))
        if act_a[ia] < 1e-4 or plan_a[ip] < 1e-4:
            continue
        iua = _first_cross(t, act_a <= UNWIND_FRACTION * act_a[ia], ia + 1)
        iup = _first_cross(t, plan_a <= UNWIND_FRACTION * plan_a[ip], ip + 1)
        if iua is None or iup is None:
            continue
        cf.unwind.append({"side": ep.side, "drive": ep.drive,
                          "lag": float(t[iup] - t[iua])})

    # ---------------- X1: stop-approach onset (vision plan), lead / no-lead
    for drive, _seg, da, evs in per_drive:
        a_s = smooth(da.t, da.a)
        for ev in evs:
            if ev.kind == "stop" and not ev.engaged and da.vis_accel is not None:
                # search onsets from up to 8 s BEFORE the stop window opens
                # (braking usually starts before v falls under the window's
                # v>=8 anchor; without the extension both onsets pin to the
                # window start and every lag reads 0)
                i_pre = int(np.searchsorted(da.t, da.t[ev.i0] - 8.0))
                if np.any(np.diff(da.t[i_pre:ev.i0 + 1]) > 1.0):
                    i_pre = ev.i0  # don't cross an inter-segment gap
                sl = slice(i_pre, ev.i1)
                t = da.t[sl]
                has_lead = bool(
                    da.lead_status is not None and da.d_rel is not None
                    and bool(da.lead_status[ev.i0])
                    and float(da.d_rel[ev.i0]) < BRAKE_LEAD_DREL
                )
                i_driver = _first_cross(t, a_s[sl] < BRAKE_ACCEL)
                if i_driver is None:
                    continue
                i_plan = _first_cross(t, da.vis_accel[sl] < BRAKE_ACCEL)
                if i_plan is None:
                    if has_lead:
                        cf.braking_never_lead += 1
                    else:
                        cf.braking_never_nolead += 1
                    lag = None
                else:
                    lag = float(t[i_plan] - t[i_driver])
                    cf.braking.append({"drive": drive.name, "t0": float(t[0]),
                                       "lag": lag, "lead": has_lead})
                # unscored diagnostic: where would the chill MPC have braked?
                if da.a_target is not None and i_plan is not None:
                    im = _first_cross(t, da.a_target[sl] < BRAKE_ACCEL)
                    if im is not None:
                        cf.mpc_brake_lags.append(float(t[im] - t[i_driver]))
                events.append(Event(
                    kind="cf_brake", engaged=False, drive=drive.name,
                    t0=float(t[0]), t1=float(t[-1]), i0=i_pre, i1=ev.i1,
                    has_override=False,
                    values={"lag": lag, "never_planned": i_plan is None,
                            "lead": has_lead},
                ))
            elif ev.kind == "pullaway" and not ev.engaged and da.vis_accel is not None:
                # ------------- X3 (lead): launch onset on manual pull-aways
                sl = slice(ev.i0, ev.i1)
                t = da.t[sl]
                t_onset = ev.values.get("t_onset")
                driver_lat = ev.values.get("latency")
                if t_onset is None or driver_lat is None:
                    continue
                horizon = (t >= t_onset) & (t <= t_onset + ONSET_CAP_S)
                ip = _first_cross(t, horizon & (da.vis_accel[sl] > LAUNCH_ACCEL))
                plan_lat = float(t[ip] - t_onset) if ip is not None else ONSET_CAP_S
                lag = plan_lat - float(driver_lat)
                cf.launch.append({"drive": drive.name, "t0": float(t[0]), "lag": lag,
                                  "censored": ip is None, "lead": True})
                events.append(Event(
                    kind="cf_launch", engaged=False, drive=drive.name,
                    t0=float(t[0]), t1=float(t[-1]), i0=ev.i0, i1=ev.i1,
                    has_override=False,
                    values={"lag": lag, "censored": ip is None, "lead": True},
                ))
            elif ev.kind == "launch" and not ev.engaged and da.vis_accel is not None:
                # ------------- X3 (no lead): light turns green, no lead ahead
                if da.lead_status is not None and bool(da.lead_status[ev.i0]):
                    continue  # lead launches are covered by pull-aways
                sl = slice(ev.i0, ev.i1)
                t = da.t[sl]
                t_fm = ev.values.get("t_first_motion")
                if t_fm is None:
                    continue
                # plan rise relative to the driver's first motion (the plan
                # may rise BEFORE the driver moves -> negative lag)
                ip = _first_cross(t, da.vis_accel[sl] > LAUNCH_ACCEL)
                if ip is None:
                    continue
                lag = float(t[ip] - t_fm)
                cf.launch.append({"drive": drive.name, "t0": float(t[0]), "lag": lag,
                                  "censored": False, "lead": False})
                events.append(Event(
                    kind="cf_launch", engaged=False, drive=drive.name,
                    t0=float(t[0]), t1=float(t[-1]), i0=ev.i0, i1=ev.i1,
                    has_override=False,
                    values={"lag": lag, "censored": False, "lead": False},
                ))

    # ---------------- X2: desired-speed agreement over manual cruising
    rms_ss, rms_n, rms_secs = 0.0, 0, 0.0
    op_ratios: dict[str, list] = {"free": [], "lead": []}
    op_secs: dict[str, float] = {"free": 0.0, "lead": 0.0}
    for drive, _seg, da, _evs in per_drive:
        if da.vis_accel is None or len(da.t) < 10:
            continue
        t = da.t
        v = da.v
        dt = float(np.median(np.diff(t)))
        # exclude the STANDSTILL_EXCLUDE_S run-up to every stop
        excl = np.zeros(len(t), bool)
        for a0i, _b in _contiguous_runs(v < 0.3):
            j = np.searchsorted(t, t[a0i] - STANDSTILL_EXCLUDE_S)
            excl[j:a0i + 1] = True
        base = ~da.long_model & (v > 5.0) & ~excl

        # (a) planned vs realized accel RMS
        m = base & np.isfinite(da.vis_accel)
        if m.any():
            diff = da.vis_accel[m] - da.a[m]
            rms_ss += float(np.sum(np.square(diff)))
            rms_n += int(m.sum())
            rms_secs += float(m.sum()) * dt

        # (b) speed opinion: planned speed 4 s ahead vs actual speed 4 s later
        if da.vis_v4 is None:
            continue
        i4 = np.searchsorted(t, t + SPEED_OPINION_HORIZON_S)
        ok = base & (i4 < len(t)) & np.isfinite(da.vis_v4)
        # the future sample must be continuous driving (no gap crossing)
        ok &= (t[np.minimum(i4, len(t) - 1)] - t) < SPEED_OPINION_HORIZON_S + 1.0
        if not ok.any():
            continue
        actual4 = v[np.minimum(i4, len(t) - 1)]
        ratio = da.vis_v4[ok] / np.maximum(actual4[ok], 0.5)
        lead_here = (
            da.lead_status[ok]
            if da.lead_status is not None
            else np.zeros(int(ok.sum()), bool)
        )
        for name, mm in (("lead", lead_here), ("free", ~lead_here)):
            if mm.any():
                op_ratios[name].append(ratio[mm])
                op_secs[name] += float(mm.sum()) * dt
    if rms_n > 100:
        cf.accel_rms = float(np.sqrt(rms_ss / rms_n))
        cf.accel_rms_seconds = rms_secs
    for name, chunks in op_ratios.items():
        if chunks and op_secs[name] >= 10.0:
            allr = np.concatenate(chunks)
            cf.speed_opinion[name] = {
                "median_ratio": float(np.median(allr)),
                "pct": 100.0 * (float(np.median(allr)) - 1.0),
                "seconds": op_secs[name],
            }

    return cf, events
