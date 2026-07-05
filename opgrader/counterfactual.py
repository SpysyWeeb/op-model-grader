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
- longitudinalPlan.aTarget is live while long is disengaged and properly
  negative when braking behind a lead (median -1.09 m/s2) but weak for
  no-lead stops (median -0.15, cruise unset plans toward max speed) =>
  braking-onset comparison is gated on a lead being present at approach
  start. The follow-gap opinion is additionally gated on
  longitudinalPlanSource being a lead source (lead0/1/2): in experimental
  mode the e2e source dominates and is not target-following.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import DriveArrays, Event
from .lateral import MPH, PP_BINS_MPH, TurnEpisode, IntentWindow
from .metrics import effective_t_follow, smooth
from .segments import _contiguous_runs

# thresholds
BRAKE_ACCEL = -0.5  # m/s^2: "braking has begun" for both plan and driver
LAUNCH_ACCEL = 0.3  # m/s^2: "launch has begun" for the plan
TURN_IN_FRACTION = 0.30  # plan onset = crossing 30% of driver's peak curvature
UNWIND_FRACTION = 0.5
MIN_SPEED_PATH = 3.0  # m/s: path agreement needs the car to be moving
BRAKE_LEAD_DREL = 60.0  # lead must be present & closer than this at approach start
LEAD_SOURCES = (1, 2, 3)  # longitudinalPlanSource: lead0, lead1, lead2
ONSET_CAP_S = 6.0


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
    # C1 / C2
    braking: list[dict] = field(default_factory=list)
    launch: list[dict] = field(default_factory=list)
    braking_never: int = 0  # windows where the plan never reached BRAKE_ACCEL
    braking_skipped_no_lead: int = 0
    # C3 per personality: driver's gap vs the target the plan pursues
    follow_opinion: dict = field(default_factory=dict)
    follow_opinion_note: str = ""

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
        lags = [r["lag"] for r in self.braking]
        return {
            "n": len(lags),
            "median": float(np.median(lags)) if lags else None,
            "never": self.braking_never,
            "skipped_no_lead": self.braking_skipped_no_lead,
            "lags": lags,
        }

    def launch_summary(self) -> dict:
        lags = [r["lag"] for r in self.launch]
        return {"n": len(lags), "median": float(np.median(lags)) if lags else None,
                "lags": lags}


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
    have_plan = any(da.a_target is not None for _d, _s, da, _e in per_drive)
    if not (have_curv or have_plan):
        cf.why_unavailable = "logs carry neither modelV2 nor longitudinalPlan"
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

    # ---------------- C1: braking onset on manual stops behind a lead
    for drive, _seg, da, evs in per_drive:
        if da.a_target is None:
            continue
        a_s = smooth(da.t, da.a)
        for ev in evs:
            if ev.kind == "stop" and not ev.engaged:
                sl = slice(ev.i0, ev.i1)
                t = da.t[sl]
                lead_ok = (
                    da.lead_status is not None and da.d_rel is not None
                    and bool(da.lead_status[ev.i0])
                    and float(da.d_rel[ev.i0]) < BRAKE_LEAD_DREL
                )
                if not lead_ok:
                    cf.braking_skipped_no_lead += 1
                    continue
                i_driver = _first_cross(t, a_s[sl] < BRAKE_ACCEL)
                i_plan = _first_cross(t, da.a_target[sl] < BRAKE_ACCEL)
                if i_driver is None:
                    continue
                if i_plan is None:
                    cf.braking_never += 1
                else:
                    lag = float(t[i_plan] - t[i_driver])
                    cf.braking.append({"drive": drive.name, "t0": float(t[0]), "lag": lag})
                events.append(Event(
                    kind="cf_brake", engaged=False, drive=drive.name,
                    t0=float(t[0]), t1=float(t[-1]), i0=ev.i0, i1=ev.i1,
                    has_override=False,
                    values={"lag": None if i_plan is None else float(t[i_plan] - t[i_driver]),
                            "never_planned": i_plan is None},
                ))
            elif ev.kind == "pullaway" and not ev.engaged:
                # ---------------- C2: launch onset on manual lead pull-aways
                sl = slice(ev.i0, ev.i1)
                t = da.t[sl]
                t_onset = ev.values.get("t_onset")
                driver_lat = ev.values.get("latency")
                if t_onset is None or driver_lat is None:
                    continue
                after = t >= t_onset
                horizon = after & (t <= t_onset + ONSET_CAP_S)
                ip = _first_cross(t, horizon & (da.a_target[sl] > LAUNCH_ACCEL))
                plan_lat = float(t[ip] - t_onset) if ip is not None else ONSET_CAP_S
                lag = plan_lat - float(driver_lat)
                cf.launch.append({"drive": drive.name, "t0": float(t[0]), "lag": lag,
                                  "censored": ip is None})
                events.append(Event(
                    kind="cf_launch", engaged=False, drive=drive.name,
                    t0=float(t[0]), t1=float(t[-1]), i0=ev.i0, i1=ev.i1,
                    has_override=False,
                    values={"lag": lag, "censored": ip is None},
                ))

    # ---------------- C3: follow-gap opinion (lead-source samples only)
    by_p: dict[str, list] = {}
    secs: dict[str, float] = {}
    have_source = any(da.plan_source is not None for _d, _s, da, _e in per_drive)
    for drive, _seg, da, _evs in per_drive:
        if da.d_rel is None or da.v_lead is None or da.plan_source is None \
                or da.personality is None:
            continue
        dt = float(np.median(np.diff(da.t))) if len(da.t) > 1 else 0.01
        # whole-timeline steady-follow mask (a 15 s contiguous follow window
        # is not required: lead-source plan samples are sparse under e2e)
        v = da.v
        vl = da.v_lead
        mask = (
            ~da.long_model
            & (v > 8.0)
            & (da.lead_status if da.lead_status is not None else True)
            & (np.abs(vl - v) < 1.5)
            & (np.abs(da.a) < 0.5)
            & np.isin(da.plan_source, LEAD_SOURCES)
        )
        if not mask.any():
            continue
        eff = effective_t_follow(v, vl, da.d_rel)
        pers = da.personality
        for idx, name in ((0, "aggressive"), (1, "standard"), (2, "relaxed")):
            m = mask & (pers == idx) & np.isfinite(eff)
            if m.any():
                by_p.setdefault(name, []).append(eff[m])
                secs[name] = secs.get(name, 0.0) + float(m.sum()) * dt
    for name, chunks in by_p.items():
        allv = np.concatenate(chunks)
        cf.follow_opinion[name] = {
            "driver_median": float(np.median(allv)),
            "target": float(t_follow_targets.get(name, 0.0)),
            "seconds": secs.get(name, 0.0),
        }
    if not have_source:
        cf.follow_opinion_note = "plan source not in these logs; C3 skipped"
    elif not cf.follow_opinion:
        cf.follow_opinion_note = (
            "no manual steady-follow samples where the lead constraint binds "
            "(plan source is mostly e2e in experimental mode)"
        )

    return cf, events
