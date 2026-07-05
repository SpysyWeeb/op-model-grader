"""Pure numpy metric functions: arrays in, floats out.

Jerk is the derivative of aEgo after a centered moving-average smooth with a
~0.3 s window (SMOOTH_WINDOW_S); at carState's 100 Hz that is ~31 samples.
The same smoothing is applied before counting accel sign reversals.
"""

from __future__ import annotations

import numpy as np

SMOOTH_WINDOW_S = 0.3
REVERSAL_HYST = 0.05  # m/s^2 hysteresis for accel sign reversals


def smooth(t: np.ndarray, x: np.ndarray, window_s: float = SMOOTH_WINDOW_S) -> np.ndarray:
    """Centered moving average with a window of ~window_s seconds."""
    n = len(x)
    if n < 3:
        return x.astype(np.float64, copy=True)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return x.astype(np.float64, copy=True)
    w = max(1, int(round(window_s / dt)))
    if w % 2 == 0:
        w += 1
    if w <= 1 or w >= n:
        return x.astype(np.float64, copy=True)
    kernel = np.ones(w) / w
    pad = w // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, kernel, mode="valid")


def derivative(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gradient of x w.r.t. t (same length as x)."""
    if len(x) < 2:
        return np.zeros_like(x, dtype=np.float64)
    return np.gradient(x.astype(np.float64), t)


def jerk(t: np.ndarray, a: np.ndarray, window_s: float = SMOOTH_WINDOW_S) -> np.ndarray:
    """Jerk (m/s^3): derivative of smoothed accel."""
    return derivative(t, smooth(t, a, window_s))


def rms(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def p95_abs(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.percentile(np.abs(x), 95))


def sign_reversals_per_min(
    t: np.ndarray, x: np.ndarray, hyst: float = REVERSAL_HYST
) -> float:
    """Rate of sign reversals of x, with +/-hyst hysteresis.

    A reversal is counted each time x, having been committed to one sign
    (|x| > hyst), crosses to being committed to the opposite sign.
    """
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    if dur <= 0:
        return float("nan")
    committed = np.where(x > hyst, 1, np.where(x < -hyst, -1, 0))
    nz = committed[committed != 0]
    if len(nz) < 2:
        return 0.0
    flips = int(np.count_nonzero(np.diff(nz) != 0))
    return flips / (dur / 60.0)


def pct_time_above(t: np.ndarray, x: np.ndarray, thresh: float) -> float:
    """Percent of samples with x > thresh (uniform-rate approximation)."""
    ok = np.isfinite(x)
    if not ok.any():
        return float("nan")
    return 100.0 * float(np.mean(x[ok] > thresh))


def detrended_std(x: np.ndarray) -> float:
    """Std after removing a linear trend."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    i = np.arange(n, dtype=np.float64)
    coef = np.polyfit(i, x, 1)
    return float(np.std(x - np.polyval(coef, i)))


def time_gap(d_rel: np.ndarray, v_ego: np.ndarray, min_v: float = 1.0) -> np.ndarray:
    """Time gap (s) = dRel / vEgo where vEgo > min_v; NaN elsewhere."""
    out = np.full(len(d_rel), np.nan)
    ok = v_ego > min_v
    out[ok] = d_rel[ok] / v_ego[ok]
    return out


# ---------------------------------------------------------------- stops


def stop_metrics(
    t: np.ndarray, v: np.ndarray, a: np.ndarray, t_standstill: float | None = None
) -> dict[str, float]:
    """Metrics for one stop-approach window (start = last v>=8, end ~ standstill).

    t_standstill anchors the "last 2 s" lurch window; defaults to window end.
    """
    out: dict[str, float] = {}
    a_s = smooth(t, a)
    j = derivative(t, a_s)
    t_stand = t[-1] if t_standstill is None else t_standstill

    approach = t <= t_stand
    a_app = a_s[approach] if approach.any() else a_s
    t_app = t[approach] if approach.any() else t
    out["peak_decel"] = float(-np.min(a_app))  # positive number, m/s^2
    dur = float(t_stand - t[0])
    if dur > 0:
        out["peak_decel_frac"] = float((t_app[int(np.argmin(a_app))] - t[0]) / dur)
    else:
        out["peak_decel_frac"] = float("nan")

    last2 = (t >= t_stand - 2.0) & (t <= t_stand)
    out["stop_lurch"] = float(np.max(np.abs(j[last2]))) if last2.any() else float("nan")

    below = np.flatnonzero(v < 0.2)
    if len(below) > 0:
        out["accel_at_crawl"] = float(abs(a_s[below[0]]))
    else:
        out["accel_at_crawl"] = float("nan")
    return out


# --------------------------------------------------------------- launches


def launch_metrics(
    t: np.ndarray, v: np.ndarray, a: np.ndarray, t_first_motion: float
) -> dict[str, float]:
    out: dict[str, float] = {}
    j = jerk(t, a)
    after = t >= t_first_motion
    hit5 = np.flatnonzero(after & (v >= 5.0))
    out["time_to_5"] = (
        float(t[hit5[0]] - t_first_motion) if len(hit5) else float("nan")
    )
    out["peak_jerk"] = float(np.max(np.abs(j))) if len(j) else float("nan")
    return out


# --------------------------------------------------------------- following


def follow_metrics(
    t: np.ndarray, v: np.ndarray, a: np.ndarray, d_rel: np.ndarray
) -> dict[str, float]:
    tg = time_gap(d_rel, v)
    tg_ok = tg[np.isfinite(tg)]
    return {
        "median_gap": float(np.median(tg_ok)) if len(tg_ok) else float("nan"),
        "gap_hunting": detrended_std(tg),
        "accel_reversals": sign_reversals_per_min(t, smooth(t, a)),
    }


# --------------------------------------------------------------- smoothness


def smoothness_metrics(t: np.ndarray, a: np.ndarray) -> dict[str, float]:
    a_s = smooth(t, a)
    j = derivative(t, a_s)
    return {
        "rms_jerk": rms(j),
        "p95_jerk": p95_abs(j),
        "accel_reversals": sign_reversals_per_min(t, a_s),
        "pct_hard_accel": pct_time_above(t, np.abs(a), 2.0),
    }


# ----------------------------------------------------------------- lateral


def lateral_metrics(
    t: np.ndarray,
    v: np.ndarray,
    steering_rate: np.ndarray,
    lat_accel: np.ndarray | None,
) -> dict[str, float]:
    """Lateral quality over a span. lat_accel may be None (no yaw source)."""
    out: dict[str, float] = {}
    fast = v > 10.0
    if fast.sum() >= 3:
        out["steer_rate_rms"] = rms(steering_rate[fast])
        out["steer_reversals"] = sign_reversals_per_min(
            t[fast], smooth(t[fast], steering_rate[fast]), hyst=1.0
        )
    if lat_accel is not None and np.isfinite(lat_accel).sum() >= 3:
        lat_s = smooth(t, lat_accel)
        out["rms_lat_jerk"] = rms(derivative(t, lat_s))
        out["pct_high_lat"] = pct_time_above(t, np.abs(lat_accel), 3.0)
    return out
