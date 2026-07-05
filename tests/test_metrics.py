import numpy as np
import pytest

from opgrader import metrics as M

DT = 0.01


def _t(duration):
    return np.arange(int(duration / DT)) * DT


def test_jerk_of_linear_accel_is_constant():
    t = _t(10)
    a = 0.5 * t  # da/dt = 0.5 everywhere
    j = M.jerk(t, a)
    interior = j[100:-100]
    assert np.allclose(interior, 0.5, atol=1e-6)


def test_sine_accel_reversal_rate():
    # a = A sin(2*pi*f*t): sign reverses 2f times per second
    f = 0.5
    t = _t(60)
    a = 0.5 * np.sin(2 * np.pi * f * t)
    rate = M.sign_reversals_per_min(t, a)
    assert rate == pytest.approx(2 * f * 60, rel=0.07)


def test_reversal_hysteresis_ignores_noise():
    t = _t(60)
    a = 0.01 * np.sin(2 * np.pi * 1.0 * t)  # inside +/-0.05 band
    assert M.sign_reversals_per_min(t, a) == 0.0


def test_stop_metrics_known_profile():
    # 10 s of constant -2 m/s^2 from 20 m/s, with an accel step at t=9
    t = _t(10)
    a = np.full(len(t), -2.0)
    a[t >= 9.0] = 0.0  # step of +2 within the last 2 s
    v = np.maximum(20 - 2 * t, 0.0)
    out = M.stop_metrics(t, v, a)
    assert out["peak_decel"] == pytest.approx(2.0, abs=0.05)
    # step of 2 m/s^2 smoothed over 0.3 s -> peak jerk ~ 2/0.3
    assert out["stop_lurch"] == pytest.approx(2.0 / M.SMOOTH_WINDOW_S, rel=0.25)


def test_stop_lurch_smooth_stop_is_small():
    t = _t(10)
    a = np.full(len(t), -2.0)
    v = np.maximum(20 - 2 * t, 0.0)
    out = M.stop_metrics(t, v, a)
    assert out["stop_lurch"] < 0.2


def test_launch_time_to_5():
    t = _t(10)
    v = 1.0 * t  # 1 m/s^2 ramp from standstill
    a = np.ones(len(t))
    out = M.launch_metrics(t, v, a, t_first_motion=0.0)
    assert out["time_to_5"] == pytest.approx(5.0, abs=0.02)


def test_detrended_std_removes_ramp():
    t = _t(30)
    x = 3.0 + 0.1 * t  # pure trend
    assert M.detrended_std(x) == pytest.approx(0.0, abs=1e-9)
    x2 = x + np.sin(2 * np.pi * 0.2 * t)
    assert M.detrended_std(x2) == pytest.approx(1 / np.sqrt(2), rel=0.05)


def test_time_gap():
    d = np.array([30.0, 30.0, 30.0])
    v = np.array([15.0, 0.5, 10.0])
    tg = M.time_gap(d, v)
    assert tg[0] == pytest.approx(2.0)
    assert np.isnan(tg[1])
    assert tg[2] == pytest.approx(3.0)


def test_rms_and_p95():
    x = np.array([3.0, -4.0])
    assert M.rms(x) == pytest.approx(np.sqrt(12.5))
    y = np.linspace(0, 100, 1001)
    assert M.p95_abs(y) == pytest.approx(95.0, abs=0.2)


def test_pct_time_above():
    t = _t(10)
    x = np.where(t < 2.5, 3.0, 0.0)
    assert M.pct_time_above(t, x, 2.0) == pytest.approx(25.0, abs=0.5)
