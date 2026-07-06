"""Self-contained HTML report: grades, distributions, event drill-downs.

Zero external requests: uPlot (vendored, MIT) plus all CSS/JS/JSON are
inlined. Event traces are downsampled to 10 Hz and rounded to 3 decimals
to keep the file small. Light and dark themes via prefers-color-scheme.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import numpy as np

from . import __version__
from .events import DriveArrays, Event
from .grading import CategoryResult, GradeReport, MetricResult, S_CUTOFF
from .metrics import derivative
from .lateral import PP_HP_WINDOW_S, highpass_angle

_ASSETS = Path(__file__).resolve().parent / "assets" / "uplot"

MAX_EVENTS_PER_KIND = 50
TRACE_HZ = 10.0

KIND_LABELS = {
    "stop": "Stop approaches",
    "launch": "Launches",
    "follow": "Lead follow windows",
    "lead_decel": "Lead decel responses",
    "pullaway": "Lead pull-aways",
    "cruise": "Free cruise windows",
    "turn": "Turn episodes",
    "intent": "Blinker turn intents",
    "pingpong": "Worst oscillation windows",
    "cf_turnin": "Plan vs You: turn-in (manual turns)",
    "cf_brake": "Plan vs You: braking onset (manual stops)",
    "cf_launch": "Plan vs You: launch onset (manual pull-aways)",
    "gas_override": "Gas overrides (Speed Disagreement)",
}

KIND_HEADLINE = {
    "stop": ("stop_lurch", "lurch m/s³"),
    "launch": ("time_to_5", "0→5 m/s s"),
    "follow": ("median_gap", "gap s"),
    "lead_decel": ("latency", "latency s"),
    "pullaway": ("latency", "latency s"),
    "cruise": (None, "duration s"),
    "turn": ("peak_deg", "peak °"),
    "intent": ("delay", "turn-in delay s"),
    "pingpong": ("osc_rms", "osc RMS °"),
    "cf_turnin": ("lag", "plan lag s"),
    "cf_brake": ("lag", "plan lag s"),
    "cf_launch": ("lag", "plan lag s"),
    "gas_override": ("magnitude", "magnitude m/s²"),
}


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{digits}f}"


def _round_list(x) -> list:
    return [None if not np.isfinite(v) else round(float(v), 3) for v in np.asarray(x, dtype=float)]


def _downsample(t: np.ndarray, x: np.ndarray, hz: float = TRACE_HZ):
    if len(t) < 2:
        return t, x
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return t, x
    step = max(1, int(round(1.0 / (hz * max(dt, 1e-3)))))
    return t[::step], x[::step]


# ------------------------------------------------------------------ payload


def _series_for_event(ev: Event, da: DriveArrays) -> list[dict]:
    sl = slice(ev.i0, ev.i1)
    t = da.t[sl]

    def ds(arr):
        _ti, x = _downsample(t, arr)
        return _round_list(x)

    ti, _ = _downsample(t, da.v[sl])
    out = [{"label": "t", "data": _round_list(ti - ti[0] if len(ti) else ti)}]

    if ev.kind == "turn":
        if da.steering_angle is not None:
            out.append({"label": "steering angle", "unit": "°", "data": ds(da.steering_angle[sl])})
        if ev.engaged and da.cmd_angle is not None:
            out.append({"label": "commanded angle", "unit": "°", "data": ds(da.cmd_angle[sl])})
        out.append({"label": "vEgo", "unit": "m/s", "data": ds(da.v[sl])})
        if da.steering_pressed is not None:
            out.append({"label": "steeringPressed", "unit": "", "data": ds(da.steering_pressed[sl].astype(float))})
        if ev.engaged and da.torque_output is not None:
            out.append({"label": "commanded torque", "unit": "-1..1", "data": ds(da.torque_output[sl])})
        if da.driver_torque is not None:
            out.append({"label": "driver torque", "unit": "raw", "data": ds(da.driver_torque[sl])})
    elif ev.kind == "pingpong":
        if da.steering_angle is not None:
            resid = highpass_angle(t, da.steering_angle[sl])
            out.append({"label": f"high-passed angle ({PP_HP_WINDOW_S:.0f}s)", "unit": "°", "data": ds(resid)})
        out.append({"label": "vEgo", "unit": "m/s", "data": ds(da.v[sl])})
    elif ev.kind == "cf_turnin":
        if da.desired_curv is not None:
            out.append({"label": "planned curvature (left+)", "unit": "1/km",
                        "data": ds(-1000.0 * da.desired_curv[sl])})
        if da.steering_angle is not None:
            out.append({"label": "steering angle", "unit": "°", "data": ds(da.steering_angle[sl])})
        out.append({"label": "vEgo", "unit": "m/s", "data": ds(da.v[sl])})
    elif ev.kind in ("cf_brake", "cf_launch"):
        if da.vis_accel is not None:
            out.append({"label": "planned accel (vision)", "unit": "m/s²", "data": ds(da.vis_accel[sl])})
        if da.a_target is not None:
            out.append({"label": "planned accel (MPC, diagnostic)", "unit": "m/s²", "data": ds(da.a_target[sl])})
        out.append({"label": "aEgo", "unit": "m/s²", "data": ds(da.a[sl])})
        out.append({"label": "vEgo", "unit": "m/s", "data": ds(da.v[sl])})
        if da.d_rel is not None and da.lead_status is not None:
            d = np.where(da.lead_status[sl], da.d_rel[sl], np.nan)
            out.append({"label": "lead distance", "unit": "m", "data": ds(d)})
    elif ev.kind == "gas_override":
        out.append({"label": "vEgo", "unit": "m/s", "data": ds(da.v[sl])})
        out.append({"label": "aEgo", "unit": "m/s²", "data": ds(da.a[sl])})
        if da.vis_accel is not None:
            out.append({"label": "planned accel (vision)", "unit": "m/s²", "data": ds(da.vis_accel[sl])})
        if da.gas_pressed is not None:
            out.append({"label": "gasPressed", "unit": "", "data": ds(da.gas_pressed[sl].astype(float))})
        if da.d_rel is not None and da.lead_status is not None:
            d = np.where(da.lead_status[sl], da.d_rel[sl], np.nan)
            out.append({"label": "lead distance", "unit": "m", "data": ds(d)})
    elif ev.kind == "intent":
        if da.steering_angle is not None:
            out.append({"label": "steering angle", "unit": "°", "data": ds(da.steering_angle[sl])})
        if ev.engaged and da.cmd_angle is not None:
            out.append({"label": "commanded angle", "unit": "°", "data": ds(da.cmd_angle[sl])})
        out.append({"label": "vEgo", "unit": "m/s", "data": ds(da.v[sl])})
        if da.steering_pressed is not None:
            out.append({"label": "steeringPressed", "unit": "", "data": ds(da.steering_pressed[sl].astype(float))})
    else:
        out.append({"label": "vEgo", "unit": "m/s", "data": ds(da.v[sl])})
        out.append({"label": "aEgo", "unit": "m/s²", "data": ds(da.a[sl])})
        out.append({"label": "jerk", "unit": "m/s³", "data": ds(derivative(t, da.a_smooth[sl]))})
        if da.d_rel is not None and da.lead_status is not None:
            d = np.where(da.lead_status[sl], da.d_rel[sl], np.nan)
            out.append({"label": "lead distance", "unit": "m", "data": ds(d)})
    return out


def _event_payload(ev: Event, da: DriveArrays, t_drive0: float) -> dict:
    key, _unit = KIND_HEADLINE[ev.kind]
    headline = (ev.t1 - ev.t0) if key is None else ev.values.get(key)
    if headline is not None and not np.isfinite(headline):
        headline = None
    tag = ""
    if ev.kind == "turn":
        tag = ("sharp " if ev.values.get("sharp") else "curve ") + str(ev.values.get("side", ""))
        if ev.values.get("rescued"):
            tag += " · rescued"
        if ev.engaged and ev.values.get("initiator") and ev.values["initiator"] != "unknown":
            tag += f" · {ev.values['initiator']}-led"
        if ev.values.get("divergence_deg") is not None:
            tag += f" · resisted {ev.values['divergence_deg']:.0f}°"
    elif ev.kind == "intent":
        tag = f"{ev.values.get('side', '')} · {ev.values.get('outcome', '')}"
        if ev.values.get("missed"):
            tag += " · MISSED"
    elif ev.kind in ("cf_turnin", "cf_brake", "cf_launch"):
        tag = str(ev.values.get("side", ""))
        if "lead" in ev.values:
            tag = (tag + (" lead" if ev.values["lead"] else " NO-LEAD")).strip()
        if ev.values.get("never_planned"):
            tag = (tag + " · NEVER PLANNED").strip(" ·")
        elif ev.values.get("censored"):
            tag = (tag + " · plan never crossed (censored)").strip(" ·")
    elif ev.kind == "gas_override":
        tag = str(ev.values.get("context", ""))
        if ev.values.get("reoverride"):
            tag += " · re-override"
    return {
        "kind": ev.kind,
        "engaged": ev.engaged,
        "override": ev.has_override,
        "drive": ev.drive,
        "t0": round(ev.t0 - t_drive0, 1),
        "dur": round(ev.t1 - ev.t0, 1),
        "headline": None if headline is None else round(float(headline), 3),
        "censored": bool(ev.values.get("censored", False)),
        "tag": tag,
        "series": _series_for_event(ev, da),
    }


# --------------------------------------------------------------- histograms


def _svg_hist(model_vals: list[float], driver_vals: list[float], title: str, unit: str) -> str:
    """Paired histogram (share-of-events per bin), model vs you."""
    allv = np.array([v for v in model_vals + driver_vals if np.isfinite(v)])
    if len(allv) < 2 or (len(model_vals) == 0 and len(driver_vals) == 0):
        return ""
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    nbins = min(12, max(5, int(np.sqrt(len(allv)) * 2)))
    edges = np.linspace(lo, hi, nbins + 1)

    def shares(vals):
        if not vals:
            return np.zeros(nbins)
        h, _ = np.histogram(vals, bins=edges)
        return h / max(1, len(vals))

    sm, sd = shares(model_vals), shares(driver_vals)
    peak = max(sm.max(initial=0), sd.max(initial=0), 1e-9)

    W, H, PAD_B, PAD_T = 560, 150, 22, 8
    plot_h = H - PAD_B - PAD_T
    bin_w = W / nbins
    bar_w = max(3.0, bin_w / 2 - 4)
    parts = [
        f'<svg class="hist" viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(title)} histogram">'
    ]
    for i in range(nbins):
        x0 = i * bin_w + 2
        for k, s in enumerate((sm, sd)):
            hh = plot_h * (s[i] / peak)
            if hh < 0.5:
                continue
            x = x0 + k * (bar_w + 2)
            y = PAD_T + plot_h - hh
            cls = "hm" if k == 0 else "hd"
            parts.append(
                f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{hh:.1f}" rx="2"/>'
            )
    parts.append(
        f'<line class="hax" x1="0" y1="{PAD_T + plot_h + 0.5}" x2="{W}" y2="{PAD_T + plot_h + 0.5}"/>'
    )
    for frac, val in ((0, lo), (0.5, (lo + hi) / 2), (1, hi)):
        anchor = "start" if frac == 0 else ("end" if frac == 1 else "middle")
        parts.append(
            f'<text class="hlab" x="{frac * W + (4 if frac == 0 else -4 if frac == 1 else 0)}" '
            f'y="{H - 6}" text-anchor="{anchor}">{val:.2f}{(" " + _esc(unit)) if unit else ""}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------- pieces


def _driver_cell(m: MetricResult) -> str:
    """'You' column: the combined (this-drive + pooled-profile) aggregate,
    with provenance when a driver profile actually contributed something --
    invisible when it didn't, so a non-pooled or --no-profile run looks
    exactly like it always has."""
    val = _fmt(m.driver_agg)
    if m.n_pooled <= 0:
        return val
    extra = f'<span class="muted"> (n={m.n_this_drive} this drive + {m.n_pooled} pooled = {m.n_driver})</span>'
    if (
        m.same_drive_agg is not None
        and m.driver_agg is not None
        and abs(m.same_drive_agg - m.driver_agg) > 1e-9
    ):
        extra += f'<span class="muted"> · this drive alone: {_fmt(m.same_drive_agg)}</span>'
    return val + extra


def _metric_rows(metrics: list[MetricResult], row_overrides: dict | None = None) -> str:
    """row_overrides: {metric_key: {"model_deg"|"model_txt": ..., "you_deg"|"you_txt": ...,
    "n": int}} -- lets a specific category show a more interpretable pair of
    cells than the generic Model/You aggregate (a "_deg" value is formatted
    like any other number; a "_txt" value is used verbatim, for a fixed
    reference string like "0.00 (reference)"). Either side of a pair can be
    overridden independently -- the other falls back to normal rendering.
    Keeps the shared renderer ignorant of any one specific metric.

    A metric with `desc` set renders as a clickable row (a small "▸" cue,
    cursor:pointer, matching the existing tr.evrow drill-down convention)
    that reveals a hidden explanation row below it on click -- see the
    tr.hasdesc rule in _CSS and the click handler in _JS."""
    row_overrides = row_overrides or {}
    rows = []
    for m in metrics:
        d = m.definition
        if d.scorer == "none" and not d.show_unscored:
            continue  # diagnostic-only; not worth a row (see cmd_unwind_lead_*)
        ov = row_overrides.get(d.key) or {}
        if "model_txt" in ov:
            model_txt = ov["model_txt"]
        elif "model_deg" in ov:
            model_txt = _fmt(ov["model_deg"])
        else:
            model_txt = _fmt(m.model_agg)
        if "you_txt" in ov:
            you_txt = ov["you_txt"]
        elif "you_deg" in ov:
            you_txt = _fmt(ov["you_deg"])
        else:
            you_txt = _driver_cell(m)
        if ov.get("n") and ("you_deg" in ov or "you_txt" in ov):
            you_txt += f' <span class="muted">(n={ov["n"]})</span>'

        icon = '<span class="rowinfo">▸</span>' if d.desc else ""
        if d.desc:
            desc_id = f"desc-{d.key}"
            tr_attrs = f' data-desc-id="{desc_id}"'
            desc_row = f'<tr class="descrow" id="{desc_id}"><td colspan="5">{_esc(d.desc)}</td></tr>'
        else:
            tr_attrs = ""
            desc_row = ""

        if d.scorer == "none":
            # show_unscored=True got it past the hide rule above, but it's
            # still not a grade -- "insufficient data" below would be a lie
            # when there's plenty of data, just no scoring formula for it.
            note = d.note or "not scored"
            if m.n_model == 0:
                note = "no data"
            cls = "hasdesc insuff" if d.desc else "insuff"
            rows.append(
                f'<tr class="{cls}"{tr_attrs}><td>{_esc(d.label)}{icon}</td>'
                f"<td>{model_txt}</td><td>{you_txt}</td>"
                f"<td>{_esc(d.unit)}</td><td>{_esc(note)}</td></tr>{desc_row}"
            )
        elif m.score is None:
            need_d = " each" if d.needs_driver else ""
            note = f"insufficient data (model n={m.n_model}, you n={m.n_driver}, need ≥3{need_d})"
            cls = "hasdesc insuff" if d.desc else "insuff"
            rows.append(
                f'<tr class="{cls}"{tr_attrs}><td>{_esc(d.label)}{icon}</td>'
                f"<td>{model_txt}</td><td>{you_txt}</td>"
                f"<td>{_esc(d.unit)}</td><td>{_esc(note)}</td></tr>{desc_row}"
            )
        else:
            star = "*" if (d.scorer == "abs" or (d.scorer == "ratio_or_abs" and (m.n_driver < 3 or (m.driver_agg or 0) < (d.abs_when_driver_below or 0)))) else ""
            cls = ' class="hasdesc"' if d.desc else ""
            rows.append(
                f"<tr{cls}{tr_attrs}><td>{_esc(d.label)}{star}{icon}</td>"
                f"<td>{model_txt}</td><td>{you_txt}</td>"
                f"<td>{_esc(d.unit)}</td><td>{m.score:.0f}</td></tr>{desc_row}"
            )
    return "".join(rows)


def _pingpong_card(cat: CategoryResult) -> str:
    bins = cat.extra.get("bins", [])
    sub_bins = cat.extra.get("sub_bins", [])
    worst = cat.extra.get("worst")

    def bin_rows(bs):
        rows = []
        for b in bs:
            score = f"{b.score:.0f}" if b.score is not None else "–"
            manual_rms = _fmt(b.manual_rms)
            manual_rev = _fmt(b.manual_rev, 1)
            if b.pooled_n > 0:
                manual_rms += f'<span class="muted"> ({_fmt(b.pooled_manual_rms)} w/ +{b.pooled_n} pooled)</span>'
                manual_rev += f'<span class="muted"> ({_fmt(b.pooled_manual_rev, 1)} pooled)</span>'
            rows.append(
                f"<tr><td>{b.lo_mph:.0f}–{('' if b.hi_mph < 150 else '+')}{'' if b.hi_mph >= 150 else f'{b.hi_mph:.0f}'} mph</td>"
                f"<td>{b.engaged_s:.0f}s / {b.manual_s:.0f}s</td>"
                f"<td>{_fmt(b.engaged_rms)} / {manual_rms}</td>"
                f"<td>{_fmt(b.engaged_rev, 1)} / {manual_rev}</td>"
                f"<td>{score}</td></tr>"
            )
        return "".join(rows)

    worst_html = ""
    if worst is not None:
        worst_html = (
            f'<div class="callout">Worst speed range: <strong>{worst.lo_mph:.0f}–{worst.hi_mph:.0f} mph</strong> '
            f"(bin score {worst.score:.0f})</div>"
        )
    sub_html = ""
    if sub_bins:
        sub_html = f"""
  <details><summary class="muted">1 mph resolution, 0–10 mph</summary>
  <table class="mtable">
    <thead><tr><th>Speed</th><th>Time (eng/man)</th><th>Osc RMS ° (eng/man)</th><th>Reversals/min (eng/man)</th><th>Bin score</th></tr></thead>
    <tbody>{bin_rows(sub_bins)}</tbody>
  </table></details>"""
    if not bins:
        return '<p class="muted">No steering data for ping-pong analysis.</p>'
    return f"""
  {worst_html}
  <table class="mtable">
    <thead><tr><th>Speed</th><th>Time (eng/man)</th><th>Osc RMS ° (eng/man)</th><th>Reversals/min (eng/man)</th><th>Bin score</th></tr></thead>
    <tbody>{bin_rows(bins)}</tbody>
  </table>
  {sub_html}
  <p class="muted">Oscillation = steering angle minus its centered 2 s moving average; reversals
  counted when the swing between extrema exceeds 3°. A bin is scored only with ≥30 s on each side;
  category score is the engaged-time-weighted mean of bin scores. "w/ +N pooled" means the manual
  baseline for that bin also draws on N other routes from your driver profile (model/engaged data
  is never pooled, only your own driving).</p>"""


def _breakdown_tables(breakdowns: dict) -> str:
    """Per-mode and per-personality longitudinal sub-tables.

    Columns = buckets; rows = metrics (grouped by category). Cells show the
    bucket's model aggregate with its sample count and score; buckets keep
    the n>=3 gate ("--" = insufficient data in that bucket)."""
    if not breakdowns:
        return ""
    out = []
    for dim, title in (("mode", "By mode (Chill / Experimental)"),
                       ("personality", "By personality")):
        bg = breakdowns.get(dim) or {}
        if not bg:
            continue
        buckets = list(bg)
        heads = "".join(f"<th>{_esc(b)}</th>" for b in buckets)
        grade_cells = "".join(
            f"<td><strong>{_esc(bg[b].letter or '–')}</strong>"
            + (f" <span class='muted'>{_lettered_score(bg[b].score)}</span>" if bg[b].score is not None else "")
            + "</td>"
            for b in buckets
        )
        rows = [f'<tr><td><strong>Longitudinal grade</strong></td><td></td>{grade_cells}</tr>']
        # metric rows grouped by category (same order as the bucket results)
        first = bg[buckets[0]]
        for ci, cat in enumerate(first.categories):
            cat_grades = []
            for b in buckets:
                c = bg[b].categories[ci]
                cat_grades.append(
                    f"<td>{_esc(c.letter or '–')}"
                    + (f" <span class='muted'>{_lettered_score(c.score)}</span>" if c.score is not None else "")
                    + "</td>"
                )
            rows.append(
                f'<tr class="bdcat"><td>{_esc(cat.name)}</td><td></td>{"".join(cat_grades)}</tr>'
            )
            for mi, m0 in enumerate(cat.metrics):
                if all(bg[b].categories[ci].metrics[mi].n_model == 0 for b in buckets):
                    continue
                d = m0.definition
                cells = []
                for b in buckets:
                    mres = bg[b].categories[ci].metrics[mi]
                    if mres.n_model == 0:
                        cells.append('<td class="muted">–</td>')
                    elif mres.score is None:
                        cells.append(
                            f'<td class="muted">{_fmt(mres.model_agg)} '
                            f'<span class="muted">(n={mres.n_model})</span></td>'
                        )
                    else:
                        cells.append(
                            f"<td>{_fmt(mres.model_agg)} "
                            f'<span class="muted">(n={mres.n_model})</span> → {mres.score:.0f}</td>'
                        )
                you = _fmt(m0.driver_agg)
                rows.append(
                    f'<tr><td class="bdmetric">{_esc(d.label)}'
                    f'{(" (" + _esc(d.unit) + ")") if d.unit else ""}</td>'
                    f"<td>{you}</td>{''.join(cells)}</tr>"
                )
        out.append(f"""
<div class="card" style="overflow-x:auto">
  <h3>{_esc(title)}</h3>
  <table class="mtable">
    <thead><tr><th>Metric</th><th>You</th>{heads}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>""")
    if not out:
        return ""
    return f"""
<section>
  <h2>Longitudinal breakdowns <span class="muted">same human baseline; mixed-mode events excluded</span></h2>
  {''.join(out)}
</section>"""


def _lagfmt(v, digits=2) -> str:
    if v is None:
        return "–"
    return f"{v:+.{digits}f}"


def _counterfactual_section(cf) -> str:
    data_tip = (
        '<p class="muted cathelp"><b>Getting good data for this section:</b> it is '
        'computed from YOUR driving, so it grows every mile you drive manually. Most '
        'valuable: signaled low-speed turns, red-light/stop-sign approaches without a '
        'car ahead, launches when a light turns green, and normal cruising — the plan '
        'is recorded alongside whether you followed it or not.</p>'
    )
    if cf is None or not cf.available:
        why = getattr(cf, "why_unavailable", "") if cf is not None else ""
        if why:
            return (f'<section><h2>Plan vs You (counterfactual)</h2>'
                    f'<p class="muted">Unavailable: {_esc(why)}.</p></section>')
        return ""
    parts = []

    # L1 path agreement
    if cf.path_overall is not None:
        rows = "".join(
            f"<tr><td>{b['lo_mph']:.0f}–{b['hi_mph']:.0f} mph</td>"
            f"<td>{b['rms']:.2f}</td><td>{b['seconds']:.0f}s</td></tr>"
            for b in cf.path_bins
        )
        parts.append(f"""
<div class="card">
  <h3>Path agreement</h3>
  <p class="muted">RMS lateral-accel disagreement between the model's planned curvature and your
  steering, over your manual driving: <strong>{cf.path_overall:.2f} m/s²</strong>
  ({cf.path_seconds:.0f} s of data). Lower = the model would have steered like you.</p>
  <table class="mtable"><thead><tr><th>Speed</th><th>RMS (m/s²)</th><th>Time</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>""")

    # L2 turn-in
    ts = cf.turn_in_summary()
    if ts["n"]:
        side_txt = " · ".join(
            f"{s}: {v['median']:+.2f} s (n={v['n']})" for s, v in ts["by_side"].items()
        )
        parts.append(f"""
<div class="card">
  <h3>Counterfactual turn-in <span class="muted">(your intersection turns, blinker &lt; 20 mph)</span></h3>
  <p>Planned-turn onset vs your steering onset (positive = model later than you):
  median <strong>{_lagfmt(ts['median_lag'])} s</strong> over n={ts['n_lag']} turns{(' — ' + side_txt) if side_txt else ''}.
  <strong>Model never planned the turn: {ts['never']}/{ts['n']}</strong>
  (planned curvature never reached 30% of your peak).</p>
</div>""")

    # L3 unwind
    us = cf.unwind_summary()
    if us:
        rows = " · ".join(f"{s}: {v['mean']:+.2f} s (n={v['n']})" for s, v in us.items())
        parts.append(f"""
<div class="card">
  <h3>Counterfactual unwind <span class="muted">(your sharp turns)</span></h3>
  <p>Planned-curvature unwind (to 50% of its peak) vs your actual unwind,
  positive = model would straighten later: {rows}.</p>
</div>""")

    # X1 braking onset (vision plan), no-lead row first — the red-light measure
    bs = cf.braking_summary()
    if bs["nolead"]["n"] or bs["lead"]["n"] or bs["nolead"]["never"] or bs["lead"]["never"]:
        def _brow(label, b):
            return (f"<tr><td>{label}</td><td><strong>{_lagfmt(b['median'])} s</strong></td>"
                    f"<td>{b['n']}</td><td>{b['never']}</td></tr>")
        mpc = ""
        if bs["mpc_median"] is not None:
            mpc = (f'<p class="muted">Diagnostic: the chill MPC plan would have braked '
                   f'{_lagfmt(bs["mpc_median"])} s vs you (n={bs["mpc_n"]}) — not scored, '
                   f'the MPC is meaningless without a cruise target.</p>')
        parts.append(f"""
<div class="card">
  <h3>Counterfactual braking onset <span class="muted">(vision plan, your stop approaches)</span></h3>
  <p>Planned accel crossing −0.5 m/s² vs yours (positive = model would brake later).
  The <strong>no-lead row is the red-light/stop-sign measure</strong>: nothing but the
  end-to-end model's road reading makes it brake there.</p>
  <table class="mtable"><thead><tr><th></th><th>Median lag</th><th>n</th><th>plan never braked</th></tr></thead>
  <tbody>{_brow("No lead (lights/signs)", bs["nolead"])}{_brow("Behind a lead", bs["lead"])}</tbody></table>
  {mpc}
</div>""")

    # X2 desired-speed agreement
    if cf.accel_rms is not None or cf.speed_opinion:
        srows = ""
        for name, label in (("free", "Free road"), ("lead", "Following a lead")):
            so = cf.speed_opinion.get(name)
            if so:
                direction = "slower" if so["pct"] < 0 else "faster"
                srows += (f"<tr><td>{label}</td><td><strong>{abs(so['pct']):.0f}% {direction}</strong> "
                          f"(ratio {so['median_ratio']:.3f})</td><td>{so['seconds']:.0f}s</td></tr>")
        rms_txt = (f"Planned-vs-realized accel RMS disagreement: "
                   f"<strong>{cf.accel_rms:.2f} m/s²</strong> over {cf.accel_rms_seconds:.0f} s. "
                   if cf.accel_rms is not None else "")
        parts.append(f"""
<div class="card">
  <h3>Desired-speed agreement <span class="muted">(vision plan, your cruising)</span></h3>
  <p>{rms_txt}Model's planned speed 4 s ahead vs the speed you actually drove 4 s later
  ("the model wants to go..."):</p>
  <table class="mtable"><thead><tr><th></th><th>Model wants</th><th>Time</th></tr></thead>
  <tbody>{srows or '<tr><td colspan=3 class="muted">insufficient data</td></tr>'}</tbody></table>
</div>""")

    # X3 launch
    ls = cf.launch_summary()
    if ls["lead"]["n"] or ls["nolead"]["n"]:
        rows = ""
        if ls["lead"]["n"]:
            rows += (f"<tr><td>Lead pull-away</td><td><strong>{_lagfmt(ls['lead']['median'])} s</strong></td>"
                     f"<td>{ls['lead']['n']}</td></tr>")
        if ls["nolead"]["n"]:
            rows += (f"<tr><td>No lead (green light), vs your first motion</td>"
                     f"<td><strong>{_lagfmt(ls['nolead']['median'])} s</strong></td>"
                     f"<td>{ls['nolead']['n']}</td></tr>")
        parts.append(f"""
<div class="card">
  <h3>Counterfactual launch onset <span class="muted">(vision plan)</span></h3>
  <p>Planned accel crossing +0.3 m/s² vs your response (positive = model later):</p>
  <table class="mtable"><thead><tr><th></th><th>Median lag</th><th>n</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>""")

    if not parts:
        return ""
    return f"""
<section>
  <h2>Plan vs You (counterfactual)</h2>
  <div class="warn">Computed from the model's live plan during YOUR driving (it keeps planning
  while disengaged). Longitudinal counterfactuals use the end-to-end (vision) model plan — what
  Experimental mode executes — not the chill MPC. Timing comparisons are robust; magnitudes are
  indicative only — the plan is conditioned on the situation you created. These numbers are NOT
  part of the grades above.</div>
  {''.join(parts)}
  {data_tip}
</section>"""


def _lettered_score(score: float) -> str:
    """Round a score that's displayed NEXT TO A LETTER GRADE, but never show
    '100' unless it truly clears S_CUTOFF -- plain :.0f rounding would show
    '100' for e.g. 99.6 (a real, non-perfect score) right next to a non-S
    letter, which reads as a bug ("it says 100 but isn't S-tier?"). Bare
    numeric scores with no letter alongside (metric rows, ping-pong bins)
    don't have this problem and keep using plain :.0f -- there's no S-badge
    for them to contradict."""
    if score >= S_CUTOFF:
        return "100"
    return str(min(round(score), 99))


def _grade_class(score) -> str:
    if score is None:
        return "gnone"
    if score >= S_CUTOFF:
        return "ggold"
    if score >= 78:
        return "ggood"
    if score >= 60:
        return "gmid"
    return "gbad"


# Plain-language help rendered under every category card: what the numbers
# mean, and how to drive to collect good data for that grade.
CATEGORY_HELP: dict[str, tuple[str, str]] = {
    "Smoothness": (
        "Overall acceleration/braking smoothness throughout a drive. Click any row "
        "below for what it specifically means.",
        "Any driving counts. For a fair grade, drive the same kinds of roads manually "
        "that you let the model drive — a model graded on city streets against your "
        "highway cruising will look worse than it is.",
    ),
    "Following": (
        "How well the model holds a following gap behind a lead car. Click any row "
        "below for what it specifically means.",
        "Best data: 30+ seconds of steady following behind one car above ~18 mph, "
        "without stop-and-go — both engaged and while you drive. Adherence needs the "
        "model actually controlling gas/brake behind a lead.",
    ),
    "Stopping": (
        "How the car brakes down to a complete stop. Click any row below for what it "
        "specifically means.",
        "Complete stops from ~18+ mph. Let the model finish stops without touching the "
        "pedals (a gas/brake tap removes that stop from its data), and make some full "
        "manual stops too — they are the baseline being compared against.",
    ),
    "Launch": (
        "How the car gets moving from a standstill. Click any row below for what it "
        "specifically means.",
        "Standstill-to-rolling starts with no gas pedal help for the model side, and "
        "normal manual launches for your baseline. Launches behind a lead also feed "
        "the pull-away latency metric.",
    ),
    "Responsiveness": (
        "Reaction stopwatches for how quickly the car responds to what the lead car "
        "does. Click either row below for what it specifically means — including an "
        "important caveat about reading your own numbers here.",
        "Needs following situations where the lead visibly brakes or pulls away — "
        "ordinary traffic provides these; more time spent following = better data.",
    ),
    "Ping-Pong": (
        "Steering-wheel oscillation by speed range. Oscillation RMS is the wobble left "
        "over after the intended maneuver is removed (degrees of wheel); reversal rate "
        "is direction flips per minute bigger than 3°. Each speed bin is scored "
        "against your own steering in that same bin; the worst bin is called out.",
        "Hands-off engaged (or AOL) steering at a variety of speeds — touching the "
        "wheel excludes those moments from the model's data. Low-speed creep (parking "
        "lots, drive-thrus) fills the 1–10 mph bins where ping-pong is worst; note "
        "your own low-speed manual maneuvering naturally raises the human baseline "
        "there.",
    ),
    "Turns": (
        "Everything about how turns are carried out: unwind quality after the apex, "
        "how hard the model commits at turn-in, and how far it drifts from its own "
        "plan when you resist it. Click any row below for what it specifically means.",
        "Let the model finish turns on its own when it's safe to — every rescue is "
        "itself a data point. Drive some sharp turns manually to build the comparison "
        "baseline, and actually push back when the model's steering feels wrong: "
        "resisted-divergence rows need real resistance, not just a hand on the wheel.",
    ),
    "General Smoothness": (
        "Overall steering comfort during ordinary driving at speed. Click any row "
        "below for what it specifically means.",
        "Ordinary driving above ~22 mph, engaged and manual, on similar roads for "
        "both.",
    ),
    "Speed Disagreement": (
        "How often you and the model disagree about speed — an override is you "
        "pressing the gas while the model controls longitudinal, and openpilot keeps "
        "driving, so every press is a clean signal. Click any row above for what it "
        "specifically measures. Two things separate from that scored table: brake "
        "disengagements are the opposite direction — you braking hard enough to kick "
        "the model out entirely (shown below, not part of this grade); the context "
        "table splits overrides by situation — launches, plan-was-braking, lead "
        "pulling away, and open road.",
        "This measures your tolerance as much as the model — override only when "
        "you actually want more speed; every override is a labeled data point, so "
        "normal driving is the right input. Compare the personality breakdown: an "
        "aggressive personality should need fewer overrides from you.",
    ),
}


def _speed_disagreement_extras(cat: CategoryResult) -> str:
    sd = cat.extra.get("result") if cat.extra else None
    if sd is None:
        return ""
    parts = []
    rows = "".join(
        f"<tr><td>{_esc(r['context'])}</td><td>{r['n']}</td>"
        f"<td>{_fmt(r['median_magnitude'])}</td></tr>"
        for r in sd.context_table
    )
    if any(r["n"] for r in sd.context_table):
        parts.append(f"""
  <table class="mtable">
    <thead><tr><th>Override context</th><th>Episodes</th><th>Median magnitude (m/s²)</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>""")
    if sd.brake_rate is not None:
        bc = sd.brake_context
        parts.append(
            f'<p class="muted">Brake disengagements (you braking the model out of control): '
            f'{_fmt(sd.brake_rate)}/10 min of model-long time — '
            f'{bc.get("lead_or_stop", 0)} behind a lead or into a stop, '
            f'{bc.get("free_road", 0)} on a free road.</p>'
        )
    return "".join(parts)


_INITIATOR_LABELS = {
    "model": "Model-led",
    "driver": "Driver-led",
    "lag": "Neither led (control lag)",
    "unknown": "Unknown (no cmd source)",
}


def _turn_in_breakdown_table(cat: CategoryResult) -> str:
    """Diagnostic-only texture for Turns: who moved first, and (for
    the episodes that DID have a scored conflict) the median divergence and
    torque-ceiling context. Never affects the score above."""
    bd = cat.extra.get("breakdown") if cat.extra else None
    if not bd:
        return ""
    rows = []
    for key in ("model", "driver", "lag", "unknown"):
        b = bd.get(key)
        if not b or not b["n"]:
            continue
        ceiling = ""
        if b["n_conflict"]:
            ceiling = (
                f'{b["ceiling_true"]} maxed out · {b["ceiling_false"]} not maxed'
                + (f' · {b["ceiling_unknown"]} unknown' if b["ceiling_unknown"] else "")
            )
        rows.append(
            f"<tr><td>{_esc(_INITIATOR_LABELS[key])}</td><td>{b['n']}</td>"
            f"<td>{b['n_conflict']}</td><td>{_fmt(b['median_divergence'], 0)}</td>"
            f"<td>{_esc(ceiling)}</td></tr>"
        )
    if not rows:
        return ""
    return f"""
  <table class="mtable">
    <thead><tr><th>Who moved first</th><th>Episodes</th><th>Resisted</th>
    <th>Median divergence (°)</th><th>Torque during resistance</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <p class="muted">"Who moved first" and "torque during resistance" are context only — the score above
  comes entirely from the divergence column, on episodes where you genuinely resisted.</p>"""


def _category_card(cat: CategoryResult) -> str:
    letter = cat.letter or "–"
    score_txt = _lettered_score(cat.score) if cat.score is not None else "no data"
    if cat.name == "Ping-Pong":
        body = _pingpong_card(cat)
    else:
        row_overrides = None
        if cat.name == "Turns":
            row_overrides = {}
            angles = cat.extra.get("resisted_angles") if cat.extra else None
            if angles:
                for side, vals in angles.items():
                    row_overrides[f"resisted_divergence_{side}"] = vals
            # onset timing is measured relative to when the wheel itself
            # turned in -- that instant IS "you", so it's always exactly
            # zero by construction, not a real aggregate to compute.
            for side in ("left", "right"):
                row_overrides[f"cmd_onset_lead_{side}"] = {
                    "you_txt": '0.00 <span class="muted">(reference — when the wheel actually turned in)</span>'
                }
        body = f"""
  <table class="mtable">
    <thead><tr><th>Metric</th><th>Model</th><th>You</th><th>Unit</th><th>Score</th></tr></thead>
    <tbody>{_metric_rows(cat.metrics, row_overrides)}</tbody>
  </table>"""
        tf = cat.extra.get("t_follow_targets") if cat.extra else None
        if tf:
            tgt = ", ".join(f"{p} {tf[p]:.2f} s" for p in ("aggressive", "standard", "relaxed") if p in tf)
            body += (
                f'<p class="muted">Follow-adherence rows compare the model\'s held gap to the '
                f'ACTIVE personality\'s target (the "You" column is the target, not the human). '
                f'Targets used: {_esc(tgt)} — these are fork-dependent; set yours with '
                f'--t-follow or the UI.</p>'
            )
        if cat.name == "Speed Disagreement":
            body += _speed_disagreement_extras(cat)
        if cat.name == "Turns":
            body += _turn_in_breakdown_table(cat)
    help_html = ""
    if cat.name in CATEGORY_HELP:
        what, data = CATEGORY_HELP[cat.name]
        help_html = (
            f'<p class="muted cathelp"><b>What these numbers mean:</b> {_esc(what)}</p>'
            f'<p class="muted cathelp"><b>Getting good data for this grade:</b> {_esc(data)}</p>'
        )
    return f"""
<div class="card {_grade_class(cat.score)}">
  <div class="cathead"><h3>{_esc(cat.name)}</h3>
    <div class="catgrade"><span class="letter">{_esc(letter)}</span><span class="score">{_esc(score_txt)}</span></div>
  </div>
  <div class="catweight">weight {cat.weight:.2f} within group</div>
  {body}
  {help_html}
</div>"""


def _fmt_bucket_times(bucket_times: dict, names: tuple) -> str:
    parts = [f"{n} {bucket_times[n] / 60:.1f} min" for n in names if n in bucket_times]
    return ", ".join(parts)


def _header_facts(drives, buckets, analysis=None) -> list[tuple[str, str]]:
    routes = [d.name for d in drives]
    fingerprints = sorted({d.meta.car_fingerprint for d in drives})
    versions = sorted({d.meta.version for d in drives if d.meta.version})
    persons = sorted({d.meta.personality for d in drives if d.meta.personality})
    dates = [d.meta.wall_time_start for d in drives if d.meta.wall_time_start]
    facts = [
        ("Routes", ", ".join(routes)),
        ("Car", ", ".join(fingerprints)),
        ("openpilot", ", ".join(versions) or "unknown"),
        ("Model, Lat & Long", f"{buckets['both'] / 60:.1f} min"),
        ("Lat-only (AOL/MADS)", f"{buckets['lat_only'] / 60:.1f} min"),
        ("Long-only", f"{buckets['long_only'] / 60:.1f} min"),
        ("Fully manual", f"{buckets['manual'] / 60:.1f} min"),
        ("Personality", ", ".join(persons) or "n/a (not in these logs)"),
    ]
    if dates:
        t0, t1 = min(dates), max(dates)
        fmt = lambda ts: time.strftime("%Y-%m-%d", time.gmtime(ts))
        facts.insert(1, ("Dates", fmt(t0) if fmt(t0) == fmt(t1) else f"{fmt(t0)} – {fmt(t1)}"))
    if analysis is not None:
        bt = getattr(analysis, "bucket_times", {}) or {}
        modes = _fmt_bucket_times(bt, ("chill", "experimental"))
        if modes:
            facts.append(("Time in mode (model long)", modes))
        pers = _fmt_bucket_times(bt, ("aggressive", "standard", "relaxed"))
        if pers:
            facts.append(("Time in personality (model long)", pers))
        mid = getattr(analysis, "model_id", None)
        if mid:
            label = mid["label"]
            if mid.get("provenance") not in (None, "unknown"):
                label += f" — {mid['provenance']}"
            facts.append(("Driving model", label))
        psum = getattr(analysis, "profile_summary", None)
        if psum is not None:
            facts.append(("Driver profile", "; ".join(psum.lines())))
    if any(d.meta.experimental_mode for d in drives) and (
        analysis is None or "experimental" not in (getattr(analysis, "bucket_times", {}) or {})
    ):
        facts.append(("Experimental mode", "seen enabled during these drives"))
    return facts


def _group_hero(grades: GradeReport) -> str:
    cells = []
    for g in grades.groups:
        letter = g.letter or "–"
        score = _lettered_score(g.score) if g.score is not None else "no data"
        cells.append(f"""
  <div class="ghero {_grade_class(g.score)}">
    <div class="gletter">{_esc(letter)}</div>
    <div class="gname">{_esc(g.name)}</div>
    <div class="gscore">{_esc(score)}</div>
  </div>""")
    overall = (
        f"Overall {grades.overall_letter} ({_lettered_score(grades.overall_score)}/100)"
        if grades.overall_score is not None
        else "Overall: not enough comparable data"
    )
    return f"""
<section class="hero">
  <div class="herotop">
    <div class="herotitle">Model vs You</div>
    <div class="overall {_grade_class(grades.overall_score)}">{_esc(overall)}</div>
  </div>
  <div class="gheroes">{''.join(cells)}</div>
  <div class="herohint">100 = matches you (or smoother) · 50 = twice your numbers · style metrics count deviation from you in either direction · human = ground truth</div>
</section>"""


# --------------------------------------------------------------------- main


def render_report(analysis, out_path: str | Path) -> Path:
    per_drive = analysis.per_drive
    samples = analysis.samples
    grades = analysis.grades
    drives = [d for d, _s, _a, _e in per_drive]
    buckets = {"both": 0.0, "lat_only": 0.0, "long_only": 0.0, "manual": 0.0}
    for _d, s, _a, _e in per_drive:
        for k, v in s.bucket_times().items():
            buckets[k] += v
    t_eng = buckets["both"] + buckets["long_only"]  # model longitudinal time
    t_man = buckets["manual"] + buckets["lat_only"]  # human longitudinal time

    # ---- warnings
    warnings = []
    mismatch_warning = getattr(analysis, "mismatch_warning", None)
    if mismatch_warning:
        # --allow-mixed overrode a real vehicle/model mismatch: this must be
        # the FIRST warning and unmistakable, not just another bullet.
        warnings.append(
            f"MIXED VEHICLES/MODELS (--allow-mixed was used) — {mismatch_warning} "
            "Results below may blend behavior from different cars or driving "
            "models as if it were one and may not be meaningful."
        )
    if all(d.meta.openpilot_long is False for d in drives):
        warnings.append(
            "openpilotLongitudinalControl is FALSE in these logs: while engaged, "
            "longitudinal behavior is the stock car's ACC, not the openpilot model. "
            "Longitudinal grades compare stock ACC against you."
        )
    if t_man < 60:
        warnings.append(
            f"Only {t_man:.0f} s of human longitudinal driving found — the human baseline is thin; "
            "most metrics will be 'insufficient data'. Include logs with more manual driving."
        )
    if t_eng < 60:
        warnings.append(f"Only {t_eng:.0f} s of model longitudinal driving found — the model side is thin.")
    if all(not d.meta.vm_params for d in drives):
        warnings.append(
            "carParams lacks vehicle-model fields; commanded-steering-angle metrics were skipped."
        )
    missing = sorted({m for d in drives for m in d.missing})
    if missing:
        warnings.append(
            "Channels absent in these logs (metrics needing them were skipped): " + ", ".join(missing)
        )
    notes = sorted({n for d in drives for n in d.meta.notes})

    # ---- events payload
    events_by_kind: dict[str, list] = {k: [] for k in KIND_LABELS}
    for d, seg, da, evs in per_drive:
        t0_drive = float(seg.t[0]) if len(seg.t) else 0.0
        for ev in sorted(evs, key=lambda e: e.t0):
            if ev.kind in events_by_kind and len(events_by_kind[ev.kind]) < MAX_EVENTS_PER_KIND:
                events_by_kind[ev.kind].append(_event_payload(ev, da, t0_drive))
    payload_json = json.dumps({"events": events_by_kind}, separators=(",", ":"), allow_nan=False)

    uplot_js = (_ASSETS / "uPlot.iife.min.js").read_text(encoding="utf-8")
    uplot_css = (_ASSETS / "uPlot.min.css").read_text(encoding="utf-8")

    group_sections = []
    for g in grades.groups:
        cards = "".join(_category_card(c) for c in g.categories)
        group_sections.append(f"""
<section>
  <h2>{_esc(g.name)} <span class="muted">{_esc(g.letter or 'no data')}{f" · {_lettered_score(g.score)}" if g.score is not None else ''}</span></h2>
  <div class="cards">{cards}</div>
</section>""")

    hists = []
    for key, title, unit in (
        ("rms_jerk", "RMS jerk per span", "m/s³"),
        ("median_gap", "Median time gap per follow window", "s"),
        ("stop_lurch", "Stop lurch per stop", "m/s³"),
    ):
        s = samples.get(key, {"model": [], "driver": []})
        svg = _svg_hist(s["model"], s["driver"], title, unit)
        if svg:
            hists.append(f"""
<div class="histbox">
  <h3>{_esc(title)}</h3>
  <div class="legend"><span class="swatch sm"></span>Model (n={len(s['model'])})
    <span class="swatch sd"></span>You (n={len(s['driver'])})</div>
  {svg}
</div>""")

    ev_sections = []
    for kind, label in KIND_LABELS.items():
        evs = events_by_kind[kind]
        if not evs:
            continue
        _hk, hunit = KIND_HEADLINE[kind]
        rows = []
        for i, ev in enumerate(evs):
            side = "model" if ev["engaged"] else "you"
            badge = ' <span class="badge">override</span>' if ev["override"] else ""
            cens = "≥" if ev["censored"] else ""
            head = f"{cens}{_fmt(ev['headline'])}" if ev["headline"] is not None else "–"
            tag = f' <span class="muted">{_esc(ev["tag"])}</span>' if ev.get("tag") else ""
            rows.append(
                f'<tr class="evrow" data-kind="{kind}" data-idx="{i}">'
                f"<td>{_esc(ev['drive'])} +{ev['t0']:.0f}s{tag}</td>"
                f'<td><span class="side {side}">{"model" if ev["engaged"] else "you"}</span>{badge}</td>'
                f"<td>{ev['dur']:.1f}s</td><td>{head}</td></tr>"
            )
        ev_sections.append(f"""
<details class="evkind" open>
  <summary><h3>{_esc(label)} <span class="count">({len(evs)})</span></h3></summary>
  <table class="etable">
    <thead><tr><th>When</th><th>Who</th><th>Duration</th><th>{_esc(hunit)}</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <div class="drill" id="drill-{kind}"></div>
</details>""")

    warn_html = "".join(f'<div class="warn">⚠ {_esc(w)}</div>' for w in warnings)
    notes_html = (
        '<div class="notes">' + " · ".join(_esc(n) for n in notes) + "</div>" if notes else ""
    )
    facts_html = "".join(
        f'<div class="fact"><span class="fk">{_esc(k)}</span><span class="fv">{_esc(v)}</span></div>'
        for k, v in _header_facts(drives, buckets, analysis)
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>op-model-grader report</title>
<style>{uplot_css}</style>
<style>{_CSS}</style>
</head><body>
<div class="wrap">
<header>
  <h1>op-model-grader</h1>
  <p class="sub">Your openpilot model, graded against your own manual driving.</p>
  {warn_html}
  <div class="facts">{facts_html}</div>
  {notes_html}
</header>

{_group_hero(grades)}

{''.join(group_sections)}

{_breakdown_tables(grades.breakdowns)}

{_counterfactual_section(getattr(analysis, "counterfactual", None))}

<section>
  <h2>Distributions</h2>
  <div class="hists">{''.join(hists) or '<p class="muted">Not enough events for distributions.</p>'}</div>
</section>

<section>
  <h2>Events <span class="muted">(click a row for traces)</span></h2>
  {''.join(ev_sections) or '<p class="muted">No events detected.</p>'}
</section>

<footer>
  <p><strong>Grading.</strong> For each ratio metric, model aggregate m vs your aggregate d:
  r = m / max(d, ε) (inverted for higher-is-better); score = 100 if r ≤ 1; 100 − 50·(r−1) for r ≤ 2;
  else max(0, 50 − 25·(r−2)). Metrics marked * use an absolute anchor scale (documented in their row's
  category) because there is no human counterpart or your baseline is ~zero.
  Letters: ≥93 A, ≥85 A−, ≥78 B+, ≥70 B, ≥60 C, ≥50 D, else F. Ratio metrics need ≥3 events per side.
  Longitudinal weights: Smoothness 0.25, Following 0.18, Stopping 0.17, Launch 0.15, Responsiveness 0.10, Speed Disagreement 0.15.
  Lateral weights: Ping-Pong 0.40, Turns 0.50, General Smoothness 0.10.
  Overall = ½ Longitudinal + ½ Lateral; empty categories/groups are dropped and weights renormalized.
  A local <strong>driver profile</strong> (~/.local/share/opgrader/profile.json by default) accumulates
  your manual-driving samples across every route you grade and blends them into "You" values above —
  the model side is never pooled, only your own driving, so a metric's combined n (shown in its "You"
  cell when a profile contributed) is what gates the ≥3-events rule. Skip it for one run with
  --no-profile; inspect or delete it with --clear-profile.</p>
  <p><strong>Longitudinal definitions.</strong> Jerk = d(aEgo)/dt after a centered 0.3 s moving average.
  Stops: from last vEgo ≥ 8 m/s to standstill (&lt; 0.3 m/s for ≥ 0.5 s, ≤ 45 s).
  Launch: standstill → 5 m/s (must reach 3 m/s within 10 s of first motion).
  Follow: lead &lt; 80 m, vEgo &gt; 8 m/s, ≥ 15 s. Lead decel stimulus: aLeadK &lt; −1.2 m/s²
  for ≥ 0.4 s; response = own smoothed accel drops 0.3 m/s² (≤ 4 s, censored).
  Pull-away: lead &lt; 12 m from standstill, vLead &gt; 0.5 m/s; go = vEgo &gt; 0.15 or aEgo &gt; 0.2 (≤ 6 s).
  Engaged moments with gas/brake override are excluded from the model side.
  <strong>Speed Disagreement</strong>: gas-override episode = gasPressed while the model controls
  longitudinal (gaps &lt; 1 s merged, ≥ 0.3 s). openpilot drops longActive during a gas override while
  staying enabled, so the control mask used is longActive OR (enabled AND gasPressed); rates are per
  time under that mask. Magnitude = median (aEgo − vision-plan accel) over the episode; brake
  disengagement = the model losing longitudinal control with brakePressed within ±0.5 s. Scored on
  absolute anchors (rate 0/4/8 per 10 min → 100/50/0; time 0/10/25%; magnitude 0.2/1.0/2.0 m/s²) —
  there is no human baseline for overriding yourself.</p>
  <p><strong>Mode &amp; personality.</strong> Experimental/chill mode and the longitudinal
  personality are tracked per sample (they change mid-drive). Events keep their tag only when
  ≥90% of the window agrees; mixed events count toward the overall grades but are excluded
  from the breakdown buckets. Breakdown buckets are scored against the same human baseline
  with the usual ≥3-events gate. <strong>Follow adherence</strong> inverts the openpilot long-MPC
  distance: t_follow = (dRel − 6.0 − (vEgo² − vLead²)/5.0) / vEgo over steady-follow samples
  (model long active, vEgo &gt; 8, lead present, |vRel| &lt; 1.5 m/s, |aEgo| &lt; 0.5 m/s²); the median per ACTIVE personality is compared
  to that personality's target and scored absolutely (100 at ≤5% error, 50 at 25%, 0 at ≥50%).
  Targets are fork-dependent.</p>
  <p><strong>Per-axis attribution (AOL/MADS aware).</strong> Longitudinal metrics attribute by
  carControl.longActive, lateral metrics by carControl.latActive — so Always-On-Lateral time counts
  as the model for steering and as you for gas/brake. Events whose window has mixed control
  (&gt;10% of samples disagreeing) are discarded rather than mis-attributed. Old logs without the
  per-axis flags fall back to the single enabled flag (noted above).</p>
  <p><strong>Lateral definitions</strong> (matching the on-device analyzer): turn onset at |angle| ≥ 20°
  (actual or commanded, whichever first); unwind point = first fall to ≤ 50% of that signal's peak;
  sharp turn = peak ≥ 90° with onset speed &lt; 15 mph; positive angle = left (ISO). Commanded angle =
  VehicleModel(carParams).get_steer_from_curvature(−actuators.curvature, vEgo). Ping-pong = high-passed
  steering angle (minus centered 2 s mean), per speed bin, standstill and steeringPressed excluded.
  <strong>Peak turn-in effort</strong> (Turns, shown not scored) is the highest
  |controlsState.lateralControlState.torqueState.output| reached from turn onset up to the instant you
  first touched the wheel (steeringPressed), or across the whole episode if you never did — deliberately
  stops there because torque_output changes once you take over (it may be fighting your input rather
  than executing its own plan at that point), so looking past that instant would misread a correction as
  an admission the model wasn't trying.
  <strong>Turns is blinker-free</strong>: every engaged turn is scored regardless of band, signaled or
  not. <strong>Cmd-onset timing</strong> is the model's own
  commanded-angle 20° crossing minus the actual angle's 20° crossing — positive means the model's own
  plan called for the turn AFTER the wheel had already gotten there, negative means it called for the
  turn first. That crossing instant on the actual (wheel) side is the reference point the lead/lag is
  measured against, so the You column always reads "0.00" by construction rather than a real aggregate —
  the Model column is what actually varies. Scored on absolute anchors 0.0/0.5/1.5s → 100/50/0.
  <strong>Resisted cmd-vs-actual divergence</strong> replaces a torque-ceiling
  gate: a conflict window opens where steeringPressed is true AND your steering torque opposes the
  model's own commanded torque (torqueState.output; falls back to opposing actual-vs-commanded angle
  signs when either torque channel is unavailable), sustained ≥ 0.3 s to filter out an incidental hand on
  the wheel. The metric is the peak |actual − commanded| angle during that window; episodes with no such
  window score nothing at all, not a zero. The Model/You columns show the median |commanded| and
  |actual| angle at that same peak-disagreement instant, per side, across all qualifying episodes — two
  independently-aggregated numbers (medians), so they won't exactly arithmetically match the median score
  above them, but together they tell you whether the model typically wanted a sharper or softer turn than
  you were willing to give it. Scored on absolute anchors 15°/75°/300° → 100/50/0, set from
  the real divergence distribution across two Palisade routes (35 conflict episodes out of 46 engaged
  turns: median ≈ 79°, p90 ≈ 337°, max 823° on one genuine multi-second tug-of-war) so the scale
  distinguishes a little push-back from a full fight rather than being picked in a vacuum. Whether the
  model was already at its torque ceiling during the resistance, and who moved first (model/driver/lag),
  are shown as unscored context in the category's breakdown table — never a scoring gate. Note:
  controlsState.lateralControlState.torqueState.saturated was False for every sample on both real
  routes tested, so ceiling detection is done by thresholding |torque_output| ourselves (CEILING_FRAC =
  0.92) rather than trusting that flag. <strong>Turn intents</strong> (a separate, unscored diagnostic
  elsewhere in this report, NOT used for Turns' grade): blinker on below 20 mph opens a window
  (blinker-on + 20 s or blinker-off + 5 s); |net heading| ≥ 45° = intersection turn, &lt; 20° = lane
  change, else ambiguous — this population still feeds the separate Plan-vs-You counterfactual turn-in
  section, whose definition of "missed"/"never planned" is blinker-gated and independent of the metric
  above.</p>
  <p class="muted">op-model-grader v{_esc(__version__)} · charts by <a href="https://github.com/leeoniya/uPlot">uPlot</a> (MIT, vendored)</p>
</footer>
</div>
<script>{uplot_js}</script>
<script>
const DATA = {payload_json};
{_JS}
</script>
</body></html>"""

    out_path = Path(out_path)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


_CSS = """
:root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --model:#2a78d6; --driver:#1baf7a;
  --good:#0ca30c; --warnc:#fab219; --bad:#d03b3b; --gold:#a8790a;
}
@media (prefers-color-scheme: dark){
  :root{
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --model:#3987e5; --driver:#199e70; --gold:#f0c030;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:1.7rem;margin:0}
h2{font-size:1.2rem;margin:36px 0 12px;border-bottom:1px solid var(--grid);padding-bottom:6px}
h3{font-size:1rem;margin:0;display:inline}
a{color:var(--model)}
.sub{color:var(--ink2);margin:.2em 0 1em}
.muted{color:var(--muted);font-weight:normal;font-size:.85em}
.cathelp{margin:8px 4px 2px;line-height:1.45}
.warn{background:color-mix(in srgb, var(--warnc) 14%, var(--surface));border:1px solid var(--warnc);
  border-radius:8px;padding:10px 14px;margin:10px 0;font-size:.95em}
.notes{color:var(--muted);font-size:.85em;margin-top:8px}
.facts{display:flex;flex-wrap:wrap;gap:8px 24px;margin-top:10px}
.fact .fk{color:var(--muted);font-size:.8em;display:block}
.fact .fv{font-size:.95em}
.hero{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:20px 28px;margin-top:24px}
.herotop{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
.herotitle{font-size:1.25rem;font-weight:600}
.overall{font-size:1rem;color:var(--ink2)}
.gheroes{display:flex;gap:20px;margin-top:14px;flex-wrap:wrap}
.ghero{flex:1;min-width:200px;text-align:center;border:1px solid var(--border);
  border-radius:12px;padding:14px}
.ghero.ggold{border-color:var(--gold);box-shadow:0 0 0 1px var(--gold), 0 0 16px -4px var(--gold)}
.gletter{font-size:3.6rem;font-weight:700;line-height:1.1}
.ggood .gletter{color:var(--good)} .gmid .gletter{color:var(--warnc)}
.gbad .gletter{color:var(--bad)} .gnone .gletter{color:var(--muted)}
.ggold .gletter{color:var(--gold)}
.gname{font-weight:600;margin-top:4px}
.gscore{color:var(--muted);font-size:.9em}
.herohint{color:var(--muted);font-size:.85em;margin-top:12px}
.callout{background:color-mix(in srgb, var(--model) 10%, var(--surface));
  border:1px solid var(--model);border-radius:8px;padding:8px 12px;margin:8px 0;font-size:.9em}
/* CSS multi-column, not grid: grid rows stretch every item in a row to
   match its tallest neighbor (Turns, after merging two categories into
   one, towers over Ping-Pong/General Smoothness and leaves dead space
   inside their shorter cards). Columns pack cards by actual height instead,
   same responsive column-count-by-width behavior as the old grid. */
.cards{column-width:340px;column-gap:14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;
  overflow-x:auto;break-inside:avoid;margin:0 0 14px}
.card.ggold{border-color:var(--gold)}
.cathead{display:flex;justify-content:space-between;align-items:baseline}
.catgrade .letter{font-size:1.5rem;font-weight:700;margin-right:8px}
.ggood .catgrade .letter{color:var(--good)} .gmid .catgrade .letter{color:var(--warnc)}
.gbad .catgrade .letter{color:var(--bad)} .gnone .catgrade .letter{color:var(--muted)}
.ggold .catgrade .letter{color:var(--gold)}
.catgrade .score{color:var(--muted);font-size:.9em}
.catweight{color:var(--muted);font-size:.8em;margin:2px 0 8px}
table{border-collapse:collapse;width:100%;font-size:.86em}
th{text-align:left;color:var(--muted);font-weight:500;border-bottom:1px solid var(--grid);padding:4px 8px 4px 0}
td{padding:4px 8px 4px 0;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tr.insuff td{color:var(--muted)}
tr.bdcat td{color:var(--ink2);font-weight:600;border-top:2px solid var(--grid)}
td.bdmetric{padding-left:14px}
tr.hasdesc{cursor:pointer}
tr.hasdesc:hover td{background:color-mix(in srgb, var(--model) 6%, transparent)}
tr.hasdesc.open{background:color-mix(in srgb, var(--model) 4%, transparent)}
.rowinfo{display:inline-block;margin-left:6px;color:var(--muted);font-size:.75em;transition:transform .15s}
tr.hasdesc.open .rowinfo{transform:rotate(90deg)}
tr.descrow{display:none}
tr.descrow.open{display:table-row}
tr.descrow td{color:var(--ink2);font-size:.92em;line-height:1.5;padding:2px 8px 12px 0}
/* a hasdesc row followed by its own (possibly hidden) descrow is never
   really the table's last row visually, even when descrow is last-child */
tr.hasdesc:has(+ tr.descrow:last-child) td{border-bottom:none}
.hists{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.histbox{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.legend{color:var(--ink2);font-size:.85em;margin:6px 0}
.swatch{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 6px 0 14px;vertical-align:baseline}
.swatch:first-child{margin-left:0}
.swatch.sm{background:var(--model)} .swatch.sd{background:var(--driver)}
svg.hist{width:100%;height:auto;display:block}
svg.hist .hm{fill:var(--model)} svg.hist .hd{fill:var(--driver)}
svg.hist .hax{stroke:var(--axis);stroke-width:1}
svg.hist .hlab{fill:var(--muted);font-size:11px}
.evkind{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:10px 16px;margin:12px 0}
.evkind summary{cursor:pointer;padding:4px 0}
.evkind .count{color:var(--muted);font-weight:normal}
.etable{margin-top:8px}
tr.evrow{cursor:pointer}
tr.evrow:hover td{background:color-mix(in srgb, var(--model) 8%, transparent)}
tr.evrow.sel td{background:color-mix(in srgb, var(--model) 16%, transparent)}
.side{font-size:.8em;padding:1px 8px;border-radius:9px;border:1px solid var(--border)}
.side.model{color:var(--model);border-color:var(--model)}
.side.you{color:var(--driver);border-color:var(--driver)}
.badge{font-size:.75em;color:var(--warnc);border:1px solid var(--warnc);border-radius:9px;padding:0 6px;margin-left:6px}
.drill{margin-top:10px}
.drill .chart{margin-bottom:6px}
.drill .charttitle{color:var(--ink2);font-size:.82em;margin:8px 0 2px}
.u-legend{display:none}
footer{margin-top:48px;color:var(--ink2);font-size:.85em;border-top:1px solid var(--grid);padding-top:16px}
"""

_JS = """
function css(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim();}
const SYNC = uPlot.sync("drill");
let charts = [];

function mkChart(el, title, t, ys, color, unit){
  const div = document.createElement('div');
  div.className = 'chart';
  const cap = document.createElement('div');
  cap.className = 'charttitle';
  const baseTitle = title + (unit ? ' (' + unit + ')' : '');
  cap.textContent = baseTitle;
  el.appendChild(cap); el.appendChild(div);
  const w = Math.min(el.clientWidth || 900, 980);
  const opts = {
    width: w, height: 110,
    cursor: {sync: {key: SYNC.key}, points: {size: 7}},
    scales: {x: {time: false}},
    axes: [
      {stroke: css('--muted'), grid: {stroke: css('--grid'), width: 1}, ticks: {stroke: css('--axis')}},
      {stroke: css('--muted'), grid: {stroke: css('--grid'), width: 1}, ticks: {stroke: css('--axis')}, size: 52},
    ],
    series: [
      {label: 't'},
      {label: title, stroke: color, width: 2, points: {show: false}},
    ],
    legend: {show: false},
    hooks: {
      // live readout: hovering any (cursor-synced) chart shows each chart's
      // value at that time in its title line
      setCursor: [u => {
        const i = u.cursor.idx;
        if (i == null || i < 0 || u.data[0][i] == null) { cap.textContent = baseTitle; return; }
        const v = u.data[1][i];
        cap.innerHTML = baseTitle + ' — <b>' +
          (v == null ? '–' : (+v).toFixed(2)) + (unit ? ' ' + unit : '') +
          '</b> @ ' + (+u.data[0][i]).toFixed(1) + ' s';
      }],
    },
  };
  charts.push(new uPlot(opts, [t, ys], div));
}

function showDrill(kind, idx, row){
  const host = document.getElementById('drill-' + kind);
  charts.forEach(c => c.destroy()); charts = [];
  document.querySelectorAll('.drill').forEach(d => d.innerHTML = '');
  document.querySelectorAll('tr.evrow.sel').forEach(r => r.classList.remove('sel'));
  row.classList.add('sel');
  const ev = DATA.events[kind][idx];
  const t = ev.series[0].data;
  const color = ev.engaged ? css('--model') : css('--driver');
  for (let i = 1; i < ev.series.length; i++) {
    const s = ev.series[i];
    mkChart(host, s.label, t, s.data, color, s.unit || '');
  }
  host.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

document.querySelectorAll('tr.evrow').forEach(row => {
  row.addEventListener('click', () => showDrill(row.dataset.kind, +row.dataset.idx, row));
});

document.querySelectorAll('tr.hasdesc').forEach(row => {
  row.addEventListener('click', () => {
    row.classList.toggle('open');
    const d = document.getElementById(row.dataset.descId);
    if (d) d.classList.toggle('open');
  });
});
"""
