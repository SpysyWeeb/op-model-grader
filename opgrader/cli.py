"""Command-line interface: opgrader LOG_PATH... [--route ...] -o report.html"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from collections import Counter
from pathlib import Path

from . import __version__
from .download import DownloadError, download_route
from .events import build_arrays, detect_events
from .extract import extract_drive
from .logreader import find_segments, group_segments, route_name_for_group
from .pipeline import analyze
from .report import render_report
from .segments import segment_drive


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="opgrader",
        description="Grade your openpilot model's driving against your own "
        "manual driving, from rlogs.",
    )
    p.add_argument(
        "logs",
        nargs="*",
        help="rlog files, directories (scanned recursively), or globs",
    )
    p.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="DONGLEID|ROUTENAME",
        help="download a route from comma connect (repeatable)",
    )
    p.add_argument("--jwt", help="comma API JWT (default: ~/.comma/auth.json)")
    p.add_argument(
        "--t-follow",
        metavar="P=SECONDS,...",
        help="fork follow targets, e.g. aggressive=1.0,standard=1.45,relaxed=2.0 "
        "(default: ~/.config/opgrader/config.json, else stock openpilot)",
    )
    p.add_argument("-o", "--out", default="report.html", help="output HTML path")
    p.add_argument("--open", action="store_true", help="open the report in a browser")
    p.add_argument(
        "--ui",
        action="store_true",
        help="open the simple desktop UI (browse drives, request uploads, grade)",
    )
    p.add_argument(
        "--no-profile",
        action="store_true",
        help="don't load or save the local driver-baseline profile for this run "
        "(grade in isolation; the profile file is left untouched)",
    )
    p.add_argument(
        "--clear-profile",
        action="store_true",
        help="delete the local driver-baseline profile (standalone action -- "
        "does not grade anything)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt for --clear-profile",
    )
    p.add_argument("--version", action="version", version=f"opgrader {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.clear_profile:
        from . import profile as P

        return P.clear_profile_cli(yes=args.yes)

    from .config import resolve_t_follow

    try:
        t_follow = resolve_t_follow(args.t_follow)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.ui:
        if args.logs or args.route:
            print(
                "warning: --ui ignores log paths and --route; "
                "use the window to pick drives",
                file=sys.stderr,
            )
        from .gui import run

        run()
        return 0

    if not args.logs and not args.route:
        build_parser().print_help()
        return 2

    inputs = list(args.logs)
    for route in args.route:
        print(f"fetching route {route} from comma connect...")
        try:
            paths = download_route(route, args.jwt)
        except DownloadError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"  {len(paths)} segments ready")
        inputs.extend(str(p) for p in paths)

    seg_files = find_segments(inputs)
    if not seg_files:
        print("error: no rlog segments found in the given paths", file=sys.stderr)
        return 1
    groups = group_segments(seg_files)
    print(f"{len(seg_files)} segment file(s) in {len(groups)} route(s)")

    per_drive = []
    ev_counter: Counter = Counter()
    for g in groups:
        name = route_name_for_group(g)
        print(f"route {name}: decoding {len(g)} segment(s)...")
        drive = extract_drive(name, g)
        seg = segment_drive(drive)
        if seg is None:
            print(f"  no carState data; skipping {name}")
            continue
        da = build_arrays(drive, seg)
        events = detect_events(drive, seg, da)
        b = seg.bucket_times()
        print(
            f"  control: both-model {b['both'] / 60:.1f} min, "
            f"lat-only {b['lat_only'] / 60:.1f}, long-only {b['long_only'] / 60:.1f}, "
            f"manual {b['manual'] / 60:.1f}"
            + ("" if seg.per_axis else "  [no per-axis flags; using enabled]")
        )
        per_drive.append((drive, seg, da, events))

    if not per_drive:
        print("error: no usable drives decoded", file=sys.stderr)
        return 1

    analysis = analyze(per_drive, t_follow_targets=t_follow, use_profile=not args.no_profile)
    grades = analysis.grades

    for _d, _s, _a, events in per_drive:
        for ev in events:
            ev_counter[(ev.kind, "engaged" if ev.engaged else "manual")] += 1
    print("events found (kind: engaged/manual):")
    for kind in sorted({k for k, _s in ev_counter}):
        print(
            f"  {kind}: {ev_counter.get((kind, 'engaged'), 0)}/"
            f"{ev_counter.get((kind, 'manual'), 0)}"
        )

    out = render_report(analysis, args.out)
    if analysis.profile_summary is not None:
        for line in analysis.profile_summary.lines():
            print(f"driver profile: {line}")
    if analysis.model_id and analysis.model_id.get("provenance") != "unknown":
        print(f"driving model: {analysis.model_id['label']} ({analysis.model_id['provenance']})")
    if analysis.bucket_times:
        bt = analysis.bucket_times
        chips = ", ".join(f"{k} {v / 60:.1f}m" for k, v in bt.items())
        print(f"model-long time by bucket: {chips}")
    for p, info in sorted(analysis.adherence.items()):
        tgt = analysis.t_follow_targets.get(p)
        if tgt and info["seconds"] >= 1.0:
            pct = abs(info["median_eff"] - tgt) / tgt * 100
            print(
                f"follow adherence ({p}): holds {info['median_eff']:.2f}s vs "
                f"{tgt:.2f}s target ({pct:.0f}% off, {info['seconds']:.0f}s of data)"
            )
    sd = analysis.speed_disagreement
    if sd is not None and sd.overall_rate is not None:
        mag = f"{sd.overall_magnitude:.2f} m/s²" if sd.overall_magnitude is not None else "n/a"
        pct = f"{sd.overall_pct:.1f}%" if sd.overall_pct is not None else "n/a"
        print(
            f"speed disagreement: {sd.overall_rate:.1f} gas overrides/10min "
            f"({pct} of model-long time), magnitude {mag}"
        )
        top = sd.biggest_context
        if top:
            n = next(r["n"] for r in sd.context_table if r["context"] == top)
            print(f"  biggest context: {top} ({n} of {len(sd.episodes)} episodes)")
    cf = analysis.counterfactual
    if cf is not None and cf.available:
        ts = cf.turn_in_summary()
        if ts["n"]:
            lag = f"{ts['median_lag']:+.2f}s" if ts["median_lag"] is not None else "n/a"
            print(f"plan-vs-you turn-in: median lag {lag} (n={ts['n_lag']}), "
                  f"never planned {ts['never']}/{ts['n']}")
        bs = cf.braking_summary()
        for name, label in (("nolead", "NO LEAD (red lights/signs)"), ("lead", "behind a lead")):
            b = bs[name]
            if b["n"] or b["never"]:
                med = f"{b['median']:+.2f}s" if b["median"] is not None else "n/a"
                print(f"plan-vs-you braking onset {label}: median lag {med} "
                      f"(n={b['n']}, plan-never-braked {b['never']})")
        if bs["mpc_median"] is not None:
            print(f"  (diagnostic: chill MPC would brake {bs['mpc_median']:+.2f}s, n={bs['mpc_n']})")
        ls = cf.launch_summary()
        for name, label in (("lead", "lead pull-away"), ("nolead", "no-lead green light")):
            if ls[name]["n"]:
                print(f"plan-vs-you launch onset ({label}): median lag "
                      f"{ls[name]['median']:+.2f}s (n={ls[name]['n']})")
        for name, label in (("free", "free road"), ("lead", "following")):
            so = cf.speed_opinion.get(name)
            if so:
                d = "slower" if so["pct"] < 0 else "faster"
                print(f"plan-vs-you speed opinion ({label}): model wants "
                      f"{abs(so['pct']):.0f}% {d} than you drive ({so['seconds']:.0f}s)")
        if cf.accel_rms is not None:
            print(f"plan-vs-you accel agreement: RMS {cf.accel_rms:.2f} m/s² "
                  f"over {cf.accel_rms_seconds:.0f}s of your cruising")
        us = cf.unwind_summary()
        if us:
            print("plan-vs-you unwind: " + ", ".join(
                f"{s} {v['mean']:+.2f}s (n={v['n']})" for s, v in us.items()))
        if cf.path_overall is not None:
            print(f"plan-vs-you path agreement: RMS {cf.path_overall:.2f} m/s² "
                  f"over {cf.path_seconds:.0f}s of your steering")
    for dim in ("mode", "personality"):
        bg = grades.breakdowns.get(dim) or {}
        scored = {b: g for b, g in bg.items() if g.score is not None}
        if scored:
            print(
                f"longitudinal by {dim}: "
                + ", ".join(f"{b} {g.letter} ({g.score:.0f})" for b, g in scored.items())
            )
    for g in grades.groups:
        if g.score is not None:
            print(f"{g.name.lower()}: {g.letter} ({g.score:.1f}/100)")
        else:
            print(f"{g.name.lower()}: insufficient data")
    if grades.overall_letter:
        print(f"overall: {grades.overall_letter} ({grades.overall_score:.1f}/100)")
    else:
        print("overall: not enough comparable data to grade")
    print(f"report: {out.resolve()}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
