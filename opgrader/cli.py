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
    p.add_argument("-o", "--out", default="report.html", help="output HTML path")
    p.add_argument("--open", action="store_true", help="open the report in a browser")
    p.add_argument("--version", action="version", version=f"opgrader {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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

    analysis = analyze(per_drive)
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
