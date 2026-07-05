"""End-to-end on the demo route (skipped when testdata is absent)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "testdata" / "demo"

pytestmark = pytest.mark.skipif(
    not DEMO.is_dir() or not list(DEMO.glob("rlog*")),
    reason="demo rlogs not downloaded",
)


@pytest.fixture(scope="module")
def demo_analysis():
    from opgrader.events import build_arrays, detect_events
    from opgrader.extract import extract_drive
    from opgrader.logreader import find_segments, group_segments, route_name_for_group
    from opgrader.pipeline import analyze
    from opgrader.segments import segment_drive

    groups = group_segments(find_segments([str(DEMO)]))
    assert len(groups) == 1
    per = []
    for g in groups:
        d = extract_drive(route_name_for_group(g), g)
        assert d.n_segments == 13
        seg = segment_drive(d)
        da = build_arrays(d, seg)
        per.append((d, seg, da, detect_events(d, seg, da)))
    return analyze(per)


def test_demo_decodes_and_finds_spans(demo_analysis):
    d, seg, _da, _e = demo_analysis.per_drive[0]
    assert d.meta.car_fingerprint.startswith("TOYOTA")
    assert seg.time_of("engaged", "long") > 300
    assert seg.time_of("manual", "long") > 30


def test_demo_finds_events_both_kinds(demo_analysis):
    events = demo_analysis.per_drive[0][3]
    kinds = {e.kind for e in events}
    assert "follow" in kinds
    assert "stop" in kinds
    assert "turn" in kinds


def test_demo_numbers_are_sane(demo_analysis):
    s = demo_analysis.samples
    for v in s["rms_jerk"]["model"] + s["rms_jerk"]["driver"]:
        assert 0.05 < v < 3.0, f"absurd RMS jerk {v}"
    for v in s["median_gap"]["model"] + s["median_gap"]["driver"]:
        assert 0.3 < v < 5.0, f"absurd time gap {v}"
    for v in s["lead_decel_latency"]["model"]:
        assert 0.0 < v <= 4.0


def test_demo_report_renders(demo_analysis, tmp_path):
    from opgrader.report import render_report

    out = render_report(demo_analysis, tmp_path / "report.html")
    doc = out.read_text()
    assert len(doc) < 10 * 1024 * 1024
    assert "<title>" in doc
    assert doc.count('class="card ') >= 9  # 5 long + 4 lat categories
    m = re.search(r"const DATA = (\{.*?\});\n", doc, re.S)
    assert m, "embedded payload missing"
    data = json.loads(m.group(1))
    assert sum(len(v) for v in data["events"].values()) > 10
    ev = data["events"]["stop"][0]
    tr = {s["label"]: s["data"] for s in ev["series"]}
    assert "vEgo" in tr and len(tr["vEgo"]) == len(ev["series"][0]["data"])
    # traces are 10 Hz-ish and rounded
    assert all(v is None or abs(v) < 1e5 for v in tr["vEgo"])


def test_demo_grades_exist(demo_analysis):
    g = demo_analysis.grades
    assert g.overall_score is not None
    assert g.overall_letter in {"A", "A-", "B+", "B", "C", "D", "F"}
    names = {gr.name for gr in g.groups}
    assert names == {"Longitudinal", "Lateral"}
