"""Vehicle/model mismatch validation gate: analyze() refuses (by default) to
combine drives from different vehicles or driving models. All model
resolution is mocked here -- see test_modelid.py for modelid.py itself."""

import pytest

from opgrader import modelid, pipeline
from opgrader.events import build_arrays, detect_events
from opgrader.report import render_report
from opgrader.segments import segment_drive
from tests.conftest import make_drive

CONCLUSIVE_A = {
    "label": "CD210 (sha256 6597a1b2c3d4…)",
    "provenance": "from build commit",
    "sha256": "6597" + "a" * 60,
}
CONCLUSIVE_B = {
    "label": "CD211 (sha256 a1b2c3d4e5f6…)",
    "provenance": "from build commit",
    "sha256": "a1b2" + "b" * 60,
}
UNKNOWN = {
    "label": "unavailable (offline or commit/model not found; switcher forks "
             "without a persisted selection can't be identified from logs)",
    "provenance": "unknown",
    "sha256": None,
}


def _route(name, fingerprint, git_remote="https://github.com/o/r.git", git_commit="c" * 40):
    """A minimal but fully-decodable engaged drive -- the mismatch gate runs
    before any of the heavier per-drive analysis, so drive content doesn't
    matter beyond decoding cleanly (mirrors test_modelid.py's
    test_secrets_never_reach_the_report minimal-drive pattern)."""
    d = make_drive(30.0, name=name, vEgo=15.0, enabled=True, latActive=True, longActive=True)
    d.meta.car_fingerprint = fingerprint
    d.meta.git_remote = git_remote
    d.meta.git_commit = git_commit
    seg = segment_drive(d)
    da = build_arrays(d, seg)
    return (d, seg, da, detect_events(d, seg, da))


# --------------------------------------------------------------- vehicle gate


def test_different_car_fingerprint_fires(monkeypatch):
    monkeypatch.setattr(modelid, "resolve", lambda meta, session=None: dict(CONCLUSIVE_A))
    a = _route("routeA", "HYUNDAI_PALISADE")
    b = _route("routeB", "HONDA_CIVIC")
    with pytest.raises(pipeline.MismatchError) as exc:
        pipeline.analyze([a, b])
    msg = str(exc.value)
    assert "HYUNDAI_PALISADE" in msg
    assert "HONDA_CIVIC" in msg


def test_same_vehicle_same_model_unaffected():
    """Must not regress the existing multi-route-same-car workflow."""
    a = _route("routeA", "HYUNDAI_PALISADE", git_commit="c" * 40)
    b = _route("routeB", "HYUNDAI_PALISADE", git_commit="c" * 40)
    an = pipeline.analyze([a, b])  # must not raise
    assert an.mismatch_warning is None


# ----------------------------------------------------------------- model gate


def test_different_commit_same_resolved_sha256_does_not_fire(monkeypatch):
    """Different git_commit -> different internal dedup key -- but BOTH
    resolve to the identical sha256, so this must NOT be treated as a real
    model mismatch (proves the gate keys on resolved identity, not the raw
    (remote, commit, params) tuple)."""
    monkeypatch.setattr(modelid, "resolve", lambda meta, session=None: dict(CONCLUSIVE_A))
    a = _route("routeA", "HYUNDAI_PALISADE", git_commit="c" * 40)
    b = _route("routeB", "HYUNDAI_PALISADE", git_commit="d" * 40)
    an = pipeline.analyze([a, b])  # must not raise
    assert an.mismatch_warning is None
    assert an.model_id["label"] == CONCLUSIVE_A["label"]


def test_one_inconclusive_does_not_fire(monkeypatch):
    """One drive resolves conclusively, the other can't be resolved at all
    (offline/no selector/unknown commit). An unprovable case must never
    count as a confirmed conflict."""
    def fake(meta, session=None):
        return dict(CONCLUSIVE_A) if meta.git_commit == "c" * 40 else dict(UNKNOWN)

    monkeypatch.setattr(modelid, "resolve", fake)
    a = _route("routeA", "HYUNDAI_PALISADE", git_commit="c" * 40)
    b = _route("routeB", "HYUNDAI_PALISADE", git_commit="d" * 40)
    an = pipeline.analyze([a, b])  # must not raise
    assert an.mismatch_warning is None


def test_two_different_conclusive_identities_fires(monkeypatch):
    def fake(meta, session=None):
        return dict(CONCLUSIVE_A) if meta.git_commit == "c" * 40 else dict(CONCLUSIVE_B)

    monkeypatch.setattr(modelid, "resolve", fake)
    a = _route("routeA", "HYUNDAI_PALISADE", git_commit="c" * 40)
    b = _route("routeB", "HYUNDAI_PALISADE", git_commit="d" * 40)
    with pytest.raises(pipeline.MismatchError) as exc:
        pipeline.analyze([a, b])
    msg = str(exc.value)
    assert "CD210" in msg
    assert "CD211" in msg


# --------------------------------------------------------------- allow_mixed


def test_allow_mixed_proceeds_and_report_has_banner(monkeypatch, tmp_path):
    monkeypatch.setattr(modelid, "resolve", lambda meta, session=None: dict(CONCLUSIVE_A))
    a = _route("routeA", "HYUNDAI_PALISADE")
    b = _route("routeB", "HONDA_CIVIC")
    an = pipeline.analyze([a, b], allow_mixed=True)  # must not raise
    assert an.mismatch_warning is not None
    assert "HYUNDAI_PALISADE" in an.mismatch_warning
    assert "HONDA_CIVIC" in an.mismatch_warning

    out = render_report(an, tmp_path / "r.html")
    html = out.read_text()
    assert "MIXED VEHICLES/MODELS" in html
    assert "HYUNDAI_PALISADE" in html
    assert "HONDA_CIVIC" in html
    assert 'class="warn"' in html


def test_allow_mixed_with_no_actual_mismatch_has_no_banner(tmp_path):
    a = _route("routeA", "HYUNDAI_PALISADE", git_commit="c" * 40)
    b = _route("routeB", "HYUNDAI_PALISADE", git_commit="c" * 40)
    an = pipeline.analyze([a, b], allow_mixed=True)
    assert an.mismatch_warning is None

    out = render_report(an, tmp_path / "r.html")
    html = out.read_text()
    assert "MIXED VEHICLES/MODELS" not in html
