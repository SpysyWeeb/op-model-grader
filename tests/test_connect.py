"""connect layer: upload payload construction, badges, job state machine.

No network: the comma API is mocked by monkeypatching opgrader.connect.requests.
"""

import json

import pytest

from opgrader import connect
from opgrader.connect import (
    ApiError,
    JobManager,
    build_athena_payload,
    build_upload_paths,
    files_badge,
    request_upload,
    summarize_route,
)


# ------------------------------------------------------- payload construction


def test_build_upload_paths():
    paths = build_upload_paths("dead1234|0000001c--3d3b422b76", 3)
    assert paths == [
        "0000001c--3d3b422b76--0/rlog.zst",
        "0000001c--3d3b422b76--1/rlog.zst",
        "0000001c--3d3b422b76--2/rlog.zst",
    ]
    # already-bare route names work too
    assert build_upload_paths("r--x", 1) == ["r--x--0/rlog.zst"]


def test_build_athena_payload():
    paths = ["r--0/rlog.zst", "r--1/rlog.zst"]
    url_items = [
        {"url": "https://blob/0", "headers": {"x-extra": "1"}},
        {"url": "https://blob/1"},
    ]
    p = build_athena_payload(paths, url_items, allow_cellular=True)
    assert p["method"] == "uploadFilesToUrls"
    assert p["jsonrpc"] == "2.0"
    fd = p["params"]["files_data"]
    assert len(fd) == 2
    assert fd[0]["fn"] == "r--0/rlog.zst"
    assert fd[0]["url"] == "https://blob/0"
    assert fd[0]["headers"]["x-ms-blob-type"] == "BlockBlob"
    assert fd[0]["headers"]["x-extra"] == "1"  # server-provided headers kept
    assert fd[1]["headers"] == {"x-ms-blob-type": "BlockBlob"}
    assert all(f["allow_cellular"] is True for f in fd)

    p2 = build_athena_payload(paths, url_items, allow_cellular=False)
    assert all(f["allow_cellular"] is False for f in p2["params"]["files_data"])


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def test_request_upload_offline_device(monkeypatch):
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        if "upload_urls" in url:
            return _Resp(200, [{"url": f"https://blob/{i}"} for i in range(2)])
        return _Resp(404, {})  # athena: device offline

    monkeypatch.setattr(connect.requests, "post", fake_post)
    with pytest.raises(ApiError) as e:
        request_upload("dongle1", "r--x", 2, "jwt", allow_cellular=False)
    assert "offline" in str(e.value)
    assert e.value.status == 404
    assert calls[0].endswith("/dongle1/upload_urls/")
    assert calls[1].endswith("/dongle1")


def test_request_upload_success_and_cellular_flag(monkeypatch):
    seen = {}

    def fake_post(url, json=None, **kw):
        if "upload_urls" in url:
            seen["paths"] = json["paths"]
            return _Resp(200, [{"url": f"https://blob/{i}"} for i in range(len(json["paths"]))])
        seen["athena"] = json
        return _Resp(200, {"result": 123, "id": 0})

    monkeypatch.setattr(connect.requests, "post", fake_post)
    res = request_upload("d", "route--abc", 3, "jwt", allow_cellular=True)
    assert res["ok"] is True
    assert "queued" in res["message"]
    assert seen["paths"] == [f"route--abc--{i}/rlog.zst" for i in range(3)]
    assert all(f["allow_cellular"] for f in seen["athena"]["params"]["files_data"])


def test_request_upload_athena_error_verbatim(monkeypatch):
    def fake_post(url, **kw):
        if "upload_urls" in url:
            return _Resp(200, [{"url": "https://blob/0"}])
        return _Resp(200, {"error": {"code": -32000, "message": "queue full"}})

    monkeypatch.setattr(connect.requests, "post", fake_post)
    with pytest.raises(ApiError) as e:
        request_upload("d", "r--x", 1, "jwt")
    assert "queue full" in str(e.value)


# --------------------------------------------------------- route summarizing


def test_summarize_route_and_badge():
    r = summarize_route(
        {
            "fullname": "d1|0000001c--3d3b422b76",
            "segment_numbers": [0, 1, 2, 3],
            "start_time_utc_millis": 1_700_000_000_000,
            "end_time_utc_millis": 1_700_000_600_000,
            "git_branch": "master",
        }
    )
    assert r["n_segments"] == 4
    assert r["duration_s"] == 600.0
    assert r["name"] == "0000001c--3d3b422b76"

    b = files_badge({"logs": ["u"] * 4, "qlogs": ["u"] * 4}, 4)
    assert b["kind"] == "ready" and "4/4" in b["label"]
    b = files_badge({"logs": ["u"] * 2, "qlogs": ["u"] * 4}, 4)
    assert b["kind"] == "partial" and "2/4" in b["label"]
    b = files_badge({"logs": [], "qlogs": ["u"] * 4}, 4)
    assert b["kind"] == "none" and "qlog" in b["label"]
    b = files_badge({}, 0)
    assert b["kind"] == "none"


# ------------------------------------------------------------- job machinery


def test_job_manager_state_machine():
    jm = JobManager()
    snap = jm.snapshot()
    assert snap["active"] is False and snap["phase"] == "idle"

    assert jm.try_start("job A") is True
    assert jm.try_start("job B") is False  # one at a time

    jm.update(phase="downloading", detail="seg 1/5", progress=(1, 5))
    s = jm.snapshot()
    assert s["phase"] == "downloading" and s["progress"] == (1, 5)

    jm.finish("/tmp/x.html")
    s = jm.snapshot()
    assert s["active"] is False and s["phase"] == "done"
    assert s["report"] == "/tmp/x.html"

    # can start again after finishing
    assert jm.try_start("job C") is True
    try:
        raise ValueError("boom")
    except ValueError as e:
        jm.fail(e)
    s = jm.snapshot()
    assert s["active"] is False and s["phase"] == "error"
    assert s["error"]["message"] == "boom"
    assert any("ValueError" in line for line in s["error"]["traceback"])
    # updates after completion are ignored
    jm.update(phase="downloading")
    assert jm.snapshot()["phase"] == "error"


def test_run_grade_job_bad_path_fails_cleanly():
    jm = JobManager()
    assert jm.try_start("bad path")
    connect.run_grade_job(jm, [], ["/nonexistent/rlogs"], None)
    s = jm.snapshot()
    assert s["phase"] == "error"
    assert "not found" in s["error"]["message"]


def test_run_grade_job_demo_route(tmp_path, monkeypatch):
    """Local-path grading end to end through the job runner (no network)."""
    from pathlib import Path

    demo = Path(__file__).resolve().parent.parent / "testdata" / "demo"
    if not demo.is_dir() or not list(demo.glob("rlog*")):
        pytest.skip("demo rlogs not downloaded")
    monkeypatch.setattr(connect, "REPORTS_DIR", tmp_path)
    # keep tests offline: stub the model-id lookup analyze() would perform
    from opgrader import modelid

    monkeypatch.setattr(
        modelid, "resolve",
        lambda meta, session=None: {"label": "stub", "provenance": "unknown", "sha256": None},
    )
    jm = JobManager()
    assert jm.try_start("demo")
    phases = []

    orig_update = jm.update

    def spy(phase=None, **kw):
        if phase:
            phases.append(phase)
        orig_update(phase=phase, **kw)

    jm.update = spy
    connect.run_grade_job(jm, [], [str(demo)], None)
    s = jm.snapshot()
    assert s["phase"] == "done", s
    report = Path(s["report"])
    assert report.is_file() and report.stat().st_size > 100_000
    assert "decoding" in phases and "grading" in phases


def test_clear_jwt_removes_token_keeps_other_keys(tmp_path, monkeypatch):
    f = tmp_path / "auth.json"
    f.write_text(json.dumps({"access_token": "x" * 30, "github_key": "keepme"}))
    monkeypatch.setattr(connect, "AUTH_FILE", f)
    connect.clear_jwt()
    data = json.loads(f.read_text())
    assert "access_token" not in data
    assert data["github_key"] == "keepme"
    assert connect.read_jwt() is None
    connect.clear_jwt()  # signing out twice is a no-op, not an error
    assert json.loads(f.read_text()) == {"github_key": "keepme"}


def test_delete_report_guard_and_cache_clear(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    monkeypatch.setattr(connect, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(connect, "REPORTS_DIR", reports)

    rpt = reports / "x.html"
    rpt.write_text("<html></html>")
    connect.delete_report(str(rpt))
    assert not rpt.exists()
    # outside the reports dir, or wrong extension -> refused
    outside = tmp_path / "evil.html"
    outside.write_text("no")
    with pytest.raises(connect.ApiError):
        connect.delete_report(str(outside))
    with pytest.raises(connect.ApiError):
        connect.delete_report(str(reports / "sneaky.txt"))

    # route cache: dirs other than reports/ count and get cleared
    route = tmp_path / "dongle|2026-01-01--00-00-00"
    route.mkdir()
    (route / "rlog_00.zst").write_bytes(b"z" * 1000)
    assert connect.route_cache_size() == 1000
    freed = connect.clear_route_cache()
    assert freed == 1000
    assert not route.exists() and reports.exists()
