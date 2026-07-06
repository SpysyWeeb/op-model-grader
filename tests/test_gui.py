"""Tkinter UI smoke test (skipped headless or without tkinter)."""

import os

import pytest

tk = pytest.importorskip("tkinter")

if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    pytest.skip("no display", allow_module_level=True)


@pytest.fixture()
def root():
    try:
        r = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"cannot open display: {e}")
    r.withdraw()
    yield r
    r.destroy()


def test_app_builds_and_polls(root, monkeypatch):
    from opgrader import gui
    from opgrader import connect as C

    # keep the constructor offline: no token found, nothing to check
    monkeypatch.setattr(C, "read_jwt", lambda: None)
    app = gui.App(root)

    # let the auth-check thread finish and its after() callback run
    import time

    deadline = time.time() + 5
    while time.time() < deadline and app.token_entry.winfo_manager() == "":
        root.update()
        time.sleep(0.05)

    # unauthenticated -> the token entry is shown
    assert app.token_entry.winfo_manager() != ""
    # core widgets exist
    assert app.tree.winfo_exists()
    assert app.grade_btn.winfo_exists()

    # job poller handles an idle job without touching the network
    app._poll_job()
    assert "idle" in app.status.cget("text")

    # route rendering works from plain dicts
    app.routes = [
        {
            "fullname": "d|r--1",
            "name": "r--1",
            "start_utc_millis": 1_700_000_000_000,
            "duration_s": 300,
            "n_segments": 5,
            "git_branch": "master",
            "git_remote": "",
            "platform": "",
        }
    ]
    app.badges["d|r--1"] = {"label": "rlogs ready 5/5", "kind": "ready",
                            "n_logs": 5, "n_segments": 5}
    app._render_routes()
    assert app.tree.exists("d|r--1")
    # columns: started, duration, segments, branch, vehicle, rlogs
    assert app.tree.item("d|r--1")["values"][4] == "–"  # no platform given -> placeholder
    assert app.tree.item("d|r--1")["values"][5] == "rlogs ready 5/5"


def _route_dict(fullname, platform):
    return {
        "fullname": fullname, "name": fullname.split("|", 1)[-1],
        "start_utc_millis": 1_700_000_000_000, "duration_s": 120,
        "n_segments": 2, "git_branch": "master", "git_remote": "", "platform": platform,
    }


def test_grade_prompts_and_can_be_declined_on_different_vehicle_platforms(root, monkeypatch, tmp_path):
    """The cheap pre-check (platform strings, no decode) must catch an
    obviously-different-vehicle selection before a job is even started, and
    declining it must leave no job running."""
    from tkinter import messagebox

    from opgrader import config
    from opgrader import connect as C
    from opgrader import gui

    monkeypatch.setattr(C, "read_jwt", lambda: None)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    app = gui.App(root)

    app.routes = [
        _route_dict("d|rA", "HYUNDAI_PALISADE"),
        _route_dict("d|rB", "HONDA_CIVIC"),
    ]
    app.badges["d|rA"] = {"label": "rlogs ready 2/2", "kind": "ready", "n_logs": 2, "n_segments": 2}
    app.badges["d|rB"] = {"label": "rlogs ready 2/2", "kind": "ready", "n_logs": 2, "n_segments": 2}
    app._render_routes()
    app.tree.selection_set("d|rA", "d|rB")

    asked = {}

    def fake_askyesno(title, msg):
        asked["title"], asked["msg"] = title, msg
        return False  # decline

    monkeypatch.setattr(messagebox, "askyesno", fake_askyesno)
    app._grade()

    assert "vehicles" in asked["title"]
    assert "HYUNDAI_PALISADE" in asked["msg"] and "HONDA_CIVIC" in asked["msg"]
    assert not app.jobs.snapshot()["active"]  # declined -> no job launched


def test_grade_proceeds_without_prompt_when_platforms_match(root, monkeypatch, tmp_path):
    """Same-vehicle multi-select must not be interrupted by the pre-check."""
    from tkinter import messagebox

    from opgrader import config
    from opgrader import connect as C
    from opgrader import gui

    monkeypatch.setattr(C, "read_jwt", lambda: None)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    app = gui.App(root)

    app.routes = [
        _route_dict("d|rA", "HYUNDAI_PALISADE"),
        _route_dict("d|rB", "HYUNDAI_PALISADE"),
    ]
    app.badges["d|rA"] = {"label": "rlogs ready 2/2", "kind": "ready", "n_logs": 2, "n_segments": 2}
    app.badges["d|rB"] = {"label": "rlogs ready 2/2", "kind": "ready", "n_logs": 2, "n_segments": 2}
    app._render_routes()
    app.tree.selection_set("d|rA", "d|rB")

    def fail_if_asked(*a, **kw):
        raise AssertionError("askyesno should not be called when platforms match")

    monkeypatch.setattr(messagebox, "askyesno", fail_if_asked)
    app._grade()  # must not raise, must reach job launch (no JWT -> job fails async, that's fine)

    import time

    deadline = time.time() + 5
    while time.time() < deadline and app.jobs.snapshot()["active"]:
        root.update()
        time.sleep(0.05)
    assert app._last_grade_args is not None
    assert set(app._last_grade_args[0]) == {"d|rA", "d|rB"}


def test_network_error_offers_plain_retry_not_a_traceback(root, monkeypatch, tmp_path):
    """A transient connection failure (what a dropped wifi/VPN blip looks
    like downloading rlogs) should offer a plain "retry?" dialog, same
    treatment as MismatchError already gets -- not a raw traceback dump,
    which is what every OTHER exception type still correctly shows."""
    import requests
    from tkinter import messagebox

    from opgrader import config
    from opgrader import connect as C
    from opgrader import gui

    monkeypatch.setattr(C, "read_jwt", lambda: None)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    app = gui.App(root)

    app._last_grade_args = (["d|rA"], [], config.get_t_follow(), False)
    app.jobs.try_start("d|rA")
    app.jobs.fail(requests.exceptions.ConnectionError("Max retries exceeded"))

    asked = {}

    def fake_askyesno(title, msg):
        asked["title"], asked["msg"] = title, msg
        return False  # decline the retry -- just verifying the right dialog fired

    def fail_if_shown(*a, **kw):
        raise AssertionError("showerror (raw traceback) must not be used for a network error")

    monkeypatch.setattr(messagebox, "askyesno", fake_askyesno)
    monkeypatch.setattr(messagebox, "showerror", fail_if_shown)

    app._poll_job()

    assert asked.get("title") == "grading failed"
    assert "network" in asked["msg"].lower()
    assert "Retry" in asked["msg"]
    assert "Traceback" not in asked["msg"] and "ConnectionError" not in asked["msg"]
