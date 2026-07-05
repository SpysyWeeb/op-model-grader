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
    assert app.tree.item("d|r--1")["values"][4] == "rlogs ready 5/5"
