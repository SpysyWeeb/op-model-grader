"""comma-connect client + grading-job machinery (UI-agnostic).

Used by the Tkinter UI (gui.py). Talks only to api.comma.ai and
athena.comma.ai with the JWT from ~/.comma/auth.json.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import traceback
import urllib.parse
from pathlib import Path

import requests

from .download import AUTH_FILE, CACHE_DIR, cache_dir_for_route, route_log_urls

API_BASE = "https://api.comma.ai/v1"
ATHENA_BASE = "https://athena.comma.ai"
REPORTS_DIR = CACHE_DIR / "reports"


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------- auth


def read_jwt() -> str | None:
    try:
        tok = json.loads(AUTH_FILE.read_text(encoding="utf-8")).get("access_token")
        return tok or None
    except (OSError, json.JSONDecodeError):
        return None


def save_jwt(token: str) -> None:
    """Merge access_token into ~/.comma/auth.json, keeping other keys, 0600."""
    token = token.strip()
    if not token or len(token) < 20:
        raise ApiError("that doesn't look like a JWT", status=400)
    data = {}
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        pass
    data["access_token"] = token
    _write_auth(data)


def clear_jwt() -> None:
    """Remove access_token from ~/.comma/auth.json, keeping any other keys."""
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict) or "access_token" not in data:
        return
    del data["access_token"]
    _write_auth(data)


def _write_auth(data: dict) -> None:
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUTH_FILE.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, AUTH_FILE)
    os.chmod(AUTH_FILE, 0o600)


# --------------------------------------------------------- comma API client


def api_get(path: str, jwt: str, timeout: float = 30.0):
    r = requests.get(
        f"{API_BASE}{path}", headers={"Authorization": f"JWT {jwt}"}, timeout=timeout
    )
    if r.status_code == 401:
        raise ApiError("token rejected by api.comma.ai (expired?)", status=401)
    if r.status_code >= 400:
        raise ApiError(f"api.comma.ai returned {r.status_code} for {path}")
    return r.json()


def get_me(jwt: str) -> dict:
    return api_get("/me/", jwt)


def get_devices(jwt: str) -> list:
    return api_get("/me/devices/", jwt)


def get_routes(dongle: str, jwt: str, limit: int = 30) -> list:
    return api_get(f"/devices/{dongle}/routes_segments?limit={limit}", jwt)


def get_route_files(fullname: str, jwt: str) -> dict:
    quoted = urllib.parse.quote(fullname, safe="")
    return api_get(f"/route/{quoted}/files", jwt)


def summarize_route(r: dict) -> dict:
    """Tolerant extraction of the routes_segments fields the UI shows."""
    fullname = r.get("fullname") or ""
    seg_numbers = r.get("segment_numbers") or []
    if seg_numbers:
        n_segments = int(max(seg_numbers)) + 1
    elif r.get("maxqlog") is not None and r["maxqlog"] >= 0:
        n_segments = int(r["maxqlog"]) + 1
    else:
        n_segments = 0
    start_ms = r.get("start_time_utc_millis") or 0
    end_ms = r.get("end_time_utc_millis") or 0
    return {
        "fullname": fullname,
        "name": fullname.split("|", 1)[-1],
        "start_utc_millis": start_ms,
        "duration_s": max(0, (end_ms - start_ms) / 1000.0) if start_ms and end_ms else None,
        "n_segments": n_segments,
        "git_branch": r.get("git_branch") or "",
        "git_remote": r.get("git_remote") or "",
        "platform": r.get("platform") or "",
    }


def files_badge(files: dict, n_segments: int) -> dict:
    n_logs = len(files.get("logs") or [])
    n_qlogs = len(files.get("qlogs") or [])
    total = max(n_segments, n_logs, n_qlogs)
    if n_logs >= total and total > 0:
        label, kind = f"rlogs ready {n_logs}/{total}", "ready"
    elif n_logs > 0:
        label, kind = f"rlogs {n_logs}/{total}", "partial"
    elif n_qlogs > 0:
        label, kind = "qlogs only", "none"
    else:
        label, kind = "no logs uploaded", "none"
    return {"label": label, "kind": kind, "n_logs": n_logs, "n_segments": total}


def local_badge(fullname: str, n_segments: int) -> dict | None:
    """Badge for a route whose rlogs are ALREADY fully present in the local
    download cache (e.g. a previous grading run) -- None if not, so the
    caller falls back to the normal server-side files_badge. Same filename
    convention download.py/run_grade_job use when saving segments."""
    if not n_segments:
        return None
    d = cache_dir_for_route(fullname)
    if not d.is_dir():
        return None
    have = sum(
        1
        for i in range(n_segments)
        if any((p := d / f"rlog_{i:03d}{ext}").exists() and p.stat().st_size > 0
               for ext in (".zst", ".bz2"))
    )
    if have < n_segments:
        return None
    return {"label": "rlogs downloaded", "kind": "downloaded", "n_logs": have, "n_segments": n_segments}


# ------------------------------------------------------------ upload request


def build_upload_paths(route_name: str, n_segments: int, ext: str = "zst") -> list[str]:
    """Device-side paths for upload_urls: <routename>--<i>/rlog.<ext>."""
    if "|" in route_name:
        route_name = route_name.split("|", 1)[1]
    return [f"{route_name}--{i}/rlog.{ext}" for i in range(n_segments)]


def build_athena_payload(paths: list[str], url_items: list[dict], allow_cellular: bool) -> dict:
    """JSON-RPC uploadFilesToUrls payload from the upload_urls response."""
    files_data = []
    for p, item in zip(paths, url_items):
        headers = dict(item.get("headers") or {})
        headers.setdefault("x-ms-blob-type", "BlockBlob")
        files_data.append(
            {
                "fn": p,
                "url": item["url"],
                "headers": headers,
                "allow_cellular": bool(allow_cellular),
            }
        )
    return {
        "method": "uploadFilesToUrls",
        "params": {"files_data": files_data},
        "jsonrpc": "2.0",
        "id": 0,
    }


def request_upload(
    dongle: str, route_name: str, n_segments: int, jwt: str, allow_cellular: bool = False
) -> dict:
    if n_segments <= 0:
        raise ApiError("route has no segments to request", status=400)
    paths = build_upload_paths(route_name, n_segments)
    r = requests.post(
        f"{API_BASE}/{dongle}/upload_urls/",
        json={"paths": paths},
        headers={"Authorization": f"JWT {jwt}"},
        timeout=30,
    )
    if r.status_code >= 400:
        raise ApiError(f"upload_urls failed: HTTP {r.status_code} {r.text[:200]}")
    url_items = r.json()
    if not isinstance(url_items, list) or len(url_items) != len(paths):
        raise ApiError(
            f"upload_urls returned {len(url_items) if isinstance(url_items, list) else '?'} "
            f"urls for {len(paths)} paths"
        )

    payload = build_athena_payload(paths, url_items, allow_cellular)
    ar = requests.post(
        f"{ATHENA_BASE}/{dongle}",
        json=payload,
        headers={"Authorization": f"JWT {jwt}"},
        timeout=45,
    )
    if ar.status_code == 404:
        raise ApiError(
            "device is offline — start the car (or wake the device) and try again",
            status=404,
        )
    if ar.status_code >= 400:
        raise ApiError(f"athena error: HTTP {ar.status_code} {ar.text[:200]}")
    resp = ar.json()
    if isinstance(resp, dict) and resp.get("error"):
        raise ApiError(f"athena error: {json.dumps(resp['error'])[:200]}")
    return {
        "ok": True,
        "message": "upload queued — uploads when the device is on WiFi"
        + (" (cellular allowed)" if allow_cellular else ""),
        "n_files": len(paths),
    }


# --------------------------------------------------------------------- jobs


class JobManager:
    """One grading job at a time; lock-protected state dict for polling."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = {"active": False, "phase": "idle"}

    def try_start(self, description: str) -> bool:
        with self._lock:
            if self._state.get("active"):
                return False
            self._state = {
                "active": True,
                "phase": "starting",
                "detail": description,
                "progress": None,  # (done, total) or None
                "report": None,
                "error": None,
                "started": time.time(),
            }
            return True

    def update(self, phase: str | None = None, detail: str | None = None,
               progress: tuple[int, int] | None = None):
        with self._lock:
            if not self._state.get("active"):
                return
            if phase is not None:
                self._state["phase"] = phase
            if detail is not None:
                self._state["detail"] = detail
            self._state["progress"] = progress

    def finish(self, report_path: str):
        with self._lock:
            self._state.update(
                active=False, phase="done", report=report_path, progress=None,
                detail="report ready",
            )

    def fail(self, exc: BaseException):
        tail = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ).strip().splitlines()[-12:]
        with self._lock:
            self._state.update(
                active=False,
                phase="error",
                # "type" lets callers (gui.py) recognize a specific failure
                # (e.g. pipeline.MismatchError) without keeping the live
                # exception object around across the thread boundary.
                error={"message": str(exc) or type(exc).__name__, "traceback": tail,
                       "type": type(exc).__name__},
                progress=None,
            )

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)


def _slug(parts: list[str]) -> str:
    text = "_".join(p.rsplit("/", 1)[-1] for p in parts)[:60]
    return re.sub(r"[^\w.-]+", "-", text).strip("-") or "drive"


def run_grade_job(job: JobManager, routes: list[str], paths: list[str], jwt: str | None,
                  t_follow_targets: dict | None = None, use_profile: bool = True,
                  allow_mixed: bool = False):
    """Download missing rlogs, run the pipeline, write the report.

    Meant to run in a background thread; never raises (failures land in the
    job state). On success job.report is the absolute report path. If the
    selected routes are from different vehicles/driving models and
    allow_mixed is False, analyze() raises pipeline.MismatchError, which
    lands here as a normal job failure (job["error"]["message"] carries its
    text) -- gui.py offers a retry-with-allow_mixed action on that failure.
    """
    from .events import build_arrays, detect_events
    from .extract import extract_drive
    from .logreader import find_segments, group_segments, route_name_for_group
    from .pipeline import analyze
    from .report import render_report
    from .segments import segment_drive

    try:
        inputs: list[str] = []
        for route in routes:
            if not jwt:
                raise ApiError("no JWT for downloading routes", status=401)
            job.update(phase="downloading", detail=f"{route}: listing files")
            urls = route_log_urls(route, jwt)
            dest = cache_dir_for_route(route)
            dest.mkdir(parents=True, exist_ok=True)
            for i, url in enumerate(urls):
                ext = ".zst" if urllib.parse.urlparse(url).path.endswith(".zst") else ".bz2"
                p = dest / f"rlog_{i:03d}{ext}"
                job.update(
                    phase="downloading",
                    detail=f"{route}: segment {i + 1}/{len(urls)}",
                    progress=(i, len(urls)),
                )
                if p.exists() and p.stat().st_size > 0:
                    continue
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                tmp = p.with_suffix(p.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.rename(p)
            inputs.append(str(dest))

        for p in paths:
            pp = Path(p).expanduser()
            if not pp.exists():
                raise ApiError(f"local path not found: {p}", status=400)
            inputs.append(str(pp))

        job.update(phase="scanning", detail="finding segments", progress=None)
        seg_files = find_segments(inputs)
        if not seg_files:
            raise ApiError("no rlog segments found in the selected inputs", status=400)
        groups = group_segments(seg_files)

        per_drive = []
        for g in groups:
            name = route_name_for_group(g)

            def cb(i, n, _ev, _name=name):
                job.update(
                    phase="decoding",
                    detail=f"{_name}: segment {i + 1}/{n}",
                    progress=(i + 1, n),
                )

            job.update(phase="decoding", detail=f"{name}: segment 1/{len(g)}",
                       progress=(0, len(g)))
            drive = extract_drive(name, g, progress=cb)
            seg = segment_drive(drive)
            if seg is None:
                continue
            da = build_arrays(drive, seg)
            per_drive.append((drive, seg, da, detect_events(drive, seg, da)))

        if not per_drive:
            raise ApiError("no usable drives decoded from the inputs", status=400)

        job.update(phase="grading", detail="detecting maneuvers and scoring", progress=None)
        if t_follow_targets is None:
            from .config import get_t_follow

            t_follow_targets = get_t_follow()
        analysis = analyze(
            per_drive, t_follow_targets=t_follow_targets, use_profile=use_profile,
            allow_mixed=allow_mixed,
        )

        job.update(phase="rendering", detail="writing report")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{time.strftime('%Y%m%d-%H%M%S')}_{_slug(routes + paths)}.html"
        out = REPORTS_DIR / fname
        render_report(analysis, out)
        job.finish(str(out))
    except BaseException as e:  # noqa: BLE001 - job thread must never raise
        job.fail(e)


def list_reports() -> list[dict]:
    items = []
    if REPORTS_DIR.is_dir():
        for p in sorted(REPORTS_DIR.glob("*.html"), reverse=True):
            st = p.stat()
            items.append({"name": p.name, "path": str(p),
                          "mtime": st.st_mtime, "size": st.st_size})
    return items


def delete_report(path: str) -> None:
    """Delete a generated report; refuses anything outside the reports dir."""
    p = Path(path).resolve()
    if p.parent != REPORTS_DIR.resolve() or p.suffix != ".html":
        raise ApiError("refusing to delete a file outside the reports folder", status=400)
    p.unlink(missing_ok=True)


def cached_route_dirs() -> list[Path]:
    """Downloaded-rlog cache directories (everything in CACHE_DIR except reports)."""
    if not CACHE_DIR.is_dir():
        return []
    reports = REPORTS_DIR.resolve()
    return [d for d in CACHE_DIR.iterdir() if d.is_dir() and d.resolve() != reports]


def route_cache_size() -> int:
    """Total bytes of downloaded rlogs in the cache."""
    return sum(
        f.stat().st_size
        for d in cached_route_dirs()
        for f in d.rglob("*")
        if f.is_file()
    )


def clear_route_cache() -> int:
    """Delete all downloaded rlogs from the cache; returns bytes freed."""
    freed = route_cache_size()
    for d in cached_route_dirs():
        shutil.rmtree(d, ignore_errors=True)
    return freed
