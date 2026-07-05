"""Local web UI: browse comma-connect routes, request uploads, grade drives.

Runs a stdlib ThreadingHTTPServer bound to 127.0.0.1 only. The page it
serves (assets/ui.html) talks to the JSON endpoints below; the server talks
to api.comma.ai / athena.comma.ai with the JWT from ~/.comma/auth.json and
never sends it anywhere else.

Endpoints:
  GET  /                    the single-page UI
  GET  /api/me              auth check (identity or authed:false)
  POST /api/auth            save a pasted JWT to ~/.comma/auth.json (0600)
  GET  /api/devices         the account's devices
  GET  /api/routes?dongle=  recent routes (routes_segments, newest first)
  GET  /api/route_files?route=   rlog/qlog availability for one route
  POST /api/request_upload  ask the device (via athena) to upload rlogs
  POST /api/grade           start a grading job (one at a time)
  GET  /api/job             job phase/progress/result
  GET  /api/reports         past reports in the reports dir
  GET  /reports/<name>      serve a finished report
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from .download import AUTH_FILE, CACHE_DIR, route_log_urls

API_BASE = "https://api.comma.ai/v1"
ATHENA_BASE = "https://athena.comma.ai"
REPORTS_DIR = CACHE_DIR / "reports"

_CACHE_TTL = {"devices": 60.0, "routes": 60.0, "files": 20.0}
_api_cache: dict[tuple, tuple[float, object]] = {}
_api_cache_lock = threading.Lock()


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------- auth


def read_jwt() -> str | None:
    try:
        tok = json.loads(AUTH_FILE.read_text()).get("access_token")
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
        data = json.loads(AUTH_FILE.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        pass
    data["access_token"] = token
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUTH_FILE.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
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


def _cached(kind: str, key: tuple, fetch, fresh: bool):
    now = time.time()
    with _api_cache_lock:
        hit = _api_cache.get((kind, key))
        if hit and not fresh and now - hit[0] < _CACHE_TTL[kind]:
            return hit[1]
    data = fetch()
    with _api_cache_lock:
        _api_cache[(kind, key)] = (now, data)
    return data


def get_me(jwt: str) -> dict:
    return api_get("/me/", jwt)


def get_devices(jwt: str, fresh: bool = False) -> list:
    return _cached("devices", (), lambda: api_get("/me/devices/", jwt), fresh)


def get_routes(dongle: str, jwt: str, limit: int = 30, fresh: bool = False) -> list:
    return _cached(
        "routes",
        (dongle, limit),
        lambda: api_get(f"/devices/{dongle}/routes_segments?limit={limit}", jwt),
        fresh,
    )


def get_route_files(fullname: str, jwt: str, fresh: bool = False) -> dict:
    quoted = urllib.parse.quote(fullname, safe="")
    return _cached(
        "files", (fullname,), lambda: api_get(f"/route/{quoted}/files", jwt), fresh
    )


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
        raise ApiError(f"upload_urls returned {len(url_items) if isinstance(url_items, list) else '?'} urls for {len(paths)} paths")

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

    def finish(self, report_url: str):
        with self._lock:
            self._state.update(
                active=False, phase="done", report=report_url, progress=None,
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
                error={"message": str(exc) or type(exc).__name__, "traceback": tail},
                progress=None,
            )

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)


JOBS = JobManager()


def _slug(parts: list[str]) -> str:
    text = "_".join(p.rsplit("/", 1)[-1] for p in parts)[:60]
    return re.sub(r"[^\w.-]+", "-", text).strip("-") or "drive"


def run_grade_job(job: JobManager, routes: list[str], paths: list[str], jwt: str | None):
    """Download missing rlogs, run the pipeline, write the report."""
    # imports here so a UI session without grading stays snappy to start
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
            dest = CACHE_DIR / route.replace("/", "_")
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
        for gi, g in enumerate(groups):
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
        analysis = analyze(per_drive)

        job.update(phase="rendering", detail="writing report")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{time.strftime('%Y%m%d-%H%M%S')}_{_slug(routes + paths)}.html"
        render_report(analysis, REPORTS_DIR / fname)
        job.finish(f"/reports/{fname}")
    except BaseException as e:  # noqa: BLE001 - job thread must never raise
        job.fail(e)


# ------------------------------------------------------------------- server


_UI_PATH = Path(__file__).resolve().parent / "assets" / "ui.html"
_REPORT_NAME_RE = re.compile(r"^[\w.-]+\.html$")


class Handler(BaseHTTPRequestHandler):
    server_version = "opgrader"

    # ---- plumbing
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _err(self, message: str, code: int = 500):
        self._json({"error": message}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except json.JSONDecodeError:
            return {}

    # ---- routing
    def do_GET(self):
        try:
            self._route_get()
        except ApiError as e:
            self._err(str(e), e.status)
        except Exception as e:  # noqa: BLE001 - never crash the server
            self._err(f"{type(e).__name__}: {e}", 500)

    def do_POST(self):
        try:
            self._route_post()
        except ApiError as e:
            self._err(str(e), e.status)
        except Exception as e:  # noqa: BLE001
            self._err(f"{type(e).__name__}: {e}", 500)

    def _route_get(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        fresh = q.get("fresh", ["0"])[0] == "1"
        path = url.path

        if path == "/" or path == "/index.html":
            self._send(200, _UI_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/me":
            jwt = read_jwt()
            if not jwt:
                self._json({"authed": False, "reason": "no token"})
                return
            try:
                me = get_me(jwt)
                self._json({"authed": True, "email": me.get("email"),
                            "user_id": me.get("user_id") or me.get("id")})
            except ApiError as e:
                self._json({"authed": False, "reason": str(e)})
        elif path == "/api/devices":
            self._json({"devices": get_devices(self._jwt(), fresh)})
        elif path == "/api/routes":
            dongle = q.get("dongle", [""])[0]
            if not dongle:
                raise ApiError("missing ?dongle=", status=400)
            routes = get_routes(dongle, self._jwt(), fresh=fresh)
            routes = sorted(
                (summarize_route(r) for r in routes),
                key=lambda r: -(r["start_utc_millis"] or 0),
            )
            self._json({"routes": routes})
        elif path == "/api/route_files":
            fullname = q.get("route", [""])[0]
            if "|" not in fullname:
                raise ApiError("missing/invalid ?route=", status=400)
            n_seg = int(q.get("segments", ["0"])[0])
            files = get_route_files(fullname, self._jwt(), fresh=fresh)
            self._json({"route": fullname, "badge": files_badge(files, n_seg)})
        elif path == "/api/job":
            self._json(JOBS.snapshot())
        elif path == "/api/reports":
            items = []
            if REPORTS_DIR.is_dir():
                for p in sorted(REPORTS_DIR.glob("*.html"), reverse=True):
                    st = p.stat()
                    items.append({"name": p.name, "url": f"/reports/{p.name}",
                                  "mtime": st.st_mtime, "size": st.st_size})
            self._json({"reports": items})
        elif path.startswith("/reports/"):
            name = urllib.parse.unquote(path[len("/reports/"):])
            if not _REPORT_NAME_RE.match(name):
                raise ApiError("bad report name", status=400)
            f = (REPORTS_DIR / name).resolve()
            if not str(f).startswith(str(REPORTS_DIR.resolve())) or not f.is_file():
                raise ApiError("report not found", status=404)
            self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        else:
            self._err("not found", 404)

    def _route_post(self):
        path = urllib.parse.urlparse(self.path).path
        body = self._body()

        if path == "/api/auth":
            token = str(body.get("token") or "")
            try:
                get_me(token)  # validate before saving
            except ApiError:
                raise ApiError("api.comma.ai rejected that token", status=401)
            save_jwt(token)
            self._json({"ok": True})
        elif path == "/api/request_upload":
            res = request_upload(
                dongle=str(body.get("dongle") or ""),
                route_name=str(body.get("route") or ""),
                n_segments=int(body.get("segments") or 0),
                jwt=self._jwt(),
                allow_cellular=bool(body.get("allow_cellular")),
            )
            self._json(res)
        elif path == "/api/grade":
            routes = [str(r) for r in body.get("routes") or []]
            paths = [str(p) for p in body.get("paths") or [] if str(p).strip()]
            if not routes and not paths:
                raise ApiError("select at least one route or local path", status=400)
            for p in paths:
                if not Path(p).expanduser().exists():
                    raise ApiError(f"local path not found: {p}", status=400)
            desc = ", ".join(routes + paths)
            if not JOBS.try_start(desc):
                raise ApiError("a grading job is already running — wait for it to finish", status=409)
            jwt = read_jwt()
            t = threading.Thread(
                target=run_grade_job, args=(JOBS, routes, paths, jwt), daemon=True
            )
            t.start()
            self._json({"ok": True})
        else:
            self._err("not found", 404)

    def _jwt(self) -> str:
        jwt = read_jwt()
        if not jwt:
            raise ApiError("not authenticated — paste a JWT first", status=401)
        return jwt


def make_server(port: int = 8385) -> ThreadingHTTPServer:
    """Build the server (127.0.0.1 only). port=0 picks a free port."""
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(port: int = 8385, open_browser: bool = True) -> None:
    httpd = make_server(port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"opgrader web UI: {url}  (Ctrl-C to stop; local-only, your JWT stays on this machine)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
