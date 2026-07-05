"""Download rlogs for a route from the comma connect API.

API notes (verified against api.comma.ai):
- GET https://api.comma.ai/v1/route/<dongleid>%7C<routename>/files
  with header `Authorization: JWT <token>` (literally "JWT", not "Bearer";
  the `|` between dongle id and route name must be URL-encoded as %7C).
- Response JSON key "logs" is a list of time-limited download URLs for the
  rlogs; those are fetched with plain GETs (no auth header).

Downloads are cached under ~/.cache/opgrader/<dongle>|<route>/.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from pathlib import Path

import requests

API_BASE = "https://api.comma.ai/v1"
AUTH_FILE = Path("~/.comma/auth.json").expanduser()
CACHE_DIR = Path(
    os.environ.get("OPGRADER_CACHE", "~/.cache/opgrader")
).expanduser()

# Windows forbids < > : " / \ | ? * and control chars in path components;
# route fullnames are "<dongle>|<route>" and comma's own "|" separator is
# one of them, so this only ever broke for Windows users (POSIX allows it).
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def cache_dir_for_route(route: str) -> Path:
    """Filesystem-safe cache directory for a route fullname.

    Callers must keep using the raw `route` (with "|") for API calls -- only
    the local directory name needs sanitizing.
    """
    return CACHE_DIR / _UNSAFE_PATH_CHARS.sub("_", route)


class DownloadError(RuntimeError):
    pass


def load_jwt(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if AUTH_FILE.exists():
        try:
            tok = json.loads(AUTH_FILE.read_text(encoding="utf-8")).get("access_token")
            if tok:
                return tok
        except (json.JSONDecodeError, OSError) as e:
            raise DownloadError(f"could not read {AUTH_FILE}: {e}") from e
    raise DownloadError(
        "no JWT: pass --jwt or log in once with comma connect so that "
        f"{AUTH_FILE} exists (key 'access_token')"
    )


def route_log_urls(route: str, jwt: str) -> list[str]:
    """route is '<dongleid>|<routename>'."""
    if "|" not in route:
        raise DownloadError(f"route must look like 'dongleid|routename', got {route!r}")
    quoted = urllib.parse.quote(route, safe="")
    url = f"{API_BASE}/route/{quoted}/files"
    r = requests.get(url, headers={"Authorization": f"JWT {jwt}"}, timeout=30)
    if r.status_code == 401:
        raise DownloadError("comma API rejected the JWT (401); token expired?")
    if r.status_code == 404:
        raise DownloadError(f"route not found: {route}")
    r.raise_for_status()
    logs = r.json().get("logs", [])
    if not logs:
        raise DownloadError(f"route {route} has no rlogs on the server")
    return logs


def _seg_filename(url: str, index: int) -> str:
    # url path ends .../<segnum>/rlog.bz2 or .zst; keep the extension
    path = urllib.parse.urlparse(url).path
    ext = ".zst" if path.endswith(".zst") else ".bz2"
    return f"rlog_{index:03d}{ext}"


def download_route(route: str, jwt: str | None = None, progress=print) -> list[Path]:
    """Download (or reuse cached) rlogs for a route; returns local paths in order."""
    token = load_jwt(jwt)
    urls = route_log_urls(route, token)
    dest = cache_dir_for_route(route)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(urls):
        p = dest / _seg_filename(url, i)
        if p.exists() and p.stat().st_size > 0:
            paths.append(p)
            continue
        progress(f"  downloading segment {i + 1}/{len(urls)} -> {p.name}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.rename(p)
        paths.append(p)
    return paths
