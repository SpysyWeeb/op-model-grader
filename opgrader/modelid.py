"""Identify the driving model behind a drive. Entirely best-effort/offline-safe.

Layered resolution (first hit wins):

1. Fork model-selector params. Switcher forks (sunnypilot model manager,
   FrogPilot) download models at runtime, so the git commit does NOT
   determine the active model there. initData.params is a boot-time snapshot
   of the Params DB and those forks persist the selection in it (model
   switches require a restart, so the snapshot holds for the whole route).
   Only the whitelisted selector keys are ever read (extract.py enforces
   this at decode time) — the raw params dump contains GithubSshKeys and
   other secrets and must never reach report-visible data.
   Verified key names (from the forks' sources):
     - sunnypilot: ModelManager_ActiveBundle (JSON; displayName/internalName)
     - FrogPilot:  Model (internal name), with AvailableModels /
                   AvailableModelNames as parallel comma-separated lists
                   mapping internal -> display names
2. git-lfs pointer at initData.gitCommit (stock and non-switcher forks):
   openpilot commits driving models as LFS pointers containing
   "oid sha256:<hash>". We resolve the commit's tree via the GitHub API
   (tree/blob endpoints work even for force-push-orphaned commits), read the
   pointer, and reverse-lookup the hash in the bundled model_hashes.json.
3. Unknown: report the sha256 prefix / commit so a human can identify it.

Lookups hit the network at most once per (remote, commit) — results are
cached in ~/.cache/opgrader/modelid.json — and every failure degrades to
"unavailable"; grading works fully offline.
"""

from __future__ import annotations

import base64
import json
import re
import string
from pathlib import Path

import requests

from .download import CACHE_DIR

MODEL_HASHES_FILE = Path(__file__).resolve().parent / "model_hashes.json"
LOOKUP_CACHE_FILE = CACHE_DIR / "modelid.json"
API_TIMEOUT = 10.0

# model files to try, in priority order (dmonitoring is never a candidate)
CANDIDATE_NAMES = [
    "driving_policy.onnx",
    "driving_vision.onnx",
    "supercombo.onnx",
    "driving_supercombo.onnx",
]
MODELS_DIR_PARTS = ["selfdrive", "modeld", "models"]
# 0.11+ nests the repo under a top-level openpilot/ directory
TREE_PREFIXES = ([], ["openpilot"])

_PRINTABLE = set(string.printable) - set("\x0b\x0c")


def sanitize(text: str, max_len: int = 80) -> str:
    """Printable-only, single-line, length-capped (selector values are
    user/fork-controlled strings)."""
    out = "".join(c for c in str(text) if c in _PRINTABLE and c not in "\r\n\t")
    out = out.strip()
    return out[:max_len] + ("…" if len(out) > max_len else "")


# ------------------------------------------------- layer 1: selector params


def from_selector_params(model_params: dict[str, str]) -> dict | None:
    """Resolve the model from fork model-selector params, if present."""
    if not model_params:
        return None

    # sunnypilot model manager: JSON bundle with displayName/internalName
    raw = model_params.get("ModelManager_ActiveBundle")
    if raw:
        try:
            bundle = json.loads(raw)
            if isinstance(bundle, dict):
                name = bundle.get("displayName") or bundle.get("display_name") or \
                    bundle.get("internalName") or bundle.get("internal_name")
                if name:
                    return {
                        "label": sanitize(name),
                        "provenance": "sunnypilot model selector",
                        "sha256": None,
                    }
        except (json.JSONDecodeError, TypeError):
            pass

    # FrogPilot: Model = internal name; map to display name when the
    # parallel AvailableModels / AvailableModelNames lists are present
    raw = model_params.get("Model")
    if raw:
        name = sanitize(raw)
        internals = [s.strip() for s in (model_params.get("AvailableModels") or "").split(",")]
        displays = [s.strip() for s in (model_params.get("AvailableModelNames") or "").split(",")]
        if name in internals and len(displays) == len(internals):
            disp = displays[internals.index(name)]
            if disp:
                name = sanitize(disp)
        if name:
            return {
                "label": name,
                "provenance": "FrogPilot model selector",
                "sha256": None,
            }
    return None


# --------------------------------------------- layer 2: git-lfs at gitCommit


_REMOTE_RES = [
    re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", re.I),
    re.compile(r"^git@github\.com:([\w.-]+)/([\w.-]+?)(?:\.git)?$", re.I),
    re.compile(r"^ssh://git@github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", re.I),
]


def parse_remote(remote: str) -> tuple[str, str] | None:
    """github remote URL -> (owner, repo); None for non-GitHub remotes."""
    remote = (remote or "").strip()
    for rx in _REMOTE_RES:
        m = rx.match(remote)
        if m:
            return m.group(1), m.group(2)
    return None


def parse_lfs_pointer(text: str) -> str | None:
    """sha256 oid from a git-lfs pointer file, or None."""
    if not text.startswith("version https://git-lfs"):
        return None
    m = re.search(r"oid sha256:([0-9a-f]{64})", text)
    return m.group(1) if m else None


def _load_lookup_cache() -> dict:
    try:
        data = json.loads(LOOKUP_CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_lookup_cache(cache: dict) -> None:
    try:
        LOOKUP_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOOKUP_CACHE_FILE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    except OSError:
        pass


def _api(session, url: str, **params):
    r = session.get(url, params=params or None, timeout=API_TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def fetch_model_pointer(owner: str, repo: str, commit: str, session=None) -> dict | None:
    """Find the driving-model LFS pointer at a commit via the GitHub API.

    Uses commits -> git/trees -> git/blobs (SHA-addressed, so it works even
    when the commit is no longer reachable from any branch after a force
    push). Returns {"sha256": ..., "path": ...} or None. Raises on network
    errors (caller treats those as 'unavailable', uncached).
    """
    session = session or requests
    api = f"https://api.github.com/repos/{owner}/{repo}"
    c = _api(session, f"{api}/commits/{commit}")
    if not c:
        return None
    tree_sha = c["commit"]["tree"]["sha"]

    def get_tree(sha):
        return _api(session, f"{api}/git/trees/{sha}") or {"tree": []}

    def descend(sha, parts):
        for part in parts:
            hit = next((e for e in get_tree(sha)["tree"] if e["path"] == part), None)
            if hit is None:
                return None
            sha = hit["sha"]
        return sha

    models_sha = None
    for prefix in TREE_PREFIXES:
        models_sha = descend(tree_sha, prefix + MODELS_DIR_PARTS)
        if models_sha:
            break
    if not models_sha:
        return None

    entries = [
        e for e in get_tree(models_sha)["tree"]
        if e["type"] == "blob" and e["path"].endswith(".onnx")
        and not e["path"].startswith("dmonitoring")
    ]
    # candidates first, then any other .onnx from the tree listing
    order = {name: i for i, name in enumerate(CANDIDATE_NAMES)}
    entries.sort(key=lambda e: order.get(e["path"], len(order)))

    for e in entries:
        if e.get("size", 0) > 4096:
            continue  # a real onnx, not an LFS pointer
        blob = _api(session, f"{api}/git/blobs/{e['sha']}")
        if not blob:
            continue
        try:
            text = base64.b64decode(blob.get("content", "")).decode("utf-8", "replace")
        except Exception:
            continue
        oid = parse_lfs_pointer(text)
        if oid:
            return {"sha256": oid, "path": e["path"]}
    return None


def lookup_hash(sha256: str) -> dict | None:
    try:
        table = json.loads(MODEL_HASHES_FILE.read_text(encoding="utf-8"))
        return table.get(sha256)
    except (OSError, json.JSONDecodeError):
        return None


def from_git(remote: str, commit: str, session=None) -> dict | None:
    """Layer 2: resolve via the LFS pointer at the build commit (cached)."""
    parsed = parse_remote(remote)
    if not parsed or not commit or len(commit) < 7:
        return None
    owner, repo = parsed
    key = f"{owner}/{repo}@{commit}"
    cache = _load_lookup_cache()
    if key in cache:
        found = cache[key]  # may be None (cached 404/no-model: permanent)
    else:
        try:
            found = fetch_model_pointer(owner, repo, commit, session=session)
        except Exception:
            return None  # network trouble: don't cache, stay quiet
        cache[key] = found
        _save_lookup_cache(cache)

    if not found:
        return None
    sha = found["sha256"]
    known = lookup_hash(sha)
    if known:
        label = f"{known['name']} (sha256 {sha[:12]}…)"
    else:
        label = f"sha256 {sha[:12]}… — unknown model (add to model_hashes.json)"
    return {"label": label, "provenance": "from build commit", "sha256": sha}


# ------------------------------------------------------------------ resolve


def resolve(meta, session=None) -> dict:
    """Layered model identification for one drive's Meta.

    Returns {"label", "provenance", "sha256"}; never raises.
    """
    try:
        hit = from_selector_params(getattr(meta, "model_params", {}) or {})
        if hit:
            return _dirty_suffix(hit, meta)
        hit = from_git(getattr(meta, "git_remote", ""), getattr(meta, "git_commit", ""),
                       session=session)
        if hit:
            return _dirty_suffix(hit, meta)
        remote = getattr(meta, "git_remote", "")
        if remote and not parse_remote(remote):
            reason = "non-GitHub remote"
        elif not getattr(meta, "git_commit", ""):
            reason = "no commit in logs"
        else:
            reason = "offline or commit/model not found"
        return {
            "label": f"unavailable ({reason}; switcher forks without a persisted "
                     "selection can't be identified from logs)",
            "provenance": "unknown",
            "sha256": None,
        }
    except Exception:
        return {"label": "unavailable", "provenance": "unknown", "sha256": None}


def _dirty_suffix(hit: dict, meta) -> dict:
    if getattr(meta, "dirty", False):
        hit = dict(hit)
        hit["label"] += " (dirty build — hash may not match device)"
    return hit
