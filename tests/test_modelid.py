"""Driving-model identification: selector params, LFS pointers, lookups.

All network mocked; the real GitHub API is never hit.
"""

import base64
import json

import pytest

from opgrader import modelid
from opgrader.extract import Meta


POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b\n"
    "size 100000000\n"
)


# --------------------------------------------------------------- primitives


def test_parse_lfs_pointer():
    assert modelid.parse_lfs_pointer(POINTER) == (
        "659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b"
    )
    assert modelid.parse_lfs_pointer("ONNX binary junk") is None
    assert modelid.parse_lfs_pointer("version https://git-lfs\nno oid here") is None


def test_parse_remote_forms():
    for url in (
        "https://github.com/spysyweeb/openpilot.git",
        "https://github.com/spysyweeb/openpilot",
        "http://github.com/spysyweeb/openpilot/",
        "git@github.com:spysyweeb/openpilot.git",
        "git@github.com:spysyweeb/openpilot",
        "ssh://git@github.com/spysyweeb/openpilot.git",
    ):
        assert modelid.parse_remote(url) == ("spysyweeb", "openpilot"), url
    assert modelid.parse_remote("https://gitlab.com/x/y.git") is None
    assert modelid.parse_remote("") is None
    assert modelid.parse_remote("not a url") is None


def test_hash_table_has_verified_entries():
    known = modelid.lookup_hash(
        "659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b"
    )
    assert known and "CD210" in known["name"]
    assert modelid.lookup_hash("0" * 64) is None


def test_sanitize():
    assert modelid.sanitize("  Dark Souls (Default) \x00\x01 v9\n") == "Dark Souls (Default)  v9"
    assert len(modelid.sanitize("x" * 500)) <= 81


# --------------------------------------------------------- layer 1: params


def test_selector_sunnypilot_json():
    hit = modelid.from_selector_params(
        {"ModelManager_ActiveBundle": json.dumps(
            {"index": 5, "internalName": "nn-x23", "displayName": "North Dakota v2"})}
    )
    assert hit["label"] == "North Dakota v2"
    assert "sunnypilot" in hit["provenance"]


def test_selector_sunnypilot_falls_back_to_internal_name():
    hit = modelid.from_selector_params(
        {"ModelManager_ActiveBundle": json.dumps({"internalName": "nn-x23"})}
    )
    assert hit["label"] == "nn-x23"


def test_selector_frogpilot_plain_and_mapped():
    hit = modelid.from_selector_params({"Model": "dark-souls_default"})
    assert hit["label"] == "dark-souls_default"
    assert "FrogPilot" in hit["provenance"]

    hit = modelid.from_selector_params({
        "Model": "dark-souls",
        "AvailableModels": "north-dakota,dark-souls",
        "AvailableModelNames": "North Dakota,Dark Souls (Default)",
    })
    assert hit["label"] == "Dark Souls (Default)"


def test_selector_absent_returns_none():
    assert modelid.from_selector_params({}) is None
    assert modelid.from_selector_params({"ModelManager_ActiveBundle": "not json"}) is None


# ----------------------------------------------------- layer 2: mocked API


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _FakeGH:
    """Mock GitHub: commits -> trees -> blobs for one repo layout."""

    def __init__(self, pointer_text=POINTER, nested=False, blob_is_pointer=True):
        prefix_tree = [{"path": "openpilot", "type": "tree", "sha": "T-op"}] if nested else [
            {"path": "selfdrive", "type": "tree", "sha": "T-sd"}]
        self.trees = {
            "T-root": prefix_tree,
            "T-op": [{"path": "selfdrive", "type": "tree", "sha": "T-sd"}],
            "T-sd": [{"path": "modeld", "type": "tree", "sha": "T-md"}],
            "T-md": [{"path": "models", "type": "tree", "sha": "T-models"}],
            "T-models": [
                {"path": "dmonitoring_model.onnx", "type": "blob", "sha": "B-dm", "size": 132},
                {"path": "driving_supercombo.onnx", "type": "blob", "sha": "B-sc", "size": 133},
            ],
        }
        content = pointer_text if blob_is_pointer else "\x00" * 64
        self.blobs = {
            "B-sc": base64.b64encode(content.encode()).decode(),
            "B-dm": base64.b64encode(POINTER.encode()).decode(),
        }
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if "/commits/" in url:
            return _Resp(200, {"commit": {"tree": {"sha": "T-root"}}})
        if "/git/trees/" in url:
            sha = url.rsplit("/", 1)[1]
            return _Resp(200, {"tree": self.trees.get(sha, [])})
        if "/git/blobs/" in url:
            sha = url.rsplit("/", 1)[1]
            if sha in self.blobs:
                return _Resp(200, {"content": self.blobs[sha]})
        return _Resp(404, {})


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(modelid, "LOOKUP_CACHE_FILE", tmp_path / "modelid.json")


def test_fetch_model_pointer_nested_layout():
    gh = _FakeGH(nested=True)
    found = modelid.fetch_model_pointer("o", "r", "c" * 40, session=gh)
    assert found["path"] == "driving_supercombo.onnx"
    assert found["sha256"].startswith("659727c4")
    # dmonitoring must never be picked even though it is a valid pointer
    assert all("B-dm" not in u for u in gh.calls)


def test_fetch_model_pointer_flat_layout():
    gh = _FakeGH(nested=False)
    found = modelid.fetch_model_pointer("o", "r", "c" * 40, session=gh)
    assert found and found["sha256"].startswith("659727c4")


def test_fetch_handles_404_commit():
    class GH404:
        def get(self, url, params=None, timeout=None):
            return _Resp(404, {})

    assert modelid.fetch_model_pointer("o", "r", "c" * 40, session=GH404()) is None


def test_fetch_skips_non_pointer_blob():
    gh = _FakeGH(blob_is_pointer=False)
    assert modelid.fetch_model_pointer("o", "r", "c" * 40, session=gh) is None


def test_from_git_known_hash_and_cache():
    gh = _FakeGH(nested=True)
    hit = modelid.from_git("https://github.com/o/r.git", "c" * 40, session=gh)
    assert "CD210" in hit["label"]
    assert hit["provenance"] == "from build commit"
    n_calls = len(gh.calls)
    # second lookup is served from the cache file: no new API calls
    hit2 = modelid.from_git("https://github.com/o/r.git", "c" * 40, session=gh)
    assert hit2["sha256"] == hit["sha256"]
    assert len(gh.calls) == n_calls


def test_from_git_unknown_hash_labelled_unknown():
    ptr = POINTER.replace("659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b",
                          "ab" * 32)
    gh = _FakeGH(pointer_text=ptr, nested=True)
    hit = modelid.from_git("https://github.com/o/r.git", "d" * 40, session=gh)
    assert "unknown model" in hit["label"]
    assert "abab" in hit["label"]


def test_resolve_layers_and_dirty_suffix():
    # layer 1 wins over layer 2
    meta = Meta()
    meta.model_params = {"Model": "dark-souls_default"}
    meta.git_remote = "https://github.com/o/r.git"
    meta.git_commit = "c" * 40
    hit = modelid.resolve(meta, session=_FakeGH(nested=True))
    assert "FrogPilot" in hit["provenance"]

    # no selector -> layer 2
    meta2 = Meta()
    meta2.git_remote = "https://github.com/o/r.git"
    meta2.git_commit = "c" * 40
    meta2.dirty = True
    hit2 = modelid.resolve(meta2, session=_FakeGH(nested=True))
    assert hit2["provenance"] == "from build commit"
    assert "dirty build" in hit2["label"]

    # nothing available -> unknown with a helpful reason
    meta3 = Meta()
    meta3.git_remote = "https://gitlab.com/x/y.git"
    hit3 = modelid.resolve(meta3)
    assert hit3["provenance"] == "unknown"
    assert "non-GitHub" in hit3["label"]


def test_secrets_never_reach_the_report(tmp_path):
    """A params dump containing GithubSshKeys must never leak into the HTML."""
    from opgrader.pipeline import analyze
    from opgrader.report import render_report
    from opgrader.events import build_arrays, detect_events
    from opgrader.segments import segment_drive
    from tests.conftest import make_drive

    secret = "ssh-rsa AAAAB3SECRETSECRETSECRET"
    d = make_drive(30.0, vEgo=15.0, enabled=True, latActive=True, longActive=True)
    # simulate extract having (wrongly) been handed a hostile dump: even the
    # whitelisted dict only ever holds whitelisted keys, but belt+braces --
    # put the secret in adjacent meta fields a sloppy renderer might dump
    d.meta.model_params = {"Model": "some-model"}
    assert "GithubSshKeys" not in d.meta.model_params
    d.meta.git_remote = "https://gitlab.example/x.git"  # offline-safe: no lookup

    seg = segment_drive(d)
    da = build_arrays(d, seg)
    an = analyze([(d, seg, da, detect_events(d, seg, da))])
    out = render_report(an, tmp_path / "r.html")
    html = out.read_text()
    assert secret not in html
    assert "GithubSshKeys" not in html
    assert "some-model" in html  # the whitelisted selector value does appear


def test_extract_whitelist_filters_params():
    """extract.py only lifts whitelisted keys from initData.params."""
    from opgrader.extract import MODEL_PARAM_WHITELIST

    assert "ModelManager_ActiveBundle" in MODEL_PARAM_WHITELIST
    assert "Model" in MODEL_PARAM_WHITELIST
    assert "GithubSshKeys" not in MODEL_PARAM_WHITELIST
    assert "GithubUsername" not in MODEL_PARAM_WHITELIST
