"""Decompress rlog segments and iterate capnp events.

Segments are capnp streams compressed with zstd (.zst, modern) or bz2
(.bz2, pre-2024). Decoded with standalone pycapnp against the vendored
schemas -- no openpilot checkout or build required.
"""

from __future__ import annotations

import bz2
import os
import re
from pathlib import Path
from typing import Iterator

import capnp

_SCHEMAS = Path(__file__).resolve().parent / "schemas"
_log_schema = None


def load_log_schema():
    """Load (and cache) the vendored log.capnp schema."""
    global _log_schema
    if _log_schema is None:
        capnp.remove_import_hook()
        _log_schema = capnp.load(
            str(_SCHEMAS / "cereal" / "log.capnp"),
            imports=[str(_SCHEMAS / "cereal"), str(_SCHEMAS / "car")],
        )
    return _log_schema


def decompress(path: str | Path) -> bytes:
    """Return the raw capnp stream bytes for a segment file."""
    path = Path(path)
    raw = path.read_bytes()
    name = path.name.lower()
    if name.endswith(".bz2"):
        return bz2.decompress(raw)
    if name.endswith(".zst"):
        import zstandard

        # rlogs may be written without a content-size header; stream-decompress.
        dctx = zstandard.ZstdDecompressor()
        return dctx.decompress(raw, max_output_size=2 * 1024 * 1024 * 1024)
    # magic-byte sniffing for extensionless files
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:4] == b"\x28\xb5\x2f\xfd":
        import zstandard

        return zstandard.ZstdDecompressor().decompress(
            raw, max_output_size=2 * 1024 * 1024 * 1024
        )
    return raw  # assume uncompressed capnp stream


def read_events(path: str | Path) -> Iterator[tuple[float, str, object]]:
    """Yield (logMonoTime_seconds, which, event) for every event in a segment.

    Events that fail to decode individually are skipped (truncated segments
    are common when a drive ends mid-write).
    """
    schema = load_log_schema()
    data = decompress(path)
    try:
        it = schema.Event.read_multiple_bytes(data)
        while True:
            try:
                e = next(it)
            except StopIteration:
                return
            try:
                yield e.logMonoTime * 1e-9, e.which(), e
            except capnp.KjException:
                continue
    except capnp.KjException:
        # truncated tail -- keep whatever we already yielded
        return


_SEG_NUM_RE = re.compile(r"(\d+)")


def _natural_key(p: Path):
    """Sort key that orders embedded integers numerically."""
    parts = []
    for piece in p.parts:
        parts.extend(
            int(tok) if tok.isdigit() else tok
            for tok in _SEG_NUM_RE.split(piece)
        )
    return parts


def _looks_like_rlog(p: Path) -> bool:
    n = p.name.lower()
    if not (n.endswith(".bz2") or n.endswith(".zst")):
        return "rlog" in n and p.is_file()
    # qlogs are decimated; explicitly skip them
    if "qlog" in n:
        return False
    return True


def find_segments(inputs: list[str]) -> list[Path]:
    """Expand files/dirs/globs into an ordered list of segment files."""
    out: list[Path] = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            found = [q for q in p.rglob("*") if q.is_file() and _looks_like_rlog(q)]
            out.extend(sorted(found, key=_natural_key))
        elif p.is_file():
            out.append(p)
        else:
            import glob as _glob

            matches = [Path(m) for m in _glob.glob(str(p))]
            out.extend(sorted((m for m in matches if m.is_file()), key=_natural_key))
    # de-dup, preserve order
    seen: set[Path] = set()
    uniq = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def group_segments(paths: list[Path]) -> list[list[Path]]:
    """Group segment files into routes.

    Segments of one route share a directory and a filename prefix once the
    trailing segment number is stripped (e.g. rlog_00.bz2 / rlog_01.bz2, or
    route--0--rlog.zst / route--1--rlog.zst share 'route' with different
    segment indices). Grouping key: parent dir + filename with digit runs
    removed. Within a group, natural sort order is segment order.
    """
    groups: dict[tuple, list[Path]] = {}
    for p in paths:
        key = (str(p.parent), _SEG_NUM_RE.sub("#", p.name))
        groups.setdefault(key, []).append(p)
    result = [sorted(v, key=_natural_key) for v in groups.values()]
    result.sort(key=lambda g: str(g[0]))
    return result


def route_name_for_group(group: list[Path]) -> str:
    """Human-readable name for a group of segments."""
    p = group[0]
    parent = p.parent.name or str(p.parent)
    if os.sep in str(p) and parent not in (".", ""):
        return parent
    return p.stem
