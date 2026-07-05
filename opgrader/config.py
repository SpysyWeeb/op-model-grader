"""Persistent user config (~/.config/opgrader/config.json).

Currently holds the personality follow-distance targets (t_follow, seconds)
used by the follow-adherence metric. Defaults are stock openpilot's; forks
tune these, so they are overridable via the config file, the GUI boxes, or
the CLI --t-follow flag (flag wins over file).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_FILE = Path(
    os.environ.get("OPGRADER_CONFIG", "~/.config/opgrader/config.json")
).expanduser()

# stock openpilot longitudinal MPC follow targets (seconds)
DEFAULT_T_FOLLOW = {"aggressive": 1.25, "standard": 1.45, "relaxed": 1.75}

PERSONALITIES = tuple(DEFAULT_T_FOLLOW)


def load_config() -> dict:
    try:
        data = json.loads(CONFIG_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(data: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, CONFIG_FILE)


def get_t_follow() -> dict[str, float]:
    """Targets from the config file, with stock defaults filling gaps."""
    out = dict(DEFAULT_T_FOLLOW)
    stored = load_config().get("t_follow")
    if isinstance(stored, dict):
        for k, v in stored.items():
            k = str(k).strip().lower()
            if k in out:
                try:
                    val = float(v)
                    if 0.3 <= val <= 5.0:
                        out[k] = val
                except (TypeError, ValueError):
                    pass
    return out


def set_t_follow(targets: dict[str, float]) -> None:
    """Persist targets (only known personalities, sane range)."""
    clean = {}
    for k, v in targets.items():
        k = str(k).strip().lower()
        if k in DEFAULT_T_FOLLOW:
            try:
                val = float(v)
                if 0.3 <= val <= 5.0:
                    clean[k] = val
            except (TypeError, ValueError):
                pass
    data = load_config()
    data["t_follow"] = {**data.get("t_follow", {}), **clean} if isinstance(
        data.get("t_follow"), dict
    ) else clean
    save_config(data)


def parse_t_follow_flag(text: str) -> dict[str, float]:
    """Leniently parse "aggressive=1.0, standard=1.45,relaxed = 2.0".

    Accepts comma/semicolon separators, spaces, ':' or '=', case-insensitive
    keys, and unambiguous key prefixes ("agg=1.0"). Unknown/garbled parts
    raise ValueError with a helpful message.
    """
    out: dict[str, float] = {}
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        sep = "=" if "=" in part else (":" if ":" in part else None)
        if sep is None:
            raise ValueError(f"--t-follow: can't parse {part!r} (want name=seconds)")
        key, _, val = part.partition(sep)
        key = key.strip().lower()
        matches = [p for p in PERSONALITIES if p.startswith(key)] if key else []
        if len(matches) != 1:
            raise ValueError(
                f"--t-follow: unknown personality {key!r} "
                f"(want one of {', '.join(PERSONALITIES)})"
            )
        try:
            fval = float(val.strip())
        except ValueError:
            raise ValueError(f"--t-follow: bad number {val.strip()!r} for {matches[0]}")
        if not 0.3 <= fval <= 5.0:
            raise ValueError(f"--t-follow: {matches[0]}={fval} outside sane range 0.3-5.0 s")
        out[matches[0]] = fval
    return out


def resolve_t_follow(flag_text: str | None = None) -> dict[str, float]:
    """Config-file targets, overridden by the CLI flag when given."""
    targets = get_t_follow()
    if flag_text:
        targets.update(parse_t_follow_flag(flag_text))
    return targets
