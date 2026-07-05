"""Guards against a real Windows bug: text file I/O without an explicit
encoding uses the platform default (UTF-8 on Linux/macOS, but often cp1252
on Windows), which crashed with UnicodeEncodeError the moment the report
contained a non-ASCII character ("throttle<->brake", U+2194) -- see the
opgrader/report.py write_text fix. A static scan is used rather than
simulating a non-UTF8 default at runtime: CPython's text-mode I/O resolves
its default encoding at the C level (sys.getfilesystemencoding /
io.text_encoding), which does not consult the Python-level `locale` module,
so monkeypatching locale.getpreferredencoding does not actually force a
different default in a portable test.
"""

import ast
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent / "opgrader"


def _iter_unsafe_calls():
    for path in sorted(PKG_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                name = fn.attr
                owner = fn.value.id if isinstance(fn.value, ast.Name) else None
                # os.open() is the low-level fd-based open (flags, not text
                # mode/encoding); webbrowser.open() opens a URL, not a file.
                if name == "open" and owner in ("os", "webbrowser"):
                    continue
            elif isinstance(fn, ast.Name):
                name = fn.id
            else:
                continue
            if name not in ("write_text", "read_text", "fdopen", "open"):
                continue
            has_encoding_kw = any(kw.arg == "encoding" for kw in node.keywords)
            if not has_encoding_kw:
                # binary mode ("...b...") doesn't need/accept encoding=
                mode_arg = node.args[1] if len(node.args) > 1 else None
                mode = mode_arg.value if isinstance(mode_arg, ast.Constant) else ""
                if isinstance(mode, str) and "b" in mode:
                    continue
                yield f"{path.name}:{node.lineno}: {name}(...) with no explicit encoding"


def test_no_text_io_without_explicit_encoding():
    offenders = list(_iter_unsafe_calls())
    assert not offenders, (
        "Text file I/O must always pass encoding=\"utf-8\" explicitly -- "
        "relying on the platform default breaks on Windows:\n" + "\n".join(offenders)
    )


def test_report_help_text_survives_utf8_round_trip():
    """The actual character that crashed, straight from opgrader/report.py."""
    from opgrader.report import CATEGORY_HELP

    what, _how = CATEGORY_HELP["Smoothness"]
    assert "↔" in what  # throttle<->brake
    assert what.encode("utf-8").decode("utf-8") == what
