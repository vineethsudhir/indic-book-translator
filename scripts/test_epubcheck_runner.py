"""Tests for ``kannada_epub.epubcheck_runner`` (optional EPUBCheck integration).

Everything runs offline: parsing uses captured EPUBCheck output strings and
``subprocess.run``/``shutil.which`` are monkeypatched. A real Java run happens
only when ``java`` is on PATH and ``.omc/epubcheck/epubcheck.jar`` exists
(git-ignored); otherwise it is skipped with a note.

Run: .venv/bin/python scripts/test_epubcheck_runner.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kannada_epub import epubcheck_runner  # noqa: E402
from kannada_epub.epubcheck_runner import (  # noqa: E402
    EPUBCHECK_JAR_ENV,
    EpubcheckResult,
    find_epubcheck,
    inherited_errors,
    new_errors,
    parse_epubcheck_output,
    run_epubcheck,
)

SHERLOCK = ROOT / "data" / "sherlock_holmes.epub"
REAL_JAR = ROOT / ".omc" / "epubcheck" / "epubcheck.jar"

# Captured from `java -jar epubcheck.jar` (clean book, no version banner).
CLEAN_OUTPUT = """\
Validating using EPUB version 3.3 rules.
No errors or warnings detected.
Messages: 0 fatals / 0 errors / 0 warnings / 0 infos

EPUBCheck completed
"""

# Captured shape with a version banner, two errors and a warning.
ERROR_OUTPUT = """\
EPUBCheck v5.1.0

Validating using EPUB version 3.3 rules.
WARNING(OPF-003): /tmp/library/book.epub/EPUB/content.opf(4,7): Item "nav" is not referenced.
ERROR(RSC-005): /tmp/library/book.epub/EPUB/ch1.xhtml(1,7): Error while parsing file: element "body" not allowed yet.
ERROR(RSC-005): /tmp/library/book.epub/EPUB/ch2.xhtml(1,90): Error while parsing file: Duplicate ID "x".
Messages: 0 fatals / 2 errors / 1 warnings / 0 infos

EPUBCheck completed
"""


class _Patcher:
    """Tiny context manager to set/restore attributes without pytest."""

    def __init__(self, obj, **attributes):
        self._obj = obj
        self._attributes = attributes
        self._saved: dict[str, object] = {}

    def __enter__(self):
        for name, value in self._attributes.items():
            self._saved[name] = getattr(self._obj, name)
            setattr(self._obj, name, value)
        return self

    def __exit__(self, *_exc):
        for name, value in self._saved.items():
            setattr(self._obj, name, value)
        return False


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parsing() -> None:
    clean = parse_epubcheck_output(CLEAN_OUTPUT)
    assert clean == EpubcheckResult(0, 0, 0, [], None), clean

    result = parse_epubcheck_output(ERROR_OUTPUT)
    assert (result.fatals, result.errors, result.warnings) == (0, 2, 1), result
    assert result.version == "5.1.0", result.version
    assert result.messages == [
        'WARNING(OPF-003): EPUB/content.opf(4,7): Item "nav" is not referenced.',
        'ERROR(RSC-005): EPUB/ch1.xhtml(1,7): Error while parsing file: element "body" not allowed yet.',
        'ERROR(RSC-005): EPUB/ch2.xhtml(1,90): Error while parsing file: Duplicate ID "x".',
    ], result.messages

    # Paths are made relative to the archive, and Windows separators normalise.
    windows = parse_epubcheck_output(
        "ERROR(RSC-005): C:\\Users\\me\\book.epub\\EPUB\\ch1.xhtml(2,3): boom\n"
        "Messages: 0 fatals / 1 errors / 0 warnings / 0 infos\n"
    )
    assert windows.messages == ["ERROR(RSC-005): EPUB/ch1.xhtml(2,3): boom"], \
        windows.messages

    # At most 50 message lines are kept; the summary counts stay authoritative.
    many = "".join(
        f"ERROR(RSC-005): /tmp/book.epub/EPUB/ch{i}.xhtml(1,1): e{i}\n"
        for i in range(60)
    ) + "Messages: 0 fatals / 60 errors / 0 warnings / 0 infos\n"
    capped = parse_epubcheck_output(many)
    assert len(capped.messages) == 50, len(capped.messages)
    assert capped.errors == 60, capped.errors

    # No summary line: the run cannot be trusted.
    exc = assert_raises(
        RuntimeError, parse_epubcheck_output, "EPUBCheck exploded\nsecond line"
    )
    assert "second line" in str(exc), exc


# ---------------------------------------------------------------------------
# find_epubcheck
# ---------------------------------------------------------------------------
def _make_jar(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jar")
    return path


def test_find_epubcheck() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="epubcheck-find-"))
    saved_env = os.environ.get(EPUBCHECK_JAR_ENV)
    saved_data = os.environ.get("KANNADA_APP_DATA_DIR")
    try:
        os.environ["KANNADA_APP_DATA_DIR"] = str(tmp)
        jar = _make_jar(tmp / "epubcheck" / "epubcheck.jar")

        # Java present + default data-dir jar present.
        with _Patcher(epubcheck_runner.shutil, which=lambda _name: "/usr/bin/java"):
            found = find_epubcheck()
        assert found == ("/usr/bin/java", jar), found

        # Java missing -> None even though the jar exists.
        with _Patcher(epubcheck_runner.shutil, which=lambda _name: None):
            assert find_epubcheck() is None

        # Env var points at an existing jar and wins over the data dir.
        other = _make_jar(tmp / "elsewhere" / "my-epubcheck.jar")
        os.environ[EPUBCHECK_JAR_ENV] = str(other)
        with _Patcher(epubcheck_runner.shutil, which=lambda _name: "/usr/bin/java"):
            assert find_epubcheck() == ("/usr/bin/java", other)
        os.environ.pop(EPUBCHECK_JAR_ENV, None)

        # Jar missing altogether -> None.
        jar.unlink()
        with _Patcher(epubcheck_runner.shutil, which=lambda _name: "/usr/bin/java"):
            assert find_epubcheck() is None
    finally:
        if saved_env is None:
            os.environ.pop(EPUBCHECK_JAR_ENV, None)
        else:
            os.environ[EPUBCHECK_JAR_ENV] = saved_env
        if saved_data is None:
            os.environ.pop("KANNADA_APP_DATA_DIR", None)
        else:
            os.environ["KANNADA_APP_DATA_DIR"] = saved_data
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# run_epubcheck
# ---------------------------------------------------------------------------
def test_run_epubcheck() -> None:
    epub = Path("/tmp/some book.epub")
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=ERROR_OUTPUT, stderr="")

    with _Patcher(
        epubcheck_runner, find_epubcheck=lambda: ("/usr/bin/java", Path("/tmp/epubcheck.jar"))
    ), _Patcher(epubcheck_runner.subprocess, run=fake_run):
        result = run_epubcheck(epub, timeout=42)

    assert (result.fatals, result.errors, result.warnings) == (0, 2, 1), result
    argv, kwargs = calls[0]
    assert isinstance(argv, list), argv
    assert argv == ["/usr/bin/java", "-jar", "/tmp/epubcheck.jar", str(epub)], argv
    assert kwargs["capture_output"] is True, kwargs
    assert kwargs["text"] is True, kwargs
    assert kwargs["timeout"] == 42, kwargs
    assert kwargs["check"] is False, kwargs
    assert not kwargs.get("shell"), kwargs

    # No summary -> RuntimeError.
    def fake_no_summary(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="boom\n", stderr="")

    with _Patcher(
        epubcheck_runner, find_epubcheck=lambda: ("/usr/bin/java", Path("/tmp/epubcheck.jar"))
    ), _Patcher(epubcheck_runner.subprocess, run=fake_no_summary):
        exc = assert_raises(RuntimeError, run_epubcheck, epub)
    assert "boom" in str(exc), exc

    # Timeout -> RuntimeError mentioning the timeout.
    def fake_timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 1))

    with _Patcher(
        epubcheck_runner, find_epubcheck=lambda: ("/usr/bin/java", Path("/tmp/epubcheck.jar"))
    ), _Patcher(epubcheck_runner.subprocess, run=fake_timeout):
        exc = assert_raises(RuntimeError, run_epubcheck, epub, timeout=7)
    assert "timed out" in str(exc) and "7" in str(exc), exc

    # Not installed -> RuntimeError.
    with _Patcher(epubcheck_runner, find_epubcheck=lambda: None):
        exc = assert_raises(RuntimeError, run_epubcheck, epub)
    assert "not installed" in str(exc), exc


# ---------------------------------------------------------------------------
# inherited vs new
# ---------------------------------------------------------------------------
def test_inherited_errors() -> None:
    source = EpubcheckResult(
        0,
        1,
        0,
        [
            "ERROR(RSC-005): EPUB/ch1.xhtml(3,9): Error while parsing file: "
            'Duplicate ID "x".'
        ],
        None,
    )
    output = EpubcheckResult(
        1,
        1,
        0,
        [
            "FATAL(RSC-016): EPUB/ch1.xhtml(1,26): Fatal Error while parsing file: "
            'The element type "p" must be terminated.',
            "ERROR(RSC-005): EPUB/ch1.xhtml(1,7): Error while parsing file: "
            'Duplicate ID "x".',
        ],
        None,
    )
    # Same code, file and text (line/column ignored) counts as inherited.
    assert inherited_errors(output, source) == 1
    assert new_errors(output, source) == 1

    # The reverse comparison: the source's one error is also in the output.
    assert inherited_errors(source, output) == 1
    assert new_errors(source, output) == 0


# ---------------------------------------------------------------------------
# a real Java run, when available
# ---------------------------------------------------------------------------
def test_real_run() -> None:
    java = shutil.which("java")
    if not java or not REAL_JAR.is_file():
        print("test_epubcheck_runner: skipping real run (java or "
              f"{REAL_JAR} missing)")
        return
    saved = os.environ.get(EPUBCHECK_JAR_ENV)
    os.environ[EPUBCHECK_JAR_ENV] = str(REAL_JAR)
    try:
        result = run_epubcheck(SHERLOCK)
    finally:
        if saved is None:
            os.environ.pop(EPUBCHECK_JAR_ENV, None)
        else:
            os.environ[EPUBCHECK_JAR_ENV] = saved
    assert result.fatals == 0, result
    assert result.errors == 0, result
    assert result.messages == [], result.messages


def test_paths_are_relative() -> None:
    """Source and output messages share keys despite different file names."""
    from kannada_epub.epubcheck_runner import _relative_paths

    template = (
        "ERROR(PKG-021): {p}/OPS/images/x(-1,-1): Corrupted image file encountered.\n"
        "ERROR(PKG-006): {p}(-1,-1): Mimetype file entry is missing.\n"
        "Messages: 0 fatals / 2 errors / 0 warnings / 0 infos\n"
    )
    src = Path("/home/someone/Books/sultana.epub")
    out = Path("/home/someone/out/sultana.kn.epub")
    a = parse_epubcheck_output(_relative_paths(template.format(p=out), out))
    b = parse_epubcheck_output(_relative_paths(template.format(p=src), src))
    assert all("someone" not in m for m in a.messages + b.messages), a.messages
    assert a.messages[0].startswith("ERROR(PKG-021): OPS/images/x"), a.messages
    assert "EPUB(-1,-1)" in a.messages[1], a.messages
    assert new_errors(a, b) == 0 and inherited_errors(a, b) == 2


def main() -> None:
    test_paths_are_relative()
    test_parsing()
    test_find_epubcheck()
    test_run_epubcheck()
    test_inherited_errors()
    test_real_run()
    print("test_epubcheck_runner: all assertions passed")


if __name__ == "__main__":
    main()
