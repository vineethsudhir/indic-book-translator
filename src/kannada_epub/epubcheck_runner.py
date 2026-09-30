"""Optional integration with W3C EPUBCheck (the reference EPUB validator).

EPUBCheck is a Java program: ``java -jar epubcheck.jar book.epub``. This module
runs it when a jar is available and turns its text output into a small result
object the pipeline and app can show. It never downloads anything and never
raises for validation findings (only for a missing summary or a timeout).

Only the external Java validator is run here — no Python subprocess and no
``scripts/`` (AGENTS.md): invoking the real validator is the module's purpose.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

EPUBCHECK_JAR_ENV = "KANNADA_EPUBCHECK_JAR"

# Keep the manifest/UI small; the counts on the summary line are authoritative.
_MAX_MESSAGES = 50

_SUMMARY_RE = re.compile(
    r"Messages:\s*(\d+)\s+fatals?\s*/\s*(\d+)\s+errors?\s*/\s*(\d+)\s+warnings?"
    r"\s*/\s*\d+\s+infos?",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"EPUBCheck v(\S+)")
_MESSAGE_RE = re.compile(r"^(FATAL|ERROR|WARNING)\(([^)]+)\):\s*(.*)$")
_LOCATION_RE = re.compile(r"^(.*?)\((\d+),(\d+)\):\s*(.*)$", re.DOTALL)
_ERROR_PREFIXES = ("FATAL(", "ERROR(")


@dataclass(frozen=True)
class EpubcheckResult:
    """EPUBCheck's verdict for one EPUB."""

    fatals: int
    errors: int
    warnings: int
    messages: list[str]
    version: str | None


def _relative_path(raw: str) -> str:
    """Turn ``/abs/book.epub/EPUB/ch1.xhtml`` into ``EPUB/ch1.xhtml``.

    EPUBCheck prefixes every message with the archive path it was given. The
    first ``.epub`` path segment is the archive; everything after it is the
    in-archive path, which is stable across machines and safe to show.
    """
    match = re.match(r"^.*?\.epub[/\\](.*)$", raw, re.IGNORECASE)
    if match:
        raw = match.group(1)
    return raw.replace("\\", "/")


def parse_epubcheck_output(text: str) -> EpubcheckResult:
    """Parse EPUBCheck's console output.

    Counts come from the ``Messages: N fatals / N errors / N warnings / N infos``
    summary line; a missing summary means the run failed, so raise
    ``RuntimeError`` with the tail of the output. ``FATAL``/``ERROR``/``WARNING``
    lines become display messages (up to :data:`_MAX_MESSAGES`), with the
    archive path stripped off the file paths.
    """
    summary = _SUMMARY_RE.search(text)
    if summary is None:
        tail = "\n".join(text.strip().splitlines()[-5:])
        raise RuntimeError(
            "EPUBCheck did not report a summary line; last output:\n" + tail
        )
    fatals, errors, warnings = (int(group) for group in summary.groups()[:3])

    version_match = _VERSION_RE.search(text)
    version = version_match.group(1) if version_match else None

    messages: list[str] = []
    for raw_line in text.splitlines():
        match = _MESSAGE_RE.match(raw_line.strip())
        if match is None:
            continue
        severity, code, rest = match.groups()
        location = _LOCATION_RE.match(rest)
        if location is None:
            messages.append(f"{severity}({code}): {rest.strip()}")
        else:
            path, line_number, column, message = location.groups()
            messages.append(
                f"{severity}({code}): {_relative_path(path.strip())}"
                f"({line_number},{column}): {message.strip()}"
            )
        if len(messages) >= _MAX_MESSAGES:
            break

    return EpubcheckResult(fatals, errors, warnings, messages, version)


def find_epubcheck() -> tuple[str, Path] | None:
    """``(java executable, jar path)`` when EPUBCheck can run, else ``None``.

    The jar is ``KANNADA_EPUBCHECK_JAR`` when set, otherwise
    ``<app data dir>/epubcheck/epubcheck.jar``. The app ``paths`` module is
    imported lazily so the core package does not depend on the app.
    """
    env_jar = os.environ.get(EPUBCHECK_JAR_ENV)
    if env_jar:
        jar = Path(env_jar)
    else:
        from .app.paths import data_dir  # lazy: core must not need the app

        jar = data_dir() / "epubcheck" / "epubcheck.jar"
    java = shutil.which("java")
    if java is None or not jar.is_file():
        return None
    return java, jar


def run_epubcheck(epub_path: Path, *, timeout: float = 180) -> EpubcheckResult:
    """Validate ``epub_path`` with EPUBCheck and parse the result.

    Raises ``RuntimeError`` if EPUBCheck is not installed, produced no summary,
    or timed out. Validation findings themselves are returned, not raised.
    """
    found = find_epubcheck()
    if found is None:
        raise RuntimeError("EPUBCheck is not installed (Java or the jar is missing).")
    java, jar = found
    # AGENTS.md forbids Python/subprocess scripts, not the external validator
    # this module exists to run; a list argv keeps it shell-injection safe.
    try:
        completed = subprocess.run(
            [java, "-jar", str(jar), str(epub_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"EPUBCheck timed out after {timeout} seconds") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    return parse_epubcheck_output(_relative_paths(output, epub_path))


def _relative_paths(output: str, epub_path: Path) -> str:
    """Strip the EPUB's own location from EPUBCheck's messages.

    EPUBCheck prefixes every location with the path of the file it checked
    (``/…/book.kn.epub/OPS/ch1.xhtml``). Removing it keeps private folder
    names out of the manifest and the UI, and lets a source and an output
    with different file names share message keys, so inherited errors are
    recognised. A message about the file itself names it ``EPUB``.
    """
    spellings = {str(epub_path), str(Path(epub_path).resolve()), str(Path(epub_path).absolute())}
    for spelling in sorted(spellings, key=len, reverse=True):
        output = output.replace(spelling + "/", "").replace(spelling, "EPUB")
    return output


def message_key(message: str) -> str:
    """A message's identity ignoring line/column numbers.

    Used to tell an error inherited from the source EPUB apart from one the
    translation introduced: ``ERROR(RSC-005): EPUB/ch1.xhtml(1,7): text`` and
    ``... (2,9): text`` share a key.
    """
    match = _MESSAGE_RE.match(message.strip())
    if match is None:
        return message.strip()
    severity, code, rest = match.groups()
    location = _LOCATION_RE.match(rest)
    if location is None:
        return f"{severity}({code}): {rest.strip()}"
    path, _line, _column, text = location.groups()
    return f"{severity}({code}): {path.strip()}: {text.strip()}"


def inherited_errors(output: EpubcheckResult, source: EpubcheckResult) -> int:
    """Fatal+error messages in ``output`` whose key also appears in ``source``."""
    source_keys = {message_key(message) for message in source.messages}
    return sum(
        1
        for message in output.messages
        if message.startswith(_ERROR_PREFIXES) and message_key(message) in source_keys
    )


def new_errors(output: EpubcheckResult, source: EpubcheckResult) -> int:
    """Fatal+error messages in ``output`` that are not in ``source``."""
    source_keys = {message_key(message) for message in source.messages}
    return sum(
        1
        for message in output.messages
        if message.startswith(_ERROR_PREFIXES) and message_key(message) not in source_keys
    )
