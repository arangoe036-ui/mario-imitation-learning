"""Parser for BizHawk ``.bk2`` movies.

A .bk2 is a zip archive.  The interesting members are:

``Header.txt``
    Whitespace-separated ``Key Value`` lines: ``Platform NES``, ``SHA1 <rom>``,
    ``Core NesHawk``, ``rerecordCount``, ...

``Input Log.txt``
    ::

        [Input]
        LogKey:#Power|Reset|#P1 Up|P1 Down|...|P1 A|#P2 Up|...|P2 A|
        |..|........|........|
        |..|...R..B.|........|
        [/Input]

    The log key is a ``|``-separated list of button names in which a leading
    ``#`` opens a new *group*.  Each group corresponds to one ``|``-delimited
    field on every frame line, and within a field there is exactly one character
    per button in that group.  The character is the button's mnemonic when held
    and ``.`` when released.

The parser's contract: give it a path, get back a
:class:`~tasdata.movie.Movie` whose ``states`` is a
``(n_frames, n_buttons)`` boolean numpy array.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .formats import (
    CorruptMovieError,
    MovieFormat,
    UnsupportedMovieFormatError,
    sniff,
)
from .movie import Movie

_LOG_KEY_PREFIX = "LogKey:"


class Bk2ParseError(CorruptMovieError):
    """The archive is a .bk2 but its Input Log does not parse."""


@dataclass(frozen=True)
class LogKey:
    """The parsed ``LogKey:`` header line."""

    #: One list of button names per ``|``-delimited field, in field order.
    groups: tuple[tuple[str, ...], ...]
    raw: str

    @property
    def names(self) -> list[str]:
        """All button names flattened in column order."""
        return [n for g in self.groups for n in g]

    @property
    def widths(self) -> tuple[int, ...]:
        """Expected character width of each field."""
        return tuple(len(g) for g in self.groups)

    @classmethod
    def parse(cls, line: str) -> LogKey:
        """Parse ``#Power|Reset|#P1 Up|...|`` into grouped button names."""
        body = line[len(_LOG_KEY_PREFIX):] if line.startswith(_LOG_KEY_PREFIX) else line
        tokens = [t for t in body.split("|") if t != ""]
        if not tokens:
            raise Bk2ParseError(f"empty log key: {line!r}")
        if not tokens[0].startswith("#"):
            raise Bk2ParseError(
                f"log key does not start with a '#' group marker: {line!r}"
            )
        groups: list[list[str]] = []
        for token in tokens:
            if token.startswith("#"):
                groups.append([token[1:]])
            else:
                groups[-1].append(token)
        return cls(groups=tuple(tuple(g) for g in groups), raw=body)


def _read_member(z: zipfile.ZipFile, wanted: str) -> str | None:
    """Case-insensitive zip member read, decoded as UTF-8 (BOM tolerated)."""
    for name in z.namelist():
        if name.lower() == wanted.lower():
            return z.read(name).decode("utf-8-sig", errors="replace")
    return None


def _parse_header(text: str) -> dict[str, str]:
    """``Key Value`` lines -> dict. Keys are unique in practice; last wins."""
    header: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        header[parts[0]] = parts[1].strip() if len(parts) == 2 else ""
    return header


def _parse_input_log(text: str, path: Path) -> tuple[LogKey, np.ndarray]:
    """Parse ``Input Log.txt`` into a log key and a boolean frame matrix."""
    log_key: LogKey | None = None
    frame_lines: list[str] = []

    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line:
            continue
        if line.startswith(_LOG_KEY_PREFIX):
            if log_key is None:
                log_key = LogKey.parse(line)
            continue
        if line.startswith("[") and line.endswith("]"):
            continue  # [Input] / [/Input] section markers
        if line.startswith("|"):
            frame_lines.append(line)
        # Anything else (stray comments) is ignored on purpose.

    if log_key is None:
        raise Bk2ParseError(
            f"{path.name}: no 'LogKey:' line in Input Log.txt. Movies this old "
            "predate the self-describing log format and cannot be parsed without "
            "an out-of-band button list."
        )
    if not frame_lines:
        raise Bk2ParseError(f"{path.name}: Input Log.txt contains no frame rows")

    widths = log_key.widths
    n_buttons = sum(widths)
    states = np.zeros((len(frame_lines), n_buttons), dtype=bool)

    for frame_no, line in enumerate(frame_lines):
        # A frame row is |field|field|...| -- strip the leading and trailing pipe.
        body = line[1:]
        if body.endswith("|"):
            body = body[:-1]
        fields = body.split("|")
        if len(fields) != len(widths):
            raise Bk2ParseError(
                f"{path.name}: frame {frame_no}: log key declares {len(widths)} "
                f"field(s) {widths} but the row has {len(fields)}: {line!r}"
            )
        col = 0
        for field_idx, (chunk, width) in enumerate(zip(fields, widths)):
            if len(chunk) != width:
                raise Bk2ParseError(
                    f"{path.name}: frame {frame_no}: field {field_idx} should be "
                    f"{width} char(s) for buttons {log_key.groups[field_idx]} but "
                    f"is {len(chunk)} ({chunk!r}) in row {line!r}"
                )
            for char in chunk:
                if char not in (".", " ", "\t", "\0"):
                    states[frame_no, col] = True
                col += 1

    return log_key, states


def parse_bk2(path: Path | str, *, allow_tasproj: bool = False) -> Movie:
    """Parse a BizHawk ``.bk2`` movie into a :class:`~tasdata.movie.Movie`.

    Args:
        path: path to the movie. A single outer gzip layer is handled
            transparently, since that is how TASVideos serves user files.
        allow_tasproj: also accept ``.tasproj`` TAStudio projects. They embed the
            same ``Input Log.txt``, so the inputs are readable, but the file also
            carries editor state and may contain a savestate anchor rather than a
            power-on start -- see :func:`starts_from_savestate`.

    Raises:
        UnsupportedMovieFormatError: the file is a different TAS movie format.
        Bk2ParseError: the file is a .bk2 whose contents do not parse.
    """
    path = Path(path)
    result = sniff(path)

    if result.format is MovieFormat.TASPROJ and allow_tasproj:
        pass  # explicitly opted in
    elif result.format is not MovieFormat.BK2:
        hint = ""
        if result.format is MovieFormat.TASPROJ:
            hint = (
                "A .tasproj carries the same 'Input Log.txt' as a .bk2, so you can "
                "opt in with allow_tasproj=True (CLI: --allow-tasproj)."
            )
        raise UnsupportedMovieFormatError(path, result.format, hint)

    with zipfile.ZipFile(io.BytesIO(result.data)) as z:
        input_log = _read_member(z, "Input Log.txt")
        if input_log is None:  # sniff() should have caught this
            raise Bk2ParseError(f"{path.name}: archive has no 'Input Log.txt'")
        header_text = _read_member(z, "Header.txt") or ""
        comments = _read_member(z, "Comments.txt") or ""
        subtitles = _read_member(z, "Subtitles.txt") or ""
        sync_raw = _read_member(z, "SyncSettings.json") or ""
        members = z.namelist()

    try:
        sync_settings = json.loads(sync_raw) if sync_raw.strip() else {}
    except json.JSONDecodeError:
        sync_settings = {"_unparsed": sync_raw}

    log_key, states = _parse_input_log(input_log, path)
    header = _parse_header(header_text)
    header.setdefault("_members", ", ".join(members))

    notes: list[str] = []
    rom_hashes: dict[str, str] = {}
    if header.get("SHA1"):
        rom_hashes["sha1-file"] = header["SHA1"].strip().lower()
    else:
        notes.append("header has no SHA1; ROM identity cannot be verified")

    # BizHawk records region in SyncSettings rather than a top-level header key.
    pal = _detect_pal(header, sync_settings)
    if pal:
        notes.append(
            "sync settings request PAL/50 Hz. nes-py emulates NTSC only, so "
            "timing-sensitive sections may not reproduce."
        )

    anchored = any(
        marker in header["_members"]
        for marker in ("BizState", "CoreState", "Framebuffer", "SaveRam")
    )
    if anchored:
        notes.append(
            "movie is anchored to an embedded savestate, so replaying from "
            "power-on will not reproduce the run"
        )
    if header.get("Core") == "SubNESHawk":
        notes.append(
            "recorded on SubNESHawk (sub-frame input core); nes-py steps whole "
            "frames only and cannot reproduce sub-frame timing"
        )

    return Movie(
        path=path,
        format=MovieFormat.BK2,
        header=header,
        button_names=log_key.names,
        states=states,
        groups=log_key.groups,
        rom_hashes=rom_hashes,
        pal=pal,
        comments=comments.strip(),
        subtitles=subtitles.strip(),
        sync_settings=sync_settings,
        savestate_anchored=anchored,
        notes=notes,
    )


def _detect_pal(header: dict[str, str], sync_settings: dict) -> bool:
    """BizHawk spells the region in a few different places depending on version."""
    if header.get("PAL", "").strip() in ("1", "True", "true"):
        return True
    inner = sync_settings.get("o", sync_settings) if isinstance(sync_settings, dict) else {}
    if isinstance(inner, dict):
        # RegionOverride: 0 = auto, 1 = NTSC, 2 = PAL (BizHawk NES).
        if str(inner.get("RegionOverride", "")) == "2":
            return True
    return False


def starts_from_savestate(movie: Movie) -> bool:
    """True if the movie is anchored to an embedded savestate, not power-on.

    Such movies cannot be replayed from a fresh emulator: the first logged frame
    assumes RAM contents we do not have.
    """
    return movie.savestate_anchored


#: Backwards-compatible alias. The parser now returns the format-neutral
#: :class:`~tasdata.movie.Movie`, which .fm2 movies share.
Bk2Movie = Movie
