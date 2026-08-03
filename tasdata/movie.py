"""The format-neutral movie representation shared by the parsers.

``.bk2`` and ``.fm2`` disagree about almost everything -- container, header keys,
button order, ROM fingerprint algorithm -- but once parsed they carry the same
information: a per-frame boolean button matrix plus enough metadata to check that
you are about to replay it against the right ROM.  Everything downstream
(:mod:`tasdata.replay`, :mod:`tasdata.verify`) works on this class only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .formats import MovieFormat
from .rom import NesRom, RomCheck, load_rom

#: NES frame rates, used only to render human-readable durations.
FPS_NTSC = 60.0988
FPS_PAL = 50.007


@dataclass
class Movie:
    """A parsed TAS movie, independent of its on-disk format."""

    path: Path
    format: MovieFormat
    #: Raw header key/values as the file spelled them.
    header: dict[str, str]
    #: Column names for ``states``, e.g. ``["Reset", "Power", "P1 Right", ...]``.
    button_names: list[str]
    #: bool array, shape ``(n_frames, n_buttons)``.
    states: np.ndarray
    #: Button-name groups, mirroring the file's field layout.
    groups: tuple[tuple[str, ...], ...] = ()
    #: Recorded ROM fingerprints, keyed by algorithm: ``sha1-file``, ``md5-prgchr``.
    rom_hashes: dict[str, str] = field(default_factory=dict)
    #: True when the movie was recorded in PAL/50 Hz mode.
    pal: bool = False
    comments: str = ""
    subtitles: str = ""
    sync_settings: dict = field(default_factory=dict)
    #: True when the first logged frame assumes a savestate, not power-on.
    savestate_anchored: bool = False
    #: Parser observations worth showing the user.
    notes: list[str] = field(default_factory=list)

    # -- basic properties ------------------------------------------------- #

    @property
    def n_frames(self) -> int:
        return int(self.states.shape[0])

    @property
    def fps(self) -> float:
        return FPS_PAL if self.pal else FPS_NTSC

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / self.fps

    @property
    def platform(self) -> str:
        return self.header.get("Platform", "NES")

    @property
    def core(self) -> str:
        return self.header.get("Core", self.header.get("emuVersion", "?"))

    @property
    def game_name(self) -> str:
        return self.header.get("GameName") or self.header.get("romFilename") or "?"

    @property
    def author(self) -> str:
        return self.header.get("Author") or self.header.get("author") or "?"

    @property
    def rerecord_count(self) -> str:
        return self.header.get("rerecordCount", "?")

    def summary(self) -> str:
        secs = self.duration_seconds
        region = "PAL" if self.pal else "NTSC"
        return (
            f"{self.path.name}: {self.game_name!r} by {self.author} "
            f"[{self.format.value}/{self.platform}/{self.core}/{region}] "
            f"{self.n_frames} frames ({int(secs // 60)}m{secs % 60:04.1f}s), "
            f"{len(self.button_names)} buttons"
        )

    # -- ROM verification ------------------------------------------------- #

    def verify_rom(self, rom: NesRom | Path | str) -> RomCheck:
        """Compare this movie's recorded ROM fingerprint against a real ROM.

        Uses whichever algorithm the movie's format recorded: ``sha1-file`` for
        .bk2, ``md5-prgchr`` for .fm2.
        """
        if not isinstance(rom, NesRom):
            rom = load_rom(rom)
        candidates = (
            ("md5-prgchr", rom.md5_prgchr),
            ("sha1-file", rom.sha1_file),
        )
        for algorithm, actual in candidates:
            expected = self.rom_hashes.get(algorithm)
            if not expected:
                continue
            expected = expected.lower()
            matched = expected == actual.lower()
            detail = "" if matched else (
                f"movie was recorded against a different dump of the game "
                f"({algorithm} {expected} vs {actual})"
            )
            return RomCheck(True, matched, algorithm, expected, actual, detail)
        return RomCheck(
            False,
            None,
            detail=f"{self.format.value} header carries no ROM fingerprint",
        )

    # -- serialisation ---------------------------------------------------- #

    def to_dict(self) -> dict:
        """JSON-safe metadata (without the frame array)."""
        return {
            "path": str(self.path),
            "format": self.format.value,
            "header": self.header,
            "button_names": self.button_names,
            "groups": [list(g) for g in self.groups],
            "n_frames": self.n_frames,
            "n_buttons": len(self.button_names),
            "pal": self.pal,
            "rom_hashes": self.rom_hashes,
            "savestate_anchored": self.savestate_anchored,
            "presses_per_button": {
                name: int(self.states[:, i].sum())
                for i, name in enumerate(self.button_names)
            },
            "sync_settings": self.sync_settings,
            "comments": self.comments,
            "notes": self.notes,
        }


def parse_movie(path: Path | str, *, allow_tasproj: bool = False) -> Movie:
    """Parse any supported movie, dispatching on sniffed format.

    Supported: BizHawk ``.bk2`` and FCEUX ``.fm2`` (plus ``.tasproj`` on request).
    Anything else raises :class:`~tasdata.formats.UnsupportedMovieFormatError`.
    """
    from .bk2 import parse_bk2
    from .fm2 import parse_fm2
    from .formats import UnsupportedMovieFormatError, sniff

    result = sniff(path)
    if result.format is MovieFormat.FM2:
        return parse_fm2(path)
    if result.format is MovieFormat.BK2 or (
        result.format is MovieFormat.TASPROJ and allow_tasproj
    ):
        return parse_bk2(path, allow_tasproj=allow_tasproj)
    hint = ""
    if result.format is MovieFormat.TASPROJ:
        hint = (
            "A .tasproj carries the same 'Input Log.txt' as a .bk2, so you can "
            "opt in with allow_tasproj=True (CLI: --allow-tasproj)."
        )
    raise UnsupportedMovieFormatError(path, result.format, hint)
