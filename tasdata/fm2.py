"""Parser for FCEUX ``.fm2`` movies.

An fm2 is plain text: a block of ``key value`` header lines followed by one line
per frame, each starting with ``|``::

    version 3
    emuVersion 22020
    rerecordCount 5804
    palFlag 1
    romFilename Super Mario Bros. (Europe) (Rev 0A)
    romChecksum base64:ujnd5jqyCbG8dR4FNecrGA==
    guid 8DB0C859-DDAB-BB94-95D3-1E4302ECAD1B
    fourscore 0
    port0 1
    port1 0
    port2 0
    |0|........|||
    |0|R...T...|||

Frame lines are ``|commands|port0|port1|port2|``.  ``commands`` is a small
integer bitmask (soft reset, power, FDS actions); each controller field is eight
characters in the fixed order ``RLDUTSBA`` -- Right, Left, Down, Up, sTart,
Select, B, A -- where ``.`` means released and the letter (or any other
character) means held.  Unconnected ports are empty strings, which is why an
ordinary one-player movie ends in ``|||``.

Two header fields decide whether a replay can possibly sync, so both are surfaced
as first-class fields rather than left in the header dict:

``romChecksum``
    MD5 of the ROM's PRG + CHR data with the 16-byte iNES header stripped.
    :meth:`~tasdata.movie.Movie.verify_rom` checks it against the supplied ROM.

``palFlag``
    1 means the run was recorded at 50 Hz. nes-py only emulates NTSC, so a PAL
    movie is flagged loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .buttons import is_pressed
from .formats import CorruptMovieError, MovieFormat, UnsupportedMovieFormatError, sniff
from .movie import Movie
from .rom import decode_fm2_checksum

#: Character order of an fm2 controller field. Position i is bit (7 - i) of the
#: NES joypad byte, which is exactly nes-py's action-byte layout.
FM2_BUTTON_ORDER: tuple[str, ...] = (
    "Right", "Left", "Down", "Up", "Start", "Select", "B", "A",
)

#: Width of one controller field.
CONTROLLER_WIDTH = len(FM2_BUTTON_ORDER)

#: Bits of the leading ``commands`` field (FCEUX ``MOVIECMD_*``).
COMMAND_BITS: tuple[tuple[int, str], ...] = (
    (0x01, "Reset"),
    (0x02, "Power"),
    (0x04, "FDS Insert"),
    (0x08, "FDS Select"),
    (0x10, "VS Coin"),
)

#: Header keys that may appear more than once and are joined instead of overwritten.
_MULTI_KEYS = frozenset({"comment", "subtitle"})


class Fm2ParseError(CorruptMovieError):
    """The file is an fm2 but its contents do not parse."""


def _parse_header(lines: list[str]) -> tuple[dict[str, str], list[str], list[str]]:
    """Split header lines into a dict plus the repeated comment/subtitle lines."""
    header: dict[str, str] = {}
    comments: list[str] = []
    subtitles: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        key = parts[0]
        value = parts[1].strip() if len(parts) == 2 else ""
        if key == "comment":
            comments.append(value)
        elif key == "subtitle":
            subtitles.append(value)
        else:
            header[key] = value
    return header, comments, subtitles


def _truthy(value: str | None) -> bool:
    return str(value).strip() not in ("", "0", "false", "False", "None")


def parse_fm2(path: Path | str, *, first_controller_only: bool = True) -> Movie:
    """Parse an FCEUX ``.fm2`` movie into a :class:`~tasdata.movie.Movie`.

    Args:
        path: path to the movie. A single outer gzip layer is stripped
            transparently, since that is how TASVideos serves user files.
        first_controller_only: decode only ``port0`` (the first controller field),
            which is all a single-player NES run uses. Set False to also decode
            ports 1 and 2 into ``P2``/``P3`` columns.

    Raises:
        UnsupportedMovieFormatError: the file is not an fm2 at all.
        Fm2ParseError: it is an fm2 but malformed, or a binary fm2.
    """
    path = Path(path)
    result = sniff(path)
    if result.format is not MovieFormat.FM2:
        raise UnsupportedMovieFormatError(path, result.format)

    text = result.data.decode("utf-8", errors="replace")
    raw_lines = text.splitlines()
    split_at = next(
        (i for i, line in enumerate(raw_lines) if line.startswith("|")), len(raw_lines)
    )
    header, comment_lines, subtitle_lines = _parse_header(raw_lines[:split_at])
    frame_lines = [line for line in raw_lines[split_at:] if line.startswith("|")]

    notes: list[str] = []

    # A binary fm2 stores frames as packed bytes after the header, so the text
    # scan above finds nothing usable.
    if _truthy(header.get("binary")):
        raise Fm2ParseError(
            f"{path.name}: this is a binary fm2 (header says 'binary 1'), whose "
            "frame data is packed bytes rather than text rows. Only text fm2 "
            "movies are supported; re-save it from FCEUX with binary mode off."
        )
    if not frame_lines:
        raise Fm2ParseError(
            f"{path.name}: no frame lines (no line starting with '|') after "
            f"{split_at} header line(s)"
        )

    # Column layout: a console group (from the commands field) followed by one
    # group per decoded controller. This mirrors the bk2 layout so that
    # replay/verify code does not care which format it came from.
    console_names = [name for _bit, name in COMMAND_BITS]
    n_ports = 1 if first_controller_only else 3
    player_groups = [
        tuple(f"P{port + 1} {b}" for b in FM2_BUTTON_ORDER) for port in range(n_ports)
    ]
    groups: tuple[tuple[str, ...], ...] = (tuple(console_names), *player_groups)
    button_names = [n for g in groups for n in g]

    states = np.zeros((len(frame_lines), len(button_names)), dtype=bool)
    console_span = len(console_names)
    ports_with_input: set[int] = set()

    for frame_no, line in enumerate(frame_lines):
        body = line[1:-1] if line.endswith("|") else line[1:]
        fields = body.split("|")
        if len(fields) < 2:
            raise Fm2ParseError(
                f"{path.name}: frame {frame_no}: expected at least "
                f"|commands|controller|, got {line!r}"
            )

        # -- commands field --
        command_text = fields[0].strip()
        if command_text:
            try:
                command = int(command_text)
            except ValueError:
                raise Fm2ParseError(
                    f"{path.name}: frame {frame_no}: commands field {command_text!r} "
                    f"is not an integer in {line!r}"
                ) from None
            for offset, (bit, _name) in enumerate(COMMAND_BITS):
                if command & bit:
                    states[frame_no, offset] = True

        # -- controller fields --
        # A port counts as "in use" only when something is actually held: an
        # unused-but-connected port still writes eight dots on every frame.
        controllers = fields[1:]
        for port_idx, chunk in enumerate(controllers):
            if any(is_pressed(ch) for ch in chunk):
                ports_with_input.add(port_idx)
        for port in range(n_ports):
            if port >= len(controllers):
                break
            chunk = controllers[port]
            if chunk == "":
                continue  # unconnected port
            if len(chunk) != CONTROLLER_WIDTH:
                raise Fm2ParseError(
                    f"{path.name}: frame {frame_no}: controller {port} field should "
                    f"be {CONTROLLER_WIDTH} chars ({''.join(FM2_BUTTON_ORDER)}) but is "
                    f"{len(chunk)} ({chunk!r}) in {line!r}"
                )
            base = console_span + port * CONTROLLER_WIDTH
            for offset, char in enumerate(chunk):
                if is_pressed(char):
                    states[frame_no, base + offset] = True

    # -- header interpretation --
    # fm2 has no Author key; FCEUX writes it as a "comment author <name>" line.
    for line in comment_lines:
        if line.lower().startswith("author"):
            header.setdefault("Author", line[len("author"):].strip())
            break

    pal = _truthy(header.get("palFlag"))
    if pal:
        notes.append(
            "palFlag is set: this run was recorded at 50 Hz. nes-py emulates NTSC "
            "only, so timing-sensitive sections may not reproduce."
        )

    rom_hashes: dict[str, str] = {}
    checksum_raw = header.get("romChecksum", "")
    if checksum_raw:
        decoded = decode_fm2_checksum(checksum_raw)
        if decoded:
            rom_hashes["md5-prgchr"] = decoded
        else:
            notes.append(f"could not decode romChecksum {checksum_raw!r}")
    else:
        notes.append("header has no romChecksum; ROM identity cannot be verified")

    if _truthy(header.get("FDS")):
        notes.append("FDS flag is set: this is a Famicom Disk System run, not a cartridge")
    if _truthy(header.get("fourscore")):
        notes.append("fourscore is set: this movie uses a 4-player adapter")
    ignored_ports = sorted(p for p in ports_with_input if p >= n_ports)
    if ignored_ports:
        notes.append(
            f"movie has input on controller port(s) {ignored_ports} which were not "
            f"decoded; only port(s) 0..{n_ports - 1} were read (pass "
            "first_controller_only=False to keep the rest)"
        )
    if _truthy(header.get("NewPPU")):
        notes.append("recorded with FCEUX's 'New PPU' core, which changes timing")
    if "RAMInitOption" in header and header.get("RAMInitOption") not in ("", "0"):
        notes.append(
            f"non-default power-on RAM init (RAMInitOption "
            f"{header['RAMInitOption']}, seed {header.get('RAMInitSeed', '?')}); "
            "nes-py cannot reproduce this and a desync is likely"
        )

    savestate_anchored = bool(header.get("savestate"))
    if savestate_anchored:
        notes.append(
            "movie is anchored to an embedded savestate, so replaying from "
            "power-on will not reproduce the run"
        )

    return Movie(
        path=path,
        format=MovieFormat.FM2,
        header=header,
        button_names=button_names,
        states=states,
        groups=groups,
        rom_hashes=rom_hashes,
        pal=pal,
        comments="\n".join(comment_lines),
        subtitles="\n".join(subtitle_lines),
        sync_settings={
            k: header[k]
            for k in ("palFlag", "NewPPU", "FDS", "fourscore", "RAMInitOption", "RAMInitSeed")
            if k in header
        },
        savestate_anchored=savestate_anchored,
        notes=notes,
    )
