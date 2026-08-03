"""Movie-format sniffing.

The point of this module is to fail *loudly and specifically* when someone hands
the pipeline a TAS movie that is not a BizHawk ``.bk2``.  TASVideos hosts a
couple dozen formats produced by a dozen different emulators; most of them are
binary, none of them are interchangeable, and a generic "bad zip file" traceback
is a miserable thing to debug.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MovieFormat(str, Enum):
    """Every format we can recognise, whether or not we can parse it."""

    BK2 = "bk2"              # BizHawk movie (zip)          -- supported
    TASPROJ = "tasproj"      # BizHawk TAStudio project     -- opt-in
    BKM = "bkm"              # BizHawk legacy text movie
    FM2 = "fm2"              # FCEUX text movie
    FCM = "fcm"              # FCEU legacy
    FMV = "fmv"              # Famtasia
    NMV = "nmv"              # Nintendulator
    SMV = "smv"              # Snes9x
    LSMV = "lsmv"            # lsnes (zip)
    VBM = "vbm"              # VisualBoyAdvance
    GMV = "gmv"              # Gens
    BKB = "bkb"              # BizHawk binary state blob
    M64 = "m64"              # Mupen64
    DTM = "dtm"              # Dolphin
    DSM = "dsm"              # DeSmuME
    MCM = "mcm"              # Mednafen / mcm
    PJM = "pjm"              # PSXjin
    PXM = "pxm"              # PCSX
    LTM = "ltm"              # libTAS (tar)
    UNKNOWN = "unknown"


#: Human-readable provenance, used to build actionable error messages.
FORMAT_DESCRIPTIONS: dict[MovieFormat, str] = {
    MovieFormat.BK2: "BizHawk .bk2 movie",
    MovieFormat.TASPROJ: "BizHawk TAStudio project (.tasproj)",
    MovieFormat.BKM: "legacy BizHawk .bkm movie",
    MovieFormat.FM2: "FCEUX .fm2 movie",
    MovieFormat.FCM: "FCEU .fcm movie",
    MovieFormat.FMV: "Famtasia .fmv movie",
    MovieFormat.NMV: "Nintendulator .nmv movie",
    MovieFormat.SMV: "Snes9x .smv movie",
    MovieFormat.LSMV: "lsnes .lsmv movie",
    MovieFormat.VBM: "VisualBoyAdvance .vbm movie",
    MovieFormat.GMV: "Gens .gmv movie",
    MovieFormat.BKB: "BizHawk .bkb binary blob",
    MovieFormat.M64: "Mupen64 .m64 movie",
    MovieFormat.DTM: "Dolphin .dtm movie",
    MovieFormat.DSM: "DeSmuME .dsm movie",
    MovieFormat.MCM: "Mednafen .mcm movie",
    MovieFormat.PJM: "PSXjin .pjm movie",
    MovieFormat.PXM: "PCSX .pxm movie",
    MovieFormat.LTM: "libTAS .ltm movie",
    MovieFormat.UNKNOWN: "unrecognised file",
}

#: Which console each foreign format targets, so the error can say *why* it is
#: hopeless rather than merely unsupported.
FORMAT_PLATFORM: dict[MovieFormat, str] = {
    MovieFormat.FM2: "NES",
    MovieFormat.FCM: "NES",
    MovieFormat.FMV: "NES",
    MovieFormat.NMV: "NES",
    MovieFormat.BKM: "multi-system",
    MovieFormat.SMV: "SNES",
    MovieFormat.LSMV: "SNES",
    MovieFormat.VBM: "Game Boy Advance",
    MovieFormat.GMV: "Sega Genesis",
    MovieFormat.M64: "Nintendo 64",
    MovieFormat.DTM: "GameCube / Wii",
    MovieFormat.DSM: "Nintendo DS",
    MovieFormat.MCM: "multi-system",
    MovieFormat.PJM: "PlayStation",
    MovieFormat.PXM: "PlayStation",
    MovieFormat.LTM: "Linux (libTAS)",
}

#: Binary magic numbers, checked against the first bytes of the file.
_MAGIC: tuple[tuple[bytes, MovieFormat], ...] = (
    (b"FCM\x1a", MovieFormat.FCM),
    (b"FMV\x1a", MovieFormat.FMV),
    (b"NSS\x1a", MovieFormat.NMV),
    (b"SMV\x1a", MovieFormat.SMV),
    (b"VBM\x1a", MovieFormat.VBM),
    (b"Gens Movie", MovieFormat.GMV),
    (b"M64\x1a", MovieFormat.M64),
    (b"DTM\x1a", MovieFormat.DTM),
    (b"MDFNMOVI", MovieFormat.MCM),
    (b"PJM ", MovieFormat.PJM),
    (b"PXM ", MovieFormat.PXM),
    (b"BIZHAWK", MovieFormat.BKB),
)

#: Zip members that mark an archive as a TAStudio *project* rather than a movie.
#: A .tasproj is a .bk2 plus editor state, so this is the only way to tell them
#: apart from content alone.
_TASPROJ_MARKERS = frozenset(
    {"markers.txt", "clientsettings.json", "session.txt", "laglog", "greenzone"}
)


#: Formats this pipeline has a parser for.
SUPPORTED_FORMATS: frozenset[MovieFormat] = frozenset(
    {MovieFormat.BK2, MovieFormat.FM2}
)


class MovieFormatError(Exception):
    """Base class for anything wrong with a movie file itself."""


class UnsupportedMovieFormatError(MovieFormatError):
    """The file is a real TAS movie, but not one this pipeline can replay."""

    def __init__(self, path: Path | str, fmt: MovieFormat, hint: str = "") -> None:
        self.path = Path(path)
        self.format = fmt
        desc = FORMAT_DESCRIPTIONS.get(fmt, str(fmt))
        platform = FORMAT_PLATFORM.get(fmt)
        supported = "BizHawk .bk2 and FCEUX .fm2"
        if fmt in SUPPORTED_FORMATS:
            # Raised by a format-specific parser rather than the dispatcher.
            msg = (
                f"{self.path.name}: detected {desc}, which is not what this parser "
                f"reads. Use tasdata.movie.parse_movie() to dispatch on format "
                f"automatically."
            )
        else:
            msg = f"{self.path.name}: detected {desc}, but this pipeline only reads {supported}."
            if platform and platform != "NES":
                msg += (
                    f" That format targets {platform}; a replay harness built on"
                    " nes-py cannot use it at all."
                )
            elif platform == "NES":
                msg += (
                    " It is an NES movie, so the inputs are convertible in principle:"
                    " load it in FCEUX or BizHawk and re-save as .fm2 or .bk2."
                )
        if hint:
            msg += f" {hint}"
        super().__init__(msg)


class CorruptMovieError(MovieFormatError):
    """The file claims to be a .bk2 but its contents do not hold up."""


def _looks_like_fm2(head: bytes) -> bool:
    """FCEUX .fm2 is a text key/value header followed by ``|0|........|||`` rows.

    The header always opens with ``version``, but every other key is optional in
    practice (hand-edited movies routinely drop ``romChecksum``), so identity
    rests on ``version`` plus any one of the other characteristic keys.
    """
    try:
        text = head.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    if not text.startswith("version "):
        return False
    markers = ("romChecksum", "port0", "emuVersion", "palFlag", "guid", "fourscore")
    return any(m in text for m in markers) or "\n|" in text


def _looks_like_bkm(head: bytes) -> bool:
    """Legacy BizHawk .bkm: text header with ``emuVersion`` but no zip container."""
    try:
        text = head.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return "emuVersion" in text and "MovieVersion" in text


@dataclass(frozen=True)
class SniffResult:
    """Outcome of :func:`sniff`."""

    format: MovieFormat
    #: The movie's own bytes, with any gzip or zip wrapper removed.
    data: bytes
    #: True when the on-disk file had a gzip wrapper (how TASVideos serves them).
    gzipped: bool = False
    #: Set when the movie was extracted from an enclosing zip (how TASVideos
    #: ships published .fm2 movies), naming the member it came from.
    inner_name: str | None = None

    @property
    def description(self) -> str:
        return FORMAT_DESCRIPTIONS.get(self.format, str(self.format))


def _sniff_zip(data: bytes, path: Path) -> SniffResult:
    """Handle the zip containers: .bk2, .tasproj, .lsmv, or a zipped movie."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        names = {i.filename.lower() for i in members}

        # lsnes movies are zips too, but carry a totally different manifest.
        if "systemid" in names or "gametype" in names:
            return SniffResult(MovieFormat.LSMV, data)

        if "input log.txt" in names:
            # .tasproj is structurally a superset of .bk2. Prefer the extension
            # when explicit, otherwise fall back to editor-state marker files.
            if path.suffix.lower() == ".tasproj" or (names & _TASPROJ_MARKERS):
                return SniffResult(MovieFormat.TASPROJ, data)
            return SniffResult(MovieFormat.BK2, data)

        # TASVideos serves published movies as a zip wrapping a single file.
        # Unwrap it so `parse_movie("...fm2.zip")` just works.
        if len(members) == 1:
            inner = members[0]
            inner_data = z.read(inner)
            nested = _sniff_bytes(inner_data, Path(inner.filename), depth=1)
            if nested.format in (MovieFormat.UNKNOWN,):
                raise CorruptMovieError(
                    f"{path.name}: zip contains one member "
                    f"({inner.filename!r}) but it is not a recognised movie"
                )
            return SniffResult(nested.format, nested.data, inner_name=inner.filename)

    raise CorruptMovieError(
        f"{path.name}: zip archive with no 'Input Log.txt' member and not a "
        f"single-file movie wrapper (found: {sorted(names) or 'nothing'})."
    )


def _sniff_bytes(data: bytes, path: Path, depth: int = 0) -> SniffResult:
    """Classify raw movie bytes. ``depth`` guards against nested archives."""
    if not data:
        raise CorruptMovieError(f"{path.name}: file is empty")

    gzipped = False
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
            gzipped = True
        except OSError as exc:  # truncated / not really gzip
            raise CorruptMovieError(
                f"{path.name}: gzip header but undecodable ({exc})"
            ) from exc

    if data[:2] == b"PK":
        if depth > 0:
            raise CorruptMovieError(f"{path.name}: nested archives are not unwrapped")
        result = _sniff_zip(data, path)
        return SniffResult(result.format, result.data, gzipped, result.inner_name)

    head = data[:4096]
    for magic, fmt in _MAGIC:
        if head.startswith(magic):
            return SniffResult(fmt, data, gzipped)

    if _looks_like_fm2(head):
        return SniffResult(MovieFormat.FM2, data, gzipped)
    if _looks_like_bkm(head):
        return SniffResult(MovieFormat.BKM, data, gzipped)
    # libTAS movies are uncompressed tar archives.
    if len(data) > 262 and data[257:262] == b"ustar":
        return SniffResult(MovieFormat.LTM, data, gzipped)
    if head.lstrip().startswith(b"|") and b"|" in head:
        return SniffResult(MovieFormat.DSM, data, gzipped)

    return SniffResult(MovieFormat.UNKNOWN, data, gzipped)


def sniff(path: Path | str) -> SniffResult:
    """Identify the movie format of ``path`` by content, not by extension.

    Two wrappers are stripped transparently, because that is how TASVideos
    distributes movies: a single outer gzip layer (user files) and a zip holding
    exactly one movie (publications). A gzipped or zipped .fm2 is still a .fm2.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"movie not found: {path}")
    return _sniff_bytes(path.read_bytes(), path)
