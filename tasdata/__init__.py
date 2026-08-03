"""TAS imitation-learning data pipeline.

Three stages, one module each, usable independently:

1. :mod:`tasdata.bk2` / :mod:`tasdata.fm2` -- parse a BizHawk ``.bk2`` or FCEUX
   ``.fm2`` into a numpy button matrix (:func:`tasdata.movie.parse_movie`
   dispatches on the sniffed format).
2. :mod:`tasdata.replay`  -- drive nes-py with those inputs, capturing frames + RAM.
3. :mod:`tasdata.verify`  -- decide whether the replay actually progressed.

Supporting modules: :mod:`tasdata.formats` (format sniffing and errors),
:mod:`tasdata.buttons` (bk2 button names <-> nes-py action bytes),
:mod:`tasdata.ram` (Super Mario Bros. RAM map), :mod:`tasdata.rom` (iNES ROM
fingerprints), :mod:`tasdata.tasvideos` (fetch movies from tasvideos.org).

No model code lives here by design; this package only produces training data.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .bk2 import Bk2ParseError, LogKey, parse_bk2, starts_from_savestate
from .fm2 import FM2_BUTTON_ORDER, Fm2ParseError, parse_fm2
from .movie import Movie, parse_movie
from .buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER, actions_from_states
from .formats import (
    SUPPORTED_FORMATS,
    CorruptMovieError,
    MovieFormat,
    MovieFormatError,
    UnsupportedMovieFormatError,
    sniff,
)
from .ram import SmbState, read_smb
from .replay import NesReplayer, ReplayError, ReplayResult, RomMismatchError, rom_sha1
from .rom import NesRom, RomCheck, load_rom
from .verify import SyncReport, compare_traces, verify_smb

__all__ = [
    "Bk2ParseError",
    "FM2_BUTTON_ORDER",
    "Fm2ParseError",
    "CorruptMovieError",
    "LogKey",
    "MovieFormat",
    "MovieFormatError",
    "Movie",
    "NES_BUTTON_BITS",
    "NES_BUTTON_ORDER",
    "NesReplayer",
    "NesRom",
    "ReplayError",
    "ReplayResult",
    "RomCheck",
    "RomMismatchError",
    "SUPPORTED_FORMATS",
    "SmbState",
    "SyncReport",
    "UnsupportedMovieFormatError",
    "actions_from_states",
    "compare_traces",
    "load_rom",
    "parse_bk2",
    "parse_fm2",
    "parse_movie",
    "read_smb",
    "rom_sha1",
    "sniff",
    "starts_from_savestate",
    "verify_smb",
    "__version__",
]
