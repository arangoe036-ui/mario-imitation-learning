"""Super Mario Bros. RAM map and the generic RAM-probe plumbing.

Addresses follow the community SMB disassembly (doppelganger / Data Crystal) and
match the ones ``gym_super_mario_bros`` uses, so traces recorded here are
directly comparable to that environment's ``info`` dicts.

The probe returns a fixed-width ``int32`` vector per frame rather than a dict:
over a 20-minute movie that is the difference between a 70 MB array and a
million Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Address constants
# --------------------------------------------------------------------------- #

ADDR_WORLD = 0x075F          # 0-based world number
ADDR_STAGE = 0x075C          # 0-based stage (level) within the world
ADDR_AREA = 0x0760           # 0-based area within the stage (pipes/vines)
ADDR_X_PAGE = 0x006D         # player x position, high byte (screen page)
ADDR_X_IN_PAGE = 0x0086      # player x position, low byte
ADDR_ON_GROUND = 0x001D      # 0 = standing on a surface, nonzero = airborne. VERIFIED:
                             # standing 0, rising 1, apex 1, falling 1, landed 0.
ADDR_Y_VIEWPORT = 0x00B5     # which vertical "page" the player is on
ADDR_Y_PIXEL = 0x03B8        # player y position on screen
ADDR_LEFT_X = 0x071C         # left edge of the screen, low byte
ADDR_LEFT_X_PAGE = 0x071A    # left edge of the screen, page
ADDR_PLAYER_STATE = 0x000E   # 0x06/0x0B dying, 0x08 normal, ...
ADDR_PLAYER_STATUS = 0x0756  # 0 small, 1 big, 2+ fire
ADDR_LIVES = 0x075A
ADDR_COINS = 0x075E
ADDR_SCORE = 0x07DE          # 6 BCD-ish digits, one per byte
ADDR_TIME = 0x07F8           # 3 digits, one per byte
ADDR_PREGAME = 0x0770        # 0 demo/title, 1 playing, 2 game over, 3 loading

#: ``_player_state`` values that mean Mario is losing a life.
DYING_STATES = frozenset({0x06, 0x0B})

#: ``_player_state`` while the player has normal control. Other values are
#: scripted sequences: 0x01 climbing, 0x03 changing size, 0x05 flagpole slide,
#: 0x07 entering a pipe/area, 0x06/0x0B dying, 0x00 pre-level.
PLAYER_STATE_NORMAL = 0x08

#: Number of frames the SMB level-load routine holds the player before control
#: returns. Used to avoid calling a legitimate level transition a stall.
LEVEL_LOAD_FRAMES = 120


@dataclass(frozen=True)
class SmbState:
    """One frame's worth of decoded SMB RAM."""

    frame: int
    world: int          # 1-based, as displayed
    stage: int          # 1-based, as displayed
    area: int           # 1-based
    x_position: int     # absolute, in pixels from the start of the area
    y_position: int     # screen pixels, larger = lower
    player_state: int
    player_status: int  # 0 small, 1 big, 2 fire
    lives: int
    coins: int
    time: int
    score: int
    pregame: int

    @property
    def level(self) -> int:
        """Monotonic level ordinal: ``world*4 + stage``, 0-based."""
        return (self.world - 1) * 4 + (self.stage - 1)

    @property
    def is_dying(self) -> bool:
        return self.player_state in DYING_STATES

    def label(self) -> str:
        return f"{self.world}-{self.stage}"

    def __str__(self) -> str:
        return (
            f"f{self.frame} {self.label()} area {self.area} x={self.x_position} "
            f"y={self.y_position} lives={self.lives} time={self.time} "
            f"state={self.player_state:#04x}"
        )


#: Column order of the packed trace array. Keep in sync with :class:`SmbState`.
TRACE_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(SmbState))


def _digits(ram: np.ndarray, start: int, count: int) -> int:
    """Read ``count`` single-digit bytes big-endian into an int.

    SMB stores the timer and score as one decimal digit per byte. Uninitialised
    bytes can hold values > 9 during loading screens, so digits are clamped.
    """
    value = 0
    for offset in range(count):
        digit = int(ram[start + offset])
        value = value * 10 + (digit if 0 <= digit <= 9 else 0)
    return value


def read_smb(ram: np.ndarray, frame: int = -1) -> SmbState:
    """Decode a 2 KB NES RAM buffer into an :class:`SmbState`."""
    return SmbState(
        frame=frame,
        world=int(ram[ADDR_WORLD]) + 1,
        stage=int(ram[ADDR_STAGE]) + 1,
        area=int(ram[ADDR_AREA]) + 1,
        x_position=int(ram[ADDR_X_PAGE]) * 0x100 + int(ram[ADDR_X_IN_PAGE]),
        y_position=int(ram[ADDR_Y_PIXEL]),
        player_state=int(ram[ADDR_PLAYER_STATE]),
        player_status=int(ram[ADDR_PLAYER_STATUS]),
        lives=int(ram[ADDR_LIVES]),
        coins=int(ram[ADDR_COINS]),
        time=_digits(ram, ADDR_TIME, 3),
        score=_digits(ram, ADDR_SCORE, 6) * 10,
        pregame=int(ram[ADDR_PREGAME]),
    )


def pack_smb(ram: np.ndarray, frame: int, out: np.ndarray) -> None:
    """Write one frame of decoded SMB RAM into ``out``, a 1-D int32 row.

    Hot path: called once per emulated frame, so this avoids building an
    :class:`SmbState` object.
    """
    out[0] = frame
    out[1] = ram[ADDR_WORLD] + 1
    out[2] = ram[ADDR_STAGE] + 1
    out[3] = ram[ADDR_AREA] + 1
    out[4] = int(ram[ADDR_X_PAGE]) * 0x100 + int(ram[ADDR_X_IN_PAGE])
    out[5] = ram[ADDR_Y_PIXEL]
    out[6] = ram[ADDR_PLAYER_STATE]
    out[7] = ram[ADDR_PLAYER_STATUS]
    out[8] = ram[ADDR_LIVES]
    out[9] = ram[ADDR_COINS]
    out[10] = _digits(ram, ADDR_TIME, 3)
    out[11] = _digits(ram, ADDR_SCORE, 6) * 10
    out[12] = ram[ADDR_PREGAME]


def state_from_row(row: Sequence[int]) -> SmbState:
    """Inverse of :func:`pack_smb` for a single trace row."""
    return SmbState(*(int(v) for v in row))


def trace_to_states(trace: np.ndarray) -> list[SmbState]:
    """Convert a whole packed trace back into :class:`SmbState` objects."""
    return [state_from_row(row) for row in trace]


def level_ordinal(trace: np.ndarray) -> np.ndarray:
    """Per-frame monotonic level ordinal ``(world-1)*4 + (stage-1)``."""
    world = trace[:, TRACE_COLUMNS.index("world")]
    stage = trace[:, TRACE_COLUMNS.index("stage")]
    return (world - 1) * 4 + (stage - 1)


def on_ground(ram) -> bool:
    """Is Mario standing on a surface?

    Read SMB's own indicator instead of comparing y to a constant. Seven failures in this
    project trace to the latter: the wrapped byte makes 176 mean three different heights
    (air, floor, deep below), and even unwrapped, a value test cannot tell ground from a
    platform at ground height. This byte answers the question directly.

    Verified against a scripted probe: standing 0, rising 1, apex 1, falling 1, landed 0.
    """
    return int(ram[ADDR_ON_GROUND]) == 0


def y_absolute(ram) -> int:
    """Mario's vertical position including the page byte.

    ``ADDR_Y_PIXEL`` alone is a single byte and wraps at 256, so a deep fall reads as a small
    number -- which is why a ``y > 200`` pit test could never fire. ``ADDR_Y_VIEWPORT`` was
    defined in this module from the start and read nowhere. This is the y counterpart of the
    x fix (page * 256 + offset).
    """
    return int(ram[ADDR_Y_VIEWPORT]) * 256 + int(ram[ADDR_Y_PIXEL])


def column(trace: np.ndarray, name: str) -> np.ndarray:
    """Extract a named column from a packed trace."""
    return trace[:, TRACE_COLUMNS.index(name)]
