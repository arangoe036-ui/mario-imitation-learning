"""Shared fixtures: synthetic movie files and a real-ROM discovery helper."""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

#: The two-controller NES log key BizHawk writes for NROM games.
FULL_LOG_KEY = (
    "LogKey:#Power|Reset|#P1 Up|P1 Down|P1 Left|P1 Right|P1 Start|P1 Select|P1 B|P1 A|"
    "#P2 Up|P2 Down|P2 Left|P2 Right|P2 Start|P2 Select|P2 B|P2 A|"
)

#: Single-controller variant, also common.
P1_LOG_KEY = (
    "LogKey:#Power|Reset|#P1 Up|P1 Down|P1 Left|P1 Right|P1 Start|P1 Select|P1 B|P1 A|"
)

DEFAULT_HEADER = """MovieVersion BizHawk v2.0.0
Author test
emuVersion Version 2.9
Platform NES
GameName Super Mario Bros
SHA1 EA343F4E445A9050D4B4FBAC2C77D0693B1D0922
BoardName NROM
Core NesHawk
rerecordCount 42
"""


def make_bk2(
    path: Path,
    frame_rows: list[str],
    *,
    log_key: str = FULL_LOG_KEY,
    header: str = DEFAULT_HEADER,
    extra_members: dict[str, str] | None = None,
    gzip_it: bool = False,
) -> Path:
    """Write a minimal but structurally faithful .bk2 to ``path``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Header.txt", header)
        z.writestr("Comments.txt", "")
        z.writestr("Subtitles.txt", "")
        z.writestr("SyncSettings.json", '{"o": {"RegionOverride": 0}}')
        body = "[Input]\n" + log_key + "\n" + "\n".join(frame_rows) + "\n[/Input]\n"
        z.writestr("Input Log.txt", body)
        for name, content in (extra_members or {}).items():
            z.writestr(name, content)
    data = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data) if gzip_it else data)
    return path


def blank_rows(n: int, *, players: int = 2) -> list[str]:
    """``n`` frames of no input."""
    pads = "|".join(["." * 8] * players)
    return [f"|..|{pads}|" for _ in range(n)]


@pytest.fixture
def bk2_simple(tmp_path: Path) -> Path:
    """Four frames: idle, Start, Right, Right+A."""
    rows = [
        "|..|........|........|",
        "|..|....S...|........|",
        "|..|...R....|........|",
        "|..|...R...A|........|",
    ]
    return make_bk2(tmp_path / "simple.bk2", rows)


@pytest.fixture
def bk2_p1_only(tmp_path: Path) -> Path:
    """Single-controller log key."""
    rows = ["|..|........|", "|..|...R..BA|"]
    return make_bk2(tmp_path / "p1.bk2", rows, log_key=P1_LOG_KEY)


def find_smb_rom() -> Path | None:
    """Locate an SMB ROM: the ``SMB_ROM`` env var, then gym_super_mario_bros."""
    import os

    env = os.environ.get("SMB_ROM")
    if env and Path(env).exists():
        return Path(env)
    try:
        import gym_super_mario_bros

        candidate = (
            Path(gym_super_mario_bros.__file__).parent / "_roms" / "super-mario-bros.nes"
        )
        if candidate.exists():
            return candidate
    except ImportError:
        pass
    return None


@pytest.fixture
def smb_rom() -> Path:
    rom = find_smb_rom()
    if rom is None:
        pytest.skip("no SMB ROM available (set SMB_ROM or install gym-super-mario-bros)")
    return rom


def synthetic_trace(rows: list[dict]) -> np.ndarray:
    """Build a packed RAM trace from partial column dicts (missing = sensible default)."""
    from tasdata.ram import TRACE_COLUMNS

    defaults = {
        "frame": 0, "world": 1, "stage": 1, "area": 1, "x_position": 0,
        "y_position": 100, "player_state": 0x08, "player_status": 0, "lives": 2,
        "coins": 0, "time": 400, "score": 0, "pregame": 1,
    }
    out = np.zeros((len(rows), len(TRACE_COLUMNS)), dtype=np.int32)
    for i, row in enumerate(rows):
        merged = {**defaults, "frame": i, **row}
        for j, col in enumerate(TRACE_COLUMNS):
            out[i, j] = merged[col]
    return out
