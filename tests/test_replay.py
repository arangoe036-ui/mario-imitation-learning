"""Replay harness tests.

The pure-python parts (frame downscaling, ROM gating, action plumbing) are tested
unconditionally.  Tests that actually boot the emulator are marked ``emulator``
and skip when nes-py or a ROM is unavailable.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tasdata.movie import parse_movie
from tasdata.ram import TRACE_COLUMNS
from tasdata.replay import (
    NES_HEIGHT,
    NES_WIDTH,
    NesReplayer,
    ReplayError,
    RomMismatchError,
    _resize_gray,
)

from .test_fm2 import HEADER as FM2_HEADER, make_fm2
from .test_rom import make_rom

pytestmark = []


def _nes_py_available() -> bool:
    try:
        import nes_py  # noqa: F401
    except Exception:
        return False
    return True


emulator = pytest.mark.skipif(
    not _nes_py_available(), reason="nes-py not installed"
)


class TestResizeGray:
    def test_native_size_is_passthrough_luma(self):
        frame = np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8)
        frame[..., 0] = 255  # pure red
        out = _resize_gray(frame, (NES_HEIGHT, NES_WIDTH))
        assert out.shape == (NES_HEIGHT, NES_WIDTH)
        assert out.dtype == np.uint8
        assert out[0, 0] == pytest.approx(0.299 * 255, abs=1)

    def test_downscale_shape_and_dtype(self):
        frame = np.random.randint(0, 256, (NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8)
        out = _resize_gray(frame, (84, 84))
        assert out.shape == (84, 84)
        assert out.dtype == np.uint8

    def test_uniform_frame_survives_downscale(self):
        frame = np.full((NES_HEIGHT, NES_WIDTH, 3), 200, dtype=np.uint8)
        out = _resize_gray(frame, (42, 40))
        assert out.shape == (42, 40)
        assert np.allclose(out, 200, atol=1)


class TestRomGate:
    def _movie_and_rom(self, tmp_path: Path, *, matching: bool):
        body = b"\x55" * 4096
        rom_path = make_rom(tmp_path / "g.nes", body)
        if matching:
            checksum = base64.b64encode(hashlib.md5(body).digest()).decode()
            header = FM2_HEADER.replace(
                "romChecksum base64:jjYwGG411HcjG/j9UOVM3Q==",
                f"romChecksum base64:{checksum}",
            )
        else:
            header = FM2_HEADER
        movie = parse_movie(
            make_fm2(tmp_path / "m.fm2", ["|0|........|........||"] * 4, header=header)
        )
        return movie, rom_path

    def test_mismatch_refused_by_default(self, tmp_path: Path):
        movie, rom_path = self._movie_and_rom(tmp_path, matching=False)
        replayer = NesReplayer(rom_path, capture_frames=False)
        with pytest.raises(RomMismatchError, match="ROM mismatch"):
            replayer.replay(movie)

    def test_mismatch_error_names_the_override(self, tmp_path: Path):
        movie, rom_path = self._movie_and_rom(tmp_path, matching=False)
        replayer = NesReplayer(rom_path, capture_frames=False)
        with pytest.raises(RomMismatchError, match="allow_rom_mismatch"):
            replayer.replay(movie)

    def test_missing_rom(self, tmp_path: Path):
        with pytest.raises(ReplayError, match="ROM not found"):
            NesReplayer(tmp_path / "nope.nes")

    def test_bad_frame_skip(self, tmp_path: Path):
        _movie, rom_path = self._movie_and_rom(tmp_path, matching=True)
        with pytest.raises(ValueError, match="frame_skip"):
            NesReplayer(rom_path, frame_skip=0)


# --------------------------------------------------------------------------- #
# Emulator-backed tests
# --------------------------------------------------------------------------- #

#: A 40-frame idle movie whose checksum is patched to match the ROM under test.
def _idle_movie(tmp_path: Path, rom, frames: int = 40):
    checksum = rom.fm2_checksum
    header = FM2_HEADER.replace(
        "romChecksum base64:jjYwGG411HcjG/j9UOVM3Q==", f"romChecksum {checksum}"
    )
    return parse_movie(
        make_fm2(tmp_path / "idle.fm2", ["|0|........|........||"] * frames, header=header)
    )


@emulator
class TestEmulatorReplay:
    def test_trace_shape_and_frame_count(self, tmp_path: Path, smb_rom: Path):
        from tasdata.rom import load_rom

        rom = load_rom(smb_rom)
        movie = _idle_movie(tmp_path, rom, frames=30)
        result = NesReplayer(smb_rom, capture_frames=False).replay(movie)
        assert result.n_frames == 30
        assert result.trace.shape == (30, len(TRACE_COLUMNS))
        assert result.rom_check.ok

    def test_frames_captured_at_requested_size(self, tmp_path: Path, smb_rom: Path):
        from tasdata.rom import load_rom

        rom = load_rom(smb_rom)
        movie = _idle_movie(tmp_path, rom, frames=20)
        result = NesReplayer(smb_rom, observation_shape=(84, 84)).replay(movie)
        assert result.frames.shape == (20, 84, 84)
        assert result.frames.dtype == np.uint8

    def test_screen_is_drawn_once_the_title_appears(self, tmp_path: Path, smb_rom: Path):
        """The first frames are legitimately blank; by frame ~120 SMB has drawn."""
        from tasdata.rom import load_rom

        rom = load_rom(smb_rom)
        movie = _idle_movie(tmp_path, rom, frames=150)
        result = NesReplayer(smb_rom, observation_shape=(84, 84)).replay(movie)
        assert result.frames[-1].std() > 0

    def test_frame_skip_thins_observations_only(self, tmp_path: Path, smb_rom: Path):
        from tasdata.rom import load_rom

        rom = load_rom(smb_rom)
        movie = _idle_movie(tmp_path, rom, frames=20)
        result = NesReplayer(smb_rom, frame_skip=4).replay(movie)
        assert result.n_frames == 20                    # every frame emulated
        assert result.frames.shape[0] == 5              # every 4th captured
        assert result.frame_indices.tolist() == [0, 4, 8, 12, 16]
        assert result.trace.shape[0] == 20              # RAM probed every frame

    def test_max_frames_truncates(self, tmp_path: Path, smb_rom: Path):
        from tasdata.rom import load_rom

        rom = load_rom(smb_rom)
        movie = _idle_movie(tmp_path, rom, frames=50)
        result = NesReplayer(smb_rom, capture_frames=False).replay(movie, max_frames=10)
        assert result.n_frames == 10

    def test_replay_is_deterministic(self, tmp_path: Path, smb_rom: Path):
        """Two runs of the same movie must produce identical traces."""
        from tasdata.rom import load_rom

        rom = load_rom(smb_rom)
        movie = _idle_movie(tmp_path, rom, frames=60)
        a = NesReplayer(smb_rom, capture_frames=False).replay(movie)
        b = NesReplayer(smb_rom, capture_frames=False).replay(movie)
        assert np.array_equal(a.trace, b.trace)

    def test_memmap_output(self, tmp_path: Path, smb_rom: Path):
        from tasdata.rom import load_rom

        rom = load_rom(smb_rom)
        movie = _idle_movie(tmp_path, rom, frames=12)
        path = tmp_path / "frames.npy"
        result = NesReplayer(smb_rom, observation_shape=(30, 32)).replay(
            movie, frames_path=path
        )
        assert path.exists()
        del result
        loaded = np.load(path, mmap_mode="r")
        assert loaded.shape == (12, 30, 32)

    def test_save_and_reload_run(self, tmp_path: Path, smb_rom: Path):
        from tasdata.replay import load_run
        from tasdata.rom import load_rom

        rom = load_rom(smb_rom)
        movie = _idle_movie(tmp_path, rom, frames=15)
        result = NesReplayer(smb_rom, observation_shape=(20, 20)).replay(movie)
        out = result.save(tmp_path / "run.npz")
        data = load_run(out)
        assert data["frames"].shape == (15, 20, 20)
        assert data["trace"].shape == (15, len(TRACE_COLUMNS))
        assert data["actions"].shape == (15,)
