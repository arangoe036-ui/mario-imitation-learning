"""FCEUX backend tests.

The Lua generation, record framing and FIFO reader are testable without the
emulator. Tests that launch FCEUX are marked ``fceux`` and skip when the binary
or a matching ROM is absent (and it needs a window, so it cannot run in CI).
"""

from __future__ import annotations

import base64
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from tasdata.backends import BACKENDS, DEFAULT_BACKEND, get_replayer
from tasdata.fceux_backend import (
    GD_HEADER,
    GD_HEIGHT,
    GD_LEN,
    GD_WIDTH,
    RAM_BYTES,
    FceuxError,
    FceuxReplayer,
    _lua_string,
    build_lua_script,
    find_fceux,
)
from tasdata.movie import parse_movie
from tasdata.ram import TRACE_COLUMNS
from tasdata.replay import NesReplayer, RomMismatchError

from .conftest import find_smb_rom
from .test_fm2 import HEADER as FM2_HEADER, make_fm2
from .test_rom import make_rom


def _fceux_available() -> bool:
    return shutil.which("fceux") is not None


def _ntsc_rom() -> Path | None:
    """A ROM matching the NTSC dump the warpless publication needs."""
    for candidate in (Path("smb.nes"), Path(__file__).parent.parent / "smb.nes"):
        if candidate.exists():
            from tasdata.rom import load_rom

            if load_rom(candidate).md5_prgchr == "8e3630186e35d477231bf8fd50e54cdd":
                return candidate
    return None


fceux = pytest.mark.skipif(not _fceux_available(), reason="fceux not installed")


class TestConstants:
    def test_gd_layout_matches_nes_output(self):
        assert (GD_WIDTH, GD_HEIGHT) == (256, 240)
        assert GD_LEN == GD_HEADER + 256 * 240 * 4 == 245771

    def test_ram_size(self):
        assert RAM_BYTES == 2048


class TestLuaGeneration:
    def test_embeds_parameters(self, tmp_path: Path):
        lua = build_lua_script(
            n_frames=1234, fifo=tmp_path / "f.fifo", frame_skip=3, want_screen=True
        )
        assert "local N          = 1234" in lua
        assert "local FRAME_SKIP = 3" in lua
        assert "local WANT_SCREEN= 1" in lua
        # The path goes in as a Lua string literal, so a Windows backslash is escaped.
        assert _lua_string(str(tmp_path / "f.fifo")) in lua

    def test_screen_can_be_disabled(self, tmp_path: Path):
        lua = build_lua_script(
            n_frames=10, fifo=tmp_path / "f", frame_skip=1, want_screen=False
        )
        assert "local WANT_SCREEN= 0" in lua

    def test_uses_maximum_speed_and_exits(self, tmp_path: Path):
        lua = build_lua_script(
            n_frames=10, fifo=tmp_path / "f", frame_skip=1, want_screen=True
        )
        assert 'emu.speedmode("maximum")' in lua
        assert "os.exit(0)" in lua

    def test_advances_before_sampling(self, tmp_path: Path):
        """Record i must be the state *after* movie frame i, like the nes-py path."""
        lua = build_lua_script(
            n_frames=10, fifo=tmp_path / "f", frame_skip=1, want_screen=True
        )
        advance = lua.index("emu.frameadvance()")
        sample = lua.index("memory.readbyterange")
        assert advance < sample

    def test_lua_modulo_is_escaped(self, tmp_path: Path):
        """`%` must survive Python's %-formatting as a literal Lua operator."""
        lua = build_lua_script(
            n_frames=10, fifo=tmp_path / "f", frame_skip=2, want_screen=True
        )
        assert "(i % FRAME_SKIP) == 0" in lua
        assert "%%" not in lua

    def test_paths_with_spaces_are_quoted(self, tmp_path: Path):
        odd = tmp_path / "a dir" / "cap.fifo"
        lua = build_lua_script(n_frames=1, fifo=odd, frame_skip=1, want_screen=False)
        quoted = _lua_string(str(odd))
        assert quoted.startswith('"') and quoted.endswith('"')
        assert quoted in lua


class TestBackendRegistry:
    def test_default_is_fceux(self):
        assert DEFAULT_BACKEND == "fceux"
        assert set(BACKENDS) == {"fceux", "nes-py"}

    def test_nes_py_selectable(self, tmp_path: Path):
        rom = make_rom(tmp_path / "g.nes")
        assert isinstance(get_replayer("nes-py", str(rom)), NesReplayer)

    def test_aliases(self, tmp_path: Path):
        rom = make_rom(tmp_path / "g.nes")
        for alias in ("NES-PY", "nespy", "nes_py"):
            assert isinstance(get_replayer(alias, str(rom)), NesReplayer)

    def test_unknown_backend(self, tmp_path: Path):
        rom = make_rom(tmp_path / "g.nes")
        with pytest.raises(ValueError, match="unknown backend"):
            get_replayer("mesen", str(rom))

    def test_extra_args_ignored_by_nes_py(self, tmp_path: Path):
        """One keyword set must construct either backend."""
        rom = make_rom(tmp_path / "g.nes")
        assert isinstance(
            get_replayer("nes-py", str(rom), extra_args=("--pal", "1")), NesReplayer
        )


@fceux
class TestBinaryDiscovery:
    def test_finds_fceux(self):
        assert find_fceux("fceux").exists()

    def test_missing_binary_message_is_actionable(self):
        with pytest.raises(FceuxError, match="brew install fceux"):
            find_fceux("definitely-not-fceux-xyz")

    def test_version_and_rev_recorded(self, tmp_path: Path):
        rom = make_rom(tmp_path / "g.nes")
        r = FceuxReplayer(rom, capture_frames=False)
        assert r.version != "unknown"
        assert r.backend == "fceux"
        assert "opposite-directionals=on" in r.describe()


@fceux
class TestRomGate:
    def test_mismatch_refused_before_launching(self, tmp_path: Path):
        rom = make_rom(tmp_path / "g.nes")
        movie = parse_movie(
            make_fm2(tmp_path / "m.fm2", ["|0|........|........||"] * 4)
        )
        with pytest.raises(RomMismatchError, match="ROM mismatch"):
            FceuxReplayer(rom, capture_frames=False).replay(movie)


# --------------------------------------------------------------------------- #
# End-to-end, needs the real ROM + movie
# --------------------------------------------------------------------------- #

@fceux
class TestEndToEnd:
    @pytest.fixture
    def rom(self) -> Path:
        path = _ntsc_rom()
        if path is None:
            pytest.skip("no NTSC SMB ROM (expected ./smb.nes)")
        return path

    @pytest.fixture
    def movie(self):
        path = Path("data/movies/happylee_mars608-smb-warpless.fm2")
        if not path.exists():
            pytest.skip("warpless movie not present")
        return parse_movie(path)

    def test_short_capture_is_frame_exact(self, rom: Path, movie):
        n = 400
        result = FceuxReplayer(rom, observation_shape=(84, 84)).replay(
            movie, max_frames=n
        )
        assert result.n_frames == n
        assert result.trace.shape == (n, len(TRACE_COLUMNS))
        assert result.frames.shape == (n, 84, 84)
        assert result.actions.shape == (n,)
        assert result.rom_check.ok
        assert result.backend.startswith("fceux")

    def test_frame_skip_thins_observations_only(self, rom: Path, movie):
        result = FceuxReplayer(rom, frame_skip=4).replay(movie, max_frames=400)
        assert result.n_frames == 400          # every frame emulated
        assert result.trace.shape[0] == 400    # RAM every frame
        assert result.frames.shape[0] == 100   # every 4th captured
        assert result.frame_indices.tolist()[:3] == [0, 4, 8]

    def test_screen_is_not_blank_once_running(self, rom: Path, movie):
        result = FceuxReplayer(rom, observation_shape=(84, 84)).replay(
            movie, max_frames=300
        )
        assert result.frames[-1].std() > 0

    def test_ram_only_is_faster_and_captures_no_frames(self, rom: Path, movie):
        result = FceuxReplayer(rom, capture_frames=False).replay(movie, max_frames=400)
        assert result.frames.shape[0] == 0
        assert result.trace.shape[0] == 400

    def test_deterministic_across_processes(self, rom: Path, movie):
        a = FceuxReplayer(rom, capture_frames=False).replay(movie, max_frames=600)
        b = FceuxReplayer(rom, capture_frames=False).replay(movie, max_frames=600)
        assert np.array_equal(a.trace, b.trace)

    def test_memmap_output(self, rom: Path, movie, tmp_path: Path):
        path = tmp_path / "frames.npy"
        FceuxReplayer(rom, observation_shape=(30, 32)).replay(
            movie, max_frames=120, frames_path=path
        )
        assert np.load(path, mmap_mode="r").shape == (120, 30, 32)

    def test_clears_world_1_1(self, rom: Path, movie):
        """FCEUX must actually sync, not merely produce frames."""
        from tasdata.verify import verify_smb

        result = FceuxReplayer(rom, capture_frames=False).replay(movie, max_frames=3000)
        report = verify_smb(result.trace, expect_level="1-2")
        assert report.passed, report.reason
        assert "1-2" in report.levels_reached


@fceux
class TestUnplayableGuards:
    """FCEUX silently plays nothing when handed a container or a foreign format.

    That failure mode produced a full-length capture of the attract-mode demo, with
    a correct frame count, so it has to be caught before launching.
    """

    def test_bk2_is_refused(self, tmp_path: Path, bk2_simple: Path):
        from tasdata.fceux_backend import UnplayableMovieError
        from tasdata.movie import parse_movie
        from tasdata.rom import load_rom

        rom_path = _ntsc_rom()
        if rom_path is None:
            pytest.skip("no NTSC SMB ROM")
        movie = parse_movie(bk2_simple)
        # point the movie's recorded hash at the real ROM so the ROM gate passes
        movie.rom_hashes = {"sha1-file": load_rom(rom_path).sha1_file}
        with pytest.raises(UnplayableMovieError, match="only replay .fm2"):
            FceuxReplayer(rom_path, capture_frames=False).replay(movie)

    def test_zip_wrapped_fm2_is_unwrapped_not_refused(self, tmp_path: Path):
        """A zipped fm2 must still work -- unwrapped, with a note."""
        from tasdata.movie import parse_movie
        from tasdata.rom import load_rom

        rom_path = _ntsc_rom()
        if rom_path is None:
            pytest.skip("no NTSC SMB ROM")
        rom = load_rom(rom_path)
        header = FM2_HEADER.replace(
            "romChecksum base64:jjYwGG411HcjG/j9UOVM3Q==", f"romChecksum {rom.fm2_checksum}"
        )
        path = make_fm2(
            tmp_path / "m.fm2.zip",
            ["|0|........|........||"] * 60,
            header=header,
            zip_it=True,
        )
        movie = parse_movie(path)
        result = FceuxReplayer(rom_path, capture_frames=False).replay(movie)
        assert result.n_frames == 60
        assert any("unwrapped for FCEUX" in w for w in result.warnings)
