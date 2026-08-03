"""Parser tests: log key grouping, frame decoding, and the error paths."""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tasdata.bk2 import Bk2ParseError, LogKey, parse_bk2, starts_from_savestate
from tasdata.buttons import actions_from_states, console_button_frames
from tasdata.formats import MovieFormat, UnsupportedMovieFormatError, sniff

from .conftest import FULL_LOG_KEY, P1_LOG_KEY, make_bk2


class TestLogKey:
    def test_groups_split_on_hash(self):
        key = LogKey.parse(FULL_LOG_KEY)
        assert key.widths == (2, 8, 8)
        assert key.groups[0] == ("Power", "Reset")
        assert key.groups[1][0] == "P1 Up"
        assert key.groups[2][-1] == "P2 A"
        assert len(key.names) == 18

    def test_single_controller(self):
        key = LogKey.parse(P1_LOG_KEY)
        assert key.widths == (2, 8)
        assert key.names[-1] == "P1 A"

    def test_rejects_key_without_group_marker(self):
        with pytest.raises(Bk2ParseError, match="group marker"):
            LogKey.parse("LogKey:P1 Up|P1 Down|")

    def test_rejects_empty_key(self):
        with pytest.raises(Bk2ParseError, match="empty log key"):
            LogKey.parse("LogKey:")


class TestParse:
    def test_shape_and_dtype(self, bk2_simple: Path):
        movie = parse_bk2(bk2_simple)
        assert movie.states.shape == (4, 18)
        assert movie.states.dtype == np.bool_
        assert movie.n_frames == 4

    def test_decodes_the_right_buttons(self, bk2_simple: Path):
        movie = parse_bk2(bk2_simple)
        names = movie.button_names
        # frame 0 is idle
        assert not movie.states[0].any()
        # frame 1 holds P1 Start
        assert movie.states[1, names.index("P1 Start")]
        assert movie.states[1].sum() == 1
        # frame 3 holds P1 Right and P1 A
        assert movie.states[3, names.index("P1 Right")]
        assert movie.states[3, names.index("P1 A")]
        assert movie.states[3].sum() == 2

    def test_header_parsed(self, bk2_simple: Path):
        movie = parse_bk2(bk2_simple)
        assert movie.platform == "NES"
        assert movie.core == "NesHawk"
        assert movie.game_name == "Super Mario Bros"
        assert movie.rom_hashes == {"sha1-file": "ea343f4e445a9050d4b4fbac2c77d0693b1d0922"}
        assert movie.header["rerecordCount"] == "42"

    def test_sync_settings_json(self, bk2_simple: Path):
        movie = parse_bk2(bk2_simple)
        assert movie.sync_settings["o"]["RegionOverride"] == 0

    def test_single_controller_movie(self, bk2_p1_only: Path):
        movie = parse_bk2(bk2_p1_only)
        assert movie.states.shape == (2, 10)
        names = movie.button_names
        assert movie.states[1, names.index("P1 Right")]
        assert movie.states[1, names.index("P1 B")]
        assert movie.states[1, names.index("P1 A")]

    def test_gzip_wrapped_is_transparent(self, tmp_path: Path):
        rows = ["|..|...R....|........|"]
        path = make_bk2(tmp_path / "gz.bk2", rows, gzip_it=True)
        assert sniff(path).gzipped is True
        movie = parse_bk2(path)
        assert movie.n_frames == 1

    def test_space_counts_as_released(self, tmp_path: Path):
        rows = ["|  |        |        |"]
        path = make_bk2(tmp_path / "spaces.bk2", rows)
        assert not parse_bk2(path).states.any()

    def test_console_reset_column(self, tmp_path: Path):
        rows = ["|.r|........|........|", "|..|........|........|"]
        path = make_bk2(tmp_path / "reset.bk2", rows)
        movie = parse_bk2(path)
        frames = console_button_frames(movie.states, movie.button_names, "Reset")
        assert frames.tolist() == [0]

    def test_savestate_anchor_detected(self, tmp_path: Path):
        path = make_bk2(
            tmp_path / "state.bk2",
            ["|..|........|........|"],
            extra_members={"BizState 1.0": "binary-ish", "BizVersion.txt": "2.9"},
        )
        assert starts_from_savestate(parse_bk2(path)) is True

    def test_plain_movie_is_not_savestate_anchored(self, bk2_simple: Path):
        assert starts_from_savestate(parse_bk2(bk2_simple)) is False


class TestParseErrors:
    def test_missing_log_key(self, tmp_path: Path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("Header.txt", "Platform NES\n")
            z.writestr("Input Log.txt", "[Input]\n|..|........|\n[/Input]\n")
        path = tmp_path / "nokey.bk2"
        path.write_bytes(buf.getvalue())
        with pytest.raises(Bk2ParseError, match="no 'LogKey:' line"):
            parse_bk2(path)

    def test_wrong_field_count(self, tmp_path: Path):
        path = make_bk2(tmp_path / "badfields.bk2", ["|..|........|"])
        with pytest.raises(Bk2ParseError, match="declares 3 field"):
            parse_bk2(path)

    def test_wrong_field_width(self, tmp_path: Path):
        path = make_bk2(tmp_path / "badwidth.bk2", ["|..|.....|........|"])
        with pytest.raises(Bk2ParseError, match="should be 8 char"):
            parse_bk2(path)

    def test_reports_the_offending_frame(self, tmp_path: Path):
        rows = ["|..|........|........|"] * 5 + ["|..|...|........|"]
        path = make_bk2(tmp_path / "frame5.bk2", rows)
        with pytest.raises(Bk2ParseError, match="frame 5"):
            parse_bk2(path)

    def test_no_frame_rows(self, tmp_path: Path):
        path = make_bk2(tmp_path / "empty.bk2", [])
        with pytest.raises(Bk2ParseError, match="no frame rows"):
            parse_bk2(path)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_bk2(tmp_path / "nope.bk2")


class TestActionBytes:
    def test_bit_layout_matches_nes_py(self, bk2_simple: Path):
        movie = parse_bk2(bk2_simple)
        actions = actions_from_states(movie.states, movie.button_names, player=1)
        assert actions.dtype == np.uint8
        assert actions.tolist() == [0x00, 0x08, 0x80, 0x81]

    def test_player_two_is_separable(self, tmp_path: Path):
        rows = ["|..|...R....|.......A|"]
        path = make_bk2(tmp_path / "twoplayer.bk2", rows)
        movie = parse_bk2(path)
        assert actions_from_states(movie.states, movie.button_names, 1).tolist() == [0x80]
        assert actions_from_states(movie.states, movie.button_names, 2).tolist() == [0x01]

    def test_unknown_player_raises(self, bk2_p1_only: Path):
        movie = parse_bk2(bk2_p1_only)
        with pytest.raises(ValueError, match="no P2 buttons"):
            actions_from_states(movie.states, movie.button_names, player=2)
