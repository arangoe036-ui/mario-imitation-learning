"""fm2 parser tests: header handling, RLDUTSBA decoding, commands, error paths."""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tasdata.buttons import actions_from_states, console_button_frames
from tasdata.fm2 import CONTROLLER_WIDTH, FM2_BUTTON_ORDER, Fm2ParseError, parse_fm2
from tasdata.formats import MovieFormat, UnsupportedMovieFormatError, sniff
from tasdata.movie import parse_movie
from tasdata.rom import decode_fm2_checksum

# md5 of an all-zero 40 960-byte body, base64-encoded, so tests can assert a
# real round-trip through decode_fm2_checksum.
HEADER = """version 3
emuVersion 22020
rerecordCount 184530
palFlag 0
romFilename Super Mario Bros. (JU) [!]
romChecksum base64:jjYwGG411HcjG/j9UOVM3Q==
guid A706785C-5113-D2C5-DC9C-F02A48F87F17
fourscore 0
microphone 0
port0 1
port1 1
port2 0
FDS 0
NewPPU 0
comment author  HappyLee & Mars608
"""


def make_fm2(
    path: Path,
    frame_rows: list[str],
    *,
    header: str = HEADER,
    gzip_it: bool = False,
    zip_it: bool = False,
) -> Path:
    text = header + "\n".join(frame_rows) + "\n"
    data = text.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if zip_it:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(path.name.removesuffix(".zip"), text)
        data = buf.getvalue()
    elif gzip_it:
        data = gzip.compress(data)
    path.write_bytes(data)
    return path


#: One-player rows: |commands|port0|port1|port2|
ROWS = [
    "|0|........|........||",
    "|0|R.......|........||",
    "|0|.L......|........||",
    "|0|..D.....|........||",
    "|0|...U....|........||",
    "|0|....T...|........||",
    "|0|.....S..|........||",
    "|0|......B.|........||",
    "|0|.......A|........||",
    "|0|R......A|........||",
]


class TestButtonOrder:
    def test_order_is_rldutsba(self):
        """The fm2 mnemonic RLDUTSBA: T is sTart, S is Select."""
        assert FM2_BUTTON_ORDER == (
            "Right", "Left", "Down", "Up", "Start", "Select", "B", "A",
        )
        assert CONTROLLER_WIDTH == 8

    def test_each_position_maps_to_the_right_button(self, tmp_path: Path):
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS))
        names = movie.button_names
        expected = ["Right", "Left", "Down", "Up", "Start", "Select", "B", "A"]
        for offset, button in enumerate(expected):
            frame = offset + 1  # row 0 is idle
            col = names.index(f"P1 {button}")
            assert movie.states[frame, col], f"frame {frame} should hold {button}"
            assert movie.states[frame].sum() == 1

    def test_action_bytes_match_nes_py_bit_layout(self, tmp_path: Path):
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS))
        actions = actions_from_states(movie.states, movie.button_names, 1)
        assert actions.dtype == np.uint8
        # RLDUTSBA maps MSB->LSB onto nes-py's byte, so each single press is one bit
        assert actions.tolist() == [
            0x00, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x81,
        ]


class TestHeader:
    def test_fields(self, tmp_path: Path):
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS))
        assert movie.format is MovieFormat.FM2
        assert movie.header["version"] == "3"
        assert movie.header["romFilename"] == "Super Mario Bros. (JU) [!]"
        assert movie.rerecord_count == "184530"
        assert movie.n_frames == len(ROWS)

    def test_rom_checksum_decoded_to_md5(self, tmp_path: Path):
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS))
        assert movie.rom_hashes == {"md5-prgchr": "8e3630186e35d477231bf8fd50e54cdd"}

    def test_palflag_zero_is_ntsc(self, tmp_path: Path):
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS))
        assert movie.pal is False
        assert abs(movie.fps - 60.0988) < 1e-3

    def test_palflag_one_is_pal_and_notes_it(self, tmp_path: Path):
        header = HEADER.replace("palFlag 0", "palFlag 1")
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS, header=header))
        assert movie.pal is True
        assert abs(movie.fps - 50.007) < 1e-3
        assert any("50 Hz" in n for n in movie.notes)

    def test_author_read_from_comment_line(self, tmp_path: Path):
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS))
        assert movie.author == "HappyLee & Mars608"

    def test_missing_checksum_is_noted(self, tmp_path: Path):
        header = "\n".join(
            l for l in HEADER.splitlines() if not l.startswith("romChecksum")
        ) + "\n"
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS, header=header))
        assert movie.rom_hashes == {}
        assert any("no romChecksum" in n for n in movie.notes)

    def test_fds_and_newppu_flags_noted(self, tmp_path: Path):
        header = HEADER.replace("FDS 0", "FDS 1").replace("NewPPU 0", "NewPPU 1")
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS, header=header))
        assert any("Famicom Disk System" in n for n in movie.notes)
        assert any("New PPU" in n for n in movie.notes)

    def test_ram_init_noted(self, tmp_path: Path):
        header = HEADER + "RAMInitOption 2\nRAMInitSeed 837\n"
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS, header=header))
        assert any("power-on RAM init" in n for n in movie.notes)


class TestCommandsField:
    def test_reset_bit(self, tmp_path: Path):
        rows = ["|1|........|........||", "|0|........|........||"]
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", rows))
        assert console_button_frames(movie.states, movie.button_names, "Reset").tolist() == [0]

    def test_power_bit(self, tmp_path: Path):
        rows = ["|2|........|........||"]
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", rows))
        assert console_button_frames(movie.states, movie.button_names, "Power").tolist() == [0]

    def test_combined_bits(self, tmp_path: Path):
        rows = ["|3|........|........||"]
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", rows))
        names = movie.button_names
        assert movie.states[0, names.index("Reset")]
        assert movie.states[0, names.index("Power")]

    def test_empty_commands_field_is_zero(self, tmp_path: Path):
        rows = ["||........|........||"]
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", rows))
        assert not movie.states[0, :5].any()

    def test_non_integer_commands_field(self, tmp_path: Path):
        rows = ["|x|........|........||"]
        with pytest.raises(Fm2ParseError, match="not an integer"):
            parse_fm2(make_fm2(tmp_path / "m.fm2", rows))


class TestControllerPorts:
    def test_only_first_controller_by_default(self, tmp_path: Path):
        """Port 1 input is ignored unless explicitly requested, and that is said."""
        rows = ["|0|........|R.......||"]
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", rows))
        assert all(not n.startswith("P2") for n in movie.button_names)
        assert not movie.states.any()
        assert any("port(s) [1]" in n for n in movie.notes)

    def test_second_controller_when_requested(self, tmp_path: Path):
        rows = ["|0|........|R.......||"]
        movie = parse_fm2(
            make_fm2(tmp_path / "m.fm2", rows), first_controller_only=False
        )
        assert movie.states[0, movie.button_names.index("P2 Right")]
        assert actions_from_states(movie.states, movie.button_names, 2).tolist() == [0x80]

    def test_connected_but_idle_port_is_not_flagged(self, tmp_path: Path):
        """Eight dots on port 1 means connected-and-unused, not 'has input'."""
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS))
        assert not any("controller port" in n for n in movie.notes)

    def test_single_field_rows(self, tmp_path: Path):
        rows = ["|0|R.......|"]
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", rows))
        assert movie.states[0, movie.button_names.index("P1 Right")]


class TestErrors:
    def test_wrong_controller_width(self, tmp_path: Path):
        rows = ["|0|.....|........||"]
        with pytest.raises(Fm2ParseError, match="should be 8 chars"):
            parse_fm2(make_fm2(tmp_path / "m.fm2", rows))

    def test_reports_offending_frame(self, tmp_path: Path):
        rows = ["|0|........|........||"] * 7 + ["|0|...|........||"]
        with pytest.raises(Fm2ParseError, match="frame 7"):
            parse_fm2(make_fm2(tmp_path / "m.fm2", rows))

    def test_no_frame_lines(self, tmp_path: Path):
        with pytest.raises(Fm2ParseError, match="no frame lines"):
            parse_fm2(make_fm2(tmp_path / "m.fm2", []))

    def test_binary_fm2_rejected_clearly(self, tmp_path: Path):
        header = HEADER + "binary 1\n"
        path = make_fm2(tmp_path / "m.fm2", ROWS, header=header)
        with pytest.raises(Fm2ParseError, match="binary fm2"):
            parse_fm2(path)

    def test_savestate_anchored_noted(self, tmp_path: Path):
        header = HEADER + "savestate BASE64:xyz\n"
        movie = parse_fm2(make_fm2(tmp_path / "m.fm2", ROWS, header=header))
        assert movie.savestate_anchored is True

    def test_bk2_passed_to_fm2_parser(self, bk2_simple: Path):
        with pytest.raises(UnsupportedMovieFormatError, match="parse_movie"):
            parse_fm2(bk2_simple)


class TestWrappers:
    def test_gzip_wrapped(self, tmp_path: Path):
        path = make_fm2(tmp_path / "m.fm2", ROWS, gzip_it=True)
        assert sniff(path).gzipped is True
        assert parse_fm2(path).n_frames == len(ROWS)

    def test_zip_wrapped(self, tmp_path: Path):
        """TASVideos ships published movies as a zip holding one file."""
        path = make_fm2(tmp_path / "pub.fm2.zip", ROWS, zip_it=True)
        result = sniff(path)
        assert result.format is MovieFormat.FM2
        assert result.inner_name == "pub.fm2"
        assert parse_fm2(path).n_frames == len(ROWS)


class TestDispatch:
    def test_parse_movie_routes_fm2(self, tmp_path: Path):
        movie = parse_movie(make_fm2(tmp_path / "m.fm2", ROWS))
        assert movie.format is MovieFormat.FM2

    def test_parse_movie_routes_bk2(self, bk2_simple: Path):
        assert parse_movie(bk2_simple).format is MovieFormat.BK2


class TestChecksumDecoding:
    def test_base64(self):
        assert decode_fm2_checksum("base64:jjYwGG411HcjG/j9UOVM3Q==") == (
            "8e3630186e35d477231bf8fd50e54cdd"
        )

    def test_bare_hex(self):
        value = "8e3630186e35d477231bf8fd50e54cdd"
        assert decode_fm2_checksum(value) == value

    def test_garbage(self):
        assert decode_fm2_checksum("base64:!!!not base64!!!") is None
        assert decode_fm2_checksum("nope") is None
