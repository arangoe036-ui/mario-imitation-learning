"""Format sniffing: every non-bk2 input must fail with a message that says why."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tasdata.bk2 import parse_bk2
from tasdata.formats import (
    CorruptMovieError,
    MovieFormat,
    UnsupportedMovieFormatError,
    sniff,
)

from .conftest import make_bk2

# A realistic FCEUX .fm2 preamble, taken from the shape TASVideos actually serves.
FM2 = (
    "version 3\n"
    "emuVersion 20605\n"
    "rerecordCount 13340\n"
    "palFlag 0\n"
    "romFilename Super_Mario_Bros._(JU)_(PRG0)_[!]\n"
    "romChecksum base64:jjYwGG411HcjG/j9UOVM3Q==\n"
    "guid 78C6A1CF-222D-75BB-2AE5-4FF4F78F8F21\n"
    "fourscore 0\n"
    "port0 1\nport1 0\nport2 0\n"
    "|0|........|||\n|0|R.......|||\n"
)


def write(tmp_path: Path, name: str, data: bytes | str) -> Path:
    path = tmp_path / name
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


class TestSniff:
    def test_bk2(self, bk2_simple: Path):
        assert sniff(bk2_simple).format is MovieFormat.BK2

    @pytest.mark.parametrize(
        "name,data,expected",
        [
            ("m.fm2", FM2, MovieFormat.FM2),
            ("m.fcm", b"FCM\x1a" + b"\0" * 64, MovieFormat.FCM),
            ("m.fmv", b"FMV\x1a" + b"\0" * 64, MovieFormat.FMV),
            ("m.smv", b"SMV\x1a" + b"\0" * 64, MovieFormat.SMV),
            ("m.vbm", b"VBM\x1a" + b"\0" * 64, MovieFormat.VBM),
            ("m.gmv", b"Gens Movie TEST9" + b"\0" * 64, MovieFormat.GMV),
            ("m.m64", b"M64\x1a" + b"\0" * 64, MovieFormat.M64),
            ("m.dtm", b"DTM\x1a" + b"\0" * 64, MovieFormat.DTM),
            ("m.mcm", b"MDFNMOVI" + b"\0" * 64, MovieFormat.MCM),
        ],
    )
    def test_foreign_formats(self, tmp_path: Path, name, data, expected):
        assert sniff(write(tmp_path, name, data)).format is expected

    def test_bkm(self, tmp_path: Path):
        data = "MovieVersion 1\nemuVersion 1.0\nGameName x\n[Input]\n|0|..|\n"
        assert sniff(write(tmp_path, "m.bkm", data)).format is MovieFormat.BKM

    def test_unknown(self, tmp_path: Path):
        assert sniff(write(tmp_path, "m.bin", b"\x99\x88\x77nonsense")).format is MovieFormat.UNKNOWN

    def test_empty_file(self, tmp_path: Path):
        with pytest.raises(CorruptMovieError, match="empty"):
            sniff(write(tmp_path, "m.bk2", b""))

    def test_single_file_zip_of_something_unrecognised(self, tmp_path: Path):
        """A one-member zip is unwrapped, then rejected on the member's content."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("Header.txt", "Platform NES\n")
        with pytest.raises(CorruptMovieError, match="not a recognised movie"):
            sniff(write(tmp_path, "m.bk2", buf.getvalue()))

    def test_multi_file_zip_without_input_log(self, tmp_path: Path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("Header.txt", "Platform NES\n")
            z.writestr("Comments.txt", "hello\n")
        with pytest.raises(CorruptMovieError, match="no 'Input Log.txt'"):
            sniff(write(tmp_path, "m.bk2", buf.getvalue()))

    def test_lsmv_zip(self, tmp_path: Path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("systemid", "lsnes-rr1")
            z.writestr("gametype", "snes_ntsc")
        assert sniff(write(tmp_path, "m.lsmv", buf.getvalue())).format is MovieFormat.LSMV


class TestErrorMessages:
    def test_fm2_handed_to_the_bk2_parser_points_at_the_dispatcher(self, tmp_path: Path):
        """fm2 is supported, just not by parse_bk2 -- say so, do not say 'unsupported'."""
        path = write(tmp_path, "happylee.fm2", FM2)
        with pytest.raises(UnsupportedMovieFormatError) as excinfo:
            parse_bk2(path)
        message = str(excinfo.value)
        assert "FCEUX .fm2" in message
        assert "parse_movie" in message
        assert excinfo.value.format is MovieFormat.FM2

    def test_unsupported_nes_format_points_at_conversion(self, tmp_path: Path):
        path = write(tmp_path, "old.fcm", b"FCM\x1a" + b"\0" * 64)
        with pytest.raises(UnsupportedMovieFormatError) as excinfo:
            parse_bk2(path)
        message = str(excinfo.value)
        assert "only reads BizHawk .bk2 and FCEUX .fm2" in message
        assert "re-save as .fm2 or .bk2" in message

    def test_snes_error_explains_it_is_hopeless(self, tmp_path: Path):
        path = write(tmp_path, "lagoon.smv", b"SMV\x1a" + b"\0" * 64)
        with pytest.raises(UnsupportedMovieFormatError, match="targets SNES"):
            parse_bk2(path)

    def test_n64_error_mentions_platform(self, tmp_path: Path):
        path = write(tmp_path, "sm64.m64", b"M64\x1a" + b"\0" * 64)
        with pytest.raises(UnsupportedMovieFormatError, match="Nintendo 64"):
            parse_bk2(path)

    def test_error_includes_the_filename(self, tmp_path: Path):
        path = write(tmp_path, "distinctive-name.fm2", FM2)
        with pytest.raises(UnsupportedMovieFormatError, match="distinctive-name.fm2"):
            parse_bk2(path)


class TestTasproj:
    def _make(self, tmp_path: Path) -> Path:
        return make_bk2(
            tmp_path / "project.tasproj",
            ["|..|........|........|"],
            extra_members={"Markers.txt": "0\tstart\n", "ClientSettings.json": "{}"},
        )

    def test_rejected_by_default_with_a_hint(self, tmp_path: Path):
        with pytest.raises(UnsupportedMovieFormatError) as excinfo:
            parse_bk2(self._make(tmp_path))
        assert "tasproj" in str(excinfo.value).lower()
        assert "allow_tasproj=True" in str(excinfo.value)

    def test_accepted_when_opted_in(self, tmp_path: Path):
        movie = parse_bk2(self._make(tmp_path), allow_tasproj=True)
        assert movie.n_frames == 1

    def test_detected_by_marker_even_with_bk2_extension(self, tmp_path: Path):
        path = make_bk2(
            tmp_path / "mislabelled.bk2",
            ["|..|........|........|"],
            extra_members={"Markers.txt": "0\tstart\n"},
        )
        assert sniff(path).format is MovieFormat.TASPROJ
