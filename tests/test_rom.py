"""ROM fingerprinting and the movie-vs-ROM verification path."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from tasdata.rom import INES_MAGIC, load_rom
from tasdata.movie import parse_movie

from .conftest import DEFAULT_HEADER, make_bk2
from .test_fm2 import HEADER as FM2_HEADER, make_fm2, ROWS


def make_rom(path: Path, body: bytes = b"\x11" * 1024, *, pal: bool = False) -> Path:
    """A minimal but structurally valid iNES file."""
    header = bytearray(16)
    header[:4] = INES_MAGIC
    header[4] = 2      # PRG banks
    header[5] = 1      # CHR banks
    header[6] = 0x01   # vertical mirroring, mapper low nibble 0
    header[9] = 0x01 if pal else 0x00
    path.write_bytes(bytes(header) + body)
    return path


class TestLoadRom:
    def test_header_fields(self, tmp_path: Path):
        rom = load_rom(make_rom(tmp_path / "g.nes"))
        assert rom.has_ines_header
        assert rom.prg_banks == 2
        assert rom.chr_banks == 1
        assert rom.mapper == 0
        assert rom.has_trainer is False
        assert rom.header_says_pal is False

    def test_pal_header_flag(self, tmp_path: Path):
        assert load_rom(make_rom(tmp_path / "p.nes", pal=True)).header_says_pal is True

    def test_both_fingerprints(self, tmp_path: Path):
        body = b"\x42" * 2048
        path = make_rom(tmp_path / "g.nes", body)
        rom = load_rom(path)
        assert rom.sha1_file == hashlib.sha1(path.read_bytes()).hexdigest()
        # fm2's romChecksum excludes the 16-byte iNES header
        assert rom.md5_prgchr == hashlib.md5(body).hexdigest()
        assert rom.md5_prgchr != rom.md5_file

    def test_fm2_checksum_roundtrip(self, tmp_path: Path):
        rom = load_rom(make_rom(tmp_path / "g.nes"))
        assert rom.fm2_checksum.startswith("base64:")
        raw = base64.b64decode(rom.fm2_checksum[7:])
        assert raw.hex() == rom.md5_prgchr

    def test_summary_mentions_both_hashes(self, tmp_path: Path):
        text = load_rom(make_rom(tmp_path / "g.nes")).summary()
        assert "bk2 'SHA1'" in text
        assert "fm2 'romChecksum'" in text

    def test_missing_rom(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_rom(tmp_path / "nope.nes")

    def test_too_small(self, tmp_path: Path):
        path = tmp_path / "tiny.nes"
        path.write_bytes(b"NES")
        with pytest.raises(ValueError, match="too small"):
            load_rom(path)

    def test_headerless_rom_hashes_whole_file(self, tmp_path: Path):
        path = tmp_path / "raw.bin"
        path.write_bytes(b"\x07" * 4096)
        rom = load_rom(path)
        assert rom.has_ines_header is False
        assert rom.md5_prgchr == rom.md5_file


class TestVerifyRom:
    def test_fm2_matching_rom(self, tmp_path: Path):
        body = b"\x33" * 4096
        rom = load_rom(make_rom(tmp_path / "g.nes", body))
        checksum = base64.b64encode(hashlib.md5(body).digest()).decode()
        header = FM2_HEADER.replace(
            "romChecksum base64:jjYwGG411HcjG/j9UOVM3Q==",
            f"romChecksum base64:{checksum}",
        )
        movie = parse_movie(make_fm2(tmp_path / "m.fm2", ROWS, header=header))
        check = movie.verify_rom(rom)
        assert check.checked and check.matched and check.ok
        assert check.algorithm == "md5-prgchr"

    def test_fm2_mismatching_rom(self, tmp_path: Path):
        rom = load_rom(make_rom(tmp_path / "g.nes"))
        movie = parse_movie(make_fm2(tmp_path / "m.fm2", ROWS))
        check = movie.verify_rom(rom)
        assert check.checked and check.matched is False
        assert "different dump" in check.detail
        assert "MISMATCH" in check.line()

    def test_bk2_uses_sha1_of_whole_file(self, tmp_path: Path):
        path = make_rom(tmp_path / "g.nes")
        rom = load_rom(path)
        header = DEFAULT_HEADER.replace(
            "SHA1 EA343F4E445A9050D4B4FBAC2C77D0693B1D0922",
            f"SHA1 {rom.sha1_file.upper()}",
        )
        movie = parse_movie(make_bk2(tmp_path / "m.bk2", ["|..|........|........|"], header=header))
        check = movie.verify_rom(rom)
        assert check.ok
        assert check.algorithm == "sha1-file"

    def test_no_fingerprint_recorded(self, tmp_path: Path):
        rom = load_rom(make_rom(tmp_path / "g.nes"))
        header = "\n".join(
            l for l in FM2_HEADER.splitlines() if not l.startswith("romChecksum")
        ) + "\n"
        movie = parse_movie(make_fm2(tmp_path / "m.fm2", ROWS, header=header))
        check = movie.verify_rom(rom)
        assert check.checked is False
        assert check.ok is False
        assert "no ROM fingerprint" in check.detail

    def test_accepts_a_path_as_well_as_a_nesrom(self, tmp_path: Path):
        path = make_rom(tmp_path / "g.nes")
        movie = parse_movie(make_fm2(tmp_path / "m.fm2", ROWS))
        assert movie.verify_rom(path).checked is True
