"""iNES ROM identity.

Movie formats disagree about how to fingerprint a ROM:

* BizHawk ``.bk2`` stores ``SHA1`` of the **whole file**, header included.
* FCEUX ``.fm2`` stores ``romChecksum base64:...``, which is the MD5 of the
  **PRG + CHR data only**, with the 16-byte iNES header stripped.

Both are computed here so a movie can be checked against a ROM whichever format
it came from.  The header fields are parsed too, because region is the single
most common cause of an instant desync and it is worth reporting explicitly.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path

INES_MAGIC = b"NES\x1a"
INES_HEADER_SIZE = 16


@dataclass(frozen=True)
class NesRom:
    """Hashes and header fields for a ``.nes`` file."""

    path: Path
    size: int
    has_ines_header: bool
    prg_banks: int          # 16 KB units
    chr_banks: int          # 8 KB units
    mapper: int
    has_trainer: bool
    #: iNES flags9 bit 0 / flags10 region bits: True when the header says PAL.
    header_says_pal: bool
    sha1_file: str          # BizHawk's fingerprint
    md5_file: str
    md5_prgchr: str         # FCEUX's fingerprint
    sha1_prgchr: str

    @property
    def fm2_checksum(self) -> str:
        """The ``romChecksum`` string an fm2 would carry for this ROM."""
        return "base64:" + base64.b64encode(bytes.fromhex(self.md5_prgchr)).decode()

    def summary(self) -> str:
        region = "PAL (per header)" if self.header_says_pal else "NTSC (per header)"
        return (
            f"{self.path.name}: {self.size} bytes, mapper {self.mapper}, "
            f"PRG {self.prg_banks}x16K, CHR {self.chr_banks}x8K, {region}\n"
            f"  sha1(file)    {self.sha1_file}   <- bk2 'SHA1'\n"
            f"  md5(prg+chr)  {self.md5_prgchr}   <- fm2 'romChecksum'\n"
            f"  fm2 checksum  {self.fm2_checksum}"
        )


def load_rom(path: Path | str) -> NesRom:
    """Read a ``.nes`` file and compute both fingerprints."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROM not found: {path}")
    data = path.read_bytes()
    if len(data) < INES_HEADER_SIZE:
        raise ValueError(f"{path.name}: too small to be a ROM ({len(data)} bytes)")

    has_header = data[:4] == INES_MAGIC
    if has_header:
        prg_banks = data[4]
        chr_banks = data[5]
        flags6, flags7, flags9 = data[6], data[7], data[9]
        mapper = ((flags7 >> 4) << 4) | (flags6 >> 4)
        has_trainer = bool(flags6 & 0b100)
        header_pal = bool(flags9 & 0b1)
        body = data[INES_HEADER_SIZE:]
        if has_trainer:
            body = body[512:]
    else:
        prg_banks = chr_banks = mapper = 0
        has_trainer = False
        header_pal = False
        body = data

    return NesRom(
        path=path,
        size=len(data),
        has_ines_header=has_header,
        prg_banks=prg_banks,
        chr_banks=chr_banks,
        mapper=mapper,
        has_trainer=has_trainer,
        header_says_pal=header_pal,
        sha1_file=hashlib.sha1(data).hexdigest(),
        md5_file=hashlib.md5(data).hexdigest(),
        md5_prgchr=hashlib.md5(body).hexdigest(),
        sha1_prgchr=hashlib.sha1(body).hexdigest(),
    )


def decode_fm2_checksum(value: str) -> str | None:
    """``"base64:jjYwGG..."`` -> lowercase hex MD5, or None if unrecognised."""
    value = value.strip()
    if value.startswith("base64:"):
        try:
            return base64.b64decode(value[7:]).hex()
        except Exception:
            return None
    cleaned = value.lower().removeprefix("0x")
    if len(cleaned) == 32 and all(c in "0123456789abcdef" for c in cleaned):
        return cleaned
    return None


@dataclass(frozen=True)
class RomCheck:
    """Result of comparing a movie's recorded ROM fingerprint to a real ROM."""

    checked: bool
    matched: bool | None
    algorithm: str = ""
    expected: str = ""
    actual: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True when verification ran and passed."""
        return self.checked and bool(self.matched)

    def line(self) -> str:
        if not self.checked:
            return f"rom check skipped: {self.detail}"
        verdict = "match" if self.matched else "MISMATCH"
        return f"rom {self.algorithm} {verdict}: movie={self.expected} rom={self.actual}"
