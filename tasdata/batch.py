"""Batch capture: run every movie in a shortlist, never aborting on one failure.

A single bad movie in a 34-run batch must not cost the other 33. Every run is
wrapped: a parse error, a ROM mismatch, an FCEUX crash or a frame-count mismatch is
recorded against that run and the batch moves on. The summary at the end lists what
failed and why, and the report is written to disk whether or not everything worked.

Note that a run *desyncing* is not a failure of capture -- the data is written and
the verdict recorded. Whether to train on a desynced run is a separate decision,
which is why ``sync.json`` and the manifest both carry it.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .curate import Candidate
from .dataset import dir_bytes, write_run_dataset
from .fceux_backend import FceuxReplayer
from .movie import parse_movie
from .rom import load_rom
from .verify import verify_smb


#: Distinct-level counts that identify a route, measured from the RAM trace rather
#: than guessed from a filename.
ROUTE_BY_LEVELS: tuple[tuple[int, int, str], ...] = (
    (30, 32, "warpless"),        # every level
    (7, 12, "warps"),            # 1-1,1-2,4-1,4-2,8-1..8-4
)


def route_from_levels(n_levels: int) -> str:
    """Name a route from how many distinct levels the run actually visited."""
    for low, high, name in ROUTE_BY_LEVELS:
        if low <= n_levels <= high:
            return name
    if n_levels <= 0:
        return "none"
    return f"partial-{n_levels}"


@dataclass
class Measurement:
    """Cheap RAM-only probe of one candidate: does it sync, and how far does it go?"""

    label: str
    declared_category: str
    n_frames: int
    synced: bool
    measured_levels: int
    furthest: str
    route: str
    seconds: float
    reason: str = ""
    error: str | None = None

    @property
    def duration(self) -> str:
        secs = self.n_frames / 60.0988
        return f"{int(secs // 60)}m{secs % 60:04.1f}s"

    def row(self) -> str:
        if self.error:
            return f"  {self.label:16s} {'ERROR':10s} {str(self.error)[:60]}"
        mark = "sync" if self.synced else "DESYNC"
        return (
            f"  {self.label:16s} {self.declared_category:20s} {self.n_frames:7,d}f "
            f"{self.duration:>8s} {self.measured_levels:3d}L {self.furthest:>4s} "
            f"{self.route:12s} {mark}"
        )


def measure_one(
    candidate: Candidate, rom, *, stall_frames: int = 2000
) -> Measurement:
    """Replay a candidate RAM-only and report what it actually does."""
    started = time.perf_counter()
    try:
        movie = parse_movie(candidate.path)
        result = FceuxReplayer(rom.path, capture_frames=False).replay(movie)
        report = verify_smb(
            result.trace,
            movie_name=str(movie.path),
            rom_name=str(rom.path),
            stall_frames=stall_frames,
            rom_matches_movie=result.rom_check.matched,
            movie_is_pal=movie.pal,
        )
        levels = report.levels_reached
        return Measurement(
            label=candidate.label,
            declared_category=candidate.category,
            n_frames=result.n_frames,
            synced=report.passed,
            measured_levels=len(levels),
            furthest=levels[-1] if levels else "-",
            route=route_from_levels(len(levels)),
            seconds=time.perf_counter() - started,
            reason=report.reason,
        )
    except Exception as exc:
        return Measurement(
            label=candidate.label,
            declared_category=candidate.category,
            n_frames=candidate.n_frames,
            synced=False,
            measured_levels=0,
            furthest="-",
            route="none",
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300],
        )


def measure_batch(
    candidates: list[Candidate],
    rom_path: Path | str,
    *,
    stall_frames: int = 2000,
    report_path: Path | str | None = None,
    on_event=None,
) -> list[Measurement]:
    """RAM-only probe of every candidate: measured levels and sync status."""
    rom = load_rom(rom_path)
    out: list[Measurement] = []
    for i, cand in enumerate(candidates, 1):
        m = measure_one(cand, rom, stall_frames=stall_frames)
        out.append(m)
        if on_event:
            on_event(f"[{i}/{len(candidates)}]{m.row()}")
        if report_path:
            Path(report_path).write_text(
                json.dumps([asdict(x) for x in out], indent=2)
            )
    return out


def safe_name(label: str) -> str:
    """``"pub 3728"`` -> ``"pub-3728"`` for use as a directory name."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in label).strip("-")


@dataclass
class RunOutcome:
    """What happened to one movie."""

    label: str
    category: str
    authors: str
    movie: str
    out_dir: str | None
    captured: bool
    synced: bool | None
    n_frames: int
    wall_seconds: float
    bytes_written: int
    levels_reached: int
    reason: str = ""
    error: str | None = None

    def row(self) -> str:
        if not self.captured:
            return f"  FAIL    {self.label:22s} {str(self.error)[:70]}"
        mark = "SYNCED " if self.synced else "DESYNC "
        return (
            f"  {mark} {self.label:22s} {self.category:20s} {self.n_frames:7d}f "
            f"{self.levels_reached:3d}L {self.wall_seconds:6.1f}s "
            f"{self.bytes_written / (1 << 20):7.1f} MiB"
        )


@dataclass
class BatchReport:
    """Aggregate result of a batch."""

    outcomes: list[RunOutcome] = field(default_factory=list)
    wall_seconds: float = 0.0
    rom: str = ""
    backend: str = ""

    @property
    def captured(self) -> list[RunOutcome]:
        return [o for o in self.outcomes if o.captured]

    @property
    def failed(self) -> list[RunOutcome]:
        return [o for o in self.outcomes if not o.captured]

    @property
    def synced(self) -> list[RunOutcome]:
        return [o for o in self.outcomes if o.captured and o.synced]

    @property
    def desynced(self) -> list[RunOutcome]:
        return [o for o in self.outcomes if o.captured and o.synced is False]

    def to_dict(self) -> dict:
        return {
            "rom": self.rom,
            "backend": self.backend,
            "wall_seconds": round(self.wall_seconds, 1),
            "n_attempted": len(self.outcomes),
            "n_captured": len(self.captured),
            "n_synced": len(self.synced),
            "n_desynced": len(self.desynced),
            "n_failed": len(self.failed),
            "total_frames": sum(o.n_frames for o in self.captured),
            "total_bytes": sum(o.bytes_written for o in self.captured),
            "outcomes": [asdict(o) for o in self.outcomes],
        }

    def summary(self) -> str:
        lines = [
            "",
            "=" * 96,
            f"BATCH SUMMARY  ({self.wall_seconds / 60:.1f} min wall clock)",
            "=" * 96,
        ]
        lines += [o.row() for o in self.outcomes]
        frames = sum(o.n_frames for o in self.captured)
        size = sum(o.bytes_written for o in self.captured)
        lines += [
            "-" * 96,
            f"attempted {len(self.outcomes)} | captured {len(self.captured)} "
            f"| synced {len(self.synced)} | desynced {len(self.desynced)} "
            f"| failed {len(self.failed)}",
            f"{frames:,} frames ({frames / 60.0988 / 3600:.1f} h) "
            f"| {size / (1 << 30):.2f} GiB on disk",
        ]
        if self.failed:
            lines.append("")
            lines.append("failures:")
            for o in self.failed:
                lines.append(f"  {o.label:24s} {o.error}")
        if self.desynced:
            lines.append("")
            lines.append("desynced (captured, but do not train on these):")
            for o in self.desynced:
                lines.append(f"  {o.label:24s} {o.reason[:70]}")
        return "\n".join(lines)


def capture_one(
    candidate: Candidate,
    rom,
    out_root: Path,
    *,
    observation_shape: tuple[int, int] = (84, 84),
    frame_skip: int = 1,
    expect_level: str | None = None,
    stall_frames: int = 2000,
    progress=None,
) -> RunOutcome:
    """Capture a single candidate. Raises nothing: failures come back as data."""
    out_dir = out_root / safe_name(candidate.label)
    started = time.perf_counter()
    try:
        movie = parse_movie(candidate.path)
        replayer = FceuxReplayer(
            rom.path,
            observation_shape=observation_shape,
            frame_skip=frame_skip,
            capture_frames=True,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        result = replayer.replay(
            movie, frames_path=out_dir / "frames.npy", progress=progress
        )
        report = verify_smb(
            result.trace,
            movie_name=str(movie.path),
            rom_name=str(rom.path),
            expect_level=expect_level,
            stall_frames=stall_frames,
            rom_matches_movie=result.rom_check.matched,
            rom_check_detail=result.rom_check.line(),
            movie_is_pal=movie.pal,
            replay_warnings=result.warnings,
        )
        write_run_dataset(
            out_dir,
            movie,
            result,
            report,
            extra={
                "label": candidate.label,
                "category": candidate.category,
                "authors": candidate.authors,
                "source": candidate.source,
                "source_id": candidate.source_id,
                "chain": candidate.chain,
                "chain_position": candidate.chain_position,
                "converted_from": candidate.converted_from,
                "measured_levels": len(report.levels_reached),
                "measured_route": route_from_levels(len(report.levels_reached)),
                "furthest_level": (
                    report.levels_reached[-1] if report.levels_reached else None
                ),
            },
        )
        return RunOutcome(
            label=candidate.label,
            category=candidate.category,
            authors=candidate.authors,
            movie=candidate.path,
            out_dir=str(out_dir),
            captured=True,
            synced=report.passed,
            n_frames=result.n_frames,
            wall_seconds=time.perf_counter() - started,
            bytes_written=dir_bytes(out_dir),
            levels_reached=len(report.levels_reached),
            reason=report.reason,
        )
    except Exception as exc:  # one bad movie must not end the batch
        return RunOutcome(
            label=candidate.label,
            category=candidate.category,
            authors=candidate.authors,
            movie=candidate.path,
            out_dir=None,
            captured=False,
            synced=None,
            n_frames=0,
            wall_seconds=time.perf_counter() - started,
            bytes_written=0,
            levels_reached=0,
            error=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:400],
        )


def run_batch(
    candidates: list[Candidate],
    rom_path: Path | str,
    out_root: Path | str,
    *,
    observation_shape: tuple[int, int] = (84, 84),
    frame_skip: int = 1,
    expect_level: str | None = None,
    stall_frames: int = 2000,
    report_path: Path | str | None = None,
    on_event=None,
) -> BatchReport:
    """Capture every candidate, logging failures and continuing."""
    rom = load_rom(rom_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report = BatchReport(rom=str(rom.path))
    started = time.perf_counter()

    for i, cand in enumerate(candidates, 1):
        if on_event:
            on_event(
                f"[{i}/{len(candidates)}] {cand.label} ({cand.category}, "
                f"{cand.n_frames} frames)"
            )
        outcome = capture_one(
            cand,
            rom,
            out_root,
            observation_shape=observation_shape,
            frame_skip=frame_skip,
            expect_level=expect_level,
            stall_frames=stall_frames,
        )
        report.outcomes.append(outcome)
        if on_event:
            on_event(outcome.row().strip())
        # Write the report after every run so a crash mid-batch loses nothing.
        if report_path:
            report.wall_seconds = time.perf_counter() - started
            Path(report_path).write_text(json.dumps(report.to_dict(), indent=2))

    report.wall_seconds = time.perf_counter() - started
    if report.captured:
        report.backend = "fceux"
    if report_path:
        Path(report_path).write_text(json.dumps(report.to_dict(), indent=2))
    return report
