"""On-disk layout of one captured run, and helpers to read it back.

Shared by ``tasdata run`` and ``tasdata batch`` so a single run and a batch member
are byte-for-byte the same shape.

    frames.npy         uint8  (n_obs, H, W)      downscaled grayscale observations
    actions.npy        uint8  (n_frames,)        one action byte per frame, ALWAYS
    button_states.npy  bool   (n_frames, n_btn)  raw per-button matrix
    trace.npy          int32  (n_frames, 13)     decoded RAM per frame
    frame_indices.npy  int64  (n_obs,)           movie frame each observation is
    movie.json / sync.json / manifest.json

``actions.npy`` and ``trace.npy`` are always full rate regardless of
``--frame-skip``: the inputs are the labels, and they cost ~3 MB against ~450 MB of
images. Only ``frames.npy`` is ever thinned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .movie import Movie
from .replay import ReplayResult
from .verify import SyncReport

MANIFEST = "manifest.json"


def write_run_dataset(
    out_dir: Path | str,
    movie: Movie,
    result: ReplayResult,
    report: SyncReport,
    *,
    extra: dict | None = None,
) -> Path:
    """Write one run's arrays and metadata to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # frames.npy may already exist as a memmap written during replay.
    if not isinstance(result.frames, np.memmap) and result.frames.size:
        np.save(out_dir / "frames.npy", np.asarray(result.frames))
    np.save(out_dir / "actions.npy", result.actions)
    np.save(out_dir / "trace.npy", result.trace)
    np.save(out_dir / "frame_indices.npy", result.frame_indices)
    np.save(out_dir / "button_states.npy", movie.states)
    (out_dir / "movie.json").write_text(json.dumps(movie.to_dict(), indent=2))
    (out_dir / "sync.json").write_text(json.dumps(report.to_dict(), indent=2))

    manifest = {
        "tasdata_version": __version__,
        "backend": result.backend,
        "movie": str(movie.path),
        "movie_format": movie.format.value,
        "movie_pal": movie.pal,
        "rom": str(result.rom_path),
        "rom_sha1_file": result.rom.sha1_file,
        "rom_md5_prgchr": result.rom.md5_prgchr,
        "rom_matches_movie": result.rom_check.matched,
        "n_frames": result.n_frames,
        "frame_skip": result.frame_skip,
        "observation_shape": list(result.observation_shape),
        "n_observations": int(len(result.frame_indices)),
        "trace_columns": list(result.trace_columns),
        "button_names": movie.button_names,
        "wall_seconds": round(result.wall_seconds, 2),
        "synced": report.passed,
        "diverged_at": report.diverged_at,
        "reason": report.reason,
        "levels_reached": report.levels_reached,
        "files": {
            "frames": "frames.npy" if result.frames.size else None,
            "actions": "actions.npy",
            "trace": "trace.npy",
            "frame_indices": "frame_indices.npy",
            "button_states": "button_states.npy",
        },
    }
    if extra:
        manifest.update(extra)
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out_dir


def dir_bytes(path: Path | str) -> int:
    """Total size of the files in a run directory."""
    return sum(p.stat().st_size for p in Path(path).glob("*") if p.is_file())


@dataclass
class LoadedRun:
    """A captured run read back from disk. Arrays are memory-mapped where large."""

    path: Path
    manifest: dict
    actions: np.ndarray
    trace: np.ndarray

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def n_frames(self) -> int:
        return int(self.manifest["n_frames"])

    @property
    def synced(self) -> bool:
        return bool(self.manifest.get("synced"))

    @property
    def category(self) -> str:
        return str(self.manifest.get("category", "unknown"))

    @property
    def chain(self) -> str:
        return str(self.manifest.get("chain", ""))

    @property
    def route(self) -> str:
        """Measured route (``warpless``/``warps``/``partial-N``), else the category.

        Used as the split stratum: it is coarser than ``category``, which keeps
        one-group categories like ``all-items`` from being stranded entirely in
        train. all-items *is* a 32-level route, so it belongs with warpless.
        """
        return str(self.manifest.get("measured_route") or self.category)

    @property
    def label(self) -> str:
        return str(self.manifest.get("label", self.path.name))

    def frames(self) -> np.ndarray:
        return np.load(self.path / "frames.npy", mmap_mode="r")


def load_run_dir(path: Path | str) -> LoadedRun:
    """Read a run directory's manifest plus its actions and trace."""
    path = Path(path)
    manifest = json.loads((path / MANIFEST).read_text())
    return LoadedRun(
        path=path,
        manifest=manifest,
        actions=np.load(path / "actions.npy"),
        trace=np.load(path / "trace.npy", mmap_mode="r"),
    )


def discover_runs(root: Path | str, *, synced_only: bool = False) -> list[LoadedRun]:
    """Load every run directory under ``root``, sorted by name."""
    root = Path(root)
    runs: list[LoadedRun] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / MANIFEST).exists():
            run = load_run_dir(child)
            if synced_only and not run.synced:
                continue
            runs.append(run)
    return runs
