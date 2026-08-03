"""Run-directory layout: write, read back, discover."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tasdata.dataset import LoadedRun, dir_bytes, discover_runs, load_run_dir, write_run_dataset
from tasdata.movie import parse_movie
from tasdata.replay import ReplayResult
from tasdata.rom import load_rom
from tasdata.verify import verify_smb

from .conftest import synthetic_trace
from .test_fm2 import HEADER as FM2_HEADER, make_fm2
from .test_rom import make_rom


def build(tmp_path: Path, name: str = "run", frames: int = 40, skip: int = 1):
    rom = load_rom(make_rom(tmp_path / "g.nes"))
    movie = parse_movie(
        make_fm2(tmp_path / f"{name}.fm2", ["|0|R.......|........||"] * frames)
    )
    n_obs = len(range(0, frames, skip))
    result = ReplayResult(
        movie=movie,
        rom_path=rom.path,
        frames=np.zeros((n_obs, 84, 84), np.uint8),
        trace=synthetic_trace([{"x_position": i} for i in range(frames)]),
        frame_indices=np.arange(0, frames, skip),
        n_frames=frames,
        frame_skip=skip,
        observation_shape=(84, 84),
        wall_seconds=1.5,
        rom=rom,
        rom_check=movie.verify_rom(rom),
        backend="fceux 2.6.6",
    )
    result.actions = np.full(frames, 0x80, np.uint8)
    report = verify_smb(result.trace)
    out = write_run_dataset(
        tmp_path / name, movie, result, report, extra={"category": "warpless", "chain": "c/1"}
    )
    return out


class TestWriteRunDataset:
    def test_all_arrays_written(self, tmp_path: Path):
        out = build(tmp_path)
        for f in ("frames.npy", "actions.npy", "trace.npy", "frame_indices.npy",
                  "button_states.npy", "movie.json", "sync.json", "manifest.json"):
            assert (out / f).exists(), f

    def test_actions_are_full_rate_even_with_frame_skip(self, tmp_path: Path):
        """The labels must never be thinned."""
        out = build(tmp_path, frames=40, skip=4)
        assert np.load(out / "actions.npy").shape == (40,)
        assert np.load(out / "trace.npy").shape[0] == 40
        assert np.load(out / "frames.npy").shape[0] == 10

    def test_manifest_carries_category_and_provenance(self, tmp_path: Path):
        out = build(tmp_path)
        m = json.loads((out / "manifest.json").read_text())
        assert m["category"] == "warpless"
        assert m["chain"] == "c/1"
        assert m["backend"] == "fceux 2.6.6"
        assert m["rom_md5_prgchr"]
        assert m["button_names"]

    def test_dir_bytes_counts_files(self, tmp_path: Path):
        out = build(tmp_path)
        assert dir_bytes(out) > 0


class TestLoadRunDir:
    def test_roundtrip(self, tmp_path: Path):
        out = build(tmp_path, frames=30)
        run = load_run_dir(out)
        assert isinstance(run, LoadedRun)
        assert run.n_frames == 30
        assert run.actions.shape == (30,)
        assert run.category == "warpless"
        assert run.chain == "c/1"

    def test_frames_are_memmapped(self, tmp_path: Path):
        out = build(tmp_path)
        assert load_run_dir(out).frames().shape[1:] == (84, 84)


class TestDiscoverRuns:
    def test_finds_all_runs(self, tmp_path: Path):
        root = tmp_path / "runs"
        root.mkdir()
        for name in ("a", "b"):
            build(tmp_path, name).rename(root / name)
        assert len(discover_runs(root)) == 2

    def test_ignores_non_run_directories(self, tmp_path: Path):
        root = tmp_path / "runs"
        root.mkdir()
        (root / "not-a-run").mkdir()
        (root / "stray.txt").write_text("x")
        assert discover_runs(root) == []

    def test_synced_only_filter(self, tmp_path: Path):
        root = tmp_path / "runs"
        root.mkdir()
        out = build(tmp_path, "a")
        out.rename(root / "a")
        m = json.loads((root / "a" / "manifest.json").read_text())
        m["synced"] = False
        (root / "a" / "manifest.json").write_text(json.dumps(m))
        assert len(discover_runs(root)) == 1
        assert discover_runs(root, synced_only=True) == []
