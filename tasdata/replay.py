"""Deterministic replay of parsed movie inputs through nes-py.

Given a parsed :class:`~tasdata.movie.Movie` and a ROM, :class:`NesReplayer` advances
the emulator one frame per logged frame, captures a downscaled grayscale
observation, and records a packed RAM trace.  Nothing here knows about Mario --
the RAM decoding is injected as a probe so other NES games can reuse the harness.

Frame buffers are the memory hog: 240x256 grayscale x 20 000 frames is 1.2 GB.
The defaults (84x84, every frame) come to 141 MB for the same movie, and
``frames_path`` streams straight to a ``.npy`` memmap when even that is too much.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .buttons import actions_from_states, console_button_frames, describe_action
from .movie import Movie
from .ram import TRACE_COLUMNS, pack_smb
from .rom import NesRom, RomCheck, load_rom

#: Native NES output resolution.
NES_HEIGHT, NES_WIDTH = 240, 256

#: Luma weights (ITU-R BT.601), the same ones cv2.COLOR_RGB2GRAY uses.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class ReplayError(RuntimeError):
    """Replay could not be carried out as requested."""


class RomMismatchError(ReplayError):
    """The ROM's hash does not match the one recorded in the movie header."""


def rom_sha1(rom_path: Path | str) -> str:
    """SHA-1 of the ROM file -- the same hash BizHawk stores in a .bk2 header."""
    return hashlib.sha1(Path(rom_path).read_bytes()).hexdigest().lower()


def _resize_gray(frame_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Grayscale + area-downscale one RGB frame to ``(height, width)``.

    Uses OpenCV's INTER_AREA when available (the RL-preprocessing convention) and
    falls back to Pillow's BOX filter, which computes the same box average.
    """
    height, width = size
    gray = frame_rgb.astype(np.float32) @ _LUMA
    if height == NES_HEIGHT and width == NES_WIDTH:
        return gray.astype(np.uint8)
    try:
        import cv2

        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(
            np.uint8
        )
    except ImportError:
        from PIL import Image

        img = Image.fromarray(gray.astype(np.uint8), mode="L")
        return np.asarray(img.resize((width, height), Image.BOX), dtype=np.uint8)


@dataclass
class ReplayResult:
    """Everything a replay produced."""

    movie: Movie
    rom_path: Path
    #: uint8 array ``(n_captured, H, W)``, or a memmap of the same shape.
    frames: np.ndarray
    #: int32 array ``(n_frames, len(TRACE_COLUMNS))`` -- one row per emulated frame.
    trace: np.ndarray
    #: Absolute movie frame index for each row of ``frames``.
    frame_indices: np.ndarray
    n_frames: int
    frame_skip: int
    observation_shape: tuple[int, int]
    wall_seconds: float
    rom: NesRom
    rom_check: RomCheck
    #: The nes-py action byte actually applied on each frame.
    actions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    warnings: list[str] = field(default_factory=list)
    trace_columns: tuple[str, ...] = TRACE_COLUMNS
    #: Which emulator produced this run, e.g. "nes-py" or "fceux 2.6.6".
    backend: str = "nes-py"

    @property
    def fps(self) -> float:
        return self.n_frames / self.wall_seconds if self.wall_seconds else float("nan")

    def save(self, path: Path | str) -> Path:
        """Write the run to a compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frames=np.asarray(self.frames),
            trace=self.trace,
            frame_indices=self.frame_indices,
            trace_columns=np.array(self.trace_columns),
            actions=self.actions,
            meta=np.array(
                [
                    str(self.movie.path),
                    str(self.rom_path),
                    self.rom.sha1_file,
                    str(self.rom_check.matched),
                    str(self.n_frames),
                    str(self.frame_skip),
                    f"{self.observation_shape[0]}x{self.observation_shape[1]}",
                ]
            ),
        )
        return path

    def summary(self) -> str:
        obs = f"{self.observation_shape[0]}x{self.observation_shape[1]}"
        mb = np.asarray(self.frames).nbytes / 1e6
        return (
            f"[{self.backend}] replayed {self.n_frames} frames in "
            f"{self.wall_seconds:.1f}s ({self.fps:.0f} fps); captured "
            f"{len(self.frame_indices)} obs @ {obs} ({mb:.0f} MB)"
        )


class NesReplayer:
    """Applies a movie's inputs to a ROM frame by frame.

    The nes-py backend. Kept as a regression check (it is pure-python to install
    and needs no window), but it is not accurate enough to survive an SMB level
    transition -- see :class:`tasdata.fceux_backend.FceuxReplayer`.

    Args:
        rom_path: path to an iNES ``.nes`` file.
        observation_shape: ``(height, width)`` of the captured grayscale frames.
        frame_skip: capture every Nth frame. Inputs are always applied to *every*
            frame -- this only thins the observations.
        capture_frames: set False to skip image capture entirely (sync checks
            only), which roughly doubles throughput.
        player: which controller port of the movie to drive.
        allow_rom_mismatch: by default a movie whose recorded ROM fingerprint does
            not match the supplied ROM raises :class:`RomMismatchError`, because
            replaying it is guaranteed to desync. Set True to warn and continue.
    """

    #: Identifies this backend in run manifests.
    backend = "nes-py"

    def __init__(
        self,
        rom_path: Path | str,
        *,
        observation_shape: tuple[int, int] = (84, 84),
        frame_skip: int = 1,
        capture_frames: bool = True,
        player: int = 1,
        allow_rom_mismatch: bool = False,
    ) -> None:
        self.rom_path = Path(rom_path)
        if not self.rom_path.exists():
            raise ReplayError(f"ROM not found: {self.rom_path}")
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.observation_shape = observation_shape
        self.frame_skip = frame_skip
        self.capture_frames = capture_frames
        self.player = player
        self.allow_rom_mismatch = allow_rom_mismatch
        self.rom = load_rom(self.rom_path)

    def _make_env(self):
        """Import nes-py lazily so ``parse``-only workflows need no emulator."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from nes_py import NESEnv
            except ImportError as exc:  # pragma: no cover
                raise ReplayError(
                    "nes-py is not installed. `pip install nes-py` (it needs a C "
                    "compiler and numpy<2)."
                ) from exc
        return NESEnv(str(self.rom_path))

    def replay(
        self,
        movie: Movie,
        *,
        max_frames: int | None = None,
        probe: Callable[[np.ndarray, int, np.ndarray], None] = pack_smb,
        trace_columns: Sequence[str] = TRACE_COLUMNS,
        frames_path: Path | str | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ReplayResult:
        """Replay ``movie`` and return a :class:`ReplayResult`.

        Args:
            max_frames: stop after this many frames (useful for smoke tests).
            probe: called as ``probe(ram, frame_index, out_row)`` every frame.
            trace_columns: names for the probe's output columns.
            frames_path: write observations to this ``.npy`` memmap instead of
                holding them in RAM.
            progress: called as ``progress(done, total)`` roughly once a second.
        """
        notes: list[str] = list(movie.notes)

        # ROM verification. The movie's own format decides which fingerprint gets
        # checked: sha1(file) for .bk2, md5(prg+chr) for .fm2's romChecksum.
        rom_check = movie.verify_rom(self.rom)
        if not rom_check.checked:
            notes.append(f"ROM identity unverified: {rom_check.detail}")
        elif not rom_check.matched:
            message = (
                f"ROM mismatch ({rom_check.algorithm}): movie expects "
                f"{rom_check.expected} but {self.rom_path.name} is "
                f"{rom_check.actual}. Different dumps of the same game desync "
                f"within a level."
            )
            if not self.allow_rom_mismatch:
                raise RomMismatchError(
                    message + " Pass allow_rom_mismatch=True (CLI: "
                    "--allow-rom-mismatch) to replay anyway."
                )
            notes.append(message)

        # Region. nes-py has no PAL timing, so a 50 Hz movie cannot be reproduced.
        if movie.pal and not self.rom.header_says_pal:
            notes.append(
                "movie is PAL but the ROM header declares NTSC; nes-py emulates "
                "NTSC timing regardless, so lag frames will differ"
            )

        actions = actions_from_states(movie.states, movie.button_names, self.player)
        total = len(actions) if max_frames is None else min(max_frames, len(actions))
        actions = actions[:total]

        # Console-level buttons. A reset on the very first frames is just how some
        # recorders spell "power on"; anything later genuinely restarts the game
        # and nes-py has no mid-movie reset that preserves the movie timeline.
        resets = console_button_frames(movie.states[:total], movie.button_names, "Reset")
        late_resets = resets[resets > 2]
        if late_resets.size:
            notes.append(
                f"movie asserts Reset on {late_resets.size} frame(s) "
                f"(first at {int(late_resets[0])}); nes-py cannot honour a mid-movie "
                "reset, so replay will diverge there"
            )

        for other in range(2, 5):
            try:
                other_actions = actions_from_states(
                    movie.states[:total], movie.button_names, other
                )
            except ValueError:
                continue
            if other_actions.any():
                notes.append(
                    f"movie contains P{other} input on "
                    f"{int((other_actions != 0).sum())} frame(s), which is ignored"
                )

        n_cols = len(trace_columns)
        trace = np.zeros((total, n_cols), dtype=np.int32)

        capture_idx = np.arange(0, total, self.frame_skip) if self.capture_frames else np.empty(0, dtype=np.int64)
        height, width = self.observation_shape
        if self.capture_frames and frames_path is not None:
            frames_path = Path(frames_path)
            frames_path.parent.mkdir(parents=True, exist_ok=True)
            frames = np.lib.format.open_memmap(
                frames_path, mode="w+", dtype=np.uint8, shape=(len(capture_idx), height, width)
            )
        elif self.capture_frames:
            frames = np.zeros((len(capture_idx), height, width), dtype=np.uint8)
        else:
            frames = np.zeros((0, height, width), dtype=np.uint8)

        env = self._make_env()
        start = time.perf_counter()
        last_report = start
        try:
            env.reset()
            ram = env.ram
            screen = env.screen
            next_capture = 0
            for i in range(total):
                # _frame_advance sets the controller byte and steps one frame
                # without nes-py's episode bookkeeping, which is what we want:
                # the movie, not a reward function, decides when the run ends.
                env._frame_advance(int(actions[i]))
                probe(ram, i, trace[i])
                if next_capture < len(capture_idx) and capture_idx[next_capture] == i:
                    frames[next_capture] = _resize_gray(screen, self.observation_shape)
                    next_capture += 1
                if progress is not None:
                    now = time.perf_counter()
                    if now - last_report > 1.0:
                        progress(i + 1, total)
                        last_report = now
            if progress is not None:
                progress(total, total)
        finally:
            try:
                env.close()
            except Exception:  # pragma: no cover - nes-py teardown is noisy
                pass

        elapsed = time.perf_counter() - start
        result = ReplayResult(
            movie=movie,
            rom_path=self.rom_path,
            frames=frames,
            trace=trace,
            frame_indices=capture_idx,
            n_frames=total,
            frame_skip=self.frame_skip,
            observation_shape=self.observation_shape,
            wall_seconds=elapsed,
            rom=self.rom,
            rom_check=rom_check,
            backend=self.backend,
            warnings=notes,
            trace_columns=tuple(trace_columns),
        )
        result.actions = actions
        return result


def load_run(path: Path | str) -> dict:
    """Load a run written by :meth:`ReplayResult.save`."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def describe_inputs(movie: Movie, frames: Sequence[int], player: int = 1) -> list[str]:
    """Human-readable input at specific frames, for divergence reports."""
    actions = actions_from_states(movie.states, movie.button_names, player)
    out = []
    for f in frames:
        if 0 <= f < len(actions):
            out.append(f"f{f}: {describe_action(int(actions[f]))}")
    return out
