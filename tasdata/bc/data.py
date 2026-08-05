"""Frame-stacked, memory-mapped dataset.

11 GiB of frames will not fit in RAM, so ``frames.npy`` is opened with
``mmap_mode="r"`` and only the requested stack is read. Memmaps are opened lazily
per worker: a memmap handle inherited across a fork is a good way to get silent
corruption, and under ``spawn`` it would not survive pickling at all.

Frame stacking is mandatory for this task. One 84x84 frame carries no velocity, and
SMB is almost entirely momentum -- from a still image you cannot tell whether Mario
is sprinting right or has just turned around.

Label alignment
---------------
Capture advances a frame and *then* samples, so ``obs_i`` is the state **after** input
``a_i`` was applied. But the game read ``a_i`` at the start of frame ``i``, i.e. based
on ``obs_{i-1}``. The correct supervision is therefore ``(obs_i, a_{i+1})``: given what
you can see now, choose the input for the next frame. Training on ``(obs_i, a_i)`` asks
the model to name the action that *caused* the current frame, which at inference makes
it replay the previous action one frame late.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..buttons import NES_BUTTON_ORDER_BITS
from ..dataset import LoadedRun, discover_runs
from .tokens import ActionVocab

#: Number of stacked frames. 4 is the DQN convention and enough for velocity.
DEFAULT_STACK = 4

#: Label offset: obs_i is supervised with a_{i+offset}. 1 is correct; 0 reproduces the
#: original off-by-one and exists only so the bug can be re-tested deliberately.
DEFAULT_LABEL_OFFSET = 1

#: How many already-applied actions to feed alongside the frames.
DEFAULT_PREV_ACTIONS = 4


@dataclass
class RunIndex:
    """Where a run's arrays live and how many frames it contributes."""

    name: str
    frames_path: Path
    actions: np.ndarray
    n_obs: int
    #: Movie frame index for each observation row (identity when frame_skip == 1).
    frame_indices: np.ndarray
    category: str


def build_run_index(runs: list[LoadedRun]) -> list[RunIndex]:
    """Collect the per-run information the dataset needs, without loading frames."""
    out: list[RunIndex] = []
    for run in runs:
        frames_path = run.path / "frames.npy"
        if not frames_path.exists():
            continue
        header = np.load(frames_path, mmap_mode="r")
        n_obs = int(header.shape[0])
        del header
        # Which movie frame each observation row corresponds to. Identity when
        # frame_skip == 1, which it is for every captured run, but read it rather
        # than assume so a thinned capture still labels correctly.
        idx_path = run.path / "frame_indices.npy"
        frame_indices = (
            np.load(idx_path) if idx_path.exists() else np.arange(n_obs, dtype=np.int64)
        )
        out.append(
            RunIndex(
                name=run.name,
                frames_path=frames_path,
                actions=run.actions.astype(np.uint8),
                n_obs=n_obs,
                frame_indices=np.asarray(frame_indices)[:n_obs],
                category=run.category,
            )
        )
    return out


def _downscale_stack(stack: np.ndarray, size: int) -> np.ndarray:
    """Area-average a ``(T, S, S)`` uint8 stack down to ``(T, size, size)``.

    Area averaging, matching `replay._resize_gray`'s INTER_AREA, so a frame downscaled from a
    128x128 capture is the same kind of image as one captured at 84x84 directly -- not merely a
    similar size. **This is not exactly bit-identical to capturing at 84x84**: that path averages
    from the NES's native 240x256, this one from an already-averaged 128x128, so it is a second
    lossy step. The 84x84 store is therefore kept rather than regenerated from this.
    """
    try:
        import cv2

        return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
                         for f in stack])
    except ImportError:
        from PIL import Image

        return np.stack([np.asarray(Image.fromarray(f, mode="L")
                                    .resize((size, size), Image.BOX), dtype=np.uint8)
                         for f in stack])


class FrameStackDataset(Dataset):
    """``(stack, token)`` pairs drawn from memory-mapped run frames.

    Item ``i`` is a global index over the concatenation of all runs. The stack for a
    given observation is the ``stack`` most recent frames *within the same run*,
    edge-padded by repeating the first frame so early frames are still usable and no
    stack ever straddles two runs.
    """

    def __init__(
        self,
        runs: list[LoadedRun],
        vocab: ActionVocab,
        *,
        stack: int = DEFAULT_STACK,
        blind: bool = False,
        prev_actions: int = 0,
        label_offset: int = DEFAULT_LABEL_OFFSET,
        label_mode: str = "token",
        frame_size: int | None = None,
    ) -> None:
        self.index = build_run_index(runs)
        if not self.index:
            raise ValueError("no runs with frames.npy")
        self.vocab = vocab
        self.stack = stack
        self.blind = blind
        self.prev_actions = prev_actions
        self.label_offset = label_offset
        if label_mode not in ("token", "buttons"):
            raise ValueError(f"unknown label_mode {label_mode!r}")
        self.label_mode = label_mode
        #: Raw action bytes per run, for the Bernoulli path (no vocabulary folding).
        self.raw = [r.actions.astype(np.uint8) for r in self.index]
        #: Token id used when the previous-action input is dropped out or unavailable.
        self.mask_token = vocab.size
        self.lengths = np.array([r.n_obs for r in self.index], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self._maps: dict[int, np.ndarray] = {}
        #: Resolution as *stored* on disk, read from the first run rather than assumed. A corpus
        #: captured at 128x128 and one captured at 84x84 are otherwise indistinguishable to a
        #: caller, and the model raises on a mismatch rather than reshaping silently.
        probe = np.load(self.index[0].frames_path, mmap_mode="r")
        self.stored_size = int(probe.shape[-1])
        del probe
        #: Resolution *served*. `None` means "whatever is stored"; a smaller value downscales at
        #: load time, so one 128x128 capture serves both resolutions and 84x84 need not be
        #: re-captured to be available.
        self.frame_size = int(frame_size) if frame_size else self.stored_size
        if self.frame_size > self.stored_size:
            raise ValueError(
                f"cannot serve {self.frame_size}x{self.frame_size} from a "
                f"{self.stored_size}x{self.stored_size} capture -- upscaling would invent detail"
            )
        # Tokens are precomputed: 1.2M int64 is 10 MB and saves a lookup per item.
        self.tokens = [vocab.encode(r.actions) for r in self.index]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _frames(self, run_id: int) -> np.ndarray:
        """Lazily open (and cache per process) one run's frame memmap."""
        mm = self._maps.get(run_id)
        if mm is None:
            mm = np.load(self.index[run_id].frames_path, mmap_mode="r")
            self._maps[run_id] = mm
        return mm

    def locate(self, i: int) -> tuple[int, int]:
        """Global index -> ``(run_id, row within run)``."""
        run_id = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return run_id, int(i - self.offsets[run_id])

    def __getitem__(self, i: int):
        run_id, row = self.locate(i)
        entry = self.index[run_id]
        if self.blind:
            # Identical shape, no information. The blind baseline must differ from
            # the sighted model *only* in what it can see.
            stack = np.zeros((self.stack, self.frame_size, self.frame_size), dtype=np.uint8)
        else:
            frames = self._frames(run_id)
            rows = np.clip(np.arange(row - self.stack + 1, row + 1), 0, entry.n_obs - 1)
            stack = np.asarray(frames[rows])
            if self.frame_size != self.stored_size:
                stack = _downscale_stack(stack, self.frame_size)

        tokens = self.tokens[run_id]
        last = len(tokens) - 1
        movie_frame = int(entry.frame_indices[row])
        # obs_i is the state *after* a_i; the action to choose next is a_{i+1}.
        token = int(tokens[min(movie_frame + self.label_offset, last)])
        # Actions already applied, oldest first, ending with a_i.
        k = max(1, self.prev_actions)
        prev_idx = np.clip(np.arange(movie_frame - k + 1, movie_frame + 1), 0, last)
        prev = torch.from_numpy(tokens[prev_idx].astype(np.int64))

        obs = torch.from_numpy(np.ascontiguousarray(stack)).float().div_(255.0)
        if self.label_mode == "token":
            return obs, prev, token

        # Bernoulli path: 8 raw button bits of a_{i+1}, plus which of them are ONSETS
        # (released on a_i, held on a_{i+1}). Onsets are what "decide to act" means and
        # what arm B upweights.
        raw = self.raw[run_id]
        nxt = int(raw[min(movie_frame + self.label_offset, len(raw) - 1)])
        cur = int(raw[min(movie_frame, len(raw) - 1)])
        bits = np.array([(nxt >> 0) & 1] * 8, dtype=np.float32)
        onset = np.zeros(8, dtype=np.float32)
        for j, bit in enumerate(NES_BUTTON_ORDER_BITS):
            bits[j] = 1.0 if (nxt & bit) else 0.0
            onset[j] = 1.0 if ((nxt & bit) and not (cur & bit)) else 0.0
        return obs, prev, torch.from_numpy(bits), torch.from_numpy(onset)

    # -- statistics used by the baselines and reporting -------------------- #

    def all_tokens(self) -> np.ndarray:
        """Every label in the dataset, in index order (with the label offset applied)."""
        parts = []
        for run_id, entry in enumerate(self.index):
            last = len(self.tokens[run_id]) - 1
            idx = np.clip(
                entry.frame_indices[: entry.n_obs] + self.label_offset, 0, last
            )
            parts.append(self.tokens[run_id][idx])
        return np.concatenate(parts)

    def token_counts(self) -> np.ndarray:
        counts = np.zeros(self.vocab.size, dtype=np.int64)
        values, freq = np.unique(self.all_tokens(), return_counts=True)
        counts[values] = freq
        return counts


def load_split(
    runs_root: Path | str, split_path: Path | str
) -> dict[str, list[LoadedRun]]:
    """Load run directories grouped by the immutable split."""
    import json

    split = json.loads(Path(split_path).read_text())["splits"]
    by_name = {r.name: r for r in discover_runs(runs_root)}
    out: dict[str, list[LoadedRun]] = {}
    for bucket, names in split.items():
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"split names not found under {runs_root}: {missing}")
        out[bucket] = [by_name[n] for n in names]
    return out
