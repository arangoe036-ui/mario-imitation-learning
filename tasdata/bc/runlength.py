"""Run-length action tokens: predict *(button combo, how long to hold it)* instead of per-frame buttons.

Every solution the enumerative search finds is a macro-action -- "jump at x=892, hold A for 12 frames". A
policy emitting 8 independent Bernoulli buttons per frame produces one with probability ~p¹², which is why
distilling 22 verified pipe-4 demonstrations moved the A-hold distribution the wrong way. **No teacher fixes
that; the student cannot represent the answer.**

This is the data transform that lets it. The expert corpus is re-expressed as runs of a constant action:

* **981,385 frames become 77,933 run samples (7.9%).** Mean run length 12.59, median 1, p90 25, max 1200.
* A 12-frame hold is **one** training sample with an explicit duration, not twelve correlated ones.

The label is a **joint class** over (combo token, length bucket), so the existing categorical head works
unchanged at a resized width -- a data transform plus a head resize, not new ML.

**Buckets are fine where the mass is.** The expert's run lengths are heavily skewed (median 1, p90 25), so
buckets are near-exact below 6 frames and widen geometrically after. Generation uses the **expert's median
length within each joint class**, so a class means a duration the expert actually produced rather than a
bucket midpoint nobody played.

**One inherited caution:** `PolicyConfig` records that the categorical head suffered vote-splitting under
**argmax** -- the four A-containing tokens each lost to Right+B, and A was emitted on 0.03% of frames.
Sampling from the softmax does not have that failure, and it is what the live rollout here does. Argmax must
not be used with this head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

#: Half-open [lo, hi) buckets over run length in frames. Near-exact below 6, geometric after.
BUCKETS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 13), (13, 17), (17, 25),
    (25, 33), (33, 49), (49, 97), (97, 10 ** 9),
)
N_BUCKETS = len(BUCKETS)


def bucket(length: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= length < hi:
            return i
    return N_BUCKETS - 1


def joint_size(vocab_size: int) -> int:
    return vocab_size * N_BUCKETS


def encode_joint(token: int, length: int) -> int:
    return int(token) * N_BUCKETS + bucket(int(length))


def decode_joint(joint: int) -> tuple[int, int]:
    """(combo token, bucket index)."""
    return int(joint) // N_BUCKETS, int(joint) % N_BUCKETS


def build_index(ds) -> dict:
    """Rows of `ds` whose label token *starts* a run, with that run's length.

    Built by walking the same `tokens` / `frame_indices` / `label_offset` path the underlying dataset
    uses for its own labels, so a row's joint label describes exactly the action that row is asked to
    predict. Deriving the run boundaries any other way would risk labelling a row with a neighbour's run.
    """
    rows, joints, lengths = [], [], []
    for run_id, entry in enumerate(ds.index):
        tokens = ds.tokens[run_id]
        last = len(tokens) - 1
        if last < 0:
            continue
        # run id -> length, computed once per run
        run_len = np.empty(len(tokens), dtype=np.int64)
        i = 0
        while i < len(tokens):
            j = i
            while j < len(tokens) and tokens[j] == tokens[i]:
                j += 1
            run_len[i:j] = j - i
            i = j
        base = int(ds.offsets[run_id])
        fidx = entry.frame_indices
        for row in range(entry.n_obs):
            m = min(int(fidx[row]) + ds.label_offset, last)
            if m > 0 and tokens[m - 1] == tokens[m]:
                continue                      # not a run start: skip
            rows.append(base + row)
            joints.append(encode_joint(int(tokens[m]), int(run_len[m])))
            lengths.append(int(run_len[m]))
    return {"rows": np.asarray(rows, dtype=np.int64),
            "joints": np.asarray(joints, dtype=np.int64),
            "lengths": np.asarray(lengths, dtype=np.int64)}


def class_lengths(index: dict, n_classes: int) -> np.ndarray:
    """Median expert run length per joint class; bucket lower bound where a class is unseen."""
    out = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        m = index["joints"] == c
        if m.any():
            out[c] = int(np.median(index["lengths"][m]))
        else:
            out[c] = BUCKETS[c % N_BUCKETS][0]
    return np.maximum(out, 1)


class RunLengthDataset(Dataset):
    """`(obs, prev, joint_class)` for run-start rows only. Wraps a token-mode FrameStackDataset."""

    def __init__(self, base, index: dict | None = None):
        if base.label_mode != "token":
            raise ValueError("RunLengthDataset needs a label_mode='token' base dataset")
        self.base = base
        self.index = index if index is not None else build_index(base)
        self.n_classes = joint_size(base.vocab.size)

    def __len__(self) -> int:
        return int(len(self.index["rows"]))

    def __getitem__(self, i: int):
        obs, prev, _token = self.base[int(self.index["rows"][i])]
        return obs, prev, int(self.index["joints"][i])


def collate(batch):
    obs = torch.stack([b[0] for b in batch])
    prev = torch.stack([b[1] for b in batch])
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return obs, prev, y
