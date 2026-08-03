"""Stage 3 prep: validate the retrieval pseudo-expert before building on it.

The idea under test: index every expert frame by its game state
``(world, level, x_pos, y_pos, player_state)`` and, at any new state, retrieve what the
expert did there. If that works it is a free teacher -- no model, no training.

It is validated by holding out one run, querying the index at each of its frames, and
comparing the retrieved action to what that run actually did. Overall accuracy is the
easy number; the one that decides whether this can teach *jumping* is accuracy at
A-onsets, where the expert transitions from not-pressing to pressing A.

Coordinates are quantised because exact-match on raw pixel x/y would almost never hit:
two runs pass through "the same place" at slightly different subpixel positions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..dataset import LoadedRun
from ..ram import TRACE_COLUMNS, column

A_BIT = 0x01


@dataclass
class RetrievalReport:
    """How good the retrieval pseudo-expert is, per quantisation setting."""

    bin_x: int
    bin_y: int
    n_query: int
    hit_rate: float
    #: Accuracy over frames where the index had *any* entry.
    accuracy_on_hits: float
    #: Accuracy over all query frames (a miss counts as wrong).
    accuracy_overall: float
    exact_byte: bool
    a_onset_queries: int
    a_onset_hit_rate: float
    #: Of A-onset frames with an index hit, how often the retrieved action presses A.
    a_onset_recall_on_hits: float
    a_onset_recall_overall: float
    #: How often the retrieved action presses A at frames where the expert did not.
    a_false_positive: float
    index_size: int
    mean_candidates: float
    #: Fraction of hit states where expert actions disagree with each other.
    ambiguous_rate: float

    def row(self) -> str:
        return (
            f"  x/{self.bin_x:<3d} y/{self.bin_y:<3d} idx={self.index_size:8,d} "
            f"hit={self.hit_rate * 100:5.1f}%  acc(hits)={self.accuracy_on_hits * 100:5.1f}%  "
            f"acc(all)={self.accuracy_overall * 100:5.1f}%  "
            f"A-onset recall(hits)={self.a_onset_recall_on_hits * 100:5.1f}%  "
            f"(all)={self.a_onset_recall_overall * 100:5.1f}%  "
            f"A-FP={self.a_false_positive * 100:4.1f}%  ambig={self.ambiguous_rate * 100:4.1f}%"
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def state_keys(
    trace: np.ndarray, *, bin_x: int, bin_y: int
) -> np.ndarray:
    """Quantised ``(world, stage, x//bin_x, y//bin_y, player_state)`` key per frame."""
    world = column(trace, "world").astype(np.int64)
    stage = column(trace, "stage").astype(np.int64)
    x = column(trace, "x_position").astype(np.int64) // max(bin_x, 1)
    y = column(trace, "y_position").astype(np.int64) // max(bin_y, 1)
    st = column(trace, "player_state").astype(np.int64)
    # Pack into one integer so it can be used as a dict key cheaply.
    return (((((world * 8 + stage) * 4096 + x) * 512 + y) * 256) + st)


def build_index(
    runs: list[LoadedRun], *, bin_x: int, bin_y: int, label_offset: int = 1
) -> dict[int, Counter]:
    """Map quantised state -> Counter of the actions the expert took there.

    ``label_offset=1`` keeps the same convention as training: the action associated
    with the state observed at frame ``i`` is ``a_{i+1}``.
    """
    index: dict[int, Counter] = defaultdict(Counter)
    for run in runs:
        trace = np.asarray(run.trace)
        keys = state_keys(trace, bin_x=bin_x, bin_y=bin_y)
        actions = run.actions.astype(np.int64)
        n = min(len(keys), len(actions) - label_offset)
        for k, a in zip(keys[:n], actions[label_offset : label_offset + n]):
            index[int(k)][int(a)] += 1
    return index


def evaluate_retrieval(
    index: dict[int, Counter],
    held_out: LoadedRun,
    *,
    bin_x: int,
    bin_y: int,
    label_offset: int = 1,
    in_control_only: bool = True,
) -> RetrievalReport:
    """Query the index at every held-out frame and score the retrieved action."""
    trace = np.asarray(held_out.trace)
    keys = state_keys(trace, bin_x=bin_x, bin_y=bin_y)
    actions = held_out.actions.astype(np.int64)
    n = min(len(keys), len(actions) - label_offset)
    keys = keys[:n]
    truth = actions[label_offset : label_offset + n]
    prev = actions[:n]

    if in_control_only:
        pregame = column(trace, "pregame")[:n]
        mask = pregame == 1
    else:
        mask = np.ones(n, dtype=bool)

    a_onset = ((truth & A_BIT) > 0) & ((prev & A_BIT) == 0) & mask
    no_a = ((truth & A_BIT) == 0) & mask

    hits = 0
    correct = 0
    onset_hits = 0
    onset_recall = 0
    fp = 0
    candidates = 0
    ambiguous = 0
    for i in np.flatnonzero(mask):
        bucket = index.get(int(keys[i]))
        if not bucket:
            continue
        hits += 1
        candidates += sum(bucket.values())
        if len(bucket) > 1:
            ambiguous += 1
        retrieved = bucket.most_common(1)[0][0]
        if retrieved == truth[i]:
            correct += 1
        if a_onset[i]:
            onset_hits += 1
            if retrieved & A_BIT:
                onset_recall += 1
        if no_a[i] and (retrieved & A_BIT):
            fp += 1

    n_query = int(mask.sum())
    n_onset = int(a_onset.sum())
    return RetrievalReport(
        bin_x=bin_x,
        bin_y=bin_y,
        n_query=n_query,
        hit_rate=hits / n_query if n_query else 0.0,
        accuracy_on_hits=correct / hits if hits else 0.0,
        accuracy_overall=correct / n_query if n_query else 0.0,
        exact_byte=True,
        a_onset_queries=n_onset,
        a_onset_hit_rate=onset_hits / n_onset if n_onset else 0.0,
        a_onset_recall_on_hits=onset_recall / onset_hits if onset_hits else 0.0,
        a_onset_recall_overall=onset_recall / n_onset if n_onset else 0.0,
        a_false_positive=fp / int(no_a.sum()) if no_a.any() else 0.0,
        index_size=len(index),
        mean_candidates=candidates / hits if hits else 0.0,
        ambiguous_rate=ambiguous / hits if hits else 0.0,
    )
