"""Post-capture analysis: overlap, split, and action statistics.

Three separable concerns, all operating on captured run directories:

**Overlap.** Successive runs in an obsoletion chain are re-records of the same
route, so much of their input is identical. :func:`chain_overlap` reports the
fraction of frames whose action bytes match for each consecutive pair, and
:func:`effective_frames` turns that into an honest dataset size.

**Split.** Whole runs are held out, never frames -- adjacent frames are nearly
identical, so a frame-level split leaks. Whole *chains* are held out too, for the
same reason one level up: two re-records of the same route are near-duplicates, so
putting one in train and its sibling in test leaks just as badly.

**Statistics.** Action-vocabulary size, frequency, physically impossible button
combinations, and hold-length distributions.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .buttons import NES_BUTTON_BITS, describe_action
from .dataset import LoadedRun

#: The two physically impossible D-pad combinations on real hardware. A TAS can
#: request them because the movie writes the controller byte directly.
IMPOSSIBLE_PAIRS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("left+right", ("Left", "Right")),
    ("up+down", ("Up", "Down")),
)


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #

def action_agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Fraction of overlapping frames whose action bytes are equal.

    Compared over the shorter of the two, since chain members differ in length.
    Returns ``(fraction, n_compared)``.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    return float(np.count_nonzero(a[:n] == b[:n]) / n), n


@dataclass
class OverlapPair:
    older: str
    newer: str
    chain: str
    agreement: float
    n_compared: int
    n_older: int
    n_newer: int

    def row(self) -> str:
        return (
            f"  {self.older:22s} vs {self.newer:22s} "
            f"{self.agreement * 100:6.2f}% of {self.n_compared:6d} frames"
        )


def chain_overlap(runs: list[LoadedRun]) -> list[OverlapPair]:
    """Action agreement for each consecutive pair within an obsoletion chain."""
    by_chain: dict[str, list[LoadedRun]] = defaultdict(list)
    for run in runs:
        key = run.chain or f"_solo/{run.label}"
        by_chain[key].append(run)
    pairs: list[OverlapPair] = []
    for chain, members in sorted(by_chain.items()):
        if len(members) < 2:
            continue
        # chain_position 0 is the current record; higher is older.
        members = sorted(
            members, key=lambda r: int(r.manifest.get("chain_position", 0)), reverse=True
        )
        for older, newer in zip(members, members[1:]):
            frac, n = action_agreement(older.actions, newer.actions)
            pairs.append(
                OverlapPair(
                    older=older.label,
                    newer=newer.label,
                    chain=chain,
                    agreement=frac,
                    n_compared=n,
                    n_older=len(older.actions),
                    n_newer=len(newer.actions),
                )
            )
    return pairs


def effective_frames(runs: list[LoadedRun]) -> tuple[int, int, list[tuple[str, float]]]:
    """Estimate distinct frames after discounting overlap.

    Greedy novelty: runs are visited longest-first, and each contributes
    ``n_frames * (1 - max agreement with any already-accepted run)``. Using the
    single most similar predecessor avoids the double-counting you get from
    subtracting every pairwise overlap independently.

    Returns ``(raw_frames, effective_frames, per_run_novelty)``.
    """
    order = sorted(runs, key=lambda r: -len(r.actions))
    accepted: list[np.ndarray] = []
    novelty: list[tuple[str, float]] = []
    effective = 0.0
    raw = 0
    for run in order:
        raw += len(run.actions)
        best = 0.0
        for prev in accepted:
            frac, _ = action_agreement(run.actions, prev)
            best = max(best, frac)
        novel = 1.0 - best
        novelty.append((run.label, novel))
        effective += len(run.actions) * novel
        accepted.append(run.actions)
    return raw, int(round(effective)), novelty


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #

@dataclass
class Split:
    """A whole-run, whole-chain train/val/test assignment."""

    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    frames: dict[str, int] = field(default_factory=dict)
    seed: int = 0
    policy: str = ""

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "unit": "whole run; whole obsoletion chain kept together",
            "splits": {"train": self.train, "val": self.val, "test": self.test},
            "frames": self.frames,
        }


def make_split(
    runs: list[LoadedRun],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 20260729,
) -> Split:
    """Assign whole runs to train/val/test, keeping chains intact.

    The indivisible unit is an obsoletion chain (a solo run is its own group),
    because chain members are re-records of one route and leak into each other.
    That makes exact quotas impossible -- one SMB warpless chain is 8 runs and a
    third of the corpus -- so groups are placed largest-first into whichever bucket
    is furthest below its target share. Dealing groups round-robin instead puts the
    biggest chain wherever it happens to land, which is how test once ended up with
    51% of the frames.

    Strata are *measured routes*, not fine categories: ``all-items`` is a 32-level
    route and belongs with warpless rather than forming a one-group stratum that
    can only ever go to train.
    """
    groups: dict[str, list[LoadedRun]] = defaultdict(list)
    for run in runs:
        groups[run.chain or f"_solo/{run.label}"].append(run)

    by_stratum: dict[str, list[tuple[str, list[LoadedRun]]]] = defaultdict(list)
    for key, members in groups.items():
        by_stratum[members[0].route].append((key, members))

    split = Split(
        seed=seed,
        policy=(
            f"whole-run holdout; obsoletion chains kept together; stratified by "
            f"measured route; groups placed largest-first by remaining deficit "
            f"toward val~{val_fraction:.0%} test~{test_fraction:.0%} of frames"
        ),
    )
    targets = {
        "train": 1.0 - val_fraction - test_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    for stratum, entries in sorted(by_stratum.items()):
        total = sum(len(r.actions) for _k, ms in entries for r in ms)
        # Deterministic order: size desc, then a seeded shuffle to break ties.
        rng = random.Random(f"{seed}:{stratum}")
        entries = sorted(entries, key=lambda kv: kv[0])
        rng.shuffle(entries)
        entries.sort(key=lambda kv: -sum(len(r.actions) for r in kv[1]))
        got = {"train": 0.0, "val": 0.0, "test": 0.0}
        placed: dict[str, list[tuple[int, list[str]]]] = {
            "train": [], "val": [], "test": []
        }
        for _key, members in entries:
            size = sum(len(r.actions) for r in members)
            # Largest remaining deficit wins; train's larger target means the
            # biggest chain lands there, which is what we want.
            bucket = max(got, key=lambda b: targets[b] * total - got[b])
            placed[bucket].append((size, [r.name for r in members]))
            got[bucket] += size

        # Guarantee non-empty holdouts. With few, similarly sized groups the
        # deficit rule can hand everything to train (six equal groups cannot be
        # split 80/10/10 at all), leaving val or test empty -- useless for
        # evaluation. Donate train's smallest groups in that case.
        # Requires 3+ groups left in train: donating from 2 would make a 50/50
        # split, which is worse than having no holdout for that stratum.
        for bucket in ("val", "test"):
            if placed[bucket] or len(placed["train"]) < 3:
                continue
            placed["train"].sort()
            size, names = placed["train"].pop(0)
            placed[bucket].append((size, names))
            got["train"] -= size
            got[bucket] += size

        for bucket, items in placed.items():
            for _size, names in items:
                getattr(split, bucket).extend(names)
    for run in runs:
        split.frames[run.name] = len(run.actions)
    split.train.sort()
    split.val.sort()
    split.test.sort()
    return split


class SplitExistsError(RuntimeError):
    """The split file is already present and is treated as immutable."""


def write_split(path: Path | str, split: Split, *, force: bool = False) -> Path:
    """Write the split once. Refuses to overwrite unless ``force``.

    The split is immutable by policy: changing it after any model has seen the data
    invalidates every comparison made against it. A checksum is stored alongside so
    accidental edits are detectable.
    """
    path = Path(path)
    if path.exists() and not force:
        raise SplitExistsError(
            f"{path} already exists and the split is immutable. Delete it "
            "deliberately (and know that any prior results become incomparable) "
            "or pass force=True."
        )
    payload = split.to_dict()
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def verify_split(path: Path | str) -> bool:
    """True if the split file matches its recorded checksum."""
    data = json.loads(Path(path).read_text())
    recorded = data.pop("sha256", None)
    body = json.dumps(data, indent=2, sort_keys=True)
    return recorded == hashlib.sha256(body.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Action statistics
# --------------------------------------------------------------------------- #

def action_histogram(runs: list[LoadedRun]) -> Counter:
    """Count of each distinct action byte across all runs."""
    counts: Counter = Counter()
    for run in runs:
        values, freq = np.unique(run.actions, return_counts=True)
        for v, c in zip(values.tolist(), freq.tolist()):
            counts[int(v)] += int(c)
    return counts


def impossible_input_stats(runs: list[LoadedRun]) -> dict:
    """Percentage of frames requesting a physically impossible D-pad state."""
    total = 0
    hits = {name: 0 for name, _ in IMPOSSIBLE_PAIRS}
    either = 0
    for run in runs:
        a = run.actions
        total += len(a)
        masks = []
        for name, (b1, b2) in IMPOSSIBLE_PAIRS:
            m = (a & NES_BUTTON_BITS[b1]).astype(bool) & (
                a & NES_BUTTON_BITS[b2]
            ).astype(bool)
            hits[name] += int(np.count_nonzero(m))
            masks.append(m)
        either += int(np.count_nonzero(masks[0] | masks[1]))
    return {
        "total_frames": total,
        "counts": hits,
        "percentages": {
            k: (100.0 * v / total if total else 0.0) for k, v in hits.items()
        },
        "either_count": either,
        "either_percentage": 100.0 * either / total if total else 0.0,
    }


def hold_lengths(runs: list[LoadedRun]) -> dict[str, np.ndarray]:
    """Lengths of every contiguous press of each button, across all runs."""
    out: dict[str, list[np.ndarray]] = {name: [] for name in NES_BUTTON_BITS}
    for run in runs:
        a = run.actions
        for name, bit in NES_BUTTON_BITS.items():
            held = (a & bit).astype(bool)
            if not held.any():
                continue
            padded = np.concatenate(([False], held, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            out[name].append(edges[1::2] - edges[0::2])
    return {
        name: (np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64))
        for name, chunks in out.items()
    }


def summarise_holds(lengths: dict[str, np.ndarray]) -> list[dict]:
    """Per-button hold-length summary, longest-median first."""
    rows = []
    for name, arr in lengths.items():
        if arr.size == 0:
            rows.append({"button": name, "presses": 0})
            continue
        rows.append(
            {
                "button": name,
                "presses": int(arr.size),
                "frames_held": int(arr.sum()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
                "max": int(arr.max()),
                "one_frame_taps": int(np.count_nonzero(arr == 1)),
            }
        )
    rows.sort(key=lambda r: -r.get("frames_held", 0))
    return rows


def build_report(runs: list[LoadedRun]) -> dict:
    """Everything in one JSON-safe dict."""
    hist = action_histogram(runs)
    total = sum(hist.values())
    holds = hold_lengths(runs)
    raw, eff, novelty = effective_frames(runs)
    pairs = chain_overlap(runs)
    return {
        "n_runs": len(runs),
        "total_frames": total,
        "action_vocabulary_size": len(hist),
        "action_frequency": [
            {
                "byte": b,
                "buttons": describe_action(b),
                "count": c,
                "percentage": 100.0 * c / total if total else 0.0,
            }
            for b, c in hist.most_common()
        ],
        "impossible_inputs": impossible_input_stats(runs),
        "hold_lengths": summarise_holds(holds),
        "overlap": {
            "chain_pairs": [
                {
                    "chain": p.chain,
                    "older": p.older,
                    "newer": p.newer,
                    "agreement": p.agreement,
                    "n_compared": p.n_compared,
                }
                for p in pairs
            ],
            "raw_frames": raw,
            "effective_frames": eff,
            "redundancy_percentage": 100.0 * (1 - eff / raw) if raw else 0.0,
            "per_run_novelty": [{"run": n, "novelty": v} for n, v in novelty],
        },
    }
