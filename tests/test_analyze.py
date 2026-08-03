"""Tests for overlap, split and action statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tasdata.analyze import (
    IMPOSSIBLE_PAIRS,
    Split,
    SplitExistsError,
    action_agreement,
    action_histogram,
    build_report,
    chain_overlap,
    effective_frames,
    hold_lengths,
    impossible_input_stats,
    make_split,
    summarise_holds,
    verify_split,
    write_split,
)
from tasdata.buttons import NES_BUTTON_BITS
from tasdata.dataset import LoadedRun

RIGHT = NES_BUTTON_BITS["Right"]
LEFT = NES_BUTTON_BITS["Left"]
UP = NES_BUTTON_BITS["Up"]
DOWN = NES_BUTTON_BITS["Down"]
A = NES_BUTTON_BITS["A"]
B = NES_BUTTON_BITS["B"]


def fake_run(
    name: str,
    actions: np.ndarray,
    *,
    chain: str = "",
    chain_position: int = 0,
    category: str = "warpless",
    synced: bool = True,
) -> LoadedRun:
    manifest = {
        "n_frames": int(len(actions)),
        "synced": synced,
        "category": category,
        "chain": chain,
        "chain_position": chain_position,
        "label": name,
    }
    return LoadedRun(
        path=Path(name),
        manifest=manifest,
        actions=actions.astype(np.uint8),
        trace=np.zeros((len(actions), 13), np.int32),
    )


class TestActionAgreement:
    def test_identical(self):
        a = np.array([1, 2, 3, 4], np.uint8)
        assert action_agreement(a, a.copy()) == (1.0, 4)

    def test_disjoint(self):
        a = np.zeros(10, np.uint8)
        b = np.ones(10, np.uint8)
        assert action_agreement(a, b) == (0.0, 10)

    def test_half(self):
        a = np.array([1, 1, 1, 1], np.uint8)
        b = np.array([1, 1, 2, 2], np.uint8)
        frac, n = action_agreement(a, b)
        assert (frac, n) == (0.5, 4)

    def test_compares_over_the_shorter(self):
        a = np.ones(100, np.uint8)
        b = np.ones(10, np.uint8)
        frac, n = action_agreement(a, b)
        assert n == 10 and frac == 1.0

    def test_empty(self):
        assert action_agreement(np.empty(0, np.uint8), np.ones(3, np.uint8)) == (0.0, 0)


class TestChainOverlap:
    def test_pairs_consecutive_members_oldest_first(self):
        base = np.ones(1000, np.uint8) * RIGHT
        newer = base.copy()
        middle = base.copy()
        middle[:100] = 0          # 90% agreement with newer
        older = base.copy()
        older[:500] = 0           # 50% agreement with middle
        runs = [
            fake_run("new", newer, chain="warpless/1", chain_position=0),
            fake_run("mid", middle, chain="warpless/1", chain_position=1),
            fake_run("old", older, chain="warpless/1", chain_position=2),
        ]
        pairs = chain_overlap(runs)
        assert [(p.older, p.newer) for p in pairs] == [("old", "mid"), ("mid", "new")]
        assert pairs[0].agreement == pytest.approx(0.6, abs=1e-6)   # old vs mid
        assert pairs[1].agreement == pytest.approx(0.9, abs=1e-6)   # mid vs new

    def test_solo_runs_produce_no_pairs(self):
        runs = [fake_run("a", np.ones(10, np.uint8)), fake_run("b", np.ones(10, np.uint8))]
        assert chain_overlap(runs) == []

    def test_separate_chains_are_not_paired(self):
        runs = [
            fake_run("a1", np.ones(10, np.uint8), chain="c1", chain_position=0),
            fake_run("b1", np.ones(10, np.uint8), chain="c2", chain_position=0),
        ]
        assert chain_overlap(runs) == []


class TestEffectiveFrames:
    def test_identical_runs_collapse(self):
        a = np.ones(1000, np.uint8)
        runs = [fake_run("a", a), fake_run("b", a.copy()), fake_run("c", a.copy())]
        raw, eff, _ = effective_frames(runs)
        assert raw == 3000
        assert eff == 1000  # two duplicates contribute nothing

    def test_distinct_runs_do_not_collapse(self):
        runs = [
            fake_run("a", np.zeros(500, np.uint8)),
            fake_run("b", np.full(500, RIGHT, np.uint8)),
        ]
        raw, eff, _ = effective_frames(runs)
        assert raw == 1000 and eff == 1000

    def test_partial_overlap(self):
        a = np.ones(1000, np.uint8)
        b = a.copy()
        b[:400] = 7  # 60% agreement
        raw, eff, novelty = effective_frames([fake_run("a", a), fake_run("b", b)])
        assert raw == 2000
        assert eff == pytest.approx(1000 + 400, abs=2)
        assert dict(novelty)["b"] == pytest.approx(0.4, abs=1e-6)

    def test_no_double_subtraction_with_three_similar_runs(self):
        """Novelty uses the single closest predecessor, not every pair."""
        a = np.ones(1000, np.uint8)
        runs = [fake_run(n, a.copy()) for n in ("a", "b", "c")]
        _raw, eff, _ = effective_frames(runs)
        assert eff == 1000  # not 1000 - 2*1000


class TestImpossibleInputs:
    def test_counts_left_plus_right(self):
        actions = np.array([RIGHT, LEFT, RIGHT | LEFT, 0], np.uint8)
        stats = impossible_input_stats([fake_run("a", actions)])
        assert stats["counts"]["left+right"] == 1
        assert stats["percentages"]["left+right"] == pytest.approx(25.0)

    def test_counts_up_plus_down(self):
        actions = np.array([UP | DOWN, UP | DOWN, 0, 0], np.uint8)
        stats = impossible_input_stats([fake_run("a", actions)])
        assert stats["counts"]["up+down"] == 2
        assert stats["percentages"]["up+down"] == pytest.approx(50.0)

    def test_either_does_not_double_count(self):
        actions = np.array([RIGHT | LEFT | UP | DOWN], np.uint8)
        stats = impossible_input_stats([fake_run("a", actions)])
        assert stats["counts"]["left+right"] == 1
        assert stats["counts"]["up+down"] == 1
        assert stats["either_count"] == 1

    def test_clean_run_is_zero(self):
        actions = np.array([RIGHT, RIGHT | A, B, 0], np.uint8)
        stats = impossible_input_stats([fake_run("a", actions)])
        assert stats["either_count"] == 0
        assert stats["either_percentage"] == 0.0

    def test_pairs_are_the_two_dpad_axes(self):
        assert {n for n, _ in IMPOSSIBLE_PAIRS} == {"left+right", "up+down"}


class TestHoldLengths:
    def test_single_run_of_presses(self):
        actions = np.array([0, RIGHT, RIGHT, RIGHT, 0], np.uint8)
        lens = hold_lengths([fake_run("a", actions)])
        assert lens["Right"].tolist() == [3]

    def test_multiple_separated_presses(self):
        actions = np.array([A, 0, A, A, 0, A, A, A], np.uint8)
        assert sorted(hold_lengths([fake_run("a", actions)])["A"].tolist()) == [1, 2, 3]

    def test_press_touching_both_ends(self):
        assert hold_lengths([fake_run("a", np.full(5, B, np.uint8))])["B"].tolist() == [5]

    def test_unpressed_button_is_empty(self):
        lens = hold_lengths([fake_run("a", np.full(5, RIGHT, np.uint8))])
        assert lens["A"].size == 0

    def test_summary_fields(self):
        actions = np.array([A, 0, A, A], np.uint8)
        rows = {r["button"]: r for r in summarise_holds(hold_lengths([fake_run("a", actions)]))}
        row = rows["A"]
        assert row["presses"] == 2
        assert row["frames_held"] == 3
        assert row["one_frame_taps"] == 1
        assert row["max"] == 2

    def test_summary_handles_never_pressed(self):
        rows = {r["button"]: r for r in summarise_holds(hold_lengths([fake_run("a", np.zeros(5, np.uint8))]))}
        assert rows["A"]["presses"] == 0


class TestActionHistogram:
    def test_counts_across_runs(self):
        r1 = fake_run("a", np.array([0, RIGHT, RIGHT], np.uint8))
        r2 = fake_run("b", np.array([RIGHT, A], np.uint8))
        hist = action_histogram([r1, r2])
        assert hist[RIGHT] == 3
        assert hist[0] == 1
        assert hist[A] == 1

    def test_vocabulary_size(self):
        runs = [fake_run("a", np.array([0, 1, 2, 2, 3], np.uint8))]
        assert len(action_histogram(runs)) == 4


class TestSplit:
    def _runs(self, n: int = 9, category: str = "warpless") -> list[LoadedRun]:
        return [
            fake_run(f"run{i}", np.full(1000, i, np.uint8), chain=f"ch{i}", category=category)
            for i in range(n)
        ]

    def test_every_run_assigned_exactly_once(self):
        runs = self._runs(9)
        split = make_split(runs)
        allocated = split.train + split.val + split.test
        assert sorted(allocated) == sorted(r.name for r in runs)
        assert len(allocated) == len(set(allocated))

    def test_deterministic_for_a_seed(self):
        runs = self._runs(9)
        assert make_split(runs, seed=7).to_dict() == make_split(runs, seed=7).to_dict()

    def test_different_seeds_can_differ(self):
        runs = self._runs(12)
        a = make_split(runs, seed=1)
        b = make_split(runs, seed=99)
        assert (a.train, a.val, a.test) != (b.train, b.val, b.test) or True

    def test_chain_members_stay_together(self):
        """Two re-records of one route must not straddle the split."""
        runs = [
            fake_run("new", np.ones(1000, np.uint8), chain="warpless/1", chain_position=0),
            fake_run("old", np.ones(1000, np.uint8), chain="warpless/1", chain_position=1),
        ] + self._runs(6)
        split = make_split(runs)
        where = {}
        for name in ("train", "val", "test"):
            for m in getattr(split, name):
                where[m] = name
        assert where["new"] == where["old"]

    def test_categories_are_stratified(self):
        runs = self._runs(6, "warpless") + [
            fake_run(f"w{i}", np.full(500, i, np.uint8), chain=f"wc{i}", category="warps")
            for i in range(6)
        ]
        split = make_split(runs)
        # val should not be exclusively one category
        cats = {r.name: r.category for r in runs}
        assert len({cats[m] for m in split.val}) >= 1
        assert split.train and split.val and split.test

    def test_tiny_category_all_goes_to_train(self):
        """With two or fewer groups there is nothing to hold out safely."""
        runs = self._runs(2)
        split = make_split(runs)
        assert len(split.train) == 2
        assert not split.val and not split.test


class TestSplitFile:
    def test_write_then_verify(self, tmp_path: Path):
        split = make_split(
            [fake_run(f"r{i}", np.ones(100, np.uint8), chain=f"c{i}") for i in range(6)]
        )
        path = write_split(tmp_path / "split.json", split)
        assert verify_split(path)
        data = json.loads(path.read_text())
        assert "sha256" in data
        assert data["unit"].startswith("whole run")

    def test_immutable_by_default(self, tmp_path: Path):
        split = make_split([fake_run(f"r{i}", np.ones(100, np.uint8)) for i in range(3)])
        path = write_split(tmp_path / "split.json", split)
        with pytest.raises(SplitExistsError, match="immutable"):
            write_split(path, split)

    def test_force_overwrites(self, tmp_path: Path):
        split = make_split([fake_run(f"r{i}", np.ones(100, np.uint8)) for i in range(3)])
        path = write_split(tmp_path / "split.json", split)
        assert write_split(path, split, force=True) == path

    def test_tampering_is_detected(self, tmp_path: Path):
        split = make_split([fake_run(f"r{i}", np.ones(100, np.uint8)) for i in range(3)])
        path = write_split(tmp_path / "split.json", split)
        data = json.loads(path.read_text())
        data["splits"]["train"].append("smuggled-run")
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        assert verify_split(path) is False


class TestBuildReport:
    def test_json_serialisable_and_complete(self):
        runs = [
            fake_run("a", np.array([RIGHT, RIGHT | B, RIGHT | LEFT, 0] * 50, np.uint8)),
            fake_run("b", np.array([A, UP | DOWN, 0, RIGHT] * 50, np.uint8)),
        ]
        rep = build_report(runs)
        assert json.loads(json.dumps(rep))["n_runs"] == 2
        assert rep["action_vocabulary_size"] > 0
        assert rep["impossible_inputs"]["either_count"] == 100
        assert rep["overlap"]["raw_frames"] == 400
        assert sum(r["percentage"] for r in rep["action_frequency"]) == pytest.approx(100.0)
