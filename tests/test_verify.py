"""Verifier tests, driven by synthetic RAM traces (no emulator needed)."""

from __future__ import annotations

import numpy as np
import pytest

from tasdata.ram import TRACE_COLUMNS
from tasdata.verify import (
    DEFAULT_COMPARE_COLUMNS,
    compare_traces,
    level_spans,
    load_reference,
    parse_level_spec,
    save_reference,
    verify_smb,
)

from .conftest import synthetic_trace


def walking_run(frames: int = 2000, *, start_x: int = 0, per_frame: int = 1) -> list[dict]:
    """A boring, healthy run: Mario walks right forever in 1-1."""
    return [{"x_position": start_x + i * per_frame} for i in range(frames)]


def clears_levels(levels: list[tuple[int, int]], frames_each: int = 800) -> list[dict]:
    """Walk right through each (world, stage) in turn."""
    rows: list[dict] = []
    for world, stage in levels:
        for i in range(frames_each):
            rows.append({"world": world, "stage": stage, "x_position": i * 2})
    return rows


class TestLevelSpec:
    def test_ordinal(self):
        assert parse_level_spec("1-1") == 0
        assert parse_level_spec("1-2") == 1
        assert parse_level_spec("4-1") == 12
        assert parse_level_spec("8-4") == 31

    @pytest.mark.parametrize("bad", ["9-1", "1-5", "abc", "1", "0-1"])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(ValueError):
            parse_level_spec(bad)


class TestSpans:
    def test_splits_on_level_change(self):
        trace = synthetic_trace(clears_levels([(1, 1), (1, 2), (4, 1)], frames_each=100))
        spans = level_spans(trace)
        assert [(s.world, s.stage) for s in spans] == [(1, 1), (1, 2), (4, 1)]
        assert spans[0].first_frame == 0
        assert spans[0].max_x == 198

    def test_drops_transient_phantom_states(self):
        rows = clears_levels([(1, 1)], frames_each=100)
        # a 2-frame phantom, the kind produced by non-atomic level-load writes
        rows[50] = {**rows[50], "stage": 3}
        rows[51] = {**rows[51], "stage": 3}
        trace = synthetic_trace(rows)
        spans = level_spans(trace)
        assert all(s.stage != 3 for s in spans)


class TestHealthyRun:
    def test_passes(self):
        trace = synthetic_trace(clears_levels([(1, 1), (1, 2), (4, 1)]))
        report = verify_smb(trace)
        assert report.passed is True
        assert report.diverged_at is None
        assert report.levels_reached == ["1-1", "1-2", "4-1"]

    def test_warp_jump_is_not_backwards(self):
        trace = synthetic_trace(clears_levels([(1, 1), (1, 2), (4, 1), (8, 1)]))
        report = verify_smb(trace)
        assert report.passed is True
        assert next(c for c in report.checks if c.name == "level-monotonic").passed

    def test_expected_level_met(self):
        trace = synthetic_trace(clears_levels([(1, 1), (8, 4)]))
        assert verify_smb(trace, expect_level="8-4").passed is True


class TestFailureModes:
    def test_death_is_the_divergence_point(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        for i in range(700, 760):
            rows[i] = {**rows[i], "player_state": 0x0B}
        report = verify_smb(synthetic_trace(rows))
        assert report.passed is False
        assert report.diverged_at == 700
        assert "no-deaths" in report.reason

    def test_lost_life_detected_even_without_dying_state(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        for i in range(600, 1000):
            rows[i] = {**rows[i], "lives": 1}
        report = verify_smb(synthetic_trace(rows))
        assert report.passed is False
        assert report.diverged_at == 600

    def test_never_started(self):
        rows = [{"pregame": 0, "x_position": 0} for _ in range(500)]
        report = verify_smb(synthetic_trace(rows))
        assert report.passed is False
        assert report.diverged_at == 0
        assert "never left the title" in report.reason

    def test_level_going_backwards(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=400) + clears_levels(
            [(1, 1)], frames_each=400
        )
        report = verify_smb(synthetic_trace(rows))
        assert report.passed is False
        assert not next(c for c in report.checks if c.name == "level-monotonic").passed

    def test_regression_across_a_load_is_caught(self):
        """1-1 -> 1-2 -> (load) -> 1-1 is a death-restart, not monotone progress.

        Regression test: an adjacent-frame diff misses this, because the level byte
        changes while pregame != 1 and only the *sustained* lower level is visible.
        """
        rows = clears_levels([(1, 1), (1, 2)], frames_each=400)
        rows += [{"world": 1, "stage": 2, "pregame": 3} for _ in range(20)]  # load
        rows += clears_levels([(1, 1)], frames_each=400)
        report = verify_smb(synthetic_trace(rows))
        check = next(c for c in report.checks if c.name == "level-monotonic")
        assert check.passed is False
        assert "fell back" in check.detail
        assert report.passed is False

    def test_brief_load_glitch_is_not_a_regression(self):
        """A few frames of non-atomic level-load bytes must not trip the check."""
        rows = clears_levels([(1, 1), (1, 2)], frames_each=400)
        for i in range(400, 403):
            rows[i] = {**rows[i], "world": 1, "stage": 1}
        report = verify_smb(synthetic_trace(rows))
        assert next(c for c in report.checks if c.name == "level-monotonic").passed

    def test_scattered_timer_zeros_are_not_expiry(self):
        """Zeroes during level loads are normal; only a sustained run counts."""
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        for i in range(200, 1000, 20):  # 40 scattered zero frames
            rows[i] = {**rows[i], "time": 0}
        report = verify_smb(synthetic_trace(rows))
        assert next(c for c in report.checks if c.name == "timer").passed

    def test_contiguous_timer_zeros_are_expiry(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        for i in range(600, 700):
            rows[i] = {**rows[i], "time": 0}
        check = next(
            c for c in verify_smb(synthetic_trace(rows)).checks if c.name == "timer"
        )
        assert check.passed is False
        assert check.frame == 600
        assert "consecutive" in check.detail

    def test_too_few_levels(self):
        trace = synthetic_trace(walking_run(3000))
        report = verify_smb(trace, min_levels=2)
        assert report.passed is False
        assert "level-coverage" in report.reason

    def test_expected_level_missed_reports_where_it_stopped(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        report = verify_smb(synthetic_trace(rows), expect_level="8-4")
        assert report.passed is False
        assert "no further than 1-2" in report.reason
        # the useful frame is where forward progress actually stopped
        assert report.diverged_at == 999

    def test_timer_expiry(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        for i in range(400, 500):
            rows[i] = {**rows[i], "time": 0}
        report = verify_smb(synthetic_trace(rows))
        assert report.passed is False
        assert not next(c for c in report.checks if c.name == "timer").passed

    def test_stall_is_advisory_by_default(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=400)
        rows += [{"world": 1, "stage": 2, "x_position": 798} for _ in range(1200)]
        report = verify_smb(synthetic_trace(rows), stall_frames=600)
        stall = next(c for c in report.checks if c.name == "forward-progress")
        assert stall.passed is False
        assert stall.advisory is True
        assert report.passed is True  # advisory only
        assert "advisories" in report.reason

    def test_stall_fails_when_strict(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=400)
        rows += [{"world": 1, "stage": 2, "x_position": 798} for _ in range(1200)]
        report = verify_smb(synthetic_trace(rows), stall_frames=600, strict_stall=True)
        assert report.passed is False

    def test_empty_trace(self):
        report = verify_smb(np.zeros((0, len(TRACE_COLUMNS)), dtype=np.int32))
        assert report.passed is False
        assert report.diverged_at == 0

    def test_earliest_failure_wins(self):
        """Death at 700, backwards level at 1600 -> report the death."""
        rows = clears_levels([(1, 1), (1, 2)], frames_each=800)
        for i in range(700, 730):
            rows[i] = {**rows[i], "player_state": 0x06}
        rows += clears_levels([(1, 1)], frames_each=200)
        report = verify_smb(synthetic_trace(rows))
        assert report.diverged_at == 700


class TestReferenceDiff:
    def test_identical_traces_match(self):
        trace = synthetic_trace(clears_levels([(1, 1), (1, 2)]))
        frame, detail = compare_traces(trace, trace.copy())
        assert frame is None and detail == ""

    def test_finds_first_mismatch_and_names_the_column(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        a = synthetic_trace(rows)
        rows[321] = {**rows[321], "x_position": 99999}
        b = synthetic_trace(rows)
        frame, detail = compare_traces(a, b)
        assert frame == 321
        assert "x_position" in detail

    def test_length_mismatch_reported(self):
        trace = synthetic_trace(clears_levels([(1, 1)], frames_each=300))
        frame, detail = compare_traces(trace, trace[:200])
        assert frame == 200
        assert "lengths differ" in detail

    def test_frame_column_is_not_compared(self):
        """The frame index is bookkeeping, not game state."""
        assert "frame" not in DEFAULT_COMPARE_COLUMNS

    def test_reference_verdict_overrides_invariants(self):
        """A run that dies still 'matches' if the reference died identically."""
        rows = clears_levels([(1, 1)], frames_each=500)
        for i in range(300, 360):
            rows[i] = {**rows[i], "player_state": 0x0B}
        trace = synthetic_trace(rows)
        report = verify_smb(trace, reference=trace.copy())
        assert report.passed is True
        assert report.reference_checked is True

    def test_reference_mismatch_sets_divergence_frame(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        good = synthetic_trace(rows)
        rows[404] = {**rows[404], "y_position": 7}
        bad = synthetic_trace(rows)
        report = verify_smb(bad, reference=good)
        assert report.passed is False
        assert report.diverged_at == 404

    def test_roundtrip_save_load(self, tmp_path):
        trace = synthetic_trace(clears_levels([(1, 1)], frames_each=50))
        path = save_reference(trace, tmp_path / "ref.npz", meta={"movie": "x"})
        loaded, cols = load_reference(path)
        assert np.array_equal(loaded, trace)
        assert cols == TRACE_COLUMNS


class TestReportRendering:
    def test_text_says_synced(self):
        trace = synthetic_trace(clears_levels([(1, 1), (1, 2)]))
        text = verify_smb(trace, movie_name="m.bk2").text()
        assert "SYNCED: m.bk2" in text
        assert "1-1 -> 1-2" in text

    def test_text_says_desynced_with_frame(self):
        rows = clears_levels([(1, 1)], frames_each=900)
        for i in range(500, 540):
            rows[i] = {**rows[i], "player_state": 0x06}
        text = verify_smb(synthetic_trace(rows)).text()
        assert "DESYNCED" in text
        assert "diverged at frame: 500" in text

    def test_to_dict_is_json_serialisable(self):
        import json

        trace = synthetic_trace(clears_levels([(1, 1), (1, 2)]))
        payload = verify_smb(trace).to_dict()
        assert json.loads(json.dumps(payload))["synced"] is True


class TestLongestTrueRun:
    def test_empty_and_all_false(self):
        from tasdata.verify import longest_true_run

        assert longest_true_run(np.zeros(0, dtype=bool)) == (0, 0)
        assert longest_true_run(np.zeros(10, dtype=bool)) == (0, 0)

    def test_single_run(self):
        from tasdata.verify import longest_true_run

        mask = np.zeros(10, dtype=bool)
        mask[3:7] = True
        assert longest_true_run(mask) == (3, 4)

    def test_picks_the_longest_of_several(self):
        from tasdata.verify import longest_true_run

        mask = np.zeros(20, dtype=bool)
        mask[1:3] = True
        mask[5:11] = True
        mask[15:17] = True
        assert longest_true_run(mask) == (5, 6)

    def test_run_touching_both_ends(self):
        from tasdata.verify import longest_true_run

        assert longest_true_run(np.ones(5, dtype=bool)) == (0, 5)


class TestAttractModeFiltering:
    def test_demo_replay_after_game_over_is_not_counted(self):
        """SMB returns to an attract-mode demo of 1-1 after a game over."""
        rows = clears_levels([(1, 1), (1, 2)], frames_each=400)
        # game over -> attract-mode demo replaying 1-1, player has no control
        rows += [
            {"world": 1, "stage": 1, "pregame": 0, "x_position": i}
            for i in range(400)
        ]
        report = verify_smb(synthetic_trace(rows))
        assert report.levels_reached == ["1-1", "1-2"]

    def test_playing_only_false_keeps_demo_spans(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=400)
        rows += [
            {"world": 1, "stage": 1, "pregame": 0, "x_position": i}
            for i in range(400)
        ]
        spans = level_spans(synthetic_trace(rows), playing_only=False)
        assert [s.label for s in spans] == ["1-1", "1-2", "1-1"]


class TestAccurateBootRam:
    """Real hardware (and FCEUX) power on with RAM = 0xFF, decoding as world 256."""

    def _boot_garbage(self, rows: list[dict]) -> list[dict]:
        garbage = [
            {"world": 256, "stage": 256, "pregame": 0, "x_position": 65535, "time": 0}
            for _ in range(4)
        ]
        return garbage + rows

    def test_uninitialised_ram_does_not_trip_monotonic(self):
        rows = self._boot_garbage(clears_levels([(1, 1), (1, 2)], frames_each=400))
        report = verify_smb(synthetic_trace(rows))
        check = next(c for c in report.checks if c.name == "level-monotonic")
        assert check.passed, check.detail
        assert report.passed is True

    def test_uninitialised_ram_not_reported_as_a_level(self):
        rows = self._boot_garbage(clears_levels([(1, 1), (1, 2)], frames_each=400))
        assert verify_smb(synthetic_trace(rows)).levels_reached == ["1-1", "1-2"]


class TestEndOfLevelSequence:
    """SMB drains the timer to zero on the flagpole; x freezes. Neither is a fault."""

    def _flagpole(self, frames: int = 700) -> list[dict]:
        # state 0x05 = flagpole slide, x pinned, timer paid out to zero
        return [
            {
                "world": 1, "stage": 1, "x_position": 3266,
                "player_state": 0x05, "time": max(0, 60 - i),
            }
            for i in range(frames)
        ]

    def test_time_bonus_countdown_is_not_expiry(self):
        rows = clears_levels([(1, 1)], frames_each=400) + self._flagpole()
        rows += clears_levels([(1, 2)], frames_each=400)
        check = next(
            c for c in verify_smb(synthetic_trace(rows)).checks if c.name == "timer"
        )
        assert check.passed, check.detail

    def test_flagpole_is_not_a_stall(self):
        rows = clears_levels([(1, 1)], frames_each=400) + self._flagpole()
        rows += clears_levels([(1, 2)], frames_each=400)
        check = next(
            c
            for c in verify_smb(synthetic_trace(rows), stall_frames=600).checks
            if c.name == "forward-progress"
        )
        assert check.passed, check.detail

    def test_a_real_timeout_while_in_control_still_fails(self):
        rows = clears_levels([(1, 1), (1, 2)], frames_each=500)
        for i in range(600, 700):
            rows[i] = {**rows[i], "time": 0}  # state stays 0x08
        check = next(
            c for c in verify_smb(synthetic_trace(rows)).checks if c.name == "timer"
        )
        assert check.passed is False
