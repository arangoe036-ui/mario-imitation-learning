"""Selection logic: exclusions, ranking, caps, and measurement-driven relabelling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasdata.batch import route_from_levels
from tasdata.curate import (
    CATEGORY_LEVELS,
    EXCLUDED_CATEGORY_PATTERNS,
    Candidate,
    apply_measurements,
    category_rank,
    classify,
    estimate_bytes,
    excluded_reason,
    load_plan,
    select,
    write_plan,
)


def cand(
    label: str,
    *,
    category: str = "warpless",
    frames: int = 60000,
    source: str = "publication",
    chain: str = "",
    chain_position: int = 0,
    rejected: str | None = None,
    path: str = "",
) -> Candidate:
    return Candidate(
        source=source,
        source_id=label.split()[-1],
        label=label,
        path=path or f"/tmp/{label.replace(' ', '-')}.fm2",
        category=category,
        authors="someone",
        n_frames=frames,
        movie_format="fm2",
        pal=False,
        rom_ok=True,
        est_bytes=estimate_bytes(frames),
        chain=chain,
        chain_position=chain_position,
        rejected=rejected,
    )


class TestExclusions:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("game end glitch", "game-end-glitch"),
            ("arbitrary code execution", "arbitrary-code-execution"),
            ("minimum presses", "minimum-presses"),
            ("minimum A presses", "minimum-A-presses"),
            ("warpless, walkathon", "walkathon"),
            ("maximum score", "maximum-score"),
            ("maximum coins", "maximum-coins"),
        ],
    )
    def test_each_excluded_category_is_caught(self, text, expected):
        assert excluded_reason(text) == expected

    def test_filename_spellings_are_caught(self):
        assert excluded_reason("smb-maxscore-tas.fm2") == "maximum-score"
        assert excluded_reason("supermariobros-minimumpresses.fm2") == "minimum-presses"
        assert excluded_reason("smb-warps,walkathon.fm2") == "walkathon"
        assert excluded_reason("onehundredthcoin-smb-ace.bk2") == "arbitrary-code-execution"

    def test_min_a_presses_is_not_confused_with_min_presses(self):
        """The A-presses branch is more specific and must win."""
        assert excluded_reason("minimum A presses") == "minimum-A-presses"

    def test_wanted_categories_are_not_excluded(self):
        for text in ("warpless", "warps", "all items", "glitchless", "anti-pacifist"):
            assert excluded_reason(text) is None

    def test_every_pattern_has_a_name(self):
        assert all(name for _pat, name in EXCLUDED_CATEGORY_PATTERNS)


class TestClassify:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("warpless", "warpless"),
            ("all items", "all-items"),
            ("warps", "warps"),
            ("no glitches, warps", "warpless-glitchless"),
            ("something else entirely", "unknown"),
        ],
    )
    def test_classification(self, text, expected):
        assert classify(text) == expected

    def test_warpless_outranks_warps(self):
        assert category_rank("warpless") < category_rank("warps")

    def test_all_items_outranks_warps(self):
        assert category_rank("all-items") < category_rank("warps")

    def test_warps_visits_eight_levels_not_four(self):
        """Measured: 1-1, 1-2, 4-1, 4-2, 8-1, 8-2, 8-3, 8-4."""
        assert CATEGORY_LEVELS["warps"] == 8
        assert CATEGORY_LEVELS["warpless"] == 32


class TestRouteFromLevels:
    @pytest.mark.parametrize("n,expected", [(32, "warpless"), (31, "warpless"), (8, "warps"), (12, "warps")])
    def test_known_routes(self, n, expected):
        assert route_from_levels(n) == expected

    def test_partial_runs_are_named_by_their_reach(self):
        assert route_from_levels(1) == "partial-1"
        assert route_from_levels(19) == "partial-19"

    def test_zero_levels(self):
        assert route_from_levels(0) == "none"


class TestEstimateBytes:
    def test_matches_the_observed_layout(self):
        """84*84 image + 13 int32 trace + 1 action byte + 13 button bools per frame."""
        assert estimate_bytes(1) == 84 * 84 + 52 + 1 + 13
        # the measured warpless run was 478,549,877 B for 67,117 frames
        assert estimate_bytes(67117) == pytest.approx(478_549_877, rel=0.01)

    def test_scales_with_observation_shape(self):
        assert estimate_bytes(100, (42, 42)) < estimate_bytes(100, (84, 84))


class TestSelect:
    def test_unreadable_files_are_not_treated_as_duplicates(self, tmp_path: Path):
        """A missing file must not silently collapse distinct runs into one."""
        cands = [cand("pub 1", frames=17868), cand("pub 2", frames=17868)]
        selected, _ = select(cands, target=10)
        assert len(selected) == 2

    def test_byte_identical_files_are_deduped(self, tmp_path: Path):
        (tmp_path / "a.fm2").write_bytes(b"same content")
        (tmp_path / "b.fm2").write_bytes(b"same content")
        (tmp_path / "c.fm2").write_bytes(b"different")
        cands = [
            cand("pub 1", path=str(tmp_path / "a.fm2")),
            cand("pub 2", path=str(tmp_path / "b.fm2")),
            cand("pub 3", path=str(tmp_path / "c.fm2")),
        ]
        selected, rejected = select(cands, target=10)
        assert len(selected) == 2
        assert any("byte-identical" in str(c.rejected) for c in rejected)

    def test_rejected_candidates_are_never_selected(self):
        cands = [cand("pub 1", rejected="ROM mismatch"), cand("pub 2")]
        selected, rejected = select(cands, target=10)
        assert [c.label for c in selected] == ["pub 2"]
        assert len(rejected) == 1

    def test_warpless_before_warps(self):
        cands = [cand("pub w", category="warps"), cand("pub p", category="warpless")]
        selected, _ = select(cands, target=10)
        assert [c.label for c in selected] == ["pub p", "pub w"]

    def test_publications_before_userfiles_within_a_category(self):
        cands = [
            cand("user 1", source="userfile", frames=99999),
            cand("pub 1", source="publication", frames=1000),
        ]
        selected, _ = select(cands, target=10)
        assert selected[0].label == "pub 1"

    def test_target_is_respected(self):
        cands = [cand(f"pub {i}", frames=60000 + i) for i in range(20)]
        selected, rejected = select(cands, target=5)
        assert len(selected) == 5
        assert all(c.rejected == "beyond target count" for c in rejected)

    def test_low_coverage_cap_limits_warps(self):
        cands = [cand(f"pub w{i}", category="warps", frames=17000 + i) for i in range(30)]
        selected, rejected = select(cands, target=20, max_low_coverage=4)
        assert len(selected) == 4
        assert any("low-coverage cap" in str(c.rejected) for c in rejected)

    def test_cap_does_not_limit_full_game_runs(self):
        cands = [cand(f"pub p{i}", category="warpless", frames=60000 + i) for i in range(12)]
        selected, _ = select(cands, target=12, max_low_coverage=2)
        assert len(selected) == 12

    def test_default_cap_is_a_quarter_of_target(self):
        cands = [cand(f"pub w{i}", category="warps", frames=17000 + i) for i in range(30)]
        selected, _ = select(cands, target=40)
        assert len(selected) == 10

    def test_older_chain_members_come_after_newer(self):
        cands = [
            cand("pub old", chain="warpless/1", chain_position=3, frames=68000),
            cand("pub new", chain="warpless/1", chain_position=0, frames=67000),
        ]
        selected, _ = select(cands, target=10)
        assert [c.label for c in selected] == ["pub new", "pub old"]


class TestPlanRoundTrip:
    def test_write_and_load(self, tmp_path: Path):
        selected = [cand("pub 1"), cand("pub 2")]
        path = write_plan(tmp_path / "plan.json", selected, [])
        loaded = load_plan(path)
        assert [c.label for c in loaded] == ["pub 1", "pub 2"]

    def test_totals_recorded(self, tmp_path: Path):
        selected = [cand("pub 1", frames=1000), cand("pub 2", frames=2000)]
        path = write_plan(tmp_path / "plan.json", selected, [])
        data = json.loads(path.read_text())
        assert data["totals"]["n_frames"] == 3000
        assert data["totals"]["n_runs"] == 2

    def test_load_ignores_measurement_fields(self, tmp_path: Path):
        """Loading must survive the extra keys `measure --update-plan` adds."""
        path = write_plan(tmp_path / "plan.json", [cand("pub 1")], [])
        data = json.loads(path.read_text())
        data["selected"][0]["measured_levels"] = 32
        data["selected"][0]["some_future_field"] = "x"
        path.write_text(json.dumps(data))
        assert load_plan(path)[0].label == "pub 1"


class TestApplyMeasurements:
    def _setup(self, tmp_path: Path, declared: str, route: str, levels: int, synced=True):
        plan = write_plan(tmp_path / "plan.json", [cand("user 1", category=declared)], [])
        measurements = tmp_path / "m.json"
        measurements.write_text(
            json.dumps(
                [
                    {
                        "label": "user 1",
                        "declared_category": declared,
                        "n_frames": 18000,
                        "synced": synced,
                        "measured_levels": levels,
                        "furthest": "8-4",
                        "route": route,
                        "seconds": 1.0,
                    }
                ]
            )
        )
        return plan, measurements

    def test_relabels_from_measurement(self, tmp_path: Path):
        plan, m = self._setup(tmp_path, "warpless", "warps", 8)
        relabelled, desynced = apply_measurements(plan, m)
        assert relabelled == 1 and desynced == 0
        entry = json.loads(plan.read_text())["selected"][0]
        assert entry["category"] == "warps"
        assert entry["declared_category"] == "warpless"
        assert entry["measured_levels"] == 8

    def test_glitchless_qualifier_is_preserved(self, tmp_path: Path):
        plan, m = self._setup(tmp_path, "warpless-glitchless", "warps", 8)
        apply_measurements(plan, m)
        assert json.loads(plan.read_text())["selected"][0]["category"] == "warps-glitchless"

    def test_all_items_is_not_downgraded_to_warpless(self, tmp_path: Path):
        """all-items is a 32-level route; measuring 32 levels confirms, not renames."""
        plan, m = self._setup(tmp_path, "all-items", "warpless", 32)
        relabelled, _ = apply_measurements(plan, m)
        assert relabelled == 0
        assert json.loads(plan.read_text())["selected"][0]["category"] == "all-items"

    def test_correct_label_is_left_alone(self, tmp_path: Path):
        plan, m = self._setup(tmp_path, "warpless", "warpless", 32)
        relabelled, _ = apply_measurements(plan, m)
        assert relabelled == 0

    def test_desync_is_flagged(self, tmp_path: Path):
        plan, m = self._setup(tmp_path, "warpless", "partial-1", 1, synced=False)
        _relabelled, desynced = apply_measurements(plan, m)
        assert desynced == 1
        assert json.loads(plan.read_text())["selected"][0]["premeasured_synced"] is False
