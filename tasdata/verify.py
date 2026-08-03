"""Sync verification: did the replayed run actually progress through the game?

What this can and cannot prove
------------------------------
A bit-exact sync check needs a reference trace produced by the emulator that
recorded the movie.  We do not have BizHawk here, so :func:`verify_smb` instead
checks *game-state invariants that a synced TAS satisfies and a desynced one
essentially never does*:

* the run leaves the title screen and gains control,
* the level ordinal never goes backwards,
* Mario never dies and the timer never expires (a desynced SMB TAS keeps holding
  right into a Goomba within a second or two),
* the run reaches the level it is supposed to reach.

Passing therefore means "this run progressed as a real playthrough would", not
"this matches BizHawk frame for frame".  For the stronger claim, record a
reference trace once with :func:`save_reference` and pass it back via
``reference=`` -- :func:`compare_traces` then reports the first differing frame,
which is what you want in CI when changing the harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .ram import (
    LEVEL_LOAD_FRAMES,
    PLAYER_STATE_NORMAL,
    TRACE_COLUMNS,
    SmbState,
    column,
    state_from_row,
)

#: Columns that must match for two runs to be considered bit-identical.
#: ``frame`` is excluded (it is just the row index) and so is ``score``, which is
#: derived from the same bytes as ``coins``.
DEFAULT_COMPARE_COLUMNS: tuple[str, ...] = (
    "world", "stage", "area", "x_position", "y_position",
    "player_state", "player_status", "lives", "coins", "time",
)


@dataclass
class Check:
    """One named invariant and how it fared."""

    name: str
    passed: bool
    detail: str
    #: Frame at which the invariant first broke, if it did.
    frame: int | None = None
    #: A failure that should not by itself fail the run.
    advisory: bool = False

    def line(self) -> str:
        if self.passed:
            mark = "PASS"
        elif self.advisory:
            mark = "WARN"
        else:
            mark = "FAIL"
        where = f" @frame {self.frame}" if self.frame is not None else ""
        return f"  [{mark}] {self.name}{where}: {self.detail}"


@dataclass
class LevelSpan:
    """A contiguous stretch of frames spent on one level/area."""

    level: int
    world: int
    stage: int
    area: int
    first_frame: int
    last_frame: int
    max_x: int

    @property
    def label(self) -> str:
        return f"{self.world}-{self.stage}"


@dataclass
class SyncReport:
    """Verdict for a single run."""

    movie: str
    rom: str
    n_frames: int
    passed: bool
    #: Earliest frame at which the run demonstrably went wrong, if any.
    diverged_at: int | None
    reason: str
    checks: list[Check] = field(default_factory=list)
    spans: list[LevelSpan] = field(default_factory=list)
    final_state: SmbState | None = None
    #: True when the verdict was reached by diffing against a reference trace.
    reference_checked: bool = False
    #: Outcome of the movie-vs-ROM fingerprint check, if one ran.
    rom_check_detail: str = ""
    rom_matches_movie: bool | None = None
    #: True when the movie was recorded in PAL/50 Hz mode.
    movie_is_pal: bool = False
    replay_warnings: list[str] = field(default_factory=list)

    @property
    def levels_reached(self) -> list[str]:
        seen: list[str] = []
        for s in self.spans:
            if not seen or seen[-1] != s.label:
                seen.append(s.label)
        return seen

    def to_dict(self) -> dict:
        return {
            "movie": self.movie,
            "rom": self.rom,
            "n_frames": self.n_frames,
            "synced": self.passed,
            "diverged_at": self.diverged_at,
            "reason": self.reason,
            "reference_checked": self.reference_checked,
            "rom_matches_movie": self.rom_matches_movie,
            "rom_check_detail": self.rom_check_detail,
            "movie_is_pal": self.movie_is_pal,
            "levels_reached": self.levels_reached,
            "final_state": (
                {
                    "world": self.final_state.world,
                    "stage": self.final_state.stage,
                    "area": self.final_state.area,
                    "x_position": self.final_state.x_position,
                    "lives": self.final_state.lives,
                    "time": self.final_state.time,
                    "score": self.final_state.score,
                }
                if self.final_state
                else None
            ),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "advisory": c.advisory,
                    "frame": c.frame,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "replay_warnings": self.replay_warnings,
        }

    def text(self) -> str:
        verdict = "SYNCED" if self.passed else "DESYNCED"
        lines = [
            f"{verdict}: {self.movie}",
            f"  rom              : {self.rom}"
            + ("" if self.rom_matches_movie is None else f" (header match: {self.rom_matches_movie})"),
            f"  frames replayed  : {self.n_frames}",
            f"  region           : {'PAL (50 Hz) movie' if self.movie_is_pal else 'NTSC movie'}",
            f"  method           : "
            + ("reference-trace diff" if self.reference_checked else "progression invariants"),
            f"  levels reached   : {' -> '.join(self.levels_reached) or 'none'}",
        ]
        if self.rom_check_detail:
            lines.insert(2, f"  rom check        : {self.rom_check_detail}")
        if self.final_state is not None:
            lines.append(f"  final state      : {self.final_state}")
        if self.diverged_at is not None:
            lines.append(f"  diverged at frame: {self.diverged_at}")
        lines.append(f"  reason           : {self.reason}")
        if self.replay_warnings:
            lines.append("  replay warnings  :")
            lines += [f"    - {w}" for w in self.replay_warnings]
        lines.append("  checks:")
        lines += [c.line() for c in self.checks]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Trace analysis helpers
# --------------------------------------------------------------------------- #

def level_spans(trace: np.ndarray, *, playing_only: bool = True) -> list[LevelSpan]:
    """Split a trace into contiguous (world, stage, area) spans.

    Very short spans (< 4 frames) are dropped: during a level load the world,
    stage and area bytes are written at slightly different times, which produces
    one- or two-frame phantom combinations like 1-1 area 3.

    Args:
        playing_only: ignore frames where the player does not have control. After
            a game over SMB returns to the attract-mode demo, which replays 1-1 --
            counting that would report a run as reaching "1-1 -> 1-2 -> 1-1" when
            it actually reached 1-2 and then ran out of lives.
    """
    if len(trace) == 0:
        return []
    world = column(trace, "world")
    stage = column(trace, "stage")
    area = column(trace, "area")
    x = column(trace, "x_position")
    playing = (column(trace, "pregame") == 1) & (world >= 1) & (world <= 8) & (
        stage >= 1
    ) & (stage <= 4)

    spans: list[LevelSpan] = []
    start = 0
    key = (world[0], stage[0], area[0])
    for i in range(1, len(trace) + 1):
        cur = (world[i], stage[i], area[i]) if i < len(trace) else None
        if cur != key:
            if i - start >= 4 and (not playing_only or playing[start:i].any()):
                spans.append(
                    LevelSpan(
                        level=int((key[0] - 1) * 4 + (key[1] - 1)),
                        world=int(key[0]),
                        stage=int(key[1]),
                        area=int(key[2]),
                        first_frame=start,
                        last_frame=i - 1,
                        max_x=int(x[start:i].max()),
                    )
                )
            start = i
            key = cur
    return spans


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Longest contiguous run of True in ``mask``, as ``(start, length)``.

    Returns ``(0, 0)`` when nothing is True. Used to distinguish "the timer read
    zero on 681 scattered frames during level loads" from "the timer actually
    expired and stayed expired".
    """
    if mask.size == 0 or not mask.any():
        return 0, 0
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(lengths[best])


def compare_traces(
    run: np.ndarray,
    reference: np.ndarray,
    columns: Sequence[str] = DEFAULT_COMPARE_COLUMNS,
    trace_columns: Sequence[str] = TRACE_COLUMNS,
) -> tuple[int | None, str]:
    """First frame where ``run`` differs from ``reference``.

    Returns ``(None, "")`` when they agree over their common length.
    """
    idx = [list(trace_columns).index(c) for c in columns]
    n = min(len(run), len(reference))
    if n == 0:
        return 0, "one of the traces is empty"
    diff = run[:n][:, idx] != reference[:n][:, idx]
    rows = np.flatnonzero(diff.any(axis=1))
    if rows.size == 0:
        if len(run) != len(reference):
            return n, (
                f"traces agree for {n} frames but lengths differ "
                f"(run={len(run)}, reference={len(reference)})"
            )
        return None, ""
    frame = int(rows[0])
    bad = [columns[j] for j in np.flatnonzero(diff[frame])]
    details = ", ".join(
        f"{c}: run={int(run[frame][list(trace_columns).index(c)])} "
        f"ref={int(reference[frame][list(trace_columns).index(c)])}"
        for c in bad
    )
    return frame, f"first mismatch on {details}"


def save_reference(trace: np.ndarray, path: Path | str, *, meta: dict | None = None) -> Path:
    """Store a known-good trace for later :func:`compare_traces` runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        trace=trace,
        trace_columns=np.array(TRACE_COLUMNS),
        meta=np.array([json.dumps(meta or {})]),
    )
    return path


def load_reference(path: Path | str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load a reference trace saved by :func:`save_reference`."""
    with np.load(Path(path), allow_pickle=False) as data:
        cols = tuple(str(c) for c in data["trace_columns"])
        return data["trace"], cols


# --------------------------------------------------------------------------- #
# The SMB verifier
# --------------------------------------------------------------------------- #

def parse_level_spec(spec: str) -> int:
    """``"8-4"`` -> level ordinal 31."""
    try:
        world_s, stage_s = spec.split("-")
        world, stage = int(world_s), int(stage_s)
    except ValueError as exc:
        raise ValueError(f"expected a level like '8-4', got {spec!r}") from exc
    if not (1 <= world <= 8 and 1 <= stage <= 4):
        raise ValueError(f"level out of range for SMB: {spec!r}")
    return (world - 1) * 4 + (stage - 1)


def verify_smb(
    trace: np.ndarray,
    *,
    movie_name: str = "?",
    rom_name: str = "?",
    expect_level: str | None = None,
    min_levels: int = 2,
    stall_frames: int = 2000,
    strict_stall: bool = False,
    reference: np.ndarray | None = None,
    reference_columns: Sequence[str] = TRACE_COLUMNS,
    rom_matches_movie: bool | None = None,
    rom_check_detail: str = "",
    movie_is_pal: bool = False,
    replay_warnings: Sequence[str] | None = None,
) -> SyncReport:
    """Decide whether a replayed SMB run progressed, and where it went wrong.

    Args:
        trace: packed RAM trace from :class:`~tasdata.replay.NesReplayer`.
        expect_level: e.g. ``"8-4"``; the run must reach it to pass.
        min_levels: how many distinct levels the run must visit to pass.
        stall_frames: forward progress must resume within this many frames. The
            default is deliberately loose: SMB has legitimate vertical sections
            with no horizontal movement, the longest measured being the 2-1 vine
            to Coin Heaven at 1,162 frames.
        strict_stall: treat a stall as a failure rather than an advisory.
        reference: a known-good trace; when given, the verdict is a frame-exact
            diff against it and the invariant checks become advisory context.
    """
    checks: list[Check] = []
    spans = level_spans(trace)
    n = len(trace)
    final = state_from_row(trace[-1]) if n else None
    report = SyncReport(
        movie=movie_name,
        rom=rom_name,
        n_frames=n,
        passed=False,
        diverged_at=None,
        reason="",
        spans=spans,
        final_state=final,
        rom_matches_movie=rom_matches_movie,
        rom_check_detail=rom_check_detail,
        movie_is_pal=movie_is_pal,
        replay_warnings=list(replay_warnings or []),
    )
    if n == 0:
        report.reason = "empty trace: nothing was replayed"
        report.diverged_at = 0
        report.checks = [Check("non-empty", False, "trace has no frames", 0)]
        return report

    x = column(trace, "x_position")
    lives = column(trace, "lives")
    state = column(trace, "player_state")
    time_left = column(trace, "time")
    pregame = column(trace, "pregame")
    world = column(trace, "world")
    stage = column(trace, "stage")
    level = (world - 1) * 4 + (stage - 1)

    # Real hardware powers on with RAM full of 0xFF, and FCEUX reproduces that, so
    # the first few frames decode as "world 256, stage 256" before SMB initialises
    # itself. (nes-py zeroes RAM instead, which is why this only shows up on the
    # accurate backend.) Every check below is gated on a plausible level.
    valid_level = (world >= 1) & (world <= 8) & (stage >= 1) & (stage <= 4)

    # PLAYER_STATE_NORMAL is the only state in which the player is actually
    # steering. The end-of-level flagpole slide and the walk to the castle are
    # scripted: x stops changing and the timer is deliberately drained to zero for
    # the time bonus. Neither is a stall or a timeout.
    in_control = (pregame == 1) & (state == PLAYER_STATE_NORMAL) & valid_level

    hard_failures: list[Check] = []

    # 1. Did the game ever start? A desync in the menus, a wrong ROM, or a movie
    #    anchored to a savestate all show up here.
    playing = np.flatnonzero(pregame == 1)
    if playing.size == 0:
        checks.append(Check("game-started", False, "never left the title/demo screen", 0))
        hard_failures.append(checks[-1])
    else:
        checks.append(
            Check("game-started", True, f"gained control at frame {int(playing[0])}", int(playing[0]))
        )

    # 2. The level ordinal must never fall below the furthest level reached. Warp
    #    pipes jump forwards (1-2 -> 4-1), which is fine; dropping back means a
    #    death-restart or a wild desync.
    #
    #    Compared against a running maximum rather than the previous frame: after a
    #    death the level changes during a load (pregame != 1), so an adjacent-diff
    #    test silently misses the regression and only sees a *lower* level from
    #    then on. The regression must also persist, since the world/stage/area
    #    bytes are written non-atomically during a level load.
    playable = np.where(valid_level & (pregame == 1), level, -1)
    high_water = np.maximum.accumulate(playable)
    regressed = (playable >= 0) & (playable < high_water)
    start, length = longest_true_run(regressed)
    if length >= 30:
        checks.append(
            Check(
                "level-monotonic",
                False,
                f"level fell back below the furthest level reached and stayed there "
                f"for {length} frames: {int(high_water[start]) // 4 + 1}-"
                f"{int(high_water[start]) % 4 + 1} -> {int(level[start]) // 4 + 1}-"
                f"{int(level[start]) % 4 + 1}",
                start,
            )
        )
        hard_failures.append(checks[-1])
    else:
        checks.append(Check("level-monotonic", True, "level ordinal never went backwards"))

    # 3. Deaths. This is the sharpest desync signal in SMB: a run that has lost
    #    frame alignment walks into the first enemy it meets.
    dying = np.flatnonzero(np.isin(state, [0x06, 0x0B]) & (pregame == 1))
    lost_life = np.flatnonzero((np.diff(lives.astype(np.int32)) < 0) & (lives[1:] != 255))
    death_frames = sorted({int(f) for f in dying} | {int(f) + 1 for f in lost_life})
    if death_frames:
        checks.append(
            Check(
                "no-deaths",
                False,
                f"player died {len(lost_life) or 1} time(s); first death state at "
                f"frame {death_frames[0]}",
                death_frames[0],
            )
        )
        hard_failures.append(checks[-1])
    else:
        checks.append(Check("no-deaths", True, "player never entered a dying state"))

    # 4. Timer expiry. Two ways to get this wrong, both fixed here: the timer reads
    #    0 during every level load before the game writes it, and SMB *deliberately*
    #    drains it to zero on the flagpole to pay out the time bonus (hundreds of
    #    frames, every single level). So require the player to be in normal control
    #    and require a sustained contiguous run. Genuinely running out of time kills
    #    the player anyway, which the death check catches.
    zeroed = (time_left == 0) & in_control & (np.arange(n) > LEVEL_LOAD_FRAMES)
    zero_start, zero_len = longest_true_run(zeroed)
    if zero_len > 30:
        checks.append(
            Check(
                "timer",
                False,
                f"timer sat at 0 for {zero_len} consecutive frames",
                zero_start,
            )
        )
        hard_failures.append(checks[-1])
    else:
        checks.append(Check("timer", True, "timer never expired"))

    # 5. Forward progress: track the best x reached per (level, area) and make
    #    sure a new best arrives within stall_frames. The same scan records the
    #    last frame that made *any* progress, which is the most informative
    #    divergence point for a run that simply stops advancing.
    stall_at: int | None = None
    stall_detail = ""
    progress_end = 0
    if playing.size:
        best_x = -1
        last_progress = int(playing[0])
        area = column(trace, "area")
        key = (int(level[playing[0]]), int(area[playing[0]]))
        for i in range(int(playing[0]), n):
            if not in_control[i]:
                # Loads, flagpole slides, castle walks and death animations all
                # freeze x legitimately; none of them is a stall.
                last_progress = i
                continue
            cur = (int(level[i]), int(area[i]))
            if cur != key:
                key, best_x, last_progress = cur, -1, i
                progress_end = i
                continue
            if x[i] > best_x:
                best_x = int(x[i])
                last_progress = i
                progress_end = i
            elif stall_at is None and i - last_progress > stall_frames:
                stall_at = last_progress
                stall_detail = (
                    f"no forward progress for {i - last_progress} frames in "
                    f"{int(level[i]) // 4 + 1}-{int(level[i]) % 4 + 1} "
                    f"area {int(area[i])} (stuck at x={best_x})"
                )
    if stall_at is None:
        checks.append(Check("forward-progress", True, f"x advanced within every {stall_frames}-frame window"))
    else:
        checks.append(
            Check("forward-progress", False, stall_detail, stall_at, advisory=not strict_stall)
        )
        if strict_stall:
            hard_failures.append(checks[-1])

    # 6. Coverage expectations. These describe the run as a whole rather than a
    #    single bad frame, so they are collected separately: a death at frame 877
    #    is a far more useful divergence point than "the run never left 1-1".
    aggregate_failures: list[Check] = []
    distinct = report.levels_reached
    if len(distinct) >= min_levels:
        checks.append(
            Check("level-coverage", True, f"visited {len(distinct)} level(s): {' -> '.join(distinct)}")
        )
    else:
        checks.append(
            Check(
                "level-coverage",
                False,
                f"only visited {len(distinct)} level(s) ({' -> '.join(distinct) or 'none'}), "
                f"expected at least {min_levels}; last forward progress was at "
                f"frame {progress_end}",
                progress_end,
            )
        )
        aggregate_failures.append(checks[-1])

    if expect_level is not None:
        want = parse_level_spec(expect_level)
        reached = int(level.max())
        if reached >= want:
            checks.append(Check("expected-level", True, f"reached {expect_level}"))
        else:
            checks.append(
                Check(
                    "expected-level",
                    False,
                    f"expected to reach {expect_level} but got no further than "
                    f"{reached // 4 + 1}-{reached % 4 + 1}; last forward progress "
                    f"was at frame {progress_end}",
                    progress_end,
                )
            )
            aggregate_failures.append(checks[-1])

    # 7. Reference diff. When available this is authoritative.
    if reference is not None:
        frame, detail = compare_traces(trace, reference, trace_columns=reference_columns)
        report.reference_checked = True
        if frame is None:
            checks.append(Check("reference-match", True, "trace is identical to the reference"))
            report.passed = True
            report.reason = "frame-exact match against reference trace"
            report.checks = checks
            return report
        checks.append(Check("reference-match", False, detail, frame))
        report.checks = checks
        report.passed = False
        report.diverged_at = frame
        report.reason = f"diverged from reference trace at frame {frame}: {detail}"
        return report

    report.checks = checks
    # A pointwise failure (death, timer, backwards level) pins the divergence to
    # a real frame and always wins. Aggregate failures are the fallback.
    if hard_failures or aggregate_failures:
        pool = hard_failures or aggregate_failures
        first = min(f.frame if f.frame is not None else 0 for f in pool)
        culprit = next(f for f in pool if (f.frame or 0) == first)
        report.passed = False
        report.diverged_at = first
        report.reason = f"{culprit.name} failed: {culprit.detail}"
        if hard_failures and aggregate_failures:
            report.reason += (
                " (also: " + ", ".join(c.name for c in aggregate_failures) + ")"
            )
    else:
        report.passed = True
        report.diverged_at = None
        advisories = [c for c in checks if not c.passed]
        report.reason = (
            "all progression invariants held"
            if not advisories
            else "progression invariants held (with advisories: "
            + ", ".join(c.name for c in advisories)
            + ")"
        )
    return report
