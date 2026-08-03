"""Savestate start-point library: which movie frames are worth rolling out from.

Uniformly sampling frames from an expert movie lands in states where a rollout means
nothing: mid-air, mid-death, flagpole slides, castle walks, pipe transitions, level
loads. The 1-2 level start is a concrete example -- Mario is falling out of a pipe, so a
jump input does nothing at all and any "does A work?" check silently passes.

Every sampled start must therefore be:

* **player-controlled** -- ``pregame == 1`` and ``player_state == 0x08``;
* **grounded** -- vertical position unchanged across the neighbouring frames, which is
  derivable from the captured trace without touching the emulator;
* **not in a transition** -- a plausible world/stage, a running timer, nonzero x, and
  clear of any level change by a margin.

Level starts are kept as a separate canonical set even when they fail the grounded test
(1-2, 4-2 and the underwater levels begin airborne), but they are flagged so callers know
which ones cannot be used for grounded-only checks.

The library itself lives in memory inside the emulator process (FCEUX cannot write
savestates to arbitrary paths). What is persisted here is the *index*: which frames to
capture, plus a hash per state so a rebuild in another process can be proven identical.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..dataset import LoadedRun
from ..ram import PLAYER_STATE_NORMAL, column

#: Frames of clearance required either side of a level change.
LEVEL_CHANGE_MARGIN = 60

#: A level start must have Mario near the left edge. The world counter increments during
#: the castle walk at the end of every W-4, so the first frame *labelled* W+1-1 is really
#: Mario at x=2430 finishing the previous level -- measured on all 7 of W-1 in the
#: warpless run. Mario always enters a level at x=40.
LEVEL_START_MAX_X = 100

#: A level start must show the expert's x actually changing within this many frames --
#: proof that the game has handed control over rather than still showing the level card.
CONTROL_PROBE_FRAMES = 10

#: y must be unchanged this many frames either side for a state to count as grounded.
#: 1 is not enough: a jump apex holds y still for a frame or two.
GROUND_STABLE_FRAMES = 4


@dataclass
class StartPoint:
    """One savestate-worthy frame of an expert movie."""

    frame: int
    world: int
    stage: int
    area: int
    x: int
    y: int
    player_state: int
    kind: str          # "level_start" or "trajectory"
    grounded: bool
    #: sha256 of the 2 KB RAM at this frame, filled in when the state is captured.
    ram_hash: str | None = None
    #: sha256 of the 84x84 grayscale observation the policy actually receives. RAM
    #: equality does not cover PPU state, and the model consumes pixels, not RAM --
    #: two states with identical RAM could still render differently.
    frame_hash: str | None = None

    @property
    def label(self) -> str:
        return f"{self.world}-{self.stage}"


@dataclass
class SelectionStats:
    """How brutal the filter is, so the cost of *not* filtering is visible."""

    n_frames: int
    n_candidates: int
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def reject_fraction(self) -> float:
        return 1.0 - (self.n_candidates / self.n_frames) if self.n_frames else 0.0

    def text(self) -> str:
        lines = [
            f"frames considered      : {self.n_frames:,}",
            f"usable start points    : {self.n_candidates:,} "
            f"({self.n_candidates / max(self.n_frames, 1) * 100:.1f}%)",
            f"rejected               : {self.reject_fraction * 100:.1f}% of uniform samples",
        ]
        for reason, n in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:28s} {n:8,d} ({n / max(self.n_frames, 1) * 100:5.1f}%)")
        return "\n".join(lines)


def usable_mask(run: LoadedRun) -> tuple[np.ndarray, SelectionStats]:
    """Boolean mask of frames that are valid rollout starts, plus rejection reasons."""
    trace = np.asarray(run.trace)
    n = len(trace)
    world = column(trace, "world")
    stage = column(trace, "stage")
    x = column(trace, "x_position")
    y = column(trace, "y_position")
    st = column(trace, "player_state")
    pg = column(trace, "pregame")
    tm = column(trace, "time")

    reasons: dict[str, np.ndarray] = {}
    reasons["not player-controlled (pregame)"] = pg != 1
    reasons["not normal player state"] = st != PLAYER_STATE_NORMAL
    reasons["implausible world/stage"] = ~(
        (world >= 1) & (world <= 8) & (stage >= 1) & (stage <= 4)
    )
    reasons["x == 0 (level load)"] = x <= 0
    reasons["timer == 0"] = tm <= 0

    # Airborne: y must be unchanged across a *window*, not just the adjacent frames.
    # At the apex of a jump y holds still for a frame or two, so a +/-1 test admits
    # mid-air states -- measured: 2 of 6 "grounded" samples were jump apexes, where
    # pressing A does nothing and a jump check silently passes.
    stable = np.ones(n, dtype=bool)
    for shift in range(1, GROUND_STABLE_FRAMES + 1):
        stable[shift:] &= y[shift:] == y[:-shift]
        stable[:-shift] &= y[:-shift] == y[shift:]
    reasons["airborne (y moving within +/-%d frames)" % GROUND_STABLE_FRAMES] = ~stable

    # Too close to a level change: transitions, cutscenes and loads cluster there.
    level = (world - 1) * 4 + (stage - 1)
    change = np.zeros(n, dtype=bool)
    change[1:] = level[1:] != level[:-1]
    near = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(change)
    for i in idx:
        near[max(0, i - LEVEL_CHANGE_MARGIN) : i + LEVEL_CHANGE_MARGIN] = True
    reasons["within 60 frames of a level change"] = near

    bad = np.zeros(n, dtype=bool)
    counts: dict[str, int] = {}
    for name, mask in reasons.items():
        counts[name] = int(mask.sum())
        bad |= mask

    good = ~bad
    return good, SelectionStats(n_frames=n, n_candidates=int(good.sum()), rejected=counts)


def level_start_frames(run: LoadedRun) -> list[int]:
    """First playable frame of each distinct level, in order of arrival.

    Deliberately *not* the first frame passing the full filter. A TAS is airborne for
    most of a level, so "first grounded frame" can land arbitrarily deep into it -- in
    1-1 it is x=2616, past both pipes, which would make "cleared pipe 1" true before the
    episode even began. Evaluation needs the level's actual beginning, so this applies
    only the in-control conditions and lets the airborne/transition filters go.

    The returned frames are flagged ``grounded=False`` when they fail the ground test, so
    callers doing grounded-only checks (does A do anything here?) can skip them.
    """
    trace = np.asarray(run.trace)
    world = column(trace, "world")
    stage = column(trace, "stage")
    x = column(trace, "x_position")
    tm = column(trace, "time")
    st = column(trace, "player_state")
    pg = column(trace, "pregame")

    playable = (
        (pg == 1)
        & (st == PLAYER_STATE_NORMAL)
        & (world >= 1)
        & (world <= 8)
        & (stage >= 1)
        & (stage <= 4)
        & (x > 0)
        & (x <= LEVEL_START_MAX_X)
        & (tm > 0)
    )
    # Mario is not controllable the instant the RAM flags say he is. For the first ~100
    # frames of a level the "WORLD 1-1" card is still up: pregame is 1, player_state is
    # 0x08, x is already 40, and inputs do nothing at all -- measured, holding Right+B for
    # 40 frames from frame 42 of 1-1 moves Mario zero pixels. An episode started there
    # wastes its first frames and can trip the stall detector before the game hands over
    # control. The expert's own trace dates the handover exactly: control begins on the
    # first frame where x actually starts to move.
    # Strictly *forward*, not merely different. Frame 42 of the warpless movie has
    # x=40 with pregame=1 -- boot-time RAM transient while the title screen is still up
    # (the expert presses Start on frame 41) -- and x then drops back to 0. An
    # "x changed" test accepts that; an "x increased" test rejects it, because Mario
    # never advances. Verified: 60 frames of Right+B from frame 42 leaves x at 0.
    advancing = np.zeros(len(x), dtype=bool)
    for shift in range(1, CONTROL_PROBE_FRAMES + 1):
        advancing[:-shift] |= x[shift:] > x[:-shift]
    playable &= advancing

    seen: set[tuple[int, int]] = set()
    out: list[int] = []
    for i in np.flatnonzero(playable):
        key = (int(world[i]), int(stage[i]))
        if key in seen:
            continue
        seen.add(key)
        out.append(int(i))
    return out


def grounded_backward_mask(run: LoadedRun) -> np.ndarray:
    """Frames whose y was unchanged over the *preceding* GROUND_STABLE_FRAMES frames.

    The symmetric test in :func:`usable_mask` also requires y to hold still going
    *forward*, which by construction rejects every jump onset -- at the frame the expert
    first presses A, y is about to start rising. On a held-out run that left 24 A-onsets
    out of 68,509 frames, which is why "can Mario jump from here?" needs the backward-only
    form: it asks whether he is on the ground *now*, not whether he stays there.

    Jump apexes are still excluded: at an apex y was rising over the preceding frames.
    """
    y = column(np.asarray(run.trace), "y_position")
    n = len(y)
    stable = np.ones(n, dtype=bool)
    for shift in range(1, GROUND_STABLE_FRAMES + 1):
        stable[shift:] &= y[shift:] == y[:-shift]
        stable[:shift] = False  # not enough history to judge
    return stable


def grounded_mask(run: LoadedRun) -> np.ndarray:
    """Frames whose y is unchanged across +/-GROUND_STABLE_FRAMES."""
    y = column(np.asarray(run.trace), "y_position")
    n = len(y)
    stable = np.ones(n, dtype=bool)
    for shift in range(1, GROUND_STABLE_FRAMES + 1):
        stable[shift:] &= y[shift:] == y[:-shift]
        stable[:-shift] &= y[:-shift] == y[shift:]
    return stable


def build_start_points(
    run: LoadedRun, *, n_trajectory: int = 500, seed: int = 0
) -> tuple[list[StartPoint], SelectionStats]:
    """Level starts plus ``n_trajectory`` filtered points spread across all levels."""
    trace = np.asarray(run.trace)
    world = column(trace, "world")
    stage = column(trace, "stage")
    area = column(trace, "area")
    x = column(trace, "x_position")
    y = column(trace, "y_position")
    st = column(trace, "player_state")

    good, stats = usable_mask(run)

    def make(frame: int, kind: str, grounded: bool) -> StartPoint:
        return StartPoint(
            frame=int(frame),
            world=int(world[frame]),
            stage=int(stage[frame]),
            area=int(area[frame]),
            x=int(x[frame]),
            y=int(y[frame]),
            player_state=int(st[frame]),
            kind=kind,
            grounded=bool(grounded),
        )

    grounded = grounded_mask(run)
    points: list[StartPoint] = []
    for frame in level_start_frames(run):
        points.append(make(frame, "level_start", bool(grounded[frame])))

    # Spread the trajectory samples evenly *per level* so late worlds are represented
    # rather than swamped by whichever level happens to be longest.
    level = (world - 1) * 4 + (stage - 1)
    rng = np.random.default_rng(seed)
    usable_idx = np.flatnonzero(good)
    by_level: dict[int, np.ndarray] = {}
    for lv in np.unique(level[usable_idx]):
        by_level[int(lv)] = usable_idx[level[usable_idx] == lv]
    if by_level:
        per = max(1, n_trajectory // len(by_level))
        chosen: list[int] = []
        for lv, frames in sorted(by_level.items()):
            take = min(per, frames.size)
            chosen.extend(rng.choice(frames, size=take, replace=False).tolist())
        # Top up to the target from whatever remains.
        remaining = np.setdiff1d(usable_idx, np.array(chosen, dtype=np.int64))
        short = n_trajectory - len(chosen)
        if short > 0 and remaining.size:
            chosen.extend(
                rng.choice(remaining, size=min(short, remaining.size), replace=False).tolist()
            )
        for frame in sorted(chosen)[:n_trajectory]:
            points.append(make(frame, "trajectory", True))

    return points, stats


def ram_hash(ram: np.ndarray) -> str:
    """Hash of the 2 KB RAM at a captured state.

    A proxy for the savestate: FCEUX exposes no way to serialise a savestate object from
    Lua, so PPU and APU state are not covered. RAM equality is still a strong check --
    any drift in emulation would show up in the game's own variables.
    """
    return hashlib.sha256(np.asarray(ram, dtype=np.uint8).tobytes()).hexdigest()[:16]


def frame_hash(rgb: np.ndarray) -> str:
    """Hash of the 84x84 grayscale observation derived from an emulator frame.

    Covers what RAM hashing cannot: the rendered pixels, which is what the policy sees.
    """
    from ..replay import _resize_gray

    obs = _resize_gray(np.asarray(rgb, dtype=np.uint8), (84, 84))
    return hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()[:16]


def save_index(
    path: Path | str,
    movie: Path | str,
    rom_md5: str,
    points: list[StartPoint],
    stats: SelectionStats,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "movie": str(movie),
                "rom_md5_prgchr": rom_md5,
                "n_points": len(points),
                "selection": {
                    "n_frames": stats.n_frames,
                    "n_candidates": stats.n_candidates,
                    "reject_fraction": stats.reject_fraction,
                    "rejected": stats.rejected,
                },
                "points": [asdict(p) for p in points],
            },
            indent=2,
        )
    )
    return path


def load_index(path: Path | str) -> tuple[dict, list[StartPoint]]:
    data = json.loads(Path(path).read_text())
    return data, [StartPoint(**p) for p in data["points"]]
