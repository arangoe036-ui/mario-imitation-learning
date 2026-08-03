"""Validate the Stage 3 search oracle against the expert, on a HELD-OUT expert run.

Sampling is stratified on purpose. A-onsets are ~1.5% of frames, so 1,000 uniform samples
would contain ~15 of them and the onset agreement number would be noise. Half the budget
goes to uniform in-control frames (which is what "overall agreement" is computed on) and
half to A-onset frames (which is what "agreement at A-onsets" is computed on). The two are
reported separately and never pooled.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.bc.oracle import HORIZON, JUMP_HOLD, OracleReport, decide  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.statelib import grounded_backward_mask  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402
from tasdata.ram import PLAYER_STATE_NORMAL, column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "smb.nes"
OUT = ROOT / "data/stage3_oracle.json"

# (horizon, jump_hold, measure). The first is the design as specified; the rest probe the
# two suspects when it disagrees with the expert -- the horizon and the progress measure.
VARIANTS = [
    (60, 20, "furthest"),
    (60, 20, "final"),
    (120, 20, "final"),
    (180, 20, "final"),
    (120, 8, "final"),
]

N_UNIFORM = 500
N_ONSET = 500
A_BIT = 0x01


def pick_held_out() -> tuple[Path, Path]:
    """A warpless run from the immutable test split, with its movie."""
    split = json.loads((ROOT / "data/split.json").read_text())
    for run_id in split["splits"]["test"]:
        run_dir = ROOT / "data/runs" / run_id
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("category") == "warpless":
            return run_dir, ROOT / str(manifest["movie"]).replace(str(ROOT) + "/", "")
    run_id = split["splits"]["test"][0]
    run_dir = ROOT / "data/runs" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return run_dir, ROOT / str(manifest["movie"]).replace(str(ROOT) + "/", "")


def score(
    decisions: dict,
    truth: np.ndarray,
    uni: np.ndarray,
    ons: np.ndarray,
    horizon: int,
    jump_hold: int,
    label_seconds: float,
) -> OracleReport:
    """Turn one variant's decisions into a report.

    ``uni`` and ``ons`` are scored separately and never pooled: the uniform stratum
    carries the overall and false-positive numbers, the onset stratum the recall number.
    """

    def agree(idx: np.ndarray) -> float:
        got = [decisions[int(f)].choose_a == bool(truth[int(f)]) for f in idx]
        return float(np.mean(got)) if got else 0.0

    em_frames = sum(d.emulator_frames for d in decisions.values())
    margins = np.array([d.margin for d in decisions.values()], dtype=float)
    no_a_uni = uni[~truth[uni]]
    return OracleReport(
        n_frames=len(decisions),
        n_decided=len(decisions),
        horizon=horizon,
        jump_hold=jump_hold,
        agreement_overall=agree(uni),
        n_a_onsets=int(ons.size),
        agreement_at_a_onsets=agree(ons),
        oracle_says_jump_at_onsets=float(
            np.mean([decisions[int(f)].choose_a for f in ons]) if ons.size else 0.0
        ),
        n_expert_no_a=int(no_a_uni.size),
        oracle_jump_where_expert_none=float(
            np.mean([decisions[int(f)].choose_a for f in no_a_uni]) if no_a_uni.size else 0.0
        ),
        oracle_jump_rate=float(np.mean([decisions[int(f)].choose_a for f in uni])),
        expert_a_rate=float(np.mean(truth[uni])),
        tie_rate=float(np.mean(margins == 0)),
        emulator_frames_total=em_frames,
        emulator_frames_per_label=em_frames / len(decisions),
        seconds_per_label=label_seconds / len(decisions),
        margin_stats={
            "mean": float(margins.mean()),
            "median": float(np.median(margins)),
            "p10": float(np.percentile(margins, 10)),
            "p90": float(np.percentile(margins, 90)),
            "abs_median": float(np.median(np.abs(margins))),
        },
    )


def main() -> None:
    run_dir, movie = pick_held_out()
    print(f"held-out run : {run_dir.name}")
    print(f"movie        : {movie.name}")
    run = load_run_dir(run_dir)
    actions = run.actions.astype(np.int64)
    trace = np.asarray(run.trace)
    pregame = column(trace, "pregame")

    # Label convention matches training: the action for the state at frame i is a_{i+1}.
    n = min(len(trace) - 1, len(actions) - 1)
    truth = (actions[1 : n + 1] & A_BIT) > 0
    prev = (actions[:n] & A_BIT) > 0
    # Eligibility for the oracle is not the savestate-library filter. That one requires y
    # to be stable in both directions, which rejects every A-onset (y rises right after).
    # Here the question is only "is Mario on the ground and under control at frame i?".
    world = column(trace, "world")[:n]
    stage = column(trace, "stage")[:n]
    xpos = column(trace, "x_position")[:n]
    timer = column(trace, "time")[:n]
    state = column(trace, "player_state")[:n]
    grounded = grounded_backward_mask(run)[:n]
    in_control = (
        (pregame[:n] == 1)
        & (state == PLAYER_STATE_NORMAL)
        & (world >= 1) & (world <= 8)
        & (stage >= 1) & (stage <= 4)
        & (xpos > 0)
        & (timer > 0)
        & grounded
    )

    onsets = np.flatnonzero(truth & ~prev & in_control)
    uniform_pool = np.flatnonzero(in_control)
    rng = np.random.default_rng(0)
    uni = rng.choice(uniform_pool, size=min(N_UNIFORM, uniform_pool.size), replace=False)
    ons = rng.choice(onsets, size=min(N_ONSET, onsets.size), replace=False)
    print(f"in-control frames: {in_control.sum():,}  A-onsets available: {onsets.size:,}")
    print(f"sampling {uni.size} uniform + {ons.size} A-onset frames")

    frames = sorted(set(uni.tolist()) | set(ons.tolist()))
    ordinal_of = {f: i for i, f in enumerate(frames)}

    started = time.perf_counter()
    decisions = {}
    with FceuxSession(ROM, movie, frames) as session:
        print(f"session up: {session.n_states} states in {session.build_seconds:.1f}s")
        build = session.build_seconds
        all_reports = []
        for horizon, jump_hold, measure in VARIANTS:
            decisions = {}
            t0 = time.perf_counter()
            for frame in frames:
                decisions[frame] = decide(
                    session, ordinal_of[frame], frame,
                    horizon=horizon, jump_hold=jump_hold, measure=measure,
                )
            label_seconds = time.perf_counter() - t0
            rep = score(decisions, truth, uni, ons, horizon, jump_hold, label_seconds)
            all_reports.append(
                {"horizon": horizon, "jump_hold": jump_hold, "measure": measure,
                 "report": rep.to_dict()}
            )
            print(f"  h={horizon:3d} hold={jump_hold:2d} {measure:8s}  "
                  f"jump={rep.oracle_jump_rate * 100:5.1f}% (expert "
                  f"{rep.expert_a_rate * 100:4.1f}%)  ties={rep.tie_rate * 100:5.1f}%  "
                  f"agree={rep.agreement_overall * 100:5.1f}%  "
                  f"onset={rep.agreement_at_a_onsets * 100:5.1f}%  "
                  f"{label_seconds / len(decisions):.3f}s/label")
        report = OracleReport(**all_reports[0]["report"])
    total = time.perf_counter() - started

    print()
    print(report.text())
    print(f"\nsession build (one-off)     : {build:.1f}s")
    print(f"total wall-clock            : {total:.1f}s")

    OUT.write_text(
        json.dumps(
            {
                "held_out_run": run_dir.name,
                "movie": movie.name,
                "sampling": {"uniform": int(uni.size), "a_onset": int(ons.size),
                             "stratified": True},
                "variants": all_reports,
                "primary_report": report.to_dict(),
                "session_build_seconds": round(build, 1),
                "total_seconds": round(total, 1),
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
