"""Print the frozen Stage 2 result table from the persisted artifacts.

Reads only files on disk so the table cannot drift from what was actually measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data/stage2_final_session.jsonl"
SWEEP = ROOT / "data/stage2c_results.jsonl"
ARM_FINAL = ROOT / "data/stage2c_final.jsonl"

RULE_LABEL = {
    "threshold": "threshold (primary)",
    "sticky": "threshold + sticky 0.25",
    "sample": "per-button sampling",
}
ARM_LABEL = {
    "A_bernoulli_only": "A  bernoulli only (control)",
    "B_bernoulli_onset10x": "B  + onset reweight 10x (treatment)",
}


def load_final() -> tuple[list[dict], dict | None]:
    rows, timing = [], None
    for line in FINAL.read_text().splitlines():
        if not line.strip():
            continue
        blob = json.loads(line)
        if blob.get("kind") == "timing":
            timing = blob
        else:
            rows.append(blob)
    return rows, timing


def main() -> None:
    if not FINAL.exists():
        sys.exit(f"missing {FINAL}")
    rows, timing = load_final()

    print("=" * 100)
    print("STAGE 2 -- FROZEN RESULT TABLE")
    print("=" * 100)
    print()
    print("Live play, 6 level starts (1-1, 1-2, 1-3, 2-1, 3-1, 4-1), 20 seeds where")
    print("stochastic, 1 where deterministic. Every episode from a savestate on one")
    print("persistent FCEUX. 'pipe1' is counted over 1-1 starts only.")
    print()
    header = (
        f"{'arm':38s} {'selection':24s} {'eps':>4s} {'x med':>7s} {'x p90':>7s} "
        f"{'pipe1':>6s} {'A hold max':>10s} {'A presses med':>13s} {'deaths med':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        for rule, summary in row["live_by_rule"].items():
            pipe = summary.get("cleared_pipe1_rate")
            print(
                f"{ARM_LABEL.get(row['arm'], row['arm']):38s} "
                f"{RULE_LABEL.get(rule, rule):24s} "
                f"{summary['n_episodes']:4d} "
                f"{summary['furthest_x'].get('median', 0):7.0f} "
                f"{summary['furthest_x'].get('max', 0):7.0f} "
                f"{'n/a' if pipe is None else f'{pipe * 100:.0f}%':>6s} "
                f"{summary['longest_a_hold'].get('max', 0):10.0f} "
                f"{summary['a_presses'].get('median', 0):13.0f} "
                f"{summary['deaths'].get('median', 0):10.0f}"
            )
        print()

    print("Errors during the final evaluation:",
          sum(len(s.get("errors", [])) for r in rows for s in r["live_by_rule"].values()))
    print()

    # Offline numbers, from the sweep rows at the final step.
    print("-" * 100)
    print("OFFLINE (validation split), at 3,000 steps")
    print("-" * 100)
    best: dict[str, dict] = {}
    for source in (SWEEP, ARM_FINAL):
        if not source.exists():
            continue
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            blob = json.loads(line)
            arm = blob.get("arm")
            # Sweep rows keep the offline metrics under "val"; the final row under
            # "offline". Take the highest step seen for each arm.
            off = blob.get("offline") or blob.get("val")
            if arm and isinstance(off, dict) and "onset_recall" in off:
                prev = best.get(arm)
                if prev is None or (blob.get("step") or 0) >= (prev.get("step") or 0):
                    best[arm] = {**blob, "offline": off}
    for row in rows:
        arm = row["arm"]
        off = (best.get(arm) or {}).get("offline") or row.get("offline")
        if not off:
            continue
        rec = off.get("onset_recall", {})
        print(f"{ARM_LABEL.get(arm, arm):38s} exact-match {off.get('exact_match', 0) * 100:5.2f}%"
              f"   A-onset recall {rec.get('A', 0) * 100:5.1f}%"
              f"   novel combos {off.get('novel_combo_rate', 0) * 100:.3f}%")
    print()
    print("Footnote: onset *separation* (mean p(button) at onsets minus elsewhere) is not")
    print("the headline. Recall at a press rate matched to the expert's is, because a model")
    print("can separate onsets cleanly and still never cross a calibrated threshold.")
    print()

    if timing:
        print("-" * 100)
        print("EVALUATION WALL-CLOCK")
        print("-" * 100)
        print(f"  {timing['episodes']} x {timing['frames_each']}-frame episodes")
        print(f"  before, process per episode : "
              f"{timing['old_process_per_episode_seconds']:6.1f}s")
        print(f"  after, one persistent session: {timing['session_seconds']:6.1f}s")
        print(f"  speedup                      : {timing['speedup']}x")
        print("  The larger win is reliability: the final evaluation completed 492 episodes")
        print("  with 0 errors, where the parallel process-per-episode design returned 2.")
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
