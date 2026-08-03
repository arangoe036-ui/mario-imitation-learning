"""Is pipe 2 reachable with the jump the model can actually produce?

Decisive and cheap: no training, no emulator. Every jump the expert makes in 1-1 is
extracted from the captured traces as an (A-hold length, height gained) pair, which gives
the game's own hold-to-height curve. The height each pipe demands is read off the same
traces -- the minimum height the expert was actually at while crossing each pipe's x span.
Then the model's measured maximum A-hold is looked up on that curve.

If the model's best hold cannot reach pipe 2's required height, no amount of selection over
rollouts could ever have cleared it: filtering a population cannot produce behaviour that
never occurs in the population.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/pipe2_ceiling.json"

GROUND_Y = 176            # 1-1 ground level, from the level-start states
PIPE1_X = (420, 480)      # x span over which pipe 1 must be crossed
PIPE2_X = (575, 640)
A_BIT = NES_BUTTON_BITS["A"]


def jumps_in_1_1(run):
    """Every A-hold in 1-1, with where it started and how high Mario got."""
    tr = np.asarray(run.trace)
    w, s = column(tr, "world"), column(tr, "stage")
    x, y = column(tr, "x_position"), column(tr, "y_position")
    a = (np.asarray(run.actions, dtype=np.uint8) & A_BIT) > 0
    n = min(len(x), len(a))
    in11 = (w[:n] == 1) & (s[:n] == 1)

    out = []
    i = 0
    while i < n:
        if in11[i] and a[i] and not (i and a[i - 1]):
            j = i
            while j < n and a[j]:
                j += 1
            # The arc continues after A is released; follow until y returns to ground.
            k = j
            while k < n and k < j + 60 and y[k] < GROUND_Y:
                k += 1
            seg_y = y[i:max(k, i + 1)]
            out.append({"hold": int(j - i), "x_at_onset": int(x[i]),
                        "peak_height": int(GROUND_Y - seg_y.min()) if seg_y.size else 0,
                        "min_y": int(seg_y.min()) if seg_y.size else int(y[i])})
            i = j
        else:
            i += 1
    return out


def crossing_height(run, span):
    """Lowest y (highest point) the expert occupied while crossing an x span in 1-1."""
    tr = np.asarray(run.trace)
    w, s = column(tr, "world"), column(tr, "stage")
    x, y = column(tr, "x_position"), column(tr, "y_position")
    m = (w == 1) & (s == 1) & (x >= span[0]) & (x <= span[1])
    if not m.any():
        return None
    return {"min_y": int(y[m].min()), "height_above_ground": int(GROUND_Y - y[m].min()),
            "frames": int(m.sum())}


def main() -> None:
    split = json.loads((ROOT / "data/split.json").read_text())["splits"]
    runs = []
    for name in split["train"]:
        r = load_run_dir(ROOT / "data/runs" / name)
        if r.manifest.get("measured_route") == "warpless":
            runs.append(r)
    print(f"{len(runs)} warpless train runs\n")

    all_jumps, p1, p2 = [], [], []
    for r in runs:
        all_jumps += jumps_in_1_1(r)
        c1, c2 = crossing_height(r, PIPE1_X), crossing_height(r, PIPE2_X)
        if c1:
            p1.append(c1)
        if c2:
            p2.append(c2)

    # The jump that actually carries each pipe: the A-hold beginning just before it.
    def holds_near(span):
        return [j["hold"] for j in all_jumps if span[0] - 90 <= j["x_at_onset"] <= span[1]]

    h1, h2 = holds_near(PIPE1_X), holds_near(PIPE2_X)
    heights = {}
    for j in all_jumps:
        heights.setdefault(j["hold"], []).append(j["peak_height"])
    curve = {k: {"n": len(v), "median_height": float(np.median(v)),
                 "max_height": int(max(v))}
             for k, v in sorted(heights.items()) if len(v) >= 3}

    need1 = int(np.median([c["height_above_ground"] for c in p1])) if p1 else None
    need2 = int(np.median([c["height_above_ground"] for c in p2])) if p2 else None

    MODEL_MAX_HOLD = 8   # measured across arm B and every arm A round
    reachable = {h: c["max_height"] for h, c in curve.items() if h <= MODEL_MAX_HOLD}
    best_model_height = max(reachable.values()) if reachable else 0

    print("expert hold -> peak height (1-1, n>=3):")
    for h, c in curve.items():
        mark = "  <- within model's reach" if h <= MODEL_MAX_HOLD else ""
        print(f"  hold {h:3d} frames  n={c['n']:4d}  median height {c['median_height']:5.0f}px"
              f"  max {c['max_height']:3d}px{mark}")

    print(f"\nheight expert reaches crossing pipe 1 ({PIPE1_X}): {need1} px above ground")
    print(f"height expert reaches crossing pipe 2 ({PIPE2_X}): {need2} px above ground")
    print(f"A-holds the expert starts near pipe 1: median "
          f"{np.median(h1) if h1 else 0:.0f}, p90 {np.percentile(h1, 90) if h1 else 0:.0f}, "
          f"max {max(h1) if h1 else 0}")
    print(f"A-holds the expert starts near pipe 2: median "
          f"{np.median(h2) if h2 else 0:.0f}, p90 {np.percentile(h2, 90) if h2 else 0:.0f}, "
          f"max {max(h2) if h2 else 0}")
    print(f"\nmodel's maximum A-hold: {MODEL_MAX_HOLD} frames -> best height "
          f"{best_model_height} px")

    verdict_2 = (best_model_height >= (need2 or 1e9))
    print(f"\nPIPE 1 clearable by the model's best jump? "
          f"{'YES' if best_model_height >= (need1 or 1e9) else 'NO'} "
          f"(needs {need1}, reaches {best_model_height})")
    print(f"PIPE 2 clearable by the model's best jump? "
          f"{'YES' if verdict_2 else 'NO'} (needs {need2}, reaches {best_model_height})")

    OUT.write_text(json.dumps({
        "ground_y": GROUND_Y, "n_runs": len(runs), "n_jumps": len(all_jumps),
        "hold_to_height": curve, "pipe1_required_height": need1,
        "pipe2_required_height": need2,
        "expert_holds_near_pipe1": {"median": float(np.median(h1)) if h1 else 0,
                                    "p90": float(np.percentile(h1, 90)) if h1 else 0,
                                    "max": int(max(h1)) if h1 else 0, "n": len(h1)},
        "expert_holds_near_pipe2": {"median": float(np.median(h2)) if h2 else 0,
                                    "p90": float(np.percentile(h2, 90)) if h2 else 0,
                                    "max": int(max(h2)) if h2 else 0, "n": len(h2)},
        "model_max_hold": MODEL_MAX_HOLD, "model_best_height": best_model_height,
        "pipe1_reachable": bool(best_model_height >= (need1 or 1e9)),
        "pipe2_reachable": bool(verdict_2),
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
