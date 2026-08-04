"""P3: was the distillation regression degeneracy-removal or catastrophic forgetting?

Screens, one seed each, explicitly not rankings (LEDGER.md §2: one seed is a screen).

The 13-epoch run at 1:1 took the A-rate from 0.852 to 0.370 and reach from x_median 723 to 437, with
zero episodes arriving at pipe 4. Two hypotheses fit that:

* **degeneracy-removal** -- the A-rate had to fall for the demonstrations to be absorbed, and reach fell
  with it because reach depended on the degeneracy. Then no schedule preserves both.
* **catastrophic forgetting** -- 800 steps over 7,670 frames is ~13 epochs on 22 near-identical scripted
  segments, and the network simply overwrote everything else. Then a shorter or more diluted schedule
  preserves reach while still moving the A-rate down.

**The discriminating observation: a schedule that lowers the A-rate AND keeps reach.** That would rule
out degeneracy-removal, because it would show the two are separable.

Why this runs despite the directive gating it on P1: P1's answer is split rather than "worth nothing."
Through pipe 2 the learned component matches a three-button script exactly (137/200 vs 137/200), but at
**pipe 4 every checkpoint with A>=0.82 beats the script by +13.5 to +20 pp with intervals excluding
zero** -- and pipe 4 is the obstacle the distillation targeted. So the learned component is not worth
nothing where this question lives.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import EARLIEST, session_when_free, train  # noqa: E402
from scripts.p1_run import episode as traced_episode  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    calibrate,
    diff_ci,
    load_policy,
    random_rows,
    save_policy,
)
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    PIPE_THRESHOLDS,
    button_marginals,
    clearance,
)
from tasdata.bc.trace_log import write_traces  # noqa: E402
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/p3_distil_screens.json"
TRACEDIR = ROOT / "data/traces"
CKPT = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
DEMOS = ROOT / "data/runs_self/pipe4_demos"
BASELINE_TRACES = ROOT / "data/traces/p1_200.json"
N = 200
LR = 1e-4

#: label -> (steps, expert-frames per demo-frame). The 13-epoch run was (800, 1).
ARMS = {
    "steps100_1to1": (100, 1),
    "steps800_1to4": (800, 4),
    "steps100_1to4": (100, 4),
}


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    b = json.loads(BASELINE_TRACES.read_text())
    b_frames = [f for e in b["episodes"] for f in e["frames"]]
    b_x = [max(f[0] for f in e["frames"]) for e in b["episodes"]]
    b_marg, b_clear = button_marginals(b_frames), clearance(b_x)
    b_a, b_xmed = b_marg["rates"]["A"], float(np.median(b_x))
    print(f"baseline {CKPT.name}: A {b_a:.3f}  x_med {b_xmed:.0f}  "
          f"pipe4 {b_clear['pipe4']['rate'] * 100:.1f}%", flush=True)
    print("13-epoch 1:1 run (data/pipe4_distil.json): A 0.370  x_med 437  pipe4 0.0%\n", flush=True)

    expert = ctx.dataset([load_run_dir(ROOT / "data/runs" / n) for n in EARLIEST])
    demo_ds = ctx.dataset([load_run_dir(DEMOS)])
    out = {"n": N, "measurement_basis": "single_life", "seeds_training": 1,
           "label": "SCREENS -- one seed each, not a ranking",
           "thresholds": PIPE_THRESHOLDS,
           "baseline": {"checkpoint": CKPT.name, "a_rate": b_a, "x_median": b_xmed,
                        "clearance": b_clear, "button_marginals": b_marg},
           "prior_13_epoch_1to1": {"a_rate": 0.370, "x_median": 437.0, "pipe4_rate": 0.0},
           "arms": {}}

    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        for label, (steps, ratio) in ARMS.items():
            n_expert = min(len(expert), len(demo_ds) * ratio)
            e_rows = random_rows(expert, n_expert, seed=0)
            mixed = ConcatDataset([Subset(expert, e_rows), demo_ds])
            epochs = steps * 128 / max(len(mixed), 1)
            print(f"[{label}] {steps} steps, expert {n_expert:,} + demo {len(demo_ds):,} "
                  f"(~{epochs:.1f} epochs)", flush=True)
            policy, cfg, _ = load_policy(CKPT)     # always from the same baseline
            policy = train(policy, mixed, steps, LR, 0)
            cal, _ = calibrate(policy, expert, ctx.target_rates)
            thr = cal.vector.astype(np.float64)
            save_policy(ROOT / f"data/bc_coverage/p3_{label}.pt", policy, cfg,
                        {n: 0.5 for n in NES_BUTTON_ORDER}, distilled_from=CKPT.name)
            traces = [traced_episode(s, policy, cfg, thr, start, i) for i in range(N)]
            write_traces(TRACEDIR / f"p3_{label}_200.json", traces, arm=label)
            xs = [max(f[0] for f in t.frames) for t in traces]
            marg = button_marginals([f for t in traces for f in t.frames])
            cl = clearance(xs)
            a_lo, a_hi = diff_ci(b_clear["pipe4"]["k"], N, cl["pipe4"]["k"], N)
            row = {"steps": steps, "expert_per_demo": ratio, "epochs": round(epochs, 1),
                   "n": N, "x_median": float(np.median(xs)), "x_max": int(max(xs)),
                   "clearance": cl, "button_marginals": marg,
                   "a_rate_moved_down": bool(marg["rates"]["A"] < b_a),
                   "reach_preserved": bool(float(np.median(xs)) >= 0.9 * b_xmed),
                   "pipe4_delta_ci_pp": [a_lo * 100, a_hi * 100],
                   "ended": {k: sum(1 for t in traces if t.ended == k)
                             for k in ("died", "stuck")}}
            out["arms"][label] = row
            print(f"  -> A {marg['rates']['A']:.3f} (was {b_a:.3f})  x_med {row['x_median']:.0f} "
                  f"(was {b_xmed:.0f})  pipe1 {cl['pipe1']['rate'] * 100:.1f}  "
                  f"pipe2 {cl['pipe2']['rate'] * 100:.1f}  pipe3 {cl['pipe3']['rate'] * 100:.1f}  "
                  f"pipe4 {cl['pipe4']['rate'] * 100:.1f}  | A down "
                  f"{row['a_rate_moved_down']}, reach kept {row['reach_preserved']}", flush=True)
    finally:
        s.close()

    both = [k for k, v in out["arms"].items()
            if v["a_rate_moved_down"] and v["reach_preserved"]]
    out["verdict"] = {
        "question": "degeneracy-removal or catastrophic forgetting?",
        "discriminator": "an arm that lowers the A-rate AND preserves reach (x_median >= 90% of "
                         f"{b_xmed:.0f})",
        "arms_satisfying_both": both,
        "answer": ("FORGETTING -- " + ", ".join(both) + " lowered the A-rate while keeping reach, so "
                   "the A-rate and reach are separable and the 13-epoch collapse was the schedule "
                   "overwriting the network, not the demonstrations removing a load-bearing "
                   "degeneracy."
                   if both else
                   "DEGENERACY-REMOVAL -- no schedule lowered the A-rate while preserving reach. "
                   "Every arm that moved the marginal toward the expert lost the level, which is "
                   "what it means for reach to depend on the degeneracy rather than on skill."),
    }
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"]["answer"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
