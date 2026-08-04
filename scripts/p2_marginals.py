"""P2: button marginals for every checkpoint carrying a headline, plus the script at n=200.

The question, from the thirty-fifth directive: **composition took pipe 2 from 21.5% to 62%. Did the
A-rate rise across that same sequence?** If it did, composition's gain is a degeneracy gain rather than
a skill improvement, and that is the largest retraction available in this project.

Five checkpoints, n=200 each, single life, per-frame retention, identical episode function and identical
seeds 0-199 to `p1_run.py` so every arm is paired with every other:

  round3_ratio1to1  -- the 21.5% pipe-2 baseline
  compose_round2    -- composition
  top20_round2      -- the top-20% filter
  surv_round2       -- the survival gate, the project's best reported model
  surv_round3

Reported beside each one's pipe 1/2/3/4 clearance: **all five button marginals, each as a ratio to the
expert's own rate, and the percentage of frames spent inside an A-hold.** `button_marginals` is now a
required artifact field (LEDGER.md §2).

The scripted control from P1 is re-run here at n=200 rather than n=20, because P1's headline comparison
rested on a 20-episode arm against a 200-episode arm and the pipe-3/pipe-4 difference was the one place
the two arms visibly disagreed. Same code path, same thresholds, same budget.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from scripts.p1_run import episode as traced_episode  # noqa: E402
from scripts.p1_script_control import scripted_episode  # noqa: E402
from tasdata.bc.overnight_lib import calibrate, diff_ci, load_policy  # noqa: E402
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    PIPE_THRESHOLDS,
    button_marginals,
    clearance,
)
from tasdata.bc.trace_log import write_traces  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/p2_marginals.json"
TRACEDIR = ROOT / "data/traces"
N = 200

#: label -> checkpoint. Order is the composition sequence the retraction question is about.
CKPTS = {
    "round3_ratio1to1": ROOT / "data/bc_overnight/round3_ratio1to1.pt",
    "compose_round2": ROOT / "data/bc_compose/compose_round2.pt",
    "top20_round2": ROOT / "data/bc_compose_top20/top20_round2.pt",
    "surv_round2": ROOT / "data/bc_compose_surv/surv_round2.pt",
    "surv_round3": ROOT / "data/bc_compose_surv/surv_round3.pt",
}
SCRIPT_P = 0.85


def summarise(label: str, traces, extra: dict | None = None) -> dict:
    xs = [max(f[0] for f in t.frames) for t in traces]
    frames = [f for t in traces for f in t.frames]
    cl = clearance(xs)
    marg = button_marginals(frames)
    row = {
        "n": len(traces), "measurement_basis": "single_life", "seeds_training": 1,
        "x_median": float(np.median(xs)), "x_p90": float(np.percentile(xs, 90)),
        "x_max": int(max(xs)),
        "clearance": cl,
        "button_marginals": marg,            # required field, LEDGER.md §2
        "ended": {k: sum(1 for t in traces if t.ended == k) for k in ("died", "stuck")},
        **(extra or {}),
    }
    print(f"  {label:18s} A {marg['rates']['A']:.3f} ({marg['over_expert']['A']:>4}x)  "
          f"B {marg['rates']['B']:.3f}  R {marg['rates']['Right']:.3f}  "
          f"D {marg['rates']['Down']:.3f}  L {marg['rates']['Left']:.3f}  "
          f"inA {marg['frames_inside_a_hold_pct']:5.1f}%  | "
          f"p1 {cl['pipe1']['rate'] * 100:5.1f}  p2 {cl['pipe2']['rate'] * 100:5.1f}  "
          f"p3 {cl['pipe3']['rate'] * 100:5.1f}  p4 {cl['pipe4']['rate'] * 100:5.1f}  "
          f"x_med {row['x_median']:.0f}", flush=True)
    return row


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    out = {"n": N, "thresholds": PIPE_THRESHOLDS, "measurement_basis": "single_life",
           "seeds": f"0-{N - 1}, identical across arms", "arms": {}}
    print(f"n={N}, single life, thresholds {PIPE_THRESHOLDS}\n", flush=True)
    print(f"  {'arm':18s} {'marginals (rate, xExpert)':46s} {'in-A':>6s}  | clearance %", flush=True)

    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        # the scripted control at full n, so P1's comparison is 200 vs 200
        traces = [scripted_episode(s, start, seed=i, p_a=SCRIPT_P) for i in range(N)]
        out["arms"][f"script_p{SCRIPT_P:g}"] = summarise(
            f"script p={SCRIPT_P:g}", traces, {"learned": False, "p_a": SCRIPT_P})
        write_traces(TRACEDIR / "p2_script_p085_200.json", traces, source="scripted_control")

        for label, path in CKPTS.items():
            if not path.exists():
                print(f"  {label}: MISSING {path}", flush=True)
                out["arms"][label] = {"error": f"missing {path.name}"}
                continue
            policy, cfg, _ = load_policy(path)
            cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
            thr = cal.vector.astype(np.float64)
            traces = [traced_episode(s, policy, cfg, thr, start, i) for i in range(N)]
            out["arms"][label] = summarise(label, traces,
                                          {"learned": True, "checkpoint": path.name})
            write_traces(TRACEDIR / f"p2_{label}_200.json", traces, checkpoint=path.name)
    finally:
        s.close()

    # the retraction question: did the A-rate rise along the composition sequence?
    seq = [k for k in CKPTS if "error" not in out["arms"].get(k, {})]
    a_rates = {k: out["arms"][k]["button_marginals"]["rates"]["A"] for k in seq}
    p2 = {k: out["arms"][k]["clearance"]["pipe2"]["rate"] for k in seq}
    inA = {k: out["arms"][k]["button_marginals"]["frames_inside_a_hold_pct"] for k in seq}
    base, best = (seq[0], max(seq, key=lambda k: p2[k])) if seq else (None, None)
    rose = bool(seq) and a_rates[best] > a_rates[base]
    if seq:
        lo, hi = diff_ci(out["arms"][base]["clearance"]["pipe2"]["k"], N,
                         out["arms"][best]["clearance"]["pipe2"]["k"], N)
    scr = out["arms"][f"script_p{SCRIPT_P:g}"]
    out["retraction_question"] = {
        "question": "Did the A-rate rise across the sequence that took pipe 2 from 21.5% to 62%?",
        "a_rate_by_arm": a_rates, "pipe2_by_arm": p2, "frames_inside_a_hold_by_arm": inA,
        "baseline_arm": base, "best_pipe2_arm": best,
        "a_rate_rose": rose,
        "pipe2_delta_ci_pp": [lo * 100, hi * 100] if seq else None,
        "answer": (
            f"YES -- the A-rate rose from {a_rates[base]:.3f} at {base} to {a_rates[best]:.3f} at "
            f"{best} while pipe 2 went {p2[base] * 100:.1f}% -> {p2[best] * 100:.1f}% "
            f"[{lo * 100:+.1f}, {hi * 100:+.1f}] pp. Composition's gain moves with the marginal, so "
            f"it cannot be reported as a skill improvement without separating the two."
            if rose else
            f"NO -- the A-rate went {a_rates[base]:.3f} at {base} to {a_rates[best]:.3f} at {best} "
            f"while pipe 2 went {p2[base] * 100:.1f}% -> {p2[best] * 100:.1f}%. Composition's gain "
            f"is not explained by a rising A-rate.") if seq else "no arms measured",
    }
    if seq:
        out["vs_script"] = {
            "script_p_a": SCRIPT_P, "script_n": N,
            "note": "same episode budget, same thresholds, same n; the script has no learned part",
            "comparisons": {
                k: {p: {"script": scr["clearance"][p]["rate"],
                        "learned": out["arms"][k]["clearance"][p]["rate"],
                        "ci_pp": [c * 100 for c in diff_ci(
                            scr["clearance"][p]["k"], N, out["arms"][k]["clearance"][p]["k"], N)],
                        "excludes_zero": bool(diff_ci(
                            scr["clearance"][p]["k"], N,
                            out["arms"][k]["clearance"][p]["k"], N)[0] > 0
                            or diff_ci(scr["clearance"][p]["k"], N,
                                       out["arms"][k]["clearance"][p]["k"], N)[1] < 0)}
                    for p in PIPE_THRESHOLDS}
                for k in seq},
        }
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["retraction_question"]["answer"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
