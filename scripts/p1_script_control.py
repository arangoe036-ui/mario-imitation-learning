"""P1: is the learned policy worth anything over a three-button script?

Right+B held on every frame, plus A sampled i.i.d. per frame at a fixed probability. No network, no
observations, no calibration -- just the emulator and a coin. This is the trivial baseline that should
have existed before the first clearance figure in this project was ever reported.

The curve is swept over p, because the learned policy's own A-rate is 0.852 and the expert's is 0.152,
and the whole question is whether the learned component does anything a marginal cannot.

Pre-committed kill condition, from the thirty-fifth directive: **if p=0.85 reaches an x median near 723
and clears pipe 2 near 62%, the learned component is worth approximately nothing** over this script and
every performance claim in the project restates as a claim about button marginals.

Episode budget, stall rule and thresholds are identical to `p1_run.py`/`pipe4_metrics.py` so the numbers
are directly comparable to the n=200 policy baseline in `data/traces/p1_200.json`. Per-frame retention
is on for every episode, so any follow-up question is a read over the file rather than a re-run.
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
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    PIPE_THRESHOLDS,
    button_marginals,
    clearance,
)
from tasdata.bc.trace_log import EpisodeTrace, write_traces  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/p1_script_control.json"
TRACES = ROOT / "data/traces/p1_script_control.json"
BASELINE = ROOT / "data/traces/p1_200.json"

A, B, RIGHT = NES_BUTTON_BITS["A"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["Right"]
P_GRID = (0.0, 0.15, 0.50, 0.85, 1.0)   # 0.15 = the expert's rate, 0.85 = the policy's
N = 20
CAP, STALL = 3000, 300                  # identical to p1_run.py


def scripted_episode(session, start, seed: int, p_a: float) -> EpisodeTrace:
    """Right+B every frame; A ~ Bernoulli(p_a) independently per frame. Single life."""
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    best = since = 0
    for _ in range(CAP):
        byte = RIGHT | B
        if rng.random() < p_a:
            byte |= A
        obs = session.step(byte)
        t.record(obs, byte)
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in (0x06, 0x0B):
            t.record_death(obs)
            break
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                t.ended = "stuck"
                break
    return t


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    b = json.loads(BASELINE.read_text())
    b_x = [max(f[0] for f in e["frames"]) for e in b["episodes"]]
    b_clear = clearance(b_x)
    b_marg = button_marginals([f for e in b["episodes"] for f in e["frames"]])
    print(f"learned policy, n=200 (data/traces/p1_200.json): x_median {np.median(b_x):.0f}  "
          f"pipe2 {b_clear['pipe2']['rate'] * 100:.1f}%  A-rate {b_marg['rates']['A']}", flush=True)
    print(f"thresholds: {PIPE_THRESHOLDS}\n", flush=True)

    out = {"n_per_arm": N, "cap": CAP, "stall": STALL, "measurement_basis": "single_life",
           "thresholds": PIPE_THRESHOLDS, "seeds_training": 0,
           "policy_baseline_n200": {"x_median": float(np.median(b_x)), "clearance": b_clear,
                                    "button_marginals": b_marg},
           "arms": {}}

    all_traces = []
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        for p_a in P_GRID:
            traces = [scripted_episode(s, start, seed=i, p_a=p_a) for i in range(N)]
            for t in traces:
                t.seed = int(p_a * 1000) * 1000 + t.seed      # keep arms distinguishable on disk
            all_traces.extend(traces)
            xs = [max(f[0] for f in t.frames) for t in traces]
            cl = clearance(xs)
            marg = button_marginals([f for t in traces for f in t.frames])
            out["arms"][f"p{p_a:g}"] = {
                "p_a": p_a, "n": N, "x_median": float(np.median(xs)),
                "x_max": int(max(xs)), "x_p90": float(np.percentile(xs, 90)),
                "clearance": cl, "button_marginals": marg,
                "ended": {k: sum(1 for t in traces if t.ended == k) for k in ("died", "stuck")},
                "death_x": sorted(int(t.death["x"]) for t in traces if t.death),
            }
            print(f"p(A)={p_a:<4g} x median {np.median(xs):6.0f}  max {max(xs):5d}  "
                  f"pipe1 {cl['pipe1']['rate'] * 100:5.1f}%  pipe2 {cl['pipe2']['rate'] * 100:5.1f}%  "
                  f"pipe3 {cl['pipe3']['rate'] * 100:5.1f}%  pipe4 {cl['pipe4']['rate'] * 100:5.1f}%  "
                  f"(realised A {marg['rates']['A']:.3f})", flush=True)
    finally:
        s.close()

    write_traces(TRACES, all_traces, arms=list(P_GRID), n_per_arm=N, source="scripted_control")

    # the binary question, evaluated against the policy's own n=200 figures
    a85 = out["arms"]["p0.85"]
    best = max(out["arms"].values(), key=lambda r: r["clearance"]["pipe2"]["rate"])
    pol_x, pol_p2 = float(np.median(b_x)), b_clear["pipe2"]["rate"]
    near_x = a85["x_median"] >= 0.85 * pol_x
    near_p2 = a85["clearance"]["pipe2"]["rate"] >= 0.85 * pol_p2
    out["verdict"] = {
        "question": "Does Right+B held permanently with A at p=0.85 reach an x median near 723?",
        "script_p085_x_median": a85["x_median"],
        "policy_x_median": pol_x,
        "script_p085_pipe2": a85["clearance"]["pipe2"]["rate"],
        "policy_pipe2": pol_p2,
        "answer": "YES" if (near_x and near_p2) else "NO",
        "caveat": f"script arms are n={N} (wide intervals); the policy figure is n=200. "
                  f"A near-miss at n=20 is not a clean negative.",
        "statement": (
            f"THE LEARNED COMPONENT IS WORTH APPROXIMATELY NOTHING over a three-button script: "
            f"Right+B+A(0.85) reaches x median {a85['x_median']:.0f} against the policy's "
            f"{pol_x:.0f} and clears pipe 2 at {a85['clearance']['pipe2']['rate'] * 100:.1f}% "
            f"against {pol_p2 * 100:.1f}%. Every performance figure in NORTH_STAR.md restates as a "
            f"claim about button marginals."
            if (near_x and near_p2) else
            f"THE POLICY IS DOING REAL WORK: the best script arm (p={best['p_a']:g}) reaches x "
            f"median {best['x_median']:.0f} and clears pipe 2 at "
            f"{best['clearance']['pipe2']['rate'] * 100:.1f}%, against the policy's {pol_x:.0f} and "
            f"{pol_p2 * 100:.1f}%. The 85% A-rate is a style of doing the work, not a substitute "
            f"for it, so the distillation question is the live one."),
    }
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"]["statement"])
    print(f"\nwrote {OUT} and {TRACES} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
