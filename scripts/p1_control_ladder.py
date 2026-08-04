"""§1: the last rungs of the control ladder. Does adding Left (or Down) close the pipe-3 gap?

The script that matched the policy through pipe 2 had **Down = 0 and Left = 0**, while every learned
checkpoint runs Down 0.086-0.284 and Left 0.135-0.265. Backing off before a tall pipe is exactly the
manoeuvre anticipation would require, so the +26.0 pp pipe-3 advantage might be nothing more than
"sometimes presses Left."

Four arms, n=200, single life, same thresholds and budget as every other arm:

  `left`         A 0.85 + Left  0.135   -- top20_round2's Left rate
  `down`         A 0.85 + Down  0.086   -- top20_round2's Down rate
  `match_top20`  A 0.85 + Left 0.135 + Down 0.086  -- **every marginal matched at once**
  `rng_matched`  A 0.85 only, but consuming the RNG exactly as the policy does (see below)

`match_top20` is the real test: a script whose four button marginals all equal the checkpoint's. If that
still does not close pipe 3, the advantage cannot be a marginal.

## Why `rng_matched` exists -- a defect in the overlap test as specified

The directive asks whether the *same episodes* clear pipe 2 in both arms under paired seeds, on the
grounds that coinciding episode sets would mean the policy is *behaving as* the script rather than merely
matching its rate.

**That test cannot work as built, and the reason is a one-line difference.** The policy draws
``rng.random(8)`` per frame (one uniform per button); the script drew ``rng.random()`` once. The two
streams therefore diverge at the first frame, so **even a policy behaviourally identical to the script
would produce a statistically independent episode set.** The measured overlap (94 shared pipe-2 clears
against 93.8 expected under independence) is exactly what independence predicts -- and would have been
whether or not the hypothesis were true.

`rng_matched` repairs it: draw all eight uniforms per frame in the same order and use the **A slot**
(index 7) for the coin. A policy emitting p=(Right 1, B 1, A 0.85, rest 0) would then produce
*bit-identical* episodes, so a coincidence test against it is finally able to fail.
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
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    PIPE_THRESHOLDS,
    button_marginals,
    clearance,
)
from tasdata.bc.trace_log import EpisodeTrace, write_traces  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/p1_control_ladder.json"
TRACEDIR = ROOT / "data/traces"
POLICY_TRACES = ROOT / "data/traces/p1_200.json"          # C_control_matched_r2, seeds 0-199
SCRIPT_TRACES = ROOT / "data/traces/p2_script_p085_200.json"

A, B, RIGHT = NES_BUTTON_BITS["A"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["Right"]
LEFT, DOWN = NES_BUTTON_BITS["Left"], NES_BUTTON_BITS["Down"]
A_SLOT = NES_BUTTON_ORDER.index("A")
N, CAP, STALL = 200, 3000, 300

#: label -> (p_A, p_Left, p_Down, rng_matched). Rates are top20_round2's measured marginals.
ARMS = {
    "left": (0.85, 0.135, 0.0, False),
    "down": (0.85, 0.0, 0.086, False),
    "match_top20": (0.85, 0.135, 0.086, False),
    "rng_matched": (0.85, 0.0, 0.0, True),
}


def scripted_episode(session, start, seed, p_a, p_left, p_down, rng_matched):
    """Right+B always; A/Left/Down i.i.d. per frame at fixed rates. Single life."""
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    best = since = 0
    for _ in range(CAP):
        byte = RIGHT | B
        if rng_matched:
            # consume the RNG exactly as the policy does: eight uniforms, A in slot 7
            u = rng.random(8)
            if u[A_SLOT] < p_a:
                byte |= A
        else:
            if rng.random() < p_a:
                byte |= A
            if p_left and rng.random() < p_left:
                byte |= LEFT
            if p_down and rng.random() < p_down:
                byte |= DOWN
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


def maxx_by_seed(path):
    d = json.loads(Path(path).read_text())
    return {e["seed"]: max(f[0] for f in e["frames"]) for e in d["episodes"]}


def overlap(a: dict, b: dict, th: int) -> dict:
    """Episode-set coincidence at a threshold. Only meaningful for RNG-matched arms."""
    common = set(a) & set(b)
    A_ = {s for s in common if a[s] > th}
    B_ = {s for s in common if b[s] > th}
    inter, union = A_ & B_, A_ | B_
    exp = len(A_) * len(B_) / len(common) if common else 0.0
    return {"n_common_seeds": len(common), "a_clears": len(A_), "b_clears": len(B_),
            "both": len(inter), "a_only": len(A_ - B_), "b_only": len(B_ - A_),
            "jaccard": (len(inter) / len(union)) if union else None,
            "expected_if_independent": round(exp, 1),
            "excess_over_independence": round(len(inter) - exp, 1)}


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    pol = maxx_by_seed(POLICY_TRACES)
    pol_clear = clearance(pol.values())
    print(f"policy C_control_matched_r2 n=200: " +
          "  ".join(f"{p} {pol_clear[p]['rate'] * 100:.1f}%" for p in PIPE_THRESHOLDS), flush=True)
    base_scr = maxx_by_seed(SCRIPT_TRACES)
    base_clear = clearance(base_scr.values())
    print(f"script p=0.85 (Left 0, Down 0) n=200:  " +
          "  ".join(f"{p} {base_clear[p]['rate'] * 100:.1f}%" for p in PIPE_THRESHOLDS), flush=True)
    print(f"\nthresholds {PIPE_THRESHOLDS}\n", flush=True)

    out = {"n": N, "measurement_basis": "single_life", "seeds": f"0-{N - 1}",
           "thresholds": PIPE_THRESHOLDS, "arm_rates": {k: dict(zip(
               ("p_A", "p_Left", "p_Down", "rng_matched"), v)) for k, v in ARMS.items()},
           "policy": {"checkpoint": "C_control_matched_r2.pt", "clearance": pol_clear},
           "script_p085_left0_down0": {"clearance": base_clear},
           "overlap_test_defect": (
               "The paired-seed episode-set test cannot detect coincidence between the policy and the "
               "plain script: the policy draws rng.random(8) per frame, the script drew rng.random() "
               "once, so the streams diverge at frame 1 and identical behaviour would still give "
               "independent episode sets. The rng_matched arm repairs this."),
           "arms": {}}

    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        for label, (pa, pl, pd, rm) in ARMS.items():
            traces = [scripted_episode(s, start, i, pa, pl, pd, rm) for i in range(N)]
            write_traces(TRACEDIR / f"ladder_{label}_200.json", traces, arm=label,
                         rates={"A": pa, "Left": pl, "Down": pd, "rng_matched": rm})
            xs = {t.seed: max(f[0] for f in t.frames) for t in traces}
            cl = clearance(xs.values())
            marg = button_marginals([f for t in traces for f in t.frames])
            row = {"rates": {"A": pa, "Left": pl, "Down": pd}, "rng_matched": rm, "n": N,
                   "x_median": float(np.median(list(xs.values()))),
                   "x_max": int(max(xs.values())),
                   "clearance": cl, "button_marginals": marg,
                   "vs_policy": {}, "overlap_with_policy": {}}
            for p, th in PIPE_THRESHOLDS.items():
                lo, hi = diff_ci(cl[p]["k"], N, pol_clear[p]["k"], N)
                row["vs_policy"][p] = {"script": cl[p]["rate"], "policy": pol_clear[p]["rate"],
                                       "policy_minus_script_pp": (pol_clear[p]["rate"]
                                                                  - cl[p]["rate"]) * 100,
                                       "ci_pp": [lo * 100, hi * 100],
                                       "gap_remains": bool(lo > 0)}
                row["overlap_with_policy"][p] = overlap(xs, pol, th)
            out["arms"][label] = row
            gap3 = row["vs_policy"]["pipe3"]
            print(f"{label:12s} A {marg['rates']['A']:.3f} L {marg['rates']['Left']:.3f} "
                  f"D {marg['rates']['Down']:.3f} | " +
                  "  ".join(f"{p} {cl[p]['rate'] * 100:5.1f}" for p in PIPE_THRESHOLDS) +
                  f"  x_med {row['x_median']:.0f} | pipe3 gap "
                  f"{gap3['policy_minus_script_pp']:+.1f} pp "
                  f"[{gap3['ci_pp'][0]:+.1f},{gap3['ci_pp'][1]:+.1f}] "
                  f"{'REMAINS' if gap3['gap_remains'] else 'CLOSED'}", flush=True)
    finally:
        s.close()

    # the binary question is about the best script arm, not any single one
    best3 = min(out["arms"].items(),
                key=lambda kv: kv[1]["vs_policy"]["pipe3"]["policy_minus_script_pp"])
    label, r = best3
    g3, g4 = r["vs_policy"]["pipe3"], r["vs_policy"]["pipe4"]
    closed = not g3["gap_remains"]
    out["verdict"] = {
        "question": "Does adding Left at p~0.15 (or Down, or both) close the pipe-3 gap?",
        "best_script_arm_at_pipe3": label,
        "pipe3": g3, "pipe4": g4,
        "answer": "YES -- closed" if closed else "NO -- the gap remains",
        "statement": (
            f"THE PIPE-3 ADVANTAGE IS A BUTTON RATE: arm `{label}` reaches "
            f"{r['clearance']['pipe3']['rate'] * 100:.1f}% at pipe 3 against the policy's "
            f"{pol_clear['pipe3']['rate'] * 100:.1f}%, difference "
            f"{g3['policy_minus_script_pp']:+.1f} pp [{g3['ci_pp'][0]:+.1f}, {g3['ci_pp'][1]:+.1f}] "
            f"which does not exclude zero. The learned component has no demonstrated value anywhere "
            f"and the thesis needs restating."
            if closed else
            f"THE PIPE-3/4 ADVANTAGE IS REAL: the best script arm `{label}` still trails by "
            f"{g3['policy_minus_script_pp']:+.1f} pp [{g3['ci_pp'][0]:+.1f}, {g3['ci_pp'][1]:+.1f}] "
            f"at pipe 3 and {g4['policy_minus_script_pp']:+.1f} pp "
            f"[{g4['ci_pp'][0]:+.1f}, {g4['ci_pp'][1]:+.1f}] at pipe 4, with every button marginal "
            f"matched. Pipes 3 and 4 are state-conditional wins and are the whole of what this "
            f"project has earned."),
    }
    rm = out["arms"]["rng_matched"]["overlap_with_policy"]["pipe2"]
    out["overlap_conclusion"] = {
        "arm": "rng_matched", **rm,
        "reading": (f"{rm['both']} of the pipe-2 clears coincide against "
                    f"{rm['expected_if_independent']} expected under independence "
                    f"(excess {rm['excess_over_independence']:+}). With the RNG consumed identically, "
                    f"a policy behaving as the script would coincide almost exactly; "
                    + ("this is consistent with coincidence." if rm["excess_over_independence"] > 20
                       else "it does not, so the policy matches the script's rate at pipe 2 while "
                            "clearing a different set of episodes.")),
    }
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"]["statement"])
    print("\n" + out["overlap_conclusion"]["reading"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
