"""§1: does a nonlinear action head reduce the on-top-of-pipe failures?

**⚠ PRE-SPECIFIED PRIMARY OUTCOME, stated here before the arms are run: the on-top failure COUNT at pipe 4**,
where on-top is half of all losses. Clearance at every wall is secondary. Recorded in the artifact so this is
not a post-hoc pick among walls.

**⚠ AND THE MOTIVATION IS WEAKER THAN BLOCK 63 CLAIMED — §2 corrected it before this ran.** Block 63 reported
the on-top-versus-at-face distinction as "present but not linearly reachable" from a linear probe AUC of 0.651
(p=0.17) against an MLP's 0.743. **At 38 on-top states instead of 11, the LINEAR probe reads AUC 0.859
(p=0.0000).** The 0.651 was a power artifact. The MLP still wins, by **+0.056, 95% CI [+0.012, +0.117]**
bootstrapped over states — real but small.

**So the head is NOT unable to read the distinction. It reads it well.** A nonlinear head buys ~0.056 of extra
decodability, and this block measures whether that converts into behaviour. That is a much weaker prior than
"the head cannot read the one thing that matters", and the result is reported against the weaker prior.

| arm | head | params | tests |
|---|---|---|---|
| **H0** | `Linear(64,300)` | 325,964 | baseline (the existing 1,000-step arms) |
| **H1** | `Linear(64,128) → GELU → Linear(128,300)` | 353,484 | the finding |
| **H2** | `Linear(64,256) → GELU → Linear(256,300)` | 400,204 | whether width beyond 128 adds |

Ten paired seeds, 1,000 steps, 84×84, `cnn(32,64,64)`, T=0.7, n=200, `STALL=6500`, from the level start.
Exact paired sign-flip permutation over seeds; floor 2/2¹⁰ = 0.00195.
"""
from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.button_mask_eval import rollout  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.runlength_script_control import rl_script_episode, token_stats  # noqa: E402
from scripts.scaleup_eval import _Ep, resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/nonlinear_head_eval.json"
TRACED = ROOT / "data/traces"

N_SEEDS = 10
N_EVAL = 200
TEMP = 0.7
GROUND_Y = 432
WALLS = {"goomba_320": 320, "pipe1_470": 470, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562}
CELLS = {"H0_linear": [f"PK32_84_s{i}" for i in range(N_SEEDS)],
         "H1_head128": [f"H1_head128_s{i}" for i in range(N_SEEDS)],
         "H2_head256": [f"H2_head256_s{i}" for i in range(N_SEEDS)]}
PARAMS = {"H0_linear": 325964, "H1_head128": 353484, "H2_head256": 400204}
ARM_BUDGET_S = 12 * 60


def wall_bin(x):
    for name, lo, hi in (("goomba_288", 240, 340), ("pipe1_432", 400, 500),
                         ("pipe2_592", 560, 660), ("pipe3_720", 660, 760),
                         ("pipe4_912", 860, 1000), ("koopas_1216", 1150, 1300),
                         ("frontier_1504", 1450, 1600), ("gap_1380", 1300, 1450)):
        if lo <= x < hi:
            return name
    return f"other_{int(x) // 200 * 200}"


def failure_kinds(traces):
    """On-top vs at-face failure counts, per wall. The mechanism this change targets."""
    out = collections.defaultdict(lambda: {"on_top": 0, "at_face": 0})
    for t in traces:
        fr = t.frames
        if not fr:
            continue
        # the failure is where forward progress stopped, not the last frame
        xs = [f[0] for f in fr]
        i = int(np.argmax(xs))
        w = wall_bin(xs[i])
        kind = "on_top" if fr[i][1] < GROUND_Y - 8 else "at_face"
        out[w][kind] += 1
    return {k: dict(v) for k, v in out.items()}


def sign_flip_p(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    cnt = sum(1 for s in itertools.product([1, -1], repeat=n)
              if abs(float(np.mean(d * np.array(s)))) >= obs - 1e-12)
    return cnt / (2 ** n), 2 ** n, 2.0 / (2 ** n)


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 200 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("skipped", [])
    out.update({
        "PRE_SPECIFIED_PRIMARY_OUTCOME": (
            "the on-top failure COUNT at pipe 4 (wall bin pipe4_912), where on-top is half of all "
            "losses. Clearance at every wall is SECONDARY. Declared before the arms were run."),
        "motivation_correction": (
            "block 63's 'not linearly reachable' was a power artifact: at 38 on-top states the LINEAR "
            "probe reads AUC 0.859 (p=0.0000), against 0.651 (p=0.1725) at 11 states. The MLP still "
            "wins by +0.056 [+0.012, +0.117] over states. So the head reads the distinction well and a "
            "nonlinear head buys ~0.056 of extra decodability -- a much weaker prior than 'the head "
            "cannot read the one thing that matters'."),
        "n_seeds": N_SEEDS, "n_eval": N_EVAL, "temperature": TEMP,
        "terminator": RB.describe(), "cells": CELLS, "params": PARAMS,
        "measurement_basis": "single_life_from_level_start",
        "test": "exact paired sign-flip permutation over seeds; floor 2/2^10 = 0.00195"})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    for cell, names in CELLS.items():
        for name in names:
            key = f"{cell}/{name}"
            if key in out["arms"]:
                continue
            if not (ROOT / f"data/bc_scaleup/{name}.pt").exists():
                continue
            cached = TRACED / f"jb_{name}_unbiased_{N_EVAL}.json"
            if cell == "H0_linear" and cached.exists():
                traces = [_Ep(e) for e in json.loads(cached.read_text())["episodes"]]
                src = "reused: jump-bias unbiased arm"
            else:
                if not dl.can_afford(150):
                    out["skipped"].append({"arm": key, "reason": "deadline"})
                    continue
                policy, cfg, _ = G.load_ckpt(name)
                tp = TRACED / f"nh_{name}_{N_EVAL}.json"
                try:
                    with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                        s = sess_get()
                        try:
                            traces = resumable(tp, N_EVAL,
                                               lambda i: rollout(s, policy, cfg, start, i, lut,
                                                                 byte_of, None, temp=TEMP))
                        finally:
                            s.close()
                except TimedOut as e:
                    out["skipped"].append({"arm": key, "reason": str(e)})
                    save()
                    continue
                src = "block 64"
            rec = score(key, traces)
            xs = [max(f[0] for f in t.frames) for t in traces]
            fk = failure_kinds(traces)
            rec.update({"cell": cell, "checkpoint": name, "source": src,
                        "terminator": RB.describe(),
                        "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                          "rate": float(np.mean([x > v for x in xs]))}
                                      for w, v in WALLS.items()},
                        "failure_kinds": fk,
                        "on_top_pipe4": int(fk.get("pipe4_912", {}).get("on_top", 0)),
                        "at_face_pipe4": int(fk.get("pipe4_912", {}).get("at_face", 0)),
                        "on_top_total": int(sum(v.get("on_top", 0) for v in fk.values())),
                        "completions": int(sum(
                            1 for t in traces
                            if any(len(f) > 7 and f[6] == 1 and f[7] == 2 for f in t.frames)))})
            out["arms"][key] = rec
            save()
            print(f"  {dl.stamp()} {key:28s} p3 {rec['past_wall']['pipe3_735']['rate']*100:5.1f}% "
                  f"p4 {rec['past_wall']['pipe4_975']['rate']*100:5.1f}% "
                  f"ONTOP-p4 {rec['on_top_pipe4']:3d} face-p4 {rec['at_face_pipe4']:3d} "
                  f"ontop-all {rec['on_top_total']:3d} comp {rec['completions']}", flush=True)

    # ---------------- analysis ----------------
    def vals(cell, f, wall=None):
        v = []
        for n in CELLS[cell]:
            r = out["arms"].get(f"{cell}/{n}")
            if r:
                v.append(r["past_wall"][wall]["rate"] * 100 if wall else r[f])
        return v

    res = {}
    for cell in ("H1_head128", "H2_head256"):
        row = {}
        # PRIMARY
        a, b = vals(cell, "on_top_pipe4"), vals("H0_linear", "on_top_pipe4")
        if len(a) >= 3 and len(a) == len(b):
            d = [x - y for x, y in zip(a, b)]
            p, nperm, floor = sign_flip_p(d)
            row["PRIMARY_on_top_pipe4"] = {
                "arm": a, "baseline": b, "diffs": d, "mean_diff": float(np.mean(d)),
                "perm_p": p, "floor": floor,
                "n_lower": int(sum(1 for x in d if x < 0)),
                "direction_wanted": "lower is better"}
        for f in ("on_top_total",):
            a, b = vals(cell, f), vals("H0_linear", f)
            if len(a) == len(b) and len(a) >= 3:
                d = [x - y for x, y in zip(a, b)]
                p, _, _ = sign_flip_p(d)
                row[f] = {"arm": a, "baseline": b, "mean_diff": float(np.mean(d)), "perm_p": p,
                          "n_lower": int(sum(1 for x in d if x < 0))}
        for w in WALLS:
            a, b = vals(cell, None, w), vals("H0_linear", None, w)
            if len(a) == len(b) and len(a) >= 3:
                d = [x - y for x, y in zip(a, b)]
                p, _, _ = sign_flip_p(d)
                row[f"clearance_{w}"] = {"arm_mean": float(np.mean(a)),
                                         "baseline_mean": float(np.mean(b)),
                                         "mean_diff": float(np.mean(d)), "perm_p": p,
                                         "n_positive": int(sum(1 for x in d if x > 0))}
        a, b = vals(cell, "completions"), vals("H0_linear", "completions")
        if len(a) == len(b):
            row["completions"] = {"arm": a, "baseline": b, "arm_total": int(sum(a)),
                                  "baseline_total": int(sum(b))}
        res[cell] = row
    out["analysis"] = res

    h1 = res.get("H1_head128", {}).get("PRIMARY_on_top_pipe4")
    parts = []
    if h1:
        p3 = res["H1_head128"].get("clearance_pipe3_735", {})
        parts.append(
            f"**PRIMARY (pre-specified): on-top failures at pipe 4, H1 vs H0 = {h1['mean_diff']:+.1f} "
            f"per 200 episodes ({h1['n_lower']}/{len(h1['diffs'])} seeds lower, paired sign-flip "
            f"p={h1['perm_p']:.4f}, floor {h1['floor']:.4f}).** Arm {h1['arm']} against baseline "
            f"{h1['baseline']}.")
        if p3:
            parts.append(
                f"Secondary, past pipe 3: {p3['baseline_mean']:.1f}% → {p3['arm_mean']:.1f}% "
                f"({p3['mean_diff']:+.1f} pp, p={p3['perm_p']:.4f}).")
        improved = h1["mean_diff"] < 0 and h1["perm_p"] < 0.05
        clear_up = bool(p3 and p3["mean_diff"] > 0 and p3["perm_p"] < 0.05)
        if improved and clear_up:
            parts.append("**Branch: on-top failures fall AND clearance rises — the linear head was a "
                         "bottleneck and this is the first architectural change here with a mechanism "
                         "behind it.**")
        elif improved:
            parts.append("**Branch: on-top failures fall and clearance does not — a dissociation, "
                         "reported not buried.**")
        else:
            parts.append("**Branch: nothing moves. The probe's linear/MLP gap did not transfer to "
                         "behaviour — and §2 already showed that gap is only +0.056, with the LINEAR "
                         "probe reading the distinction at AUC 0.859, so the weaker prior was the "
                         "right one.**")
    out["verdict"] = " ".join(parts) if parts else "insufficient arms"
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
