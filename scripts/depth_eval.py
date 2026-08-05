"""§3 and §4: does `x_max` keep rising with more training steps (L) and a wider encoder (W)?

Block 56 tested the encoder on **pipe-2 clearance** and found nothing. The owner, watching a recording, said
the wide encoder was visibly better. Re-tested on **depth** from the same checkpoints, the wide encoder gains
**+367 px of `x_max`, exact permutation p = 0.032**, and the narrow one is walled at x≈1568 in 5 of 5 seeds.
**Clearance at one obstacle is censored — it cannot see an improvement 900 px later.**

So `x_max` is the primary outcome here, with clearance reported beside it rather than instead of it.

| arm | change | steps | encoder |
|---|---|---|---|
| P (baseline) | — | 15,000 | (32,64,64) |
| **L** | training length | **60,000** | (32,64,64) |
| **W** | encoder width | 15,000 | **(48,96,96)** |

**Evaluated at `STALL=300`, the same terminator as the P baseline.** `stall_rule_audit.json` shows that rule
truncates the upper tail (p90 +82 to +127 px, 14-17% of episodes, 4 completions that were 0) while leaving the
median alone. Changing it here would confound the terminator with the thing being tested, so the comparison
runs at 300 and the tail caveat is carried in the report.

**Primary test: exact permutation on `x_max` per seed against the P baseline** — 3 seeds against 5 gives
C(8,3)=56 arrangements, so the smallest attainable two-sided p is 0.036. That is stated up front: with three
seeds this test can only detect a clean separation, nothing subtle.

Required fields per arm, per seed, listed individually: `x_max`, `x_median`, and the fraction of episodes past
each named wall.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/depth_eval.json"
PRIOR = [ROOT / "data/seed_variance_5.json", ROOT / "data/vision_2x2.json",
         ROOT / "data/temperature_ladder.json", ROOT / "data/generation_sweep.json"]

#: named walls from reach_walls.json / LEDGER §5 -- "past" means max_x strictly beyond
WALLS = {"goomba_320": 320, "pipe3_735": 735, "pipe4_975": 975,
         "koopas_1248": 1248, "frontier_1562": 1562, "flagpole_3266": 3266}
CELLS = {
    "P_15k_cnn32": ["P_84_cnn32", "P_84_cnn32_seed1", "P_84_cnn32_seed2",
                    "P_84_cnn32_seed3", "P_84_cnn32_seed4"],
    "L_60k_cnn32": ["L_84_cnn32_60k", "L_84_cnn32_60k_seed1", "L_84_cnn32_60k_seed2"],
    "W_15k_cnn48": ["W_84_cnn48", "W_84_cnn48_seed1", "W_84_cnn48_seed2"],
    "WL_60k_cnn48": ["WL_84_cnn48_60k", "WL_84_cnn48_60k_seed1", "WL_84_cnn48_60k_seed2"],
}
TEMPS = [0.7, 1.0]
N_EVAL = 200
ARM_BUDGET_S = 12 * 60


def perm_p(a, b):
    """Exact two-sided permutation p on the difference of means. Small-n honest."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None, None, None
    pool = np.concatenate([a, b])
    obs = a.mean() - b.mean()
    diffs = []
    for idx in itertools.combinations(range(len(pool)), len(a)):
        x = pool[list(idx)]
        y = pool[[i for i in range(len(pool)) if i not in idx]]
        diffs.append(x.mean() - y.mean())
    diffs = np.asarray(diffs)
    return (float((np.abs(diffs) >= abs(obs) - 1e-9).mean()),
            float(len(diffs)), float(2.0 / len(diffs)))


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 150 * 60)
    only = [c for c in (sys.argv[2].split(",") if len(sys.argv) > 2 else list(CELLS))]
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    lut_cache: dict[str, np.ndarray] = {}
    prior = {}
    for p in PRIOR:
        if p.exists():
            prior.update(json.loads(p.read_text()).get("arms", {}))

    def get_ck(name):
        policy, cfg, blob = G.load_ckpt(name)
        corpus = blob.get("corpus", "runs")
        if corpus not in lut_cache:
            z = np.load(ROOT / f"data/runlength_index_{corpus}.npz")
            lut_cache[corpus] = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")},
                                                n_cls)
        return {"name": name, "policy": policy, "cfg": cfg, "blob": blob,
                "lut": lut_cache[corpus], "byte_of": byte_of}

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("skipped", [])
    out.update({
        "n_eval": N_EVAL, "measurement_basis": "single_life", "temps": TEMPS,
        "primary_outcome": "x_max per seed; clearance reported beside it, never instead of it",
        "terminator": ("STALL=300, CAP_FRAMES=3000 -- identical to the P baseline. See "
                       "stall_rule_audit.json: that rule truncates the upper tail but not the median, "
                       "so it is held FIXED here rather than corrected mid-comparison"),
        "walls": WALLS, "cells": CELLS})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    for cell in only:
        for name in CELLS[cell]:
            if not (ROOT / f"data/bc_scaleup/{name}.pt").exists():
                continue
            ck = None
            for T in TEMPS:
                k = G.tag(name, None, T)
                if k in out["arms"]:
                    continue
                if k in prior:
                    out["arms"][k] = {**prior[k], "source": "reused", "cell": cell}
                    save()
                    continue
                if not dl.can_afford(120):
                    out["skipped"].append({"arm": k, "reason": "deadline"})
                    print(f"{dl.stamp()} SKIP {k}: deadline", flush=True)
                    continue
                if ck is None:
                    ck = get_ck(name)
                try:
                    with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), f"arm {k}"):
                        rec = G.run_arm(sess_get, ck, None, T, ctx, start, n=N_EVAL)
                    rec.update({"source": "block 57", "cell": cell})
                    # depth past each named wall, from this arm's own traces
                    tp = ROOT / rec["traces"]
                    eps = json.loads(tp.read_text())["episodes"]
                    xs = [max(f[0] for f in e["frames"]) for e in eps]
                    rec["past_wall"] = {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                            "rate": float(np.mean([x > v for x in xs]))}
                                        for w, v in WALLS.items()}
                    rec["x_p90"] = float(np.percentile(xs, 90))
                    out["arms"][k] = rec
                    save()
                except TimedOut as e:
                    out["skipped"].append({"arm": k, "reason": str(e)})
                    print(f"{dl.stamp()} TIMEOUT {k}: {e}", flush=True)
                    save()

    # ---------------- analysis ----------------
    res = {}
    for T in TEMPS:
        cells = {}
        for cell, names in CELLS.items():
            rows = []
            for name in names:
                a = out["arms"].get(G.tag(name, None, T))
                if not a:
                    continue
                rows.append({
                    "checkpoint": name, "seed": a.get("train_seed"),
                    "x_max": a["x_max"], "x_median": a["x_median"],
                    "x_p90": a.get("x_p90"),
                    "pipe2": a["clearance"]["pipe2"]["rate"] * 100,
                    "pipe3": a["clearance"]["pipe3"]["rate"] * 100,
                    "pipe4": a["clearance"]["pipe4"]["rate"] * 100,
                    "past_walls": {w: v["rate"] for w, v in (a.get("past_wall") or {}).items()},
                    "vs_script_best": {o: v["advantage_pp"] for o, v in
                                       a["vs_script_best_fixed_rate"]["per_obstacle"].items()},
                    "a_rate": a["button_marginals"]["rates"]["A"]})
            if rows:
                cells[cell] = {"n_seeds": len(rows), "per_seed": rows,
                               "x_max_values": [r["x_max"] for r in rows],
                               "x_max_median": float(np.median([r["x_max"] for r in rows])),
                               "x_max_mean": float(np.mean([r["x_max"] for r in rows])),
                               "x_median_values": [r["x_median"] for r in rows],
                               "x_median_mean": float(np.mean([r["x_median"] for r in rows]))}
        tests = {}
        base = cells.get("P_15k_cnn32")
        for cell in ("L_60k_cnn32", "W_15k_cnn48", "WL_60k_cnn48"):
            c = cells.get(cell)
            if not (base and c):
                continue
            for field in ("x_max_values", "x_median_values"):
                p, nperm, minp = perm_p(c[field], base[field])
                tests.setdefault(cell, {})[field.replace("_values", "")] = {
                    "arm_values": c[field], "baseline_values": base[field],
                    "mean_diff": float(np.mean(c[field]) - np.mean(base[field])),
                    "perm_p_two_sided": p, "n_permutations": nperm,
                    "min_attainable_p": minp}
        res[f"T{T}"] = {"cells": cells, "tests_vs_P_baseline": tests}
    out["analysis"] = res

    for T in TEMPS:
        print(f"\n--- T={T} ---")
        for cell, c in res[f"T{T}"]["cells"].items():
            print(f"  {cell:14s} n={c['n_seeds']} x_max {[int(v) for v in c['x_max_values']]} "
                  f"median {c['x_max_median']:.0f} | x_median "
                  f"{[int(v) for v in c['x_median_values']]}")
        for cell, t in res[f"T{T}"]["tests_vs_P_baseline"].items():
            xm = t.get("x_max", {})
            if xm.get("perm_p_two_sided") is not None:
                print(f"    {cell} vs P: x_max {xm['mean_diff']:+.0f} px, "
                      f"p={xm['perm_p_two_sided']:.3f} (min attainable {xm['min_attainable_p']:.3f})",
                      flush=True)

    # ---------------- §6 gate ----------------
    def gained(cell):
        best = None
        for T in TEMPS:
            t = res[f"T{T}"]["tests_vs_P_baseline"].get(cell, {}).get("x_max")
            if t and (best is None or t["mean_diff"] > best):
                best = t["mean_diff"]
        return best
    gl, gw = gained("L_60k_cnn32"), gained("W_15k_cnn48")
    out["section6_gate"] = {
        "rule": "run WL only if EITHER L or W improves x_max over the P baseline",
        "L_best_x_max_gain_px": gl, "W_best_x_max_gain_px": gw,
        "fired": bool((gl is not None and gl > 0) or (gw is not None and gw > 0)),
        "decision": ("run WL" if ((gl is not None and gl > 0) or (gw is not None and gw > 0))
                     else "skip WL -- two changes that individually do nothing are not a promising "
                          "combination")}
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(f"§6 GATE: L {gl} px, W {gw} px -> {out['section6_gate']['decision']}")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
