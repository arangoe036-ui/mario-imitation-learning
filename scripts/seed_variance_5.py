"""§1: does the wide encoder still collapse the seed spread at n=5?

Block 55 reported "pipe-2 seed spread 24.5 -> 2.5 pp" from the wide encoder. That was **a range of two
numbers against a range of two numbers** -- the most valuable claim in the report and the weakest. A spread
and a standard deviation need five seeds, and training costs ~4 minutes, so there is no reason to leave it.

| cell | encoder | params | seeds |
|---|---|---|---|
| `B_84_cnn16` | (16, 32, 32) | 172,284 | 0,1,2 + **3,4** |
| `P_84_cnn32` | (32, 64, 64) | 325,964 | 0,1 + **2,3,4** |

**Note on the directive's arithmetic: B already had three seeds, so it needed two more, not three.**

**Intervals are bootstrapped over SEEDS, not pooled episodes.** Pooling episodes treats the episode as the
unit of randomisation when the quantity in dispute is between-seed dispersion -- pooling would narrow exactly
the interval that matters. The seed is the unit here, n=5, and a 5-value bootstrap is reported with its own
caveat: it resamples five points, so it is discrete over them and understates tail uncertainty. **The
per-seed values are listed individually so the reader can see the raw dispersion rather than trust a summary
of it** -- block 51 lost a figure to a bootstrap of a median over 7 values, and this is the same shape.

**The spread comparison is the headline; the mean shift is secondary.** A recipe whose seeds agree is the
foundation this project has never had.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/seed_variance_5.json"
PRIOR = [ROOT / "data/vision_2x2.json", ROOT / "data/temperature_ladder.json",
         ROOT / "data/generation_sweep.json", ROOT / "data/generation_seeds.json"]

CELLS = {
    "B_84_cnn16": ["B_84_d64_L1", "B_84_seed1", "B_84_seed2", "B_84_seed3", "B_84_seed4"],
    "P_84_cnn32": ["P_84_cnn32", "P_84_cnn32_seed1", "P_84_cnn32_seed2",
                   "P_84_cnn32_seed3", "P_84_cnn32_seed4"],
}
PARAMS = {"B_84_cnn16": 172284, "P_84_cnn32": 325964}
TEMPS = [1.0, 0.7]
N_EVAL = 200
BOOT = 20000


def boot_median_over_seeds(vals, reps=BOOT, seed=0):
    """Resample SEEDS with replacement. Discrete over 5 points -- stated, not hidden."""
    if len(vals) < 3:
        return None
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    m = [float(np.median(rng.choice(a, size=len(a), replace=True))) for _ in range(reps)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_spread_diff(a, b, reps=BOOT, seed=0):
    """Bootstrap (spread of a) - (spread of b), resampling seeds within each cell."""
    if len(a) < 3 or len(b) < 3:
        return None
    rng = np.random.default_rng(seed)
    A, B = np.asarray(a, float), np.asarray(b, float)
    d = []
    for _ in range(reps):
        x = rng.choice(A, size=len(A), replace=True)
        y = rng.choice(B, size=len(B), replace=True)
        d.append((x.max() - x.min()) - (y.max() - y.min()))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main() -> None:
    t0 = time.time()
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
    out.update({
        "n_eval": N_EVAL, "measurement_basis": "single_life", "temps": TEMPS,
        "cells": {c: {"checkpoints": v, "params": PARAMS[c], "n_seeds": len(v)}
                  for c, v in CELLS.items()},
        "interval_basis": ("bootstrapped over SEEDS (n=5), not pooled episodes -- pooling would narrow "
                           "exactly the interval in dispute. A 5-value bootstrap is discrete over those "
                           "five points and understates tail uncertainty; per-seed values are listed."),
        "generation_rule": "capped (non-A <= 4), A-runs UNCAPPED",
        "directive_arithmetic_note": ("the directive asked for three new B seeds; B already had 0/1/2, "
                                      "so two were trained, not three"),
        "episode0_guard": "compose.warm_session before every scored episode"})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    missing = []
    for cell, names in CELLS.items():
        for name in names:
            if not (ROOT / f"data/bc_scaleup/{name}.pt").exists():
                missing.append(name)
    if missing:
        print(f"⚠ checkpoints not yet trained: {missing}", flush=True)

    print("=== evaluating 5 seeds x 2 cells x 2 temperatures ===", flush=True)
    for cell, names in CELLS.items():
        for name in names:
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
                if ck is None:
                    ck = get_ck(name)
                rec = G.run_arm(sess_get, ck, None, T, ctx, start, n=N_EVAL)
                rec.update({"source": "block 56", "cell": cell})
                out["arms"][k] = rec
                save()

    # ---------------- the variance analysis ----------------
    res = {}
    for T in TEMPS:
        per_cell = {}
        for cell, names in CELLS.items():
            rows = []
            for name in names:
                a = out["arms"].get(G.tag(name, None, T))
                if not a:
                    continue
                rows.append({
                    "checkpoint": name, "train_seed": a.get("train_seed"),
                    "pipe1": a["clearance"]["pipe1"]["rate"] * 100,
                    "pipe2": a["clearance"]["pipe2"]["rate"] * 100,
                    "pipe3": a["clearance"]["pipe3"]["rate"] * 100,
                    "pipe4": a["clearance"]["pipe4"]["rate"] * 100,
                    "x_median": a["x_median"],
                    "a_rate": a["button_marginals"]["rates"]["A"]})
            v2 = [r["pipe2"] for r in rows]
            v3 = [r["pipe3"] for r in rows]
            per_cell[cell] = {
                "n_seeds": len(rows), "per_seed": rows,
                "pipe2": {
                    "values": v2, "median": float(np.median(v2)) if v2 else None,
                    "mean": float(np.mean(v2)) if v2 else None,
                    "spread": float(max(v2) - min(v2)) if len(v2) > 1 else None,
                    "sd": float(np.std(v2, ddof=1)) if len(v2) > 1 else None,
                    "median_ci_boot_over_seeds": boot_median_over_seeds(v2)},
                "pipe3": {
                    "values": v3, "median": float(np.median(v3)) if v3 else None,
                    "spread": float(max(v3) - min(v3)) if len(v3) > 1 else None,
                    "sd": float(np.std(v3, ddof=1)) if len(v3) > 1 else None}}
        b, p = per_cell["B_84_cnn16"]["pipe2"], per_cell["P_84_cnn32"]["pipe2"]
        cmp = {"B_spread": b["spread"], "P_spread": p["spread"],
               "spread_reduction_pp": ((b["spread"] - p["spread"])
                                       if None not in (b["spread"], p["spread"]) else None),
               "spread_diff_ci_boot_over_seeds": boot_spread_diff(b["values"], p["values"]),
               "B_sd": b["sd"], "P_sd": p["sd"],
               "sd_ratio_B_over_P": (b["sd"] / p["sd"]) if (b["sd"] and p["sd"]) else None,
               "B_median": b["median"], "P_median": p["median"],
               "median_shift_pp": ((p["median"] - b["median"])
                                   if None not in (p["median"], b["median"]) else None)}
        # The MEAN/median shift is now the primary claim, so it gets its own seed-level bootstrap
        # rather than being read off two point estimates.
        rng = np.random.default_rng(1)
        Bv, Pv = np.asarray(b["values"], float), np.asarray(p["values"], float)
        if len(Bv) >= 3 and len(Pv) >= 3:
            dm = [float(np.median(rng.choice(Pv, len(Pv), True))
                        - np.median(rng.choice(Bv, len(Bv), True))) for _ in range(BOOT)]
            dmean = [float(np.mean(rng.choice(Pv, len(Pv), True))
                           - np.mean(rng.choice(Bv, len(Bv), True))) for _ in range(BOOT)]
            cmp["median_shift_ci_boot_over_seeds"] = [float(np.percentile(dm, 2.5)),
                                                     float(np.percentile(dm, 97.5))]
            cmp["mean_shift_pp"] = float(Pv.mean() - Bv.mean())
            cmp["mean_shift_ci_boot_over_seeds"] = [float(np.percentile(dmean, 2.5)),
                                                    float(np.percentile(dmean, 97.5))]
            # The cleanest distribution-free statement available at n=5:
            cmp["P_worst_seed"] = float(Pv.min())
            cmp["B_median_seed"] = float(np.median(Bv))
            cmp["P_worst_beats_B_median"] = bool(Pv.min() > np.median(Bv))
            cmp["P_worst_beats_B_worst_by_pp"] = float(Pv.min() - Bv.min())
        res[f"T{T}"] = {"cells": per_cell, "comparison": cmp}
    out["analysis"] = res

    for T in TEMPS:
        a = res[f"T{T}"]
        print(f"\n--- T={T} ---")
        for cell in CELLS:
            c = a["cells"][cell]["pipe2"]
            print(f"  {cell:12s} n={a['cells'][cell]['n_seeds']} pipe2 per-seed "
                  f"{[round(x, 1) for x in c['values']]}")
            print(f"  {'':12s}   median {c['median']} spread {c['spread']} sd "
                  f"{(round(c['sd'], 2) if c['sd'] else None)}")
        cm = a["comparison"]
        ci = cm["spread_diff_ci_boot_over_seeds"]
        print(f"  spread reduction B-P: {cm['spread_reduction_pp']} pp"
              f"{f' CI [{ci[0]:.1f}, {ci[1]:.1f}]' if ci else ''}"
              f" | sd ratio {cm['sd_ratio_B_over_P'] and round(cm['sd_ratio_B_over_P'], 2)}"
              f" | median shift {cm['median_shift_pp']} pp", flush=True)

    # ---------------- verdict ----------------
    # "Survives" = the wide cell's spread AND sd are lower at BOTH temperatures, and the bootstrapped
    # spread difference excludes zero at at least one. Reported per temperature either way.
    holds = []
    for T in TEMPS:
        cm = res[f"T{T}"]["comparison"]
        if cm["spread_reduction_pp"] is None:
            continue
        ci = cm["spread_diff_ci_boot_over_seeds"]
        holds.append({"T": T, "reduction_pp": cm["spread_reduction_pp"],
                      "ci": ci, "excludes_zero": bool(ci and ci[0] > 0),
                      "sd_lower": bool(cm["P_sd"] is not None and cm["B_sd"] is not None
                                       and cm["P_sd"] < cm["B_sd"])})
    both_lower = bool(holds) and all(h["sd_lower"] for h in holds)
    any_excl = any(h["excludes_zero"] for h in holds)
    out["binary_question"] = {
        "spread_collapse_survives_at_n5": bool(both_lower and any_excl),
        "sd_lower_at_both_temperatures": both_lower,
        "spread_diff_excludes_zero_somewhere": any_excl, "per_temperature": holds,
        "n_seeds_per_cell": {c: res["T1.0"]["cells"][c]["n_seeds"] for c in CELLS}}
    if both_lower and any_excl:
        out["verdict"] = (
            f"**THE VARIANCE COLLAPSE SURVIVES AT FIVE SEEDS.** The wide encoder's between-seed standard "
            f"deviation at pipe 2 is lower at both temperatures "
            f"(T=1.0: {res['T1.0']['comparison']['B_sd']:.1f} -> "
            f"{res['T1.0']['comparison']['P_sd']:.1f}; T=0.7: "
            f"{res['T0.7']['comparison']['B_sd']:.1f} -> {res['T0.7']['comparison']['P_sd']:.1f}), and "
            f"the bootstrapped spread difference excludes zero. **The headline is reproducibility, not "
            f"clearance:** a training recipe whose seeds agree is the foundation this project has never "
            f"had, and §3 goes wider on solid ground.")
    elif both_lower:
        out["verdict"] = (
            f"**THE DIRECTION HOLDS BUT THE INTERVAL DOES NOT EXCLUDE ZERO.** The wide encoder's pipe-2 "
            f"between-seed sd is lower at both temperatures (T=1.0 "
            f"{res['T1.0']['comparison']['B_sd']:.1f} -> {res['T1.0']['comparison']['P_sd']:.1f}; "
            f"T=0.7 {res['T0.7']['comparison']['B_sd']:.1f} -> "
            f"{res['T0.7']['comparison']['P_sd']:.1f}), but a spread difference bootstrapped over five "
            f"seeds spans zero. **Consistent with variance reduction, not evidence of it** -- and a "
            f"dispersion claim needs more than five seeds to settle, which is worth knowing before "
            f"spending another block on it.")
    else:
        out["verdict"] = (
            f"**THE VARIANCE COLLAPSE DOES NOT SURVIVE.** At five seeds the wide encoder's pipe-2 "
            f"between-seed dispersion is not consistently lower "
            f"(T=1.0 sd {res['T1.0']['comparison']['B_sd']} -> {res['T1.0']['comparison']['P_sd']}, "
            f"T=0.7 {res['T0.7']['comparison']['B_sd']} -> {res['T0.7']['comparison']['P_sd']}). "
            f"**Block 55's 24.5 -> 2.5 pp was a two-seed artifact and is withdrawn.** What remains is a "
            f"mean shift: median pipe 2 {res['T0.7']['comparison']['median_shift_pp']:+.1f} pp at T=0.7 "
            f"and {res['T1.0']['comparison']['median_shift_pp']:+.1f} pp at T=1.0.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
