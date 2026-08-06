"""§1: evaluate C0 / C1 / C2 at ten paired seeds, and the steps ladder that tests whether the peak moves.

| arm | training data | 1-1 exposure at 1,000 steps |
|---|---|---|
| **C0** | full corpus | ~1,908 samples drawn = **0.82 epochs** over the index |
| **C1** | 1-1 samples only (2,323) | **27.6 epochs** over 1-1 |
| **C2** | half of each batch from 1-1 | **13.8 epochs** over 1-1 |

**The ladder is the test of the diagnosis, not the clearance number.** The prediction is that restricting the
corpus moves the peak later, because the 0.82-epoch optimum would then be explained by over-fitting the 91% of
off-task levels rather than by the 1-1 signal being exhausted.

Ten paired seeds at 1,000 steps; ladder at 500 / 1k / 2k / 5k / 15k on one seed, n=100, with training loss at
every rung for comparison against the full corpus's 4.033 → 1.228.

Exact paired sign-flip permutation over seeds; **floor 2/2¹⁰ = 0.00195, stated with every p.**
"""
from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.button_mask_eval import rollout  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.nonlinear_head_eval import failure_kinds  # noqa: E402
from scripts.scaleup_eval import _Ep, resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/level_restricted_arms.json"
LADOUT = ROOT / "data/level_restricted_ladder.json"
TRACED = ROOT / "data/traces"
CKDIR = ROOT / "data/bc_scaleup"

N_SEEDS, N_EVAL, N_LAD = 10, 200, 100
TEMP = 0.7
RUNGS = [500, 1000, 2000, 5000, 15000]
WALLS = {"goomba_320": 320, "pipe1_470": 470, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562}
CELLS = {"C0_full": [f"PK32_84_s{i}" for i in range(N_SEEDS)],
         "C1_only11": [f"C1_84_s{i}" for i in range(N_SEEDS)],
         "C2_half11": [f"C2_84_s{i}" for i in range(N_SEEDS)]}
ARM_BUDGET_S = 12 * 60


def sign_flip_p(d):
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    c = sum(1 for s in itertools.product([1, -1], repeat=n)
            if abs(float(np.mean(d * np.array(s)))) >= obs - 1e-12)
    return c / 2 ** n, 2.0 / 2 ** n


def load(name):
    blob = torch.load(CKDIR / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig.from_dict(cfg)
    p = BCPolicy(cfg)
    p.load_state_dict(blob["model_state"])
    p.eval()
    return p, cfg, blob


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 220 * 60)
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
    out.update({"n_seeds": N_SEEDS, "n_eval": N_EVAL, "temperature": TEMP,
                "terminator": RB.describe(), "cells": CELLS,
                "measurement_basis": "single_life_from_level_start",
                "exposure_at_1000_steps": {
                    "C0": "~1,908 of 64,000 draws are 1-1 = 0.82 epochs over the full index",
                    "C1": "64,000 draws over 2,323 1-1 samples = 27.6 epochs",
                    "C2": "32,000 draws over 2,323 1-1 samples = 13.8 epochs"},
                "test": "exact paired sign-flip permutation over seeds; floor 2/2^10 = 0.00195"})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    def eval_one(key, name, n, tp, cell=None, extra=None):
        if key in out["arms"]:
            return
        if not (CKDIR / f"{name}.pt").exists():
            return
        # C0 is the block-64 H0 baseline: same recipe, already rolled out. s0-s2 came from the
        # jump-bias unbiased arm, s3-s9 from block 64's own rollouts. Reuse the TRACES, not the
        # summary records, so every field below is recomputed by identical code.
        cached = next((p for p in (TRACED / f"jb_{name}_unbiased_{N_EVAL}.json",
                                   TRACED / f"nh_{name}_{N_EVAL}.json")
                       if p.exists()), None)
        if cell == "C0_full" and cached is not None:
            traces = [_Ep(e) for e in json.loads(cached.read_text())["episodes"]]
            src = f"reused: {cached.name}"
            blob = torch.load(CKDIR / f"{name}.pt", map_location="cpu", weights_only=False)
        else:
            if not dl.can_afford(150):
                out["skipped"].append({"arm": key, "reason": "deadline"})
                return
            policy, cfg, blob = load(name)
            try:
                with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                    s = sess_get()
                    try:
                        traces = resumable(tp, n,
                                           lambda i: rollout(s, policy, cfg, start, i, lut,
                                                             byte_of, None, temp=TEMP))
                    finally:
                        s.close()
            except TimedOut as e:
                out["skipped"].append({"arm": key, "reason": str(e)})
                save()
                return
            src = "block 65"
        rec = score(key, traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        fk = failure_kinds(traces)
        rec.update({"cell": cell, "checkpoint": name, "source": src, "n": len(traces),
                    "terminator": RB.describe(),
                    "final_loss": blob.get("final_loss") or blob.get("loss_at_snapshot"),
                    "epochs_over_1_1": blob.get("epochs_over_1_1"),
                    "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                      "rate": float(np.mean([x > v for x in xs]))}
                                  for w, v in WALLS.items()},
                    "failure_kinds": fk,
                    "on_top_pipe4": int(fk.get("pipe4_912", {}).get("on_top", 0)),
                    "on_top_total": int(sum(v.get("on_top", 0) for v in fk.values())),
                    "completions": int(sum(
                        1 for t in traces
                        if any(len(f) > 7 and f[6] == 1 and f[7] == 2 for f in t.frames)))})
        if extra:
            rec.update(extra)
        out["arms"][key] = rec
        save()
        print(f"  {dl.stamp()} {key:26s} loss {(rec['final_loss'] or float('nan')):.3f} "
              f"p2 {rec['past_wall']['pipe2_630']['rate']*100:5.1f}% "
              f"p3 {rec['past_wall']['pipe3_735']['rate']*100:5.1f}% "
              f"p4 {rec['past_wall']['pipe4_975']['rate']*100:5.1f}% "
              f"x_med {rec['x_median']:4.0f} comp {rec['completions']}", flush=True)

    for cell, names in CELLS.items():
        for name in names:
            eval_one(f"{cell}/{name}", name, N_EVAL, TRACED / f"lr_{name}_{N_EVAL}.json", cell)

    # ---------------- paired analysis ----------------
    def vals(cell, field, wall=None):
        v = []
        for n in CELLS[cell]:
            r = out["arms"].get(f"{cell}/{n}")
            if r:
                v.append(r["past_wall"][wall]["rate"] * 100 if wall else r[field])
        return v

    res = {}
    for cell in ("C1_only11", "C2_half11"):
        row = {}
        for w in WALLS:
            a, b = vals(cell, None, w), vals("C0_full", None, w)
            if len(a) == len(b) and len(a) >= 3:
                d = [x - y for x, y in zip(a, b)]
                p, floor = sign_flip_p(d)
                row[w] = {"arm_mean": float(np.mean(a)), "baseline_mean": float(np.mean(b)),
                          "arm": a, "baseline": b, "mean_diff": float(np.mean(d)),
                          "perm_p": p, "floor": floor,
                          "n_positive": int(sum(1 for x in d if x > 0))}
        for f in ("x_median", "on_top_pipe4", "on_top_total", "completions"):
            a, b = vals(cell, f), vals("C0_full", f)
            if len(a) == len(b) and len(a) >= 3:
                d = [x - y for x, y in zip(a, b)]
                p, _ = sign_flip_p(d)
                row[f] = {"arm": a, "baseline": b, "mean_diff": float(np.mean(d)), "perm_p": p}
        res[cell] = row
    out["analysis"] = res
    # the run-length script bar, from the existing control artifact — the level that any learned
    # arm has to clear before "learning bought it" means anything.
    sc = ROOT / "data/runlength_script_control.json"
    if sc.exists():
        arms = json.loads(sc.read_text())["arms"]
        out["script_bar"] = {k: {"x_median": v["x_median"], "x_max": v["x_max"],
                                 "clearance": {w: v["clearance"][w]["rate"] * 100
                                               for w in v.get("clearance", {})}}
                             for k, v in arms.items()}
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()

    # ---------------- the ladder ----------------
    lad = json.loads(LADOUT.read_text()) if LADOUT.exists() else {}
    lad.setdefault("rungs", {})
    lad.update({"rungs_planned": RUNGS, "n_eval": N_LAD, "seed": 0,
                "terminator": RB.describe(),
                "full_corpus_reference": {"peak_steps": 1000,
                                          "loss_curve": "4.033 -> 1.228 over 500..60000"},
                "prediction_under_test": "restricting the corpus moves the PEAK later"})
    for tag in ("C1", "C2"):
        for st in RUNGS:
            nm = f"{tag}LAD_s0_{st}"
            key = f"{tag}@{st}"
            if key in lad["rungs"] or not (CKDIR / f"{nm}.pt").exists():
                continue
            if not dl.can_afford(120):
                lad.setdefault("skipped", []).append(key)
                continue
            policy, cfg, blob = load(nm)
            tp = TRACED / f"lrlad_{nm}_{N_LAD}.json"
            try:
                with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                    s = sess_get()
                    try:
                        traces = resumable(tp, N_LAD,
                                           lambda i: rollout(s, policy, cfg, start, i, lut,
                                                             byte_of, None, temp=TEMP))
                    finally:
                        s.close()
            except TimedOut as e:
                lad.setdefault("skipped", []).append({"rung": key, "reason": str(e)})
                LADOUT.write_text(json.dumps(lad, indent=2, default=str))
                continue
            rec = score(key, traces)
            xs = [max(f[0] for f in t.frames) for t in traces]
            rec.update({"tag": tag, "steps": st, "loss": blob.get("loss_at_snapshot"),
                        "past_wall": {w: {"rate": float(np.mean([x > v for x in xs]))}
                                      for w, v in WALLS.items()}})
            lad["rungs"][key] = rec
            LADOUT.write_text(json.dumps(lad, indent=2, default=str))
            print(f"  {dl.stamp()} ladder {key:10s} loss {(rec['loss'] or float('nan')):.3f} "
                  f"x_med {rec['x_median']:4.0f} "
                  f"p2 {rec['past_wall']['pipe2_630']['rate']*100:5.1f}% "
                  f"p3 {rec['past_wall']['pipe3_735']['rate']*100:5.1f}%", flush=True)
    for tag in ("C1", "C2"):
        rr = [lad["rungs"][f"{tag}@{s}"] for s in RUNGS if f"{tag}@{s}" in lad["rungs"]]
        if rr:
            best = max(rr, key=lambda r: r["past_wall"]["pipe3_735"]["rate"])
            lad.setdefault("peak", {})[tag] = {
                "steps": best["steps"], "past_pipe3": best["past_wall"]["pipe3_735"]["rate"] * 100,
                "moved_later_than_1000": bool(best["steps"] > 1000),
                "curve": [{"steps": r["steps"], "loss": r["loss"],
                           "x_median": r["x_median"],
                           "past_pipe3": r["past_wall"]["pipe3_735"]["rate"] * 100} for r in rr]}
    LADOUT.write_text(json.dumps(lad, indent=2, default=str))

    p3c1 = res.get("C1_only11", {}).get("pipe3_735")
    p3c2 = res.get("C2_half11", {}).get("pipe3_735")
    pk = lad.get("peak", {})
    parts = []
    if p3c1:
        parts.append(f"**C1 (1-1 only) past pipe 3: {p3c1['baseline_mean']:.1f}% → "
                     f"{p3c1['arm_mean']:.1f}% ({p3c1['mean_diff']:+.1f} pp, "
                     f"{p3c1['n_positive']}/10 seeds up, p={p3c1['perm_p']:.4f}, floor "
                     f"{p3c1['floor']:.4f}).**")
    if p3c2:
        parts.append(f"**C2 (half 1-1): {p3c2['arm_mean']:.1f}% ({p3c2['mean_diff']:+.1f} pp, "
                     f"{p3c2['n_positive']}/10 up, p={p3c2['perm_p']:.4f}).**")
    if pk:
        parts.append("Peaks: " + " · ".join(
            f"{t} at {v['steps']} steps ({'moved later' if v['moved_later_than_1000'] else 'NOT later'})"
            for t, v in pk.items()) + f", against the full corpus's 1,000.")
    out["verdict"] = " ".join(parts) if parts else "insufficient arms"
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} and {LADOUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
