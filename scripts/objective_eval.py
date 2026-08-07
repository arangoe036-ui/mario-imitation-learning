"""§1: evaluate the objective sweeps against the reused eps=0 / 1.0x baseline.

**PRE-SPECIFIED PRIMARY OUTCOME — written into the artifact before any arm is rolled out: clearance past
pipe 4 (x > 975).** Chosen by the directive because that is where the last two blocks' harm landed and where
the run-length script bar sits closest. Everything else in the table is secondary.

**Bonferroni family = 6 walls x n_arms.** Pipe 1 (470) and pipe 2 (630) have had identical clearance in every
arm this project has ever run — no episode has failed between them — so they are one measurement, not two.

Ten paired seeds per cell; exact paired sign-flip permutation over seeds; **floor 2/2^10 = 0.00195, printed
beside every p.** Baseline = `PK32_84_s0..9`, traces reused so the comparison is against the same weights the
last three blocks were measured against, and re-scored by this file so no delta is a scoring difference.
"""
from __future__ import annotations

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
TRACED = ROOT / "data/traces"
CKDIR = ROOT / "data/bc_scaleup"
LS_OUT = ROOT / "data/label_smoothing_sweep.json"
LAD_OUT = ROOT / "data/label_smoothing_ladder.json"
WR_OUT = ROOT / "data/window_reweight_sweep.json"

N_SEEDS, N_EVAL, N_LAD, TEMP = 10, 200, 100, 0.7
RUNGS = [500, 1000, 2000, 5000, 15000]
PRIMARY = "pipe4_975"
#: six, not seven: pipe1 and pipe2 are the same measurement
WALLS = {"goomba_320": 320, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562}
BASE = [f"PK32_84_s{i}" for i in range(N_SEEDS)]
LS_CELLS = {"eps0.00_baseline": BASE,
            "eps0.05": [f"LS005_s{i}" for i in range(N_SEEDS)],
            "eps0.10": [f"LS010_s{i}" for i in range(N_SEEDS)],
            "eps0.20": [f"LS020_s{i}" for i in range(N_SEEDS)]}
WR_CELLS = {"x1.0_baseline": BASE,
            "x1.5": [f"WR015_s{i}" for i in range(N_SEEDS)],
            "x2.0": [f"WR020_s{i}" for i in range(N_SEEDS)],
            "x3.0": [f"WR030_s{i}" for i in range(N_SEEDS)],
            "x8.0": [f"WR080_s{i}" for i in range(N_SEEDS)]}
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


class Runner:
    def __init__(self, dl):
        self.dl = dl
        self.ctx = O.Ctx()
        self.start = next(p for p in self.ctx.points
                          if p.kind == "level_start" and p.label == "1-1")
        n_cls = G.joint_size(self.ctx.vocab.size)
        self.byte_of = np.array([self.ctx.vocab.decode_byte(c // G.N_BUCKETS)
                                 for c in range(n_cls)], dtype=np.int64)
        z = np.load(ROOT / "data/runlength_index_runs.npz")
        self.lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    def session(self):
        s = G.session_when_free(O.ROM, O.MOVIE, self.ctx.frames_needed())
        warm_session(s, self.start.frame)
        return s

    def arm(self, name, n, tp, is_base, key):
        """Returns (record, error). Baseline traces are reused; everything else is rolled out."""
        if is_base:
            c = next((p for p in (TRACED / f"jb_{name}_unbiased_{N_EVAL}.json",
                                  TRACED / f"nh_{name}_{N_EVAL}.json") if p.exists()), None)
            if c is not None:
                traces = [_Ep(e) for e in json.loads(c.read_text())["episodes"]]
                return self.rec(key, name, traces, f"reused: {c.name}"), None
        if not (CKDIR / f"{name}.pt").exists():
            return None, "no checkpoint"
        if not self.dl.can_afford(150):
            return None, "deadline"
        policy, cfg, _ = load(name)
        try:
            with time_limit(min(ARM_BUDGET_S, self.dl.remaining() - 60), key):
                s = self.session()
                try:
                    traces = resumable(tp, n, lambda i: rollout(s, policy, cfg, self.start, i,
                                                                self.lut, self.byte_of, None,
                                                                temp=TEMP))
                finally:
                    s.close()
        except TimedOut as e:
            return None, str(e)
        return self.rec(key, name, traces, "block 66"), None

    def rec(self, key, name, traces, src):
        blob = torch.load(CKDIR / f"{name}.pt", map_location="cpu", weights_only=False)
        r = score(key, traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        fk = failure_kinds(traces)
        lh = blob.get("loss_history")
        r.update({
            "checkpoint": name, "source": src, "n": len(traces), "terminator": RB.describe(),
            "label_smoothing": blob.get("label_smoothing", 0.0),
            "window_strength": blob.get("window_strength", 1.0),
            "final_loss": blob.get("final_loss") or blob.get("loss_at_snapshot")
            or (lh[-1][1] if lh else None),
            "final_nll": blob.get("final_nll") or blob.get("plain_nll_at_snapshot")
            or (lh[-1][2] if lh and len(lh[-1]) > 2 else (lh[-1][1] if lh else None)),
            "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                              "rate": float(np.mean([x > v for x in xs]))}
                          for w, v in WALLS.items()},
            "failure_kinds": fk,
            "on_top_pipe4": int(fk.get("pipe4_912", {}).get("on_top", 0)),
            "at_face_pipe4": int(fk.get("pipe4_912", {}).get("at_face", 0)),
            "on_top_total": int(sum(v.get("on_top", 0) for v in fk.values())),
            "completions": int(sum(1 for t in traces
                                   if any(len(f) > 7 and f[6] == 1 and f[7] == 2 for f in t.frames))),
        })
        return r


def analyse(out, cells, base_key):
    def vals(cell, field, wall=None):
        v = []
        for n in cells[cell]:
            r = out["arms"].get(f"{cell}/{n}")
            if r:
                v.append(r["past_wall"][wall]["rate"] * 100 if wall else r[field])
        return v

    fam = len(WALLS) * (len(cells) - 1)
    res = {}
    for cell in cells:
        if cell == base_key:
            continue
        row = {}
        for w in WALLS:
            a, b = vals(cell, None, w), vals(base_key, None, w)
            if len(a) == len(b) and len(a) >= 3:
                d = [x - y for x, y in zip(a, b)]
                p, floor = sign_flip_p(d)
                row[w] = {"arm_mean": float(np.mean(a)), "baseline_mean": float(np.mean(b)),
                          "arm": a, "baseline": b, "mean_diff": float(np.mean(d)),
                          "perm_p": p, "floor": floor, "bonferroni_p": min(1.0, p * fam),
                          "n_positive": int(sum(1 for x in d if x > 0)),
                          "is_primary": w == PRIMARY}
        for f in ("x_median", "x_max", "on_top_pipe4", "at_face_pipe4", "on_top_total",
                  "completions", "final_nll"):
            a, b = vals(cell, f), vals(base_key, f)
            if len(a) == len(b) and len(a) >= 3 and all(x is not None for x in a + b):
                d = [x - y for x, y in zip(a, b)]
                p, _ = sign_flip_p(d)
                row[f] = {"arm_mean": float(np.mean(a)), "baseline_mean": float(np.mean(b)),
                          "arm": a, "mean_diff": float(np.mean(d)), "perm_p": p,
                          "n_positive": int(sum(1 for x in d if x > 0))}
        res[cell] = row
    out["analysis"] = res
    out["bonferroni_family"] = {"n_tests": fam, "definition": "6 walls x n non-baseline arms",
                                "walls": list(WALLS)}
    out["survivors"] = [{"cell": c, "wall": w, "diff": r[w]["mean_diff"], "p": r[w]["perm_p"],
                         "bonferroni_p": r[w]["bonferroni_p"]}
                        for c, r in res.items() for w in WALLS
                        if w in r and r[w]["bonferroni_p"] < 0.05]
    prim = [{"cell": c, "diff_pp": r[PRIMARY]["mean_diff"], "seeds_up": r[PRIMARY]["n_positive"],
             "p": r[PRIMARY]["perm_p"], "floor": r[PRIMARY]["floor"],
             "arm_mean": r[PRIMARY]["arm_mean"], "baseline_mean": r[PRIMARY]["baseline_mean"]}
            for c, r in res.items() if PRIMARY in r]
    out["PRIMARY_RESULT"] = {"outcome": f"clearance past {PRIMARY}", "pre_specified": True,
                             "cells": prim}
    return out


def run_cells(run, cells, base_key, path, extra):
    out = json.loads(path.read_text()) if path.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("skipped", [])
    out.update({"n_seeds": N_SEEDS, "n_eval": N_EVAL, "temperature": TEMP,
                "terminator": RB.describe(), "cells": cells,
                "PRE_SPECIFIED_PRIMARY": f"clearance past {PRIMARY} (x > 975)",
                "measurement_basis": "single_life_from_level_start",
                "test": "exact paired sign-flip permutation over seeds; floor 2/2^10 = 0.00195",
                **extra})
    path.write_text(json.dumps(out, indent=2, default=str))
    for cell, names in cells.items():
        for name in names:
            key = f"{cell}/{name}"
            if key in out["arms"]:
                continue
            rec, err = run.arm(name, N_EVAL, TRACED / f"obj_{name}_{N_EVAL}.json",
                               cell == base_key, key)
            if rec is None:
                out["skipped"].append({"arm": key, "reason": err})
                path.write_text(json.dumps(out, indent=2, default=str))
                continue
            rec["cell"] = cell
            out["arms"][key] = rec
            path.write_text(json.dumps(out, indent=2, default=str))
            print(f"  {run.dl.stamp()} {key:28s} nll {(rec['final_nll'] or float('nan')):.3f} "
                  f"p2 {rec['past_wall']['pipe2_630']['rate']*100:5.1f}% "
                  f"p3 {rec['past_wall']['pipe3_735']['rate']*100:5.1f}% "
                  f"**p4 {rec['past_wall']['pipe4_975']['rate']*100:5.1f}%** "
                  f"x_med {rec['x_median']:4.0f} comp {rec['completions']}", flush=True)
    out = analyse(out, cells, base_key)
    path.write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 420 * 60)
    run = Runner(dl)
    fw = json.loads((ROOT / "data/failure_windows.json").read_text())

    print("=== 1a. label smoothing ===", flush=True)
    ls = run_cells(run, LS_CELLS, "eps0.00_baseline", LS_OUT, {
        "intervention": "cross_entropy(..., label_smoothing=eps); nothing else changed",
        "prediction_before_running": ("if over-commitment to a flawless trajectory is the mechanism, the "
                                      "PEAK MOVES LATER and the collapse is slower. Failed twice already "
                                      "(augmented data, restricted corpus); a third failure retires it."),
        "loss_note": "`final_loss` is the smoothed objective and is NOT comparable across eps; "
                     "`final_nll` is plain cross-entropy on the same batches and IS."})

    print("\n=== 1b. obstacle-window reweighting ===", flush=True)
    wr = run_cells(run, WR_CELLS, "x1.0_baseline", WR_OUT, {
        "intervention": "in-window rows repeated `strength` times inside the permuted index",
        "windows": fw["windows"], "windows_cover_share_of_failures": fw["failures_covered"],
        "DESIGN_LIMIT": fw["design_check"],
        "reachable_effect": fw["REACHABLE_EFFECT"],
        "strength_note": ("1.5/2.0/3.0 are the directive's mild ladder. 8.0 is added so that a flat sweep "
                          "reads as 'reweighting does not work' rather than 'the manipulation was too "
                          "small to see' — the mild rungs are retained, so this is not an aggressive "
                          "strength run alone.")})

    # ---------------- 1a ladder on the best two epsilons by the PRIMARY outcome ----------------
    lad = json.loads(LAD_OUT.read_text()) if LAD_OUT.exists() else {}
    lad.setdefault("rungs", {})
    prim = sorted(ls["PRIMARY_RESULT"]["cells"], key=lambda c: -c["diff_pp"])
    best = [c["cell"] for c in prim[:2]]
    tags = {"eps0.05": ("LS005", 0.05), "eps0.10": ("LS010", 0.10), "eps0.20": ("LS020", 0.20)}
    lad.update({"rungs_planned": RUNGS, "n_eval": N_LAD, "seed": 0,
                "selected_by": f"the two epsilons with the highest {PRIMARY} delta in the sweep",
                "selected": best, "terminator": RB.describe(),
                "full_corpus_reference": json.loads((ROOT / "data/full_corpus_curve_ref.json").read_text()),
                "prediction_under_test": "label smoothing moves the PEAK LATER than 1,000 steps",
                "power_note": ("one training seed at n=100 per rung, and the full-corpus reference rungs "
                               "above 3,000 steps are also one seed. No single rung-to-rung difference is "
                               "a powered test; the powered comparison is the 10-paired-seed sweep.")})
    LAD_OUT.write_text(json.dumps(lad, indent=2, default=str))
    for cell in best:
        tag, eps = tags[cell]
        for st in RUNGS:
            nm = f"{tag}LAD_s0_{st}"
            key = f"{cell}@{st}"
            if key in lad["rungs"] or not (CKDIR / f"{nm}.pt").exists():
                continue
            rec, err = run.arm(nm, N_LAD, TRACED / f"objlad_{nm}_{N_LAD}.json", False, key)
            if rec is None:
                lad.setdefault("skipped", []).append({"rung": key, "reason": err})
                LAD_OUT.write_text(json.dumps(lad, indent=2, default=str))
                continue
            rec.update({"cell": cell, "eps": eps, "steps": st})
            lad["rungs"][key] = rec
            LAD_OUT.write_text(json.dumps(lad, indent=2, default=str))
            print(f"  {run.dl.stamp()} ladder {key:14s} nll {(rec['final_nll'] or float('nan')):.3f} "
                  f"x_med {rec['x_median']:4.0f} "
                  f"p3 {rec['past_wall']['pipe3_735']['rate']*100:5.1f}% "
                  f"**p4 {rec['past_wall']['pipe4_975']['rate']*100:5.1f}%**", flush=True)
    for cell in best:
        rr = [lad["rungs"][f"{cell}@{s}"] for s in RUNGS if f"{cell}@{s}" in lad["rungs"]]
        if rr:
            b = max(rr, key=lambda r: r["past_wall"][PRIMARY]["rate"])
            lad.setdefault("peak", {})[cell] = {
                "steps": b["steps"], "past_pipe4": b["past_wall"][PRIMARY]["rate"] * 100,
                "MOVED_LATER_THAN_1000": bool(b["steps"] > 1000),
                "curve": [{"steps": r["steps"], "loss": r["final_loss"], "nll": r["final_nll"],
                           "x_median": r["x_median"],
                           "past_pipe4": r["past_wall"][PRIMARY]["rate"] * 100,
                           "past_pipe3": r["past_wall"]["pipe3_735"]["rate"] * 100} for r in rr]}
    LAD_OUT.write_text(json.dumps(lad, indent=2, default=str))

    print("\n" + "=" * 78)
    print("PRIMARY (pre-specified: past pipe 4), label smoothing:")
    for c in ls["PRIMARY_RESULT"]["cells"]:
        print(f"  {c['cell']:20s} {c['baseline_mean']:5.1f} -> {c['arm_mean']:5.1f} "
              f"({c['diff_pp']:+5.1f} pp, {c['seeds_up']}/10 up, p={c['p']:.4f}, floor {c['floor']:.4f})")
    print("PRIMARY, window reweighting:")
    for c in wr["PRIMARY_RESULT"]["cells"]:
        print(f"  {c['cell']:20s} {c['baseline_mean']:5.1f} -> {c['arm_mean']:5.1f} "
              f"({c['diff_pp']:+5.1f} pp, {c['seeds_up']}/10 up, p={c['p']:.4f}, floor {c['floor']:.4f})")
    print(f"\nBonferroni survivors — LS {ls['survivors']} · WR {wr['survivors']}")
    print(f"Peak: {json.dumps(lad.get('peak', {}).get(best[0], {}).get('MOVED_LATER_THAN_1000'))} "
          f"for {best[0] if best else '-'}")
    print(f"\nwrote {LS_OUT}, {WR_OUT}, {LAD_OUT} ({dl.elapsed()/60:.1f} min)")


if __name__ == "__main__":
    main()
