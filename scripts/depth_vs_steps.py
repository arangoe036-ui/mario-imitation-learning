"""§3: where is the depth-versus-steps peak, and is it before 15,000? **The block's binary question.**

Block 57 found training loss halving from 15k to 60k while `x_max` fell 305–551 px. If the peak is well
before 15,000 then **every arm this project has trained is past its own optimum and training length is a
confound in every historical comparison.**

**⚠ DEFECT IN THE PREMISE, found while building.** The directive states the every-250-step banked checkpoints
"already exist". They do not: `scaleup_train` writes `{name}.partial.pt` and **overwrites it at every bank**,
so only the final step survives — `L_84_cnn32_60k.partial.pt` holds step 60,000 and nothing else. The curve
therefore required re-running the identical recipe with `SNAP_STEPS` set to retain intermediate weights.
**Same architecture, same seed, same data, same 60,000 steps — only the intermediate weights are kept**, so
this is not "training longer or wider", which §7 forbids.

**n=100 per rung, not 200**, to fit twelve rungs plus a second seed inside the budget alongside §2. The
quantity of interest is the *shape* of the curve and the location of its peak; at n=100 a rate carries roughly
±10 pp, which is ample to place a peak that block 57 measured as a 300–550 px effect. Stated here because
halving n without saying so is how a comparison quietly stops being comparable to the n=200 figures elsewhere.

Terminator from `tasdata.bc.rollout_budget` — identical across every rung, which is the one thing that must
not vary.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.button_mask_eval import MASK_BITS, rollout  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.scaleup_eval import resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/depth_vs_steps.json"
CKDIR = ROOT / "data/bc_scaleup"
TRACED = ROOT / "data/traces"

RUNGS = [500, 1000, 2000, 3000, 5000, 8000, 12000, 15000, 22000, 30000, 45000, 60000]
#: rungs re-run on a second seed once the peak is known; a peak on one seed is a screen
BRACKET_N = 4
SEED0, SEED1 = "CURVE_84_cnn32", "CURVE_84_cnn32_seed1"
TEMP = 0.7
N_EVAL = 100
WALLS = {"pipe3_735": 735, "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562,
         "flagpole_3266": 3266}
ARM_BUDGET_S = 12 * 60


def load_snap(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig.from_dict(cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()
    return policy, cfg, blob


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 180 * 60)
    use_mask = os.environ.get("USE_MASK", "0") == "1"
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)
    class_ok = (np.array([0.0 if (int(byte_of[c]) & MASK_BITS) else 1.0 for c in range(n_cls)])
                if use_mask else None)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("rungs", {})
    out.setdefault("skipped", [])
    out.update({
        "question": "where is the depth-vs-steps peak, and is it before 15,000?",
        "n_eval": N_EVAL,
        "n_eval_note": ("100, not 200, to fit 12 rungs plus a second seed in budget; the target is the "
                        "SHAPE of the curve, and these figures are therefore not directly comparable to "
                        "the project's n=200 numbers"),
        "measurement_basis": "single_life_from_level_start",
        "terminator": RB.describe(), "temperature": TEMP, "rungs_planned": RUNGS,
        "button_mask_applied": bool(use_mask),
        "premise_defect": ("the directive assumed every-250-step checkpoints existed; partials are "
                           "OVERWRITTEN each bank, so the identical recipe was re-run with SNAP_STEPS "
                           "to retain intermediate weights"),
        "walls": WALLS})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    def eval_rung(arm: str, step: int):
        key = f"{arm}@{step}"
        if key in out["rungs"]:
            return
        snap = CKDIR / f"{arm}.snap{step}.pt"
        if not snap.exists():
            return
        if not dl.can_afford(120):
            out["skipped"].append({"rung": key, "reason": "deadline"})
            return
        policy, cfg, blob = load_snap(snap)
        tp = TRACED / f"curve_{arm}_s{step}_{N_EVAL}.json"
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                s = sess_get()
                try:
                    traces = resumable(tp, N_EVAL,
                                       lambda i: rollout(s, policy, cfg, start, i, lut, byte_of,
                                                         class_ok, temp=TEMP))
                finally:
                    s.close()
        except TimedOut as e:
            out["skipped"].append({"rung": key, "reason": str(e)})
            save()
            return
        rec = score(key, traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        rec.update({
            "arm": arm, "steps": step, "train_seed": blob.get("seed"),
            "loss_at_snapshot": blob.get("loss_at_snapshot"),
            "x_p90": float(np.percentile(xs, 90)),
            "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                              "rate": float(np.mean([x > v for x in xs]))}
                          for w, v in WALLS.items()},
            "terminator": RB.describe(), "traces": str(tp.relative_to(ROOT))})
        out["rungs"][key] = rec
        save()
        pw = rec["past_wall"]
        print(f"  {dl.stamp()} {key:28s} loss {rec['loss_at_snapshot'] or float('nan'):.3f} "
              f"x_max {rec['x_max']:5d} x_med {rec['x_median']:5.0f} "
              f"p3 {pw['pipe3_735']['rate']*100:5.1f}% p4 {pw['pipe4_975']['rate']*100:4.1f}% "
              f"pipe2 {rec['clearance']['pipe2']['rate']*100:5.1f}%", flush=True)

    print(f"{dl.stamp()} seed-0 ladder, mask={'on' if use_mask else 'off'}", flush=True)
    for st in RUNGS:
        eval_rung(SEED0, st)

    # ---- locate the peak on seed 0, then bracket it on seed 1 ----
    def series(arm):
        rows = [v for v in out["rungs"].values() if v["arm"] == arm]
        return sorted(rows, key=lambda r: r["steps"])
    s0 = series(SEED0)
    peak_step = None
    if s0:
        # peak on x_median, which is far less noisy at n=100 than x_max
        peak = max(s0, key=lambda r: r["x_median"])
        peak_step = peak["steps"]
        order = [r["steps"] for r in s0]
        i = order.index(peak_step)
        bracket = order[max(0, i - 2):i + 3][:BRACKET_N + 1]
        print(f"\n{dl.stamp()} seed-0 peak at {peak_step} steps (x_median "
              f"{peak['x_median']:.0f}); bracketing on seed 1: {bracket}", flush=True)
        for st in bracket:
            eval_rung(SEED1, st)

    # ---- analysis ----
    def table(arm):
        return [{"steps": r["steps"], "loss": r["loss_at_snapshot"],
                 "x_max": r["x_max"], "x_median": r["x_median"], "x_p90": r["x_p90"],
                 "pipe2": r["clearance"]["pipe2"]["rate"] * 100,
                 "past_pipe3": r["past_wall"]["pipe3_735"]["rate"] * 100,
                 "past_pipe4": r["past_wall"]["pipe4_975"]["rate"] * 100}
                for r in series(arm)]
    out["curve_seed0"] = table(SEED0)
    out["curve_seed1"] = table(SEED1)

    c0 = out["curve_seed0"]
    if c0:
        best = max(c0, key=lambda r: r["x_median"])
        first = c0[0]
        monotone_down = all(c0[i]["x_median"] >= c0[i + 1]["x_median"] - 1e-9
                            for i in range(len(c0) - 1))
        c1 = out["curve_seed1"]
        best1 = max(c1, key=lambda r: r["x_median"]) if c1 else None
        out["peak"] = {
            "seed0_peak_steps": best["steps"], "seed0_peak_x_median": best["x_median"],
            "seed1_peak_steps": (best1["steps"] if best1 else None),
            "seed1_peak_x_median": (best1["x_median"] if best1 else None),
            "seed1_rungs_evaluated": [r["steps"] for r in c1],
            "monotone_decline_from_first_rung": bool(monotone_down),
            "loss_at_peak": best["loss"], "loss_at_60k": c0[-1]["loss"],
            "x_median_at_peak": best["x_median"], "x_median_at_60k": c0[-1]["x_median"]}
        agree = (best1 is not None and abs(best1["steps"] - best["steps"]) <= 0
                 or (best1 is not None and best1["steps"] in
                     [r["steps"] for r in c1] and best1["steps"] == best["steps"]))
        if monotone_down:
            out["branch"] = "monotone_decline"
            out["verdict"] = (
                f"**MONOTONE DECLINE FROM THE FIRST RUNG.** x_median falls from {first['x_median']:.0f} "
                f"at {first['steps']} steps to {c0[-1]['x_median']:.0f} at 60,000, with no interior "
                f"peak, while loss falls from {first['loss']:.3f} to {c0[-1]['loss']:.3f}. **The corpus "
                f"or the objective is wrong from step one** — no amount of scaling, sharpening or "
                f"distillation was ever going to work on this signal. This is the most consequential of "
                f"the three branches and should be written up as such.")
        elif best["steps"] < 15000:
            out["branch"] = "peak_before_15k"
            out["verdict"] = (
                f"**THE PEAK IS BEFORE 15,000: {best['steps']} steps** (x_median "
                f"{best['x_median']:.0f} against {c0[-1]['x_median']:.0f} at 60k), seed 1 peaking at "
                f"{out['peak']['seed1_peak_steps']}. **Every arm this project has trained is past its "
                f"own optimum, and training length is a confound in every historical comparison.** The "
                f"cheapest available improvement is to stop earlier. Loss at the peak "
                f"{best['loss']:.3f} against {c0[-1]['loss']:.3f} at 60k — the objective keeps "
                f"improving well past the point where behaviour stops.")
        else:
            out["branch"] = "peak_at_or_near_15k"
            out["verdict"] = (
                f"**THE PEAK IS AT OR NEAR 15,000: {best['steps']} steps** (x_median "
                f"{best['x_median']:.0f}), seed 1 peaking at {out['peak']['seed1_peak_steps']}. P was "
                f"accidentally right and the ceiling is genuine, so the objective itself is the target "
                f"— early stopping on a behavioural criterion, or reweighting.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out.get("verdict", "no rungs evaluated"))
    print(f"\n{'steps':>7s}{'loss':>8s}{'x_med':>7s}{'x_max':>7s}{'pipe2':>7s}{'>p3':>6s}{'>p4':>6s}")
    for r in out["curve_seed0"]:
        print(f"{r['steps']:>7d}{(r['loss'] or float('nan')):>8.3f}{r['x_median']:>7.0f}"
              f"{r['x_max']:>7d}{r['pipe2']:>7.1f}{r['past_pipe3']:>6.1f}{r['past_pipe4']:>6.1f}")
    if out["curve_seed1"]:
        print("  -- seed 1 bracket --")
        for r in out["curve_seed1"]:
            print(f"{r['steps']:>7d}{(r['loss'] or float('nan')):>8.3f}{r['x_median']:>7.0f}"
                  f"{r['x_max']:>7d}{r['pipe2']:>7.1f}{r['past_pipe3']:>6.1f}{r['past_pipe4']:>6.1f}")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
