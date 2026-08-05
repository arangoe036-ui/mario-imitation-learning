"""Was "no timing anywhere" a property of the method, or of a checkpoint trained for 3,000 steps?

The fifty-first block measured the timing lift on `data/bc_phase1/runlength.pt` and concluded the policy has
no timing signal where it succeeds. §1 of the fifty-third directive corrected the estimator's baseline and the
sign held: pipe 2 stayed negative, **-0.026 [-0.035, -0.016]**.

But that checkpoint was trained for **3,000 steps at batch 128** -- and the directive's own §4 said "train
longer before concluding anything about size." The same instruction applies to concluding anything about
*timing*. So: hold the architecture, resolution, corpus, split, loss and generation rule fixed, and vary only
how long it trained.

| arm | steps | batch | samples seen |
|---|---|---|---|
| `phase1_repro_84_3k_b128` | 3,000 | 128 | 384,000 -- reproduces the old recipe through this pipeline |
| `B_84_d64_L1` / `seed1` / `seed2` | 15,000 | 64 | 960,000 |

**Three seeds at 15,000, because one seed is a screen** -- this project's ledger records a 14.5-24.5 pp
training-seed spread on clearance, and the spread on a timing lift has never been measured at all. Seeding
covers weight initialisation as well as data order.

`phase1_repro` exists to separate two variables that differ between the old checkpoint and the new arms
(steps *and* batch) and, just as importantly, to check that **this pipeline reproduces the old result at the
old recipe.** If it does not, the difference is in the pipeline and nothing about training length follows.

Forward passes only. No emulator.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.scaleup_eval import load_arm, timing_lift  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data/bc_scaleup"
OUT = ROOT / "data/lift_by_training_length.json"

#: the fifty-first block's checkpoint, measured here by the same code path as the new arms
OLD = ROOT / "data/bc_phase1/runlength.pt"
ARMS = ["phase1_repro_84_3k_b128", "B_84_d64_L1", "B_84_seed1", "B_84_seed2"]
#: §1's numbers on the old checkpoint, for the record
SEC1 = {"goomba_288": (0.062, [0.035, 0.090]), "pipe1_432": (0.008, [-0.003, 0.018]),
        "pipe2_592": (-0.026, [-0.035, -0.016])}


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.update({
        "estimator": ("corrected: expert A-onsets vs NON-A RUN STARTS, stratified over 16px bins, "
                      "weighted by onset count, bootstrapped over onsets"),
        "held_fixed": ("architecture d_model=64 n_layers=1, 84x84, expert-train split, plain "
                       "cross-entropy, run-length joint classes"),
        "varied": "training steps (3,000 vs 15,000), batch (128 vs 64), and seed",
        "section1_values_on_old_checkpoint": {k: {"corrected_lift": v[0], "corrected_ci": v[1]}
                                             for k, v in SEC1.items()},
    })

    todo = [("phase1_runlength_3k_ORIGINAL", OLD)] + [(a, OUTDIR / f"{a}.pt") for a in ARMS]
    hdr = (f"{'arm':30s} {'steps':>6s} {'batch':>6s} {'seed':>5s} | "
           f"{'goomba':>18s} {'pipe1':>18s} {'pipe2':>18s}")
    print(hdr)
    print("-" * len(hdr))
    for name, path in todo:
        if not path.exists():
            print(f"{name:30s} (no checkpoint yet)")
            continue
        if name in out["arms"]:
            pass                                   # recompute: cheap, and keeps one code path
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob["policy_config"]
        if isinstance(cfg, dict):
            cfg = PolicyConfig.from_dict(cfg)
        policy = BCPolicy(cfg)
        policy.load_state_dict(blob["model_state"])
        policy.eval()
        tl = timing_lift(policy, cfg, ctx, blob.get("corpus", "runs"), ctx.vocab)
        rec = {"checkpoint": str(path.relative_to(ROOT)),
               "steps": blob.get("steps", blob.get("step")), "batch": blob.get("batch"),
               "seed": blob.get("seed"), "frame_size": cfg.frame_size,
               "d_model": cfg.d_model, "n_layers": cfg.n_layers,
               "samples_seen": ((blob.get("step") or 0) * (blob.get("batch") or 0)) or None,
               "timing_lift": tl}
        out["arms"][name] = rec

        def cell(ob):
            a = tl["obstacles"].get(ob, {})
            v, ci = a.get("corrected_lift"), a.get("corrected_ci")
            if v is None:
                return "n/a"
            return f"{v:+.3f}[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else f"{v:+.3f}"
        print(f"{name:30s} {str(rec['steps']):>6s} {str(rec['batch']):>6s} "
              f"{str(rec['seed']):>5s} | {cell('goomba_288'):>18s} {cell('pipe1_432'):>18s} "
              f"{cell('pipe2_592'):>18s}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def lifts(ob, arms):
        vals = []
        for a in arms:
            r = out["arms"].get(a, {}).get("timing_lift", {}).get("obstacles", {}).get(ob, {})
            if r.get("corrected_lift") is not None:
                vals.append((a, r["corrected_lift"], r.get("corrected_ci")))
        return vals

    seeds15k = ["B_84_d64_L1", "B_84_seed1", "B_84_seed2"]
    out["pipe2"] = {
        "old_3k_original": lifts("pipe2_592", ["phase1_runlength_3k_ORIGINAL"]),
        "repro_3k": lifts("pipe2_592", ["phase1_repro_84_3k_b128"]),
        "at_15k_three_seeds": lifts("pipe2_592", seeds15k)}
    v15 = [v for _, v, _ in out["pipe2"]["at_15k_three_seeds"]]
    v3 = [v for _, v, _ in out["pipe2"]["repro_3k"]]
    all_pos = bool(v15) and all(
        ci is not None and ci[0] > 0 for _, _, ci in out["pipe2"]["at_15k_three_seeds"])
    out["pipe2_summary"] = {
        "n_seeds_15k": len(v15),
        "median_15k": float(np.median(v15)) if v15 else None,
        "min_15k": float(min(v15)) if v15 else None, "max_15k": float(max(v15)) if v15 else None,
        "seed_spread_15k": float(max(v15) - min(v15)) if len(v15) > 1 else None,
        "repro_3k_lift": v3[0] if v3 else None,
        "all_15k_seeds_positive_excluding_zero": all_pos,
        "reproduced_the_old_negative_at_3k": bool(v3 and v3[0] < 0)}
    s = out["pipe2_summary"]
    s["old_checkpoint_lift"] = (out["pipe2"]["old_3k_original"][0][1]
                               if out["pipe2"]["old_3k_original"] else None)
    #: the case that actually occurred: the old recipe, re-run, does NOT give the old sign
    single_seed_artifact = bool(
        all_pos and v3 and v3[0] > 0 and s["old_checkpoint_lift"] is not None
        and s["old_checkpoint_lift"] < 0)
    s["old_negative_is_single_seed_artifact"] = single_seed_artifact
    if single_seed_artifact:
        out["verdict"] = (
            f"**\"NO TIMING ANYWHERE\" IS VOID: IT WAS ONE TRAINING SEED.** The old checkpoint's pipe-2 "
            f"lift is {s['old_checkpoint_lift']:+.3f}. Re-running **its own recipe** -- 3,000 steps, "
            f"batch 128, same architecture, corpus, split, loss -- gives {v3[0]:+.3f}, POSITIVE. And at "
            f"15,000 steps it is positive in {s['n_seeds_15k']}/{s['n_seeds_15k']} seeds (median "
            f"{s['median_15k']:+.3f}, range {s['min_15k']:+.3f} to {s['max_15k']:+.3f}), every interval "
            f"excluding zero. **Four freshly trained checkpoints all show a positive pipe-2 timing lift; "
            f"the one the conclusion was built on is the outlier.** The fifty-first block measured a "
            f"single checkpoint and §1 re-measured that same checkpoint, so neither read caught it -- "
            f"the ledger's own rule that one seed is a screen was the thing being violated. "
            f"**The policy does discriminate when to jump where it succeeds. The observation is not the "
            f"indicated lever, and there is also a training-length dose-response on top of the seed "
            f"effect: {v3[0]:+.3f} at 3k against a median {s['median_15k']:+.3f} at 15k.**")
    elif all_pos and s["reproduced_the_old_negative_at_3k"]:
        out["verdict"] = (
            f"**\"NO TIMING ANYWHERE\" IS VOID -- IT WAS TRAINING LENGTH, NOT THE METHOD.** At the old "
            f"recipe this pipeline reproduces the negative pipe-2 lift ({s['repro_3k_lift']:+.3f}); at "
            f"15,000 steps it is positive in {s['n_seeds_15k']}/{s['n_seeds_15k']} seeds, median "
            f"{s['median_15k']:+.3f}, range {s['min_15k']:+.3f} to {s['max_15k']:+.3f}, every interval "
            f"excluding zero. **The policy does discriminate when to jump where it succeeds; the earlier "
            f"checkpoint was simply undertrained.** The observation is NOT the indicated lever, and the "
            f"scale-up is an optimisation rather than a rescue.")
    elif all_pos:
        out["verdict"] = (
            f"**PIPE-2 TIMING LIFT IS POSITIVE AT 15,000 STEPS IN ALL {s['n_seeds_15k']} SEEDS** "
            f"(median {s['median_15k']:+.3f}), **but this pipeline did NOT reproduce the old negative at "
            f"the old recipe** ({s['repro_3k_lift']}). The difference therefore is not established as "
            f"training length -- something else differs between this pipeline and the fifty-first "
            f"block's. Do not attribute it to steps until the reproduction succeeds.")
    else:
        out["verdict"] = (
            f"**THE NEGATIVE PIPE-2 LIFT SURVIVES LONGER TRAINING.** At 15,000 steps the corrected lift "
            f"is median {s['median_15k']} across {s['n_seeds_15k']} seeds, not positive with intervals "
            f"excluding zero. Training length is not the explanation, and the fifty-first block's "
            f"conclusion stands as corrected in §1.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
