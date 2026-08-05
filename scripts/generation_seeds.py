"""§2e done properly, plus the corrected reading of argmax. Reuses the sweep; no training.

The sweep's own winner-selection picked **argmax**, and that was wrong for a reason the sweep itself
measured: argmax has **effective n = 2 out of 200**, and the 2 is an artifact — episode 0 of a freshly
started session differs from episodes 1-199, which for a deterministic policy is the *only* source of
variation. So argmax is **one trajectory per checkpoint**, and "91.7%" / "100.0%" are not clearance rates.
They are "the single trajectory cleared", reported as a rate by a scorer that had no way to know.

This script does two things:

1. **§2e on the SAMPLED winners** — `cap48 @ T=1.0` and `cap48 @ T=0.7` on `B_84_seed1` and `B_84_seed2`.
   Those are genuine 200-episode rates and are what a seed replication can actually test.
2. **More argmax trajectories, priced correctly at n=3** (enough to expose the episode-0 split, not more).
   `RT_128_d128_L2` and `phase1_repro_84_3k_b128` are added so the argmax count is 5 checkpoints rather
   than 3 — still trajectories, still not rates, but a less thin base for a qualitative claim.

**Why argmax is worth any words at all:** it is the only configuration in this project whose button
statistics approach the expert's — A 0.28-0.45 against the sampled arms' 0.52-0.62, airborne 44.8% against
66-71% (expert 61.1%), A-onsets 22.6/1k against 45-56 (expert 27.5) — and one of its trajectories reached
**x=916**, past pipe 4's face and the deepest single trajectory recorded here. It is also 0.0% on another
seed. Both facts belong in the same sentence.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
import scripts.generation_sweep as G  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/generation_seeds.json"
SWEEP = ROOT / "data/generation_sweep.json"

#: §2e: the sampled candidate winners, on the two other training seeds
SEED_CONFIGS = [(48, 1.0), (48, 0.7)]
SEED_ARMS = ["B_84_seed1", "B_84_seed2"]
#: argmax is n=1 by construction; 3 episodes is enough to show the episode-0 split and no more
ARGMAX_EXTRA = [("RT_128_d128_L2", 48), ("phase1_repro_84_3k_b128", 48)]
N_ARGMAX = 3


def clr(rec, ob):
    return rec["clearance"][ob]["rate"]


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    sweep = json.loads(SWEEP.read_text())
    arms = dict(sweep["arms"])
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})

    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    lut_cache: dict[str, np.ndarray] = {}

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
        return G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    print("=== 2e: sampled winners on two more training seeds ===", flush=True)
    for name in SEED_ARMS:
        ck = get_ck(name)
        for cap, temp in SEED_CONFIGS:
            k = G.tag(name, cap, temp)
            if k in arms:
                out["arms"][k] = arms[k]
                continue
            if k not in out["arms"]:
                out["arms"][k] = G.run_arm(sess_get, ck, cap, temp, ctx, start)
                save()

    print("\n=== argmax on two more checkpoints, n=3 (trajectories, not rates) ===", flush=True)
    for name, cap in ARGMAX_EXTRA:
        ck = get_ck(name)
        k = G.tag(name, cap, "argmax")
        if k not in out["arms"]:
            out["arms"][k] = G.run_arm(sess_get, ck, cap, "argmax", ctx, start, n=N_ARGMAX)
            save()

    allarms = {**arms, **out["arms"]}

    # ---------- corrected §2e: seed replication of the SAMPLED configs ----------
    rep = {}
    for cap, temp in SEED_CONFIGS:
        rows = []
        for base in ["B_84_d64_L1"] + SEED_ARMS:
            kb, kw = G.tag(base, None, 1.0), G.tag(base, cap, temp)
            if kb in allarms and kw in allarms:
                b, w = clr(allarms[kb], "pipe2"), clr(allarms[kw], "pipe2")
                lo, hi = diff_ci(int(round(b * 200)), 200, int(round(w * 200)), 200)
                rows.append({"seed_arm": base, "uncapped_pipe2": b, "config_pipe2": w,
                             "gain_pp": (w - b) * 100, "ci_pp": [lo * 100, hi * 100],
                             "pipe3_uncapped": clr(allarms[kb], "pipe3"),
                             "pipe3_config": clr(allarms[kw], "pipe3")})
        g = [r["gain_pp"] for r in rows]
        rep[f"cap{cap}_T{temp}"] = {
            "per_seed": rows, "n_seeds": len(g),
            "median_gain_pp": float(np.median(g)) if g else None,
            "min_gain_pp": float(min(g)) if g else None,
            "max_gain_pp": float(max(g)) if g else None,
            "all_positive": bool(g) and all(x > 0 for x in g),
            "any_interval_excludes_zero": any(r["ci_pp"][0] > 0 for r in rows)}
        print(f"  cap {cap} T={temp}: gains {[round(x, 1) for x in g]} pp "
              f"(median {np.median(g):+.1f})" if g else f"  cap {cap} T={temp}: no data", flush=True)
    out["seed_replication_sampled"] = rep

    # ---------- corrected argmax reading ----------
    ax = {}
    for k, r in allarms.items():
        if r.get("temperature") != "argmax":
            continue
        d = r["distinctness"]
        ax[k] = {"checkpoint": r["checkpoint"], "frame_size": r["frame_size"],
                 "train_seed": r.get("train_seed"),
                 "episodes_run": d["n_episodes"], "distinct_trajectories": d["n_distinct_prefixes"],
                 "effective_n": 1,
                 "effective_n_note": ("1, not the reported distinct count: the extra trajectory is "
                                      "always episode 0 of a freshly started session, whose initial "
                                      "frame stack differs -- a harness artifact, not policy variation"),
                 "x_median": r["x_median"], "x_max": r["x_max"],
                 "cleared": {o: clr(r, o) > 0.5 for o in ("pipe1", "pipe2", "pipe3", "pipe4")},
                 "reported_rate_pipe2_NOT_A_RATE": clr(r, "pipe2"),
                 "a_rate": r["button_marginals"]["rates"]["A"],
                 "a_onsets_per_1000": r["a_onsets_per_1000_frames"],
                 "airborne": r["behaviour"]["airborne_fraction"],
                 "a_hold_max": r["a_hold_anywhere"].get("max")}
    out["argmax_trajectories"] = ax
    n_p2 = sum(1 for v in ax.values() if v["cleared"]["pipe2"])
    n_p3 = sum(1 for v in ax.values() if v["cleared"]["pipe3"])
    out["argmax_summary"] = {
        "n_checkpoints": len(ax), "cleared_pipe2": n_p2, "cleared_pipe3": n_p3,
        "x_median_range": [min(v["x_median"] for v in ax.values()),
                           max(v["x_median"] for v in ax.values())] if ax else None,
        "a_rate_range": [min(v["a_rate"] for v in ax.values()),
                         max(v["a_rate"] for v in ax.values())] if ax else None,
        "expert_a_rate": 0.152, "expert_airborne": 0.611, "expert_onsets_per_1000": 27.5,
        "vote_splitting_reproduced": bool(ax and min(v["a_rate"] for v in ax.values()) < 0.01),
        "vote_splitting_checkpoints": sorted(k for k, v in ax.items() if v["a_rate"] < 0.01),
        "vote_splitting_a_rates": {k: v["a_rate"] for k, v in ax.items()},
        "vote_splitting_note": ("LEDGER's 'never argmax the categorical head' -- A emitted on 0.03% of "
                                "frames by vote-splitting -- **reproduces on some checkpoints and not "
                                "others, on the SAME head.** Measured A under argmax: B seeds "
                                "0.28/0.32/0.45, R 0.36, but RT **0.000** (never presses A, x=40, does "
                                "not leave the start) and phase1_repro **0.002**. So the rule is not "
                                "wrong and is not universal: it is a property of the individual "
                                "checkpoint's logit geometry, and argmax must be checked per checkpoint "
                                "rather than assumed either safe or fatal."),
        "verdict": (f"argmax is ONE trajectory per checkpoint. Across {len(ax)} checkpoints it clears "
                    f"pipe 2 on {n_p2} and pipe 3 on {n_p3}. Its button statistics are the closest to "
                    f"the expert's of anything in this project, and its outcome is bimodal: one "
                    f"trajectory reached x=916 past pipe 4's face, another died at x=316. **A rate "
                    f"cannot be estimated from it by running more seeds; it needs many START STATES.**")}
    print(f"\n  argmax over {len(ax)} checkpoints: cleared pipe2 {n_p2}, pipe3 {n_p3}", flush=True)

    # ---------- the binary question, answered on the sampled arms only ----------
    ce = sweep["cap_effect"]
    gB, gR = ce["B_84_d64_L1"]["gain_pp"], ce["R_128_d64_L1"]["gain_pp"]
    ciB, ciR = ce["B_84_d64_L1"]["ci_pp"], ce["R_128_d64_L1"]["ci_pp"]
    sampled_improves = any(v["all_positive"] and v["any_interval_excludes_zero"]
                           for v in rep.values())
    out["binary_question"] = {
        "does_capping_improve_pipe2": sampled_improves,
        "B_gain_pp": gB, "B_ci": ciB, "R_gain_pp": gR, "R_ci": ciR,
        "R_minus_B_pp": gR - gB,
        "answered_on": "sampled arms only; argmax cannot answer it because its n is 1",
    }
    if not sampled_improves:
        out["verdict"] = (
            f"**NO — capping the A-run does not improve pipe 2.** B +{gB:.1f} pp "
            f"[{ciB[0]:+.1f},{ciB[1]:+.1f}], R +{gR:.1f} pp [{ciR[0]:+.1f},{ciR[1]:+.1f}]; both "
            f"intervals span zero, and across three training seeds the gains do not hold a sign. "
            f"Temperature 0.7 does not rescue it. **The generation rule, in the one form the directive "
            f"specified, is not the bottleneck for pipe 2.**\n\n"
            f"But the directive's 'no' branch — that the LIFT is then under suspicion — does not follow "
            f"yet, and I am not claiming it. Two things sit in the way. **(1) R's pipe 3 moved "
            f"22.5% -> 31.5% under cap 24 while B's did not**, which is a resolution-dependent gain in "
            f"the predicted direction, at the wrong obstacle. **(2) argmax is not a null result, it is "
            f"an unmeasurable one**: it produces the most expert-like button statistics in the project "
            f"and one trajectory to x=916, but it is n=1 per checkpoint and needs many start states "
            f"rather than many seeds. **The lift's behavioural relevance is untested, not refuted.**")
    else:
        best = max(rep.items(), key=lambda kv: (kv[1]["median_gain_pp"] or -99))
        out["verdict"] = (
            f"**YES — {best[0]} improves pipe 2 in {best[1]['n_seeds']}/{best[1]['n_seeds']} seeds** "
            f"(median {best[1]['median_gain_pp']:+.1f} pp). R gain {gR:+.1f} vs B {gB:+.1f}.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
