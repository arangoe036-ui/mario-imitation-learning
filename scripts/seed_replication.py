"""P1: how much of the filter comparison is training-seed noise?

Every composition and filter result in this project is **one training run per configuration**.
Round 1 swings 28.0 -> 43.5 -> 51.5 across three filters (23.5 pp), while the claim that top-20%
beats survival-gating rests on 62.0 vs 60.0 (2 pp). Seed variance has never been measured.

Three seeds of the round1->round2 sequence for each filter, from the shared deterministic base
checkpoint. The seed drives both the rollout RNG and the training shuffle.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset, Subset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import EARLIEST, ROUND_STEPS, ROLLOUTS, evaluate, session_when_free, train
from scripts.compose_survival import rollout_round_survival
from scripts.overnight import write_self_run
from tasdata.bc.overnight_lib import diff_ci, load_policy, random_rows, save_policy, wilson
from tasdata.bc.session_player import play_episode
from tasdata.buttons import NES_BUTTON_ORDER
from tasdata.dataset import load_run_dir

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/seed_replication.json"
CKPTS = ROOT / "data/bc_seeds"; CKPTS.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "data/bc_compose/compose_base.pt"
MAX_FRAMES, MIN_PROGRESS = 500, 120

def rollout_plain(ctx, session, policy, cfg, thr, seed, frac):
    """Top-`frac` by progress, no survival test."""
    rng = np.random.default_rng(seed)
    picks = [ctx.traj[i] for i in rng.choice(len(ctx.traj), size=ROLLOUTS, replace=False)]
    scored = []
    for k, st in enumerate(picks):
        try:
            ep = play_episode(session, policy, st, ctx.vocab, seed=seed * 10_000 + k,
                              selection="sample", thresholds=thr, head_type=cfg.head_type,
                              stack=cfg.stack, max_frames=MAX_FRAMES)
        except Exception:
            continue
        g = ep.max_x_by_level.get(st.label, st.x) - st.x
        scored.append({"score": g + 4000 * max(0, ep.levels_reached - 1), "k": k,
                       "start": st, "deaths": ep.deaths})
    sc = np.array([s["score"] for s in scored], float)
    cutoff = max(float(np.quantile(sc, 1 - frac)), MIN_PROGRESS)
    acc = [s for s in scored if s["score"] >= cutoff]
    frames, bytes_ = [], []
    for s in acc:
        rec = []
        try:
            play_episode(session, policy, s["start"], ctx.vocab, seed=seed * 10_000 + s["k"],
                         selection="sample", thresholds=thr, head_type=cfg.head_type,
                         stack=cfg.stack, max_frames=MAX_FRAMES, record=rec)
        except Exception:
            continue
        if rec:
            frames.append(np.stack([r[0] for r in rec]))
            bytes_.append(np.array([r[1] for r in rec], dtype=np.uint8))
    return {"accepted": len(acc), "cutoff": cutoff, "scored": len(scored)}, frames, bytes_

def main():
    ctx = O.Ctx()
    ds = ctx.dataset([load_run_dir(ROOT / "data/runs" / n) for n in EARLIEST])
    out = {"configs": {}, "note": "3 seeds x 2 filters, rounds 1-2, shared deterministic base"}
    for cfgname in ("top20", "surv"):
        out["configs"][cfgname] = []
        for seed in (0, 1, 2):
            policy, pcfg, _ = load_policy(BASE)
            _, thr = evaluate(ctx, policy, pcfg, f"{cfgname}_s{seed}_base", log=lambda *a: None)
            dirs = []
            for rnd in (1, 2):
                s = session_when_free(O.ROM, O.MOVIE,
                                      ctx.frames_needed(p.frame for p in ctx.traj))
                try:
                    if cfgname == "top20":
                        st, fr, by = rollout_plain(ctx, s, policy, pcfg, thr,
                                                   1000 + seed * 10 + rnd, 0.20)
                    else:
                        st, fr, by = rollout_round_survival(ctx, s, policy, pcfg, thr,
                                                            1000 + seed * 10 + rnd)
                finally:
                    s.close()
                if not fr:
                    break
                d = ROOT / f"data/runs_self/{cfgname}_s{seed}_r{rnd}"
                write_self_run(d, np.concatenate(fr), np.concatenate(by)); dirs.append(d)
                sd = ctx.dataset([load_run_dir(x) for x in dirs])
                e = random_rows(ds, min(len(ds), len(sd)), seed=seed * 10 + rnd)
                policy = train(policy, ConcatDataset([Subset(ds, e), sd]), ROUND_STEPS, 1e-4,
                               seed * 10 + rnd, log=lambda *a: None)
                res, thr = evaluate(ctx, policy, pcfg, f"{cfgname}_s{seed}_r{rnd}")
                res.update({"seed": seed, "round": rnd, "filter": cfgname, **st})
                if rnd == 2:
                    out["configs"][cfgname].append(res)
                    save_policy(CKPTS / f"{cfgname}_s{seed}_r2.pt", policy, pcfg,
                                {n: 0.5 for n in NES_BUTTON_ORDER})
            OUT.write_text(json.dumps(out, indent=2, default=str))

    print("\nROUND 2, three seeds per filter")
    summ = {}
    for c, rs in out["configs"].items():
        p2 = [r["pipe2"]["rate"] for r in rs]; dd = [r["deaths"] for r in rs]
        summ[c] = {"pipe2": p2, "deaths": dd, "pipe2_mean": float(np.mean(p2)),
                   "pipe2_spread_pp": float((max(p2) - min(p2)) * 100),
                   "k_total": sum(r["pipe2"]["k"] for r in rs),
                   "n_total": sum(r["pipe2"]["n"] for r in rs)}
        print(f"  {c}: pipe2 " + " ".join(f"{v*100:.1f}%" for v in p2) +
              f"  spread {summ[c]['pipe2_spread_pp']:.1f} pp  deaths {dd}")
    lo, hi = diff_ci(summ["surv"]["k_total"], summ["surv"]["n_total"],
                     summ["top20"]["k_total"], summ["top20"]["n_total"])
    gap = (summ["top20"]["pipe2_mean"] - summ["surv"]["pipe2_mean"]) * 100
    sprd = max(summ["top20"]["pipe2_spread_pp"], summ["surv"]["pipe2_spread_pp"])
    out["summary"] = summ
    out["difference_pooled"] = {"delta_pp": gap, "ci_pp": [lo*100, hi*100],
                               "excludes_zero": bool(lo > 0 or hi < 0)}
    out["verdict"] = (
        f"NOISE: within-filter seed spread is {sprd:.1f} pp against a {abs(gap):.1f} pp "
        f"between-filter gap -- 'top-20% is better' is unsupported"
        if sprd >= abs(gap) else
        f"REAL: seed spread {sprd:.1f} pp is smaller than the {gap:+.1f} pp gap "
        f"[{lo*100:+.1f},{hi*100:+.1f}] -- top-20% holds up")
    print("\n" + "="*78); print(out["verdict"])
    OUT.write_text(json.dumps(out, indent=2, default=str)); print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
