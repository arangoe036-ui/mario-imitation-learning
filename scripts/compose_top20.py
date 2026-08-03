"""P1 control: plain top-20% by progress, NO survival test.

The survival gate cut the accepted set from ~38 to 29-30, which is also what tightening the
quantile from 25% to 20% would do. Survival and strictness are confounded. This changes only the
quantile, so if it reproduces the +8/+6/+7.5 pp at pipe 2 then the gain was strictness and the
survival condition is irrelevant.

Original docstring follows.

P1: re-run the composition rounds with a survival-gated acceptance filter.

The old filter accepted the top 25% of rollouts by **progress-from-start**, which cannot tell
"got far and survived" from "got far and died". A rollout that dies at x=1216 after excellent
progress scores highly and is trained on, so the filter actively selects for reckless play --
and deaths rose monotonically across rounds, 30 -> 56 -> 153 -> 173, which is what that predicts.

This changes exactly one condition: a rollout must also **survive to the end of its window**.
Same recipe, same seeds, same steps, so the comparison against `compose.json` is clean.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset, Subset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import (BASELINE, EARLIEST, ROUND_STEPS, ROUNDS, ROLLOUTS,
                             evaluate, session_when_free, train)
from scripts.overnight import write_self_run
from tasdata.bc.overnight_lib import load_policy, random_rows, save_policy
from tasdata.bc.session_player import play_episode
from tasdata.buttons import NES_BUTTON_ORDER
from tasdata.dataset import load_run_dir

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/compose_top20.json"
CKPTS = ROOT / "data/bc_compose_top20"
BASE_CKPT = ROOT / "data/bc_compose/compose_base.pt"
MAX_FRAMES, ACCEPT_FRAC, MIN_PROGRESS = 500, 0.20, 120   # top 20%, no survival test

def rollout_round_survival(ctx, session, policy, cfg, thr, rnd):
    """Top quartile by progress AND zero deaths in the rollout window."""
    rng = np.random.default_rng(1000 + rnd)
    picks = [ctx.traj[i] for i in
             rng.choice(len(ctx.traj), size=min(ROLLOUTS, len(ctx.traj)), replace=False)]
    scored = []
    for k, start in enumerate(picks):
        try:
            ep = play_episode(session, policy, start, ctx.vocab, seed=rnd * 10_000 + k,
                              selection="sample", thresholds=thr, head_type=cfg.head_type,
                              stack=cfg.stack, max_frames=MAX_FRAMES)
        except Exception:
            continue
        gained = ep.max_x_by_level.get(start.label, start.x) - start.x
        score = gained + 4000 * max(0, ep.levels_reached - 1)
        scored.append({"score": score, "k": k, "start": start, "deaths": ep.deaths})
    scores = np.array([s["score"] for s in scored], dtype=float)
    cutoff = max(float(np.quantile(scores, 1 - ACCEPT_FRAC)), MIN_PROGRESS)
    survived = [s for s in scored if s["deaths"] == 0]      # measured, NOT used as a gate
    accepted = [s for s in scored if s["score"] >= cutoff]

    frames, bytes_ = [], []
    for s in accepted:
        rec: list = []
        try:
            play_episode(session, policy, s["start"], ctx.vocab, seed=rnd * 10_000 + s["k"],
                         selection="sample", thresholds=thr, head_type=cfg.head_type,
                         stack=cfg.stack, max_frames=MAX_FRAMES, record=rec)
        except Exception:
            continue
        if rec:
            frames.append(np.stack([r[0] for r in rec]))
            bytes_.append(np.array([r[1] for r in rec], dtype=np.uint8))
    return {"scored": len(scored), "survived": len(survived), "accepted": len(accepted),
            "cutoff": cutoff, "score_median": float(np.median(scores)),
            "died_in_rollout": len(scored) - len(survived)}, frames, bytes_

def main():
    CKPTS.mkdir(parents=True, exist_ok=True)
    ctx = O.Ctx()
    ds = ctx.dataset([load_run_dir(ROOT / "data/runs" / n) for n in EARLIEST])
    policy, cfg, _ = load_policy(BASE_CKPT)     # same base model, unchanged
    print(f"reusing compose_base ({len(ds):,} earliest-chain frames); "
          f"survival-gated rounds only")
    out = {"gate": "plain top 20% by progress; NO survival test (strictness control)",
           "base_checkpoint": BASE_CKPT.name, "stages": []}
    res, thr = evaluate(ctx, policy, cfg, "top20_base")
    out["stages"].append(res)

    self_dirs = []
    for rnd in range(1, ROUNDS + 1):
        print(f"\n[round {rnd}] survival-gated rollouts")
        s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed(p.frame for p in ctx.traj))
        try:
            stats, frames, bytes_ = rollout_round_survival(ctx, s, policy, cfg, thr, rnd)
        finally:
            s.close()
        print(f"  scored {stats['scored']}  died in rollout {stats['died_in_rollout']}  "
              f"survived {stats['survived']}  accepted {stats['accepted']}  "
              f"cutoff {stats['cutoff']:.0f}")
        if not frames:
            print("  no accepted rollouts; stopping"); break
        d = ROOT / f"data/runs_self/top20_round{rnd}"
        write_self_run(d, np.concatenate(frames), np.concatenate(bytes_))
        self_dirs.append(d)
        self_ds = ctx.dataset([load_run_dir(x) for x in self_dirs])
        e_rows = random_rows(ds, min(len(ds), len(self_ds)), seed=rnd)
        policy = train(policy, ConcatDataset([Subset(ds, e_rows), self_ds]),
                       ROUND_STEPS, 1e-4, rnd)
        save_policy(CKPTS / f"surv_round{rnd}.pt", policy, cfg,
                    {n: 0.5 for n in NES_BUTTON_ORDER})
        res, thr = evaluate(ctx, policy, cfg, f"top20_round{rnd}")
        res.update({"round": rnd, **stats})
        out["stages"].append(res)
        OUT.write_text(json.dumps(out, indent=2, default=str))

    old = json.loads((ROOT / "data/compose.json").read_text())
    best_new = max(out["stages"], key=lambda r: r["pipe2"]["rate"])
    deaths = best_new["deaths"]
    p2 = best_new["pipe2"]["rate"]
    out["comparison_vs_old_filter"] = {
        "old_best_pipe2": 0.54, "old_best_deaths_at_n200": 153,
        "new_best_tag": best_new["tag"], "new_best_pipe2": p2, "new_best_deaths": deaths}
    # Matched-round comparison only. No best-of-run figures.
    surv = json.loads((ROOT / "data/compose_survival.json").read_text())["stages"]
    by_tag = {s["tag"]: s for s in out["stages"]}
    matched = []
    for r in (1, 2, 3):
        n_ = by_tag.get(f"top20_round{r}")
        s_ = next((x for x in surv if x["tag"] == f"surv_round{r}"), None)
        o_ = next((x for x in json.loads((ROOT / "data/compose.json").read_text())["stages"]
                   if x["tag"] == f"compose_round{r}"), None)
        if n_ and s_ and o_:
            matched.append({"round": r,
                            "top25_pipe2": o_["pipe2"]["rate"], "top25_deaths": o_["deaths"],
                            "surv_pipe2": s_["pipe2"]["rate"], "surv_deaths": s_["deaths"],
                            "top20_pipe2": n_["pipe2"]["rate"], "top20_deaths": n_["deaths"]})
    out["matched_rounds"] = matched
    print("\nMATCHED ROUNDS -- pipe 2 (deaths/200)")
    for m in matched:
        print(f"  round {m['round']}: top25 {m['top25_pipe2']*100:5.1f}% ({m['top25_deaths']:3d})  "
              f"surv {m['surv_pipe2']*100:5.1f}% ({m['surv_deaths']:3d})  "
              f"top20 {m['top20_pipe2']*100:5.1f}% ({m['top20_deaths']:3d})")
    if matched:
        rep = sum(1 for m in matched if m["top20_pipe2"] >= m["top25_pipe2"] + 0.04)
        out["verdict"] = (
            f"STRICTNESS, NOT SURVIVAL: plain top-20% reproduces the pipe-2 gain in "
            f"{rep}/{len(matched)} matched rounds -- the survival condition is not the cause"
            if rep >= 2 else
            f"SURVIVAL IS DOING REAL WORK: plain top-20% reproduces the gain in only "
            f"{rep}/{len(matched)} matched rounds despite the same accepted-set size")
    print("\n" + "=" * 78); print(out["verdict"])
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
