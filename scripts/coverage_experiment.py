"""P1: does surface-route coverage of x=916-2616 fix the frontier deaths?

The unit is not frames. Two covering runs x ~two gaps is about **four demonstrations** of the actual
behaviour, so inclusion alone is unlikely to teach it -- this project's two most successful
interventions (onset x10, sustain x5) exist because rare events need amplification. So reweighting
is the experiment, not a refinement.

And adding the two covering runs whole would change four things at once: +66% volume, breaking
earliest-chain-only selection, overshooting the 25% subset, and (incidentally) adding coverage. Only
the ~1,606 qualifying frames are added, and a volume-matched control adds the same number of frames
from runs that do NOT cover the stretch.

  arm B  earliest-chain + covering segments, duplicated x20
  arm C  earliest-chain + frame-matched segments from non-covering runs  (control)

Primary outcome: deaths in x=1216-1248 and 1504-1536, currently ~24 per 200 episodes. A global rate
at n=200 cannot resolve this intervention; a targeted count going 24 -> 5 can.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset, Subset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import EARLIEST, ROUND_STEPS, evaluate, session_when_free, train
from scripts.compose_survival import rollout_round_survival
from scripts.overnight import write_self_run
from tasdata.bc.overnight_lib import fresh_policy, random_rows, save_policy, wilson
from tasdata.buttons import NES_BUTTON_ORDER
from tasdata.dataset import load_run_dir
from tasdata.ram import column

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/coverage_experiment.json"
CKPTS = ROOT / "data/bc_coverage"; CKPTS.mkdir(parents=True, exist_ok=True)
COVERING = ["pub-3648", "pub-4313"]
CONTROL_SRC = ["pub-1331", "pub-1349"]      # train, warpless, do NOT cover the stretch
LO, HI = 916, 2616
REWEIGHT, BASE_STEPS, ROUNDS = 20, 2000, 2
FRONTIER_BINS = (1216, 1248, 1504, 1536)

def stretch_rows(ctx, names):
    """Row indices whose observation lies in the undemonstrated stretch of 1-1."""
    runs = [load_run_dir(ROOT / "data/runs" / n) for n in names]
    ds = ctx.dataset(runs)
    rows, off = [], 0
    for r in runs:
        tr = np.asarray(r.trace)
        w, s = column(tr, "world"), column(tr, "stage")
        x, st, pg = column(tr, "x_position"), column(tr, "player_state"), column(tr, "pregame")
        n = min(len(x), len(ctx.dataset([r])))
        m = ((w[:n] == 1) & (s[:n] == 1) & (pg[:n] == 1) & (st[:n] == 8)
             & (x[:n] >= LO) & (x[:n] <= HI))
        rows += (np.flatnonzero(m) + off).tolist()
        off += len(ctx.dataset([r]))
    return ds, rows

def frontier_deaths(res):
    return sum(v for b, c in res["cause_by_bin_32px"].items() if int(b) in FRONTIER_BINS
               for v in c.values())

def main():
    ctx = O.Ctx()
    base_ds = ctx.dataset([load_run_dir(ROOT / "data/runs" / n) for n in EARLIEST])
    cov_ds, cov_rows = stretch_rows(ctx, COVERING)
    print(f"earliest-chain base: {len(base_ds):,} frames")
    print(f"covering segments  : {len(cov_rows):,} frames from {COVERING}")
    ctl_ds = ctx.dataset([load_run_dir(ROOT / "data/runs" / n) for n in CONTROL_SRC])
    ctl_rows = random_rows(ctl_ds, len(cov_rows), seed=0)
    print(f"control segments   : {len(ctl_rows):,} frames from {CONTROL_SRC} (frame-matched)\n")

    arms = {
        "B_coverage_x20": ConcatDataset([base_ds] + [Subset(cov_ds, cov_rows)] * REWEIGHT),
        "C_control_matched": ConcatDataset([base_ds, Subset(ctl_ds, ctl_rows)]),
    }
    out = {"covering_runs": COVERING, "control_runs": CONTROL_SRC,
           "segment_frames": len(cov_rows), "reweight": REWEIGHT,
           "baseline_frontier_deaths_per_200": 24,
           "note": "single seed; screen, not a ranking", "arms": {}}

    for tag, ds in arms.items():
        print(f"[{tag}] training base, {len(ds):,} effective frames")
        policy = fresh_policy(ctx.cfg, seed=0)
        policy = train(policy, ds, BASE_STEPS, 3e-4, 0, log=lambda *a: None)
        res, thr = evaluate(ctx, policy, ctx.cfg, f"{tag}_base")
        stages = [dict(res, frontier_deaths=frontier_deaths(res))]
        print(f"  base: frontier deaths {stages[-1]['frontier_deaths']}/200  "
              f"pipe2 {res['pipe2']['rate']*100:.1f}%  total deaths {res['deaths']}")
        dirs = []
        for rnd in range(1, ROUNDS + 1):
            s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed(p.frame for p in ctx.traj))
            try:
                st, fr, by = rollout_round_survival(ctx, s, policy, ctx.cfg, thr, rnd)
            finally:
                s.close()
            if not fr:
                break
            d = ROOT / f"data/runs_self/{tag}_r{rnd}"
            write_self_run(d, np.concatenate(fr), np.concatenate(by)); dirs.append(d)
            sd = ctx.dataset([load_run_dir(x) for x in dirs])
            e = random_rows(base_ds, min(len(base_ds), len(sd)), seed=rnd)
            mixed = ConcatDataset([Subset(base_ds, e), sd]
                                  + ([Subset(cov_ds, cov_rows)] * REWEIGHT
                                     if tag.startswith("B") else [Subset(ctl_ds, ctl_rows)]))
            policy = train(policy, mixed, ROUND_STEPS, 1e-4, rnd, log=lambda *a: None)
            save_policy(CKPTS / f"{tag}_r{rnd}.pt", policy, ctx.cfg,
                        {n: 0.5 for n in NES_BUTTON_ORDER})
            res, thr = evaluate(ctx, policy, ctx.cfg, f"{tag}_r{rnd}")
            fd = frontier_deaths(res)
            stages.append(dict(res, round=rnd, frontier_deaths=fd, **st))
            print(f"  round {rnd}: frontier deaths {fd}/200  "
                  f"pipe1 {res['pipe1']['rate']*100:.1f}%  pipe2 {res['pipe2']['rate']*100:.1f}%  "
                  f"pipe3 {res['past720']['rate']*100:.1f}%  total deaths {res['deaths']}")
        out["arms"][tag] = stages
        OUT.write_text(json.dumps(out, indent=2, default=str))

    b = min(s["frontier_deaths"] for s in out["arms"].get("B_coverage_x20", [{"frontier_deaths": 99}]))
    c = min(s["frontier_deaths"] for s in out["arms"].get("C_control_matched", [{"frontier_deaths": 99}]))
    out["verdict"] = (
        f"COVERAGE HELPS: arm B reaches {b} frontier deaths/200 against control {c} and a "
        f"~24 baseline"
        if b < c and b <= 12 else
        f"NOT DEMONSTRATED: arm B {b} frontier deaths/200, control {c}, baseline ~24. "
        f"1,606 frames and ~4 crossing demonstrations, amplified 20x, do not fix the frontier -- "
        f"imitation cannot use the data even when it has it.")
    print("\n" + "="*78); print(out["verdict"])
    OUT.write_text(json.dumps(out, indent=2, default=str)); print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
