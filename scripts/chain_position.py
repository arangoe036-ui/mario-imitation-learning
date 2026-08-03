"""Chain-position experiment: does older, less-optimised TAS data train a better policy?

Replaces the glitchless-vs-glitchy comparison, which the corpus cannot support (no
warpless-glitchless runs exist; the pilot rested on a single run). Obsoletion position is
a cleaner proxy for the same underlying question. Within one chain every run completes the
same route with the same level count, so position is the only thing that varies: position
0 is the current publication -- fastest, most heavily optimised, most glitch-dependent --
and higher positions are the older records it obsoleted.

Hypothesis: earlier runs produce a better *live* policy despite being slower runs, because
frame-perfect glitch execution only works from states a learned policy cannot reach.

Arms are drawn from ``warpless/3728``, which has nine runs, all in the train split, all 32
levels. Frame budgets are matched by subsampling the larger arm.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    diff_ci,
    fresh_policy,
    random_rows,
    save_policy,
    train_policy,
)
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/chain_position.jsonl"
CHAIN = "warpless/3728"
SEEDS = (0, 1, 2)
STEPS = 2000
EVAL_SEEDS = 200


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def emit(**row):
    with OUT.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def main() -> None:
    split = json.loads((ROOT / "data/split.json").read_text())["splits"]
    train_names = set(split["train"])
    members = []
    for m in sorted(glob.glob(str(ROOT / "data/runs/*/manifest.json"))):
        name = os.path.basename(os.path.dirname(m))
        j = json.loads(Path(m).read_text())
        if j.get("chain") == CHAIN and name in train_names:
            members.append((int(j["chain_position"]), name, int(j["n_frames"]),
                            j.get("measured_levels")))
    members.sort()
    log(f"chain {CHAIN}: {len(members)} runs in train -> {members}")
    if len(members) < 6:
        log("not enough runs in this chain")
        return

    latest = members[:3]                  # positions 0,1,2 -- newest, most optimised
    earliest = members[-3:]               # highest positions -- oldest records
    log(f"latest   arm: {[(p, n) for p, n, _, _ in latest]}")
    log(f"earliest arm: {[(p, n) for p, n, _, _ in earliest]}")

    ctx = O.Ctx()
    arms = {}
    for tag, group in (("latest", latest), ("earliest", earliest)):
        runs = [load_run_dir(ROOT / "data/runs" / n) for _, n, _, _ in group]
        arms[tag] = ctx.dataset(runs)
    budget = min(len(arms["latest"]), len(arms["earliest"]))
    log(f"matched frame budget: {budget:,}")

    results = []
    for seed in SEEDS:
        for tag in ("earliest", "latest"):
            t0 = time.time()
            try:
                rows = random_rows(arms[tag], budget, seed=seed)
                policy = fresh_policy(ctx.cfg, seed=seed)
                policy = train_policy(policy, Subset(arms[tag], rows), steps=STEPS,
                                      lr=3e-4, onset_weight=10.0, seed=seed, log=log)
                save_policy(ROOT / f"data/bc_overnight/chain_{tag}_seed{seed}.pt",
                            policy, ctx.cfg, {n: 0.5 for n in NES_BUTTON_ORDER})
                row = O.full_eval(ctx, policy, ctx.cfg, f"chain_{tag}_seed{seed}",
                                  seeds=EVAL_SEEDS)
                row.update({"arm": tag, "seed": seed, "frames": budget,
                            "runs": [n for _, n, _, _ in
                                     (earliest if tag == "earliest" else latest)],
                            "minutes": round((time.time() - t0) / 60, 1)})
                results.append(row)
                emit(kind="chain_arm", **row)
                one = row["live"].get("1-1", {})
                log(f"  {tag} seed{seed}: A recall "
                    f"{row['offline']['onset_recall']['A'] * 100:.1f}%  "
                    f"pipe1 {one.get('pipe1_rate', 0) * 100:.1f}%  "
                    f"x2 {row['live'].get('2-1', {}).get('x_median', 0):.0f}")
            except Exception as exc:
                emit(kind="chain_failed", arm=tag, seed=seed, error=f"{type(exc).__name__}: {exc}")
                log(f"  {tag} seed{seed} FAILED: {exc}")

    # Pooled comparison across seeds.
    pooled = {}
    for tag in ("earliest", "latest"):
        rs = [r for r in results if r["arm"] == tag]
        k = sum(int(r["live"].get("1-1", {}).get("pipe1_k", 0)) for r in rs)
        n = sum(int(r["live"].get("1-1", {}).get("n", 0)) for r in rs)
        pooled[tag] = {"pipe1_k": k, "pipe1_n": n, "pipe1_rate": (k / n if n else 0),
                       "A_recall_mean": float(np.mean(
                           [r["offline"]["onset_recall"]["A"] for r in rs])) if rs else 0.0,
                       "seeds": len(rs)}
    if pooled["earliest"]["pipe1_n"] and pooled["latest"]["pipe1_n"]:
        lo, hi = diff_ci(pooled["latest"]["pipe1_k"], pooled["latest"]["pipe1_n"],
                         pooled["earliest"]["pipe1_k"], pooled["earliest"]["pipe1_n"])
        pooled["difference_earliest_minus_latest"] = {
            "delta": pooled["earliest"]["pipe1_rate"] - pooled["latest"]["pipe1_rate"],
            "ci": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}
    emit(kind="chain_pooled", **pooled)
    log(f"pooled: {json.dumps(pooled, default=str)}")


if __name__ == "__main__":
    main()
