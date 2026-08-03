"""P1: compose the four wins into one model.

Sustain+onset reweighting, earliest-in-chain data, the ~25% subset, and self-imitation. Every
result in the project was measured against a baseline that has none of these.

**Note on the recipe.** "Earliest chain" and "25% subset" very nearly coincide: the three
earliest `warpless/3728` publications total 203,865 frames, which is 20.8% of the 981,385-frame
training corpus, against the scaling table's 25% optimum of 245,346. So this trains on the
earliest-chain data at full size and treats that as the subset — the two wins are not
independent, and pretending otherwise would double-count one choice. The subset size is in any
case a hypothesis rather than a settled result, because the scaling table is still multi-life.

Base training needs no emulator. The self-imitation rounds do, and wait for the lock rather
than failing, so this can be launched alongside emulator work.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.overnight import write_self_run  # noqa: E402
from scripts.single_life import episode, summarise  # noqa: E402
from tasdata.bc.bernoulli import bce_with_onset_weights  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    calibrate,
    diff_ci,
    fresh_policy,
    load_policy,
    random_rows,
    save_policy,
)
from tasdata.bc.session import FceuxSession, TooManyEmulators  # noqa: E402
from tasdata.bc.train import make_loader  # noqa: E402
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/compose.json"
CKPTS = ROOT / "data/bc_compose"
EARLIEST = ["pub-1194", "pub-1106", "pub-262"]
BASE_STEPS = 2000
ROUND_STEPS = 800
ROUNDS = 3
EVAL_SEEDS = 200
ROLLOUTS = 150
ONSET_W, SUSTAIN_W = 10.0, 5.0

# Single-life baselines this must beat (round3_ratio1to1).
BASELINE = {"pipe1": (163, 200), "pipe2": (43, 200), "past720": (38, 200)}


def sustain_loss(logits, bits, onset):
    base = torch.nn.functional.binary_cross_entropy_with_logits(logits, bits, reduction="none")
    w = torch.ones_like(base) + (ONSET_W - 1.0) * onset
    cont = (bits > 0).float() * (1.0 - onset)
    w = w + (SUSTAIN_W - 1.0) * cont
    return (base * w).mean()


def train(policy, ds, steps, lr, seed, log=print):
    policy = policy.to(torch.device("cpu"))
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(ds, batch_size=128, shuffle=True, num_workers=0, seed=seed)
    step, running = 0, 0.0
    while step < steps:
        for obs, _p, bits, onset in loader:
            loss = sustain_loss(policy(obs), bits.float(), onset.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            step += 1
            if step % 400 == 0:
                log(f"    step {step}/{steps} loss {running / 400:.4f}")
                running = 0.0
            if step >= steps:
                break
    policy.eval()
    return policy


def session_when_free(rom, movie, frames, *, tries: int = 240, wait: float = 30.0):
    """Wait for the one-emulator lock instead of failing."""
    for i in range(tries):
        try:
            return FceuxSession(rom, movie, frames).__enter__()
        except TooManyEmulators:
            if i == 0:
                print("    emulator busy; waiting for the lock", flush=True)
            time.sleep(wait)
    raise SystemExit("emulator never became free")


def evaluate(ctx, policy, cfg, tag: str, log=print):
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        rows = [episode(s, policy, cfg, thr, start, i) for i in range(EVAL_SEEDS)]
    finally:
        s.close()
    res = summarise(rows)
    res["tag"] = tag
    for m in ("pipe1", "pipe2", "past720"):
        bk, bn = BASELINE[m]
        lo, hi = diff_ci(bk, bn, res[m]["k"], res[m]["n"])
        res[m]["vs_baseline"] = {"delta": res[m]["rate"] - bk / bn, "ci": [lo, hi],
                                "excludes_zero": bool(lo > 0 or hi < 0)}
    log(f"  {tag}: pipe1 {res['pipe1']['rate'] * 100:5.1f}% "
        f"pipe2 {res['pipe2']['rate'] * 100:5.1f}% "
        f"(vs baseline {res['pipe2']['vs_baseline']['delta'] * 100:+5.1f} pp "
        f"[{res['pipe2']['vs_baseline']['ci'][0] * 100:+.1f}, "
        f"{res['pipe2']['vs_baseline']['ci'][1] * 100:+.1f}]) "
        f"past720 {res['past720']['rate'] * 100:5.1f}%  x_med {res['x_median']:.0f}  "
        f"deaths {res['deaths']}")
    return res, thr


def main() -> None:
    CKPTS.mkdir(parents=True, exist_ok=True)
    ctx = O.Ctx()
    runs = [load_run_dir(ROOT / "data/runs" / n) for n in EARLIEST]
    ds = ctx.dataset(runs)
    frac = len(ds) / sum(len(ctx.dataset([r])) for r in ctx.expert_train)
    print(f"earliest-chain data: {EARLIEST} -> {len(ds):,} frames "
          f"({frac * 100:.1f}% of the training corpus; the scaling optimum is 25%)")

    out = {"recipe": {"data": EARLIEST, "frames": len(ds),
                      "fraction_of_corpus": frac, "onset_weight": ONSET_W,
                      "sustain_weight": SUSTAIN_W, "base_steps": BASE_STEPS,
                      "rounds": ROUNDS, "round_steps": ROUND_STEPS,
                      "note": "earliest-chain and 25%-subset nearly coincide; not independent"},
           "baseline_single_life": {k: {"k": v[0], "n": v[1], "rate": v[0] / v[1]}
                                    for k, v in BASELINE.items()},
           "stages": []}

    print(f"\n[base] sustain+onset on earliest-chain data, {BASE_STEPS} steps")
    policy = fresh_policy(ctx.cfg, seed=0)
    policy = train(policy, ds, BASE_STEPS, 3e-4, 0)
    save_policy(CKPTS / "compose_base.pt", policy, ctx.cfg,
                {n: 0.5 for n in NES_BUTTON_ORDER})
    res, thr = evaluate(ctx, policy, ctx.cfg, "compose_base")
    out["stages"].append(res)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    self_dirs: list[Path] = []
    from scripts.overnight import rollout_round
    for rnd in range(1, ROUNDS + 1):
        print(f"\n[round {rnd}] rollouts")
        s = session_when_free(O.ROM, O.MOVIE,
                              ctx.frames_needed(p.frame for p in ctx.traj))
        try:
            stats, frames, bytes_ = rollout_round(ctx, s, policy, ctx.cfg, thr, rnd,
                                                  episodes=ROLLOUTS)
        finally:
            s.close()
        print(f"  accepted {stats['accepted']}/{stats['scored']} "
              f"cutoff {stats['cutoff']:.0f} score_med {stats['score_median']:.0f}")
        if not frames:
            print("  no accepted rollouts; stopping rounds")
            break
        d = ROOT / f"data/runs_self/compose_round{rnd}"
        write_self_run(d, np.concatenate(frames), np.concatenate(bytes_))
        self_dirs.append(d)

        self_ds = ctx.dataset([load_run_dir(x) for x in self_dirs])
        n_self = len(self_ds)
        e_rows = random_rows(ds, min(len(ds), n_self), seed=rnd)   # 1:1 expert:self
        mixed = ConcatDataset([Subset(ds, e_rows), self_ds])
        print(f"  training 1:1 -> expert {len(e_rows):,} + self {n_self:,}")
        policy = train(policy, mixed, ROUND_STEPS, 1e-4, rnd)
        save_policy(CKPTS / f"compose_round{rnd}.pt", policy, ctx.cfg,
                    {n: 0.5 for n in NES_BUTTON_ORDER})
        res, thr = evaluate(ctx, policy, ctx.cfg, f"compose_round{rnd}")
        res.update({"round": rnd, **stats})
        out["stages"].append(res)
        OUT.write_text(json.dumps(out, indent=2, default=str))

    best = max(out["stages"], key=lambda r: r["pipe2"]["rate"])
    out["best"] = best["tag"]
    out["verdict"] = (
        f"COMPOSITION IMPROVES pipe 2: {best['tag']} at {best['pipe2']['rate'] * 100:.1f}% vs "
        f"baseline 21.5%, difference "
        f"{best['pipe2']['vs_baseline']['delta'] * 100:+.1f} pp "
        f"[{best['pipe2']['vs_baseline']['ci'][0] * 100:+.1f}, "
        f"{best['pipe2']['vs_baseline']['ci'][1] * 100:+.1f}]"
        if best["pipe2"]["vs_baseline"]["excludes_zero"] and
        best["pipe2"]["rate"] > BASELINE["pipe2"][0] / BASELINE["pipe2"][1] else
        f"COMPOSITION EXHAUSTED: best is {best['tag']} at {best['pipe2']['rate'] * 100:.1f}% "
        f"pipe 2 against the 21.5% baseline; the difference does not exclude zero. "
        f"Search is the only remaining move.")
    print("\n" + "=" * 78)
    print(out["verdict"])
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
