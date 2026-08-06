"""§1: train on 1-1 instead of on 91% levels that are never evaluated. C1 restricted, C2 half-and-half.

**Two premise corrections from `corpus_composition.json`, both measured before building:**

1. **The dataset does not filter to in-control frames, but it barely needs to.** Neither `FrameStackDataset`
   nor `build_index` references `pregame` or `player_state` — yet only **3.3%** of the 77,916 run-length
   training samples are out-of-control. Run-length encoding already removes most of them, because a pregame or
   death-animation stretch is one long constant-action run and contributes a single sample. So the directive's
   "82% of the 1-1 label is pregame, loading and death animation" is true of *frames* and not of *samples*.

2. **1-1 is 3.0% of training samples, not 8.9%.** 2,323 of 77,916 (1,737 in-control). The 8.9% is a frame
   share; run-length compresses levels unequally, so the sample share is what the sampler actually draws.
   **At 1,000 steps × batch 64 = 64,000 draws the model sees ~1,908 samples of 1-1 — and restricting to 1-1
   gives 27.6 epochs, an exposure multiplier of ~34×, not 11×.**

**⚠ That makes the over-fitting branch more likely, not less: C1 trains on 2,323 samples.** The directive
anticipated 6,949. Reported so the branch is read against the real number.

C1 draws every batch from 1-1 rows only. C2 draws half of each batch from 1-1 and half from the rest, so
variety is retained while exposure rises ~17×.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.data import FrameStackDataset  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig, pick_device  # noqa: E402
from tasdata.bc.provenance import recipe  # noqa: E402
from tasdata.bc.runlength import RunLengthDataset, joint_size  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data/bc_scaleup"
IDXCACHE = ROOT / "data/level_index_map.npz"
OUT = ROOT / "data/level_restricted_train.json"

BATCH, LR = 64, 3e-4
LEVEL = "1-1"


def build(seed, steps, mode, cache):
    """mode: 'only' = C1 (1-1 only) · 'half' = C2 (half of each batch from 1-1)."""
    ctx = O.Ctx()
    n_cls = joint_size(ctx.vocab.size)
    base = FrameStackDataset(ctx.expert_train, ctx.vocab, stack=4, label_mode="token")
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    ds = RunLengthDataset(base, {k: z[k] for k in ("rows", "joints", "lengths")})
    zz = np.load(IDXCACHE, allow_pickle=True)
    lvl = np.array([str(x) for x in zz["level"]])
    is11 = np.flatnonzero(lvl == LEVEL)
    rest = np.flatnonzero(lvl != LEVEL)

    torch.manual_seed(seed)
    dev = pick_device("mps")
    cfg = PolicyConfig(n_actions=n_cls, stack=4, frame_size=84, d_model=64, n_layers=1,
                       head_type="categorical", cnn_channels=(32, 64, 64))
    policy = BCPolicy(cfg).to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    policy.train()
    g = np.random.default_rng(seed)
    losses, hist = [], []
    snaps = sorted(int(x) for x in os.environ.get("SNAP_STEPS", "").split(",") if x.strip())
    for step in range(1, steps + 1):
        if mode == "only":
            pick = g.choice(is11, size=BATCH, replace=len(is11) < BATCH)
        else:
            k = BATCH // 2
            pick = np.concatenate([g.choice(is11, size=k, replace=len(is11) < k),
                                   g.choice(rest, size=BATCH - k, replace=False)])
        xb, yb = [], []
        for i in pick:
            key = int(i)
            if key not in cache:
                o, _p, y = ds[key]
                cache[key] = (o, y)
            o, y = cache[key]
            xb.append(o)
            yb.append(y)
        ob = torch.stack(xb).to(dev)
        yv = torch.tensor(yb, dtype=torch.long).to(dev)
        loss = torch.nn.functional.cross_entropy(policy(ob), yv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 250 == 0:
            hist.append([step, float(np.mean(losses[-250:]))])
        if step in snaps:
            tag = "C1" if mode == "only" else "C2"
            sp = OUTDIR / f"{tag}LAD_s{seed}_{step}.pt"
            torch.save({"model_state": {k: v.cpu() for k, v in policy.state_dict().items()},
                        "policy_config": cfg.to_dict(), "step": step, "mode": mode,
                        "loss_at_snapshot": float(np.mean(losses[-250:])),
                        "corpus": "runs", "cnn_channels": [32, 64, 64]}, sp)
            print(f"    snapshot @ {step} -> {sp.name} loss {np.mean(losses[-250:]):.4f}",
                  flush=True)
    return policy, cfg, float(np.mean(losses[-250:])), hist, len(is11), len(rest)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    specs = []
    for a in sys.argv[1:]:
        tag, seed, steps = a.split(":")
        specs.append((tag, int(seed), int(steps)))
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.update({"batch": BATCH, "lr": LR, "level": LEVEL,
                "composition_note": ("1-1 is 2,323 of 77,916 run-length training samples (3.0%); the "
                                     "8.9% in the directive is a FRAME share. C1 therefore trains on "
                                     "2,323 samples, not the 6,949 anticipated"),
                "modes": {"C1": "every batch drawn from 1-1 only",
                          "C2": "half of each batch from 1-1, half from the rest"}})
    cache: dict = {}
    for tag, seed, steps in specs:
        name = f"{tag}_84_s{seed}" if steps == 1000 else f"{tag}_84_s{seed}_{steps}"
        ck = OUTDIR / f"{name}.pt"
        if ck.exists():
            continue
        t0 = time.time()
        mode = "only" if tag == "C1" else "half"
        policy, cfg, fl, hist, n11, nrest = build(seed, steps, mode, cache)
        torch.save({"model_state": {k: v.cpu() for k, v in policy.state_dict().items()},
                    "policy_config": cfg.to_dict(), "corpus": "runs",
                    "cnn_channels": [32, 64, 64], "loss": "plain cross-entropy",
                    "final_loss": fl, "loss_history": hist, "mode": mode,
                    "n_samples_1_1": n11, "n_samples_rest": nrest,
                    "epochs_over_1_1": steps * BATCH / max(1, n11) if mode == "only"
                    else steps * (BATCH // 2) / max(1, n11),
                    **recipe(steps=steps, batch=BATCH, seed=seed, frame_size=84)}, ck)
        out["arms"][name] = {"tag": tag, "seed": seed, "steps": steps, "mode": mode,
                             "final_loss": fl, "n_samples_1_1": n11,
                             "minutes": round((time.time() - t0) / 60, 1)}
        OUT.write_text(json.dumps(out, indent=2, default=str))
        print(f"  {name}: loss {fl:.4f} ({out['arms'][name]['minutes']} min)", flush=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
