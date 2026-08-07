"""§1: change the OBJECTIVE — the only component never varied in 65 blocks.

Two interventions, each one line, both on the full corpus with everything else held at the peak recipe
(84x84, `cnn(32,64,64)`, linear head, d64/L1, 1,000 steps, batch 64, lr 3e-4):

| arm | one line | diagnosis it attacks |
|---|---|---|
| **LS** | `cross_entropy(..., label_smoothing=eps)` | over-commitment: forbid the output collapsing onto the demonstrated token |
| **WR** | sample with weight `s` on in-window rows | the flawless trajectory spends almost no time where the policy loses |

**The ε=0 / 1.0x cell is the existing `PK32_84_s0..9`** — identical recipe, already trained and already rolled
out. It is reused rather than retrained so the baseline is literally the same weights the last three blocks
were measured against.

**⚠ DESIGN LIMIT ON WR, MEASURED BEFORE TRAINING (`data/failure_windows.json`).** The windows are 1-1
coordinates, so only in-control 1-1 samples can fall inside one: **295 of 77,916 rows, 0.38%.** At the
directive's strongest mild strength (3.0x) that is 1.13% of draws — 721 per 1,000 steps against 242 at
baseline. **A 479-draw change in 64,000 may well be null by construction, and a sweep that cannot express its
own answer is a defect this project has shipped before (block 57).** So the swept strengths keep the
directive's mild ladder *and add one strong rung*, 8.0x (2.95% of draws, ~1,890 per 1,000 steps, 7.8x
exposure), purely so that a flat result across the whole sweep can be read as "reweighting does not work"
rather than "the manipulation was too small to see." **The mild rungs remain, so this is not an aggressive
strength run alone** — which the directive forbids.

**PRE-SPECIFIED PRIMARY OUTCOME, written here before any arm runs: clearance past pipe 4 (x > 975).**
Everything else is secondary. Bonferroni family = **6 walls** (pipe 1 and pipe 2 are the same measurement).
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
OUT = ROOT / "data/objective_train.json"

BATCH, LR, STEPS = 64, 3e-4, 1000


def epoch_stream(n, inwin, strength, g):
    """Yield batches the way the BASELINE does: a permutation per epoch, no replacement within it.

    **This matters and it nearly went wrong.** `scaleup_train.py` — which produced the `PK32_84_s*`
    checkpoints reused as the eps=0 / 1.0x cell — uses `DataLoader(shuffle=True)`, i.e. sampling *without*
    replacement across an epoch. An i.i.d. sampler would show the model ~43,650 distinct rows in 64,000 draws
    where a permutation shows 64,000. Comparing a label-smoothed arm trained with an i.i.d. sampler against
    that baseline would confound the objective with the sampler.

    The reweighting is therefore implemented as **repetition inside the index**, not as a sampling
    distribution: an in-window row appears `strength` times (fractional part resolved by a seeded coin), and
    the expanded index is permuted. **At `strength == 1.0` this reduces exactly to the baseline scheme**, so
    `PK32` is a valid control for the WR arms as well as the LS arms. The realised order differs from
    torch's, but the sampling scheme is identical; only the scheme is what the comparison rests on.
    """
    reps = np.where(inwin, float(strength), 1.0)
    whole = np.floor(reps).astype(np.int64)
    idx = np.repeat(np.arange(n, dtype=np.int64), whole)
    frac = reps - whole
    if frac.any():
        idx = np.concatenate([idx, np.flatnonzero(g.random(n) < frac)])
    while True:
        perm = g.permutation(idx)
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            yield perm[i:i + BATCH]


def build(seed, steps, eps, strength, cache):
    """`eps` = label-smoothing epsilon. `strength` = repetition weight on in-window rows (1.0 = baseline)."""
    ctx = O.Ctx()
    n_cls = joint_size(ctx.vocab.size)
    base = FrameStackDataset(ctx.expert_train, ctx.vocab, stack=4, label_mode="token")
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    ds = RunLengthDataset(base, {k: z[k] for k in ("rows", "joints", "lengths")})
    inwin = np.load(IDXCACHE, allow_pickle=True)["in_window"]
    n = len(inwin)

    torch.manual_seed(seed)
    dev = pick_device("mps")
    cfg = PolicyConfig(n_actions=n_cls, stack=4, frame_size=84, d_model=64, n_layers=1,
                       head_type="categorical", cnn_channels=(32, 64, 64))
    policy = BCPolicy(cfg).to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    policy.train()
    g = np.random.default_rng(seed)
    stream = epoch_stream(n, inwin, strength, g)
    losses, plain, hist = [], [], []
    snaps = sorted(int(x) for x in os.environ.get("SNAP_STEPS", "").split(",") if x.strip())
    tag = name_of(eps, strength)
    for step in range(1, steps + 1):
        pick = next(stream)
        xb, yb = [], []
        for i in pick:
            key = int(i)
            if key not in cache:
                o, _pp, y = ds[key]
                cache[key] = (o, y)
            o, y = cache[key]
            xb.append(o)
            yb.append(y)
        ob = torch.stack(xb).to(dev)
        yv = torch.tensor(yb, dtype=torch.long).to(dev)
        logits = policy(ob)
        loss = torch.nn.functional.cross_entropy(logits, yv, label_smoothing=eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        # the smoothed loss is not comparable across epsilon, so record the plain NLL too --
        # otherwise "LS0.20 has a higher loss" is an artifact of the objective, not of fit.
        with torch.no_grad():
            plain.append(float(torch.nn.functional.cross_entropy(logits, yv).detach()))
        if step % 250 == 0:
            hist.append([step, float(np.mean(losses[-250:])), float(np.mean(plain[-250:]))])
        if step in snaps:
            sp = OUTDIR / f"{tag}LAD_s{seed}_{step}.pt"
            torch.save({"model_state": {k: v.cpu() for k, v in policy.state_dict().items()},
                        "policy_config": cfg.to_dict(), "step": step,
                        "label_smoothing": eps, "window_strength": strength,
                        "loss_at_snapshot": float(np.mean(losses[-250:])),
                        "plain_nll_at_snapshot": float(np.mean(plain[-250:])),
                        "corpus": "runs", "cnn_channels": [32, 64, 64]}, sp)
            print(f"    snapshot @ {step} -> {sp.name} loss {np.mean(losses[-250:]):.4f} "
                  f"nll {np.mean(plain[-250:]):.4f}", flush=True)
    return (policy, cfg, float(np.mean(losses[-250:])), float(np.mean(plain[-250:])), hist)


def name_of(eps, strength):
    return f"LS{round(eps * 100):03d}" if strength == 1.0 else f"WR{round(strength * 10):03d}"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    specs = []
    for a in sys.argv[1:]:
        kind, val, seed, steps = a.split(":")
        specs.append((kind, float(val), int(seed), int(steps)))
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.update({
        "batch": BATCH, "lr": LR, "steps": STEPS,
        "PRE_SPECIFIED_PRIMARY": "clearance past pipe 4 (x > 975)",
        "bonferroni_family": "6 walls x n_arms (pipe1 and pipe2 are one measurement)",
        "baseline_cell": "PK32_84_s0..9 — eps=0, strength=1.0, identical recipe, reused not retrained",
        "prediction_before_running": (
            "if over-commitment to a flawless trajectory is the mechanism, label smoothing moves the PEAK "
            "LATER and slows the collapse. This prediction has already failed twice (augmented data, "
            "restricted corpus); a third failure retires the over-commitment story."),
        "wr_design_limit": (
            "only 295 of 77,916 rows (0.38%) can be reweighted, because the windows are 1-1 coordinates. "
            "3.0x moves them to 1.13% of draws. An 8.0x rung is added so a flat sweep is interpretable."),
        "loss_note": ("`final_loss` is the SMOOTHED objective and is not comparable across epsilon; "
                      "`final_nll` is plain cross-entropy on the same batches and is."),
    })
    cache: dict = {}
    for kind, val, seed, steps in specs:
        eps, strength = (val, 1.0) if kind == "LS" else (0.0, val)
        tag = name_of(eps, strength)
        name = f"{tag}_s{seed}" if steps == STEPS else f"{tag}_s{seed}_{steps}"
        ck = OUTDIR / f"{name}.pt"
        if ck.exists():
            continue
        t0 = time.time()
        policy, cfg, fl, nll, hist = build(seed, steps, eps, strength, cache)
        torch.save({"model_state": {k: v.cpu() for k, v in policy.state_dict().items()},
                    "policy_config": cfg.to_dict(), "corpus": "runs",
                    "cnn_channels": [32, 64, 64],
                    "loss": "cross-entropy with label smoothing" if eps else "plain cross-entropy",
                    "label_smoothing": eps, "window_strength": strength,
                    "final_loss": fl, "final_nll": nll, "loss_history": hist,
                    **recipe(steps=steps, batch=BATCH, seed=seed, frame_size=84)}, ck)
        out["arms"][name] = {"kind": kind, "label_smoothing": eps, "window_strength": strength,
                             "seed": seed, "steps": steps, "final_loss": fl, "final_nll": nll,
                             "minutes": round((time.time() - t0) / 60, 1)}
        OUT.write_text(json.dumps(out, indent=2, default=str))
        print(f"  {name}: loss {fl:.4f} nll {nll:.4f} ({out['arms'][name]['minutes']} min)", flush=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
