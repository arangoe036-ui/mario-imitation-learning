"""§4 arms R and RT: train on 128x128 on MPS. **This process never spawns FCEUX.**

Two arms, one variable each, because this project has been burned four times by moving several at once:

| arm | resolution | transformer | isolates |
|---|---|---|---|
| **R** | 128x128 | `d_model=64`, `n_layers=1` -- unchanged | **information**: was the Goomba too small to see? |
| **RT** | 128x128 | `d_model=128`, `n_layers=2` | **capacity**: can a bigger network use it? |

Plain cross-entropy over the run-length joint classes, no reweighting, nothing else changed. A **B** arm at
84x84 with the same step count is trained too, because R-minus-baseline is only interpretable against a
baseline trained for the same 15,000 steps -- the existing `runlength.pt` was 3,000, so comparing against it
would confound resolution with training length.

**Evaluation is deliberately absent from this file.** Touching MPS poisons every later FCEUX child, so the
rollouts run in a separate CPU process (`scripts/scaleup_eval.py`). The split is the whole reason MPS is
usable at all; see `data/mps_boundary.json` for what was and was not demonstrated about it.

Banking every `CHUNK_STEPS` because jobs on this machine are killed every few minutes; a resumed arm reloads
its optimiser state as well as its weights, or the effective schedule differs from the un-interrupted one.
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
from tasdata.bc.runlength import RunLengthDataset, build_index, collate, joint_size  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS128 = ROOT / "data/runs128"
OUTDIR = ROOT / "data/bc_scaleup"
LR = 3e-4
CHUNK_STEPS = 250
#: Block 58 §3 needs the depth-vs-steps curve, and the directive assumed the every-250-step banked
#: checkpoints "already exist". They do not: `{name}.partial.pt` is OVERWRITTEN at every bank, so only
#: the final step survives. Set SNAP_STEPS to keep permanent snapshots at chosen steps -- same recipe,
#: same seed, nothing trained longer or wider, just intermediate weights retained.
SNAP_STEPS = sorted(int(x) for x in os.environ.get("SNAP_STEPS", "").split(",") if x.strip())

#: Full spec per arm, so every difference between two arms is visible in one place. `phase1` reproduces
#: the existing `runlength.pt` recipe exactly -- **batch 128, 3,000 steps** -- because that checkpoint
#: differs from the 15,000-step arms in *two* variables and the timing-lift comparison needs them
#: separated. The seed replicas exist because this project's ledger records a 14.5-24.5 pp training-seed
#: spread: one seed is a screen, not a result.
ARMS = {
    "phase1_repro_84_3k_b128": dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=3_000, batch=128, seed=0),
    "B_84_d64_L1":             dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=0),
    "B_84_seed1":              dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=1),
    "B_84_seed2":              dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=2),
    "R_128_d64_L1":            dict(size=128, d_model=64, n_layers=1, corpus="runs128",
                                    steps=15_000, batch=64, seed=0),
    "RT_128_d128_L2":          dict(size=128, d_model=128, n_layers=2, corpus="runs128",
                                    steps=15_000, batch=64, seed=0),
    # ---- block 55 §3: the 2x2 on ENCODER WIDTH. `cnn_channels` has been (16, 32, 32) in every arm
    # this project has ever trained -- block 53 widened `d_model` and `n_layers`, i.e. the reasoning,
    # and never the vision. So "a bigger model did not clear more" was only ever tested as a bigger
    # thinker behind the same small eyes. (32, 64, 64) is the standard encoder for this work.
    # V - P is the parameter-matched resolution control: 84 -> 128 raised the count 172k -> 367k by
    # itself, so B -> R was never a clean test of resolution.
    "P_84_cnn32":              dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=0, cnn=(32, 64, 64)),
    "P_84_cnn32_seed1":        dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=1, cnn=(32, 64, 64)),
    "V_128_cnn32":             dict(size=128, d_model=64, n_layers=1, corpus="runs128",
                                    steps=15_000, batch=64, seed=0, cnn=(32, 64, 64)),
    "V_128_cnn32_seed1":       dict(size=128, d_model=64, n_layers=1, corpus="runs128",
                                    steps=15_000, batch=64, seed=1, cnn=(32, 64, 64)),
    # ---- block 56 §1: FIVE seeds per cell. The variance-collapse claim was a range of two numbers
    # against a range of two numbers; a spread and a standard deviation need n=5, and training is
    # ~4 min. B already had seeds 0/1/2, so it needs two more, not three.
    "B_84_seed3":              dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=3),
    "B_84_seed4":              dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=4),
    "P_84_cnn32_seed2":        dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=2, cnn=(32, 64, 64)),
    "P_84_cnn32_seed3":        dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=3, cnn=(32, 64, 64)),
    "P_84_cnn32_seed4":        dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=4, cnn=(32, 64, 64)),
    # ---- block 56 §3: wider still, 84x84 only. Trained ONLY if §1 confirms the spread separation.
    "W_84_cnn48":              dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=0, cnn=(48, 96, 96)),
    "W_84_cnn48_seed1":        dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=1, cnn=(48, 96, 96)),
    "W_84_cnn48_seed2":        dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=15_000, batch=64, seed=2, cnn=(48, 96, 96)),
    # ---- block 57 §3: TRAINING LENGTH, the cheapest untested axis. Only ever varied 3k -> 15k.
    # 60k steps is ~100 epochs over 77,916 run samples. If x_max improves while loss falls, longer
    # works; if x_max degrades, that is the ceiling and it is a result.
    "L_84_cnn32_60k":          dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=60_000, batch=64, seed=0, cnn=(32, 64, 64)),
    "L_84_cnn32_60k_seed1":    dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=60_000, batch=64, seed=1, cnn=(32, 64, 64)),
    "L_84_cnn32_60k_seed2":    dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=60_000, batch=64, seed=2, cnn=(32, 64, 64)),
    # ---- block 57 §6: the compound arm. Gated -- only run if L or W beats the P baseline on x_max.
    "WL_84_cnn48_60k":         dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=60_000, batch=64, seed=0, cnn=(48, 96, 96)),
    "WL_84_cnn48_60k_seed1":   dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=60_000, batch=64, seed=1, cnn=(48, 96, 96)),
    "WL_84_cnn48_60k_seed2":   dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=60_000, batch=64, seed=2, cnn=(48, 96, 96)),
    # ---- block 58 §3: identical recipe to L, re-run only to RETAIN intermediate snapshots so the
    # depth-vs-steps curve can be measured. Two seeds, because a peak on one seed is a screen.
    "CURVE_84_cnn32":          dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=60_000, batch=64, seed=0, cnn=(32, 64, 64)),
    "CURVE_84_cnn32_seed1":    dict(size=84, d_model=64, n_layers=1, corpus="runs",
                                    steps=60_000, batch=64, seed=1, cnn=(32, 64, 64)),
}


def index_path(corpus: str, size: int) -> Path:
    """Run-boundary index is a property of the ACTIONS, not the pixels, so it is shared per corpus."""
    return ROOT / f"data/runlength_index_{corpus}.npz"


def load_index(ds, corpus: str, size: int) -> dict:
    p = index_path(corpus, size)
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in ("rows", "joints", "lengths")}
    t0 = time.time()
    idx = build_index(ds)          # walks ~1M rows in Python; cache it or every restart pays again
    np.savez(p, **idx)
    print(f"    built run-length index in {time.time() - t0:.0f}s -> {p.name}", flush=True)
    return idx


def runs_for(corpus: str):
    """Expert train split, from whichever capture the arm uses.

    **The same frozen split, by name, for every arm.** Re-deriving a split per corpus would confound
    resolution with which runs were trained on, which is the one thing these arms exist to separate.
    """
    ctx = O.Ctx()
    if corpus == "runs":
        return ctx.expert_train
    from tasdata.dataset import load_run_dir
    names = [r.name for r in ctx.expert_train]
    missing = [n for n in names if not (RUNS128 / n / "frames.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(names)} expert-train runs absent from {RUNS128.name}: "
            f"{missing[:4]}{'...' if len(missing) > 4 else ''} -- the capture is incomplete"
        )
    return [load_run_dir(RUNS128 / n) for n in names]


def train_arm(name: str, size: int, d_model: int, n_layers: int, corpus: str,
              steps: int, batch: int, seed: int, cnn: tuple[int, ...] = (16, 32, 32)) -> dict:
    STEPS, BATCH, SEED = steps, batch, seed
    # Seed BOTH weight init and shuffling. Seeding only the loader generator would make a "seed
    # replica" differ in data order alone, which understates the spread the ledger records at
    # 14.5-24.5 pp -- that spread comes from initialisation as much as from ordering.
    torch.manual_seed(SEED)
    dev = pick_device("mps")                       # explicit; never pick_device("auto") here
    ctx = O.Ctx()
    runs = runs_for(corpus)
    stored = None
    base = FrameStackDataset(runs, ctx.vocab, stack=4, label_mode="token", frame_size=size)
    stored = base.stored_size
    ds = RunLengthDataset(base, load_index(base, corpus, size))
    n_cls = joint_size(ctx.vocab.size)
    cfg = PolicyConfig(n_actions=n_cls, stack=4, frame_size=size, d_model=d_model,
                       n_layers=n_layers, head_type="categorical", cnn_channels=tuple(cnn))
    policy = BCPolicy(cfg)
    ckpt = OUTDIR / f"{name}.pt"
    partial = OUTDIR / f"{name}.partial.pt"
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    done = 0
    if partial.exists():
        blob = torch.load(partial, map_location="cpu", weights_only=False)
        policy.load_state_dict(blob["model_state"])
        if "opt_state" in blob:
            opt.load_state_dict(blob["opt_state"])
        done = int(blob["step"])
        print(f"    resuming {name} from step {done}/{STEPS}", flush=True)
    policy = policy.to(dev)
    # AdamW state loaded on CPU must follow the parameters, or the first step runs on mixed devices
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(dev)

    print(f"[{name}] {stored}x{stored} stored -> {size}x{size} served | d_model {d_model} "
          f"L{n_layers} cnn {tuple(cnn)} | {sum(p.numel() for p in policy.parameters()):,} params | "
          f"{len(ds):,} run samples | device {dev}", flush=True)
    if done >= STEPS:
        print("    already complete", flush=True)
    else:
        policy.train()
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                            collate_fn=collate, generator=torch.Generator().manual_seed(SEED))
        step, t0, losses = done, time.time(), []
        # Loss history at every bank point, so "did loss still fall at 60k" is answerable from the
        # checkpoint instead of from a log that may not survive.
        hist = list(blob.get("loss_history", [])) if partial.exists() else []
        while step < STEPS:
            for obs, prev, y in loader:
                obs, prev, y = obs.to(dev), prev.to(dev), y.to(dev)
                loss = torch.nn.functional.cross_entropy(policy(obs, prev), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                step += 1
                losses.append(float(loss.detach()))
                if step in SNAP_STEPS:
                    snap = OUTDIR / f"{name}.snap{step}.pt"
                    torch.save({"model_state": {k: v.cpu() for k, v in
                                                policy.state_dict().items()},
                                "policy_config": cfg.to_dict(), "step": step,
                                "loss_at_snapshot": float(np.mean(losses[-CHUNK_STEPS:]))
                                if losses else None,
                                "loss_history": hist,
                                **recipe(steps=step, batch=BATCH, seed=SEED, frame_size=size),
                                "corpus": corpus, "cnn_channels": list(cnn),
                                "snapshot_of": name}, snap)
                    print(f"    snapshot @ {step} -> {snap.name}", flush=True)
                if step % CHUNK_STEPS == 0 or step >= STEPS:
                    hist.append([step, float(np.mean(losses[-CHUNK_STEPS:]))])
                    torch.save({"model_state": {k: v.cpu() for k, v in
                                                policy.state_dict().items()},
                                "opt_state": opt.state_dict(), "step": step,
                                "loss_history": hist,
                                "policy_config": cfg.to_dict()}, partial)
                    rate = (step - done) / max(time.time() - t0, 1e-9)
                    print(f"    step {step}/{STEPS} loss {np.mean(losses[-CHUNK_STEPS:]):.4f} "
                          f"({rate:.1f} steps/s)", flush=True)
                if step >= STEPS:
                    break
        policy.eval()
    blob = torch.load(partial, map_location="cpu", weights_only=False)
    torch.save({"model_state": blob["model_state"], "policy_config": cfg.to_dict(),
                "loss_history": blob.get("loss_history", []),
                "step": blob["step"], "arm": name,
                "stored_size": stored, "corpus": corpus, "loss": "plain cross-entropy",
                "lr": LR, "n_train_runs": len(runs), "n_samples": len(ds),
                # steps / batch / seed / git_sha / git_dirty / frame_size -- the five fields whose
                # absence from runlength.pt made "outlier seed" and "different recipe"
                # indistinguishable after the fact.
                **recipe(steps=STEPS, batch=BATCH, seed=SEED, frame_size=size)}, ckpt)
    return {"arm": name, "checkpoint": str(ckpt.relative_to(ROOT)), "frame_size": size,
            "stored_size": stored, "d_model": d_model, "n_layers": n_layers, "corpus": corpus,
            "steps": int(blob["step"]), "batch": BATCH, "seed": SEED, "lr": LR,
            "cnn_channels": list(cnn),
            "samples_seen": int(blob["step"]) * BATCH, "n_samples": len(ds),
            "params": sum(v.numel() for v in blob["model_state"].values())}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or list(ARMS)
    out_path = OUTDIR / "train_summary.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {"arms": {}}
    out.setdefault("arms", {})
    out.update({"lr": LR, "loss": "plain cross-entropy, no reweighting",
                "device": "mps (explicit); this process never spawns FCEUX",
                "arm_specs": {k: dict(v) for k, v in ARMS.items()},
                "note": ("B is the 84x84 control at the SAME 15,000 steps -- the existing "
                         "runlength.pt was 3,000 steps AND batch 128, two differences, so "
                         "phase1_repro reproduces that recipe exactly to separate them")})
    for name in only:
        a = ARMS[name]
        t0 = time.time()
        rec = train_arm(name, a["size"], a["d_model"], a["n_layers"], a["corpus"],
                        a["steps"], a["batch"], a["seed"], a.get("cnn", (16, 32, 32)))
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        out["arms"][name] = rec
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"    {name} done in {rec['minutes']} min\n", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
