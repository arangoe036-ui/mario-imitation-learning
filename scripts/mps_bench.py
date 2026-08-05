"""Is MPS actually faster for these configs, and does the process boundary hold?

§2 of the fifty-third directive asks for MPS training so §4's arms cost tens of minutes rather than hours.
Two things have to be true before that is worth building on, and neither is obvious:

1. **MPS must be faster for a 200k-parameter model.** Kernel-launch overhead dominates small graphs, and this
   network is deliberately tiny (~0.25M parameters, batch 64). A GPU is not automatically a speed-up here.
2. **The process boundary must hold.** `pick_device`'s docstring records that probing MPS poisons every later
   FCEUX child irreversibly. This process therefore never spawns FCEUX, and a *separate*, fresh process is
   used to confirm the emulator still gets Metal afterwards (`scripts/fceux_after_mps.py`).

Throughput is measured on **synthetic tensors**, which isolates compute from the memmap read path, and then
on the **real dataset**, which is what actually gets paid. Both numbers matter: if the real figure is far
below the synthetic one, the bottleneck is data loading and the device is irrelevant.

**`prefer` is always explicit.** `pick_device("auto")` is never called here.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.bc.model import BCPolicy, PolicyConfig, pick_device  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/mps_bench.json"
BATCH = 64
WARMUP = 10
STEPS = 60
N_ACTIONS = 300

#: the two §4 arms, plus the current model for reference
ARMS = {
    "current_84_d64_L1": {"res": 84, "d_model": 64, "n_layers": 2 - 1},
    "R_128_d64_L1": {"res": 128, "d_model": 64, "n_layers": 1},
    "RT_128_d128_L2": {"res": 128, "d_model": 128, "n_layers": 2},
}


def make(res: int, d_model: int, n_layers: int) -> tuple[BCPolicy, PolicyConfig]:
    cfg = PolicyConfig(n_actions=N_ACTIONS, stack=4, d_model=d_model, n_layers=n_layers,
                       head_type="categorical", frame_size=res)
    return BCPolicy(cfg), cfg


def bench_synth(device: str, res: int, d_model: int, n_layers: int) -> dict:
    """Steps/s on random tensors of the right shape: pure compute, no data pipeline."""
    dev = pick_device(device)
    policy, cfg = make(res, d_model, n_layers)
    n_par = sum(p.numel() for p in policy.parameters())
    policy = policy.to(dev)
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    obs = torch.rand(BATCH, cfg.stack, res, res, device=dev)
    prev = torch.zeros(BATCH, 1, dtype=torch.long, device=dev)
    y = torch.randint(0, N_ACTIONS, (BATCH,), device=dev)

    def one():
        loss = torch.nn.functional.cross_entropy(policy(obs, prev), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    for _ in range(WARMUP):
        one()
    if dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        one()
    if dev.type == "mps":
        torch.mps.synchronize()
    el = time.time() - t0
    return {"device": device, "res": res, "d_model": d_model, "n_layers": n_layers,
            "params": int(n_par), "steps": STEPS, "seconds": round(el, 3),
            "steps_per_s": round(STEPS / el, 1)}


def bench_real(device: str) -> dict:
    """Steps/s on the actual run-length dataset at 84x84 -- what training currently pays."""
    import scripts.overnight as O
    from tasdata.bc.data import FrameStackDataset
    from tasdata.bc.runlength import RunLengthDataset, collate, joint_size
    from torch.utils.data import DataLoader

    ctx = O.Ctx()
    tok = FrameStackDataset(ctx.expert_train, ctx.vocab, stack=4, label_mode="token")
    idx_path = ROOT / "data/phase1_runlength_index.npz"
    idx = None
    if idx_path.exists():
        z = dict(__import__("numpy").load(idx_path))
        idx = {k: z[k] for k in ("rows", "joints", "lengths")}
    ds = RunLengthDataset(tok, idx)
    dev = pick_device(device)
    n_cls = joint_size(ctx.vocab.size)
    if n_cls != N_ACTIONS:                      # 25 combos x 12 buckets = 300; assert, do not adapt
        raise ValueError(f"joint size {n_cls} != N_ACTIONS {N_ACTIONS}")
    policy, _ = make(84, 64, 1)
    policy = policy.to(dev)
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, collate_fn=collate,
                        generator=torch.Generator().manual_seed(0))
    it = iter(loader)
    n = 0
    t0 = None
    while n < WARMUP + STEPS:
        try:
            obs, prev, y = next(it)
        except StopIteration:
            it = iter(loader)
            continue
        obs, prev, y = obs.to(dev), prev.to(dev), y.to(dev)
        loss = torch.nn.functional.cross_entropy(policy(obs, prev), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n += 1
        if n == WARMUP:
            if dev.type == "mps":
                torch.mps.synchronize()
            t0 = time.time()
    if dev.type == "mps":
        torch.mps.synchronize()
    el = time.time() - t0
    return {"device": device, "n_samples": len(ds), "steps": STEPS,
            "seconds": round(el, 3), "steps_per_s": round(STEPS / el, 1)}


def main() -> None:
    only = os.environ.get("BENCH_DEVICES", "cpu,mps").split(",")
    out = {"batch": BATCH, "warmup": WARMUP, "timed_steps": STEPS,
           "note": ("this process never spawns FCEUX; the boundary is verified separately by "
                    "scripts/fceux_after_mps.py in a fresh process"),
           "synthetic": {}, "real_84x84": {}}
    print(f"{'arm':22s} {'device':>7s} {'params':>9s} {'steps/s':>9s}")
    for name, a in ARMS.items():
        for dev in only:
            r = bench_synth(dev, a["res"], a["d_model"], a["n_layers"])
            out["synthetic"][f"{name}/{dev}"] = r
            print(f"{name:22s} {dev:>7s} {r['params']:>9,} {r['steps_per_s']:>9.1f}", flush=True)
    for name, a in ARMS.items():
        c = out["synthetic"].get(f"{name}/cpu")
        m = out["synthetic"].get(f"{name}/mps")
        if c and m:
            out["synthetic"][f"{name}/speedup_mps_over_cpu"] = round(
                m["steps_per_s"] / c["steps_per_s"], 2)

    print()
    for dev in only:
        r = bench_real(dev)
        out["real_84x84"][dev] = r
        print(f"real dataset 84x84    {dev:>7s} {'':>9s} {r['steps_per_s']:>9.1f}", flush=True)
    c, m = out["real_84x84"].get("cpu"), out["real_84x84"].get("mps")
    if c and m:
        out["real_84x84"]["speedup_mps_over_cpu"] = round(m["steps_per_s"] / c["steps_per_s"], 2)

    sp = out["synthetic"].get("RT_128_d128_L2/speedup_mps_over_cpu")
    rsp = out["real_84x84"].get("speedup_mps_over_cpu")
    out["verdict"] = (
        f"MPS speedup on the largest arm (synthetic) {sp}x; on the real 84x84 data pipeline {rsp}x. "
        f"If the real speedup is near 1.0 the bottleneck is the memmap read path, not the device, and "
        f"moving training to MPS buys nothing until the loader is addressed.")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + out["verdict"])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
