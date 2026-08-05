"""Phase 2, obstacle 1, second attempt: the same demos through a mix that is not self-defeating.

**The first attempt tested my mixing bug, not the hypothesis.** Run-length encoding compresses ~100 frames
of demonstration into ~3 run samples, so 48 winning attempts became **144 samples across 7 classes**. A 1:1
expert:demo ratio then *capped the expert side at 144 too* — 288 samples total, and 300 steps over them is
**133 epochs**. That is the exact failure the directive named (13 epochs over 22 near-identical segments at
pipe 4), reproduced at ten times the severity, and it discarded 77,772 of 77,916 expert samples to do it.

Result of that arm, kept for the record: Goomba clearance **65.0% → 53.5%**, deaths in 272–319 **69 → 92**.

This arm changes the mix and nothing else:

* **expert side large** -- 20,000 run-length samples, not capped to the demo count
* **demos repeated to ~9% of the mixture** rather than 50%
* **~1.7 epochs** instead of 133

If the corrected mix still fails to reduce Goomba deaths, the kill condition fires on a fair test.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from scripts.phase1_duration import _Ep  # noqa: E402
from scripts.phase1_variants import rollout  # noqa: E402
from scripts.phase2_goomba import (  # noqa: E402
    CAPPED_TRACES,
    CLEAR_X,
    DEMOS,
    IDX,
    TRACEDIR,
    resumable_eval,
    score,
)
from scripts.rate_matched_control import scripted_episode  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import PIPE_THRESHOLDS  # noqa: E402
from tasdata.bc.runlength import N_BUCKETS, class_lengths, joint_size  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE_CKPT = ROOT / "data/bc_phase1/runlength.pt"
NEW_CKPT = ROOT / "data/bc_phase1/goomba_distilled_v2.pt"
OUT = ROOT / "data/phase2_goomba_v2.json"

N_EXPERT = 20_000
DEMO_REPEATS = 14          # 144 x 14 = 2,016 -> ~9% of the mixture
STEPS, LR, CHUNK_STEPS = 300, 1e-4, 100
N_EVAL = 200


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = joint_size(ctx.vocab.size)
    z = np.load(IDX)
    eidx = {k: z[k] for k in ("rows", "joints", "lengths")}
    lut = class_lengths(eidx, n_cls)
    dists = {c: eidx["lengths"][eidx["joints"] == c] for c in range(n_cls)}
    byte_of = np.array([ctx.vocab.decode_byte(c // N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    blob = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)
    dz = np.load(DEMOS)
    demo_obs, demo_y = dz["obs"], dz["y"]

    print(f"demos: {len(demo_y)} samples, {len(set(demo_y.tolist()))} classes")
    print(f"mix: expert {N_EXPERT:,} + demo {len(demo_y) * DEMO_REPEATS:,} "
          f"({len(demo_y) * DEMO_REPEATS / (N_EXPERT + len(demo_y) * DEMO_REPEATS) * 100:.1f}% demo)",
          flush=True)

    out = {"fix": ("first attempt used a 1:1 ratio, which capped the expert side at the demo count "
                   "(144) and ran 133 epochs over 288 samples"),
           "v1_result": {"goomba_rate_before": 0.65, "goomba_rate_after": 0.535,
                         "deaths_272_319_before": 69, "deaths_272_319_after": 92},
           "n_expert": N_EXPERT, "demo_repeats": DEMO_REPEATS,
           "demo_samples": int(len(demo_y)), "steps": STEPS, "lr": LR}

    if NEW_CKPT.exists():
        pol = BCPolicy(cfg)
        pol.load_state_dict(torch.load(NEW_CKPT, map_location="cpu",
                                       weights_only=False)["model_state"])
        pol.eval()
        print("checkpoint resumed", flush=True)
    else:
        class Mixed(Dataset):
            def __init__(self):
                self.base = ctx.dataset(ctx.expert_train)
                self.base.label_mode = "token"
                rng = np.random.default_rng(0)
                self.pick = rng.choice(len(eidx["rows"]),
                                       size=min(N_EXPERT, len(eidx["rows"])), replace=False)
                self.n_demo = len(demo_y) * DEMO_REPEATS

            def __len__(self):
                return len(self.pick) + self.n_demo

            def __getitem__(self, i):
                if i < len(self.pick):
                    j = int(self.pick[i])
                    obs, _prev, _tok = self.base[int(eidx["rows"][j])]
                    return obs, int(eidx["joints"][j])
                d = (i - len(self.pick)) % len(demo_y)
                return torch.from_numpy(demo_obs[d].astype(np.float32) / 255.0), int(demo_y[d])

        def coll(batch):
            return (torch.stack([b[0] for b in batch]),
                    torch.tensor([b[1] for b in batch], dtype=torch.long))

        ds = Mixed()
        epochs = STEPS * 128 / len(ds)
        print(f"training: {len(ds):,} samples, {STEPS} steps ~ {epochs:.2f} epochs, plain CE",
              flush=True)
        out["training"] = {"samples": len(ds), "epochs": round(epochs, 2),
                           "loss": "plain_cross_entropy"}
        part = NEW_CKPT.with_suffix(".partial.pt")
        pol = BCPolicy(cfg)
        pol.load_state_dict(blob["model_state"])
        done = 0
        if part.exists():
            pb = torch.load(part, map_location="cpu", weights_only=False)
            pol.load_state_dict(pb["model_state"])
            done = int(pb["step"])
            print(f"    resuming from step {done}/{STEPS}", flush=True)
        if done < STEPS:
            pol.train()
            opt = torch.optim.AdamW(pol.parameters(), lr=LR, weight_decay=1e-4)
            g = torch.Generator().manual_seed(0)
            step = done
            while step < STEPS:
                for obs, y in DataLoader(ds, batch_size=128, shuffle=True, num_workers=0,
                                         collate_fn=coll, generator=g):
                    loss = torch.nn.functional.cross_entropy(pol(obs), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                    opt.step()
                    step += 1
                    if step % CHUNK_STEPS == 0 or step >= STEPS:
                        torch.save({"model_state": pol.state_dict(), "step": step}, part)
                        print(f"    step {step}/{STEPS} loss {float(loss.detach()):.4f}",
                              flush=True)
                    if step >= STEPS:
                        break
        pol.eval()
        torch.save({"model_state": pol.state_dict(), "policy_config": cfg,
                    "loss": "plain_cross_entropy", "base": BASE_CKPT.name}, NEW_CKPT)
        part.unlink(missing_ok=True)

    print("\nevaluating, capped generation rule, n=200", flush=True)
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        tr = resumable_eval(TRACEDIR / "goomba_v2_200.json", N_EVAL,
                            lambda i: rollout(s, pol, cfg, start, i, mode="capped",
                                              lut=lut, dists=dists, byte_of=byte_of))
        base = score("capped (before)",
                     [_Ep(e) for e in json.loads(CAPPED_TRACES.read_text())["episodes"]])
        got = score("goomba-distilled v2", tr)
        rates = {k: round(got["button_marginals"]["rates"][k], 3)
                 for k in ("A", "B", "Right", "Down", "Left")}
        print(f"\nrate-matched control at v2's marginals {rates}", flush=True)
        ctl = resumable_eval(TRACEDIR / "goomba_v2_ratematched_200.json", N_EVAL,
                             lambda i: scripted_episode(s, start, i, rates))
        ctlr = score("rate-matched (v2)", ctl)
    finally:
        s.close()

    out["arms"] = {"capped_baseline": base, "distilled_v2": got, "rate_matched_v2": ctlr}
    lo, hi = diff_ci(base["goomba_cleared"], base["n"], got["goomba_cleared"], got["n"])
    dlo, dhi = diff_ci(base["deaths_272_319"], base["n"], got["deaths_272_319"], got["n"])
    out["goomba_comparison"] = {
        "capped_rate": base["goomba_rate"], "v2_rate": got["goomba_rate"],
        "delta_pp": (got["goomba_rate"] - base["goomba_rate"]) * 100,
        "ci_pp": [lo * 100, hi * 100], "improved": bool(lo > 0),
        "deaths_272_319": {"capped": base["deaths_272_319"], "v2": got["deaths_272_319"],
                           "ci_pp": [dlo * 100, dhi * 100], "reduced": bool(dhi < 0)}}
    # both bars, unconditional and conditional
    out["bars"] = {"best_script": {ob: got["vs_script"]["per_obstacle"][ob]["advantage_pp"]
                                   for ob in PIPE_THRESHOLDS}, "rate_matched_v2": {}}
    for ob in PIPE_THRESHOLDS:
        x, y = ctlr["clearance"][ob], got["clearance"][ob]
        l2, h2 = diff_ci(x["k"], x["n"], y["k"], y["n"])
        out["bars"]["rate_matched_v2"][ob] = {
            "advantage_pp": (y["rate"] - x["rate"]) * 100, "ci_pp": [l2 * 100, h2 * 100],
            "beats": bool(l2 > 0), "loses": bool(h2 < 0)}
    gc = out["goomba_comparison"]
    out["verdict"] = (
        f"CORRECTED MIX REDUCED GOOMBA DEATHS: clearance past x>{CLEAR_X} went "
        f"{base['goomba_rate'] * 100:.1f}% -> {got['goomba_rate'] * 100:.1f}% "
        f"({gc['delta_pp']:+.1f} pp [{gc['ci_pp'][0]:+.1f}, {gc['ci_pp'][1]:+.1f}]); deaths in "
        f"272-319 {base['deaths_272_319']} -> {got['deaths_272_319']}. The first attempt's failure was "
        f"the mixing bug, not the method."
        if gc["improved"] else
        f"EVEN WITH A CORRECTED MIX, DISTILLATION DID NOT REDUCE GOOMBA DEATHS: clearance past "
        f"x>{CLEAR_X} went {base['goomba_rate'] * 100:.1f}% -> {got['goomba_rate'] * 100:.1f}% "
        f"({gc['delta_pp']:+.1f} pp [{gc['ci_pp'][0]:+.1f}, {gc['ci_pp'][1]:+.1f}]); deaths in "
        f"272-319 {base['deaths_272_319']} -> {got['deaths_272_319']} of 200. 1,005 verified solutions, "
        f"a representation that can express them, a sane schedule -- and no improvement. The kill "
        f"condition fires on a fair test.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
