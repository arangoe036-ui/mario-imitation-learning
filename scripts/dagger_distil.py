"""§3c: distil the search corrections into the policy. Expert corpus + search solutions, 1,000 steps.

The labels come from **search applied to states where the policy failed** — not from its own successes, which
is the loop that failed three times and has a fixed point.

**Mixing is by SAMPLE COUNT with an absolute floor, never by ratio** (`LEDGER.md` §1): run-length encoding
compresses ~100 demo frames into ~3 samples, so a 1:1 ratio once capped the *expert* side at 144 samples,
gave 133 epochs over 288 rows, and regressed the Goomba from 65.0% to 53.5%. Exact counts are reported.

**The whole solution sequence is trained on, not only its first token** — the continuation is part of the
correction.

Pixels are needed for the labels and the solutions are stored as bytes, so each state's prefix is replayed
once, snapshotted, and every solution for that state is stepped from the snapshot. Restores are O(1), which
is what makes this affordable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.data import FrameStackDataset  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig, pick_device  # noqa: E402
from tasdata.bc.provenance import recipe  # noqa: E402
from tasdata.bc.runlength import RunLengthDataset, encode_joint, joint_size  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATES = ROOT / "data/dagger_round1_states.json"
SOLS = ROOT / "data/dagger_round1_solutions.json"
OUT = ROOT / "data/dagger_round1_distil.json"
CACHE = ROOT / "data/dagger_round1_samples.npz"
OUTDIR = ROOT / "data/bc_scaleup"

ARM = "P_84_cnn32"
MAX_SOLS_PER_STATE = 8
STEPS, BATCH, LR = 1_000, 64, 3e-4
#: absolute floor on the expert side -- never a ratio
EXPERT_FLOOR = 20_000
SEEDS = [0, 1, 2]
ARM_BUDGET_S = 40 * 60


def capture_samples(dl):
    """Replay each state's prefix once, then step every kept solution from the snapshot."""
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    sols = json.loads(SOLS.read_text())["solutions"]
    states = json.loads(STATES.read_text())
    eps = {e["seed"]: e for e in json.loads(
        (ROOT / "data/dagger_round1.partial.json").read_text())["episodes"]}
    policy, cfg, _ = G.load_ckpt(ARM)
    s_ = cfg.frame_size

    by_state = {}
    for s in sols:
        by_state.setdefault(s["state"], []).append(s)
    # prefer retreat solutions -- the corpus has 0.526% of them and they are the scarce label
    for k in by_state:
        by_state[k].sort(key=lambda r: (r["kind"] != "retreat", len(r["bytes"])))
        by_state[k] = by_state[k][:MAX_SOLS_PER_STATE]

    obs_list, lab_list, meta = [], [], []
    sess = None
    try:
        with time_limit(min(ARM_BUDGET_S, dl.remaining() - 120), "capture"):
            sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
            warm_session(sess, start.frame)
            for si, group in sorted(by_state.items()):
                if dl.remaining() < 180:
                    break
                g0 = group[0]
                ep = eps.get(g0["seed"])
                if ep is None:
                    continue
                obs = sess.reset(start.frame)
                for b in ep["bytes"][:g0["prefix_frames"]]:
                    obs = sess.step(int(b))
                sess.save_scratch(500)
                base_win = np.zeros((cfg.stack, s_, s_), np.uint8)
                base_win[:] = _resize_gray(obs.rgb, (s_, s_))
                for sol in group:
                    sess.load_scratch(500)
                    win = base_win.copy()
                    bs = sol["bytes"]
                    # run boundaries: a new run starts wherever the byte changes
                    i = 0
                    while i < len(bs):
                        j = i
                        while j < len(bs) and bs[j] == bs[i]:
                            j += 1
                        tok = int(ctx.vocab.encode(np.array([bs[i]], dtype=np.uint8))[0])
                        obs_list.append(win.copy())
                        lab_list.append(encode_joint(tok, j - i))
                        meta.append({"state": si, "kind": sol["kind"], "wall": sol["wall"]})
                        for k in range(i, j):
                            o = sess.step(int(bs[k]))
                            win = np.roll(win, -1, 0)
                            win[-1] = _resize_gray(o.rgb, (s_, s_))
                        i = j
                print(f"  {dl.stamp()} captured {len(obs_list)} correction samples "
                      f"from {len(by_state)} states", flush=True)
    except TimedOut as e:
        print(f"capture timeout: {e}", flush=True)
    finally:
        if sess is not None:
            try:
                sess.close()
            except BaseException:
                pass
    return np.asarray(obs_list, dtype=np.uint8), np.asarray(lab_list, dtype=np.int64), meta


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 90 * 60)
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        obs, lab = z["obs"], z["lab"]
        meta = list(z["meta"]) if "meta" in z else []
        print(f"reusing {len(lab)} cached correction samples", flush=True)
    else:
        obs, lab, meta = capture_samples(dl)
        np.savez_compressed(CACHE, obs=obs, lab=lab,
                            meta=np.array([json.dumps(m) for m in meta]))
        print(f"captured {len(lab)} correction samples", flush=True)
    if len(lab) == 0:
        print("no correction samples; nothing to distil")
        return

    ctx = O.Ctx()
    n_cls = joint_size(ctx.vocab.size)
    base = FrameStackDataset(ctx.expert_train, ctx.vocab, stack=4, label_mode="token")
    zz = np.load(ROOT / "data/runlength_index_runs.npz")
    idx = {k: zz[k] for k in ("rows", "joints", "lengths")}
    expert = RunLengthDataset(base, idx)

    n_new = len(lab)
    n_expert = max(EXPERT_FLOOR, 0)
    n_expert = min(n_expert, len(expert))
    out = {"arm": ARM, "steps": STEPS, "batch": BATCH, "lr": LR, "seeds": SEEDS,
           "mix": {"expert_samples": int(n_expert), "correction_samples": int(n_new),
                   "correction_share": float(n_new / (n_expert + n_new)),
                   "epochs_over_mixture": float(STEPS * BATCH / (n_expert + n_new)),
                   "rule": "SAMPLE COUNT with an absolute floor -- never a ratio (LEDGER §1)"},
           "max_solutions_per_state": MAX_SOLS_PER_STATE,
           "solution_kinds": dict(zip(*np.unique([m.get("kind", "?") if isinstance(m, dict)
                                                  else json.loads(m).get("kind", "?")
                                                  for m in meta], return_counts=True)))
           if meta else {},
           "checkpoints": []}
    print(json.dumps(out["mix"], indent=2), flush=True)

    rng = np.random.default_rng(0)
    pick = rng.choice(len(expert), size=n_expert, replace=False)
    new_obs = torch.from_numpy(obs).float().div_(255.0)
    new_lab = torch.from_numpy(lab)

    for seed in SEEDS:
        name = f"DAG1_84_cnn32_s{seed}"
        ck = OUTDIR / f"{name}.pt"
        if ck.exists():
            out["checkpoints"].append(str(ck.relative_to(ROOT)))
            continue
        torch.manual_seed(seed)
        dev = pick_device("mps")
        cfg = PolicyConfig(n_actions=n_cls, stack=4, frame_size=84, d_model=64, n_layers=1,
                           head_type="categorical", cnn_channels=(32, 64, 64))
        policy = BCPolicy(cfg).to(dev)
        opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
        policy.train()
        g = np.random.default_rng(seed)
        losses = []
        for step in range(STEPS):
            # draw the batch from the MIXTURE by sample count
            k_new = int(round(BATCH * n_new / (n_expert + n_new)))
            k_new = max(1, min(BATCH - 1, k_new))
            k_exp = BATCH - k_new
            ei = g.choice(n_expert, size=k_exp, replace=False)
            ni = g.choice(n_new, size=k_new, replace=(n_new < k_new))
            xb, yb = [], []
            for t in ei:
                o, _p, y = expert[int(pick[t])]
                xb.append(o)
                yb.append(y)
            ob = torch.stack(xb + [new_obs[int(j)] for j in ni])
            yv = torch.tensor(list(yb) + [int(new_lab[int(j)]) for j in ni], dtype=torch.long)
            ob, yv = ob.to(dev), yv.to(dev)
            loss = torch.nn.functional.cross_entropy(policy(ob), yv)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 250 == 0:
                print(f"    {name} step {step + 1}/{STEPS} loss "
                      f"{np.mean(losses[-250:]):.4f}", flush=True)
        torch.save({"model_state": {k: v.cpu() for k, v in policy.state_dict().items()},
                    "policy_config": cfg.to_dict(), "corpus": "runs",
                    "cnn_channels": [32, 64, 64], "loss": "plain cross-entropy",
                    "final_loss": float(np.mean(losses[-250:])),
                    "mix": out["mix"],
                    **recipe(steps=STEPS, batch=BATCH, seed=seed, frame_size=84)}, ck)
        out["checkpoints"].append(str(ck.relative_to(ROOT)))
        print(f"  wrote {ck.name}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, default=str))

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
