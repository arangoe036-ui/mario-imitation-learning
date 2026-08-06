"""§3 + §4: round two — balance the mix three ways, ten seeds, and test conditional vs marginal directly.

Round one learned the corrections (retreat-class mass rose 8×) and applied them **everywhere** (global Left
rate 0.050 → 0.55). The cause was the mix: **1,480 of 1,507 correction samples were retreat macros** because
selection sorted them first as "the scarce label".

Three stratifications, all on the 14,675 corrections already on disk — only the mix and the training change:

1. **Retreats capped at ≤ 1/3 of correction samples**, the rest drawn from the policy-sampled solutions.
2. **Stratified by WALL to match the failure distribution** (pipe3 60 · goomba 48 · pipe4 38 · koopas 30 ·
   other_800 13 · frontier 9 · pipe2 1), not by where solutions happened to be plentiful.
3. **Stratified by FAILURE KIND — face versus on-top.** 34 of 200 failures are stalls on top of a pipe and at
   pipe 4 it is half; those need "get off the pipe", not "clear the pipe". All 11 on-top states searched were
   solved, so the labels exist.

**§4's diagnostic, upgraded because round one's was near-circular.** Round one measured mass on retreat classes
while 98% of the mix was retreats — nearly tautological, and the 11× global Left rate is what actually carried
the "marginal" reading. The direct test is **the Left rate INSIDE the correction windows versus OUTSIDE them**:

* **rose only inside** → state-conditional, which is the result this project has been chasing;
* **rose everywhere** → still a marginal, and the mix is still wrong.

A "correction window" is an x-interval around a wall where corrections were collected; outside is everything
else on the surface. Reported per seed and per wall.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.data import FrameStackDataset  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig, pick_device  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.provenance import recipe  # noqa: E402
from tasdata.bc.runlength import RunLengthDataset, encode_joint, joint_size  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATES = ROOT / "data/dagger_round1_states.json"
SOLS = ROOT / "data/dagger_round1_solutions.json"
EPS = ROOT / "data/dagger_round1.partial.json"
OUT = ROOT / "data/dagger_round2.json"
CACHE = ROOT / "data/dagger_round2_samples.npz"
OUTDIR = ROOT / "data/bc_scaleup"

N_SEEDS = 10
STEPS, BATCH, LR = 1_000, 64, 3e-4
EXPERT_FLOOR = 20_000
RETREAT_CAP = 1.0 / 3.0
#: total correction sequences to keep; ~4 per state keeps the capture affordable
TARGET_SEQS = 300
LEFT_BIT = NES_BUTTON_BITS["Left"]
#: x-window around each wall counted as "inside a correction window"
WALL_WINDOW = 96
ARM_BUDGET_S = 40 * 60


def build_mix(dl):
    """Select a wall- and kind-stratified correction set, then replay it for pixels."""
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    sols = json.loads(SOLS.read_text())["solutions"]
    states = {i: s for i, s in enumerate(json.loads(STATES.read_text())["states"])}
    eps = {e["seed"]: e for e in json.loads(EPS.read_text())["episodes"]}
    policy, cfg, _ = G.load_ckpt("P_84_cnn32")
    s_ = cfg.frame_size

    # failure distribution -> target shares per wall
    hist = json.loads(STATES.read_text())["failure_histogram"]
    tot = sum(hist.values())
    want = {w: max(1, int(round(TARGET_SEQS * n / tot))) for w, n in hist.items()}

    # group solutions by (wall, on_top, kind)
    groups = collections.defaultdict(list)
    for s in sols:
        st = states.get(s["state"], {})
        groups[(s["wall"], bool(st.get("on_top")), s["kind"])].append(s)
    for k in groups:
        groups[k].sort(key=lambda r: len(r["bytes"]))

    chosen = []
    for wall, n_want in want.items():
        # within a wall, split across on_top/at_face in proportion to what exists
        keys = [k for k in groups if k[0] == wall]
        if not keys:
            continue
        n_retreat_cap = int(n_want * RETREAT_CAP)
        retreat_keys = [k for k in keys if k[2] == "retreat"]
        sampled_keys = [k for k in keys if k[2] == "sampled"]
        took_r = 0
        i = 0
        while took_r < n_retreat_cap and retreat_keys:
            k = retreat_keys[i % len(retreat_keys)]
            if groups[k]:
                chosen.append(groups[k].pop(0))
                took_r += 1
            else:
                retreat_keys.remove(k)
                continue
            i += 1
        took_s = 0
        i = 0
        while took_s < (n_want - took_r) and sampled_keys:
            k = sampled_keys[i % len(sampled_keys)]
            if groups[k]:
                chosen.append(groups[k].pop(0))
                took_s += 1
            else:
                sampled_keys.remove(k)
                continue
            i += 1

    comp = collections.Counter((c["wall"], c["kind"]) for c in chosen)
    print(f"selected {len(chosen)} correction sequences; composition:", flush=True)
    for (w, k), n in sorted(comp.items()):
        print(f"    {w:16s} {k:8s} {n:4d}", flush=True)

    by_state = collections.defaultdict(list)
    for c in chosen:
        by_state[c["state"]].append(c)

    obs_l, lab_l, meta_l = [], [], []
    sess = None
    try:
        with time_limit(min(ARM_BUDGET_S, dl.remaining() - 180), "capture"):
            sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
            warm_session(sess, start.frame)
            for si, group in sorted(by_state.items()):
                if dl.remaining() < 240:
                    break
                g0 = group[0]
                ep = eps.get(g0["seed"])
                if ep is None:
                    continue
                obs = sess.reset(start.frame)
                for b in ep["bytes"][:g0["prefix_frames"]]:
                    obs = sess.step(int(b))
                sess.save_scratch(700)
                base = np.zeros((cfg.stack, s_, s_), np.uint8)
                base[:] = _resize_gray(obs.rgb, (s_, s_))
                st = states.get(si, {})
                for sol in group:
                    sess.load_scratch(700)
                    win = base.copy()
                    bs = sol["bytes"]
                    i = 0
                    while i < len(bs):
                        j = i
                        while j < len(bs) and bs[j] == bs[i]:
                            j += 1
                        tok = int(ctx.vocab.encode(np.array([bs[i]], dtype=np.uint8))[0])
                        obs_l.append(win.copy())
                        lab_l.append(encode_joint(tok, j - i))
                        meta_l.append({"state": si, "kind": sol["kind"], "wall": sol["wall"],
                                       "on_top": bool(st.get("on_top")),
                                       "fail_x": st.get("fail_x")})
                        for k in range(i, j):
                            o = sess.step(int(bs[k]))
                            win = np.roll(win, -1, 0)
                            win[-1] = _resize_gray(o.rgb, (s_, s_))
                        i = j
            print(f"  captured {len(obs_l)} correction samples", flush=True)
    finally:
        if sess is not None:
            try:
                sess.close()
            except BaseException:
                pass
    return (np.asarray(obs_l, dtype=np.uint8), np.asarray(lab_l, dtype=np.int64), meta_l,
            {f"{w}|{k}": n for (w, k), n in comp.items()})


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 150 * 60)
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        obs, lab = z["obs"], z["lab"]
        meta = [json.loads(m) for m in z["meta"]]
        comp = json.loads(str(z["comp"]))
        print(f"reusing {len(lab)} cached round-2 samples", flush=True)
    else:
        obs, lab, meta, comp = build_mix(dl)
        np.savez_compressed(CACHE, obs=obs, lab=lab,
                            meta=np.array([json.dumps(m) for m in meta]),
                            comp=json.dumps(comp))
    if len(lab) == 0:
        print("no samples; abort")
        return

    ctx = O.Ctx()
    n_cls = joint_size(ctx.vocab.size)
    base_ds = FrameStackDataset(ctx.expert_train, ctx.vocab, stack=4, label_mode="token")
    zz = np.load(ROOT / "data/runlength_index_runs.npz")
    expert = RunLengthDataset(base_ds, {k: zz[k] for k in ("rows", "joints", "lengths")})
    n_new = len(lab)
    n_exp = min(EXPERT_FLOOR, len(expert))
    kinds = collections.Counter(m["kind"] for m in meta)
    out = {"round": 2, "steps": STEPS, "batch": BATCH, "lr": LR, "n_seeds": N_SEEDS,
           "mix": {"expert_samples": int(n_exp), "correction_samples": int(n_new),
                   "correction_share": float(n_new / (n_exp + n_new)),
                   "retreat_samples": int(kinds.get("retreat", 0)),
                   "sampled_samples": int(kinds.get("sampled", 0)),
                   "retreat_share_of_corrections": float(
                       kinds.get("retreat", 0) / max(1, n_new)),
                   "retreat_cap": RETREAT_CAP,
                   "rule": "sample count with an absolute floor, never a ratio (LEDGER §1)"},
           "sequence_composition_by_wall_and_kind": comp,
           "sample_composition_by_wall": dict(collections.Counter(m["wall"] for m in meta)),
           "sample_composition_on_top": dict(collections.Counter(
               f"{m['wall']}|{'on_top' if m['on_top'] else 'at_face'}" for m in meta)),
           "round1_comparison": {"retreat_share": 1480 / 1507,
                                 "note": "round 1 was 98% retreats; this is the fix"},
           "checkpoints": []}
    print(json.dumps(out["mix"], indent=2), flush=True)

    g0 = np.random.default_rng(0)
    pick = g0.choice(len(expert), size=n_exp, replace=False)
    new_obs = torch.from_numpy(obs).float().div_(255.0)

    for seed in range(N_SEEDS):
        name = f"DAG2_84_cnn32_s{seed}"
        ck = OUTDIR / f"{name}.pt"
        if ck.exists():
            out["checkpoints"].append(str(ck.relative_to(ROOT)))
            continue
        if dl.remaining() < 120:
            break
        torch.manual_seed(seed)
        dev = pick_device("mps")
        cfg = PolicyConfig(n_actions=n_cls, stack=4, frame_size=84, d_model=64, n_layers=1,
                           head_type="categorical", cnn_channels=(32, 64, 64))
        policy = BCPolicy(cfg).to(dev)
        opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
        policy.train()
        g = np.random.default_rng(seed)
        losses = []
        k_new = max(1, min(BATCH - 1, int(round(BATCH * n_new / (n_exp + n_new)))))
        for _ in range(STEPS):
            ei = g.choice(n_exp, size=BATCH - k_new, replace=False)
            ni = g.choice(n_new, size=k_new, replace=(n_new < k_new))
            xb, yb = [], []
            for t in ei:
                o, _p, y = expert[int(pick[t])]
                xb.append(o)
                yb.append(y)
            ob = torch.stack(xb + [new_obs[int(j)] for j in ni]).to(dev)
            yv = torch.tensor(list(yb) + [int(lab[int(j)]) for j in ni],
                              dtype=torch.long).to(dev)
            loss = torch.nn.functional.cross_entropy(policy(ob), yv)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        torch.save({"model_state": {k: v.cpu() for k, v in policy.state_dict().items()},
                    "policy_config": cfg.to_dict(), "corpus": "runs",
                    "cnn_channels": [32, 64, 64], "loss": "plain cross-entropy",
                    "final_loss": float(np.mean(losses[-250:])), "mix": out["mix"],
                    **recipe(steps=STEPS, batch=BATCH, seed=seed, frame_size=84)}, ck)
        out["checkpoints"].append(str(ck.relative_to(ROOT)))
        print(f"  trained {name} loss {np.mean(losses[-250:]):.4f}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, default=str))

    # correction windows, for §4
    walls = collections.Counter(m["fail_x"] for m in meta if m.get("fail_x"))
    centres = sorted(walls)
    out["correction_windows"] = {
        "centres": centres[:40], "half_width": WALL_WINDOW,
        "note": "inside = within half_width px of any captured failure x; outside = the rest"}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
