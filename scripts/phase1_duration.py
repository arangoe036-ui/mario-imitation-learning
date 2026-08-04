"""Phase 1: can a duration-aware action space produce expert-like holds?

**The one binary question:** trained on expert data alone, is a run-length policy's A-hold length
distribution closer to the expert's than a per-frame policy's is?

Two arms, **identical trunk, identical optimisation budget, expert data only, no press-weighted loss** --
differing only in the action space:

* **`runlength`** -- joint (combo, length-bucket) classes, categorical head, unweighted cross-entropy.
  Sampled from the softmax, never argmax: the categorical head's recorded failure was vote-splitting under
  argmax, where the four A-containing tokens each lost to Right+B and A was emitted on 0.03% of frames.
* **`perframe`** -- 8 independent Bernoulli buttons, unweighted BCE. The control.

**The reference is measured here, not cited.** The expert's A-hold at pipe 2 is recomputed from the expert
corpus with the same windowed-onset function used on the policies, so all three numbers come from one
definition. The ledger's "median 30, p90 66, max 72" came from a different segmentation and is quoted only
for comparison.

Clearance is secondary in this block, but `vs_script` is reported anyway because a clearance figure without
it is not interpretable.

**Kill condition, pre-committed:** if the run-length policy's A-hold distribution at pipe 2 is no closer to
the expert's than the per-frame policy's is, the representation is not the constraint.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    a_hold_onsets,
    button_marginals,
    clearance,
    hold_stats,
)
from tasdata.bc.runlength import (  # noqa: E402
    N_BUCKETS,
    RunLengthDataset,
    build_index,
    class_lengths,
    collate,
    joint_size,
)
from tasdata.bc.script_baseline import behaviour_stats, vs_script  # noqa: E402
from tasdata.bc.tokens import LIVE_MASK  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace, write_traces  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.ram import column, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/phase1_duration.json"
TRACEDIR = ROOT / "data/traces"
CKPTS = ROOT / "data/bc_phase1"
A_BIT = NES_BUTTON_BITS["A"]

PIPE2_WINDOW = (575, 640)      # the window the expert reference is quoted for
STEPS, LR, BATCH = 3000, 3e-4, 128
N_EVAL, CAP, STALL = 200, 3000, 300
CHUNK_EPISODES, CHUNK_STEPS = 20, 250


# --------------------------------------------------------------------- expert reference

def expert_pipe2_holds(ctx) -> dict:
    """The expert's A-hold lengths whose onset falls in the pipe-2 window, on 1-1 surface frames.

    Recomputed rather than cited: the archived figure used a jump segmentation that was later declared
    broken, and the policies are measured with `a_hold_onsets`. One definition for all three arms.
    """
    holds = []
    for run in ctx.expert_train:
        tr = np.asarray(run.trace)
        w, s = column(tr, "world"), column(tr, "stage")
        x, st, pg = column(tr, "x_position"), column(tr, "player_state"), column(tr, "pregame")
        a = np.asarray(run.actions, dtype=np.uint8)
        n = min(len(x), len(a))
        m = (w[:n] == 1) & (s[:n] == 1) & (pg[:n] == 1) & (st[:n] == 8)
        if not m.any():
            continue
        frames = [(int(x[i]), 0, 0, int(a[i]), 8) for i in range(n) if m[i]]
        holds.extend(a_hold_onsets(frames, PIPE2_WINDOW))
    return hold_stats(holds)


# --------------------------------------------------------------------- training

def train_categorical(policy, ds, steps, lr, seed, ckpt_partial: Path):
    from torch.utils.data import DataLoader
    done = 0
    if ckpt_partial.exists():
        blob = torch.load(ckpt_partial, map_location="cpu", weights_only=False)
        policy.load_state_dict(blob["model_state"])
        done = int(blob["step"])
        print(f"    resuming from step {done}/{steps}", flush=True)
    if done >= steps:
        return policy
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                        collate_fn=collate, generator=g)
    step = done
    while step < steps:
        for obs, prev, y in loader:
            loss = torch.nn.functional.cross_entropy(policy(obs, prev), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if step % CHUNK_STEPS == 0 or step >= steps:
                torch.save({"model_state": policy.state_dict(), "step": step}, ckpt_partial)
                print(f"    step {step}/{steps} loss {float(loss.detach()):.4f}", flush=True)
            if step >= steps:
                break
    policy.eval()
    return policy


def train_bernoulli(policy, ds, steps, lr, seed, ckpt_partial: Path):
    from tasdata.bc.train import make_loader
    done = 0
    if ckpt_partial.exists():
        blob = torch.load(ckpt_partial, map_location="cpu", weights_only=False)
        policy.load_state_dict(blob["model_state"])
        done = int(blob["step"])
        print(f"    resuming from step {done}/{steps}", flush=True)
    if done >= steps:
        return policy
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    step = done
    while step < steps:
        for obs, _prev, bits, _onset in make_loader(ds, batch_size=BATCH, shuffle=True,
                                                    num_workers=0, seed=seed):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(policy(obs), bits.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if step % CHUNK_STEPS == 0 or step >= steps:
                torch.save({"model_state": policy.state_dict(), "step": step}, ckpt_partial)
                print(f"    step {step}/{steps} loss {float(loss.detach()):.4f}", flush=True)
            if step >= steps:
                break
    policy.eval()
    return policy


# --------------------------------------------------------------------- live rollouts

def episode_runlength(session, policy, cfg, start, seed, lut) -> EpisodeTrace:
    """Sample (combo, duration) and hold that combo for the sampled number of frames."""
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8)
    win[:] = _resize_gray(obs.rgb, (84, 84))
    best = since = frames = 0
    byte_of = policy._byte_of                       # set by the caller
    while frames < CAP:
        with torch.no_grad():
            logits = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
            p = torch.softmax(logits, dim=-1).numpy()
        cls = int(rng.choice(len(p), p=p / p.sum()))
        byte = byte_of[cls] & LIVE_MASK
        hold = int(lut[cls])
        for _ in range(hold):
            obs = session.step(byte)
            win = np.roll(win, -1, 0)
            win[-1] = _resize_gray(obs.rgb, (84, 84))
            t.record(obs, byte)
            frames += 1
            st = read_smb(obs.ram, obs.framecount)
            if st.player_state in (0x06, 0x0B):
                t.record_death(obs)
                return t
            if st.x_position > best:
                best, since = st.x_position, 0
            else:
                since += 1
                if since > STALL:
                    t.ended = "stuck"
                    return t
            if frames >= CAP:
                break
    return t


def episode_perframe(session, policy, cfg, start, seed) -> EpisodeTrace:
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8)
    win[:] = _resize_gray(obs.rgb, (84, 84))
    best = since = 0
    for _ in range(CAP):
        with torch.no_grad():
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0 / (1.0 + np.exp(-lg))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]:
                byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        obs = session.step(byte)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (84, 84))
        t.record(obs, byte)
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in (0x06, 0x0B):
            t.record_death(obs)
            break
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                t.ended = "stuck"
                break
    return t


class _Ep:
    def __init__(self, e):
        self.raw, self.frames, self.ended = e, e["frames"], e.get("ended", "budget")

    def to_dict(self):
        return self.raw


def resumable(path: Path, n: int, make, **meta):
    """Bank every CHUNK_EPISODES; the environment kills long jobs every few minutes."""
    if path.exists() and json.loads(path.read_text()).get("n_episodes") == n:
        return [_Ep(e) for e in json.loads(path.read_text())["episodes"]]
    partial = path.with_suffix(".partial.json")
    done = json.loads(partial.read_text())["episodes"] if partial.exists() else []
    traces = [_Ep(e) for e in done]
    while len(traces) < n:
        for i in range(len(traces), min(len(traces) + CHUNK_EPISODES, n)):
            traces.append(make(i))
        partial.write_text(json.dumps(
            {"episodes": [t.to_dict() for t in traces]}, separators=(",", ":")))
        print(f"    {path.stem}: {len(traces)}/{n} banked", flush=True)
    path.write_text(json.dumps(
        {"schema": "per-frame (x, y_absolute, speed_byte, buttons, player_state, grounded)",
         "n_episodes": len(traces), **meta,
         "episodes": [t.to_dict() for t in traces]}, separators=(",", ":")))
    partial.unlink(missing_ok=True)
    return traces


def score(label: str, traces, loss: str) -> dict:
    xs = [max(f[0] for f in t.frames) for t in traces]
    frames = [f for t in traces for f in t.frames]
    holds_p2 = [h for t in traces for h in a_hold_onsets(t.frames, PIPE2_WINDOW)]
    all_holds = [h for t in traces for h in a_hold_onsets(t.frames, (0, 10 ** 9))]
    row = {"label": label, "loss": loss, "n": len(traces), "measurement_basis": "single_life",
           "x_median": float(np.median(xs)), "x_max": int(max(xs)),
           "a_hold_pipe2": hold_stats(holds_p2), "a_hold_anywhere": hold_stats(all_holds),
           "clearance": clearance(xs), "vs_script": vs_script(xs),
           "button_marginals": button_marginals(frames), "behaviour": behaviour_stats(frames)}
    h = row["a_hold_pipe2"]
    print(f"  {label:10s} pipe2 A-hold: n={h['n']:5d} median {h['median']} p90 {h['p90']} "
          f"max {h['max']} | x_med {row['x_median']:.0f} "
          f"A {row['button_marginals']['rates']['A']:.3f}", flush=True)
    return row


def main() -> None:
    t0 = time.time()
    CKPTS.mkdir(parents=True, exist_ok=True)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    ref_path = ROOT / "data/phase1_expert_reference.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
    else:
        ref = expert_pipe2_holds(ctx)
        ref_path.write_text(json.dumps(ref, indent=2, default=str))
    print(f"EXPERT A-hold at pipe 2 {PIPE2_WINDOW}, recomputed: n={ref['n']} "
          f"median {ref['median']} p90 {ref['p90']} max {ref['max']}")
    print("  (archived figure, different segmentation: median 30 p90 66 max 72)\n", flush=True)

    tok_ds = ctx.dataset(ctx.expert_train)          # label_mode='buttons' by default
    tok_ds.label_mode = "token"                     # switch labels; frames path is identical
    # `build_index` walks ~1M rows in Python. That is fine once, but the environment restarts this job
    # every few minutes and rebuilding it consumed most of each cycle, so training barely advanced.
    # It is a pure function of the run set, so it is cached.
    icache = ROOT / "data/phase1_runlength_index.npz"
    if icache.exists():
        z = np.load(icache)
        idx = {k: z[k] for k in ("rows", "joints", "lengths")}
    else:
        idx = build_index(tok_ds)
        np.savez_compressed(icache, **idx)
    rl_ds = RunLengthDataset(tok_ds, idx)
    n_cls = joint_size(ctx.vocab.size)
    lut = class_lengths(idx, n_cls)
    # `decode_byte` already applies LIVE_MASK, which is what the emulator should receive
    byte_of = np.array([ctx.vocab.decode_byte(c // N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    print(f"run-length dataset: {len(rl_ds):,} run samples from {len(tok_ds):,} frames "
          f"({len(rl_ds) / len(tok_ds) * 100:.1f}%), {n_cls} joint classes "
          f"({ctx.vocab.size} combos x {N_BUCKETS} buckets)", flush=True)
    print(f"  classes seen in expert data: {len(set(idx['joints'].tolist()))}\n", flush=True)

    out = {"expert_reference_pipe2": ref, "pipe2_window": list(PIPE2_WINDOW),
           "archived_expert_figure": {"median": 30, "p90": 66, "max": 72,
                                      "note": "different segmentation; quoted, not used"},
           "steps": STEPS, "lr": LR, "batch": BATCH, "data": "expert only",
           "run_samples": len(rl_ds), "frames": len(tok_ds), "joint_classes": n_cls,
           "arms": {}}

    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        # ---- arm 1: run-length ------------------------------------------------------------
        print("[runlength] categorical over (combo, duration), unweighted CE", flush=True)
        cfg_rl = PolicyConfig(n_actions=n_cls, stack=4, head_type="categorical")
        ck = CKPTS / "runlength.pt"
        if ck.exists():
            pol = BCPolicy(cfg_rl)
            pol.load_state_dict(torch.load(ck, map_location="cpu",
                                           weights_only=False)["model_state"])
            pol.eval()
        else:
            torch.manual_seed(0)
            pol = train_categorical(BCPolicy(cfg_rl), rl_ds, STEPS, LR, 0,
                                    CKPTS / "runlength.partial.pt")
            torch.save({"model_state": pol.state_dict(), "policy_config": cfg_rl,
                        "loss": "cross_entropy", "lut": lut.tolist()}, ck)
            (CKPTS / "runlength.partial.pt").unlink(missing_ok=True)
        pol._byte_of = byte_of
        tr = resumable(TRACEDIR / "phase1_runlength_200.json", N_EVAL,
                       lambda i: episode_runlength(s, pol, cfg_rl, start, i, lut),
                       arm="runlength", loss="cross_entropy")
        out["arms"]["runlength"] = score("runlength", tr, "cross_entropy")

        # ---- arm 2: per-frame control ------------------------------------------------------
        print("\n[perframe] 8 Bernoulli buttons, unweighted BCE -- same trunk, same steps",
              flush=True)
        bern_ds = ctx.dataset(ctx.expert_train)     # label_mode='buttons'
        cfg_pf = PolicyConfig(n_actions=ctx.vocab.size, stack=4, head_type="bernoulli")
        ck2 = CKPTS / "perframe.pt"
        if ck2.exists():
            pol2 = BCPolicy(cfg_pf)
            pol2.load_state_dict(torch.load(ck2, map_location="cpu",
                                            weights_only=False)["model_state"])
            pol2.eval()
        else:
            torch.manual_seed(0)
            pol2 = train_bernoulli(BCPolicy(cfg_pf), bern_ds, STEPS, LR, 0,
                                   CKPTS / "perframe.partial.pt")
            torch.save({"model_state": pol2.state_dict(), "policy_config": cfg_pf,
                        "loss": "plain_BCE"}, ck2)
            (CKPTS / "perframe.partial.pt").unlink(missing_ok=True)
        tr2 = resumable(TRACEDIR / "phase1_perframe_200.json", N_EVAL,
                        lambda i: episode_perframe(s, pol2, cfg_pf, start, i),
                        arm="perframe", loss="plain_BCE")
        out["arms"]["perframe"] = score("perframe", tr2, "plain_BCE")
    finally:
        s.close()

    # ---- the one question ---------------------------------------------------------------
    e_med = ref["median"]
    rl = out["arms"]["runlength"]["a_hold_pipe2"]
    pf = out["arms"]["perframe"]["a_hold_pipe2"]
    d_rl = abs((rl["median"] or 0) - e_med)
    d_pf = abs((pf["median"] or 0) - e_med)
    closer = d_rl < d_pf
    out["comparison"] = {
        "expert_median": e_med,
        "runlength_median": rl["median"], "perframe_median": pf["median"],
        "abs_gap_runlength": d_rl, "abs_gap_perframe": d_pf,
        "runlength_closer": bool(closer),
        "p90": {"expert": ref["p90"], "runlength": rl["p90"], "perframe": pf["p90"]},
        "frac_ge_12": {"expert": ref["frac_ge_required"], "runlength": rl["frac_ge_required"],
                       "perframe": pf["frac_ge_required"]},
    }
    out["verdict"] = (
        f"THE REPRESENTATION IS THE CONSTRAINT AND IT IS REMOVABLE: at pipe 2 the run-length policy's "
        f"A-hold median is {rl['median']} against the expert's {e_med}, versus the per-frame policy's "
        f"{pf['median']} — a gap of {d_rl:.1f} against {d_pf:.1f} frames. A duration-aware action space "
        f"produces expert-like holds where per-frame independence cannot."
        if closer else
        f"THE REPRESENTATION IS NOT THE CONSTRAINT: at pipe 2 the run-length policy's A-hold median is "
        f"{rl['median']} against the expert's {e_med}, and the per-frame policy's is {pf['median']} — a "
        f"gap of {d_rl:.1f} against {d_pf:.1f} frames, so the duration-aware action space is no closer. "
        f"Phase 2 proceeds on the existing action space with this question closed rather than assumed.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
