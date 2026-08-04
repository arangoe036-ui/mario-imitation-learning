"""§1: three generation rules over the *same* trained run-length policy.

The three variants change **how a predicted (combo, duration) is executed**, not how the model was trained,
so all four arms below share one checkpoint and differ in exactly one thing. No retraining is needed or
performed -- that is what makes this a screen rather than a day.

| arm | rule |
|---|---|
| `median` | baseline: hold the combo for the class's expert-median length. Open-loop for the whole run |
| `sampled` | (a) draw the length from the class's empirical expert distribution instead of its median |
| `capped` | (b) cap runs whose combo has no A at `CAP_NON_A` frames; A-containing runs uncapped |
| `interrupt` | (c) re-predict every frame; the sampled combo overrides the held one when it differs, so the duration acts as a maximum rather than a commitment |

**Why (b) is primary:** `data/idle_audit.json` shows the run-length arm's no-op runs reach **347 frames**
against the expert's longest of 53, while its median (10) and total no-op share (17.6%) match the expert
(9, 17.0%). The aggregate is right and the tail is fatal, which is exactly what a cap fixes.

**A correction carried in from that audit, and it changes how §1 should be judged.** A-onsets per 1,000
*grounded* frames is not comparable across arms with different airborne fractions: the run-length arm is
grounded 57% of the time and the per-frame arm 22%, so the same jump rate divides by very different
denominators. Per 1,000 **total** frames -- a denominator available for the expert too -- the run-length arm
sits at **20.9 against the expert's 27.5**, while the per-frame arms are at **148.0 and 116.5**, five times
too many. Both normalisations are reported here; the total-frames one is the one with an expert reference.
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
from scripts.phase1_duration import PIPE2_WINDOW, _Ep  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    A_BIT,
    a_hold_onsets,
    button_marginals,
    clearance,
    hold_stats,
)
from tasdata.bc.runlength import N_BUCKETS, class_lengths, joint_size  # noqa: E402
from tasdata.bc.script_baseline import behaviour_stats, vs_script  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "data/bc_phase1/runlength.pt"
IDX = ROOT / "data/phase1_runlength_index.npz"
TRACEDIR = ROOT / "data/traces"
OUT = ROOT / "data/phase1_variants.json"

CAP_NON_A = 4          # (b): frames. The expert's no-op run median is 9; this is deliberately tighter
N_EVAL, CAP_FRAMES, STALL = 200, 3000, 300
CHUNK = 20
EXPERT_ONSETS_PER_1K = 27.5      # data/idle_audit.json, 1-1 surface frames, total-frame denominator
EXPERT_AIRBORNE = 0.611


def rollout(session, policy, cfg, start, seed, *, mode, lut, dists, byte_of) -> EpisodeTrace:
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8)
    win[:] = _resize_gray(obs.rgb, (84, 84))
    best = since = frames = 0
    held_byte, remaining = None, 0

    def predict():
        with torch.no_grad():
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
            p = torch.softmax(lg, dim=-1).numpy()
        c = int(rng.choice(len(p), p=p / p.sum()))
        if mode == "sampled":
            pool = dists.get(c)
            length = int(rng.choice(pool)) if pool is not None and len(pool) else int(lut[c])
        else:
            length = int(lut[c])
        b = int(byte_of[c])
        if mode == "capped" and not (b & A_BIT):
            length = min(length, CAP_NON_A)
        return b, max(1, length)

    while frames < CAP_FRAMES:
        if mode == "interrupt":
            # re-predict every frame; a differing combo overrides the held one, so the predicted
            # duration is a maximum rather than a commitment
            b, length = predict()
            if held_byte is None or b != held_byte or remaining <= 0:
                held_byte, remaining = b, length
            byte = held_byte
            remaining -= 1
        else:
            if remaining <= 0:
                held_byte, remaining = predict()
            byte = held_byte
            remaining -= 1
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
    return t


def resumable(path: Path, n: int, make):
    if path.exists() and json.loads(path.read_text()).get("n_episodes") == n:
        return [_Ep(e) for e in json.loads(path.read_text())["episodes"]]
    partial = path.with_suffix(".partial.json")
    traces = [_Ep(e) for e in (json.loads(partial.read_text())["episodes"]
                               if partial.exists() else [])]
    while len(traces) < n:
        for i in range(len(traces), min(len(traces) + CHUNK, n)):
            traces.append(make(i))
        partial.write_text(json.dumps({"episodes": [t.to_dict() for t in traces]},
                                      separators=(",", ":")))
        print(f"    {path.stem}: {len(traces)}/{n}", flush=True)
    path.write_text(json.dumps(
        {"schema": "per-frame (x, y_absolute, speed_byte, buttons, player_state, grounded)",
         "n_episodes": len(traces), "episodes": [t.to_dict() for t in traces]},
        separators=(",", ":")))
    partial.unlink(missing_ok=True)
    return traces


def noop_runs(frames) -> dict:
    b = np.asarray([f[3] for f in frames], dtype=np.int64)
    out, i = [], 0
    while i < len(b):
        if b[i] == 0:
            j = i
            while j < len(b) and b[j] == 0:
                j += 1
            out.append(j - i)
            i = j
        else:
            i += 1
    if not out:
        return {"n": 0}
    a = np.asarray(out, dtype=float)
    return {"n": len(out), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def score(label: str, traces) -> dict:
    xs = [max(f[0] for f in t.frames) for t in traces]
    frames = [f for t in traces for f in t.frames]
    h2 = hold_stats([h for t in traces for h in a_hold_onsets(t.frames, PIPE2_WINDOW)])
    ha = hold_stats([h for t in traces for h in a_hold_onsets(t.frames, (0, 10 ** 9))])
    b = np.asarray([f[3] for f in frames], dtype=np.int64)
    a = (b & A_BIT) > 0
    prev = np.zeros_like(a)
    prev[1:] = a[:-1]
    beh = behaviour_stats(frames)
    row = {"label": label, "n": len(traces), "measurement_basis": "single_life",
           "x_median": float(np.median(xs)), "x_max": int(max(xs)),
           "a_hold_pipe2": h2, "a_hold_anywhere": ha,
           "a_onsets_per_1000_frames": float((a & ~prev).sum()) / len(b) * 1000,
           "expert_onsets_per_1000_frames": EXPERT_ONSETS_PER_1K,
           "noop_runs": noop_runs(frames),
           "clearance": clearance(xs), "vs_script": vs_script(xs),
           "button_marginals": button_marginals(frames), "behaviour": beh,
           "ended": {k: sum(1 for t in traces if t.ended == k) for k in ("died", "stuck")}}
    print(f"  {label:10s} >=12 {(h2['frac_ge_required'] or 0) * 100:5.1f}%  "
          f"airb {beh['airborne_fraction'] * 100:5.1f}%  "
          f"onset/1k-gr {beh['a_onsets_while_grounded_per_1000']:5.1f}  "
          f"/1k-all {row['a_onsets_per_1000_frames']:5.1f}  "
          f"noop max {row['noop_runs'].get('max', 0):5.0f}  "
          f"x_med {row['x_median']:4.0f}  "
          f"p1 {row['clearance']['pipe1']['rate'] * 100:4.1f} "
          f"p2 {row['clearance']['pipe2']['rate'] * 100:4.1f} "
          f"p3 {row['clearance']['pipe3']['rate'] * 100:4.1f} "
          f"p4 {row['clearance']['pipe4']['rate'] * 100:4.1f}", flush=True)
    return row


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = joint_size(ctx.vocab.size)
    z = np.load(IDX)
    idx = {k: z[k] for k in ("rows", "joints", "lengths")}
    lut = class_lengths(idx, n_cls)
    dists = {c: idx["lengths"][idx["joints"] == c] for c in range(n_cls)}
    byte_of = np.array([ctx.vocab.decode_byte(c // N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)
    pol = BCPolicy(cfg)
    pol.load_state_dict(blob["model_state"])
    pol.eval()
    print(f"one checkpoint, four generation rules. non-A cap = {CAP_NON_A} frames")
    print(f"expert reference: {EXPERT_ONSETS_PER_1K} A-onsets/1k frames, "
          f"airborne {EXPERT_AIRBORNE * 100:.1f}%, no-op run max 53\n", flush=True)

    out = {"checkpoint": CKPT.name, "cap_non_a": CAP_NON_A,
           "expert": {"a_onsets_per_1000_frames": EXPERT_ONSETS_PER_1K,
                      "airborne_fraction": EXPERT_AIRBORNE, "noop_run_max": 53,
                      "a_hold_pipe2": {"median": 32.0, "p90": 70.0, "max": 72}},
           "note": ("all four arms share one trained checkpoint; only the generation rule differs, "
                    "so no retraining was performed"),
           "arms": {}}

    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        # (b) first: it is the primary hypothesis
        for mode in ("capped", "sampled", "interrupt", "median"):
            print(f"[{mode}]", flush=True)
            tr = resumable(TRACEDIR / f"variant_{mode}_200.json", N_EVAL,
                           lambda i, m=mode: rollout(s, pol, cfg, start, i, mode=m,
                                                     lut=lut, dists=dists, byte_of=byte_of))
            out["arms"][mode] = score(mode, tr)
    finally:
        s.close()

    # the gate: onsets/1k grounded above ~2 while >=12 fraction stays above ~20%
    passed = {k: bool((v["behaviour"]["a_onsets_while_grounded_per_1000"] or 0) > 2.0
                      and (v["a_hold_pipe2"]["frac_ge_required"] or 0) > 0.20)
              for k, v in out["arms"].items()}
    out["gate"] = {"rule": "A-onsets/1k grounded > 2 AND pipe-2 hold fraction >=12 above 20%",
                   "passed": passed,
                   "note": ("the grounded denominator is not comparable across arms with different "
                            "airborne fractions and has no expert reference; the total-frame "
                            "denominator is reported beside it")}
    winners = [k for k, v in passed.items() if v]
    best = max(out["arms"].items(),
               key=lambda kv: abs(kv[1]["a_onsets_per_1000_frames"] - EXPERT_ONSETS_PER_1K) * -1)
    out["verdict"] = (
        f"GATE PASSED by {winners}: a generation rule raises A-onsets/1k grounded above 2 while keeping "
        f"the pipe-2 >=12 hold fraction above 20%."
        if winners else
        f"GATE NOT PASSED on its stated terms: no variant exceeds 2 A-onsets/1k *grounded* while holding "
        f">=12 above 20%. But the grounded denominator has no expert reference and is not comparable "
        f"across arms; on the total-frame denominator the closest arm is `{best[0]}` at "
        f"{best[1]['a_onsets_per_1000_frames']:.1f} against the expert's {EXPERT_ONSETS_PER_1K}.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
