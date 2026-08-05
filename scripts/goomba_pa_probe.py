"""§1: at x=272-304, does the policy assign the same p(A) in the deaths that never jumped as in clearers?

The distinction the answer decides:

* **p(A) similar in both** -- the policy assigns middling probability and sampling picks. It knows and
  gambles, and **more demonstrations will not help**; the lever is the sampling or generation rule.
* **p(A) lower in the deaths** -- it fails to recognise the state, and **targeted demonstrations from those
  exact states are the right fix.**

**One correction to the request: this cannot be forward passes alone.** The retained traces hold
`(x, y_absolute, speed, buttons, player_state, grounded)` -- RAM-derived fields, not the 84x84 observations
the network consumes. Reconstructing the frame stacks means replaying each episode's recorded byte prefix in
the emulator, which is exactly deterministic (132,844/132,844 frames verified earlier) but is not free.
~164 episodes replayed to x≈272.

**p(A) for a run-length head is not one logit.** The head is a 300-way softmax over (combo, length-bucket)
classes, so p(A) is the summed probability of every class whose combo contains A. That is the quantity the
sampler draws against.

**A caveat stated up front:** the policy only *decides* at run boundaries, so p(A) at an arbitrary frame is
its assessment of the state, not necessarily a choice it acted on. Both are reported -- all frames in the
window, and the subset where a new run actually began.
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
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.runlength import N_BUCKETS, joint_size  # noqa: E402
from tasdata.ram import on_ground, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "data/traces/variant_capped_200.json"
CKPT = ROOT / "data/bc_phase1/runlength.pt"
OUT = ROOT / __import__("os").environ.get("PA_OUT","data/goomba_pa_probe.json")
PARTIAL = ROOT / __import__("os").environ.get("PA_PARTIAL","data/goomba_pa_probe.partial.json")

WIN = tuple(int(v) for v in (__import__("os").environ.get("PA_WIN","272,304").split(",")))
DEATH_BAND = (272, 320)
CLEAR_X = 320
CHUNK = 15


def stats(v) -> dict:
    a = np.asarray(list(v), dtype=float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "min": float(a.min())}


def classify(eps) -> dict:
    """Split into the three groups the forensics identified."""
    def onsets(fr, lo, hi):
        out, prev = [], False
        for f in fr:
            a = bool(f[3] & A_BIT)
            if a and not prev and lo <= f[0] <= hi:
                out.append(f[0])
            prev = a
        return out

    groups = {"death_no_jump": [], "death_jumped": [], "cleared": []}
    for e in eps:
        mx = max(f[0] for f in e["frames"])
        died_band = bool(e.get("death") and DEATH_BAND[0] <= e["death"]["x"] < DEATH_BAND[1])
        if mx > CLEAR_X:
            groups["cleared"].append(e)
        elif died_band:
            (groups["death_jumped"] if onsets(e["frames"], 260, 320)
             else groups["death_no_jump"]).append(e)
    return groups


def probe_episode(session, policy, cfg, start, e, a_mask) -> dict:
    """Replay the episode's recorded bytes; at every frame in WIN, read the head's p(A)."""
    bytes_ = [f[3] for f in e["frames"]]
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8)
    win[:] = _resize_gray(obs.rgb, (84, 84))
    pa_all, pa_grounded, pa_decision = [], [], []
    prev_byte = None
    for i, byte in enumerate(bytes_):
        st = read_smb(obs.ram, obs.framecount)
        x = st.x_position
        if WIN[0] <= x <= WIN[1]:
            with torch.no_grad():
                p = torch.softmax(policy(
                    torch.from_numpy(win[None]).float().div_(255.0))[0], dim=-1).numpy()
            pa = float(p[a_mask].sum())
            pa_all.append(pa)
            if on_ground(obs.ram):
                pa_grounded.append(pa)
            if prev_byte is None or byte != prev_byte:
                pa_decision.append(pa)
        prev_byte = byte
        obs = session.step(byte)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (84, 84))
        if read_smb(obs.ram, obs.framecount).x_position > WIN[1] + 24:
            break
    return {"seed": e["seed"], "n_frames": len(pa_all),
            "pa_all": pa_all, "pa_grounded": pa_grounded, "pa_decision": pa_decision}


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = joint_size(ctx.vocab.size)
    a_mask = np.array([(ctx.vocab.decode_byte(c // N_BUCKETS) & A_BIT) > 0
                       for c in range(n_cls)])
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()

    eps = json.loads(TRACES.read_text())["episodes"]
    groups = classify(eps)
    print(f"groups: death_no_jump {len(groups['death_no_jump'])}, "
          f"death_jumped {len(groups['death_jumped'])}, cleared {len(groups['cleared'])}")
    print(f"A-containing classes: {int(a_mask.sum())} of {n_cls}\n", flush=True)

    done = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    todo = [(g, e) for g in ("death_no_jump", "death_jumped", "cleared") for e in groups[g]
            if f"{g}:{e['seed']}" not in done]
    if todo:
        s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        try:
            for k, (g, e) in enumerate(todo):
                r = probe_episode(s, policy, cfg, start, e, a_mask)
                done[f"{g}:{e['seed']}"] = {"group": g, **r}
                if (k + 1) % CHUNK == 0 or k == len(todo) - 1:
                    PARTIAL.write_text(json.dumps(done, separators=(",", ":")))
                    print(f"    probed {len(done)}/{len(todo) + len(done) - (k + 1)}"
                          f" ... {k + 1}/{len(todo)} this pass", flush=True)
        finally:
            s.close()

    out = {"window_x": list(WIN), "checkpoint": CKPT.name,
           "note": ("p(A) is the summed softmax over classes whose combo contains A; the head is a "
                    "300-way (combo, length-bucket) softmax, not a per-button logit"),
           "caveat": ("the policy decides only at run boundaries, so pa_all is its assessment of the "
                      "state and pa_decision is the subset where a new run actually began"),
           "reconstruction": ("frame stacks rebuilt by replaying each episode's recorded bytes; the "
                              "traces store RAM fields, not observations, so this needed the emulator"),
           "groups": {}}
    for g in ("death_no_jump", "death_jumped", "cleared"):
        rows = [v for v in done.values() if v["group"] == g]
        out["groups"][g] = {
            "n_episodes": len(rows),
            "pa_all": stats([p for r in rows for p in r["pa_all"]]),
            "pa_grounded": stats([p for r in rows for p in r["pa_grounded"]]),
            "pa_decision": stats([p for r in rows for p in r["pa_decision"]]),
            "per_episode_mean_pa": stats([float(np.mean(r["pa_all"])) for r in rows
                                          if r["pa_all"]]),
        }
        a = out["groups"][g]
        print(f"{g:16s} eps {a['n_episodes']:3d} | all frames p(A) median "
              f"{a['pa_all'].get('median')} p99 {a['pa_all'].get('p99')} "
              f"max {a['pa_all'].get('max')} | grounded median {a['pa_grounded'].get('median')} "
              f"| at decisions median {a['pa_decision'].get('median')}", flush=True)

    d = out["groups"]["death_no_jump"]["pa_grounded"]
    c = out["groups"]["cleared"]["pa_grounded"]
    gap = (c.get("median") or 0) - (d.get("median") or 0)
    # a fraction-based comparison too: how often is p(A) above 0.5 while grounded?
    def frac_above(g, thr=0.5):
        rows = [v for v in done.values() if v["group"] == g]
        vals = [p for r in rows for p in r["pa_grounded"]]
        return (sum(1 for v in vals if v > thr), len(vals))
    kd, nd = frac_above("death_no_jump")
    kc, nc = frac_above("cleared")
    lo, hi = diff_ci(kd, nd, kc, nc) if nd and nc else (0, 0)
    out["comparison"] = {
        "grounded_median_death_no_jump": d.get("median"),
        "grounded_median_cleared": c.get("median"),
        "median_gap": gap,
        "frac_pa_above_0.5_grounded": {
            "death_no_jump": {"k": kd, "n": nd, "rate": kd / nd if nd else None},
            "cleared": {"k": kc, "n": nc, "rate": kc / nc if nc else None},
            "difference_pp": ((kc / nc) - (kd / nd)) * 100 if nd and nc else None,
            "ci_pp": [lo * 100, hi * 100]},
    }
    similar = abs(gap) < 0.10
    out["verdict"] = (
        f"IT KNOWS AND GAMBLES: while grounded in x {WIN[0]}-{WIN[1]}, p(A) median is "
        f"{d.get('median')} in the deaths that never jumped and {c.get('median')} in the clearers, a gap "
        f"of {gap:+.3f}. The policy assigns similar probability in both and the sampler decides. **More "
        f"demonstrations will not fix this**; the lever is the sampling or generation rule."
        if similar else
        f"IT FAILS TO RECOGNISE THE STATE: while grounded in x {WIN[0]}-{WIN[1]}, p(A) median is "
        f"{d.get('median')} in the deaths that never jumped against {c.get('median')} in the clearers, a "
        f"gap of {gap:+.3f}. The policy discriminates these states and gets them wrong, so targeted "
        f"demonstrations from exactly those states are the right fix.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
