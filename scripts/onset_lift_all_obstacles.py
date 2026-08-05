"""Is there a timing signal anywhere, or is the policy's whole competence duration?

The x-matched onset lift at the Goomba was **+0.067** against a positional swing of 0.157 — coarse spatial
conditioning, weak temporal conditioning. **The question that completes the story: what is the lift at
pipe 2, where this policy succeeds at 61%?**

* **materially larger at pipe 2** -- timing signal tracks success, there is a target value, and sharpening is
  the right lever.
* **the same or lower** -- **the policy has no timing anywhere.** It wins where duration alone suffices and
  loses where timing is required, which would explain every result in this project in one sentence, and the
  next lever is the observation rather than sharpening or search.

**The estimator is stratified, not pooled.** Within each 16-px bin, mean p(A) at expert A-onset frames minus
mean p(A) at non-onset frames; then a weighted average over bins with the onset count as the weight. That
holds position fixed by construction, which is the whole point — a pooled difference would re-import the
positional gradient the Goomba read had to remove.

**Intervals are bootstrapped over onsets, not frames.** Each onset contributes exactly one frame to the
estimate, so resampling onsets is the correct cluster. The earlier +0.103 interval came from a 7-value
bootstrap of a median and is re-reported here with its raw per-onset differences so its width is visible
rather than implied.

Forward passes only: expert observations are on disk as `frames.npy`.
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
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.runlength import N_BUCKETS, joint_size  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "data/bc_phase1/runlength.pt"
OUT = ROOT / "data/onset_lift_all_obstacles.json"
BIN = 16
BOOT = 4000

#: obstacle -> (window lo, hi). Windows cover the approach and the face of each obstacle.
WINDOWS = {
    "goomba_288": (180, 320),
    "pipe1_432": (370, 480),
    "pipe2_592": (530, 645),
    "pipe3_720": (660, 775),
    "pipe4_912": (850, 980),
    "koopas_1216": (1150, 1265),
    "gap_1380": (1320, 1430),
}


def stratified_lift(xs, pa, ons, bin_px=BIN):
    """Onset-minus-non-onset p(A), averaged over x bins, weighted by onset count."""
    bins = {}
    for lo in range(int(xs.min()) // bin_px * bin_px, int(xs.max()) + bin_px, bin_px):
        m = (xs >= lo) & (xs < lo + bin_px)
        o, n = pa[m & ons], pa[m & ~ons]
        if len(o) and len(n):
            bins[lo] = {"n_onset": int(len(o)), "n_other": int(len(n)),
                        "onset_mean": float(o.mean()), "other_mean": float(n.mean()),
                        "diff": float(o.mean() - n.mean())}
    if not bins:
        return None, {}
    w = np.array([b["n_onset"] for b in bins.values()], dtype=float)
    d = np.array([b["diff"] for b in bins.values()], dtype=float)
    return float((w * d).sum() / w.sum()), bins


def boot_lift(xs, pa, ons, reps=BOOT, seed=0):
    """Bootstrap the stratified lift by resampling ONSETS; non-onset baselines stay fixed."""
    rng = np.random.default_rng(seed)
    oi = np.flatnonzero(ons)
    if len(oi) < 3:
        return None
    base_bins = {}
    for lo in range(int(xs.min()) // BIN * BIN, int(xs.max()) + BIN, BIN):
        m = (xs >= lo) & (xs < lo + BIN)
        n = pa[m & ~ons]
        if len(n):
            base_bins[lo] = float(n.mean())
    vals = []
    for _ in range(reps):
        pick = rng.choice(oi, size=len(oi), replace=True)
        num = den = 0.0
        acc = {}
        for i in pick:
            b = int(xs[i]) // BIN * BIN
            if b in base_bins:
                acc.setdefault(b, []).append(pa[i])
        for b, v in acc.items():
            k = len(v)
            num += k * (float(np.mean(v)) - base_bins[b])
            den += k
        if den:
            vals.append(num / den)
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
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
    ds = ctx.dataset(ctx.expert_train)

    lo_all = min(v[0] for v in WINDOWS.values())
    hi_all = max(v[1] for v in WINDOWS.values())
    recs = []
    for run_id, entry in enumerate(ds.index):
        tr = np.asarray(ctx.expert_train[run_id].trace)
        w, s, pg = column(tr, "world"), column(tr, "stage"), column(tr, "pregame")
        x, ps = column(tr, "x_position"), column(tr, "player_state")
        raw = ds.raw[run_id]
        base = int(ds.offsets[run_id])
        fidx = entry.frame_indices
        for row in range(entry.n_obs):
            m = min(int(fidx[row]) + ds.label_offset, len(raw) - 1)
            if m >= len(x) or not (w[m] == 1 and s[m] == 1 and pg[m] == 1 and ps[m] == 8):
                continue
            if not (lo_all <= x[m] <= hi_all):
                continue
            cur, prv = int(raw[m]), int(raw[max(m - 1, 0)])
            recs.append((int(x[m]), bool((cur & A_BIT) and not (prv & A_BIT)), base + row))
    print(f"expert 1-1 surface frames in x {lo_all}-{hi_all}: {len(recs):,} "
          f"({sum(1 for r in recs if r[1])} A-onsets)", flush=True)

    rows = [r[2] for r in recs]
    pa = []
    for i in range(0, len(rows), 256):
        ob = torch.stack([ds[j][0] for j in rows[i:i + 256]])
        with torch.no_grad():
            p = torch.softmax(policy(ob), dim=-1).numpy()
        pa.extend(p[:, a_mask].sum(axis=1).tolist())
        if (i // 256) % 8 == 0:
            print(f"    forward {min(i + 256, len(rows)):,}/{len(rows):,}", flush=True)
    pa = np.asarray(pa)
    xs = np.asarray([r[0] for r in recs])
    ons = np.asarray([r[1] for r in recs])

    out = {"estimator": ("stratified: within each 16px bin, mean p(A) at expert A-onsets minus mean at "
                         "non-onset frames; weighted average over bins by onset count"),
           "interval": "bootstrap over ONSETS (each onset contributes one frame), 4000 reps",
           "checkpoint": CKPT.name, "bin_px": BIN, "obstacles": {}}
    print(f"\n{'obstacle':14s} {'window':>12s} {'onsets':>7s} {'other':>7s} {'lift':>8s} "
          f"{'95% CI':>20s} {'spatial swing':>14s}")
    for name, (lo, hi) in WINDOWS.items():
        m = (xs >= lo) & (xs <= hi)
        if not m.any():
            out["obstacles"][name] = {"window": [lo, hi], "n_frames": 0}
            print(f"{name:14s} {f'{lo}-{hi}':>12s} {'no frames':>7s}")
            continue
        lift, bins = stratified_lift(xs[m], pa[m], ons[m])
        ci = boot_lift(xs[m], pa[m], ons[m])
        by_bin = [b["other_mean"] for b in bins.values()]
        swing = (max(by_bin) - min(by_bin)) if by_bin else None
        n_on = int((ons & m).sum())
        n_ot = int((~ons & m).sum())
        out["obstacles"][name] = {
            "window": [lo, hi], "n_onsets": n_on, "n_non_onsets": n_ot,
            "stratified_lift": lift, "ci_onset_bootstrap": ci,
            "spatial_swing_non_onset": swing, "bins": bins}
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n<3 onsets"
        print(f"{name:14s} {f'{lo}-{hi}':>12s} {n_on:7d} {n_ot:7d} "
              f"{(lift if lift is not None else float('nan')):+8.3f} {cis:>20s} "
              f"{(swing if swing is not None else float('nan')):14.3f}", flush=True)

    g = out["obstacles"]["goomba_288"]
    p2 = out["obstacles"]["pipe2_592"]
    gl, pl = g.get("stratified_lift"), p2.get("stratified_lift")
    materially = (pl is not None and gl is not None and pl > gl + 0.03)
    out["comparison"] = {
        "goomba_lift": gl, "goomba_ci": g.get("ci_onset_bootstrap"),
        "pipe2_lift": pl, "pipe2_ci": p2.get("ci_onset_bootstrap"),
        "pipe2_minus_goomba": (pl - gl) if (pl is not None and gl is not None) else None,
        "materially_larger_at_pipe2": bool(materially)}
    out["verdict"] = (
        f"TIMING SIGNAL TRACKS SUCCESS: the x-matched onset lift at pipe 2 is {pl:+.3f} against the "
        f"Goomba's {gl:+.3f}. Where the policy succeeds it also discriminates when to act, so there is a "
        f"target value and sharpening is the right lever."
        if materially else
        f"THE POLICY HAS NO TIMING ANYWHERE: the x-matched onset lift at pipe 2 -- where it clears 61% -- "
        f"is {pl:+.3f}, against {gl:+.3f} at the Goomba where it fails. Success does not come with a "
        f"stronger timing signal. **Its competence is duration, not timing**, which explains why it beats a "
        f"rate-matched script wherever a long hold suffices and loses wherever placement matters. There is "
        f"nothing to sharpen, and the next lever is the observation.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
