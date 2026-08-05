"""Does p(A) spike at the expert's own A-onsets near the Goomba, or is it flat across the window?

The fork this decides:

* **spikes** -- the 84x84 observation carries the timing signal and the network can read it. The deficit is
  training/sampling, and sharpening is the lever.
* **flat** -- **the observation cannot resolve the Goomba's position.** A Goomba is 16 px wide; at 84x84 over
  a 256-px screen that is roughly 5 px. No amount of imitation, search or distillation on this input fixes
  the obstacle, and the project's shape changes.

**This one really is forward passes only.** Expert observations are on disk as `frames.npy` per run, and
`FrameStackDataset` already yields the stacked observation together with per-button **onset** flags using the
exact alignment training used. No emulator, no replay.

Method: for every expert A-onset whose x lies in 272-304 on 1-1 surface frames, run the policy over the
**+/-10 neighbouring dataset rows** and record p(A) as a function of offset from the onset. p(A) is the summed
softmax over the 107 A-containing classes, the same quantity the sampler draws against.

**n is small -- roughly 30 expert onsets in this window across 25 runs. Reported as a screen and labelled
one.** The contrast that matters is offset 0 against the flanks, aggregated, not any single onset.
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
from tasdata.bc.overnight_lib import boot_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.runlength import N_BUCKETS, joint_size  # noqa: E402
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "data/bc_phase1/runlength.pt"
OUT = ROOT / "data/expert_onset_probe.json"
WIN = (272, 304)
FLANK = 10
A_INDEX = NES_BUTTON_ORDER.index("A")
#: the policy's own p(A) in this window, from data/goomba_pa_probe.json, for context
POLICY_PA = {"death_no_jump": 0.430, "death_jumped": 0.424, "cleared": 0.459}


def stats(v) -> dict:
    a = np.asarray(list(v), dtype=float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "min": float(a.min())}


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

    ds = ctx.dataset(ctx.expert_train)          # label_mode='buttons': gives bits and onsets
    print(f"expert dataset {len(ds):,} rows; A-containing classes {int(a_mask.sum())}/{n_cls}",
          flush=True)

    # locate the onsets in dataset-row space, using the run's own trace for x
    onset_rows: list[int] = []
    for run_id, entry in enumerate(ds.index):
        run = ctx.expert_train[run_id] if run_id < len(ctx.expert_train) else None
        tr = np.asarray(ds.index[run_id].run.trace if hasattr(ds.index[run_id], "run")
                        else run.trace)
        w, s, pg = column(tr, "world"), column(tr, "stage"), column(tr, "pregame")
        x, ps = column(tr, "x_position"), column(tr, "player_state")
        raw = ds.raw[run_id]
        base = int(ds.offsets[run_id])
        fidx = entry.frame_indices
        for row in range(entry.n_obs):
            m = min(int(fidx[row]) + ds.label_offset, len(raw) - 1)
            if m >= len(x):
                continue
            if not (w[m] == 1 and s[m] == 1 and pg[m] == 1 and ps[m] == 8):
                continue
            if not (WIN[0] <= x[m] <= WIN[1]):
                continue
            cur, prev = int(raw[m]), int(raw[max(m - 1, 0)])
            if (cur & A_BIT) and not (prev & A_BIT):
                onset_rows.append(base + row)
    print(f"expert A-onsets with x in {WIN[0]}-{WIN[1]}: {len(onset_rows)}\n", flush=True)

    # forward pass over each onset's neighbourhood
    by_offset: dict[int, list[float]] = {o: [] for o in range(-FLANK, FLANK + 1)}
    per_onset = []
    for k, r in enumerate(onset_rows):
        run_id, row = ds.locate(r)
        n_obs = ds.index[run_id].n_obs
        rows, offsets = [], []
        for o in range(-FLANK, FLANK + 1):
            rr = row + o
            if 0 <= rr < n_obs:
                rows.append(int(ds.offsets[run_id]) + rr)
                offsets.append(o)
        if not rows:
            continue
        obs = torch.stack([ds[i][0] for i in rows])
        with torch.no_grad():
            p = torch.softmax(policy(obs), dim=-1).numpy()
        pa = p[:, a_mask].sum(axis=1)
        for o, v in zip(offsets, pa):
            by_offset[o].append(float(v))
        at0 = dict(zip(offsets, pa)).get(0)
        flank = [float(v) for o, v in zip(offsets, pa) if abs(o) >= 5]
        per_onset.append({"row": int(r), "pa_at_onset": (float(at0) if at0 is not None else None),
                          "pa_flank_mean": (float(np.mean(flank)) if flank else None)})
        if (k + 1) % 10 == 0:
            print(f"    {k + 1}/{len(onset_rows)} onsets", flush=True)

    profile = {o: stats(v) for o, v in by_offset.items() if v}
    print(f"\n{'offset':>7s} {'n':>4s} {'p(A) median':>12s} {'mean':>8s} {'max':>8s}")
    for o in sorted(profile):
        a = profile[o]
        star = "  <-- onset" if o == 0 else ""
        print(f"{o:>7d} {a['n']:>4d} {a['median']:>12.3f} {a['mean']:>8.3f} {a['max']:>8.3f}{star}",
              flush=True)

    at0 = [r["pa_at_onset"] for r in per_onset if r["pa_at_onset"] is not None]
    fl = [r["pa_flank_mean"] for r in per_onset if r["pa_flank_mean"] is not None]
    paired = [(a, b) for r in per_onset
              if (a := r["pa_at_onset"]) is not None and (b := r["pa_flank_mean"]) is not None]
    diffs = [a - b for a, b in paired]
    lo, hi = boot_ci(diffs) if diffs else (0.0, 0.0)
    spread = (max(a["median"] for a in profile.values())
              - min(a["median"] for a in profile.values())) if profile else 0.0
    print(f"\nonset p(A)  median {np.median(at0):.3f}   flank (|offset|>=5) median "
          f"{np.median(fl):.3f}")
    print(f"paired difference (onset - own flanks): median {np.median(diffs):+.4f} "
          f"bootstrap CI [{lo:+.4f}, {hi:+.4f}]  n={len(diffs)}")
    print(f"spread of the median profile across offsets: {spread:.4f}", flush=True)

    spikes = lo > 0 and np.median(diffs) > 0.02
    out = {
        "window_x": list(WIN), "flank": FLANK, "checkpoint": CKPT.name,
        "label": "SCREEN -- n is ~30 expert onsets across 25 runs",
        "method": ("forward passes only; expert observations come from frames.npy via "
                   "FrameStackDataset, using the same alignment training used"),
        "pa_definition": "summed softmax over the 107 A-containing classes of 300",
        "n_onsets": len(onset_rows),
        "profile_by_offset": {str(k): v for k, v in profile.items()},
        "onset_vs_flank": {
            "onset": stats(at0), "flank": stats(fl),
            "paired_difference_median": (float(np.median(diffs)) if diffs else None),
            "paired_bootstrap_ci": [lo, hi], "n_paired": len(diffs)},
        "median_profile_spread": spread,
        "policy_pa_in_window_for_context": POLICY_PA,
        "verdict": (
            f"p(A) SPIKES AT THE EXPERT'S ONSETS: median {np.median(at0):.3f} at the onset against "
            f"{np.median(fl):.3f} in its own flanks, paired difference {np.median(diffs):+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]. The observation carries the timing signal and the network reads it; "
            f"the deficit is training and sampling, and sharpening is the lever."
            if spikes else
            f"p(A) IS FLAT ACROSS THE WINDOW: median {np.median(at0):.3f} at the expert's onsets against "
            f"{np.median(fl):.3f} in its own flanks, paired difference {np.median(diffs):+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}], and the median profile varies by only {spread:.3f} across 21 "
            f"offsets. **The 84x84 observation does not resolve when to jump at the Goomba.** No amount of "
            f"imitation, search or distillation on this input fixes the obstacle."),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({round((time.time() - t0) / 60, 1)} min)")


if __name__ == "__main__":
    main()
