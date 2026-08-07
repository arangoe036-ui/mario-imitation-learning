"""§1b precondition: define the obstacle windows from the FAILURE HISTOGRAM, not from the wall list.

The directive is explicit that the windows come from where the policy actually loses. So the histogram is
built from the **2,000 baseline episodes** (`PK32_84_s0..9`, the C0/ε=0 cell), taking each episode's
**maximum-x frame** — the point at which forward progress stopped, which is what `failure_kinds` already
treats as the failure location — and reading the modes off it.

**⚠ The windows are 1-1 coordinates and are therefore applied only to 1-1 training samples.** x means a
different piece of geometry in every level, so a window at x∈[560,660] is the second pipe in 1-1 and something
unrelated in 7-2. Applying it corpus-wide would be a different, meaningless intervention. This is a real limit
on the reach of §1b and it is measured here rather than discovered afterwards: the artifact reports the
**share of training draws** the reweighting can move at each strength, so the sweep is known to be able to
express an effect before it is run. (Block 57 shipped a sweep that could not express its own answer; the fix
is to compute the reachable share first.)

Writes `data/failure_windows.json` and caches per-sample x in `data/level_index_map.npz`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.data import FrameStackDataset  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACED = ROOT / "data/traces"
OUT = ROOT / "data/failure_windows.json"
IDXCACHE = ROOT / "data/level_index_map.npz"

BIN = 20            # px per histogram bin
MIN_SHARE = 0.02    # a bin must hold >=2% of failures to seed a window
HALF_WIDTH = 60     # window is mode +- 60 px: the approach, not just the obstacle face
STRENGTHS = [1.0, 1.5, 2.0, 3.0]


def baseline_traces():
    """The ten C0 arms: s0-s2 from the jump-bias unbiased cell, s3-s9 from block 64."""
    out, used = [], []
    for i in range(10):
        for p in (TRACED / f"jb_PK32_84_s{i}_unbiased_200.json", TRACED / f"nh_PK32_84_s{i}_200.json"):
            if p.exists():
                out += json.loads(p.read_text())["episodes"]
                used.append(p.name)
                break
    return out, used


def main() -> None:
    eps, used = baseline_traces()
    fail_x = np.array([max(f[0] for f in e["frames"]) for e in eps], dtype=float)
    n = len(fail_x)

    edges = np.arange(0, fail_x.max() + BIN, BIN)
    cnt, _ = np.histogram(fail_x, bins=edges)
    share = cnt / n

    # local maxima above the share floor, merged when their windows would overlap
    seeds = [int(i) for i in range(len(cnt)) if share[i] >= MIN_SHARE
             and cnt[i] >= cnt[max(0, i - 1)] and cnt[i] >= cnt[min(len(cnt) - 1, i + 1)]]
    wins = []
    for i in seeds:
        c = float(edges[i] + BIN / 2)
        lo, hi = c - HALF_WIDTH, c + HALF_WIDTH
        if wins and lo <= wins[-1][1]:
            wins[-1] = (wins[-1][0], hi, wins[-1][2] + [c])
        else:
            wins.append((lo, hi, [c]))
    windows = [{"lo": w[0], "hi": w[1], "modes": w[2],
                "failures_inside": int(((fail_x >= w[0]) & (fail_x < w[1])).sum()),
                "share_of_failures": float(((fail_x >= w[0]) & (fail_x < w[1])).mean())}
               for w in wins]

    # ---- per training sample: x at the labelled frame (and reuse the cached level map) ----
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    rows = z["rows"]
    ctx = O.Ctx()
    base = FrameStackDataset(ctx.expert_train, ctx.vocab, stack=4, label_mode="token")
    xs = np.full(len(rows), -1, dtype=np.int64)
    offs = base.offsets
    for run_id, entry in enumerate(base.index):
        tr = np.asarray(ctx.expert_train[run_id].trace)
        x = column(tr, "x_position")
        last = len(base.tokens[run_id]) - 1
        lo, hi = int(offs[run_id]), int(offs[run_id + 1])
        sel = np.flatnonzero((rows >= lo) & (rows < hi))
        if not len(sel):
            continue
        m = np.minimum(entry.frame_indices[rows[sel] - lo] + base.label_offset, last)
        xs[sel] = x[np.minimum(m, len(x) - 1)]

    cached = np.load(IDXCACHE, allow_pickle=True)
    lvl = np.array([str(v) for v in cached["level"]])
    inctl = cached["in_control"]
    is11 = lvl == "1-1"
    inwin = np.zeros(len(rows), dtype=bool)
    for w in windows:
        inwin |= is11 & (xs >= w["lo"]) & (xs < w["hi"])
    inwin &= inctl  # a pregame frame at x=40 is not an obstacle approach

    N = len(rows)
    k = int(inwin.sum())
    reach = {f"{s:g}x": {"weighted_share_of_draws": s * k / (s * k + (N - k)),
                         "expected_samples_per_1000_steps": 64000 * s * k / (s * k + (N - k)),
                         "multiplier_vs_baseline": (s * k / (s * k + (N - k))) / (k / N)}
             for s in STRENGTHS}

    out = {
        "source": {"episodes": n, "arms": 10, "trace_files": used,
                   "failure_definition": "the frame of maximum x in the episode"},
        "method": {"bin_px": BIN, "min_bin_share": MIN_SHARE, "half_width_px": HALF_WIDTH,
                   "note": "windows seeded at local maxima of the failure histogram, merged on overlap"},
        "histogram_top15": sorted(
            [{"x_lo": float(edges[i]), "x_hi": float(edges[i] + BIN), "n": int(cnt[i]),
              "share": float(share[i])} for i in range(len(cnt)) if cnt[i]],
            key=lambda r: -r["n"])[:15],
        "windows": windows,
        "failures_covered": float(sum(w["share_of_failures"] for w in windows)),
        "training_samples": {
            "total": N, "in_window": k, "share": k / N,
            "restriction": "windows are 1-1 coordinates, so only in-control 1-1 samples can be in one",
            "samples_1_1_in_control": int((is11 & inctl).sum())},
        "REACHABLE_EFFECT": reach,
        "strengths": STRENGTHS,
    }
    out["design_check"] = (
        f"The reweighting can move **{k:,} of {N:,} training samples ({k / N * 100:.2f}%)**. At the strongest "
        f"swept strength (3.0x) they are {reach['3x']['weighted_share_of_draws'] * 100:.2f}% of draws against "
        f"{k / N * 100:.2f}% at baseline — **{reach['3x']['multiplier_vs_baseline']:.2f}x the exposure**, or "
        f"{reach['3x']['expected_samples_per_1000_steps']:.0f} draws per 1,000 steps against "
        f"{64000 * k / N:.0f}. The windows cover {out['failures_covered'] * 100:.1f}% of baseline failures.")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    np.savez_compressed(IDXCACHE, rows=rows, level=lvl, in_control=inctl, x=xs, in_window=inwin)

    print(json.dumps({k2: out[k2] for k2 in ("source", "windows", "training_samples",
                                             "REACHABLE_EFFECT")}, indent=1, default=str)[:2400])
    print("\n" + "=" * 78)
    print(out["design_check"])
    print(f"\nwrote {OUT} and {IDXCACHE}")


if __name__ == "__main__":
    main()
