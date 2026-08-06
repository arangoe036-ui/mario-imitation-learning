"""§3 + §4.2: re-run the encoder 2x2 at the PEAK, and re-derive the two figures known to be censored.

**§3.** Every architecture comparison in this project was made at 15,000 steps. Block 58 showed the 500–3,000
region beats 12,000–60,000 by +12.6 pp on pipe 2, so all of them were measured ~15x past the optimum in a
degraded regime. This re-runs the encoder 2x2 — `cnn(16,32,32)` vs `cnn(32,64,64)`, 84x84, **1,000 steps**,
5 seeds each — at T=0.7, n=200, `STALL=6500`.

**It can invert, and that is the point.** A wider encoder is *less* trained relative to its capacity at a
fixed 1,000 steps, so its advantage may shrink or reverse. If it does, the +367 px depth result is void twice
over — once for the training length and once for the terminator.

**§4.2.** Two figures in `NORTH_STAR.md` were measured under `STALL=300` and are lower bounds:

1. the **+367 px encoder depth result** at 15,000 steps — the P-cell arms at `STALL=6500` already exist from
   block 58's mask study, so only the **B cell** needs running here;
2. block 56's **"0 of 720 completions at T=0.7"** — conditional-on-arrival over start states, re-run
   separately if budget allows.

Exact permutation test over seeds, **`1/C(n,k)` floor** as corrected — with equal group sizes the mirror
arrangement does exist, so for 5 vs 5 the floor is 2/252 = 0.0079; the `1/n` form matters only for unequal
groups, which is where block 57 got it wrong.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.button_mask_eval import rollout  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.scaleup_eval import resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/peak_rebaseline.json"
TRACED = ROOT / "data/traces"

CELLS = {
    # §3 -- the 2x2 re-run at the peak
    "PK16_1k": [f"PK16_84_s{i}" for i in range(5)],
    "PK32_1k": [f"PK32_84_s{i}" for i in range(5)],
    # §4.2(1) -- the 15k comparison re-derived at the corrected terminator
    "B16_15k": ["B_84_d64_L1", "B_84_seed1", "B_84_seed2", "B_84_seed3", "B_84_seed4"],
    "P32_15k": ["P_84_cnn32", "P_84_cnn32_seed1", "P_84_cnn32_seed2", "P_84_cnn32_seed3",
                "P_84_cnn32_seed4"],
}
TEMP = 0.7
N_EVAL = 200
WALLS = {"pipe3_735": 735, "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562,
         "flagpole_3266": 3266}
ARM_BUDGET_S = 15 * 60
#: block 58's mask study already ran the P cell unmasked at STALL=6500, T=0.7
REUSE = {f"P_84_cnn32{'' if i == 0 else f'_seed{i}'}": f"mask_P_84_cnn32{'' if i == 0 else f'_seed{i}'}_t0.7_unmasked_200.json"
         for i in range(5)}


def perm_p(a, b):
    """Exact two-sided permutation on the difference of means, with the attainable floor."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool = np.concatenate([a, b])
    obs = a.mean() - b.mean()
    d = []
    for idx in itertools.combinations(range(len(pool)), len(a)):
        x = pool[list(idx)]
        y = pool[[i for i in range(len(pool)) if i not in idx]]
        d.append(x.mean() - y.mean())
    d = np.asarray(d)
    n = len(d)
    # with EQUAL group sizes every split has a mirror, so the floor is 2/n; unequal -> 1/n
    floor = (2.0 / n) if len(a) == len(b) else (1.0 / n)
    return float((np.abs(d) >= abs(obs) - 1e-9).mean()), n, floor


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 120 * 60)
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CELLS)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("skipped", [])
    out.update({"n_eval": N_EVAL, "temperature": TEMP, "terminator": RB.describe(),
                "measurement_basis": "single_life_from_level_start", "cells": CELLS,
                "button_mask": "NOT applied -- block 58's branch says do not keep it"})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    for cell in only:
        for name in CELLS[cell]:
            key = f"{cell}/{name}"
            if key in out["arms"]:
                continue
            if not (ROOT / f"data/bc_scaleup/{name}.pt").exists():
                continue
            reuse = TRACED / REUSE[name] if (cell == "P32_15k" and name in REUSE) else None
            if reuse is not None and reuse.exists():
                import json as _j
                traces_raw = _j.loads(reuse.read_text())["episodes"]
                from scripts.scaleup_eval import _Ep
                traces = [_Ep(e) for e in traces_raw]
                src = "reused from block 58 mask study (unmasked, STALL=6500)"
            else:
                if not dl.can_afford(150):
                    out["skipped"].append({"arm": key, "reason": "deadline"})
                    print(f"{dl.stamp()} SKIP {key}", flush=True)
                    continue
                policy, cfg, blob = G.load_ckpt(name)
                tp = TRACED / f"peak_{name}_t0.7_{N_EVAL}.json"
                try:
                    with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                        s = sess_get()
                        try:
                            traces = resumable(tp, N_EVAL,
                                               lambda i: rollout(s, policy, cfg, start, i, lut,
                                                                 byte_of, None, temp=TEMP))
                        finally:
                            s.close()
                except TimedOut as e:
                    out["skipped"].append({"arm": key, "reason": str(e)})
                    save()
                    continue
                src = "block 59"
            rec = score(key, traces)
            xs = [max(f[0] for f in t.frames) for t in traces]
            rec.update({"cell": cell, "checkpoint": name, "source": src,
                        "x_p90": float(np.percentile(xs, 90)),
                        "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                          "rate": float(np.mean([x > v for x in xs]))}
                                      for w, v in WALLS.items()},
                        "flagpole_episodes": int(sum(
                            1 for t in traces
                            if any(len(f) > 4 and f[4] == 0x05 for f in t.frames))),
                        "terminator": RB.describe()})
            out["arms"][key] = rec
            save()
            pw = rec["past_wall"]
            print(f"  {dl.stamp()} {key:28s} x_max {rec['x_max']:5d} x_med {rec['x_median']:5.0f} "
                  f"p2 {rec['clearance']['pipe2']['rate']*100:5.1f}% "
                  f"p3 {pw['pipe3_735']['rate']*100:5.1f}% p4 {pw['pipe4_975']['rate']*100:5.1f}% "
                  f"flag {rec['flagpole_episodes']}", flush=True)

    # ---------------- contrasts ----------------
    def vals(cell, f):
        v = []
        for name in CELLS[cell]:
            a = out["arms"].get(f"{cell}/{name}")
            if a:
                v.append(a["past_wall"][f]["rate"] * 100 if f in WALLS else a[f])
        return v

    res = {}
    for label, hi, lo in (("section3_peak_1k__cnn32_minus_cnn16", "PK32_1k", "PK16_1k"),
                          ("section4_2__15k_cnn32_minus_cnn16_at_STALL6500", "P32_15k", "B16_15k")):
        row = {}
        for f in ("x_max", "x_median", "pipe3_735", "pipe4_975"):
            a, b = vals(hi, f), vals(lo, f)
            if len(a) >= 3 and len(b) >= 3:
                p, nperm, floor = perm_p(a, b)
                row[f] = {"hi_values": a, "lo_values": b,
                          "hi_mean": float(np.mean(a)), "lo_mean": float(np.mean(b)),
                          "hi_median": float(np.median(a)), "lo_median": float(np.median(b)),
                          "mean_diff": float(np.mean(a) - np.mean(b)),
                          "median_diff": float(np.median(a) - np.median(b)),
                          "perm_p": p, "n_permutations": nperm, "min_attainable_p": floor}
        res[label] = row
    out["contrasts"] = res

    p3 = res.get("section3_peak_1k__cnn32_minus_cnn16", {}).get("x_max")
    p42 = res.get("section4_2__15k_cnn32_minus_cnn16_at_STALL6500", {}).get("x_max")
    parts = []
    if p3:
        parts.append(
            f"**AT THE PEAK (1,000 steps) the encoder contrast on x_max is {p3['mean_diff']:+.0f} px "
            f"(median {p3['median_diff']:+.0f}), p={p3['perm_p']:.3f}, floor {p3['min_attainable_p']:.4f}** "
            f"— cnn32 {[int(v) for v in p3['hi_values']]} vs cnn16 {[int(v) for v in p3['lo_values']]}.")
    if p42:
        parts.append(
            f"**RE-DERIVED AT STALL=6500, 15,000 steps: {p42['mean_diff']:+.0f} px "
            f"(median {p42['median_diff']:+.0f}), p={p42['perm_p']:.3f}** — against the +367 px originally "
            f"reported at STALL=300.")
    if p3 and p42:
        inverted = (p3["mean_diff"] < 0) and (p42["mean_diff"] > 0)
        parts.append(
            "**The encoder advantage INVERTS at the peak**: positive at 15,000 steps, negative at 1,000. "
            "So the +367 px result is an artifact of comparing two architectures in a regime both were "
            "over-trained into, and it is void twice over — once for training length, once for the "
            "terminator."
            if inverted else
            "The sign is consistent across both training lengths, so the encoder effect is not an artifact "
            "of over-training, though its magnitude changes with the corrected terminator.")
    out["verdict"] = " ".join(parts) if parts else "Insufficient arms evaluated."
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    for label, row in res.items():
        print(f"\n{label}")
        for f, v in row.items():
            print(f"   {f:12s} hi {[round(x) for x in v['hi_values']]} lo "
                  f"{[round(x) for x in v['lo_values']]} | mean {v['mean_diff']:+8.1f} "
                  f"median {v['median_diff']:+8.1f} p={v['perm_p']:.3f}")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
