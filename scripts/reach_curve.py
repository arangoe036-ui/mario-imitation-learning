"""§2: where does reach actually collapse? Sampled T=0.7, all 72 start states, binned by start x.

From x=1264 this policy finishes 1-1. From x=0 it stalls before x=1,000. **Fifty-five blocks of measurement
sit on four obstacles inside the first 975 px of a 3,266 px level, and nobody has ever measured where reach
breaks as a function of where you start.**

What this buys: a decomposition of "cannot finish 1-1" into "reach collapses at x≈N". If the curve is flat
after some x, the remaining problem is a **bounded early segment**, which is a much smaller target than the
whole level.

**⚠ `measurement_basis: conditional_on_arrival`.** Each number is "given the policy stands here, how much
further does it get". **Never table these beside an n=200 single-life figure.** And the start library is
harvested from *one* policy's own play, so the states are places that policy reached -- not a sample of the
level. Five states sit at x=722 and five at x=898.

**`REPEATS` episodes per start state**, because the policy is stochastic at T=0.7: one episode per state would
make each bin's number a draw of one per state. Per-bin n is reported as both states and episodes, and any bin
with fewer than `THIN_STATES` distinct states is marked `thin: true` and must not carry a claim.

Reported per bin: **median and max Δx** (how much further it got), the absolute max_x, and whether the
flagpole at x=3266 was reached.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.argmax_startstates import restore_state, rollout_from  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc.overnight_lib import wilson  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STARTLIB = ROOT / "data/startlib_policy.json"
P1_TRACES = ROOT / "data/traces/p1_200.json"
OUT = ROOT / f"data/reach_curve_{(sys.argv[1] if len(sys.argv) > 1 else 'P_84_cnn32')}.json"
PARTIAL = ROOT / "data/reach_curve.partial.json"

TEMP = 0.7
REPEATS = 5
FLAG = 3266
THIN_STATES = 4
CHUNK = 30
#: x-bin edges. Chosen to put each known obstacle in its own bin and keep the late level coarse,
#: since the library thins out there.
EDGES = [0, 200, 350, 500, 650, 800, 1000, 1300, 1600, 2000]


def bin_of(x: int) -> str:
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= x < EDGES[i + 1]:
            return f"{EDGES[i]}-{EDGES[i + 1]}"
    return f"{EDGES[-1]}+"


def main() -> None:
    t0 = time.time()
    arm = sys.argv[1] if len(sys.argv) > 1 else "P_84_cnn32"
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    states = json.loads(STARTLIB.read_text())["states"]
    p1 = {e["seed"]: e for e in json.loads(P1_TRACES.read_text())["episodes"]}
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    policy, cfg, blob = G.load_ckpt(arm)
    corpus = blob.get("corpus", "runs")
    z = np.load(ROOT / f"data/runlength_index_{corpus}.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    done = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    todo = [(s, r) for s in states for r in range(REPEATS)
            if f"{arm}:{s['seed']}:{s['frame_index']}:{r}" not in done]
    print(f"arm {arm} | {len(states)} states x {REPEATS} repeats | {len(todo)} to run", flush=True)
    if todo:
        sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(sess, start.frame)
        try:
            for i, (st, rep) in enumerate(todo):
                ep = p1.get(st["seed"])
                if ep is None:
                    continue
                obs = restore_state(sess, st, [f[3] for f in ep["frames"]], start.frame)
                got = read_smb(obs.ram, obs.framecount).x_position
                tr = rollout_from(sess, policy, cfg, obs, byte_of, lut, temp=TEMP,
                                  seed=st["frame_index"] * 100 + rep)
                fr = tr.frames
                mx = int(max(f[0] for f in fr)) if fr else int(got)
                done[f"{arm}:{st['seed']}:{st['frame_index']}:{rep}"] = {
                    "arm": arm, "start_x": st["x"], "restored_x": int(got),
                    "restore_ok": bool(abs(int(got) - st["x"]) <= 1),
                    "max_x": mx, "delta_x": mx - int(got), "n_frames": len(fr),
                    "ended": getattr(tr, "ended", None), "reached_flag": bool(mx >= FLAG)}
                if (i + 1) % CHUNK == 0 or i == len(todo) - 1:
                    PARTIAL.write_text(json.dumps(done, separators=(",", ":")))
                    print(f"  {i + 1}/{len(todo)}", flush=True)
        finally:
            sess.close()

    rows = [v for v in done.values() if v["arm"] == arm]
    out = {
        "arm": arm, "temperature": TEMP, "repeats_per_state": REPEATS,
        "measurement_basis": "conditional_on_arrival",
        "NOT_comparable_to": "any n=200 single-life figure in this project",
        "start_library_caveat": ("harvested from ONE policy's own play; these are places that policy "
                                "reached, not a sample of the level. Five states at x=722, five at x=898."),
        "flagpole_x": FLAG, "bin_edges": EDGES,
        "n_states": len({r["start_x"] for r in rows}),
        "n_states_note": ("distinct start_x values; the library has 72 states but duplicates x "
                          "(five at 722, five at 898), so distinct-x is lower than 72"),
        "n_episodes": len(rows),
        "restore_mismatches": sum(1 for r in rows if not r["restore_ok"]),
        "bins": {}}

    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(bin_of(r["start_x"]), []).append(r)
    print(f"\n{'bin':>12s}{'states':>7s}{'eps':>5s}{'med Δx':>8s}{'max Δx':>8s}"
          f"{'med max_x':>10s}{'best x':>8s}{'flag':>6s}")
    for b in sorted(groups, key=lambda s: int(s.split("-")[0].rstrip("+"))):
        g = groups[b]
        nst = len({r["start_x"] for r in g})
        dx = [r["delta_x"] for r in g]
        mx = [r["max_x"] for r in g]
        kf = sum(1 for r in g if r["reached_flag"])
        lo, hi = wilson(kf, len(g))
        rec = {"n_states": nst, "n_episodes": len(g), "thin": nst < THIN_STATES,
               "start_x_values": sorted({r["start_x"] for r in g}),
               "delta_x": {"median": float(np.median(dx)), "max": int(max(dx)),
                           "p25": float(np.percentile(dx, 25)),
                           "p75": float(np.percentile(dx, 75))},
               "max_x": {"median": float(np.median(mx)), "max": int(max(mx))},
               "reached_flagpole": {"k": kf, "n": len(g), "rate": kf / len(g),
                                    "wilson": [lo, hi]},
               "measurement_basis": "conditional_on_arrival"}
        out["bins"][b] = rec
        print(f"{b:>12s}{nst:>7d}{len(g):>5d}{rec['delta_x']['median']:>8.0f}"
              f"{rec['delta_x']['max']:>8d}{rec['max_x']['median']:>10.0f}"
              f"{rec['max_x']['max']:>8d}{kf:>4d}/{len(g)}"
              f"{'  ⚠thin' if rec['thin'] else ''}", flush=True)

    # ---- where does reach break? ----
    solid = {b: v for b, v in out["bins"].items() if not v["thin"]}
    by_start = sorted(solid.items(), key=lambda kv: int(kv[0].split("-")[0]))
    dmed = [(b, v["delta_x"]["median"]) for b, v in by_start]
    flag_bins = [b for b, v in out["bins"].items() if v["reached_flagpole"]["k"] > 0]
    # the furthest absolute x reached, per bin, tells you whether a bin can finish at all
    reach_ceiling = {b: v["max_x"]["max"] for b, v in out["bins"].items()}
    out["analysis"] = {
        "median_delta_x_by_bin": dict(dmed),
        "reach_ceiling_by_bin": reach_ceiling,
        "bins_reaching_flagpole": flag_bins,
        "n_solid_bins": len(solid),
        "thin_bins": [b for b, v in out["bins"].items() if v["thin"]],
        "flagpole_episodes": sum(v["reached_flagpole"]["k"] for v in out["bins"].values()),
        "flagpole_total_episodes": len(rows)}
    lows = [b for b, d in dmed if d < 200]
    out["verdict"] = (
        f"Reach curve for {arm} at T={TEMP}, conditional on arrival, {len(rows)} episodes over "
        f"{REPEATS} repeats of 72 start states. Median Δx by start bin: "
        f"{', '.join(f'{b}:{d:.0f}' for b, d in dmed)}. "
        f"Flagpole reached from {len(flag_bins)} bin(s) "
        f"({out['analysis']['flagpole_episodes']}/{len(rows)} episodes). "
        + (f"**Reach is worst from bins {lows}** — the early segment is where it collapses, not the late "
           f"one, which is where every prior measurement in this project has pointed."
           if lows else
           "**No bin has a median Δx below 200 px**, so there is no single early cliff in this library.")
        + " **conditional_on_arrival — do not table beside any n=200 single-life figure.**")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
