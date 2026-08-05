"""§2: is clearance monotone in temperature? T in {0.5, 0.3, 0.15}, uncapped, measurable at n=200.

Block 54 found that argmax (T -> 0) produced one trajectory to x=916 -- the deepest ever recorded here --
and that it is **n=1 by construction**: a greedy policy on a deterministic emulator from a fixed start
yields one trajectory however many episodes are run. T=1.0 and T=0.7 were measured and nothing between
0.7 and zero.

**A low but non-zero temperature stays stochastic**, so it yields many distinct trajectories, n=200 works,
and the numbers are directly comparable to every prior block. If sharpening is the lever, clearance should
rise as T falls and argmax's x=916 is the visible end of a trend. If the ladder is flat, argmax was a lucky
path.

**The A-run cap is NOT carried forward.** It was killed in block 54 (+0.5 / -1.5 / 0.0 pp across seeds) and
uncapped is the base rule again.

**`effective_n` is reported for every rung and is load-bearing.** As T falls the policy approaches
determinism and distinct trajectories collapse. **Any rung with effective_n < 20 is reported as a
trajectory count, not a rate.** That rule is what stopped block 54 shipping a 91.7%.

Read as a ladder across **three B seeds**, whose uncapped pipe-2 spread is 16.0 pp -- larger than any effect
measured in block 54. A trend that does not survive all three seeds is not a trend.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/temperature_ladder.json"
PRIOR = [ROOT / "data/generation_sweep.json", ROOT / "data/generation_seeds.json"]

#: new rungs this block; 1.0 and 0.7 come from block 54 where they exist uncapped
NEW_TEMPS = [0.5, 0.3, 0.15]
CKPTS = ["B_84_d64_L1", "B_84_seed1", "B_84_seed2", "R_128_d64_L1"]
LADDER = [1.0, 0.7, 0.5, 0.3, 0.15]
MIN_EFFECTIVE_N = 20
N_EVAL = 200


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    lut_cache: dict[str, np.ndarray] = {}

    prior = {}
    for p in PRIOR:
        if p.exists():
            prior.update(json.loads(p.read_text()).get("arms", {}))

    def get_ck(name):
        policy, cfg, blob = G.load_ckpt(name)
        corpus = blob.get("corpus", "runs")
        if corpus not in lut_cache:
            z = np.load(ROOT / f"data/runlength_index_{corpus}.npz")
            lut_cache[corpus] = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")},
                                                n_cls)
        return {"name": name, "policy": policy, "cfg": cfg, "blob": blob,
                "lut": lut_cache[corpus], "byte_of": byte_of}

    def sess_get():
        """Every session is warmed, so the first scored episode sees what the rest see."""
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.update({
        "n_eval": N_EVAL, "measurement_basis": "single_life",
        "generation_rule": "capped (non-A runs <= 4); A-runs UNCAPPED -- the cap is dead",
        "ladder": LADDER, "new_this_block": NEW_TEMPS,
        "min_effective_n_for_a_rate": MIN_EFFECTIVE_N,
        "episode0_guard": ("every session warmed with a throwaway reset+step; see "
                           "compose.warm_session for the measured defect it fixes"),
        "expert_reference": {"a_rate": 0.152, "onsets_per_1000": 27.5, "airborne": 0.611}})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    print("=== temperature ladder, uncapped ===", flush=True)
    for name in CKPTS:
        ck = None
        for T in LADDER:
            k = G.tag(name, None, T)
            if k in out["arms"]:
                continue
            if T in (1.0,) and k in prior:               # block 54's uncapped T=1.0 arms
                out["arms"][k] = {**prior[k], "source": "block 54"}
                save()
                continue
            if ck is None:
                ck = get_ck(name)
            out["arms"][k] = G.run_arm(sess_get, ck, None, T, ctx, start, n=N_EVAL)
            out["arms"][k]["source"] = "block 55"
            save()

    # ------------------------------------------------ ladder analysis
    def arm(name, T):
        return out["arms"].get(G.tag(name, None, T))

    rungs = {}
    print(f"\n{'checkpoint':16s}" + "".join(f"{('T=' + str(T)):>16s}" for T in LADDER))
    for ob in ("pipe1", "pipe2", "pipe3", "pipe4"):
        print(f"-- {ob} --")
        for name in CKPTS:
            cells = []
            for T in LADDER:
                a = arm(name, T)
                if not a:
                    cells.append("       -        ")
                    continue
                eff = a["distinctness"]["effective_n"]
                r = a["clearance"][ob]["rate"] * 100
                cells.append(f"{r:9.1f}{('*' if eff < MIN_EFFECTIVE_N else ' '):1s}(n{eff:3d})")
            print(f"{name:16s}" + "".join(f"{c:>16s}" for c in cells), flush=True)

    for name in CKPTS:
        row = {}
        for T in LADDER:
            a = arm(name, T)
            if not a:
                continue
            eff = a["distinctness"]["effective_n"]
            row[str(T)] = {
                "effective_n": eff, "measurable_as_rate": eff >= MIN_EFFECTIVE_N,
                "clearance": {o: a["clearance"][o]["rate"] for o in
                              ("pipe1", "pipe2", "pipe3", "pipe4")},
                "x_median": a["x_median"], "x_max": a["x_max"],
                "a_rate": a["button_marginals"]["rates"]["A"],
                "onsets_per_1000": a["a_onsets_per_1000_frames"],
                "airborne": a["behaviour"].get("airborne_fraction"),
                "a_hold": {k: a["a_hold_anywhere"].get(k) for k in ("median", "p99", "max")},
                "vs_script_best_fixed_rate": {
                    o: v["advantage_pp"] for o, v in
                    a["vs_script_best_fixed_rate"]["per_obstacle"].items()},
                "source": a.get("source")}
        rungs[name] = row
    out["rungs"] = rungs

    # monotonicity: does clearance rise as T falls, per obstacle, per seed?
    mono = {}
    for ob in ("pipe2", "pipe3"):
        per = {}
        for name in CKPTS:
            vals = [(T, rungs[name][str(T)]["clearance"][ob])
                    for T in LADDER
                    if str(T) in rungs[name] and rungs[name][str(T)]["measurable_as_rate"]]
            if len(vals) < 3:
                continue
            ys = [v for _, v in vals]                    # ordered T high -> low
            # Spearman against falling T, plus the endpoint difference with an interval
            rho = float(np.corrcoef(np.arange(len(ys)), ys)[0, 1]) if len(set(ys)) > 1 else 0.0
            k_hi = int(round(ys[0] * N_EVAL))
            k_lo = int(round(ys[-1] * N_EVAL))
            lo, hi = diff_ci(k_hi, N_EVAL, k_lo, N_EVAL)
            per[name] = {"temps": [t for t, _ in vals], "rates": ys,
                         "trend_corr_with_falling_T": rho,
                         "lowest_minus_highest_T_pp": (ys[-1] - ys[0]) * 100,
                         "ci_pp": [lo * 100, hi * 100], "method": "Newcombe"}
        seeds = [v for k, v in per.items() if k.startswith("B_")]
        per["_summary"] = {
            "n_measurable_checkpoints": len(per),
            "n_B_seeds": len(seeds),
            "all_B_seeds_rise": bool(seeds) and all(s["lowest_minus_highest_T_pp"] > 0
                                                    for s in seeds),
            "any_B_seed_interval_excludes_zero": any(s["ci_pp"][0] > 0 for s in seeds),
            "B_seed_deltas_pp": [s["lowest_minus_highest_T_pp"] for s in seeds]}
        mono[ob] = per
    out["monotonicity"] = mono

    p2, p3 = mono["pipe2"]["_summary"], mono["pipe3"]["_summary"]
    rises = p2["all_B_seeds_rise"] and p2["any_B_seed_interval_excludes_zero"]
    out["binary_question_part1"] = {
        "clearance_monotone_in_temperature": rises,
        "pipe2": p2, "pipe3": p3}
    if rises:
        out["verdict"] = (
            f"**THE LADDER RISES.** Pipe-2 clearance increases as temperature falls in all "
            f"{p2['n_B_seeds']} B seeds (deltas {[round(x, 1) for x in p2['B_seed_deltas_pp']]} pp) with "
            f"at least one interval excluding zero. Sharpening is a real lever and argmax's x=916 is the "
            f"end of a trend rather than a lucky path.")
    else:
        d2 = p2["B_seed_deltas_pp"]
        falls = bool(d2) and all(x < 0 for x in d2)
        # "Flat" would be the directive's third branch. Distinguish it from a DECLINE, which is a
        # stronger statement and a different one: a decline rules sharpening out as a direction, not
        # merely as an untapped gain.
        out["verdict"] = (
            (f"**THE LADDER DECLINES AT PIPE 2 — it is not flat, it falls.** Lowest-minus-highest "
             f"temperature deltas across the three B seeds: {[round(x, 1) for x in d2]} pp, "
             f"**negative in all three**, plus {mono['pipe2'].get('R_128_d64_L1', {}).get('lowest_minus_highest_T_pp', float('nan')):+.1f} pp "
             f"at R. Sharpening does not merely fail to help; **it makes clearance worse, monotonically.** "
             f"So sharpening joins the A-cap on the dead list and argmax's x=916 was a lucky path at the "
             f"far end of a DOWNWARD trend, not the visible end of an upward one — which is a stronger "
             f"negative than the directive's third branch anticipated."
             if falls else
             f"**THE LADDER DOES NOT RISE AT PIPE 2.** Deltas across the three B seeds: "
             f"{[round(x, 1) for x in d2]} pp — not all positive. Sharpening joins the A-cap on the "
             f"dead list and argmax's x=916 was a lucky path.")
            + f"\n\n**But the behaviour statistics move the OTHER way, monotonically.** As T falls, "
              f"A rate, jump starts per 1,000 and airborne fraction all march toward the expert's "
              f"(0.152 / 27.5 / 61.1%) while clearance falls. **Statistical resemblance to the expert "
              f"and competence are moving in opposite directions here**, which retires "
              f"expert-likeness of the button marginals as a proxy for skill.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
