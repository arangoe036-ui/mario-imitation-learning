"""§3: the 2x2 on ENCODER WIDTH. Does making the part that SEES bigger move clearance?

`cnn_channels` has been **(16, 32, 32) in every arm this project has ever trained.** Block 53 widened
`d_model` and `n_layers` -- the reasoning -- and never the vision. So "a bigger model did not clear more"
has only ever been tested as a bigger thinker behind the same small eyes.

| | cnn (16,32,32) | cnn (32,64,64) |
|---|---|---|
| **84x84** | **B** 172,284 | **P** 325,964 |
| **128x128** | **R** 366,844 | **V** 715,084 |

* **P − B** = vision width at 84x84
* **V − R** = vision width at 128x128
* **V − P** = resolution at matched *encoder width*

**⚠ One correction to the directive's framing: V − P is NOT parameter-matched.** Raising resolution raises
the encoder's flatten width, so V has 715,084 parameters against P's 325,964 -- a 2.2x gap. What V − P holds
fixed is the *encoder channel width*, not the parameter count. A genuinely parameter-matched resolution
control is not available from this factorial and would need a third encoder setting chosen to equalise
counts. Reported as measured, labelled as what it is.

Evaluated at **T=1.0** (comparable to every prior block) and at **T=0.7**, the best rung from §2 -- best by
median pipe-2 clearance across the three B seeds (65.5 vs 63.5 at T=1.0). Two seeds per new arm, because
block 54's single-seed arms are why three of block 53's claims did not stand.
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
OUT = ROOT / "data/vision_2x2.json"
PRIOR = [ROOT / "data/temperature_ladder.json", ROOT / "data/generation_sweep.json",
         ROOT / "data/generation_seeds.json"]

TEMPS = [1.0, 0.7]
#: cell -> checkpoints (seed 0, seed 1). B/R seed-1 partners differ in name, mapped explicitly.
CELLS = {
    "B_84_cnn16":  ["B_84_d64_L1", "B_84_seed1"],
    "P_84_cnn32":  ["P_84_cnn32", "P_84_cnn32_seed1"],
    "R_128_cnn16": ["R_128_d64_L1"],
    "V_128_cnn32": ["V_128_cnn32", "V_128_cnn32_seed1"],
}
NEW = ["P_84_cnn32", "P_84_cnn32_seed1", "V_128_cnn32", "V_128_cnn32_seed1"]
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
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.update({
        "n_eval": N_EVAL, "measurement_basis": "single_life", "temps": TEMPS,
        "best_rung_from_section2": 0.7,
        "generation_rule": "capped (non-A <= 4), A-runs UNCAPPED",
        "factorial": {c: v for c, v in CELLS.items()},
        "parameter_counts": {"B_84_cnn16": 172284, "P_84_cnn32": 325964,
                             "R_128_cnn16": 366844, "V_128_cnn32": 715084},
        "V_minus_P_caveat": ("holds ENCODER CHANNEL WIDTH fixed, not parameter count: V has 715,084 "
                             "parameters against P's 325,964 because resolution changes the flatten "
                             "width. Not the parameter-matched control; that needs a third encoder "
                             "setting chosen to equalise counts."),
        "episode0_guard": "compose.warm_session before every scored episode"})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    print("=== 2x2 on encoder width ===", flush=True)
    for cell, ckpts in CELLS.items():
        for name in ckpts:
            ck = None
            for T in TEMPS:
                k = G.tag(name, None, T)
                if k in out["arms"]:
                    continue
                if k in prior:
                    out["arms"][k] = {**prior[k], "source": "reused", "cell": cell}
                    save()
                    continue
                if ck is None:
                    ck = get_ck(name)
                rec = G.run_arm(sess_get, ck, None, T, ctx, start, n=N_EVAL)
                rec.update({"source": "block 55", "cell": cell})
                out["arms"][k] = rec
                save()

    # --------- cell means (pooled over the cell's seeds) ---------
    def cell_stats(cell, T, ob):
        vals, ns = [], 0
        for name in CELLS[cell]:
            a = out["arms"].get(G.tag(name, None, T))
            if a:
                vals.append(a["clearance"][ob]["rate"])
                ns += a["n"]
        return (float(np.mean(vals)) if vals else None, len(vals), ns)

    out["cells"] = {}
    for T in TEMPS:
        for cell in CELLS:
            for ob in ("pipe1", "pipe2", "pipe3", "pipe4"):
                m, nseeds, ntot = cell_stats(cell, T, ob)
                out["cells"].setdefault(f"T{T}", {}).setdefault(cell, {})[ob] = {
                    "mean_rate": m, "n_seeds": nseeds, "n_episodes_total": ntot}

    print(f"\n{'cell':14s}{'params':>9s}" +
          "".join(f"{f'{ob} T{T}':>13s}" for T in TEMPS for ob in ("pipe2", "pipe3")))
    for cell in CELLS:
        cells = [out["cells"][f"T{T}"][cell][ob]["mean_rate"]
                 for T in TEMPS for ob in ("pipe2", "pipe3")]
        print(f"{cell:14s}{out['parameter_counts'][cell]:>9,}" +
              "".join(f"{(v*100 if v is not None else float('nan')):>13.1f}" for v in cells),
              flush=True)

    # --------- the three contrasts ---------
    def contrast(hi, lo, T, ob):
        a, na, ea = cell_stats(hi, T, ob)
        b, nb, eb = cell_stats(lo, T, ob)
        if a is None or b is None:
            return None
        ka, kb = int(round(a * ea)), int(round(b * eb))
        loci, hici = diff_ci(kb, eb, ka, ea)
        return {"hi_cell": hi, "lo_cell": lo, "hi_rate": a, "lo_rate": b,
                "difference_pp": (a - b) * 100, "ci_pp": [loci * 100, hici * 100],
                "method": "Newcombe on pooled episodes", "n_seeds_hi": na, "n_seeds_lo": nb,
                "n_episodes_hi": ea, "n_episodes_lo": eb,
                "⚠": ("pooling seeds inside a cell narrows the interval but does NOT remove "
                      "training-seed variance; the ledger's clearance seed band is 14.5-24.5 pp and "
                      "each cell has at most 2 seeds")}
    out["contrasts"] = {}
    for T in TEMPS:
        for label, hi, lo in (("P_minus_B__vision_at_84", "P_84_cnn32", "B_84_cnn16"),
                              ("V_minus_R__vision_at_128", "V_128_cnn32", "R_128_cnn16"),
                              ("V_minus_P__resolution_at_matched_encoder", "V_128_cnn32",
                               "P_84_cnn32")):
            for ob in ("pipe2", "pipe3"):
                c = contrast(hi, lo, T, ob)
                if c:
                    out["contrasts"].setdefault(f"T{T}", {}).setdefault(label, {})[ob] = c

    print("\ncontrasts (pp, Newcombe on pooled episodes):")
    for T in TEMPS:
        for label, obs in out["contrasts"].get(f"T{T}", {}).items():
            s = "  ".join(f"{ob} {v['difference_pp']:+.1f}[{v['ci_pp'][0]:+.1f},{v['ci_pp'][1]:+.1f}]"
                          for ob, v in obs.items())
            print(f"  T={T} {label:44s} {s}", flush=True)

    # --------- binary question part 2 ---------
    def sig(label, ob):
        best = None
        for T in TEMPS:
            c = out["contrasts"].get(f"T{T}", {}).get(label, {}).get(ob)
            if c and (c["ci_pp"][0] > 0 or c["ci_pp"][1] < 0):
                best = c if best is None or abs(c["difference_pp"]) > abs(best["difference_pp"]) else best
        return best
    moved = {lab: {ob: sig(lab, ob) for ob in ("pipe2", "pipe3")}
             for lab in ("P_minus_B__vision_at_84", "V_minus_R__vision_at_128")}
    helped = [(lab, ob, c) for lab, d in moved.items() for ob, c in d.items()
              if c and c["difference_pp"] > 0]
    hurt = [(lab, ob, c) for lab, d in moved.items() for ob, c in d.items()
            if c and c["difference_pp"] < 0]
    out["binary_question_part2"] = {
        "encoder_width_moves_clearance": bool(helped or hurt),
        "helped": [f"{lab}/{ob} {c['difference_pp']:+.1f} pp" for lab, ob, c in helped],
        "hurt": [f"{lab}/{ob} {c['difference_pp']:+.1f} pp" for lab, ob, c in hurt]}
    if helped:
        out["verdict"] = (
            f"**WIDENING THE ENCODER HELPS: {'; '.join(out['binary_question_part2']['helped'])}.** "
            f"Every previous 'bigger model' arm widened the reasoning and left the vision at "
            f"(16,32,32), so this was untested until now.")
    elif hurt:
        out["verdict"] = (
            f"**WIDENING THE ENCODER HURTS: {'; '.join(out['binary_question_part2']['hurt'])}.** "
            f"More vision capacity on this corpus makes clearance worse, not merely flat.")
    else:
        out["verdict"] = (
            "**WIDENING THE ENCODER DOES NOTHING MEASURABLE.** No vision contrast has an interval "
            "excluding zero at pipe 2 or pipe 3, at either temperature. Combined with §2's declining "
            "ladder: the sampling rule is not the lever, resolution is not the lever, and vision width "
            "is not the lever either.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
