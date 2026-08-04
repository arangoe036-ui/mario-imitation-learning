"""P1: how much of the variance is *measurement* rather than training?

Nobody has ever evaluated the same checkpoint twice with different evaluation seeds, so the
measurement noise floor under every number in this project is unknown. One fixed checkpoint,
same weights, same calibrated thresholds, no retraining -- only the episode seeds change.

n=200 at p~0.54 predicts +/-6.9 pp from episode sampling alone. If the observed spread is near
that, the 14.5-24.5 pp seed spreads are mostly training instability. If it is near 15 pp, most of
what we called seed noise is just too few episodes, and the fix is bigger evaluations rather than
triplicate training.

Also emits cause x bin (P3), since these evaluations produce it for free.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import session_when_free
from scripts.single_life import episode, summarise
from tasdata.bc.overnight_lib import calibrate, load_policy, wilson

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/eval_noise.json"
CKPT = ROOT / "data/bc_compose_top20/top20_round2.pt"
N, BLOCKS = 200, 3

def main():
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    print(f"checkpoint {CKPT.name}; identical weights and thresholds in every block\n")

    reps = []
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        for b in range(BLOCKS):
            lo = b * N
            rows = [episode(s, policy, cfg, thr, start, seed) for seed in range(lo, lo + N)]
            r = summarise(rows)
            r["eval_seed_block"] = [lo, lo + N - 1]
            reps.append(r)
            print(f"  block {b} (seeds {lo}-{lo+N-1}): pipe1 {r['pipe1']['rate']*100:5.1f}%  "
                  f"pipe2 {r['pipe2']['rate']*100:5.1f}% [{r['pipe2']['ci'][0]*100:.1f},"
                  f"{r['pipe2']['ci'][1]*100:.1f}]  deaths {r['deaths']:3d}  "
                  f"x_med {r['x_median']:.0f}")
    finally:
        s.close()

    p2 = [r["pipe2"]["rate"] for r in reps]
    p1 = [r["pipe1"]["rate"] for r in reps]
    dd = [r["deaths"] for r in reps]
    spread = (max(p2) - min(p2)) * 100
    pbar = float(np.mean(p2))
    predicted = 1.96 * math.sqrt(pbar * (1 - pbar) / N) * 100 * 2   # full width of a 95% band
    print(f"\npipe 2 across blocks: " + "  ".join(f"{v*100:.1f}%" for v in p2))
    print(f"  observed spread        {spread:5.1f} pp")
    print(f"  predicted 95% width at n={N}, p={pbar:.2f}: {predicted:5.1f} pp")
    print(f"pipe 1 across blocks: " + "  ".join(f"{v*100:.1f}%" for v in p1))
    print(f"deaths across blocks: {dd}  spread {max(dd)-min(dd)}")

    # P3: cause x bin, pooled over the three blocks (same checkpoint, so pooling is valid here)
    merged = {}
    for r in reps:
        for b, c in r["cause_by_bin_32px"].items():
            for k, v in c.items():
                merged.setdefault(int(b), {}).setdefault(k, 0)
                merged[int(b)][k] += v
    print("\nP3 -- cause x bin, pooled over 3 blocks (600 episodes, same checkpoint):")
    for b in sorted(merged):
        tot = sum(merged[b].values())
        tag = "   <-- pipe 3 face" if 640 <= b <= 704 else (
              "   <-- Goomba" if b <= 320 else "")
        print(f"  x {b:5d}: {tot:3d}  {merged[b]}{tag}")
    face = {k: v for b in (640, 672, 704) for k, v in merged.get(b, {}).items()}
    face_tot = sum(v for b in (640, 672, 704) for v in merged.get(b, {}).values())
    enemy = sum(v for b in (640, 672, 704) for k, v in merged.get(b, {}).items()
                if k.startswith("enemy"))
    verdict_p3 = (f"pipe-3 face deaths: {face_tot} total, {enemy} enemy "
                  f"({enemy/face_tot*100:.0f}%)" if face_tot else "no deaths at the pipe-3 face")
    print(f"\n  {verdict_p3}")

    verdict = (f"MEASUREMENT-DOMINATED: same checkpoint spans {spread:.1f} pp across evaluation "
               f"seeds alone, close to the {predicted:.1f} pp predicted by episode sampling. "
               f"Most of the 14.5-24.5 pp 'seed noise' is too-few-episodes; the fix is larger n."
               if spread >= 0.6 * 14.5 else
               f"TRAINING-DOMINATED: same checkpoint spans only {spread:.1f} pp across evaluation "
               f"seeds against 14.5-24.5 pp across training seeds, so the training pipeline is "
               f"the unstable part and larger evaluations will not fix it.")
    print("\n" + "="*78); print(f"BINARY: {verdict}")
    OUT.write_text(json.dumps({"checkpoint": CKPT.name, "n_per_block": N,
                               "blocks": reps, "pipe2_rates": p2,
                               "observed_spread_pp": spread,
                               "predicted_95_width_pp": predicted,
                               "deaths": dd, "cause_by_bin_pooled": merged,
                               "pipe3_face": verdict_p3, "verdict": verdict},
                              indent=2, default=str))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
