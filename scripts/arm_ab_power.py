"""Arm A vs arm B at n=200 per arm per start, with confidence intervals.

At n=20 the pipe-1 rates were 45% and 40% -- 9 episodes against 8, with a Wilson interval
on 9/20 spanning roughly 26-66%. That is underpowered, not a null result. This reruns the
one selection rule that actually clears the pipe (per-button sampling) at n=200 on 1-1 and
2-1, and reports intervals so "no difference" can be distinguished from "no power".
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.session_player import play_episode  # noqa: E402
from tasdata.bc.statelib import load_index  # noqa: E402
from tasdata.bc.tokens import ActionVocab  # noqa: E402
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "smb.nes"
MOVIE = ROOT / "data/movies/happylee_mars608-smb-warpless.fm2"
OUT = ROOT / "data/stage2_power.json"

LEVELS = ["1-1", "2-1"]
SEEDS = 200
ARMS = {
    "A_bernoulli_only": ROOT / "data/bc3/A_bernoulli_only_step3000_recal.pt",
    "B_bernoulli_onset10x": ROOT / "data/bc3/B_bernoulli_onset10x_step3000_recal.pt",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- behaves at the extremes where normal approximation does not."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def boot_ci(values: list[float], reps: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the median."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    meds = np.median(rng.choice(arr, size=(reps, arr.size), replace=True), axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def diff_ci(ka: int, na: int, kb: int, nb: int, z: float = 1.96) -> tuple[float, float]:
    """Newcombe interval for the difference of two proportions (B minus A)."""
    la, ua = wilson(ka, na, z)
    lb, ub = wilson(kb, nb, z)
    pa, pb = ka / na, kb / nb
    lo = (pb - pa) - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2)
    hi = (pb - pa) + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)
    return (lo, hi)


def load_policy(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()
    thr = blob["thresholds"]
    return policy, cfg, np.array([thr[n] for n in NES_BUTTON_ORDER], dtype=np.float64)


def main() -> None:
    vocab = ActionVocab.load(ROOT / "data/action_vocab.json")
    expert_bytes = set(json.loads((ROOT / "data/expert_bytes.json").read_text()))
    _, points = load_index(ROOT / "data/state_index.json")
    starts = [
        next(p for p in points if p.label == lv and p.kind == "level_start") for lv in LEVELS
    ]
    print(f"starts: {[f'{s.label}@{s.frame} x={s.x}' for s in starts]}")
    print(f"{SEEDS} seeds per arm per start, per-button sampling\n")

    results: dict = {"seeds": SEEDS, "levels": LEVELS, "arms": {}}
    t_start = time.perf_counter()
    with FceuxSession(ROM, MOVIE, sorted({s.frame for s in starts})) as session:
        for arm, ckpt in ARMS.items():
            policy, cfg, thresholds = load_policy(ckpt)
            per_level = {}
            for start in starts:
                eps = []
                t0 = time.perf_counter()
                for seed in range(SEEDS):
                    eps.append(
                        play_episode(
                            session, policy, start, vocab, seed=seed, selection="sample",
                            thresholds=thresholds, head_type=cfg.head_type,
                            stack=cfg.stack, expert_bytes=expert_bytes,
                        )
                    )
                xs = [e.furthest_x for e in eps]
                k1 = sum(e.cleared_pipe1 for e in eps)
                k2 = sum(e.cleared_pipe2 for e in eps)
                per_level[start.label] = {
                    "n": len(eps),
                    "pipe1_k": k1,
                    "pipe1_rate": k1 / len(eps),
                    "pipe1_ci": wilson(k1, len(eps)),
                    "pipe2_k": k2,
                    "pipe2_rate": k2 / len(eps),
                    "x_median": float(np.median(xs)),
                    "x_median_ci": boot_ci(xs),
                    "x_mean": float(np.mean(xs)),
                    "x_p90": float(np.percentile(xs, 90)),
                    "deaths_mean": float(np.mean([e.deaths for e in eps])),
                    "a_presses_median": float(np.median([e.a_presses for e in eps])),
                    "longest_a_hold_max": int(max(e.longest_a_hold for e in eps)),
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                d = per_level[start.label]
                print(
                    f"  {arm:22s} {start.label}  pipe1 {d['pipe1_k']:3d}/{d['n']:3d} "
                    f"= {d['pipe1_rate'] * 100:5.1f}% "
                    f"[{d['pipe1_ci'][0] * 100:4.1f}, {d['pipe1_ci'][1] * 100:4.1f}]  "
                    f"x_med {d['x_median']:6.0f} "
                    f"[{d['x_median_ci'][0]:.0f}, {d['x_median_ci'][1]:.0f}]  "
                    f"{d['seconds']:5.1f}s"
                )
            results["arms"][arm] = per_level
        print()

    print("Difference, arm B minus arm A (Newcombe 95% interval):")
    a, b = results["arms"]["A_bernoulli_only"], results["arms"]["B_bernoulli_onset10x"]
    for lv in LEVELS:
        lo, hi = diff_ci(a[lv]["pipe1_k"], a[lv]["n"], b[lv]["pipe1_k"], b[lv]["n"])
        delta = b[lv]["pipe1_rate"] - a[lv]["pipe1_rate"]
        verdict = "excludes zero" if lo > 0 or hi < 0 else "includes zero"
        print(f"  {lv} pipe1: {delta * 100:+5.1f} pp  "
              f"[{lo * 100:+5.1f}, {hi * 100:+5.1f}]  {verdict}")
        results.setdefault("difference", {})[lv] = {
            "delta_pipe1": delta, "ci": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)
        }

    results["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\ntotal {results['wall_seconds']:.0f}s -> wrote {OUT}")


if __name__ == "__main__":
    main()
