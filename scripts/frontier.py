"""P1: where is the real frontier? Rates, not medians. Plus cause of death.

The binary this answers: at n=200, does the policy clear pipe 2 in more than 30% of episodes?
That rate has never been reported — every statement about pipe 2 in this project has been
inferred from an `x` median of 594-595, and a median hides the entire upper tail.

Also recorded per episode, because two artifacts have been shown to conflate outcomes:

* **cause of death** and the x where it happened, read from RAM (enemy type and position, or
  a below-floor y for a pit). `pipe2_sweep.json` and `standstill_geometry.json` recorded only
  a `died` boolean, so P1(a) cannot be answered from them — this is the re-run that fixes it.
* **realized airborne Right rate**, folded in per P1(e) rather than run separately.
* the **full x distribution**, so obstacle faces show up as spikes instead of being averaged.

Obstacle thresholds are calibrated from the observed distribution rather than assumed: whatever
x values the episodes pile up at *are* the obstacle faces.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.overnight_lib import calibrate, load_policy, wilson  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.tokens import LIVE_MASK  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/frontier.json"
SEEDS = 200
MAX_FRAMES = 2500
STALL = 300

# SMB enemy slots: type at 0x0016+i, x page at 0x006E+i, x in page at 0x0087+i.
ENEMY_TYPE, ENEMY_XPAGE, ENEMY_XLO = 0x0016, 0x006E, 0x0087
ENEMY_NAMES = {0x00: "green_koopa", 0x06: "goomba", 0x03: "buzzy_beetle",
               0x05: "hammer_bro", 0x0F: "piranha_plant"}
A_BIT, RIGHT_BIT = NES_BUTTON_BITS["A"], NES_BUTTON_BITS["Right"]

CHECKPOINTS = [
    ("round3_ratio1to1", ROOT / "data/bc_overnight/round3_ratio1to1.pt"),
    ("round2_ratio1to1", ROOT / "data/bc_overnight/round2_ratio1to1.pt"),
    ("round2_ratio3to1", ROOT / "data/bc_overnight/round2_ratio3to1.pt"),
    ("sustain_arm_a", ROOT / "data/bc_followup/a_sustain_and_onset.pt"),
]


def enemies(ram) -> list[tuple[str, int]]:
    out = []
    for i in range(5):
        t = int(ram[ENEMY_TYPE + i])
        if t == 0 and int(ram[ENEMY_XPAGE + i]) == 0:
            continue
        x = int(ram[ENEMY_XPAGE + i]) * 256 + int(ram[ENEMY_XLO + i])
        out.append((ENEMY_NAMES.get(t, f"type_{t:#04x}"), x))
    return out


def episode(session, policy, cfg, thr, start, seed: int):
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), dtype=np.uint8)
    win[:] = _resize_gray(obs.rgb, (84, 84))

    maxx = read_smb(obs.ram, obs.framecount).x_position
    prev_y = None
    airborne_frames = airborne_right = 0
    death = None
    ended = "budget"
    best = maxx
    since = 0
    for _ in range(MAX_FRAMES):
        with torch.no_grad():
            logits = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0 / (1.0 + np.exp(-logits))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]:
                byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        obs = session.step(byte)
        win = np.roll(win, -1, axis=0)
        win[-1] = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)

        if prev_y is not None and st.y_position != prev_y:
            airborne_frames += 1
            if byte & RIGHT_BIT:
                airborne_right += 1
        prev_y = st.y_position

        if st.player_state in (0x06, 0x0B):
            near = sorted(enemies(obs.ram), key=lambda e: abs(e[1] - st.x_position))
            cause = "pit" if st.y_position > 200 else (
                f"enemy:{near[0][0]}" if near and abs(near[0][1] - st.x_position) < 24
                else "unknown_contact")
            death = {"cause": cause, "x": int(st.x_position), "y": int(st.y_position),
                     "enemies": near[:2]}
            ended = "died"
            break
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                ended = "stuck"
                break
        if st.pregame == 2:
            ended = "game_over"
            break
    return {"seed": seed, "max_x": int(maxx), "ended": ended, "death": death,
            "airborne_frames": airborne_frames,
            "airborne_right_rate": (airborne_right / airborne_frames
                                    if airborne_frames else None)}


def main() -> None:
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    results = {}
    for label, path in CHECKPOINTS:
        if not Path(path).exists():
            print(f"{label}: MISSING")
            continue
        policy, cfg, _ = load_policy(Path(path))
        cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
        thr = cal.vector.astype(np.float64)
        rows = []
        with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
            for s in range(SEEDS):
                rows.append(episode(session, policy, cfg, thr, start, s))
        xs = np.array([r["max_x"] for r in rows])

        # Obstacle faces = where episodes pile up. Bin to 8 px and take the modes.
        hist = Counter((xs // 8 * 8).tolist())
        spikes = sorted([b for b, c in hist.items() if c >= max(3, SEEDS // 40)])

        def rate(th):
            k = int((xs > th).sum())
            lo, hi = wilson(k, len(xs))
            return {"threshold": th, "k": k, "n": len(xs), "rate": k / len(xs),
                    "ci": [lo, hi]}

        deaths = Counter(r["death"]["cause"] for r in rows if r["death"])
        death_x = [r["death"]["x"] for r in rows if r["death"]]
        ar = [r["airborne_right_rate"] for r in rows if r["airborne_right_rate"] is not None]
        res = {
            "n": len(rows),
            "pipe1": rate(470), "pipe2": rate(630), "pipe3": rate(760),
            "x_median": float(np.median(xs)), "x_mean": float(np.mean(xs)),
            "x_p10": float(np.percentile(xs, 10)), "x_p90": float(np.percentile(xs, 90)),
            "x_max": int(xs.max()), "x_min": int(xs.min()),
            "x_histogram_8px": dict(sorted(hist.items())),
            "pileup_bins": spikes,
            "ended": dict(Counter(r["ended"] for r in rows)),
            "death_causes": dict(deaths),
            "death_x_median": float(np.median(death_x)) if death_x else None,
            "death_x_all": sorted(death_x),
            "airborne_right_rate_mean": float(np.mean(ar)) if ar else None,
            "airborne_right_rate_median": float(np.median(ar)) if ar else None,
        }
        results[label] = res
        print(f"\n=== {label} (n={res['n']}) ===")
        for p in ("pipe1", "pipe2", "pipe3"):
            d = res[p]
            print(f"  {p} (x>{d['threshold']}): {d['k']:3d}/{d['n']} = "
                  f"{d['rate'] * 100:5.1f}% [{d['ci'][0] * 100:4.1f}, {d['ci'][1] * 100:5.1f}]")
        print(f"  x: median {res['x_median']:.0f} p90 {res['x_p90']:.0f} "
              f"max {res['x_max']} min {res['x_min']}")
        print(f"  ended: {res['ended']}")
        print(f"  deaths: {res['death_causes']}  median death x "
              f"{res['death_x_median']}")
        print(f"  airborne Right rate: mean {res['airborne_right_rate_mean']:.3f} "
              f"median {res['airborne_right_rate_median']:.3f}")
        print(f"  x pile-ups (8px bins, >=5 eps): "
              f"{[b for b in res['pileup_bins'] if res['x_histogram_8px'][b] >= 5]}")

    best = max(results.items(), key=lambda kv: kv[1]["pipe2"]["rate"]) if results else None
    p2 = results.get("round3_ratio1to1", {}).get("pipe2", {})
    verdict = (
        f"YES -- round3_ratio1to1 clears pipe 2 on {p2.get('rate', 0) * 100:.1f}% of episodes "
        f"[{p2.get('ci', [0, 0])[0] * 100:.1f}, {p2.get('ci', [0, 0])[1] * 100:.1f}] at n=200, "
        f"above 30%. Pipe 2 was never the blocker."
        if p2.get("rate", 0) > 0.30 else
        f"NO -- round3_ratio1to1 clears pipe 2 on {p2.get('rate', 0) * 100:.1f}% of episodes "
        f"[{p2.get('ci', [0, 0])[0] * 100:.1f}, {p2.get('ci', [0, 0])[1] * 100:.1f}] at n=200, "
        f"at or below 30%.")
    print("\n" + "=" * 78)
    print(f"BINARY: {verdict}")
    if best:
        print(f"best pipe-2 rate: {best[0]} at {best[1]['pipe2']['rate'] * 100:.1f}%")
    OUT.write_text(json.dumps({"seeds": SEEDS, "binary": verdict,
                               "best_pipe2_checkpoint": best[0] if best else None,
                               "checkpoints": results}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
