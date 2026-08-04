"""§2: the per-start reach table -- what the canonical fixed-rate script achieves from each state.

Credit a rollout by where it lands in the script's own `max_x` distribution **from the same start**. No
terrain knowledge, no per-level thresholds, and not gameable by the button marginal, because the script
already has the best marginal there is.

One canonical script rather than the per-obstacle maximum, per the directive: `match_top20`
(A 0.848, Left 0.136, Down 0.086), which matches a real checkpoint's marginals on every button. Taking a
maximum over arms would multiply emulator cost by the number of arms for a slightly stronger bar; a single
un-gameable bar is the property that matters.

Cost is what makes this possible at all: each state's prefix is replayed **once**, snapshotted with
`session.save_scratch`, and then restored in O(1) for every rollout. Without that, every rollout would pay
its full prefix again -- for 72 states x 16 rollouts that is the difference between ~20 minutes and hours.

**The saturation risk is measured, not assumed.** A quantile is bounded on [0, 1], so it cannot separate
"slightly better than the script" from "far better." This script reports how often the quantile pins at
1.0, and writes a standardised margin `(max_x - script_median) / script_IQR` alongside it so the fallback
the directive asked for is already on disk if the quantile turns out flat.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "data/startlib_policy.json"
TRACES = ROOT / "data/traces/p1_200.json"
OUT = ROOT / "data/reach_table.json"

A, B, RIGHT = NES_BUTTON_BITS["A"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["Right"]
LEFT, DOWN = NES_BUTTON_BITS["Left"], NES_BUTTON_BITS["Down"]

#: The canonical opponent: `match_top20`'s measured marginals.
CANON = {"A": 0.848, "Left": 0.136, "Down": 0.086}
ROLLOUTS = 16
MAX_FRAMES, STALL = 400, 120


def script_rollout(session, seed: int, max_frames: int = MAX_FRAMES) -> dict:
    """Canonical fixed-rate script from the *currently restored* state."""
    rng = np.random.default_rng(seed)
    maxx = best = 0
    since = 0
    died = False
    frames = 0
    obs = None
    for i in range(max_frames):
        byte = RIGHT | B
        if rng.random() < CANON["A"]:
            byte |= A
        if rng.random() < CANON["Left"]:
            byte |= LEFT
        if rng.random() < CANON["Down"]:
            byte |= DOWN
        obs = session.step(byte)
        frames = i + 1
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if st.player_state in (0x06, 0x0B):
            died = True
            break
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                break
    return {"max_x": int(maxx), "died": died, "frames": frames}


def main() -> None:
    t0 = time.time()
    lib = json.loads(LIB.read_text())
    states = lib["states"]
    bytes_by_seed = {e["seed"]: [f[3] for f in e["frames"]]
                     for e in json.loads(TRACES.read_text())["episodes"]}
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    print(f"{len(states)} start states, canonical script {CANON}, {ROLLOUTS} rollouts each",
          flush=True)

    rows = {}
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    prefix_frames = rollout_frames = 0
    try:
        for si, stt in enumerate(states):
            seq = bytes_by_seed[stt["seed"]][:stt["frame_index"]]
            obs = s.reset(start.frame)
            for byte in seq:
                obs = s.step(byte)
            prefix_frames += len(seq)
            live = read_smb(obs.ram, obs.framecount)
            s.save_scratch(0)                       # replay the prefix once, then restore for free
            outs = []
            for r in range(ROLLOUTS):
                o = s.load_scratch(0)
                x_at_start = read_smb(o.ram, o.framecount).x_position
                res = script_rollout(s, seed=si * 1000 + r)
                res["max_x"] = max(res["max_x"], int(x_at_start))
                rollout_frames += res["frames"]
                outs.append(res)
            xs = np.array([o["max_x"] for o in outs], dtype=float)
            q1, q3 = np.percentile(xs, [25, 75])
            key = f"{stt['seed']}:{stt['frame_index']}"
            rows[key] = {
                "seed": stt["seed"], "frame_index": stt["frame_index"],
                "x_at_state": stt["x"], "x_after_restore": int(x_at_start), "bin": stt["bin"],
                "n": ROLLOUTS,
                "script_max_x": [int(v) for v in xs],
                "median": float(np.median(xs)), "q1": float(q1), "q3": float(q3),
                "iqr": float(q3 - q1), "min": float(xs.min()), "max": float(xs.max()),
                "deaths": sum(o["died"] for o in outs),
                "gain_median": float(np.median(xs) - stt["x"]),
            }
            if (si + 1) % 12 == 0 or si == len(states) - 1:
                print(f"  {si + 1}/{len(states)}  x={stt['x']:5d}  script median "
                      f"{np.median(xs):6.0f} (gain {np.median(xs) - stt['x']:+5.0f})  "
                      f"IQR {q3 - q1:5.0f}  deaths {sum(o['died'] for o in outs)}/{ROLLOUTS}",
                      flush=True)
    finally:
        s.close()

    iqrs = np.array([r["iqr"] for r in rows.values()])
    zero_iqr = int((iqrs == 0).sum())
    gains = np.array([r["gain_median"] for r in rows.values()])
    out = {
        "canonical_script": CANON, "rollouts_per_state": ROLLOUTS,
        "max_frames": MAX_FRAMES, "stall": STALL,
        "n_states": len(rows), "source_library": LIB.name,
        "credit": {
            "primary": "quantile of rollout max_x within this state's script_max_x list",
            "fallback": "(rollout_max_x - median) / IQR, for when the quantile saturates",
            "iqr_zero_states": zero_iqr,
            "note": ("a state whose IQR is 0 cannot use the standardised margin; the quantile still "
                     "works there, so the primary is the quantile and the margin is the fallback"),
        },
        "diagnostics": {
            "script_gain_median_over_states": float(np.median(gains)),
            "script_gain_min": float(gains.min()), "script_gain_max": float(gains.max()),
            "iqr_median": float(np.median(iqrs)), "iqr_zero_states": zero_iqr,
            "prefix_frames_replayed": prefix_frames,
            "rollout_frames": rollout_frames,
            "frames_saved_by_scratch": int(prefix_frames * (ROLLOUTS - 1)),
        },
        "states": rows,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nscript gain from these states: median {np.median(gains):+.0f} px "
          f"(range {gains.min():+.0f} to {gains.max():+.0f})")
    print(f"IQR median {np.median(iqrs):.0f}px; {zero_iqr}/{len(rows)} states have IQR 0")
    print(f"scratch savestates avoided {prefix_frames * (ROLLOUTS - 1):,} prefix frames "
          f"({prefix_frames:,} replayed instead of "
          f"{prefix_frames * ROLLOUTS:,})")
    print(f"wrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
