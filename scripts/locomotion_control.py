"""P1(a) + P1(b): is the ~0.62 Right threshold about the pipe, or just about walking?

(a) The missing trivial baseline. Run the identical i.i.d. Right dose-response on flat ground
    with no pipe and no jump, scoring forward travel. If travel is a smooth function of the
    Right rate with a knee in the same place, then the pipe-2 dose-response was measuring how
    fast Mario walks and the obstacle is incidental.

(b) Reissue the structured conditions with consistent accounting. The previous table reported
    `right_frames_in_flight` as `min(total_right_frames, 40)`, which is a whole-window total
    clamped to 40 rather than a count over the first 40 frames — so `alt` printed 40 where the
    honest first-40 count is 20, and `full` printed 40 where it executed 78 frames before
    dying. Every condition here reports frames executed, Right frames, the realized rate, and
    the longest run, all over the window actually run.

Flat ground is the opening of 1-1: from the level start at x=40 there is clear ground until the
first Goomba near x=300, so an 80-frame window at a maximum 2.5 px/f cannot reach it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.partial_right import A_HOLD, CKPT, FLIGHT, right_on  # noqa: E402
from scripts.pipe2_sweep import PIPE2_CLEARED_X, policy_prefix, replay_prefix  # noqa: E402
from scripts.standstill_geometry import settle  # noqa: E402
from tasdata.bc.overnight_lib import calibrate, load_policy, wilson  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/locomotion_control.json"
RIGHT, B, A = NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["A"]
WALK_FRAMES = 80
RATES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]
SEEDS = 20
POST = 300


def walk(session, start, *, rate: float, seed: int, jump: bool = False):
    """Flat ground, no pipe: how far does Mario travel at this Right rate?"""
    obs = session.reset(start.frame)
    st = read_smb(obs.ram, obs.framecount)
    x0 = st.x_position
    rng = np.random.default_rng(seed)
    right_frames = run = longest = 0
    a_left = A_HOLD if jump else 0
    died = False
    x = x0
    for i in range(WALK_FRAMES):
        byte = B
        if rng.random() < rate:
            byte |= RIGHT
            right_frames += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
        if a_left > 0:
            byte |= A
            a_left -= 1
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        x = st.x_position
        if st.player_state in (0x06, 0x0B):
            died = True
            break
    return {"rate": rate, "seed": seed, "x0": int(x0), "x_end": int(x),
            "distance": int(x - x0), "frames": i + 1,
            "right_frames": right_frames, "realized_rate": right_frames / (i + 1),
            "longest_run": longest, "died": died,
            "px_per_frame": (x - x0) / (i + 1)}


def structured(session, start, seq, *, mode: str, seed: int = 0):
    """P1(b): the pipe-2 structured conditions with honest, full-window accounting."""
    obs = replay_prefix(session, start, seq)
    obs, log = settle(session, obs, "at_pipe")
    rng = np.random.default_rng(seed)
    maxx = log["x_before_jump"]
    a_left = A_HOLD
    right_frames = run = longest = 0
    died = False
    executed = 0
    for i in range(POST):
        byte = B
        if right_on(mode, i, rng):
            byte |= RIGHT
            right_frames += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
        if a_left > 0:
            byte |= A
            a_left -= 1
        obs = session.step(byte)
        executed = i + 1
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if st.player_state in (0x06, 0x0B):
            died = True
            break
    right_first40 = sum(1 for i in range(min(FLIGHT, executed))
                        if right_on(mode, i, np.random.default_rng(seed)))
    return {"mode": mode, "frames_executed": executed, "right_frames_total": right_frames,
            "realized_right_rate": right_frames / executed,
            "right_frames_first40_deterministic": (right_first40
                                                   if not mode.startswith("iid") else None),
            "longest_consecutive_right": longest, "max_x": int(maxx),
            "cleared_pipe2": bool(maxx > PIPE2_CLEARED_X), "died": died}


def main() -> None:
    ctx = O.Ctx()
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    loco, struct = [], []
    with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
        print(f"(a) LOCOMOTION on flat ground, {WALK_FRAMES} frames from x={start.x}, "
              f"B held, no jump\n")
        for rate in RATES:
            rs = [walk(session, start, rate=rate, seed=s) for s in range(SEEDS)]
            loco += rs
            d = [r["distance"] for r in rs]
            print(f"  Right p={rate:.2f}  distance mean {np.mean(d):6.1f} px  "
                  f"median {np.median(d):6.1f}  min {min(d):4d} max {max(d):4d}  "
                  f"px/frame {np.mean([r['px_per_frame'] for r in rs]):.3f}  "
                  f"longest_run {np.mean([r['longest_run'] for r in rs]):5.1f}  "
                  f"deaths {sum(r['died'] for r in rs)}")

        print("\n(b) STRUCTURED pipe-2 conditions, honest full-window accounting\n")
        seq = None
        for sd in range(12):
            seq = policy_prefix(session, policy, cfg, thr, start, seed=sd)
            if seq:
                break
        for mode in ("full", "none", "alt", "first_half", "second_half"):
            r = structured(session, start, seq, mode=mode)
            struct.append(r)
            print(f"  {mode:12s} frames={r['frames_executed']:3d} "
                  f"right={r['right_frames_total']:3d} "
                  f"rate={r['realized_right_rate']:.3f} "
                  f"first40={r['right_frames_first40_deterministic']} "
                  f"longest_run={r['longest_consecutive_right']:3d} "
                  f"max_x={r['max_x']:4d} cleared={r['cleared_pipe2']} died={r['died']}")

    by_rate = {}
    for rate in RATES:
        rs = [r for r in loco if r["rate"] == rate]
        d = [r["distance"] for r in rs]
        by_rate[f"{rate:.2f}"] = {
            "n": len(rs), "mean_distance": float(np.mean(d)),
            "median_distance": float(np.median(d)), "min": int(min(d)), "max": int(max(d)),
            "mean_px_per_frame": float(np.mean([r["px_per_frame"] for r in rs])),
            "mean_longest_run": float(np.mean([r["longest_run"] for r in rs])),
            "deaths": int(sum(r["died"] for r in rs))}

    d45 = by_rate["0.45"]["mean_distance"]
    d70 = by_rate["0.70"]["mean_distance"]
    ratio = d45 / d70 if d70 else 0.0
    # Smoothness: is there a cliff between 0.60 and 0.65 like the pipe result, or a smooth ramp?
    steps = [(RATES[i + 1], by_rate[f'{RATES[i+1]:.2f}']["mean_distance"]
              - by_rate[f'{RATES[i]:.2f}']["mean_distance"]) for i in range(len(RATES) - 1)]
    answer = (
        f"YES -- p=0.45 travels {d45:.0f} px against {d70:.0f} px at p=0.70 "
        f"({ratio * 100:.0f}% as far). Locomotion itself scales strongly with the Right rate, "
        f"so the pipe-2 dose-response is confounded with walking speed and the 0.60-0.65 cliff "
        f"needs reinterpreting."
        if ratio < 0.75 else
        f"NO -- p=0.45 travels {d45:.0f} px against {d70:.0f} px at p=0.70 "
        f"({ratio * 100:.0f}% as far), i.e. comparable. Forward travel is not what separates "
        f"the rates, so the pipe-2 threshold is a property of the obstacle.")
    print("\n" + "=" * 78)
    print(f"BINARY: {answer}")
    print("\nper-step distance increments (is there a cliff at 0.60->0.65?):")
    for rate, dd in steps:
        print(f"  ->{rate:.2f}: {dd:+7.1f} px")

    OUT.write_text(json.dumps(
        {"walk_frames": WALK_FRAMES, "seeds": SEEDS, "locomotion_by_rate": by_rate,
         "distance_increments": [{"to_rate": r, "delta_px": d} for r, d in steps],
         "ratio_045_over_070": ratio, "binary_answer": answer,
         "structured_reissued": struct, "locomotion_rows": loco}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
