"""P1: does *partial* Right in flight clear pipe 2, or does it need consecutive frames?

The previous experiment tested horizontal input as all-or-nothing, which cannot distinguish
"the architecture cannot sustain a button" from "the policy is not competent at this". In-air
horizontal acceleration in SMB accumulates, and accumulation does not obviously require
consecutive frames.

So: same harness, same dead stop flush against pipe 2, A held 11 frames, and Right applied on
a *fraction* of the flight. If i.i.d. Right at the policy's own marginal rate (p=0.45) clears
at an appreciable rate, then per-frame independent sampling already produces the required
input and the "output parameterisation is the blocker" story is withdrawn.

B is held throughout in every condition, exactly as in the `at_pipe` condition it is being
compared against, so the only variable is Right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.pipe2_sweep import PIPE2_CLEARED_X, policy_prefix, replay_prefix  # noqa: E402
from scripts.standstill_geometry import settle  # noqa: E402
from tasdata.bc.overnight_lib import calibrate, load_policy, wilson  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/partial_right.json"
CKPT = ROOT / "data/bc_overnight/round3_ratio1to1.pt"

RIGHT, B, A = NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["A"]
A_HOLD = 11              # the minimum that clears from a dead stop with full Right
POST = 300
FLIGHT = 40              # nominal flight length, for the first-half / second-half split
ADDR_X_SPEED = 0x0057


def right_on(mode: str, i: int, rng) -> bool:
    if mode == "full":
        return True
    if mode == "none":
        return False
    if mode == "alt":
        return i % 2 == 0
    if mode == "first_half":
        return i < FLIGHT // 2
    if mode == "second_half":
        return FLIGHT // 2 <= i < FLIGHT
    if mode.startswith("iid:"):
        return rng.random() < float(mode.split(":")[1])
    raise ValueError(mode)


def one(session, start, seq, *, mode: str, seed: int = 0):
    obs = replay_prefix(session, start, seq)
    obs, log = settle(session, obs, "at_pipe")
    rng = np.random.default_rng(seed)
    takeoff_x = log["x_before_jump"]
    maxx = takeoff_x
    a_left = A_HOLD
    right_frames = 0
    longest_right_run = run = 0
    died = False
    for i in range(POST):
        byte = B
        if right_on(mode, i, rng):
            byte |= RIGHT
            right_frames += 1
            run += 1
            longest_right_run = max(longest_right_run, run)
        else:
            run = 0
        if a_left > 0:
            byte |= A
            a_left -= 1
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if st.player_state in (0x06, 0x0B):
            died = True
            break
    return {"mode": mode, "seed": seed, "takeoff_x": takeoff_x,
            "takeoff_speed_byte": log["speed_byte_before_jump"],
            "right_frames_in_flight": min(right_frames, FLIGHT),
            "longest_consecutive_right": longest_right_run,
            "max_x": int(maxx), "cleared_pipe2": bool(maxx > PIPE2_CLEARED_X),
            "died": died}


def main() -> None:
    ctx = O.Ctx()
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    print(f"policy realized Right rate (calibration): {cal.realized_rate['Right']:.4f}, "
          f"expert target {cal.target_rate['Right']:.4f}")

    plan = ([("full", [0]), ("none", [0]), ("alt", [0]),
             ("first_half", [0]), ("second_half", [0])]
            + [("iid:0.45", list(range(20))), ("iid:0.70", list(range(20)))])

    rows = []
    with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
        seq = None
        for seed in range(12):
            seq = policy_prefix(session, policy, cfg, thr, start, seed=seed)
            if seq:
                break
        if not seq:
            raise SystemExit("policy never reached handover")

        for mode, seeds in plan:
            rs = [one(session, start, seq, mode=mode, seed=s) for s in seeds]
            rows += rs
            k = sum(r["cleared_pipe2"] for r in rs)
            n = len(rs)
            lo, hi = wilson(k, n)
            print(f"  {mode:12s} n={n:2d}  cleared {k:2d}/{n:2d} = {100 * k / n:5.1f}% "
                  f"[{lo * 100:4.1f}, {hi * 100:5.1f}]  "
                  f"right_frames(mean of first {FLIGHT}) "
                  f"{np.mean([r['right_frames_in_flight'] for r in rs]):4.1f}  "
                  f"longest_run {np.mean([r['longest_consecutive_right'] for r in rs]):5.1f}  "
                  f"max_x median {np.median([r['max_x'] for r in rs]):.0f}")

    def cell(mode):
        rs = [r for r in rows if r["mode"] == mode]
        k, n = sum(r["cleared_pipe2"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        return {"n": n, "cleared": k, "rate": k / n, "ci": [lo, hi],
                "mean_right_frames_in_flight":
                    float(np.mean([r["right_frames_in_flight"] for r in rs])),
                "mean_longest_consecutive_right":
                    float(np.mean([r["longest_consecutive_right"] for r in rs])),
                "median_max_x": float(np.median([r["max_x"] for r in rs]))}

    per = {m: cell(m) for m, _ in plan}
    iid = per["iid:0.45"]
    kill = iid["rate"] >= 0.10 and iid["ci"][0] > 0
    survives = iid["cleared"] == 0 and iid["ci"][1] < 0.14
    verdict = (
        "ARCHITECTURAL FRAMING DEAD: i.i.d. Right at p=0.45 clears pipe 2 at "
        f"{iid['rate'] * 100:.1f}% [{iid['ci'][0] * 100:.1f}, {iid['ci'][1] * 100:.1f}], "
        "so per-frame independent sampling already produces the required input. The blocker "
        "is competence, not the output parameterisation."
        if kill else
        ("ARCHITECTURAL FRAMING SURVIVES P1: i.i.d. Right at p=0.45 cleared "
         f"{iid['cleared']}/{iid['n']}, 95% upper bound {iid['ci'][1] * 100:.1f}% < 14%. "
         "Consecutive frames appear to be required."
         if survives else
         f"INCONCLUSIVE against the pre-committed gate: i.i.d. p=0.45 cleared "
         f"{iid['cleared']}/{iid['n']} = {iid['rate'] * 100:.1f}% "
         f"[{iid['ci'][0] * 100:.1f}, {iid['ci'][1] * 100:.1f}] -- neither >=10% with the "
         f"interval excluding zero, nor 0 with an upper bound below 14%."))

    print("\n" + "=" * 78)
    print(verdict)
    OUT.write_text(json.dumps(
        {"checkpoint": CKPT.name, "a_hold": A_HOLD, "flight_window": FLIGHT,
         "policy_realized_right_rate": float(cal.realized_rate["Right"]),
         "expert_right_rate": float(cal.target_rate["Right"]),
         "per_condition": per, "verdict": verdict,
         "kill_condition_met": bool(kill), "framing_survives": bool(survives),
         "rows": rows}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
