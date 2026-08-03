"""P1: what was the standstill arm of pipe2_sweep actually testing?

`data/pipe2_sweep.json` records only (hold, trigger, with_b, standstill, max_x,
cleared_pipe2, takeoff_velocity, died). It does **not** record where Mario was when A was
pressed, whether he was touching pipe 2, or what horizontal input was applied in flight, so
the advisor's binary cannot be answered from it. This re-runs only that arm with those things
logged, plus the two trivial controls the sweep never had.

Four takeoff conditions, because "standstill" in the original run was none of the interesting
ones:

* ``as_run``     -- reproduces the original: coast until three consecutive integer x values
                    match, then press A with Right+B held through the whole flight.
* ``true_rest``  -- coast until the *velocity byte* 0x0057 reads 0, then the same.
* ``at_pipe``    -- hold Right into pipe 2 until x stops rising, so Mario is flush against
                    it, then jump with Right+B held.
* ``at_pipe_noh``-- flush against pipe 2, jump with **no horizontal input at all** during
                    flight.

``at_pipe_noh`` is the condition that matches the policy's situation. ``as_run`` is what was
actually measured and reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.overnight_lib import calibrate, load_policy  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from scripts.pipe2_sweep import HANDOVER, PIPE2_CLEARED_X, policy_prefix, replay_prefix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/standstill_geometry.json"
CKPT = ROOT / "data/bc_overnight/round3_ratio1to1.pt"

RIGHT, B, A = NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["A"]
ADDR_X_SPEED = 0x0057
CONTACT_X = 585          # the advisor's threshold for "in contact with pipe 2"
HOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
POST = 300


def speed(obs) -> int:
    v = int(obs.ram[ADDR_X_SPEED])
    return v - 256 if v > 127 else v


def settle(session, obs, mode):
    """Bring Mario to the requested pre-jump condition. Returns (obs, log)."""
    xs = [read_smb(obs.ram, obs.framecount).x_position]
    frames = 0
    if mode == "as_run":
        for _ in range(90):
            obs = session.step(0)
            xs.append(read_smb(obs.ram, obs.framecount).x_position)
            frames += 1
            if len(xs) >= 3 and xs[-1] == xs[-2] == xs[-3]:
                break
    elif mode == "true_rest":
        for _ in range(240):
            obs = session.step(0)
            xs.append(read_smb(obs.ram, obs.framecount).x_position)
            frames += 1
            if speed(obs) == 0:
                break
    elif mode in ("at_pipe", "at_pipe_noh"):
        # Walk into the pipe (no B) until x stops rising for 8 frames.
        stall = 0
        for _ in range(400):
            obs = session.step(RIGHT)
            x = read_smb(obs.ram, obs.framecount).x_position
            frames += 1
            stall = stall + 1 if xs and x <= xs[-1] else 0
            xs.append(x)
            if stall >= 8:
                break
    return obs, {"settle_frames": frames, "x_before_jump": int(xs[-1]),
                 "speed_byte_before_jump": speed(obs),
                 "recent_dx": [int(b - a) for a, b in zip(xs[-5:], xs[-4:])]}


def one(session, start, seq, *, hold, mode, control=None):
    obs = replay_prefix(session, start, seq)
    obs, log = settle(session, obs, mode)
    horiz_in_flight = mode != "at_pipe_noh"
    base = 0 if mode == "at_pipe_noh" else (RIGHT | B)

    takeoff_x = log["x_before_jump"]
    takeoff_speed = log["speed_byte_before_jump"]
    xs = [takeoff_x]
    maxx = takeoff_x
    died = False
    left = hold
    airborne_h_frames = 0
    for i in range(POST):
        byte = base
        if control == "no_jump":
            pass
        elif control == "always_a":
            byte |= A
        elif left > 0:
            byte |= A
            left -= 1
        if byte & (RIGHT | NES_BUTTON_BITS["Left"]):
            airborne_h_frames += 1
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        xs.append(st.x_position)
        maxx = max(maxx, st.x_position)
        if st.player_state in (0x06, 0x0B):
            died = True
            break
    return {"mode": mode, "hold": hold, "control": control,
            "takeoff_x": takeoff_x, "takeoff_speed_byte": takeoff_speed,
            "in_contact_with_pipe2": bool(takeoff_x >= CONTACT_X),
            "horizontal_input_in_flight": horiz_in_flight,
            "frames_with_horizontal_input": airborne_h_frames,
            "max_x": int(maxx), "cleared_pipe2": bool(maxx > PIPE2_CLEARED_X),
            "died": died, **{k: v for k, v in log.items() if k != "x_before_jump"}}


def main() -> None:
    ctx = O.Ctx()
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    rows = []
    with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
        seq = None
        for seed in range(12):
            seq = policy_prefix(session, policy, cfg, thr, start, seed=seed)
            if seq:
                break
        if not seq:
            raise SystemExit("policy never reached handover")
        print(f"handover reached in {len(seq)} frames (x>={HANDOVER})\n")

        for mode in ("as_run", "true_rest", "at_pipe", "at_pipe_noh"):
            for ctrl in (None, "no_jump", "always_a"):
                if ctrl is not None:
                    r = one(session, start, seq, hold=0, mode=mode, control=ctrl)
                    rows.append(r)
                    print(f"  {mode:12s} CONTROL {ctrl:9s} takeoff_x={r['takeoff_x']} "
                          f"speed={r['takeoff_speed_byte']} max_x={r['max_x']} "
                          f"cleared={r['cleared_pipe2']}")
            cleared = []
            for hold in HOLDS:
                r = one(session, start, seq, hold=hold, mode=mode)
                rows.append(r)
                if r["cleared_pipe2"]:
                    cleared.append(hold)
            first = rows[-len(HOLDS)]
            print(f"  {mode:12s} takeoff_x={first['takeoff_x']} "
                  f"speed_byte={first['takeoff_speed_byte']} "
                  f"contact={first['in_contact_with_pipe2']} "
                  f"h_in_flight={first['horizontal_input_in_flight']} "
                  f"-> min clearing hold {min(cleared) if cleared else None} "
                  f"(cleared: {cleared})\n")

    per_mode = {}
    for mode in ("as_run", "true_rest", "at_pipe", "at_pipe_noh"):
        ms = [r for r in rows if r["mode"] == mode and r["control"] is None]
        cl = [r["hold"] for r in ms if r["cleared_pipe2"]]
        ctrl = {r["control"]: r["max_x"] for r in rows
                if r["mode"] == mode and r["control"]}
        per_mode[mode] = {
            "takeoff_x": ms[0]["takeoff_x"],
            "takeoff_speed_byte": ms[0]["takeoff_speed_byte"],
            "in_contact_with_pipe2": ms[0]["in_contact_with_pipe2"],
            "horizontal_input_in_flight": ms[0]["horizontal_input_in_flight"],
            "min_clearing_hold": min(cl) if cl else None,
            "clearing_holds": cl,
            "max_x_no_jump_control": ctrl.get("no_jump"),
            "max_x_always_a_control": ctrl.get("always_a"),
            "distinct_max_x": len({r["max_x"] for r in ms}),
        }

    answer = per_mode["at_pipe_noh"]
    binary = (
        "YES -- a jump initiated in contact with pipe 2 (x>=585) with no horizontal input in "
        f"flight clears it at hold {answer['min_clearing_hold']}"
        if answer["min_clearing_hold"] is not None else
        "NO -- a jump initiated in contact with pipe 2 with no horizontal input in flight "
        "does NOT clear at any hold up to 40")
    out = {"checkpoint": CKPT.name, "contact_threshold_x": CONTACT_X,
           "per_mode": per_mode, "binary_answer": binary,
           "original_sweep_standstill_was": per_mode["as_run"], "rows": rows}
    print("=" * 78)
    for m, d in per_mode.items():
        print(f"{m:12s} takeoff_x={d['takeoff_x']:4d} speed={d['takeoff_speed_byte']:3d} "
              f"contact={str(d['in_contact_with_pipe2']):5s} "
              f"h_flight={str(d['horizontal_input_in_flight']):5s} "
              f"min_hold={d['min_clearing_hold']}  "
              f"no_jump_ctrl={d['max_x_no_jump_control']} "
              f"always_a_ctrl={d['max_x_always_a_control']}")
    print(f"\nBINARY: {binary}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
