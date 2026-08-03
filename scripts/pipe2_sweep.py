"""Priority 2: is pipe 2 clearable by this action space? Sweep hold x trigger x B.

The previous attempt was void: a scripted run-right agent died on the Goomba at x~300 and
every configuration returned max x = 314, so it measured the harness rather than the
variable. Two fixes here.

**Getting to the pipe.** No grounded expert frame exists in 1-1 between x=250 and x=480 --
the TAS is airborne continuously through there -- so there is no savestate to start from.
Instead the trained policy drives from the 1-1 level start until x >= HANDOVER, which it does
reliably (it clears pipe 1 on 99% of episodes), and scripted control takes over from there.
The policy is seeded and the emulator is deterministic from a savestate, so the handover
state is byte-identical for every configuration. Its action prefix is computed once and
replayed thereafter, which removes ~450 network evaluations per configuration.

**Sanity guard.** If every configuration returns the same max x, the harness is broken again;
that is asserted rather than reported.

Takeoff velocity is measured as mean delta-x over the 5 frames before A is first pressed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.overnight_lib import calibrate, load_policy  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.tokens import LIVE_MASK  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/pipe2_sweep.json"
CKPT = ROOT / "data/bc_overnight/round3_ratio1to1.pt"

RIGHT, B, A, LEFT = (NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["B"],
                     NES_BUTTON_BITS["A"], NES_BUTTON_BITS["Left"])
HANDOVER = 500            # policy drives to here, then the script takes over
PIPE2_CLEARED_X = 630     # past the far side of pipe 2
HOLDS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
TRIGGERS = [510, 520, 530, 540, 550, 560, 570, 580, 590]
POST_FRAMES = 260


def policy_prefix(session, policy, cfg, thr, start, seed=0, cap=900):
    """Actions the policy takes from the level start until x >= HANDOVER."""
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), dtype=np.uint8)
    win[:] = _resize_gray(obs.rgb, (84, 84))
    seq = []
    for _ in range(cap):
        with torch.no_grad():
            logits = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0 / (1.0 + np.exp(-logits))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]:
                byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        seq.append(byte)
        obs = session.step(byte)
        win = np.roll(win, -1, axis=0)
        win[-1] = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in (0x06, 0x0B):
            return None
        if st.x_position >= HANDOVER:
            return seq
    return None


def replay_prefix(session, start, seq):
    obs = session.reset(start.frame)
    for byte in seq:
        obs = session.step(byte)
    return obs


def run_config(session, start, seq, *, hold, trigger, with_b, standstill=False):
    """Replay to handover, then approach and jump under scripted control."""
    obs = replay_prefix(session, start, seq)
    base = RIGHT | (B if with_b else 0)
    xs = [read_smb(obs.ram, obs.framecount).x_position]
    takeoff_v = None
    jumped = 0
    maxx = xs[0]
    died = False

    if standstill:
        # Let Mario come to rest first: no input until delta-x is zero.
        for _ in range(90):
            obs = session.step(0)
            x = read_smb(obs.ram, obs.framecount).x_position
            xs.append(x)
            maxx = max(maxx, x)
            if len(xs) >= 3 and xs[-1] == xs[-2] == xs[-3]:
                break

    for i in range(POST_FRAMES):
        st = read_smb(obs.ram, obs.framecount)
        x = st.x_position
        byte = base
        if jumped == 0 and (standstill or x >= trigger):
            takeoff_v = float(np.mean(np.diff(xs[-6:]))) if len(xs) >= 3 else 0.0
            jumped = 1
            left = hold
        if jumped == 1 and left > 0:
            byte |= A
            left -= 1
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        xs.append(st.x_position)
        maxx = max(maxx, st.x_position)
        if st.player_state in (0x06, 0x0B):
            died = True
            break
    return {"hold": hold, "trigger": trigger, "with_b": with_b, "standstill": standstill,
            "max_x": int(maxx), "cleared_pipe2": bool(maxx > PIPE2_CLEARED_X),
            "takeoff_velocity": (round(takeoff_v, 2) if takeoff_v is not None else None),
            "died": died}


def main() -> None:
    ctx = O.Ctx()
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    results = []
    with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
        seq = None
        for seed in range(12):
            seq = policy_prefix(session, policy, cfg, thr, start, seed=seed)
            if seq:
                print(f"policy reached x>={HANDOVER} on seed {seed} in {len(seq)} frames")
                break
        if not seq:
            raise SystemExit("policy never reached the handover point")

        obs = replay_prefix(session, start, seq)
        st = read_smb(obs.ram, obs.framecount)
        print(f"handover state: x={st.x_position} y={st.y_position} "
              f"state={st.player_state}")

        for with_b in (True, False):
            for trigger in TRIGGERS:
                for hold in HOLDS:
                    r = run_config(session, start, seq, hold=hold, trigger=trigger,
                                   with_b=with_b)
                    results.append(r)
                if any(x["cleared_pipe2"] for x in results[-len(HOLDS):]):
                    good = [x for x in results[-len(HOLDS):] if x["cleared_pipe2"]]
                    print(f"  B={with_b} trigger={trigger}: CLEARED with holds "
                          f"{[g['hold'] for g in good]} "
                          f"(v={good[0]['takeoff_velocity']})")
                else:
                    best = max(results[-len(HOLDS):], key=lambda x: x["max_x"])
                    print(f"  B={with_b} trigger={trigger}: best max_x {best['max_x']} "
                          f"(hold {best['hold']}, v={best['takeoff_velocity']})")

        # standstill: decelerate to rest, then jump
        stand = [run_config(session, start, seq, hold=h, trigger=0, with_b=True,
                            standstill=True) for h in HOLDS]
        results += stand
        s_ok = [r for r in stand if r["cleared_pipe2"]]
        print(f"\nstandstill takeoff (v=0): "
              f"{'CLEARED with holds ' + str([r['hold'] for r in s_ok]) if s_ok else 'NEVER cleared'}"
              f"  best max_x {max(r['max_x'] for r in stand)}")

    xs = {r["max_x"] for r in results}
    assert len(xs) > 1, (
        f"every configuration returned max_x={xs.pop()} -- the harness is broken, not the "
        f"variable")

    running = [r for r in results if not r["standstill"]
               and (r["takeoff_velocity"] or 0) >= 2.0 and r["cleared_pipe2"]]
    min_running_hold = min((r["hold"] for r in running), default=None)
    good_triggers = sorted({r["trigger"] for r in results
                            if r["cleared_pipe2"] and not r["standstill"]})
    out = {
        "handover_x": HANDOVER, "n_configs": len(results),
        "distinct_max_x": len(xs),
        "min_hold_clearing_from_running": min_running_hold,
        "standstill_ever_cleared": bool(s_ok),
        "working_trigger_range": [min(good_triggers), max(good_triggers)] if good_triggers else None,
        "working_triggers": good_triggers,
        "any_cleared": any(r["cleared_pipe2"] for r in results),
        "results": results,
    }
    out["verdict"] = (
        f"pipe 2 IS clearable by this action space: minimum {min_running_hold}-frame A-hold "
        f"from a running takeoff (v>=2.0), trigger window x={good_triggers}. "
        f"Standstill takeoff: {'also clears' if s_ok else 'NEVER clears at any hold up to 40'}. "
        f"The ceiling is placement."
        if min_running_hold is not None else
        ("no running configuration cleared pipe 2 -- the action space or the approach is the "
         "limit" if not out["any_cleared"] else
         "cleared, but not from a running takeoff -- inspect the results"))
    print(f"\nVERDICT: {out['verdict']}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
