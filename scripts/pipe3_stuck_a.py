"""§2: is the hidden variable at pipe 3 a stuck A button?

`pipe3_requirement.json` left five of twelve arrival states unsolvable by any of 350 (combo, hold) sequences,
and the pattern is suspicious: **four of the five sit at exactly x=678**, while x=677 at identical speed
(2.50 px/frame) clears 77 of 350. **A one-pixel cliff at 2.5 px/frame is not geometry.**

Hypothesis: **A is already held at those arrival frames.** In SMB the jump button must be RELEASED before a
new jump can start, and `live.py` already names "A held while airborne" as a pathology that blocks every
later jump. If so the mistake happens *before* arrival and the correct label is **"release A"**, not
"jump here" — which changes what §3 has to capture.

Recorded at each arrival, so a negative is informative too: the **button byte**, the **A-hold length running
into the frame**, **y sub-position**, the **enemy slots**, and whether the frame is grounded.

The arrivals are re-captured rather than read from the old artifact, because that artifact stored only
(x, y, grounded, speed) — the four fields already shown not to determine the outcome.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.argmax_startstates import CAP_NON_A, restore_state  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT, PIPE_THRESHOLDS  # noqa: E402
from tasdata.bc.trace_log import (  # noqa: E402
    ADDR_ENEMY_ACTIVE,
    ADDR_ENEMY_TYPE,
    ADDR_ENEMY_XLO,
    ADDR_ENEMY_XPAGE,
    ENEMY_SLOTS,
)
from tasdata.ram import on_ground, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/pipe3_stuck_a.json"

ARM = "P_84_cnn32_seed4"
TEMP = 0.7
CAPTURE_WINDOW = (660, 700)
CLEAR_X = PIPE_THRESHOLDS["pipe3"]
N_ARRIVALS = 12
PROBE_FRAMES = 150
LOCOMOTION = 0x82
HOLDS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
#: §2b: retreat macros -- the corpus has 695 retreats in 132,005 tokens (0.526%), so sampling never
#: finds "back off and re-approach". Injected explicitly.
RETREAT_L = [8, 16, 24, 32]
RETREAT_M = [16, 24, 32, 48]
RETREAT_H = [12, 16, 20, 24]
LEFT_BIT = 0x40


def a_hold_len(hist):
    """How many consecutive frames ending here had A held."""
    n = 0
    for b in reversed(hist):
        if b & A_BIT:
            n += 1
        else:
            break
    return n


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 40 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    lib = [s for s in json.loads((ROOT / "data/startlib_policy.json").read_text())["states"]
           if s["x"] < CAPTURE_WINDOW[0]]
    p1 = {e["seed"]: e
          for e in json.loads((ROOT / "data/traces/p1_200.json").read_text())["episodes"]}
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)
    policy, cfg, _ = G.load_ckpt(ARM)
    tokens = [(t, int(ctx.vocab.decode_byte(t))) for t in range(ctx.vocab.size)]

    out = {"arm": ARM, "hypothesis": "A is already held at the unsolvable arrival frames",
           "capture_window_x": list(CAPTURE_WINDOW), "clear_threshold_x": CLEAR_X,
           "measurement_basis": "conditional_on_arrival",
           "retreat_macro_grid": {"L": RETREAT_L, "M": RETREAT_M, "H": RETREAT_H,
                                  "n": len(RETREAT_L) * len(RETREAT_M) * len(RETREAT_H),
                                  "why": ("the corpus holds 695 retreats in 132,005 run tokens "
                                          "(0.526%), so ~1.8 expected among 350 sampled sequences; "
                                          "a retreat+re-approach is 2-3 tokens and effectively "
                                          "unreachable by sampling")},
           "arrivals": []}
    sess = None
    try:
        with time_limit(min(30 * 60, dl.remaining() - 60), "pipe3 stuck-A"):
            sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
            warm_session(sess, start.frame)
            captured = 0
            for st in lib:
                if captured >= N_ARRIVALS or dl.remaining() < 120:
                    break
                ep = p1.get(st["seed"])
                if ep is None:
                    continue
                obs = restore_state(sess, st, [f[3] for f in ep["frames"]], start.frame)
                s_ = cfg.frame_size
                win = np.zeros((cfg.stack, s_, s_), np.uint8)
                win[:] = _resize_gray(obs.rgb, (s_, s_))
                rng = np.random.default_rng(st["frame_index"])
                held, remaining = None, 0
                bytes_hist = []
                arr = None
                for _ in range(2500):
                    if remaining <= 0:
                        with torch.no_grad():
                            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
                        p = torch.softmax(lg / TEMP, dim=-1).numpy()
                        c = int(rng.choice(len(p), p=p / p.sum()))
                        b, L = int(byte_of[c]), max(1, int(lut[c]))
                        if not (b & A_BIT):
                            L = min(L, CAP_NON_A)
                        held, remaining = b, L
                    remaining -= 1
                    obs = sess.step(held)
                    bytes_hist.append(int(held))
                    win = np.roll(win, -1, 0)
                    win[-1] = _resize_gray(obs.rgb, (s_, s_))
                    r = read_smb(obs.ram, obs.framecount)
                    if r.player_state in (0x06, 0x0B):
                        break
                    if CAPTURE_WINDOW[0] <= r.x_position <= CAPTURE_WINDOW[1] and on_ground(obs.ram):
                        slot = 1 + captured
                        sess.save_scratch(slot)
                        enemies = [{"slot": i, "raw_id": int(obs.ram[ADDR_ENEMY_TYPE + i]),
                                    "active": int(obs.ram[ADDR_ENEMY_ACTIVE + i]),
                                    "x": int(obs.ram[ADDR_ENEMY_XPAGE + i]) * 256
                                         + int(obs.ram[ADDR_ENEMY_XLO + i])}
                                   for i in range(ENEMY_SLOTS)]
                        arr = {"from_start_x": st["x"], "x": int(r.x_position),
                               "y_absolute": int(obs.ram[0x00B5]) * 256 + int(obs.ram[0x03B8]),
                               "y_sub": int(obs.ram[0x0416]) if len(obs.ram) > 0x0416 else None,
                               "speed_byte": int(obs.ram[0x0057]),
                               "speed_px_per_frame": int(obs.ram[0x0057]) / 16.0,
                               "grounded": True,
                               "button_byte": int(held),
                               "A_held_at_arrival": bool(held & A_BIT),
                               "A_hold_len_running_in": a_hold_len(bytes_hist),
                               "enemies": enemies, "scratch_slot": slot}
                        break
                    if r.x_position > CAPTURE_WINDOW[1] + 40:
                        break
                if arr is None:
                    continue
                captured += 1
                # --- sweep: single (combo, hold) on a sustained Right+B baseline ---
                winners = []
                for tok, b in tokens:
                    for hold in HOLDS:
                        sess.load_scratch(arr["scratch_slot"])
                        best = 0
                        for f in range(PROBE_FRAMES):
                            o = sess.step((b | LOCOMOTION) if f < hold else LOCOMOTION)
                            rr = read_smb(o.ram, o.framecount)
                            best = max(best, rr.x_position)
                            if rr.player_state in (0x06, 0x0B) or best > CLEAR_X:
                                break
                        if best > CLEAR_X:
                            winners.append({"kind": "single", "byte": b, "hold": hold})
                # --- §2b: retreat macros, injected explicitly ---
                retreat_winners = []
                for L in RETREAT_L:
                    for M in RETREAT_M:
                        for H in RETREAT_H:
                            sess.load_scratch(arr["scratch_slot"])
                            best = 0
                            dead = False
                            for f in range(L + M + 200):
                                if f < L:
                                    byte = LEFT_BIT
                                elif f < L + M:
                                    byte = LOCOMOTION
                                else:
                                    byte = LOCOMOTION | (A_BIT if f < L + M + H else 0)
                                o = sess.step(byte)
                                rr = read_smb(o.ram, o.framecount)
                                best = max(best, rr.x_position)
                                if rr.player_state in (0x06, 0x0B):
                                    dead = True
                                    break
                                if best > CLEAR_X:
                                    break
                            if best > CLEAR_X and not dead:
                                retreat_winners.append({"kind": "retreat", "L": L, "M": M, "H": H})
                arr.update({
                    "n_single_configs": len(tokens) * len(HOLDS),
                    "n_single_cleared": len(winners),
                    "n_retreat_configs": len(RETREAT_L) * len(RETREAT_M) * len(RETREAT_H),
                    "n_retreat_cleared": len(retreat_winners),
                    "solvable_by_single": len(winners) > 0,
                    "solvable_by_retreat_only": len(winners) == 0 and len(retreat_winners) > 0,
                    "min_single_hold": min((w["hold"] for w in winners), default=None),
                    "retreat_examples": retreat_winners[:6]})
                out["arrivals"].append(arr)
                print(f"  {dl.stamp()} x={arr['x']} spd={arr['speed_px_per_frame']:.2f} "
                      f"byte={arr['button_byte']:#04x} A_held={arr['A_held_at_arrival']} "
                      f"A_run={arr['A_hold_len_running_in']:3d} | single {len(winners):3d}/350 "
                      f"| retreat {len(retreat_winners):2d}/64", flush=True)
                OUT.write_text(json.dumps(out, indent=2, default=str))
    except TimedOut as e:
        out["timed_out"] = str(e)
    finally:
        if sess is not None:
            try:
                sess.close()
            except BaseException:
                pass

    arr = out["arrivals"]
    if arr:
        solv = [a for a in arr if a["solvable_by_single"]]
        unsolv = [a for a in arr if not a["solvable_by_single"]]
        out["cross_tab"] = {
            "n": len(arr), "n_solvable_by_single": len(solv), "n_unsolvable_by_single": len(unsolv),
            "A_held_among_solvable": sum(1 for a in solv if a["A_held_at_arrival"]),
            "A_held_among_unsolvable": sum(1 for a in unsolv if a["A_held_at_arrival"]),
            "mean_A_run_solvable": float(np.mean([a["A_hold_len_running_in"] for a in solv]))
            if solv else None,
            "mean_A_run_unsolvable": float(np.mean([a["A_hold_len_running_in"] for a in unsolv]))
            if unsolv else None,
            "x_solvable": [a["x"] for a in solv], "x_unsolvable": [a["x"] for a in unsolv],
            "retreat_rescued": [a["x"] for a in arr if a["solvable_by_retreat_only"]],
            "n_retreat_rescued": sum(1 for a in arr if a["solvable_by_retreat_only"])}
        ct = out["cross_tab"]
        held_u = ct["A_held_among_unsolvable"]
        held_s = ct["A_held_among_solvable"]
        supports = (len(unsolv) > 0 and held_u == len(unsolv) and held_s < len(solv))
        parts = []
        if supports:
            parts.append(
                f"**THE STUCK-A HYPOTHESIS HOLDS: A is held at {held_u}/{len(unsolv)} unsolvable "
                f"arrivals and {held_s}/{len(solv)} solvable ones.** The mistake is made before "
                f"arrival and the label is 'release A', not 'jump here'.")
        else:
            parts.append(
                f"**THE STUCK-A HYPOTHESIS DOES NOT HOLD.** A is held at {held_u} of {len(unsolv)} "
                f"unsolvable arrivals and {held_s} of {len(solv)} solvable ones; mean A-run "
                f"{ct['mean_A_run_unsolvable']} vs {ct['mean_A_run_solvable']}. **The hidden variable "
                f"is still unidentified, so §3 must capture states earlier to compensate.**")
        if ct["n_retreat_rescued"]:
            parts.append(
                f"**RETREAT MACROS RESCUE {ct['n_retreat_rescued']} STATE(S) THAT SINGLE-ACTION SEARCH "
                f"CANNOT** (x = {ct['retreat_rescued']}). Backing off and re-approaching is a solution "
                f"the policy has ~0.5% mass on and sampling would never find.")
        else:
            parts.append("**No state was rescued by a retreat macro that single-action search could "
                         "not already solve.**")
        out["verdict"] = " ".join(parts)
    else:
        out["verdict"] = "No arrivals captured."
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
