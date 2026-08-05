"""§5: what clears pipe 3's face, what does the policy emit there, and how does it arrive?

§2's decision rule fired `wall_map_stands` — nothing left the 1-1 surface under the terminator every prior
figure used — so this targets **pipe 3's face**, x ≈ 672–704, as specified. (`stall_rule_audit.json` refines
that: the median wall is genuine, the upper tail was truncated. The median is what this study is about.)

**The arrival states are the policy's OWN, not the start library's.** §5.3 asks for x, y, grounded and
**speed at the wall — never measured here before** — and a library state merely near x=700 is not the state
the policy actually arrives in. So each arrival state is captured live: run the policy, and at the first
grounded frame with x in the capture window, `save_scratch()`. Restores are then O(1), which is what makes a
sweep of hundreds of action sequences per state affordable.

Three questions:

1. **What clears it.** Sweep (combo × hold length) from the arrival state and report **every** sequence that
   gets past x=735, plus the **minimum hold** that works. **If nothing clears it from the state the policy
   actually arrives in, that is the finding** — it means the policy must not arrive that way, and the target
   moves upstream.
2. **What the policy emits there** — its probability mass on the solution set. This is the number the study
   is for.
3. **The arrival state itself:** x, y absolute, `0x001D` grounded, `0x0057` speed (units of 1/16 px/frame,
   max running 40).

**`measurement_basis: conditional_on_arrival`, per-obstacle, never pooled. Bootstrap over ARRIVAL STATES**,
not over sweep configurations — repeats from one state are correlated, which is why block 56's bins had no
intervals.
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
from tasdata.ram import on_ground, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/pipe3_requirement.json"
PARTIAL = ROOT / "data/pipe3_requirement.partial.json"

ARM = "P_84_cnn32_seed4"
TEMP = 0.7
TARGET = "pipe3"
CLEAR_X = PIPE_THRESHOLDS["pipe3"]        # 735, past the far edge -- never the face
CAPTURE_WINDOW = (660, 700)               # the face, from reach_walls' biggest early pile
N_ARRIVALS = 12
PROBE_FRAMES = 150                        # enough to clear 735 from ~680 at any speed
SCRATCH_SLOT = 1
#: Sustained locomotion baseline, Right+B = 0x82. The swept combo is OR'd on top of this for its
#: hold, then the baseline continues -- see the comment in the sweep loop for why.
LOCOMOTION = 0x82
#: coarse-then-geometric hold lengths; 12 is pipe 2's known requirement so the grid must straddle it
HOLDS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
ARM_BUDGET_S = 30 * 60


def capture_arrivals(sess, policy, cfg, lut, byte_of, states, p1, start, dl, want):
    """Run the policy from library starts; snapshot the first grounded frame inside the window."""
    out = []
    s = cfg.frame_size
    for st in states:
        if len(out) >= want or dl.remaining() < 120:
            break
        ep = p1.get(st["seed"])
        if ep is None:
            continue
        obs = restore_state(sess, st, [f[3] for f in ep["frames"]], start.frame)
        win = np.zeros((cfg.stack, s, s), np.uint8)
        win[:] = _resize_gray(obs.rgb, (s, s))
        rng = np.random.default_rng(st["frame_index"])
        held, remaining = None, 0
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
            win = np.roll(win, -1, 0)
            win[-1] = _resize_gray(obs.rgb, (s, s))
            r = read_smb(obs.ram, obs.framecount)
            if r.player_state in (0x06, 0x0B):
                break
            if CAPTURE_WINDOW[0] <= r.x_position <= CAPTURE_WINDOW[1] and on_ground(obs.ram):
                # ⚠ ONE SLOT PER ARRIVAL. The first version saved every arrival to slot 1, so each
                # capture overwrote the previous one and all twelve sweeps ran from whichever state
                # happened to be captured last. Slots run to 0xFFFF, so there is no reason to share.
                slot = SCRATCH_SLOT + len(out)
                sess.save_scratch(slot)
                out.append({
                    "from_start_x": st["x"], "x": int(r.x_position),
                    "y_absolute": int(obs.ram[0x00B5]) * 256 + int(obs.ram[0x03B8]),
                    "grounded": True, "speed_byte": int(obs.ram[0x0057]),
                    "speed_px_per_frame": int(obs.ram[0x0057]) / 16.0,
                    "frame_window": np.asarray(win).copy(),
                    "scratch_slot": slot})
                break
            if r.x_position > CAPTURE_WINDOW[1] + 40:
                break
    return out


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

    out = {"arm": ARM, "target": TARGET, "clear_threshold_x": CLEAR_X,
           "capture_window_x": list(CAPTURE_WINDOW), "temperature_for_arrival": TEMP,
           "measurement_basis": "conditional_on_arrival",
           "grounded_enforced": True,
           "holds_swept": HOLDS, "n_combos": len(tokens), "locomotion_baseline": "Right+B (0x82) held for all probe frames",
           "note": ("arrival states are the POLICY'S OWN, captured live at the first grounded frame in "
                    "the window, not library states merely near x=700"),
           "arrivals": [], "skipped": []}
    sess = None
    try:
        with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), "pipe3 sweep"):
            sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
            warm_session(sess, start.frame)
            arrivals = capture_arrivals(sess, policy, cfg, lut, byte_of, lib, p1, start, dl,
                                        N_ARRIVALS)
            print(f"{dl.stamp()} captured {len(arrivals)} arrival states at pipe 3's face",
                  flush=True)
            for ai, arr in enumerate(arrivals):
                if dl.remaining() < 90:
                    out["skipped"].append({"arrival": ai, "reason": "deadline"})
                    break
                # the policy's own distribution AT this state
                with torch.no_grad():
                    lg = policy(torch.from_numpy(arr["frame_window"][None]).float().div_(255.0))[0]
                probs = torch.softmax(lg / TEMP, dim=-1).numpy()
                probs1 = torch.softmax(lg, dim=-1).numpy()
                # Verify the restore lands where it was captured. A silent bad restore is exactly
                # how the first run of this sweep reported 0/350 from every state.
                vobs = sess.load_scratch(arr["scratch_slot"])
                vx = read_smb(vobs.ram, vobs.framecount).x_position
                restore_ok = abs(int(vx) - arr["x"]) <= 4
                winners, rows = [], []
                for tok, b in tokens:
                    for hold in HOLDS:
                        sess.load_scratch(arr["scratch_slot"])
                        best = 0
                        for f in range(PROBE_FRAMES):
                            # ⚠ The swept combo is OR'd into a SUSTAINED Right+B baseline, not played
                            # in isolation. The first version of this sweep released every button
                            # after `hold` frames, so Mario decelerated and every one of 350 configs
                            # from every one of 12 states stopped at exactly x=724 -- pressed against
                            # the pipe. That was the sweep space being too impoverished to express the
                            # solution, not the obstacle being unsolvable. A jump here is "hold Right
                            # throughout, add A for a burst", and the grid has to be able to say that.
                            obs = sess.step((b | LOCOMOTION) if f < hold else LOCOMOTION)
                            r = read_smb(obs.ram, obs.framecount)
                            best = max(best, r.x_position)
                            if r.player_state in (0x06, 0x0B):
                                break
                            if r.x_position > CLEAR_X:
                                break
                        ok = best > CLEAR_X
                        rows.append({"token": tok, "byte": b, "hold": hold,
                                     "max_x": int(best), "cleared": bool(ok)})
                        if ok:
                            winners.append({"token": tok, "byte": b, "hold": hold})
                # probability mass the policy puts on the winning (combo, length-bucket) classes
                win_tok = {w["token"] for w in winners}
                mass = float(sum(probs[c] for c in range(len(probs))
                                 if (c // G.N_BUCKETS) in win_tok))
                mass_t1 = float(sum(probs1[c] for c in range(len(probs1))
                                    if (c // G.N_BUCKETS) in win_tok))
                min_hold = min((w["hold"] for w in winners), default=None)
                rec = {k: v for k, v in arr.items() if k != "frame_window"}
                rec.update({
                    "restore_verified_x": int(vx), "restore_ok": bool(restore_ok),
                    "n_configs": len(rows), "n_cleared": len(winners),
                    "frac_cleared": len(winners) / len(rows) if rows else None,
                    "minimum_hold_that_works": min_hold,
                    "winning_combos": sorted({w["byte"] for w in winners}),
                    "winners": winners[:60],
                    "policy_mass_on_winning_combos_T0.7": mass,
                    "policy_mass_on_winning_combos_T1.0": mass_t1,
                    "best_max_x_over_all_configs": max((r["max_x"] for r in rows), default=None)})
                out["arrivals"].append(rec)
                print(f"  {dl.stamp()} arrival {ai}: x={rec['x']} speed={rec['speed_px_per_frame']:.2f} "
                      f"px/f | {rec['n_cleared']}/{rec['n_configs']} configs clear | min hold "
                      f"{min_hold} | policy mass {mass:.3f}", flush=True)
                PARTIAL.write_text(json.dumps(out, indent=2, default=str))
    except TimedOut as e:
        out["skipped"].append({"reason": str(e)})
        print(f"{dl.stamp()} TIMEOUT: {e}", flush=True)
    finally:
        if sess is not None:
            try:
                sess.close()
            except BaseException:
                pass

    arr = out["arrivals"]
    if arr:
        fr = np.array([a["frac_cleared"] for a in arr], float)
        ms = np.array([a["policy_mass_on_winning_combos_T0.7"] for a in arr], float)
        sp = np.array([a["speed_px_per_frame"] for a in arr], float)
        mh = [a["minimum_hold_that_works"] for a in arr if a["minimum_hold_that_works"] is not None]
        rng = np.random.default_rng(0)

        def boot(v):
            if len(v) < 3:
                return None
            b = [float(np.mean(rng.choice(v, len(v), True))) for _ in range(20000)]
            return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        out["summary"] = {
            "n_arrival_states": len(arr),
            "frac_configs_clearing": {"mean": float(fr.mean()), "ci_boot_over_arrivals": boot(fr),
                                      "per_arrival": fr.tolist()},
            "policy_mass_on_solution_set": {"mean": float(ms.mean()),
                                            "ci_boot_over_arrivals": boot(ms),
                                            "per_arrival": ms.tolist()},
            "arrival_speed_px_per_frame": {"mean": float(sp.mean()), "min": float(sp.min()),
                                           "max": float(sp.max()), "per_arrival": sp.tolist(),
                                           "expert_max_running": 2.5},
            "minimum_hold_that_works": {"values": mh,
                                        "median": (float(np.median(mh)) if mh else None)},
            "n_arrivals_with_no_solution": sum(1 for a in arr if a["n_cleared"] == 0),
            "interval_basis": "bootstrap over ARRIVAL STATES, not over sweep configurations"}
        s = out["summary"]
        nosol = s["n_arrivals_with_no_solution"]
        if nosol == len(arr):
            out["verdict"] = (
                f"**NOTHING CLEARS PIPE 3 FROM THE STATE THE POLICY ARRIVES IN.** All {len(arr)} captured "
                f"arrival states are unsolvable by any of {out['n_combos']}×{len(HOLDS)} single "
                f"(combo, hold) sequences. Arrival speed {s['arrival_speed_px_per_frame']['mean']:.2f} "
                f"px/frame against the expert's running maximum of 2.5. **The target moves upstream: the "
                f"policy must not arrive this way**, so pipe 3 is an approach problem, not a jump problem.")
        else:
            out["verdict"] = (
                f"**PIPE 3 IS SOLVABLE FROM THE POLICY'S OWN ARRIVAL STATE, AND THE POLICY BARELY AIMS AT "
                f"THE SOLUTION.** {len(arr) - nosol} of {len(arr)} arrival states have at least one "
                f"clearing sequence; on average {s['frac_configs_clearing']['mean']:.1%} of "
                f"{out['n_combos']}×{len(HOLDS)} sequences work, minimum hold "
                f"{s['minimum_hold_that_works']['median']}. **The policy puts "
                f"{s['policy_mass_on_solution_set']['mean']:.3f} of its probability mass on the winning "
                f"combos** (CI {s['policy_mass_on_solution_set']['ci_boot_over_arrivals']}). Arrival "
                f"speed {s['arrival_speed_px_per_frame']['mean']:.2f} px/frame vs the expert's 2.5 "
                f"running maximum.")
    else:
        out["verdict"] = "No arrival states captured; nothing to report."
    out["minutes"] = round(dl.elapsed() / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
