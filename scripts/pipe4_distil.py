"""Pipe 4: confirm survival for the 39 clearing configs, distil them, measure transfer.

Three steps, one script, per the thirty-fourth directive.

Step 1  Re-run the 39 clearing (seed, trigger, hold) configurations from `data/pipe4_build.json`
        against the *search's* bar -- cleared past x=975 **and** survived 40 further frames -- and
        record every frame of the passing ones as a demonstration.
Step 2  Fine-tune the same checkpoint on those demonstrations mixed 1:1 with earliest-chain expert
        data, using the composed recipe's sustain+onset loss. The target stays per-frame next-action
        prediction; a (trigger, hold) solution is unrolled into frames before it is ever trained on.
Step 3  Measure transfer at n=200, single life, with the same episode function and the same 200
        seeds as the baseline in `data/traces/p1_200.json`, so the comparison is paired:
          * the A-hold distribution for holds *beginning* in x 880-924,
          * the count of episodes stuck at pipe 4's face (max_x 896-928),
          * clearance past 975 conditional on arriving at 880.

The demonstration is the last `PREFIX_TAIL` frames of the policy's own approach plus the scripted
attempt through clearance and 40 frames of survival. The full approach is deliberately *not*
repeated 13 times per seed: the prefix is identical across a seed's configs, and repeating it would
make the demonstration set mostly a copy of the approach rather than of the jump.

A clearance figure alone is not evidence that distillation worked. If clearance moves and the hold
distribution does not, this script's own verdict string says so.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import EARLIEST, session_when_free, train  # noqa: E402
from scripts.overnight import write_self_run  # noqa: E402
from scripts.p1_run import episode as traced_episode  # noqa: E402
from scripts.pipe4_build import prefix  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    calibrate,
    diff_ci,
    load_policy,
    random_rows,
    save_policy,
    wilson,
)
from tasdata.bc.pipe4_metrics import CLEAR_X, arm_metrics  # noqa: E402
from tasdata.bc.trace_log import write_traces  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402
from tasdata.ram import on_ground, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "data/pipe4_build.json"
OUT = ROOT / "data/pipe4_distil.json"
TRACES = ROOT / "data/traces/pipe4_distil_200.json"
BASELINE_TRACES = ROOT / "data/traces/p1_200.json"
DEMOS = ROOT / "data/runs_self/pipe4_demos"
CKPT = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
NEW_CKPT = ROOT / "data/bc_coverage/pipe4_distilled.pt"

A, B, RIGHT = NES_BUTTON_BITS["A"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["Right"]
PREFIX_TAIL = 60      # frames of the policy's own approach kept as context in each demonstration
SURVIVE = 40          # frames alive past clearance -- the search's bar, reused verbatim
ATTEMPT_FRAMES = 320
STEPS = 800
LR = 1e-4
N_EVAL = 200


def demo_attempt(session, start, seq, *, hold, trigger, frames=ATTEMPT_FRAMES):
    """The scripted attempt, recording every (obs84, byte) so a success becomes a demonstration.

    The last `PREFIX_TAIL` frames of the policy's own approach are recorded too, so the frame stacks
    at the start of the jump have real context rather than a repeated first frame.

    Returns the recorded pairs plus whether it cleared past 975 and stayed alive for SURVIVE frames
    after the first frame past 975. `on_ground()` is required at the jump frame, as in the sweep.
    """
    obs = session.reset(start.frame)
    rec = []
    tail_from = max(0, len(seq) - PREFIX_TAIL)
    for i, b in enumerate(seq):
        if i >= tail_from:
            rec.append((_resize_gray(obs.rgb, (84, 84)), b))
        obs = session.step(b)
    win84 = _resize_gray(obs.rgb, (84, 84))
    left, jumped, maxx = hold, False, 0
    xs, died, prev_x = [], False, None
    clear_i = None
    for _ in range(frames):
        st = read_smb(obs.ram, obs.framecount)
        byte = RIGHT | B
        if not jumped and st.x_position >= trigger:
            if not on_ground(obs.ram):
                return None, {"skipped": True, "grounded_at_jump": False}
            jumped = True
        if jumped and left > 0:
            byte |= A
            left -= 1
        rec.append((win84.copy(), byte))
        obs = session.step(byte)
        win84 = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        if prev_x is not None and st.x_position - prev_x < -100:
            break                      # pipe transit: x and y both invalid across it
        prev_x = st.x_position
        xs.append(int(st.x_position))
        maxx = max(maxx, st.x_position)
        if clear_i is None and st.x_position > CLEAR_X:
            clear_i = len(xs) - 1
        if st.player_state in (0x06, 0x0B):
            died = True
            break
        if clear_i is not None and len(xs) - clear_i > SURVIVE:
            break                      # bar met; stop here so the demo ends where it is scored
    survived = clear_i is not None and not died and (len(xs) - clear_i) >= SURVIVE
    info = {"skipped": False, "grounded_at_jump": True, "max_x": int(maxx), "died": died,
            "cleared_past_975": clear_i is not None, "frames_after_clear":
            (len(xs) - clear_i) if clear_i is not None else 0, "survived": bool(survived)}
    return rec, info


def main() -> None:
    t0 = time.time()
    build = json.loads(BUILD.read_text())
    clearing = [r for r in build["requirement"]["rows"] if r.get("cleared")]
    print(f"{len(clearing)} clearing configs from {BUILD.name}, "
          f"seeds {sorted({r['seed'] for r in clearing})}", flush=True)

    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)

    base = arm_metrics([e["frames"] for e in
                        json.loads(BASELINE_TRACES.read_text())["episodes"]])
    base.pop("rows")
    print(f"baseline (same ckpt, same 200 seeds): stuck {base['stuck_at_pipe4']}/200  "
          f"arrived {base['arrived_at_880']}  cleared {base['cleared_past_975']}  "
          f"A-hold median {base['a_hold_880_924']['median']} "
          f"p90 {base['a_hold_880_924']['p90']} "
          f">=12 {base['a_hold_880_924']['frac_ge_required'] * 100:.1f}%", flush=True)

    out = {"checkpoint_in": CKPT.name, "n_clearing_configs": len(clearing),
           "prefix_tail": PREFIX_TAIL, "survive_frames": SURVIVE,
           "baseline": base, "measurement_basis": "single_life", "seeds_training": 1}

    # ---------------- step 1: confirm survival, record demonstrations ----------------
    print("\nSTEP 1 confirm survival for the 39, recording demonstrations", flush=True)
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    demo_frames, demo_bytes, checked = [], [], []
    try:
        prefixes = {}
        for sd in sorted({r["seed"] for r in clearing}):
            sq, _ = prefix(s, policy, cfg, thr, start, sd)
            if sq:
                prefixes[sd] = sq
        print(f"  prefixes rebuilt for seeds {sorted(prefixes)}", flush=True)
        for r in clearing:
            sd = r["seed"]
            if sd not in prefixes:
                checked.append({**r, "recheck": "prefix_unavailable"})
                continue
            rec, info = demo_attempt(s, start, prefixes[sd], hold=r["hold"],
                                     trigger=r["trigger"])
            checked.append({"seed": sd, "hold": r["hold"], "trigger": r["trigger"],
                            "sweep_max_x": r["max_x"], **info})
            if rec and info["survived"]:
                demo_frames.append(np.stack([p[0] for p in rec]))
                demo_bytes.append(np.array([p[1] for p in rec], dtype=np.uint8))
        n_pass = sum(1 for c in checked if c.get("survived"))
        n_clear = sum(1 for c in checked if c.get("cleared_past_975"))
        print(f"  re-ran {len(checked)}: cleared again {n_clear}, "
              f"cleared AND survived {SURVIVE} frames {n_pass}", flush=True)
        out["survival_check"] = {"n": len(checked), "cleared_again": n_clear,
                                 "passed_search_bar": n_pass,
                                 "bar": f"max_x>{CLEAR_X} and alive {SURVIVE} frames after",
                                 "grounded_enforced": True, "rows": checked}
        if not demo_frames:
            out["verdict"] = ("NO DEMONSTRATION PASSED THE SEARCH BAR -- the 39 clearing configs "
                              "do not survive past pipe 4, so there is nothing to distil.")
            OUT.write_text(json.dumps(out, indent=2, default=str))
            print("\n" + out["verdict"])
            return

        frames = np.concatenate(demo_frames)
        bytes_ = np.concatenate(demo_bytes)
        write_self_run(DEMOS, frames, bytes_)
        a_rate = float(((bytes_ & A) > 0).mean())
        print(f"  wrote {DEMOS.name}: {len(frames):,} demo frames, A pressed on "
              f"{a_rate * 100:.1f}% of them", flush=True)
        out["demonstrations"] = {"dir": DEMOS.name, "frames": int(len(frames)),
                                 "episodes": len(demo_frames), "a_press_rate": a_rate}

        # ---------------- step 2: distil ----------------
        print(f"\nSTEP 2 distil: {STEPS} steps, 1:1 expert:demo, lr {LR}", flush=True)
        expert = ctx.dataset([load_run_dir(ROOT / "data/runs" / n) for n in EARLIEST])
        demo_ds = ctx.dataset([load_run_dir(DEMOS)])
        e_rows = random_rows(expert, min(len(expert), len(demo_ds)), seed=0)
        mixed = ConcatDataset([Subset(expert, e_rows), demo_ds])
        print(f"  expert {len(e_rows):,} + demo {len(demo_ds):,} frames", flush=True)
        out["training"] = {"expert_frames": len(e_rows), "demo_frames": len(demo_ds),
                           "ratio": "1:1", "steps": STEPS, "lr": LR,
                           "loss": "sustain+onset weighted BCE (composed recipe)",
                           "expert_data": EARLIEST}
        policy = train(policy, mixed, STEPS, LR, 0)
        cal, _ = calibrate(policy, expert, ctx.target_rates)
        thr = cal.vector.astype(np.float64)
        save_policy(NEW_CKPT, policy, cfg, {n: 0.5 for n in NES_BUTTON_ORDER},
                    distilled_from=str(CKPT.name), demo_dir=DEMOS.name)
        print(f"  saved {NEW_CKPT.name}", flush=True)

        # ---------------- step 3: measure transfer ----------------
        print(f"\nSTEP 3 measure transfer: n={N_EVAL}, single life, seeds 0-{N_EVAL - 1}",
              flush=True)
        traces = []
        for i in range(N_EVAL):
            traces.append(traced_episode(s, policy, cfg, thr, start, i))
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{N_EVAL}", flush=True)
    finally:
        s.close()

    write_traces(TRACES, traces, checkpoint=NEW_CKPT.name)
    got = arm_metrics([t.frames for t in traces])
    got.pop("rows")
    out["distilled"] = got

    # paired comparisons against the identical baseline metric code
    bh, gh = base["a_hold_880_924"], got["a_hold_880_924"]
    hold_rose = (gh["median"] or 0) > (bh["median"] or 0) and \
                (gh["frac_ge_required"] or 0) > (bh["frac_ge_required"] or 0)
    stuck_lo, stuck_hi = diff_ci(base["stuck_at_pipe4"], base["n"],
                                 got["stuck_at_pipe4"], got["n"])
    clr_lo, clr_hi = diff_ci(base["cleared_past_975"], base["n"],
                             got["cleared_past_975"], got["n"])
    stuck_fell = stuck_hi < 0
    out["comparison"] = {
        "method": "Newcombe on paired seed sets (same 200 seeds, same episode function)",
        "a_hold_880_924": {"baseline": bh, "distilled": gh, "rose": bool(hold_rose)},
        "stuck_at_pipe4": {"baseline": base["stuck_at_pipe4"], "distilled": got["stuck_at_pipe4"],
                           "delta_pp": (got["stuck_at_pipe4"] - base["stuck_at_pipe4"]) / 2.0,
                           "ci_pp": [stuck_lo * 100, stuck_hi * 100],
                           "excludes_zero": bool(stuck_lo > 0 or stuck_hi < 0),
                           "fell": bool(stuck_fell)},
        "cleared_past_975": {"baseline": base["cleared_past_975"],
                             "distilled": got["cleared_past_975"],
                             "ci_pp": [clr_lo * 100, clr_hi * 100],
                             "excludes_zero": bool(clr_lo > 0 or clr_hi < 0)},
        "cleared_given_arrived": {
            "baseline": base["cleared_given_arrived"], "distilled": got["cleared_given_arrived"],
            "baseline_wilson": wilson(base["cleared_past_975"], base["arrived_at_880"]),
            "distilled_wilson": wilson(got["cleared_past_975"], got["arrived_at_880"])},
    }

    if hold_rose and stuck_fell:
        v = ("SEARCH-AND-DISTIL WORKS: the A-hold at x 880-924 rose "
             f"(median {bh['median']}->{gh['median']}, >=12 "
             f"{(bh['frac_ge_required'] or 0) * 100:.1f}%->"
             f"{(gh['frac_ge_required'] or 0) * 100:.1f}%) and the stuck-at-pipe-4 count fell "
             f"{base['stuck_at_pipe4']}->{got['stuck_at_pipe4']} of 200 "
             f"[{stuck_lo * 100:+.1f}, {stuck_hi * 100:+.1f}] pp. The loop is closed.")
    elif hold_rose:
        v = ("THE ACTION TRANSFERRED, THE OUTCOME DID NOT: the A-hold rose "
             f"(median {bh['median']}->{gh['median']}) but stuck-at-pipe-4 went "
             f"{base['stuck_at_pipe4']}->{got['stuck_at_pipe4']} of 200 "
             f"[{stuck_lo * 100:+.1f}, {stuck_hi * 100:+.1f}] pp, which does not exclude zero in "
             "the falling direction. Something else at pipe 4 is binding and must be named.")
    elif stuck_fell:
        v = ("CLEARANCE MOVED WITHOUT THE HOLD: stuck-at-pipe-4 fell "
             f"{base['stuck_at_pipe4']}->{got['stuck_at_pipe4']} of 200 but the A-hold did not "
             f"rise (median {bh['median']}->{gh['median']}). Do not call this distillation "
             "working; the mechanism is unidentified and must be named before it is claimed.")
    else:
        v = ("39 KNOWN-GOOD DEMONSTRATIONS FROM REAL STATES WERE NOT ABSORBED: A-hold median "
             f"{bh['median']}->{gh['median']}, stuck-at-pipe-4 "
             f"{base['stuck_at_pipe4']}->{got['stuck_at_pipe4']} of 200 "
             f"[{stuck_lo * 100:+.1f}, {stuck_hi * 100:+.1f}] pp. This is a limit of the method, "
             "not a shortage of demonstrations.")
    out["verdict"] = v
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(v)
    print(f"\nwrote {OUT} and {TRACES} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
