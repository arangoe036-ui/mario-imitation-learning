"""Does the model jump while walking? Takeoff velocity at pipe 2, model vs expert.

In SMB the initial vertical velocity of a jump is selected from Mario's *horizontal* speed
at takeoff -- a walking jump is strictly shorter than a running one, whatever happens to the
A button afterwards. Arm (a) already produces 21-frame A-holds, longer than the expert's
median 18 at pipe 2, and still stops at x=594. If its takeoff speed is low, the ceiling is
the B button, not the A button.

Velocity is measured as per-frame delta-x for both sides so the two are directly comparable
(the captured expert traces store decoded state, not raw RAM, so the speed byte is not
available for them). For live rollouts the raw speed byte 0x0057 is reported as well.
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
from tasdata.dataset import load_run_dir  # noqa: E402
from tasdata.ram import column, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/velocity_diagnosis.json"
ADDR_X_SPEED = 0x0057

WINDOW = (500, 600)          # the run-up to pipe 2
A_BIT, B_BIT = NES_BUTTON_BITS["A"], NES_BUTTON_BITS["B"]

CHECKPOINTS = [
    ("arm_a_round3", ROOT / "data/bc_overnight/round3_ratio1to1.pt"),
    ("sustain_arm_a", ROOT / "data/bc_followup/a_sustain_and_onset.pt"),
    ("stage2_armB", O.STAGE2_CKPT),
]


def describe(a) -> dict:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "p10": round(float(np.percentile(a, 10)), 3),
            "p90": round(float(np.percentile(a, 90)), 3),
            "max": round(float(a.max()), 3)}


def expert_stats(ctx) -> dict:
    """Takeoff velocity and B usage at A-onsets in the pipe-2 run-up."""
    vel, holds, b_rate_frames, b_at_onset = [], [], [], []
    for name in ctx.split["train"]:
        run = load_run_dir(ROOT / "data/runs" / name)
        if run.manifest.get("measured_route") != "warpless":
            continue
        tr = np.asarray(run.trace)
        w, s = column(tr, "world"), column(tr, "stage")
        x = column(tr, "x_position").astype(np.int64)
        a = np.asarray(run.actions, dtype=np.uint8)
        n = min(len(x), len(a))
        dx = np.zeros(n, dtype=np.int64)
        dx[1:] = x[1:n] - x[:n - 1]
        inwin = (w[:n] == 1) & (s[:n] == 1) & (x[:n] >= WINDOW[0]) & (x[:n] <= WINDOW[1])
        if not inwin.any():
            continue
        b_rate_frames.append(float(((a[:n] & B_BIT) > 0)[inwin].mean()))
        ap = (a[:n] & A_BIT) > 0
        for i in np.flatnonzero(inwin):
            if i == 0 or not ap[i] or ap[i - 1]:
                continue
            # Velocity just before takeoff, averaged over 4 frames to dodge single-frame noise.
            lo = max(1, i - 4)
            vel.append(float(dx[lo:i].mean()) if i > lo else float(dx[i]))
            b_at_onset.append(bool(a[i] & B_BIT))
            j = i
            while j < n and ap[j]:
                j += 1
            holds.append(int(j - i))
    return {"takeoff_velocity_px_per_frame": describe(vel),
            "A_hold_frames": describe(holds),
            "B_held_fraction_of_window": describe(b_rate_frames),
            "B_held_at_takeoff_fraction": (round(float(np.mean(b_at_onset)), 3)
                                           if b_at_onset else None),
            "n_onsets": len(vel)}


def model_stats(ctx, session, policy, cfg, thr, *, episodes: int = 40) -> dict:
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    vel, holds, b_at_onset, speed_byte, b_frac, reached = [], [], [], [], [], 0
    for seed in range(episodes):
        rng = np.random.default_rng(seed)
        obs = session.reset(start.frame)
        window = np.zeros((cfg.stack, 84, 84), dtype=np.uint8)
        window[:] = _resize_gray(obs.rgb, (84, 84))
        prev_x, prev_a = None, False
        trace_x, trace_b, pending = [], [], None
        for _ in range(1200):
            with torch.no_grad():
                logits = policy(torch.from_numpy(window[None]).float().div_(255.0)
                                )[0].numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            bits = rng.random(probs.shape) < probs
            byte = 0
            for j, nm in enumerate(NES_BUTTON_ORDER):
                if bits[j]:
                    byte |= NES_BUTTON_BITS[nm]
            byte &= LIVE_MASK
            obs = session.step(byte)
            window = np.roll(window, -1, axis=0)
            window[-1] = _resize_gray(obs.rgb, (84, 84))
            st = read_smb(obs.ram, obs.framecount)
            x = st.x_position
            a_now = bool(byte & A_BIT)

            if WINDOW[0] <= x <= WINDOW[1]:
                trace_x.append(x)
                trace_b.append(bool(byte & B_BIT))
                if a_now and not prev_a and prev_x is not None:
                    vel.append(float(np.mean(np.diff(trace_x[-5:]))) if len(trace_x) >= 3
                               else float(x - prev_x))
                    b_at_onset.append(bool(byte & B_BIT))
                    sp = int(obs.ram[ADDR_X_SPEED])
                    speed_byte.append(sp - 256 if sp > 127 else sp)
                    pending = 1
                elif a_now and pending is not None:
                    pending += 1
                elif not a_now and pending is not None:
                    holds.append(pending)
                    pending = None
            prev_x, prev_a = x, a_now
            if st.player_state in (0x06, 0x0B):
                break
        if trace_x:
            reached += 1
            b_frac.append(float(np.mean(trace_b)))
    return {"episodes": episodes, "episodes_reaching_window": reached,
            "takeoff_velocity_px_per_frame": describe(vel),
            "raw_speed_byte_0x0057": describe(speed_byte),
            "A_hold_frames": describe(holds),
            "B_held_fraction_of_window": describe(b_frac),
            "B_held_at_takeoff_fraction": (round(float(np.mean(b_at_onset)), 3)
                                           if b_at_onset else None),
            "n_onsets": len(vel)}


def main() -> None:
    ctx = O.Ctx()
    out = {"window_x": WINDOW, "expert": expert_stats(ctx), "models": {}}
    e = out["expert"]
    print("EXPERT in x=500-600 of 1-1:")
    print(f"  takeoff velocity  {e['takeoff_velocity_px_per_frame']}")
    print(f"  A-hold            {e['A_hold_frames']}")
    print(f"  B held, window    {e['B_held_fraction_of_window']}")
    print(f"  B held at takeoff {e['B_held_at_takeoff_fraction']}")

    for label, path in CHECKPOINTS:
        if not Path(path).exists():
            print(f"\n{label}: MISSING")
            continue
        policy, cfg, _ = load_policy(Path(path))
        cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
        thr = cal.vector.astype(np.float64)
        with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
            m = model_stats(ctx, session, policy, cfg, thr)
        out["models"][label] = m
        print(f"\n{label}: reached window in {m['episodes_reaching_window']}/{m['episodes']} eps")
        print(f"  takeoff velocity  {m['takeoff_velocity_px_per_frame']}")
        print(f"  raw speed byte    {m['raw_speed_byte_0x0057']}")
        print(f"  A-hold            {m['A_hold_frames']}")
        print(f"  B held, window    {m['B_held_fraction_of_window']}")
        print(f"  B held at takeoff {m['B_held_at_takeoff_fraction']}")

    ev = out["expert"]["takeoff_velocity_px_per_frame"].get("median", 0)
    verdicts = []
    for label, m in out["models"].items():
        mv = m["takeoff_velocity_px_per_frame"].get("median", 0)
        mb = m["B_held_at_takeoff_fraction"] or 0
        eb = out["expert"]["B_held_at_takeoff_fraction"] or 0
        verdicts.append({
            "model": label, "model_takeoff_velocity": mv, "expert_takeoff_velocity": ev,
            "model_B_at_takeoff": mb, "expert_B_at_takeoff": eb,
            "walking_takeoff": bool(mv < 0.75 * ev if ev else False),
        })
    out["verdicts"] = verdicts
    walking = [v for v in verdicts if v["walking_takeoff"]]
    out["conclusion"] = (
        "THE CEILING IS B, NOT A: the model takes off at a materially lower horizontal speed "
        "than the expert, and in SMB jump height is selected from horizontal speed, so no "
        "A-hold length can compensate."
        if walking else
        "NOT a velocity ceiling: takeoff speeds are comparable to the expert's, so the "
        "failure is elsewhere (timing or placement).")
    print(f"\nCONCLUSION: {out['conclusion']}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
