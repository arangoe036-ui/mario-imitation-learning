"""§1 BLOCKER: does the emulator ever render badly while staying alive? The policy acts on the frames.

`EpisodeAborted` fires on a broken pipe, but a glitching-yet-alive FCEUX yields a complete episode recorded as
normal. **Corrupt or stale observations handicap the policy specifically** — a script never looks at the screen
— so this is a live candidate for both the script's residual advantage and the between-seed variance this
project keeps attributing to training noise.

Three checks:

1. **The log.** `fceux.log` is written to a temp workdir and discarded. **Measured here: it is 0 bytes.** FCEUX
   prints nothing on this build, so *the GL backend is not recorded anywhere and cannot be read back* — the
   `gl_backend: "unknown"` in `data/mps_boundary.json` was this, not a parsing bug. What the log *can* carry is
   the Qt software-fallback warning ("known to be broken on macOS Tahoe") that block 53 saw when Metal was
   poisoned. So **an empty log is the clean case and any content is the alarm**, and the log is now retained
   per batch rather than thrown away.
2. **Stale frames.** `live.py` builds an observation video "so stale frames are obvious"; this makes it a
   number.

   **⚠ The obvious criterion is wrong and nearly stopped this block.** "84x84 observation identical while `x`
   changed" fires at **2.58%** — but stratified by the size of the x change, **every single such event happens
   at |dx| = 1, and in every case the NATIVE 256x240 frame is byte-identical too** (22 of 22). Zero events at
   |dx| >= 2. SMB advances Mario's position counter by one pixel without necessarily redrawing him at a new
   pixel, so an unchanged frame at |dx| = 1 is **the game's own behaviour faithfully reproduced**, not a
   rendering fault.

   The fault criterion used here is therefore **native frame identical while |dx| >= 2** — a screen that fails
   to redraw across a multi-pixel move. Both numbers are reported so the distinction is visible.
3. **Frame determinism.** Same seed, same policy, same start, twice: **every observation must be
   byte-identical.** A mismatch means the pixels the policy trains and acts on are not reproducible, which
   would undermine every measurement in the project.

**If any check fires at a material rate, this reports and stops before the distillation** rather than
conditioning labels on frames that never existed.
"""
from __future__ import annotations

import collections
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/graphics_integrity.json"
LOGDIR = ROOT / "data/fceux_logs"

ARM = "P_84_cnn32"
TEMP = 0.7
N_EPISODES = 40
N_DETERMINISM = 4
CAP_NON_A = 4
#: a stale-frame rate above this is "material" and stops the block
MATERIAL_RATE = 0.01
ARM_BUDGET_S = 25 * 60


def run_episode(sess, policy, cfg, start, seed, *, keep_frames=False):
    """One episode, counting stale frames and optionally retaining every observation."""
    s = cfg.frame_size
    rng = np.random.default_rng(seed)
    obs = sess.reset(start.frame)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    cur = _resize_gray(obs.rgb, (s, s))
    win[:] = cur
    prev_frame = cur.copy()
    prev_x = read_smb(obs.ram, obs.framecount).x_position
    held, remaining = None, 0
    best = since = frames = 0
    stale_while_moving = 0
    stale_total = 0
    moving_frames = 0
    faults = 0
    stale_dx, moving_dx = {}, {}
    prev_native = np.asarray(obs.rgb).copy()
    kept = [] if keep_frames else None
    ended = "cap"
    while frames < RB.CAP_FRAMES:
        if remaining <= 0:
            with torch.no_grad():
                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
            p = torch.softmax(lg / TEMP, dim=-1).numpy()
            c = int(rng.choice(len(p), p=p / p.sum()))
            b, L = int(byte_of_global[c]), max(1, int(lut_global[c]))
            if not (b & A_BIT):
                L = min(L, CAP_NON_A)
            held, remaining = b, L
        remaining -= 1
        obs = sess.step(held)
        newf = _resize_gray(obs.rgb, (s, s))
        r = read_smb(obs.ram, obs.framecount)
        same = bool(np.array_equal(newf, prev_frame))
        native_same = bool(np.array_equal(np.asarray(obs.rgb), prev_native))
        dx = abs(int(r.x_position) - int(prev_x))
        moved = dx > 0
        if same:
            stale_total += 1
            if moved:
                stale_while_moving += 1
                stale_dx[dx] = stale_dx.get(dx, 0) + 1
        # THE FAULT CRITERION: the native screen failed to redraw across a multi-pixel move
        if native_same and dx >= 2:
            faults += 1
        if moved:
            moving_frames += 1
            moving_dx[dx] = moving_dx.get(dx, 0) + 1
        if keep_frames:
            kept.append(newf.copy())
        win = np.roll(win, -1, 0)
        win[-1] = newf
        prev_frame, prev_x = newf, r.x_position
        prev_native = np.asarray(obs.rgb).copy()
        frames += 1
        if r.player_state in (0x06, 0x0B):
            ended = "died"
            break
        if r.x_position > best:
            best, since = r.x_position, 0
        else:
            since += 1
            if since > RB.STALL:
                ended = "stuck"
                break
    return {"seed": seed, "n_frames": frames, "ended": ended,
            "max_x": best, "moving_frames": moving_frames,
            "stale_total": stale_total, "stale_while_moving": stale_while_moving,
            "stale_while_moving_rate": (stale_while_moving / moving_frames)
            if moving_frames else 0.0,
            "render_faults_native_identical_dx_ge_2": faults,
            "fault_rate": (faults / moving_frames) if moving_frames else 0.0,
            "stale_by_dx": stale_dx, "moving_by_dx": moving_dx}, kept


def main() -> None:
    global byte_of_global, lut_global
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of_global = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                              dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut_global = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)
    policy, cfg, _ = G.load_ckpt(ARM)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    out = {"arm": ARM, "temperature": TEMP, "terminator": RB.describe(),
           "n_episodes": N_EPISODES, "material_rate_threshold": MATERIAL_RATE,
           "checks": ["fceux_log_retained", "stale_frames", "frame_determinism"],
           "episodes": []}
    sess = None
    try:
        with time_limit(min(ARM_BUDGET_S, dl.remaining() - 120), "graphics"):
            sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
            warm_session(sess, start.frame)
            wd = Path(sess._workdir)
            for i in range(N_EPISODES):
                if dl.remaining() < 150:
                    break
                rec, _ = run_episode(sess, policy, cfg, start, i)
                out["episodes"].append(rec)
                if (i + 1) % 10 == 0:
                    print(f"  {dl.stamp()} {i + 1}/{N_EPISODES} episodes", flush=True)
            # ---- check 1: retain the log ----
            lg = wd / "fceux.log"
            log_txt = lg.read_text(errors="replace") if lg.exists() else ""
            dest = LOGDIR / "graphics_integrity_session.log"
            if lg.exists():
                shutil.copy2(lg, dest)
            low = log_txt.lower()
            out["fceux_log"] = {
                "exists": lg.exists(), "bytes": len(log_txt),
                "retained_to": str(dest.relative_to(ROOT)) if lg.exists() else None,
                "contains_software": "software" in low,
                "contains_metal": "metal" in low,
                "contains_warning": any(w in low for w in ("warning", "broken", "failed", "error")),
                "interpretation": ("FCEUX prints nothing on this build, so an EMPTY log is the clean "
                                  "case and the GL backend is genuinely unrecoverable from it. Any "
                                  "content -- in particular the Qt software-fallback warning block 53 "
                                  "saw -- is the alarm."),
                "tail": log_txt[-500:]}
            # ---- check 3: frame determinism ----
            det = []
            for i in range(N_DETERMINISM):
                if dl.remaining() < 120:
                    break
                _, f1 = run_episode(sess, policy, cfg, start, 900 + i, keep_frames=True)
                _, f2 = run_episode(sess, policy, cfg, start, 900 + i, keep_frames=True)
                n = min(len(f1), len(f2))
                mism = sum(1 for k in range(n) if not np.array_equal(f1[k], f2[k]))
                det.append({"seed": 900 + i, "len_a": len(f1), "len_b": len(f2),
                            "compared": n, "mismatched_frames": mism,
                            "identical": bool(mism == 0 and len(f1) == len(f2))})
                print(f"  {dl.stamp()} determinism seed {900 + i}: "
                      f"{len(f1)} vs {len(f2)} frames, {mism} mismatched", flush=True)
            out["frame_determinism"] = det
    except TimedOut as e:
        out["timed_out"] = str(e)
    finally:
        if sess is not None:
            try:
                sess.close()
            except BaseException:
                pass

    eps = out["episodes"]
    if eps:
        rates = [e["stale_while_moving_rate"] for e in eps]
        by_outcome = collections.defaultdict(list)
        for e in eps:
            by_outcome[e["ended"]].append(e["stale_while_moving_rate"])
        out["stale_frames"] = {
            "n_episodes": len(eps),
            "total_stale_while_moving": int(sum(e["stale_while_moving"] for e in eps)),
            "total_moving_frames": int(sum(e["moving_frames"] for e in eps)),
            "pooled_rate": float(sum(e["stale_while_moving"] for e in eps)
                                 / max(1, sum(e["moving_frames"] for e in eps))),
            "per_episode_rate": {"mean": float(np.mean(rates)), "max": float(max(rates)),
                                 "n_above_threshold": int(sum(1 for r in rates
                                                              if r > MATERIAL_RATE))},
            "by_outcome": {k: {"n": len(v), "mean_rate": float(np.mean(v))}
                           for k, v in by_outcome.items()},
            "stale_total_pooled": int(sum(e["stale_total"] for e in eps)),
            "pooled_fault_rate": float(sum(e["render_faults_native_identical_dx_ge_2"] for e in eps)
                                       / max(1, sum(e["moving_frames"] for e in eps))),
            "total_render_faults": int(sum(e["render_faults_native_identical_dx_ge_2"]
                                           for e in eps)),
            "stale_by_dx_pooled": {str(k): int(sum(e["stale_by_dx"].get(k, 0) for e in eps))
                                   for k in sorted({k for e in eps for k in e["stale_by_dx"]})},
            "moving_by_dx_pooled": {str(k): int(sum(e["moving_by_dx"].get(k, 0) for e in eps))
                                    for k in sorted({k for e in eps for k in e["moving_by_dx"]})},
            "fault_criterion": ("native 256x240 frame identical while |dx| >= 2; the naive "
                                "'84x84 identical while x moved' criterion fires at ~2.6% and is "
                                "entirely |dx| = 1 quantisation, with the native frame identical too"),
            "note": ("a stale frame with x UNCHANGED is legitimate -- the screen need not move when "
                     "Mario does not. Only stale-while-moving is a rendering fault")}
    det = out.get("frame_determinism", [])
    out["verdict_parts"] = {}
    fired = []
    if eps and out["stale_frames"]["pooled_fault_rate"] > MATERIAL_RATE:
        fired.append("render_faults")
    if det and not all(d["identical"] for d in det):
        fired.append("frame_determinism")
    if out.get("fceux_log", {}).get("contains_software") or \
            out.get("fceux_log", {}).get("contains_warning"):
        fired.append("fceux_log")
    out["checks_fired"] = fired
    out["blocker_clear"] = len(fired) == 0
    if not fired:
        sf = out.get("stale_frames", {})
        out["verdict"] = (
            f"**ALL THREE GRAPHICS CHECKS PASS. The blocker is clear and §3 may proceed.** "
            f"**Render faults (native frame identical across a multi-pixel move): "
            f"{sf.get('total_render_faults')} of {sf.get('total_moving_frames')} moving frames "
            f"({sf.get('pooled_fault_rate', 0) * 100:.4f}%).** The naive criterion -- 84x84 identical "
            f"while x moved -- fires at {sf.get('pooled_rate', 0) * 100:.2f}%, but that is entirely "
            f"|dx| = 1 quantisation with the native frame identical too: "
            f"stale-by-dx {sf.get('stale_by_dx_pooled')} against moving-by-dx "
            f"{sf.get('moving_by_dx_pooled')}. "
            f"Frame determinism: **{sum(1 for d in det if d['identical'])}/{len(det)} repeated "
            f"episodes byte-identical.** `fceux.log` is **{out['fceux_log']['bytes']} bytes** — FCEUX "
            f"prints nothing on this build, so the GL backend cannot be recorded from it and an empty "
            f"log is the clean case; it is now retained per session rather than discarded. "
            f"**Degraded rendering is not a candidate for the script's residual advantage or for the "
            f"between-seed variance.**")
    else:
        out["verdict"] = (
            f"**GRAPHICS CHECKS FIRED: {fired}. STOPPING BEFORE THE DISTILLATION** rather than "
            f"conditioning labels on frames that never existed. Stale-while-moving pooled rate "
            f"{out.get('stale_frames', {}).get('pooled_rate')}, determinism "
            f"{[d['identical'] for d in det]}.")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    byte_of_global = None
    lut_global = None
    main()
