"""P1: is the 21.5% real, or is Mario clipping through pipes?

The owner reports glitchy play. The per-frame trajectories needed to check this were never
saved -- previous runs stored only max_x, ended and death -- so this re-runs the same seeded
episodes with (x, y, action) logged per frame, then applies four checks.

Pipe surfaces, measured independently from expert traces (data/terrain_profile.json):
pipe 1 y=144 (2 tiles), pipe 2 y=128 (3 tiles), pipe 3 y=112 (4 tiles). Floor is y=176.
Crossing a pipe's x-range while y stays at floor level is a clip, not a clearance.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from tasdata.bc.overnight_lib import calibrate, load_policy, wilson
from scripts.compose import session_when_free
from tasdata.bc.session import FceuxSession
from tasdata.bc.tokens import LIVE_MASK
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from tasdata.dataset import load_run_dir
from tasdata.ram import column, read_smb
from tasdata.replay import _resize_gray

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/clip_test.json"
CKPT = ROOT / "data/bc_overnight/round3_ratio1to1.pt"
SEEDS, MAX_FRAMES, STALL, FLOOR = 200, 2500, 300, 176
PIPES = {"pipe1": (420, 470, 144), "pipe2": (575, 640, 128), "pipe3": (690, 768, 112)}
MAX_DX, MAX_DY = 4, 8      # generous SMB limits: run ~2.5 px/f, fall ~4-5 px/f

def rollout(session, policy, cfg, thr, start, seed):
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), dtype=np.uint8); win[:] = _resize_gray(obs.rgb, (84, 84))
    xs, ys, acts = [], [], []
    best, since = read_smb(obs.ram, obs.framecount).x_position, 0
    ended = "budget"
    for _ in range(MAX_FRAMES):
        with torch.no_grad():
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0/(1.0+np.exp(-lg))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]: byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        obs = session.step(byte)
        win = np.roll(win, -1, axis=0); win[-1] = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        xs.append(st.x_position); ys.append(st.y_position); acts.append(byte)
        if st.player_state in (0x06, 0x0B): ended = "died"; break
        if st.x_position > best: best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL: ended = "stuck"; break
    return np.array(xs), np.array(ys), np.array(acts, dtype=np.uint8), ended

def main():
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)

    eps = []
    # The composition job holds the one-emulator lock intermittently; wait rather than fail.
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        for i in range(SEEDS):
            eps.append(rollout(s, policy, cfg, thr, start, i))
        # 4. determinism
        a = rollout(s, policy, cfg, thr, start, 0); b = rollout(s, policy, cfg, thr, start, 0)
        det = bool(len(a[0]) == len(b[0]) and (a[0] == b[0]).all() and (a[1] == b[1]).all()
                   and (a[2] == b[2]).all())
    finally:
        s.close()
    print(f"4. determinism: seed 0 re-run identical frame-for-frame: {det}")

    # 1. clip test. CROSSED means max_x got past the far edge -- an episode that merely
    # ENTERS the x-range and stalls against the pipe face is sitting at floor level without
    # clipping anything, and counting those as clips is a category error.
    clip = {}
    for name, (lo, hi, surf) in PIPES.items():
        entered = crossed_over = crossed_floor = stalled = 0
        for xs, ys, _a, _e in eps:
            m = (xs >= lo) & (xs <= hi)
            if not m.any(): continue
            entered += 1
            if xs.max() <= hi:
                stalled += 1                      # never got past the far edge
            elif ys[m].min() <= surf:
                crossed_over += 1                 # went over the surface: a real jump
            else:
                crossed_floor += 1                # crossed at floor level: a clip
        nc = crossed_over + crossed_floor
        clip[name] = {"entered_x_range": entered, "stalled_inside": stalled,
                      "crossed": nc, "crossed_over_surface": crossed_over,
                      "crossed_at_floor": crossed_floor, "surface_y": surf,
                      "clip_fraction_of_crossings": (crossed_floor / nc if nc else None)}
        print(f"1. {name}: entered x{lo}-{hi} {entered}; stalled inside {stalled}; "
              f"CROSSED {nc} of which over-the-surface {crossed_over}, "
              f"floor-level(clip) {crossed_floor}"
              + (f" ({crossed_floor / nc * 100:.1f}% of crossings)" if nc else ""))

    # 2. discontinuity scan
    viol = []
    for i, (xs, ys, _a, _e) in enumerate(eps):
        dx, dy = np.abs(np.diff(xs.astype(int))), np.abs(np.diff(ys.astype(int)))
        for f in np.flatnonzero(dx > MAX_DX):
            if xs[f] > 0 and xs[f+1] > 0: viol.append({"ep": i, "frame": int(f), "type": "dx",
                                                       "value": int(dx[f])})
        for f in np.flatnonzero(dy > MAX_DY):
            viol.append({"ep": i, "frame": int(f), "type": "dy", "value": int(dy[f])})
    print(f"2. discontinuities: {len(viol)} frames exceed |dx|>{MAX_DX} or |dy|>{MAX_DY} "
          f"across {len(eps)} episodes"
          + (f"; worst {max(v['value'] for v in viol)}" if viol else ""))

    # 3. input contamination: longest exact match against any expert action sequence
    experts = [np.asarray(load_run_dir(ROOT/"data/runs"/n).actions, dtype=np.uint8)
               for n in ctx.split["train"][:6]]
    worst = 0
    for xs, ys, acts, _e in eps[:40]:
        for ex in experts:
            k = min(len(acts), 600)
            for off in range(0, min(len(ex) - k, 4000), 200):
                seg = ex[off:off+k]
                eq = (seg == acts[:k])
                run = best = 0
                for v in eq:
                    run = run + 1 if v else 0
                    best = max(best, run)
                worst = max(worst, best)
    print(f"3. input contamination: longest exact action run matching any expert: {worst} frames")

    f2 = clip["pipe2"]["clip_fraction_of_crossings"]
    kill = f2 is not None and f2 > 0.10
    verdict = (f"CLIPPING: {clip['pipe2']['crossed_at_floor']}/{clip['pipe2']['crossed']} "
               f"pipe-2 CROSSINGS are floor level ({(f2 or 0) * 100:.1f}%) -- the 21.5% is void"
               if kill else
               f"NO CLIPPING at pipe 2: {clip['pipe2']['crossed_over_surface']}/"
               f"{clip['pipe2']['crossed']} crossings go over the surface, "
               f"floor-level {clip['pipe2']['crossed_at_floor']}. The 21.5% stands.")
    print("\n" + "="*78); print(f"BINARY: {verdict}")
    OUT.write_text(json.dumps({"determinism_identical": det, "clip": clip,
                               "n_discontinuities": len(viol), "discontinuities": viol[:40],
                               "longest_expert_action_match_frames": worst,
                               "verdict": verdict}, indent=2))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
