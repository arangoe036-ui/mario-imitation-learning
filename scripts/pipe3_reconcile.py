"""P1: is the x=724 ceiling real, or an artifact? Answers (a)-(d) in one run.

Four candidate explanations, all checked here:

(a) **Frame budget.** Episodes stall 300 frames at pipe 2 before arriving at pipe 3; if they
    arrive near frame 2500 of 2500, "stalled at pipe 3" means "the episode ended".
(b) **Termination reason**, split three ways -- timeout / stuck / died -- not merged.
(c) **Bigger budget**: 10,000 frames instead of 2,500.
(d) **x cross-check**, and the specific candidate I think most likely: SMB 1-1 contains an
    *enterable* pipe. If Mario goes down it the game switches `area`, x resets to a small value
    in the bonus room, and `max_x` freezes at whatever it was -- which looks exactly like a hard
    ceiling. `area` was never recorded, so this could not have been seen. Recorded now.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import session_when_free
from tasdata.bc.overnight_lib import calibrate, load_policy, wilson
from tasdata.bc.tokens import LIVE_MASK
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from tasdata.ram import read_smb
from tasdata.replay import _resize_gray

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/pipe3_reconcile.json"
CKPT = ROOT / "data/bc_overnight/round3_ratio1to1.pt"
SEEDS, BUDGET, STALL = 200, 10000, 300
ADDR_XPAGE, ADDR_XLO = 0x006D, 0x0086

def rollout(session, policy, cfg, thr, start, seed):
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), dtype=np.uint8); win[:] = _resize_gray(obs.rgb, (84, 84))
    best, since = read_smb(obs.ram, obs.framecount).x_position, 0
    maxx = best; arrival = None; ended = "timeout"; xmismatch = 0
    areas = []; levels = []; f = 0
    for f in range(BUDGET):
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
        # (d) independent x from raw RAM
        raw_x = int(obs.ram[ADDR_XPAGE]) * 256 + int(obs.ram[ADDR_XLO])
        if raw_x != st.x_position: xmismatch += 1
        areas.append(st.area); levels.append((st.world, st.stage))
        maxx = max(maxx, st.x_position)
        if arrival is None and st.x_position >= 690: arrival = f
        if st.player_state in (0x06, 0x0B): ended = "died"; break
        if st.x_position > best: best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL: ended = "stuck"; break
    return {"seed": seed, "max_x": int(maxx), "arrival_frame_690": arrival,
            "frames_used": f + 1, "ended": ended, "x_mismatches": xmismatch,
            "areas_seen": sorted(set(int(a) for a in areas)),
            "levels_seen": sorted({(int(a), int(b)) for a, b in levels}),
            "area_changed": len(set(areas)) > 1}

def main():
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        rows = [rollout(s, policy, cfg, thr, start, i) for i in range(SEEDS)]
    finally:
        s.close()

    xs = np.array([r["max_x"] for r in rows])
    arr = [r for r in rows if r["arrival_frame_690"] is not None]
    print(f"(c) budget {BUDGET}: max_x over {SEEDS} eps -> median {np.median(xs):.0f} "
          f"p90 {np.percentile(xs,90):.0f} MAX {xs.max()}")
    k = int((xs > 768).sum()); lo, hi = wilson(k, SEEDS)
    print(f"    cleared pipe 3 (x>768): {k}/{SEEDS} = {k/SEEDS*100:.1f}% [{lo*100:.1f},{hi*100:.1f}]")
    print(f"(a) arrivals at x>=690: {len(arr)}; arrival frame median "
          f"{np.median([r['arrival_frame_690'] for r in arr]):.0f} "
          f"max {max(r['arrival_frame_690'] for r in arr) if arr else '-'}")
    print(f"(b) termination among those {len(arr)}: "
          f"{dict(Counter(r['ended'] for r in arr))}")
    print(f"    frames used median {np.median([r['frames_used'] for r in arr]):.0f}")
    print(f"(d) x mismatches vs raw RAM: {sum(r['x_mismatches'] for r in rows)} frames total")
    areas = Counter(tuple(r["areas_seen"]) for r in rows)
    print(f"(d) area sets seen: {dict(areas)}")
    print(f"    episodes with an area change: {sum(r['area_changed'] for r in rows)}")
    lv = Counter(tuple(r["levels_seen"]) for r in rows)
    print(f"    level sets seen: {dict(list(lv.items())[:4])}")

    verdict = (f"BUDGET WAS NOT THE CAUSE: with {BUDGET} frames, max_x is still {xs.max()} and "
               f"pipe-3 clearance is {k}/{SEEDS}"
               if k == 0 else
               f"THE 0/200 WAS THE BUDGET: with {BUDGET} frames, pipe-3 clearance is "
               f"{k}/{SEEDS} = {k/SEEDS*100:.1f}% [{lo*100:.1f},{hi*100:.1f}] and max_x reaches {xs.max()}")
    print("\n" + "="*78); print(f"BINARY: {verdict}")
    OUT.write_text(json.dumps({"budget": BUDGET, "seeds": SEEDS, "verdict": verdict,
                               "pipe3_rate": {"k": k, "n": SEEDS, "ci": [lo, hi]},
                               "x_max": int(xs.max()), "rows": rows}, indent=2))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
