"""P1+P2: what does the POLICY do at the gap, and is it B or speed?

The sweep said what the environment requires. This says what the model does -- a distinction this
project has conflated three times, each time costing a directive.

Payloads, one run:
  1. per-frame (x, absolute y, speed byte, buttons) for any episode reaching x>=1150
  2. speed and A-press position on arrival at the near lip (x~1475), against the sweep's
     requirement of speed 40 and trigger 1435-1483
  3. death classification with the fixed y, closing the pit question
P2 is the same harness: accelerate with B, release B on the takeoff frame, arrive at speed 40
without B held. Crosses -> speed is the variable; fails -> B in flight matters.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import session_when_free
from scripts.gap_sweep import prefix_to, replay
from tasdata.bc.overnight_lib import calibrate, load_policy, wilson
from tasdata.bc.tokens import LIVE_MASK
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from tasdata.ram import read_smb, y_absolute
from tasdata.replay import _resize_gray

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/gap_policy.json"
CKPT = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
RIGHT, B, A = NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["A"]
NEAR, FAR, FLOOR = 1475, 1531, 176
WINDOW = (1435, 1483)
SPD = 0x0057

def episode(session, policy, cfg, thr, start, seed, cap=3000):
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8); win[:] = _resize_gray(obs.rgb, (84, 84))
    trail, maxx, death, lip = [], 0, None, None
    prev_a = False
    for _ in range(cap):
        with torch.no_grad():
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0/(1.0+np.exp(-lg))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]: byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        obs = session.step(byte)
        win = np.roll(win, -1, 0); win[-1] = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        ya = y_absolute(obs.ram); spd = int(obs.ram[SPD]); x = st.x_position
        maxx = max(maxx, x)
        if x >= 1150:
            trail.append((x, ya, spd, byte))
        # first A-onset inside the approach corridor
        a_now = bool(byte & A)
        if lip is None and 1400 <= x <= FAR and a_now and not prev_a:
            lip = {"a_onset_x": x, "speed_at_onset": spd, "b_held": bool(byte & B),
                   "in_window": bool(WINDOW[0] <= x <= WINDOW[1])}
        prev_a = a_now
        # speed on first arrival at the near lip
        if lip is None and x >= NEAR - 4:
            lip = {"a_onset_x": None, "speed_at_lip": spd, "b_held": bool(byte & B),
                   "in_window": None}
        if st.player_state in (0x06, 0x0B):
            death = {"x": int(x), "y_abs": ya, "y_wrapped": int(st.y_position),
                     "is_pit": bool(ya > FLOOR + 40)}
            break
    return {"seed": seed, "max_x": int(maxx), "lip": lip, "death": death,
            "reached_1150": bool(trail), "n_trail": len(trail)}

def release_b(session, start, seq, *, hold, trigger, frames=300):
    """P2: accelerate with B, drop B on the takeoff frame."""
    obs = replay(session, start, seq)
    left, jumped, maxx, landed = hold, False, 0, None
    spd0 = 0
    for _ in range(frames):
        st = read_smb(obs.ram, obs.framecount)
        if not jumped and st.x_position >= trigger:
            jumped = True; spd0 = int(obs.ram[SPD])
        byte = RIGHT if jumped else (RIGHT | B)      # B released at takeoff
        if jumped and left > 0:
            byte |= A; left -= 1
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if landed is None and jumped and st.y_position == FLOOR and st.x_position > trigger + 8:
            landed = int(st.x_position)
        if st.player_state in (0x06, 0x0B): break
    return {"hold": hold, "trigger": trigger, "takeoff_speed": spd0,
            "max_x": int(maxx), "landed_x": landed,
            "crossed": bool(landed is not None and landed > FAR)}

def main():
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    out = {"checkpoint": CKPT.name, "near_lip": NEAR, "far_lip": FAR, "window": WINDOW}
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        # y-fix assertion: a fall must read monotonically past 256
        seq = None
        for sd in range(60):
            seq = prefix_to(s, policy, cfg, thr, start, 1150, sd)
            if seq: break
        obs = replay(s, start, seq)
        ys = []
        for _ in range(200):
            obs = s.step(RIGHT); ys.append(y_absolute(obs.ram))
        out["y_fix_check"] = {"max_y_absolute": max(ys), "exceeds_256": bool(max(ys) > 256),
                              "tail": ys[-10:]}
        print(f"y fix: max absolute y {max(ys)} (wrapped version reported 252); "
              f"exceeds 256: {max(ys) > 256}")

        eps = [episode(s, policy, cfg, thr, start, i) for i in range(200)]

        p2 = [release_b(s, start, seq, hold=h, trigger=t)
              for t in range(1435, 1487, 4) for h in (4, 8, 12, 16, 20)]
    finally:
        s.close()

    reach = [e for e in eps if e["max_x"] >= NEAR - 4 and e["lip"]]
    spds = [e["lip"].get("speed_at_onset") or e["lip"].get("speed_at_lip") for e in reach]
    spds = [v for v in spds if v is not None]
    onsets = [e["lip"] for e in reach if e["lip"].get("a_onset_x")]
    inw = sum(1 for l in onsets if l["in_window"])
    deaths = [e["death"] for e in eps if e["death"]]
    pits = [d for d in deaths if d["is_pit"]]
    out.update({"n_reaching_lip": len(reach), "speeds_at_lip": spds,
                "speed_40_count": sum(1 for v in spds if v >= 40),
                "a_onsets_in_corridor": len(onsets), "a_onsets_in_window": inw,
                "n_deaths": len(deaths), "n_pits": len(pits),
                "pit_x": sorted(d["x"] for d in pits),
                "p2_release_b": p2,
                "p2_crossed": sum(1 for r in p2 if r["crossed"])})
    print(f"\nepisodes reaching the near lip: {len(reach)}/200")
    if spds:
        print(f"  speed byte at the lip: median {np.median(spds):.0f} "
              f"max {max(spds)} | at 40: {out['speed_40_count']}/{len(spds)} "
              f"({out['speed_40_count']/len(spds)*100:.0f}%)")
        print(f"  distribution: {dict(Counter(spds))}")
    print(f"  A-onsets in the corridor: {len(onsets)}, of which inside the "
          f"{WINDOW[0]}-{WINDOW[1]} window: {inw}")
    print(f"\ndeaths {len(deaths)}, classified as pits with fixed y: {len(pits)}")
    print(f"  pit x positions: {out['pit_x'][:20]}")
    print(f"\nP2 release-B-at-takeoff: {out['p2_crossed']}/{len(p2)} crossed")
    med = float(np.median(spds)) if spds else 0
    out["binary"] = (f"SPEED IS NOT THE DEFICIT: median speed at the lip is {med:.0f} "
                     f"({out['speed_40_count']}/{len(spds)} at 40)"
                     if med >= 40 else
                     f"ARRIVES TOO SLOW: median speed at the lip is {med:.0f}, not 40")
    print(f"\nBINARY: {out['binary']}")
    OUT.write_text(json.dumps(out, indent=2, default=str)); print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
