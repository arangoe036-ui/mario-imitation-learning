"""P1: locate the first gap in 1-1, validate the pit classifier on it, and sweep for a crossing.

Every previous sweep varied *duration* because every previous obstacle was *height*. A gap is a
*distance* obstacle, so the varying parameters here are takeoff speed (via B and run-up), trigger
position, and A-hold.

The traces cannot supply the gap location -- they merge the main level with the coin room, and the
expert route goes under this stretch entirely -- so the emulator locates it: walk right with no jump
and find where the floor stops.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import session_when_free
from tasdata.bc.overnight_lib import calibrate, load_policy
from tasdata.bc.tokens import LIVE_MASK
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from tasdata.ram import read_smb
from tasdata.replay import _resize_gray

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/gap_sweep.json"
CKPT = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
RIGHT, LEFT, B, A = (NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["Left"],
                     NES_BUTTON_BITS["B"], NES_BUTTON_BITS["A"])
FLOOR, HANDOVER = 176, 1150
DYING = (0x06, 0x0B)

def prefix_to(session, policy, cfg, thr, start, target, seed, cap=1500):
    """Deterministic policy actions until x >= target. Returns the action list or None."""
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8); win[:] = _resize_gray(obs.rgb, (84, 84))
    seq = []
    for _ in range(cap):
        with torch.no_grad():
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0 / (1.0 + np.exp(-lg))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]: byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        seq.append(byte)
        obs = session.step(byte)
        win = np.roll(win, -1, 0); win[-1] = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in DYING: return None
        if st.x_position >= target: return seq
    return None

def replay(session, start, seq):
    obs = session.reset(start.frame)
    for b in seq: obs = session.step(b)
    return obs

def walk_probe(session, start, seq, frames=260):
    """Walk right, no jump. Where does the floor stop, and does the classifier see a pit?"""
    obs = replay(session, start, seq)
    trail, death = [], None
    for _ in range(frames):
        obs = session.step(RIGHT)
        st = read_smb(obs.ram, obs.framecount)
        trail.append((int(st.x_position), int(st.y_position)))
        if st.player_state in DYING:
            death = {"x": int(st.x_position), "y": int(st.y_position),
                     "classifier_says_pit": bool(st.y_position > 200)}
            break
    floor_xs = [x for x, y in trail if y == FLOOR]
    return {"near_lip": max(floor_xs) if floor_xs else None, "death": death,
            "max_y": max(y for _, y in trail), "trail_tail": trail[-14:]}

def cross(session, start, seq, *, hold, trigger, with_b, standstill=False, frames=300):
    obs = replay(session, start, seq)
    base = RIGHT | (B if with_b else 0)
    if standstill:                       # decelerate to a true stop first
        for _ in range(120):
            obs = session.step(0)
            if int(obs.ram[0x0057]) == 0: break
    st = read_smb(obs.ram, obs.framecount)
    x0, spd0 = st.x_position, int(obs.ram[0x0057])
    left, jumped, maxx, died, landed = hold, standstill, x0, False, None
    for _ in range(frames):
        st = read_smb(obs.ram, obs.framecount)
        byte = base
        if not jumped and st.x_position >= trigger:
            jumped = True; spd0 = int(obs.ram[0x0057])
        if jumped and left > 0:
            byte |= A; left -= 1
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if landed is None and jumped and st.y_position == FLOOR and st.x_position > x0 + 8:
            landed = int(st.x_position)
        if st.player_state in DYING: died = True; break
    return {"hold": hold, "trigger": trigger, "with_b": with_b, "standstill": standstill,
            "takeoff_speed_byte": spd0, "max_x": int(maxx), "landed_x": landed,
            "died": died}

def main():
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    out = {"checkpoint": CKPT.name, "handover_target": HANDOVER}
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        seq = None
        for sd in range(60):
            seq = prefix_to(s, policy, cfg, thr, start, HANDOVER, sd)
            if seq: out["prefix_seed"], out["prefix_frames"] = sd, len(seq); break
        if not seq:
            print(f"no seed reached x>={HANDOVER} in 60 tries"); return
        st = read_smb(replay(s, start, seq).ram, 0)
        print(f"handover at x={st.x_position} y={st.y_position} (seed {out['prefix_seed']}, "
              f"{len(seq)} frames)")

        probe = walk_probe(s, start, seq); out["walk_probe"] = probe
        print(f"\nwalk-right probe: near lip x={probe['near_lip']}  max y {probe['max_y']}")
        print(f"  death: {probe['death']}")
        print(f"  last frames (x,y): {probe['trail_tail']}")

        lip = probe["near_lip"] or HANDOVER
        rows = []
        for with_b in (True, False):
            for trig in range(lip - 40, lip + 12, 4):
                for hold in (4, 8, 12, 16, 20, 26, 32):
                    rows.append(cross(s, start, seq, hold=hold, trigger=trig, with_b=with_b))
        stand = [cross(s, start, seq, hold=h, trigger=0, with_b=True, standstill=True)
                 for h in (8, 12, 16, 20, 26, 32)]
        rows += stand
    finally:
        s.close()

    assert len({r["max_x"] for r in rows}) > 1, "degenerate harness: identical max_x everywhere"
    far = (probe["near_lip"] or 0) + 48
    ok = [r for r in rows if r["max_x"] > far and not r["died"]]
    st_ok = [r for r in stand if r["max_x"] > far and not r["died"]]
    out["rows"], out["far_threshold"] = rows, far
    out["crossed"] = len(ok)
    out["min_crossing"] = (min(((r["hold"], r["trigger"], r["takeoff_speed_byte"]) for r in ok),
                               key=lambda t: t[0]) if ok else None)
    out["standstill_crossed"] = bool(st_ok)
    out["verdict"] = (
        f"CROSSABLE: {len(ok)} of {len(rows)} configurations clear past x={far}. Minimum "
        f"(hold, trigger, speed byte) = {out['min_crossing']}. From a standstill: "
        f"{'yes' if st_ok else 'NO'}."
        if ok else
        f"NOT CROSSABLE by any tested configuration ({len(rows)} tried, threshold x>{far}). "
        f"The gaps are not a one-jump problem.")
    print(f"\nsweep: {len(rows)} configs, {len({r['max_x'] for r in rows})} distinct max_x")
    print(f"BINARY: {out['verdict']}")
    OUT.write_text(json.dumps(out, indent=2, default=str)); print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
