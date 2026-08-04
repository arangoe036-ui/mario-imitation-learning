"""P1: one 200-episode instrumented run + a chained walk probe for the gap map.

Everything is written to disk per-frame; the seven analyses are reads over that file.
FLOOR is 432 in absolute y (page 1 x 256 + 176) -- the constant that zeroed two checks yesterday.
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
from tasdata.bc.trace_log import EpisodeTrace, write_traces
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from tasdata.ram import read_smb, y_absolute
from tasdata.replay import _resize_gray

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/traces/p1_200.json"
PROBE = ROOT / "data/traces/p1_walkprobe.json"
CKPT = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
A, B, RIGHT = NES_BUTTON_BITS["A"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["Right"]
N, CAP, STALL = 200, 3000, 300

def episode(session, policy, cfg, thr, start, seed):
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8); win[:] = _resize_gray(obs.rgb, (84, 84))
    best = since = 0
    for _ in range(CAP):
        with torch.no_grad():
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0/(1.0+np.exp(-lg))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]: byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        obs = session.step(byte)
        win = np.roll(win, -1, 0); win[-1] = _resize_gray(obs.rgb, (84, 84))
        t.record(obs, byte)
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in (0x06, 0x0B): t.record_death(obs); break
        if st.x_position > best: best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL: t.ended = "stuck"; break
    return t

def walk_leg(session, start, prefix, jump_period, jump_hold, frames=900):
    """Scripted traversal leg: run right, jump on a fixed period, record every frame."""
    obs = session.reset(start.frame)
    for b in prefix: obs = session.step(b)
    rows = []
    for i in range(frames):
        byte = RIGHT | B
        if (i % jump_period) < jump_hold: byte |= A
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        rows.append((int(st.x_position), y_absolute(obs.ram), int(st.player_state)))
        if st.player_state in (0x06, 0x0B): break
    return rows

def main():
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        print("walk probe: scripted traversals with swept jump phase", flush=True)
        legs = {}
        for per, hold in ((48, 12), (56, 14), (40, 10), (64, 16), (72, 18)):
            legs[f"p{per}h{hold}"] = walk_leg(s, start, [], per, hold)
            xs = [r[0] for r in legs[f"p{per}h{hold}"]]
            print(f"  period {per} hold {hold}: max x {max(xs)}, {len(xs)} frames", flush=True)
        print(f"\n{N} instrumented episodes", flush=True)
        traces = []
        for i in range(N):
            traces.append(episode(s, policy, cfg, thr, start, i))
            if (i + 1) % 50 == 0: print(f"  {i+1}/{N}", flush=True)
    finally:
        s.close()
    PROBE.write_text(json.dumps({"floor_absolute_y": 432, "legs": legs}, separators=(",", ":")))
    p = write_traces(OUT, traces, checkpoint=CKPT.name, cap=CAP, stall=STALL)
    print(f"\nwrote {p} ({p.stat().st_size/1e6:.1f} MB) and {PROBE.name}")

if __name__ == "__main__":
    main()
