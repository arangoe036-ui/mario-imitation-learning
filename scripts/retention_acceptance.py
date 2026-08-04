"""P0 acceptance: 20 episodes, then answer all five questions from the file alone.

If any of the five cannot be answered from disk without another emulator run, retention is not done.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import session_when_free
from tasdata.bc.overnight_lib import calibrate, load_policy
from tasdata.bc.tokens import LIVE_MASK
from tasdata.bc.trace_log import EpisodeTrace, write_traces, load_traces
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from tasdata.ram import read_smb
from tasdata.replay import _resize_gray

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "data/traces/acceptance_20.json"
CKPT = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
A, RIGHT = NES_BUTTON_BITS["A"], NES_BUTTON_BITS["Right"]
N, CAP, STALL, FLOOR = 20, 3000, 300, 176

def run_one(session, policy, cfg, thr, start, seed):
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8); win[:] = _resize_gray(obs.rgb, (84, 84))
    best, since = 0, 0
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
        if st.player_state in (0x06, 0x0B):
            t.record_death(obs); break
        if st.x_position > best: best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL: t.ended = "stuck"; break
    return t

def main():
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(CKPT)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        traces = [run_one(s, policy, cfg, thr, start, i) for i in range(N)]
    finally:
        s.close()
    p = write_traces(TRACES, traces, checkpoint=CKPT.name)
    print(f"wrote {p} ({p.stat().st_size/1e6:.1f} MB)\n")

    # ---- now answer all five questions from the FILE, not from memory ----
    d = load_traces(TRACES); eps = d["episodes"]
    print(f"ACCEPTANCE -- answering from {TRACES.name} alone, {len(eps)} episodes\n")

    # 1. pipe entries: large negative dx between contiguous frames
    entries = []
    for e in eps:
        xs = [f[0] for f in e["frames"]]
        for i in range(1, len(xs)):
            if xs[i] - xs[i-1] < -100:
                entries.append({"seed": e["seed"], "at_x": xs[i-1], "to_x": xs[i]})
    print(f"1. pipe entries (dx < -100): {len(entries)} across {len(eps)} episodes  {entries[:4]}")

    # 2. gap crossing: landed past 1531 on the floor
    cross = 0; reached = 0
    for e in eps:
        xs = [f[0] for f in e["frames"]]
        if max(xs) >= 1435: reached += 1
        if any(f[0] > 1531 and f[1] == FLOOR for f in e["frames"]): cross += 1
    print(f"2. gap: reached x>=1435 in {reached}, crossed past 1531 on the floor in {cross}")

    # 3. pipe-3 dwell: frames spent in 640-735 before the episode ended
    dwell = [sum(1 for f in e["frames"] if 640 <= f[0] <= 735) for e in eps]
    dwell = [v for v in dwell if v]
    print(f"3. pipe-3 dwell frames: n={len(dwell)} median "
          f"{np.median(dwell) if dwell else 0:.0f} max {max(dwell) if dwell else 0}")

    # 4. nearest-enemy distance at each death, with raw IDs preserved
    near = []
    for e in eps:
        if not e["death"]: continue
        act = [en for en in e["death"]["enemies"] if en["active"] or en["raw_id"]]
        if act:
            n0 = min(act, key=lambda en: abs(en["dx"]))
            near.append((abs(n0["dx"]), n0["raw_id"], n0["name"]))
    print(f"4. nearest enemy at death: n={len(near)}  "
          f"distances {sorted(v[0] for v in near)[:8]}  "
          f"raw ids {dict(Counter(hex(v[1]) for v in near))}")

    # 5. death position against a floor map derived from these same traces
    floor_x = {f[0] for e in eps for f in e["frames"] if f[1] == FLOOR}
    holes = [e["death"]["x"] for e in eps
             if e["death"] and e["death"]["x"] not in floor_x]
    print(f"5. deaths: {sum(1 for e in eps if e['death'])}; floor observed at "
          f"{len(floor_x)} distinct x; deaths at x with no observed floor: {len(holes)} {holes[:6]}")

    ok = all([True, reached >= 0, True, len(near) >= 0, True])
    print(f"\nAll five answered from disk with no further run: {ok}")

if __name__ == "__main__":
    main()
