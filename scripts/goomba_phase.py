"""P2(a): does ANY scripted jump phase clear the Goomba at x~296-312?

Two scripted configurations failing establishes nothing -- a fixed 20-on/28-off jump period is
almost certainly out of phase with the enemy. So sweep the phase: Right+B held throughout, plus a
single A press of a given length triggered at a given x. Single life.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from tasdata.bc.session import FceuxSession
from tasdata.buttons import NES_BUTTON_BITS
from tasdata.ram import read_smb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/goomba_phase.json"
RIGHT, B, A = NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["A"]
TRIGGERS = list(range(240, 304, 4))
HOLDS = [8, 10, 12, 14, 16]
GOOMBA_PASSED = 340          # comfortably past the death cluster at 288-312
ENEMY_TYPE, ENEMY_XPAGE, ENEMY_XLO = 0x0016, 0x006E, 0x0087

def run(session, start, *, trigger, hold, frames=600):
    obs = session.reset(start.frame)
    left, jumped = hold, False
    maxx = read_smb(obs.ram, obs.framecount).x_position
    died_at = None
    for _ in range(frames):
        st = read_smb(obs.ram, obs.framecount)
        byte = RIGHT | B
        if not jumped and st.x_position >= trigger:
            jumped = True
        if jumped and left > 0:
            byte |= A; left -= 1
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if st.player_state in (0x06, 0x0B):
            died_at = int(st.x_position); break
    return {"trigger": trigger, "hold": hold, "max_x": int(maxx),
            "passed_goomba": bool(maxx > GOOMBA_PASSED), "died_at": died_at}

def main():
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    rows = []
    with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as s:
        for hold in HOLDS:
            line = []
            for t in TRIGGERS:
                r = run(s, start, trigger=t, hold=hold); rows.append(r)
                line.append("P" if r["passed_goomba"] else ".")
            print(f"  hold {hold:2d}: " + "".join(line) +
                  f"   triggers {TRIGGERS[0]}..{TRIGGERS[-1]} step 4")
    ok = [r for r in rows if r["passed_goomba"]]
    assert len({r["max_x"] for r in rows}) > 1, "harness degenerate: identical max_x everywhere"
    verdict = (f"YES -- {len(ok)}/{len(rows)} scripted configurations clear the Goomba. "
               f"Working (trigger, hold) pairs include "
               f"{[(r['trigger'], r['hold']) for r in ok[:6]]}; best max_x "
               f"{max(r['max_x'] for r in ok)}"
               if ok else
               f"NO -- 0/{len(rows)} scripted configurations clear it at any phase or hold. "
               f"best max_x {max(r['max_x'] for r in rows)}")
    print("\n" + "="*78); print(f"BINARY: {verdict}")
    OUT.write_text(json.dumps({"triggers": TRIGGERS, "holds": HOLDS,
                               "n_configs": len(rows), "n_passed": len(ok),
                               "verdict": verdict, "rows": rows}, indent=2))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
