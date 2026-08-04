"""The gap's geometry from the two surface-route demonstrations, using on_ground().

`pub-3648` and `pub-4313` are the only runs in the corpus that traverse x=916-2616 of 1-1 on the
surface. Where they are grounded there is ground; where they are airborne between two grounded
stretches, that is the pit.

Measured with SMB's own on-ground byte (0x001D), not by comparing y to a constant -- every previous
geometry for this obstacle came from a value comparison or from a falling policy, and there have been
three of them.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasdata.fceux_backend import FceuxReplayer
from tasdata.movie import parse_movie
from tasdata.ram import (ADDR_ON_GROUND, ADDR_PREGAME, ADDR_STAGE, ADDR_WORLD,
                         ADDR_X_IN_PAGE, ADDR_X_PAGE, ADDR_Y_PIXEL, ADDR_Y_VIEWPORT)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/pipe4_geometry.json"
RUNS = {"pub-3648": None, "pub-4313": None}
COLS = ("frame", "world", "stage", "x", "on_ground", "y_abs", "pregame")

def probe(ram, i, row):
    row[0] = i
    row[1] = ram[ADDR_WORLD] + 1
    row[2] = ram[ADDR_STAGE] + 1
    row[3] = int(ram[ADDR_X_PAGE]) * 256 + int(ram[ADDR_X_IN_PAGE])
    row[4] = 1 if int(ram[ADDR_ON_GROUND]) == 0 else 0     # 1 = grounded
    row[5] = int(ram[ADDR_Y_VIEWPORT]) * 256 + int(ram[ADDR_Y_PIXEL])
    row[6] = ram[ADDR_PREGAME]

def main():
    out = {"note": "grounded measured from SMB 0x001D; 1 = on a surface", "runs": {}}
    for name in RUNS:
        mpath = json.loads((ROOT / "data/runs" / name / "manifest.json").read_text())["movie"]
        mpath = ROOT / str(mpath).replace(str(ROOT) + "/", "")
        movie = parse_movie(mpath)
        rep = FceuxReplayer(ROOT / "smb.nes")
        res = rep.replay(movie, probe=probe, trace_columns=COLS)
        tr = np.asarray(res.trace)
        m = (tr[:, 1] == 1) & (tr[:, 2] == 1) & (tr[:, 6] == 1)
        x, g = tr[m, 3], tr[m, 4]
        sel = (x >= 860) & (x <= 1010)
        xs, gs = x[sel], g[sel]
        print(f"\n{name}: {sel.sum()} frames in x=860-1010 of 1-1")
        # POSITIVE grounding only: where is the run standing, and at what height?
        ya = tr[m, 5][sel]
        FLOORY = 432
        gnd = [(int(a), int(b)) for a, b, c in zip(xs, ya, gs) if c == 1]
        elevated = [(a, b) for a, b in gnd if b < FLOORY - 8]
        print(f"  grounded frames: {len(gnd)}; of those ABOVE floor level (y<{FLOORY-8}): "
              f"{len(elevated)}")
        if elevated:
            hs = sorted({b for _, b in elevated})
            for h in hs:
                at = sorted({a for a, b in elevated if b == h})
                print(f"    y={h} ({(FLOORY-h)/16:.2f} tiles above floor) at x={at[0]}-{at[-1]} "
                      f"({len(at)} distinct x)")
        else:
            print("    none -- this run never stands above floor level here")
        floor_x = sorted({a for a, b in gnd if b >= FLOORY - 8})
        print(f"  grounded AT floor level at x: {floor_x[:6]}...{floor_x[-6:] if len(floor_x)>6 else ''}")
        out["runs"][name] = {"n_frames": int(sel.sum()), "n_grounded": len(gnd),
                             "elevated": [[a, b, round((FLOORY-b)/16, 2)] for a, b in elevated],
                             "floor_grounded_x": floor_x}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
