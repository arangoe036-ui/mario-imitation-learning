"""P2(b) + P5: what is at x≈720, read off the expert RAM traces. No emulator, no ROM parsing.

The expert's y as a function of x *is* the terrain profile, for the parts of the terrain the
expert stands on. 25 synced runs traverse 1-1, and `ram.py` recorded x and y per frame for all
of them, so this is one pass over data already on disk.

Two things are computed:

* **The ground profile.** For each x, the *lowest* point (largest y) at which any expert was
  stationary in y — i.e. standing on something. Airborne frames are excluded by requiring y to
  be unchanged from the previous frame. Height above the floor line (y=176) in 16 px tiles gives
  the obstacle height.
* **The expert A-hold statistics near each obstacle**, re-derived here independently, because the
  previous figures came from the jump segmentation in `pipe2_ceiling.py` that was declared broken
  and must not be cited.

Caveat stated up front: this reveals only surfaces the expert actually stood on. If the expert
flies over an obstacle without touching it, that obstacle is invisible to this method, and the
report says so rather than inferring a height.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/terrain_profile.json"
FLOOR_Y = 176
A_BIT = NES_BUTTON_BITS["A"]
XLO, XHI = 200, 900
OBSTACLES = {"goomba_zone": (280, 330), "pipe1": (420, 470),
             "pipe2": (575, 640), "wall720": (690, 760)}


def main() -> None:
    split = json.loads((ROOT / "data/split.json").read_text())["splits"]
    names = [n for b in ("train", "val", "test") for n in split[b]]

    stand = defaultdict(list)       # x -> y values while y is unchanged (standing)
    airborne_y = defaultdict(list)  # x -> y while y is changing
    holds_by_zone = defaultdict(list)
    n_runs = 0
    for name in names:
        run = load_run_dir(ROOT / "data/runs" / name)
        if run.manifest.get("measured_route") not in ("warpless", "warps"):
            continue
        tr = np.asarray(run.trace)
        w, s = column(tr, "world"), column(tr, "stage")
        x, y = column(tr, "x_position"), column(tr, "y_position")
        st, pg = column(tr, "player_state"), column(tr, "pregame")
        a = (np.asarray(run.actions, dtype=np.uint8) & A_BIT) > 0
        n = min(len(x), len(a))
        m = (w[:n] == 1) & (s[:n] == 1) & (pg[:n] == 1) & (st[:n] == 8)
        if not m.any():
            continue
        n_runs += 1
        idx = np.flatnonzero(m)
        for i in idx:
            if i == 0 or x[i] < XLO or x[i] > XHI:
                continue
            (stand if y[i] == y[i - 1] else airborne_y)[int(x[i]) // 8 * 8].append(int(y[i]))
        # A-holds beginning inside each obstacle zone, re-derived from scratch
        for zone, (lo, hi) in OBSTACLES.items():
            for i in idx:
                if i == 0 or not (lo <= x[i] <= hi) or not a[i] or a[i - 1]:
                    continue
                j = i
                while j < n and a[j]:
                    j += 1
                holds_by_zone[zone].append(int(j - i))

    print(f"{n_runs} runs contributed 1-1 frames\n")
    print("GROUND PROFILE, x 400-800 (8 px bins) -- surfaces the expert stood on")
    print(f"{'x':>5s} {'n_stand':>8s} {'lowest y':>9s} {'highest y':>10s} "
          f"{'tiles above floor':>18s}")
    profile = {}
    for xb in range(400, 808, 8):
        ys = stand.get(xb, [])
        if not ys:
            profile[xb] = None
            continue
        lo_y, hi_y = max(ys), min(ys)          # y grows downward
        tiles = round((FLOOR_Y - lo_y) / 16, 2)
        profile[xb] = {"n": len(ys), "lowest_y": lo_y, "highest_y": hi_y, "tiles": tiles}
        flag = ""
        if 690 <= xb <= 760:
            flag = "   <-- the wall at 720"
        elif 575 <= xb <= 640:
            flag = "   <-- pipe 2"
        elif 420 <= xb <= 470:
            flag = "   <-- pipe 1"
        print(f"{xb:5d} {len(ys):8d} {lo_y:9d} {hi_y:10d} {tiles:18.2f}{flag}")

    print("\nOBSTACLE SUMMARY")
    summary = {}
    for zone, (lo, hi) in OBSTACLES.items():
        bins = [profile[b] for b in range(lo // 8 * 8, hi + 8, 8)
                if profile.get(b)]
        if bins:
            tiles = max(b["tiles"] for b in bins)
            n = sum(b["n"] for b in bins)
        else:
            tiles, n = None, 0
        air = sum(len(airborne_y.get(b, [])) for b in range(lo // 8 * 8, hi + 8, 8))
        h = holds_by_zone.get(zone, [])
        summary[zone] = {
            "x_range": [lo, hi], "max_tiles_above_floor": tiles,
            "standing_frames": n, "airborne_frames": air,
            "airborne_fraction": (air / (air + n) if (air + n) else None),
            "expert_A_holds": ({"n": len(h), "median": float(np.median(h)),
                                "p90": float(np.percentile(h, 90)),
                                "max": int(max(h))} if h else {"n": 0}),
        }
        d = summary[zone]
        print(f"  {zone:14s} x{lo}-{hi}: height "
              f"{'unknown (expert never stood here)' if tiles is None else f'{tiles} tiles'}"
              f"  standing {n:6d} airborne {air:6d} "
              f"({(d['airborne_fraction'] or 0) * 100:.0f}% airborne)  "
              f"A-holds n={d['expert_A_holds']['n']} "
              f"median {d['expert_A_holds'].get('median')} "
              f"p90 {d['expert_A_holds'].get('p90')} max {d['expert_A_holds'].get('max')}")

    w = summary["wall720"]
    verdict = (
        f"x=690-760 is a surface {w['max_tiles_above_floor']} tiles above the floor line; "
        f"the expert is airborne there {(w['airborne_fraction'] or 0) * 100:.0f}% of the time"
        if w["max_tiles_above_floor"] is not None else
        "the expert never stood anywhere in x=690-760, so this method cannot give its height -- "
        "it is either flown over entirely or is not a standable surface")
    print(f"\nVERDICT on x~720: {verdict}")
    OUT.write_text(json.dumps({"n_runs": n_runs, "floor_y": FLOOR_Y,
                               "profile_8px": profile, "obstacles": summary,
                               "verdict_wall720": verdict}, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
