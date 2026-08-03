"""Build (or verify) the savestate start-point index, hashing RAM *and* pixels.

``build`` captures every state once and records both hashes in the persisted index.
``check`` rebuilds from scratch in a fresh process and asserts both match. RAM equality
alone would not catch PPU drift, and the policy consumes pixels, not RAM.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.statelib import (  # noqa: E402
    build_start_points,
    frame_hash,
    load_index,
    ram_hash,
    save_index,
)
from tasdata.dataset import load_run_dir  # noqa: E402
from tasdata.rom import load_rom  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "smb.nes"
MOVIE = ROOT / "data/movies/happylee_mars608-smb-warpless.fm2"
RUN = ROOT / "data/runs/pub-3728"
INDEX = ROOT / "data/state_index.json"


def capture(points):
    """Load each state on a live session and hash what comes back."""
    frames = sorted({p.frame for p in points})
    with FceuxSession(ROM, MOVIE, frames) as session:
        print(f"  session: {session.n_states} states in {session.build_seconds:.1f}s")
        out = {}
        for frame in frames:
            obs = session.reset(frame)
            out[frame] = (ram_hash(obs.ram), frame_hash(obs.rgb))
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    run = load_run_dir(RUN)
    points, stats = build_start_points(run, n_trajectory=500, seed=0)
    n_starts = sum(1 for p in points if p.kind == "level_start")
    print(f"[{mode}] {len(points)} points ({n_starts} level starts, "
          f"{len(points) - n_starts} trajectory)")
    print(stats.text())

    grounded_starts = sum(1 for p in points if p.kind == "level_start" and p.grounded)
    print(f"  level starts grounded: {grounded_starts}/{n_starts} "
          f"(airborne starts are canonical but excluded from grounded-only checks)")
    for p in points[:8]:
        if p.kind == "level_start":
            print(f"    {p.label} frame={p.frame} x={p.x} y={p.y} grounded={p.grounded}")

    hashes = capture(points)

    if mode == "build":
        for p in points:
            p.ram_hash, p.frame_hash = hashes[p.frame]
        save_index(INDEX, MOVIE, load_rom(ROM).md5_prgchr, points, stats)
        distinct_ram = len({h[0] for h in hashes.values()})
        distinct_frm = len({h[1] for h in hashes.values()})
        print(f"[build] wrote {INDEX}")
        print(f"[build] distinct RAM hashes {distinct_ram}/{len(hashes)}, "
              f"distinct FRAME hashes {distinct_frm}/{len(hashes)}")
    else:
        _, stored = load_index(INDEX)
        ram_ok = frm_ok = 0
        missing = 0
        for p in stored:
            got = hashes.get(p.frame)
            if got is None:
                missing += 1
                continue
            ram_ok += got[0] == p.ram_hash
            frm_ok += got[1] == p.frame_hash
        n = len(stored) - missing
        print(f"[check] RAM hashes identical  : {ram_ok}/{n}")
        print(f"[check] FRAME hashes identical: {frm_ok}/{n}")
        assert missing == 0, f"{missing} indexed frames were not captured on rebuild"
        assert ram_ok == n, "RAM drift between builds"
        assert frm_ok == n, "PIXEL drift between builds (RAM alone would have missed this)"
        print("[check] both assertions passed")


if __name__ == "__main__":
    main()
