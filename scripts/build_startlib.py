"""§1: a start-state library from the policy's own retained traces.

The existing start library is unusable under the new objective. All 16 of 1-1's trajectory points sit at
**x = 2,616-2,636** -- past pipe 4, past the gap, past every obstacle the script-net credit pays for. So
self-imitation had zero start states from which practising those obstacles was even possible. Verified,
not assumed: the x values are printed at the top of the run.

The fix needs no ROM, no expert grounding and no new geometry. `data/traces/p1_200.json` holds 200
episodes of per-frame `(x, y_absolute, speed, buttons, player_state)`, and **the buttons are the datum
that matters**: replaying a recorded byte prefix reproduces the state deterministically, so any frame of
any episode is a restorable start state. `on_ground()` is checked live during the replay, because the
trace tuple never stored it.

Two things fall out of doing it this way:

* **The early-1-1 absence closes.** The expert is airborne through most of early 1-1, so a
  grounded-state filter over expert traces found nothing there. The policy is grounded there constantly.
* **Replay determinism gets measured.** Every frame's replayed x is compared against the recorded x, per
  episode. 17 of 39 pipe-4 configurations failed to reproduce two blocks ago and the cause was never
  found; if replay diverges anywhere, this prints which seeds and where.

Selection is stratified by x so early 1-1 and the approaches to pipes 3 and 4 are all represented, and
spread across distinct seeds inside each bin so a single unusual episode cannot dominate a stratum.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from tasdata.ram import on_ground, read_smb, y_absolute  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "data/traces/p1_200.json"
OUT = ROOT / "data/startlib_policy.json"

#: x strata. Boundaries are the measured obstacle positions (pipe1 470, pipe2 630 with its face at
#: 592, pipe3 735 with its face at 720, pipe4 975 with its face at ~896) plus even coverage of early
#: 1-1, which no previous library reached at all.
BINS = [(0, 120), (120, 240), (240, 360), (360, 470), (470, 592), (592, 700),
        (700, 760), (760, 880), (880, 975), (975, 1216), (1216, 1400), (1400, 4000)]
PER_BIN = 6
MIN_GAP = 24          # frames; two states from one episode must be this far apart


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    traj11 = [p for p in ctx.points if p.kind == "trajectory" and p.label == "1-1"]
    print(f"AUDIT of the existing library: {len(traj11)} 1-1 trajectory points at x "
          f"{sorted(p.x for p in traj11)}", flush=True)
    print("  -> all past pipe 4 (975) and the gap (~1380); unusable for practising obstacles\n",
          flush=True)

    blob = json.loads(TRACES.read_text())
    eps = blob["episodes"]
    print(f"{len(eps)} episodes from {TRACES.name} (checkpoint {blob.get('checkpoint')})",
          flush=True)

    cands: list[dict] = []
    mism = Counter()
    total_frames = checked = 0
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        for ei, e in enumerate(eps):
            seed = e["seed"]
            frames = e["frames"]
            bytes_ = [f[3] for f in frames]
            obs = s.reset(start.frame)
            last_kept = -10 ** 9
            for i, byte in enumerate(bytes_):
                obs = s.step(byte)
                st = read_smb(obs.ram, obs.framecount)
                total_frames += 1
                # determinism: the replayed x must equal the recorded x at the same index
                rec_x = frames[i][0]
                if st.x_position != rec_x:
                    mism[seed] += 1
                else:
                    checked += 1
                if (st.player_state == 8 and on_ground(obs.ram)
                        and i - last_kept >= MIN_GAP and i >= 4):
                    cands.append({"seed": seed, "frame_index": i + 1,  # bytes[:i+1] reproduces it
                                  "x": int(st.x_position), "y_absolute": y_absolute(obs.ram),
                                  "speed": int(obs.ram[0x0057]),
                                  "recorded_x": int(rec_x)})
                    last_kept = i
                if st.player_state in (0x06, 0x0B):
                    break
            if (ei + 1) % 25 == 0:
                print(f"  {ei + 1}/{len(eps)} episodes, {len(cands):,} grounded candidates, "
                      f"{sum(mism.values())} x-mismatches", flush=True)
    finally:
        s.close()

    print(f"\nreplay determinism: {checked:,}/{total_frames:,} frames reproduced the recorded x "
          f"({100 * checked / max(total_frames, 1):.3f}%)", flush=True)
    if mism:
        print(f"  MISMATCHES on {len(mism)} seeds: {dict(mism.most_common(10))}", flush=True)
    else:
        print("  no mismatches: replaying a recorded byte prefix is exactly deterministic",
              flush=True)

    by_bin = defaultdict(list)
    for c in cands:
        for lo, hi in BINS:
            if lo <= c["x"] < hi:
                by_bin[(lo, hi)].append(c)
                break

    rng = np.random.default_rng(0)
    chosen: list[dict] = []
    print("\nstratified selection:", flush=True)
    for lo, hi in BINS:
        pool = by_bin.get((lo, hi), [])
        # spread across distinct seeds: round-robin over seeds, then fill
        byseed = defaultdict(list)
        for c in pool:
            byseed[c["seed"]].append(c)
        seeds = sorted(byseed)
        rng.shuffle(seeds)
        picked = []
        while len(picked) < PER_BIN and seeds:
            for sd in list(seeds):
                if not byseed[sd]:
                    seeds.remove(sd)
                    continue
                picked.append(byseed[sd].pop(rng.integers(len(byseed[sd]))))
                if len(picked) >= PER_BIN:
                    break
        for c in picked:
            c["bin"] = [lo, hi]
        chosen.extend(picked)
        print(f"  x {lo:5d}-{hi:5d}: {len(pool):6,} candidates from "
              f"{len(set(c['seed'] for c in pool)):3d} seeds -> picked {len(picked)}", flush=True)

    out = {"source_traces": TRACES.name, "source_checkpoint": blob.get("checkpoint"),
           "restore": ("replay data/traces/p1_200.json episode `seed`, stepping its recorded "
                       "buttons for `frame_index` frames from the 1-1 level_start savestate; then "
                       "session.save_scratch() to make further restores O(1)"),
           "grounded_enforced": True, "player_state_required": 8,
           "min_frame_gap": MIN_GAP, "bins": BINS, "per_bin": PER_BIN,
           "n_candidates": len(cands), "n_selected": len(chosen),
           "replay_determinism": {"frames": total_frames, "reproduced": checked,
                                  "mismatched_seeds": dict(mism)},
           "existing_library_1_1_x": sorted(p.x for p in traj11),
           "states": chosen,
           "minutes": round((time.time() - t0) / 60, 1)}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    xs = [c["x"] for c in chosen]
    print(f"\nselected {len(chosen)} start states, x {min(xs)}-{max(xs)}, "
          f"{len(set(c['seed'] for c in chosen))} distinct seeds")
    print(f"wrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
