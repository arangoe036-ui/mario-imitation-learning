"""§2: does anything ever leave the 1-1 surface? If so, `reach_walls.json`'s bins are partly mislabelled.

The owner watched a recording and said the wide encoder "goes a lot more into the hidden level." If episodes
are entering a pipe into a bonus area, then **x is a different coordinate system in there** -- `ram.py` says
`x_position` is "absolute, in pixels from the start of the AREA" -- and stops binned at "pipe 3's face" or
"pipe 4's face" in `data/reach_walls.json` may be **pipe entries rather than stalls.**

This cannot be answered from the existing artifacts: `reach_curve.partial.json` stored per-episode summaries
only, and `EpisodeTrace` records `(x, y_absolute, speed, buttons, player_state, grounded)` -- **no world,
stage or area.** So the reach-curve episodes are re-run with those three bytes logged. They reproduce exactly:
`rollout_from` seeds its RNG with `frame_index * 100 + rep`, and the prefix replay is deterministic.

Three questions, then a decision rule applied here in code rather than left to prose:

1. how many episodes ever leave area 1, and from where;
2. what x means after entry, for those that do;
3. the wall histogram recomputed with pipe-entry episodes separated out.

**DECISION RULE (§2 of the directive):** zero pipe entries → the wall map stands and §5 targets pipe 3's face.
Non-zero → §5 re-points at the largest surviving genuine stall, and the mislabelled bins are named.

`player_state 0x07` is "entering a pipe/area" per `ram.py`; `area` is `0x0760`, 1-based after `read_smb`.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.argmax_startstates import CAP_NON_A, restore_state  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/hidden_area_check.json"
PARTIAL = ROOT / "data/hidden_area_check.partial.json"
STARTLIB = ROOT / "data/startlib_policy.json"
P1_TRACES = ROOT / "data/traces/p1_200.json"

ARMS = ["P_84_cnn32_seed1", "P_84_cnn32_seed4"]
TEMP, REPEATS = 0.7, 5
CAP_FRAMES, STALL = 3000, 300
CHUNK = 30
ARM_BUDGET_S = 20 * 60          # per-arm wall clock; on expiry log, skip, continue
ENTER_PIPE_STATE = 0x07


def rollout_logged(session, policy, cfg, obs, byte_of, lut, *, temp, seed) -> dict:
    """Same generation as `reach_curve`, but logging world/stage/area and pipe-entry states."""
    s = cfg.frame_size
    rng = np.random.default_rng(seed)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    win[:] = _resize_gray(obs.rgb, (s, s))
    held, remaining = None, 0
    best = since = 0
    log = []
    st0 = read_smb(obs.ram, obs.framecount)
    for _ in range(CAP_FRAMES):
        if remaining <= 0:
            with torch.no_grad():
                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
            p = torch.softmax(lg / float(temp), dim=-1).numpy()
            c = int(rng.choice(len(p), p=p / p.sum()))
            b, L = int(byte_of[c]), max(1, int(lut[c]))
            if not (b & A_BIT):
                L = min(L, CAP_NON_A)
            held, remaining = b, L
        remaining -= 1
        obs = session.step(held)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (s, s))
        r = read_smb(obs.ram, obs.framecount)
        log.append((r.x_position, r.world, r.stage, r.area, r.player_state))
        if r.player_state in (0x06, 0x0B):
            break
        if r.x_position > best:
            best, since = r.x_position, 0
        else:
            since += 1
            if since > STALL:
                break
    areas = sorted({l[3] for l in log})
    ws = sorted({(l[1], l[2]) for l in log})
    on_surface = [l for l in log if l[1] == 1 and l[2] == 1 and l[3] == 1]
    off = [l for l in log if not (l[1] == 1 and l[2] == 1 and l[3] == 1)]
    ent = [i for i, l in enumerate(log) if l[4] == ENTER_PIPE_STATE]
    return {
        "start_area": st0.area, "start_ws": [st0.world, st0.stage],
        "n_frames": len(log),
        "areas_seen": areas, "world_stage_seen": [list(x) for x in ws],
        "left_surface": bool(off),
        "n_frames_off_surface": len(off),
        "entered_pipe_state": bool(ent),
        "first_pipe_entry_frame": (ent[0] if ent else None),
        "x_at_first_pipe_entry": (log[ent[0]][0] if ent else None),
        # x_max ON THE SURFACE ONLY -- the number reach_walls should have used
        "max_x_surface": (max(l[0] for l in on_surface) if on_surface else None),
        # and the naive one, which mixes coordinate systems if an area was entered
        "max_x_naive": max(l[0] for l in log) if log else None,
        "max_x_off_surface": (max(l[0] for l in off) if off else None),
    }


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    states = json.loads(STARTLIB.read_text())["states"]
    p1 = {e["seed"]: e for e in json.loads(P1_TRACES.read_text())["episodes"]}
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    done = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    skipped = []
    for arm in ARMS:
        if not (ROOT / f"data/bc_scaleup/{arm}.pt").exists():
            continue
        todo = [(s, r) for s in states for r in range(REPEATS)
                if f"{arm}:{s['seed']}:{s['frame_index']}:{r}" not in done]
        if not todo:
            continue
        if not dl.can_afford(120):
            skipped.append({"arm": arm, "reason": "deadline", "n_unrun": len(todo)})
            print(f"{dl.stamp()} SKIP {arm}: deadline", flush=True)
            continue
        policy, cfg, blob = G.load_ckpt(arm)
        print(f"{dl.stamp()} {arm}: {len(todo)} episodes", flush=True)
        sess = None
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 30), f"arm {arm}"):
                sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
                warm_session(sess, start.frame)
                for i, (st, rep) in enumerate(todo):
                    ep = p1.get(st["seed"])
                    if ep is None:
                        continue
                    obs = restore_state(sess, st, [f[3] for f in ep["frames"]], start.frame)
                    rec = rollout_logged(sess, policy, cfg, obs, byte_of, lut,
                                         temp=TEMP, seed=st["frame_index"] * 100 + rep)
                    rec.update({"arm": arm, "start_x": st["x"], "rep": rep})
                    done[f"{arm}:{st['seed']}:{st['frame_index']}:{rep}"] = rec
                    if (i + 1) % CHUNK == 0 or i == len(todo) - 1:
                        PARTIAL.write_text(json.dumps(done, separators=(",", ":")))
                        print(f"  {dl.stamp()} {i + 1}/{len(todo)}", flush=True)
        except TimedOut as e:
            skipped.append({"arm": arm, "reason": str(e)})
            print(f"{dl.stamp()} TIMEOUT on {arm}: {e}", flush=True)
        finally:
            if sess is not None:
                try:
                    sess.close()
                except BaseException:
                    pass
            PARTIAL.write_text(json.dumps(done, separators=(",", ":")))

    rows = list(done.values())
    out = {"arms": ARMS, "temperature": TEMP, "repeats": REPEATS,
           "measurement_basis": "conditional_on_arrival",
           "n_episodes": len(rows), "skipped": skipped,
           "method": ("reach-curve episodes re-run with world/stage/area logged; EpisodeTrace does not "
                      "record those bytes so the existing artifacts could not answer this"),
           "per_arm": {}}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        left = [r for r in rs if r["left_surface"]]
        ent = [r for r in rs if r["entered_pipe_state"]]
        # wall histogram, surface-only x, split by whether the episode ever left the surface
        def hist(sub):
            return dict(sorted(collections.Counter(
                (r["max_x_surface"] // 32) * 32 for r in sub
                if r["max_x_surface"] is not None).items()))
        out["per_arm"][arm] = {
            "n": len(rs),
            "n_left_surface": len(left), "frac_left_surface": len(left) / len(rs),
            "n_entered_pipe_state": len(ent),
            "areas_seen_union": sorted({a for r in rs for a in r["areas_seen"]}),
            "world_stage_union": sorted({tuple(x) for r in rs for x in r["world_stage_seen"]}),
            "start_x_of_leavers": sorted({r["start_x"] for r in left}),
            "x_at_entry": sorted(r["x_at_first_pipe_entry"] for r in ent
                                 if r["x_at_first_pipe_entry"] is not None),
            "max_x_off_surface": sorted({r["max_x_off_surface"] for r in left
                                         if r["max_x_off_surface"] is not None}),
            "wall_histogram_surface_x_stayers": hist([r for r in rs if not r["left_surface"]]),
            "wall_histogram_surface_x_leavers": hist(left),
            "naive_vs_surface_disagreements": sum(
                1 for r in rs if r["max_x_surface"] is not None
                and r["max_x_naive"] != r["max_x_surface"])}

    total_left = sum(v["n_left_surface"] for v in out["per_arm"].values())
    total_ent = sum(v["n_entered_pipe_state"] for v in out["per_arm"].values())
    out["decision"] = {
        "rule": ("zero pipe entries -> the wall map stands and §5 targets pipe 3's face; non-zero -> "
                 "§5 re-points at the largest surviving genuine stall"),
        "n_episodes_leaving_surface": total_left,
        "n_episodes_in_pipe_entry_state": total_ent,
        "branch": ("wall_map_stands" if total_left == 0 else "wall_map_partly_mislabelled")}
    if total_left == 0:
        out["verdict"] = (
            f"**NOTHING LEAVES THE 1-1 SURFACE.** Across {len(rows)} episodes, every frame is world 1, "
            f"stage 1, area 1; {total_ent} episodes ever entered the pipe-entry player state (0x07). "
            f"**`reach_walls.json`'s bins are genuine stalls, x is one coordinate system throughout, and "
            f"the wall map stands.** The owner's 'hidden level' observation therefore describes depth on "
            f"the surface, not a bonus area. **§5 targets pipe 3's face as specified.**")
    else:
        out["verdict"] = (
            f"**EPISODES DO LEAVE THE 1-1 SURFACE: {total_left} of {len(rows)}**, and {total_ent} entered "
            f"the pipe-entry state. **x is a different coordinate system inside an area, so some "
            f"`reach_walls.json` bins are pipe entries rather than stalls.** The surface-only wall "
            f"histogram is reported separately per arm; §5 re-points at the largest surviving genuine "
            f"stall.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    for arm, v in out["per_arm"].items():
        print(f"\n{arm}: n={v['n']} left_surface={v['n_left_surface']} "
              f"pipe_entry_state={v['n_entered_pipe_state']} areas={v['areas_seen_union']} "
              f"world_stage={v['world_stage_union']}")
        print(f"   naive-vs-surface max_x disagreements: {v['naive_vs_surface_disagreements']}")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
