"""⚠⚠ How much of this project's reach measurement was the STALL RULE, not the policy?

`STALL = 300` — end an episode after 300 frames without a new maximum x — appears in **twelve** scripts and
governs **every rollout this project has ever run**. `CAP_FRAMES = 3000` caps the rest.

Found while answering §2: four episodes flagged as entering the pipe state were re-run with the stall rule
**disabled**, and three of them went far past the "wall":

| start x | with STALL=300 | with the rule off |
|---|---|---|
| 56 | died at 682 | died at 682 |
| 100 | ~900 | **max_x 2710** |
| 66 | ~900 | **max_x 2712** |
| **158** | ~900 | **max_x 3266 — the flagpole**, areas 1→2→3, world/stage 1-1→**1-2** |

So the "walls" at 672–704, 896 and 1504–1536 in `reach_walls.json` are at least partly **our own terminator
firing**, and `hidden_area_check`'s conclusion that nothing leaves the 1-1 surface was itself an artifact of
it — nothing leaves the surface *because episodes are killed before the transit completes*.

This is the paired experiment that says how much. Identical arms, identical start states, identical RNG seeds
— **only the terminator differs**:

* **old**: `STALL=300`, `CAP_FRAMES=3000` — reused from `hidden_area_check.partial.json`, so the control
  costs nothing and is exactly the condition every prior figure was measured under.
* **new**: `STALL=1200`, `CAP_FRAMES=9000`.

Also recorded, because it is the number that sets a defensible rule: **the longest run of frames without a new
maximum x** within each episode. If legitimate pauses routinely exceed 300 frames, 300 was simply too tight
and the right value is measurable rather than guessed.
"""
from __future__ import annotations

import collections
import json
import sys
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
OUT = ROOT / "data/stall_rule_audit.json"
PARTIAL = ROOT / "data/stall_rule_audit.partial.json"
OLD_DATA = ROOT / "data/hidden_area_check.partial.json"

ARMS = ["P_84_cnn32_seed1", "P_84_cnn32_seed4"]
TEMP = 0.7
REPS = 3                      # paired against the same reps of the old condition
NEW_STALL, NEW_CAP = 1200, 9000
OLD_STALL, OLD_CAP = 300, 3000
FLAG = 3266
CHUNK = 20
ARM_BUDGET_S = 45 * 60


def rollout(session, policy, cfg, obs, byte_of, lut, *, temp, seed, stall, cap) -> dict:
    s = cfg.frame_size
    rng = np.random.default_rng(seed)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    win[:] = _resize_gray(obs.rgb, (s, s))
    held, remaining = None, 0
    best = since = 0
    longest_freeze = 0
    log = []
    ended = "cap"
    for _ in range(cap):
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
            ended = "died"
            break
        if r.x_position > best:
            best, since = r.x_position, 0
        else:
            since += 1
            longest_freeze = max(longest_freeze, since)
            if since > stall:
                ended = "stuck"
                break
    on_surf = [l for l in log if l[1] == 1 and l[2] == 1 and l[3] == 1]
    off = [l for l in log if not (l[1] == 1 and l[2] == 1 and l[3] == 1)]
    ws = sorted({(l[1], l[2]) for l in log})
    return {"max_x": max(l[0] for l in log) if log else 0,
            "max_x_surface": (max(l[0] for l in on_surf) if on_surf else None),
            "n_frames": len(log), "ended": ended, "longest_freeze": longest_freeze,
            "areas_seen": sorted({l[3] for l in log}),
            "world_stage_seen": [list(x) for x in ws],
            "left_surface": bool(off),
            "completed_1_1": bool(any(w == 1 and s_ == 2 for w, s_ in ws)),
            "reached_flag": bool(any(l[0] >= FLAG for l in on_surf))}


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 100 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    lib = json.loads((ROOT / "data/startlib_policy.json").read_text())["states"]
    p1 = {e["seed"]: e for e in json.loads((ROOT / "data/traces/p1_200.json").read_text())["episodes"]}
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)
    old = json.loads(OLD_DATA.read_text()) if OLD_DATA.exists() else {}

    done = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    skipped = []
    for arm in ARMS:
        todo = [(s, r) for s in lib for r in range(REPS)
                if f"{arm}:{s['seed']}:{s['frame_index']}:{r}" not in done]
        if not todo:
            continue
        if not dl.can_afford(180):
            skipped.append({"arm": arm, "reason": "deadline", "n_unrun": len(todo)})
            print(f"{dl.stamp()} SKIP {arm}", flush=True)
            continue
        policy, cfg, _ = G.load_ckpt(arm)
        print(f"{dl.stamp()} {arm}: {len(todo)} episodes at STALL={NEW_STALL} CAP={NEW_CAP}",
              flush=True)
        sess = None
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), f"arm {arm}"):
                sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
                warm_session(sess, start.frame)
                for i, (st, rep) in enumerate(todo):
                    ep = p1.get(st["seed"])
                    if ep is None:
                        continue
                    obs = restore_state(sess, st, [f[3] for f in ep["frames"]], start.frame)
                    rec = rollout(sess, policy, cfg, obs, byte_of, lut, temp=TEMP,
                                  seed=st["frame_index"] * 100 + rep,
                                  stall=NEW_STALL, cap=NEW_CAP)
                    rec.update({"arm": arm, "start_x": st["x"], "rep": rep})
                    done[f"{arm}:{st['seed']}:{st['frame_index']}:{rep}"] = rec
                    if (i + 1) % CHUNK == 0 or i == len(todo) - 1:
                        PARTIAL.write_text(json.dumps(done, separators=(",", ":")))
                        print(f"  {dl.stamp()} {i + 1}/{len(todo)}", flush=True)
        except TimedOut as e:
            skipped.append({"arm": arm, "reason": str(e)})
            print(f"{dl.stamp()} TIMEOUT {arm}: {e}", flush=True)
        finally:
            if sess is not None:
                try:
                    sess.close()
                except BaseException:
                    pass
            PARTIAL.write_text(json.dumps(done, separators=(",", ":")))

    out = {"question": "how much of this project's reach measurement was the stall rule?",
           "old_condition": {"stall": OLD_STALL, "cap_frames": OLD_CAP,
                             "note": ("reused from hidden_area_check.partial.json -- identical arms, "
                                      "states and RNG seeds, so the pairing is exact and the control "
                                      "is the condition EVERY prior figure was measured under")},
           "new_condition": {"stall": NEW_STALL, "cap_frames": NEW_CAP},
           "arms": ARMS, "temperature": TEMP, "reps": REPS,
           "measurement_basis": "conditional_on_arrival",
           "skipped": skipped, "per_arm": {}}

    for arm in ARMS:
        pairs = []
        for k, nv in done.items():
            if nv["arm"] != arm:
                continue
            ov = old.get(k)
            if not ov:
                continue
            pairs.append({"start_x": nv["start_x"], "rep": nv["rep"],
                          "old_max_x": ov["max_x_naive"], "new_max_x": nv["max_x"],
                          "delta": nv["max_x"] - (ov["max_x_naive"] or 0),
                          "old_frames": ov["n_frames"], "new_frames": nv["n_frames"],
                          "new_ended": nv["ended"], "longest_freeze": nv["longest_freeze"],
                          "left_surface": nv["left_surface"],
                          "completed": nv["completed_1_1"]})
        if not pairs:
            continue
        d = np.array([p["delta"] for p in pairs], float)
        frz = np.array([p["longest_freeze"] for p in pairs], float)
        out["per_arm"][arm] = {
            "n_paired": len(pairs),
            "old_max_x_median": float(np.median([p["old_max_x"] for p in pairs])),
            "new_max_x_median": float(np.median([p["new_max_x"] for p in pairs])),
            "old_max_x_p90": float(np.percentile([p["old_max_x"] for p in pairs], 90)),
            "new_max_x_p90": float(np.percentile([p["new_max_x"] for p in pairs], 90)),
            "new_max_x_max": int(max(p["new_max_x"] for p in pairs)),
            "delta_median": float(np.median(d)), "delta_mean": float(d.mean()),
            "delta_max": float(d.max()),
            "frac_improved": float((d > 0).mean()), "frac_unchanged": float((d == 0).mean()),
            "longest_freeze": {"median": float(np.median(frz)), "p90": float(np.percentile(frz, 90)),
                               "p99": float(np.percentile(frz, 99)), "max": float(frz.max()),
                               "frac_above_300": float((frz > 300).mean())},
            "n_left_surface": sum(1 for p in pairs if p["left_surface"]),
            "n_completed_1_1": sum(1 for p in pairs if p["completed"]),
            "ended_new": dict(collections.Counter(p["new_ended"] for p in pairs)),
            "new_wall_histogram_32px": dict(sorted(collections.Counter(
                (p["new_max_x"] // 32) * 32 for p in pairs).items()))}

    tot_left = sum(v["n_left_surface"] for v in out["per_arm"].values())
    tot_comp = sum(v["n_completed_1_1"] for v in out["per_arm"].values())
    tot_n = sum(v["n_paired"] for v in out["per_arm"].values())
    med_deltas = [v["delta_median"] for v in out["per_arm"].values()]
    out["summary"] = {
        "n_paired_total": tot_n, "median_delta_per_arm": med_deltas,
        "n_left_surface": tot_left, "n_completed_1_1": tot_comp,
        "frac_longest_freeze_above_300": [v["longest_freeze"]["frac_above_300"]
                                          for v in out["per_arm"].values()]}
    big = any(m > 0 for m in med_deltas)
    improved = "/".join(f"{v['frac_improved']:.0%}" for v in out["per_arm"].values())
    med_str = " and ".join(f"{m:+.0f}" for m in med_deltas)
    out["verdict"] = (
        f"**THE STALL RULE WAS TRUNCATING REAL PROGRESS.** Paired on identical start states and RNG "
        f"seeds, raising the terminator from {OLD_STALL} to {NEW_STALL} frames moves median max_x by "
        f"{med_str} px per arm, and {improved} of episodes get further. **{tot_left} of {tot_n} now "
        f"leave the 1-1 surface and {tot_comp} complete the level**, against zero under the old rule -- "
        f"so `hidden_area_check`'s 'nothing leaves the surface' was itself the terminator firing before "
        f"the pipe transit finished. **Every reach and x_max figure in this project is a lower bound**, "
        f"and `reach_walls.json`'s walls are partly this artifact."
        if big else
        f"**THE STALL RULE WAS NOT THE BINDING CONSTRAINT.** Raising it from {OLD_STALL} to "
        f"{NEW_STALL} moves median max_x by {med_str} px per arm and {improved} of episodes get "
        f"further. The walls in `reach_walls.json` are genuine.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    for arm, v in out["per_arm"].items():
        print(f"\n{arm}: n={v['n_paired']} median max_x {v['old_max_x_median']:.0f} -> "
              f"{v['new_max_x_median']:.0f} (p90 {v['old_max_x_p90']:.0f} -> {v['new_max_x_p90']:.0f}, "
              f"max {v['new_max_x_max']})")
        print(f"   improved {v['frac_improved']:.0%} | left surface {v['n_left_surface']} | "
              f"completed {v['n_completed_1_1']} | ended {v['ended_new']}")
        f = v["longest_freeze"]
        print(f"   longest x-freeze: median {f['median']:.0f} p90 {f['p90']:.0f} "
              f"p99 {f['p99']:.0f} max {f['max']:.0f}; {f['frac_above_300']:.0%} exceed 300")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
