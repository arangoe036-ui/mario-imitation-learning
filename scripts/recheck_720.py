"""§4.2(2): re-derive block 56's "0 of 720 completions at T=0.7" at the corrected terminator.

That figure is cited in `NORTH_STAR.md` and was measured under `STALL=300`, which block 58 showed censors
14–17% of episodes and turned two real completions on disk into `stuck` and `budget`. A zero measured under a
rule that truncates episodes before the flagpole sequence finishes is not evidence of zero.

**`measurement_basis: conditional_on_arrival`** — 72 library start states, so this is "given the policy stands
here", and must not be tabled beside any from-the-level-start figure.

**3 repeats rather than block 56's 5**, because episodes are much longer at `STALL=6500`; 432 episodes is
enough to decide the binary question (does any completion appear at all) and is reported with that limit
stated rather than presented as a matched re-run.

Completion is judged by **stage advance to 1-2**, not by `max_x` — the lesson from block 58's mislabelled
pair, and from block 59's retraction, where an area union over a whole episode was mistaken for a route.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.argmax_startstates import restore_state  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/recheck_720.json"
PARTIAL = ROOT / "data/recheck_720.partial.json"

ARMS = ["P_84_cnn32_seed1", "P_84_cnn32_seed4"]
TEMP = 0.7
REPS = 3
CAP_NON_A = 4
CHUNK = 30
ARM_BUDGET_S = 25 * 60


def episode(sess, policy, cfg, obs, byte_of, lut, seed):
    s = cfg.frame_size
    rng = np.random.default_rng(seed)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    win[:] = _resize_gray(obs.rgb, (s, s))
    held, remaining = None, 0
    best = since = 0
    max_x_11 = 0
    reached_stage2 = False
    ended = "cap"
    for _ in range(RB.CAP_FRAMES):
        if remaining <= 0:
            with torch.no_grad():
                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
            p = torch.softmax(lg / TEMP, dim=-1).numpy()
            c = int(rng.choice(len(p), p=p / p.sum()))
            b, L = int(byte_of[c]), max(1, int(lut[c]))
            if not (b & A_BIT):
                L = min(L, CAP_NON_A)
            held, remaining = b, L
        remaining -= 1
        obs = sess.step(held)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (s, s))
        r = read_smb(obs.ram, obs.framecount)
        if r.world == 1 and r.stage == 1:
            max_x_11 = max(max_x_11, r.x_position)
        elif r.world == 1 and r.stage == 2:
            reached_stage2 = True
        if r.player_state in (0x06, 0x0B):
            ended = "died"
            break
        if r.x_position > best:
            best, since = r.x_position, 0
        else:
            since += 1
            if since > RB.STALL:
                ended = "stuck"
                break
    return {"max_x_in_1_1": max_x_11, "completed_1_1": reached_stage2, "ended": ended}


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    lib = json.loads((ROOT / "data/startlib_policy.json").read_text())["states"]
    p1 = {e["seed"]: e
          for e in json.loads((ROOT / "data/traces/p1_200.json").read_text())["episodes"]}
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    done = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    skipped = []
    for arm in ARMS:
        todo = [(s, r) for s in lib for r in range(REPS)
                if f"{arm}:{s['seed']}:{s['frame_index']}:{r}" not in done]
        if not todo or not dl.can_afford(180):
            if todo:
                skipped.append({"arm": arm, "n_unrun": len(todo), "reason": "deadline"})
            continue
        policy, cfg, _ = G.load_ckpt(arm)
        print(f"{dl.stamp()} {arm}: {len(todo)} episodes at STALL={RB.STALL}", flush=True)
        sess = None
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), arm):
                sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
                warm_session(sess, start.frame)
                for j, (st, rep) in enumerate(todo):
                    ep = p1.get(st["seed"])
                    if ep is None:
                        continue
                    obs = restore_state(sess, st, [f[3] for f in ep["frames"]], start.frame)
                    rec = episode(sess, policy, cfg, obs, byte_of, lut,
                                  st["frame_index"] * 100 + rep)
                    rec.update({"arm": arm, "start_x": st["x"], "rep": rep})
                    done[f"{arm}:{st['seed']}:{st['frame_index']}:{rep}"] = rec
                    if (j + 1) % CHUNK == 0 or j == len(todo) - 1:
                        PARTIAL.write_text(json.dumps(done, separators=(",", ":")))
                        print(f"  {dl.stamp()} {j + 1}/{len(todo)}", flush=True)
        except TimedOut as e:
            skipped.append({"arm": arm, "reason": str(e)})
        finally:
            if sess is not None:
                try:
                    sess.close()
                except BaseException:
                    pass
            PARTIAL.write_text(json.dumps(done, separators=(",", ":")))

    rows = list(done.values())
    comp = [r for r in rows if r["completed_1_1"]]
    out = {"figure_under_review": ("block 56: '0 of 720 completions at T=0.7', conditional_on_arrival "
                                   "over 72 start states, measured at STALL=300"),
           "measurement_basis": "conditional_on_arrival",
           "NOT_comparable_to": "any from-the-level-start figure",
           "terminator": RB.describe(), "reps": REPS, "arms": ARMS,
           "n_episodes": len(rows), "skipped": skipped,
           "completion_criterion": "stage advance to 1-2, never max_x",
           "completions": len(comp),
           "completing": [{"arm": r["arm"], "start_x": r["start_x"], "rep": r["rep"]}
                          for r in comp],
           "max_x_in_1_1": {"median": float(np.median([r["max_x_in_1_1"] for r in rows])),
                            "max": int(max(r["max_x_in_1_1"] for r in rows))} if rows else None,
           "per_arm": {a: {"n": sum(1 for r in rows if r["arm"] == a),
                           "completions": sum(1 for r in comp if r["arm"] == a)}
                       for a in ARMS}}
    if comp:
        out["verdict"] = (
            f"**BLOCK 56'S '0 OF 720' IS VOID.** At the corrected terminator, {len(comp)} of "
            f"{len(rows)} conditional-on-arrival episodes complete 1-1 (stage advance verified), where "
            f"the censored rule reported zero. The zero was the terminator, not the policy.")
    else:
        out["verdict"] = (
            f"**BLOCK 56'S '0 OF 720' SURVIVES, WITH LESS POWER.** {len(rows)} episodes at "
            f"STALL={RB.STALL} produced 0 completions, against 720 at STALL=300. The figure is not an "
            f"artifact of the terminator — but it is now 0 of {len(rows)}, not 0 of 720, and should be "
            f"restated with that n. Best max_x in 1-1: {out['max_x_in_1_1']['max']}.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
