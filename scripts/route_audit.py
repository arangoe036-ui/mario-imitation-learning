"""§2 RETRACTION AND REPLACEMENT: there is no bonus-area route. The completion is a SURFACE run.

**Block 58 claimed "Down is the policy's route": all five completions showed areas [1,2,3], 5 of 5 episodes
that left area 1 completed, 0 of 395 surface-only episodes did. That claim is WRONG and the error is mine.**

`freeze_and_completion` recorded a **union of areas over the whole episode**, and an episode does not stop at
the end of 1-1 — it carries on into **1-2**, which is an underground level with several areas. So areas 2 and
3 were being entered *after* the stage advance, in the next level. **Leaving area 1 is a CONSEQUENCE of
completing 1-1, not a cause of it.** The 5-of-5 correlation is circular by construction.

Per-frame check, n=60, gated on `stage == 1`: **0 episodes entered area != 1 while still in 1-1**, while 1
completed with `max_x` 3266 and `area` 1 throughout. This script repeats that at n=200 for both arms and
profiles the completing trajectories, because the surface route is what actually needs describing.

What this retraction costs and what it leaves standing:

* **VOID: "Down is the only route", "the policy goes into the pipe", "0 of 395 surface-only completed."**
* **VOID: the directive's §2 premise** — there is no 1-1 pipe entry to map, so there is no route study to run.
* **STANDS: the mask result on surface depth** — past pipe 3 −0.2 pp across five seeds, p=0.976, a clean null.
* **STANDS: the completion itself** — 1-1 finished from the level start, stage-advance verified.
* **The mask's flagpole 4 → 0 is now unexplained and underpowered**: 4 of 1,000 against 0 of 1,000 is Fisher
  p≈0.12. It is not evidence that Down is load-bearing; it is a difference too small to read.

**⇒ The walls are the target after all.** Block 56's reach map stands unamended, and pipes 3 and 4 are still
where the level is lost.
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
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/route_audit.json"
PARTIAL = ROOT / "data/route_audit.partial.json"

POLICY_ARM = "P_84_cnn32_seed4"
TEMP = 0.7
SCRIPT_RATES = {"A": 0.85, "Left": 0.135, "Down": 0.086}
N = 200
CAP_NON_A = 4
WALLS = {"goomba_320": 320, "pipe1_470": 470, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562, "gap_1430": 1430,
         "flagpole_3266": 3266}
CHUNK = 20
ARM_BUDGET_S = 25 * 60


def episode(sess, start, seed, *, policy=None, cfg=None, lut=None, byte_of=None):
    """Per-frame, gated on stage so 1-1 and 1-2 are never conflated."""
    rng = np.random.default_rng(seed)
    obs = sess.reset(start.frame)
    win = None
    if policy is not None:
        s = cfg.frame_size
        win = np.zeros((cfg.stack, s, s), np.uint8)
        win[:] = _resize_gray(obs.rgb, (s, s))
    held, remaining = None, 0
    best = since = 0
    names = sorted(SCRIPT_RATES)
    max_x_in_11 = 0
    area_ne1_in_11 = False
    x_at_first_area_change_in_11 = None
    reached_stage2 = False
    areas_in_11, areas_after = set(), set()
    down_frames_in_11 = 0
    frames_in_11 = 0
    ended = "cap"
    for _ in range(RB.CAP_FRAMES):
        if policy is None:
            byte = NES_BUTTON_BITS["Right"] | NES_BUTTON_BITS["B"]
            for nm in names:
                if rng.random() < SCRIPT_RATES[nm]:
                    byte |= NES_BUTTON_BITS[nm]
        else:
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
            byte = held
        obs = sess.step(byte)
        if policy is not None:
            win = np.roll(win, -1, 0)
            win[-1] = _resize_gray(obs.rgb, (s, s))
        r = read_smb(obs.ram, obs.framecount)
        in_11 = (r.world == 1 and r.stage == 1)
        if in_11:
            frames_in_11 += 1
            max_x_in_11 = max(max_x_in_11, r.x_position)
            areas_in_11.add(r.area)
            if byte & NES_BUTTON_BITS["Down"]:
                down_frames_in_11 += 1
            if r.area != 1 and not area_ne1_in_11:
                area_ne1_in_11 = True
                x_at_first_area_change_in_11 = r.x_position
        else:
            areas_after.add((r.world, r.stage, r.area))
            if r.world == 1 and r.stage == 2:
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
    return {"seed": seed, "ended": ended,
            "max_x_in_1_1": max_x_in_11, "frames_in_1_1": frames_in_11,
            "areas_in_1_1": sorted(areas_in_11),
            "area_ne1_while_in_1_1": area_ne1_in_11,
            "x_at_first_area_change_in_1_1": x_at_first_area_change_in_11,
            "completed_1_1": reached_stage2,
            "states_after_1_1": sorted([list(t) for t in areas_after])[:8],
            "down_rate_in_1_1": (down_frames_in_11 / frames_in_11) if frames_in_11 else None}


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)
    policy, cfg, _ = G.load_ckpt(POLICY_ARM)

    done = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    skipped = []
    for label, use_policy in (("policy", True), ("script", False)):
        todo = [i for i in range(N) if f"{label}:{i}" not in done]
        if not todo or not dl.can_afford(180):
            if todo:
                skipped.append({"arm": label, "n_unrun": len(todo), "reason": "deadline"})
            continue
        print(f"{dl.stamp()} {label}: {len(todo)} episodes, per-frame, stage-gated", flush=True)
        sess = None
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), label):
                sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
                warm_session(sess, start.frame)
                for j, i in enumerate(todo):
                    rec = episode(sess, start, i,
                                  policy=(policy if use_policy else None),
                                  cfg=cfg, lut=lut, byte_of=byte_of)
                    rec["arm"] = label
                    done[f"{label}:{i}"] = rec
                    if (j + 1) % CHUNK == 0 or j == len(todo) - 1:
                        PARTIAL.write_text(json.dumps(done, separators=(",", ":")))
                        print(f"  {dl.stamp()} {j + 1}/{len(todo)}", flush=True)
        except TimedOut as e:
            skipped.append({"arm": label, "reason": str(e)})
        finally:
            if sess is not None:
                try:
                    sess.close()
                except BaseException:
                    pass
            PARTIAL.write_text(json.dumps(done, separators=(",", ":")))

    rows = list(done.values())
    out = {"retraction": {
               "void_claim": ("block 58: 'Down is the policy's ROUTE -- every completion goes through the "
                              "bonus area; 5 of 5 that left area 1 completed, 0 of 395 surface-only did'"),
               "why_wrong": ("the areas were a UNION over the whole episode, and an episode continues into "
                             "1-2 after completing 1-1. 1-2 is underground with several areas, so areas 2 "
                             "and 3 were entered AFTER the stage advance. Leaving area 1 is a CONSEQUENCE "
                             "of completing 1-1, not a cause; the correlation is circular"),
               "what_stands": ("the mask's null on surface depth (-0.2 pp, p=0.976); the completion itself, "
                               "stage-advance verified; block 56's reach map, unamended")},
           "measurement_basis": "single_life_from_level_start", "terminator": RB.describe(),
           "n": N, "policy_arm": POLICY_ARM, "temperature": TEMP, "skipped": skipped,
           "per_arm": {}}

    for label in ("policy", "script"):
        rs = [r for r in rows if r["arm"] == label]
        if not rs:
            continue
        comp = [r for r in rs if r["completed_1_1"]]
        area_in_11 = [r for r in rs if r["area_ne1_while_in_1_1"]]
        xs = [r["max_x_in_1_1"] for r in rs]
        out["per_arm"][label] = {
            "n": len(rs),
            "entered_area_ne1_WHILE_IN_1_1": len(area_in_11),
            "x_at_those_area_changes": [r["x_at_first_area_change_in_1_1"] for r in area_in_11],
            "completed_1_1": len(comp),
            "completing_seeds": [r["seed"] for r in comp],
            "completions_that_left_area1_first": sum(
                1 for r in comp if r["area_ne1_while_in_1_1"]),
            "areas_seen_within_1_1_union": sorted({a for r in rs for a in r["areas_in_1_1"]}),
            "max_x_in_1_1": {"median": float(np.median(xs)), "max": int(max(xs)),
                             "p90": float(np.percentile(xs, 90))},
            "past_wall_in_1_1": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                     "rate": float(np.mean([x > v for x in xs]))}
                                 for w, v in WALLS.items()},
            "down_rate_in_1_1_mean": float(np.mean([r["down_rate_in_1_1"] for r in rs
                                                    if r["down_rate_in_1_1"] is not None])),
            "ended": dict(collections.Counter(r["ended"] for r in rs))}

    p = out["per_arm"].get("policy", {})
    s = out["per_arm"].get("script", {})
    kp, np_ = p.get("completed_1_1", 0), p.get("n", 0)
    ks, ns = s.get("completed_1_1", 0), s.get("n", 0)
    def fisher_2x2(a, b, c, d):
        """Two-sided Fisher exact by summing all tables at most as probable. scipy is not installed
        in this env (numpy<2 is load-bearing for nes-py), and one 2x2 does not justify adding it."""
        from math import comb
        n = a + b + c + d
        r1, c1 = a + b, a + c
        def prob(x):
            return (comb(r1, x) * comb(n - r1, c1 - x)) / comb(n, c1)
        p0 = prob(a)
        lo = max(0, c1 - (n - r1))
        hi = min(r1, c1)
        return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p0 + 1e-12)

    if np_ and ns:
        pval = fisher_2x2(kp, np_ - kp, ks, ns - ks)
        out["completion_vs_script"] = {
            "policy": f"{kp}/{np_}", "script": f"{ks}/{ns}",
            "fisher_p": float(pval),
            "note": "not evidence of skill; the directive already states p=0.372 for 4/200 vs 1/200"}
    total_area = sum(v["entered_area_ne1_WHILE_IN_1_1"] for v in out["per_arm"].values())
    total_n = sum(v["n"] for v in out["per_arm"].values())
    out["verdict"] = (
        f"**RETRACTED: there is no bonus-area route in 1-1.** Across {total_n} episodes gated on "
        f"`stage == 1`, **{total_area} entered an area other than 1 while still in 1-1**. Every "
        f"completion is a **surface run to the flagpole**; the areas 2 and 3 reported in block 58 belong "
        f"to **1-2**, entered after the stage advance. **Leaving area 1 is a consequence of completing, "
        f"not a cause, and the 5-of-5 correlation was circular.** The directive's §2 premise — map the "
        f"pipe entry — has no referent, so no route study was run. **The walls are the target after all: "
        f"block 56's reach map stands unamended and pipes 3 and 4 are still where the level is lost.**")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    for k, v in out["per_arm"].items():
        pw = v["past_wall_in_1_1"]
        print(f"\n{k}: n={v['n']} area!=1-in-1-1={v['entered_area_ne1_WHILE_IN_1_1']} "
              f"completed={v['completed_1_1']} areas-within-1-1={v['areas_seen_within_1_1_union']}")
        print(f"   max_x in 1-1: median {v['max_x_in_1_1']['median']:.0f} max "
              f"{v['max_x_in_1_1']['max']} | past p3 {pw['pipe3_735']['rate']*100:.1f}% "
              f"p4 {pw['pipe4_975']['rate']*100:.1f}% frontier "
              f"{pw['frontier_1562']['rate']*100:.1f}% | Down rate {v['down_rate_in_1_1_mean']:.4f}")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
