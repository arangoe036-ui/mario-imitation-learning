"""§3a + §3b: collect the policy's OWN failure states and search for corrections from them.

**This is DAgger with search standing in for the expert we do not have.** It is not the self-imitation that
failed three times: those trained on the policy's lucky *successes*, which reinforces current behaviour and
has a fixed point. This trains on **corrections to its own mistakes**, and the label comes from search.

Why it is necessary, measured: the frozen training split is **25 runs, 1,223,797 frames, zero deaths**. A TAS
visits one near-optimal trajectory and contains no recoveries, so the policy is off-data within seconds and
the corpus cannot supply the correction at any training length.

**§3a — failure states, with an upstream ladder.** §2 found five of twelve at-the-wall arrivals unsolvable by
any single action *and* by retreat macros, so for those the mistake was already made and no label exists at
the wall. Every failure therefore gets savestates at **30, 60 and 120 frames before** it, and the search walks
outward until it finds a solution, recording which lead was needed.

**The wall list comes from the data.** The failure histogram is reported as collected, over the whole level to
the flagpole at 3266 — not from a pre-chosen list of three pipes.

**Failures are separated by y at the stall:** stalled at a pipe's *face* (ground level) and stalled *on top of*
a pipe land in the same x-bin and need different corrections — "clear it" versus "get off it". The owner is
watching the second happen.

**§3b — search.** ~`N_SAMPLED` sequences per state drawn from the policy at **high temperature**, which keeps
solutions on the policy's own manifold and so learnable, plus **64 hand-built retreat macros**, because the
corpus holds 695 retreats in 132,005 run tokens (0.526%) and sampling would expect **1.8** of them in 350
draws. §2 already showed a retreat rescuing a state single-action search could not.

**⚠ Reduced from the directive's 350 to fit the block's other stages: `N_SAMPLED` sequences per state.**
Stated rather than silently applied.
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
from tasdata.ram import on_ground, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/dagger_round1_states.json"
SOLS = ROOT / "data/dagger_round1_solutions.json"
PARTIAL = ROOT / "data/dagger_round1.partial.json"

ARM = "P_84_cnn32"                 # 1,000 steps -- the peak
ROLL_TEMP = 0.7
SEARCH_TEMP = 1.8                  # high: keeps solutions on the policy's manifold but widens them
N_EPISODES = 200
N_STATES = 60
LEADS = [30, 60, 120]
N_SAMPLED = 250
SEQ_FRAMES = 150
CAP_NON_A = 4
LOCOMOTION = 0x82
LEFT_BIT = 0x40
RETREAT_L, RETREAT_M, RETREAT_H = [8, 16, 24, 32], [16, 24, 32, 48], [12, 16, 20, 24]
#: progress past the capture x that counts as "corrected"
CLEAR_MARGIN = 64
GROUND_Y = 432                     # floor in absolute y; above this (smaller y) is elevated
ARM_BUDGET_S = 40 * 60


def wall_bin(x):
    for name, lo, hi in (("goomba_288", 240, 340), ("pipe1_432", 400, 500),
                         ("pipe2_592", 560, 660), ("pipe3_720", 660, 760),
                         ("pipe4_912", 860, 1000), ("koopas_1216", 1150, 1300),
                         ("frontier_1504", 1450, 1600), ("gap_1380", 1300, 1450)):
        if lo <= x < hi:
            return name
    return f"other_{int(x) // 200 * 200}"


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 150 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)
    policy, cfg, _ = G.load_ckpt(ARM)
    s_ = cfg.frame_size

    state = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    episodes = state.get("episodes", [])

    # ---------------- 3a: roll out and record ----------------
    if len(episodes) < N_EPISODES:
        sess = None
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 120), "rollouts"):
                sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
                warm_session(sess, start.frame)
                for seed in range(len(episodes), N_EPISODES):
                    rng = np.random.default_rng(seed)
                    obs = sess.reset(start.frame)
                    win = np.zeros((cfg.stack, s_, s_), np.uint8)
                    win[:] = _resize_gray(obs.rgb, (s_, s_))
                    held, remaining = None, 0
                    best = since = 0
                    bytes_, xs, ys, states_ = [], [], [], []
                    ended, completed = "cap", False
                    for _ in range(RB.CAP_FRAMES):
                        if remaining <= 0:
                            with torch.no_grad():
                                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
                            p = torch.softmax(lg / ROLL_TEMP, dim=-1).numpy()
                            c = int(rng.choice(len(p), p=p / p.sum()))
                            b, L = int(byte_of[c]), max(1, int(lut[c]))
                            if not (b & A_BIT):
                                L = min(L, CAP_NON_A)
                            held, remaining = b, L
                        remaining -= 1
                        obs = sess.step(held)
                        win = np.roll(win, -1, 0)
                        win[-1] = _resize_gray(obs.rgb, (s_, s_))
                        r = read_smb(obs.ram, obs.framecount)
                        bytes_.append(int(held))
                        xs.append(int(r.x_position))
                        ys.append(int(obs.ram[0x00B5]) * 256 + int(obs.ram[0x03B8]))
                        states_.append(int(r.player_state))
                        if r.world == 1 and r.stage == 2:
                            completed = True
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
                    # the failure frame: where progress stopped, not the last frame
                    fmax = int(np.argmax(xs)) if xs else 0
                    episodes.append({
                        "seed": seed, "ended": ended, "completed": completed,
                        "n_frames": len(bytes_), "max_x": int(max(xs)) if xs else 0,
                        "fail_frame": fmax, "fail_x": xs[fmax] if xs else 0,
                        "fail_y": ys[fmax] if ys else 0,
                        "bytes": bytes_})
                    if (seed + 1) % 20 == 0:
                        state["episodes"] = episodes
                        PARTIAL.write_text(json.dumps(state, separators=(",", ":")))
                        print(f"  {dl.stamp()} rollouts {seed + 1}/{N_EPISODES}", flush=True)
        except TimedOut as e:
            print(f"{dl.stamp()} rollout timeout: {e}", flush=True)
        finally:
            if sess is not None:
                try:
                    sess.close()
                except BaseException:
                    pass
            state["episodes"] = episodes
            PARTIAL.write_text(json.dumps(state, separators=(",", ":")))

    fails = [e for e in episodes if not e["completed"]]
    hist = collections.Counter(wall_bin(e["fail_x"]) for e in fails)
    elevated = collections.Counter(
        (wall_bin(e["fail_x"]), "on_top" if e["fail_y"] < GROUND_Y - 8 else "at_face")
        for e in fails)
    print(f"\n{dl.stamp()} failure histogram over {len(fails)} failures:", flush=True)
    for k, v in hist.most_common():
        ot = elevated.get((k, "on_top"), 0)
        af = elevated.get((k, "at_face"), 0)
        print(f"    {k:16s} {v:4d}   at_face {af:4d}  on_top {ot:4d}", flush=True)

    # ---------------- select states, stratified by wall ----------------
    by_wall = collections.defaultdict(list)
    for e in fails:
        by_wall[wall_bin(e["fail_x"])].append(e)
    chosen, wi = [], 0
    walls = [w for w, _ in hist.most_common()]
    while len(chosen) < N_STATES and walls:
        w = walls[wi % len(walls)]
        if by_wall[w]:
            chosen.append(by_wall[w].pop(0))
        else:
            walls.remove(w)
            continue
        wi += 1

    out = {
        "arm": ARM, "rollout_temperature": ROLL_TEMP, "search_temperature": SEARCH_TEMP,
        "n_episodes": len(episodes), "n_failures": len(fails),
        "n_completions": sum(1 for e in episodes if e["completed"]),
        "terminator": RB.describe(), "leads": LEADS,
        "n_sampled_per_state": N_SAMPLED,
        "reduction_note": (f"directive suggested ~350 sampled sequences per state; reduced to "
                           f"{N_SAMPLED} to fit distillation, measurement and the steps ladder in "
                           f"one block"),
        "failure_histogram": dict(hist),
        "failure_histogram_by_height": {f"{k[0]}|{k[1]}": v for k, v in elevated.items()},
        "height_note": ("y < 424 absolute counts as ON TOP of a pipe; the floor is 432. 'At the face' "
                        "and 'on top' land in the same x-bin and need different corrections"),
        "n_states_selected": len(chosen), "states": []}

    # ---------------- 3b: search from each state, walking the lead ladder ----------------
    solutions = []
    sess = None
    try:
        with time_limit(min(ARM_BUDGET_S, dl.remaining() - 120), "search"):
            sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
            warm_session(sess, start.frame)
            for si, ep in enumerate(chosen):
                if dl.remaining() < 180:
                    out["search_truncated_at_state"] = si
                    break
                rec = {"seed": ep["seed"], "wall": wall_bin(ep["fail_x"]),
                       "fail_x": ep["fail_x"], "fail_y": ep["fail_y"],
                       "on_top": bool(ep["fail_y"] < GROUND_Y - 8),
                       "leads_tried": [], "solved_at_lead": None,
                       "n_solutions": 0, "n_retreat_solutions": 0}
                for lead in LEADS:
                    f0 = max(0, ep["fail_frame"] - lead)
                    # replay the episode's own prefix -- exactly deterministic
                    obs = sess.reset(start.frame)
                    for b in ep["bytes"][:f0]:
                        obs = sess.step(int(b))
                    r0 = read_smb(obs.ram, obs.framecount)
                    slot = 100 + si
                    sess.save_scratch(slot)
                    target = r0.x_position + CLEAR_MARGIN
                    win0 = np.zeros((cfg.stack, s_, s_), np.uint8)
                    win0[:] = _resize_gray(obs.rgb, (s_, s_))
                    found = []
                    # -- policy-sampled sequences at high temperature --
                    for k in range(N_SAMPLED):
                        sess.load_scratch(slot)
                        win = win0.copy()
                        rng = np.random.default_rng(si * 10_000 + lead * 100 + k)
                        held, remaining = None, 0
                        seq, best = [], r0.x_position
                        dead = False
                        for _ in range(SEQ_FRAMES):
                            if remaining <= 0:
                                with torch.no_grad():
                                    lg = policy(
                                        torch.from_numpy(win[None]).float().div_(255.0))[0]
                                p = torch.softmax(lg / SEARCH_TEMP, dim=-1).numpy()
                                c = int(rng.choice(len(p), p=p / p.sum()))
                                b, L = int(byte_of[c]), max(1, int(lut[c]))
                                if not (b & A_BIT):
                                    L = min(L, CAP_NON_A)
                                held, remaining = b, L
                            remaining -= 1
                            o = sess.step(held)
                            win = np.roll(win, -1, 0)
                            win[-1] = _resize_gray(o.rgb, (s_, s_))
                            seq.append(int(held))
                            rr = read_smb(o.ram, o.framecount)
                            best = max(best, rr.x_position)
                            if rr.player_state in (0x06, 0x0B):
                                dead = True
                                break
                            if best > target:
                                break
                        if best > target and not dead:
                            found.append({"kind": "sampled", "bytes": seq})
                    # -- injected retreat macros --
                    n_retreat = 0
                    for L in RETREAT_L:
                        for M in RETREAT_M:
                            for H in RETREAT_H:
                                sess.load_scratch(slot)
                                seq, best, dead = [], r0.x_position, False
                                for f in range(L + M + 160):
                                    if f < L:
                                        byte = LEFT_BIT
                                    elif f < L + M:
                                        byte = LOCOMOTION
                                    else:
                                        byte = LOCOMOTION | (A_BIT if f < L + M + H else 0)
                                    o = sess.step(byte)
                                    seq.append(byte)
                                    rr = read_smb(o.ram, o.framecount)
                                    best = max(best, rr.x_position)
                                    if rr.player_state in (0x06, 0x0B):
                                        dead = True
                                        break
                                    if best > target:
                                        break
                                if best > target and not dead:
                                    found.append({"kind": "retreat", "bytes": seq})
                                    n_retreat += 1
                    rec["leads_tried"].append(
                        {"lead": lead, "start_x": int(r0.x_position),
                         "start_y": int(obs.ram[0x00B5]) * 256 + int(obs.ram[0x03B8]),
                         "grounded": bool(on_ground(obs.ram)),
                         "target_x": int(target), "n_found": len(found),
                         "n_retreat": n_retreat})
                    if found:
                        rec["solved_at_lead"] = lead
                        rec["n_solutions"] = len(found)
                        rec["n_retreat_solutions"] = n_retreat
                        for f in found:
                            solutions.append({"state": si, "seed": ep["seed"],
                                              "wall": rec["wall"], "lead": lead,
                                              "prefix_frames": f0, "kind": f["kind"],
                                              "bytes": f["bytes"]})
                        break
                out["states"].append(rec)
                print(f"  {dl.stamp()} state {si:3d} {rec['wall']:16s} x={rec['fail_x']:5d} "
                      f"{'on_top' if rec['on_top'] else 'at_face'} -> "
                      f"{'lead ' + str(rec['solved_at_lead']) if rec['solved_at_lead'] else 'UNSOLVED'}"
                      f" ({rec['n_solutions']} sols, {rec['n_retreat_solutions']} retreat)",
                      flush=True)
                OUT.write_text(json.dumps(out, indent=2, default=str))
                SOLS.write_text(json.dumps({"n": len(solutions), "solutions": solutions},
                                           separators=(",", ":")))
    except TimedOut as e:
        out["search_timed_out"] = str(e)
    finally:
        if sess is not None:
            try:
                sess.close()
            except BaseException:
                pass

    st = out["states"]
    solved = [s for s in st if s["solved_at_lead"]]
    out["summary"] = {
        "n_states": len(st), "n_solved": len(solved),
        "solved_at_lead": dict(collections.Counter(s["solved_at_lead"] for s in solved)),
        "n_solutions_total": len(solutions),
        "n_retreat_solutions": sum(s["n_retreat_solutions"] for s in st),
        "solved_by_wall": dict(collections.Counter(s["wall"] for s in solved)),
        "unsolved_by_wall": dict(collections.Counter(
            s["wall"] for s in st if not s["solved_at_lead"])),
        "on_top_states": sum(1 for s in st if s["on_top"]),
        "on_top_solved": sum(1 for s in solved if s["on_top"])}
    out["verdict"] = (
        f"Collected {len(st)} failure states across {len(out['failure_histogram'])} walls; "
        f"**{len(solved)} solved by search, yielding {len(solutions)} correction sequences** "
        f"({out['summary']['n_retreat_solutions']} from injected retreat macros). Lead distances "
        f"needed: {out['summary']['solved_at_lead']}. "
        f"{out['summary']['on_top_states']} states were stalled ON TOP of a pipe rather than at its "
        f"face, of which {out['summary']['on_top_solved']} were solved.")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    SOLS.write_text(json.dumps({"n": len(solutions), "solutions": solutions},
                               separators=(",", ":")))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} and {SOLS}")


if __name__ == "__main__":
    main()
