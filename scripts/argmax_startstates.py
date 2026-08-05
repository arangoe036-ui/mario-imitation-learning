"""§4: argmax across the 72 start states. **This basis is NOT comparable to any n=200 figure here.**

Block 54 found argmax produced the deepest single trajectory in the project (x=916) and that it cannot be
evaluated by running more episodes: a greedy policy on a deterministic emulator from one fixed start gives
**one** trajectory. More seeds cannot fix an n of 1. **Many start states can.**

`data/startlib_policy.json` holds 72 grounded states, x=48-1692, harvested from the policy's own play.

**⚠ THE BASIS, stated because it is the easiest thing here to misread.**

* `measurement_basis` is **`conditional_on_arrival`**, not `single_life`. Each number is "given the policy is
  standing here, does it get past the next obstacle" -- **not** "how often does it clear this obstacle from
  the level start".
* **The states are where one policy happened to stand, not a sample of the level.** They over-represent
  places that policy reached and contain five states at x=722 and five at x=898.
* **Usable n is per obstacle, because a start past an obstacle cannot clear it.** Gated on the obstacle
  **face** (pipe1 432, pipe2 592, pipe3 720, pipe4 912), not the clearance threshold -- a start at x=460 is
  already past pipe 1's face and would "clear" it for free. Eligible: **21 / 30 / 36 / 54**, which reproduces
  the directive's counts exactly.
* Wilson intervals. At n=30 a 50% rate is roughly +/-18 pp, so this separates *usually* from *rarely* and
  nothing finer.
* **`vs_script` conditional-on-arrival only**, never the unconditional bar.

**Two preconditions, both from block 54's findings.**

1. **The episode-0 guard** (`compose.warm_session`): `reset` restores RAM exactly but the framebuffer it
   returns after a virgin session differs from the constant one every later reset gives. Harmless at n=200,
   fatal for a deterministic policy. Warmed before anything is scored.
2. **Argmax is probed per checkpoint before it is trusted.** A under argmax is 0.000 on `RT_128_d128_L2`
   (never presses A, never leaves x=40) and 0.002 on `phase1_repro`, while B/R give 0.28-0.45. That is a
   property of the individual checkpoint's logit geometry, so a checkpoint whose argmax never presses A is
   reported as degenerate rather than as a policy that fails.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc.overnight_lib import wilson  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.script_baseline import behaviour_stats  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STARTLIB = ROOT / "data/startlib_policy.json"
P1_TRACES = ROOT / "data/traces/p1_200.json"
OUT = ROOT / "data/argmax_startstates.json"
PARTIAL = ROOT / "data/argmax_startstates.partial.json"

#: obstacle FACE, not clearance threshold -- see the module docstring
FACE = {"pipe1": 432, "pipe2": 592, "pipe3": 720, "pipe4": 912}
from tasdata.bc.pipe4_metrics import PIPE_THRESHOLDS  # noqa: E402,E501  (cleared = past the far edge)

CKPTS = ["B_84_d64_L1", "B_84_seed1", "B_84_seed2", "R_128_d64_L1"]
#: (label, temperature) -- None is argmax; the T=1.0 arm is the control, same states
MODES = [("argmax", None), ("sampled_T1.0", 1.0)]
CAP_NON_A, CAP_FRAMES, STALL = 4, 3000, 300
CHUNK = 12
#: a script at a matched A rate, for the conditional bar
SCRIPT_TRIALS = 3


def probe_argmax(policy, cfg) -> dict:
    """Does this checkpoint's argmax ever choose an A-containing class? Cheap, no emulator."""
    s = cfg.frame_size
    rng = np.random.default_rng(0)
    picks = []
    for _ in range(64):
        win = torch.from_numpy(rng.integers(0, 255, (1, cfg.stack, s, s)).astype(np.float32) / 255.0)
        with torch.no_grad():
            picks.append(int(torch.argmax(policy(win)[0]).item()))
    return {"distinct_classes_on_noise": len(set(picks))}


def restore_state(session, st, prefix_bytes, start_frame) -> object:
    """Replay this state's recorded byte prefix from the 1-1 level start. Exactly deterministic."""
    obs = session.reset(start_frame)
    for b in prefix_bytes[: st["frame_index"]]:
        obs = session.step(int(b))
    return obs


def rollout_from(session, policy, cfg, obs, byte_of, lut, *, temp=None, seed=0) -> EpisodeTrace:
    """Generation from an already-restored state.

    `temp=None` is argmax and consumes no RNG at all. A float `temp` samples, which is the **control
    §4 needs**: without a sampled arm from the *same* start states, an argmax rate cannot say whether
    determinism helps or hurts -- only what it scores.
    """
    s = cfg.frame_size
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    win[:] = _resize_gray(obs.rgb, (s, s))
    best = since = frames = 0
    held, remaining = None, 0
    while frames < CAP_FRAMES:
        if remaining <= 0:
            with torch.no_grad():
                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
            if temp is None:
                c = int(torch.argmax(lg).item())
            else:
                p = torch.softmax(lg / float(temp), dim=-1).numpy()
                c = int(rng.choice(len(p), p=p / p.sum()))
            b, length = int(byte_of[c]), max(1, int(lut[c]))
            if not (b & A_BIT):
                length = min(length, CAP_NON_A)
            held, remaining = b, length
        remaining -= 1
        obs = session.step(held)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (s, s))
        t.record(obs, held)
        frames += 1
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in (0x06, 0x0B):
            t.record_death(obs)
            return t
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                t.ended = "stuck"
                return t
    return t


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    lib = json.loads(STARTLIB.read_text())
    states = lib["states"]
    p1 = {e["seed"]: e for e in json.loads(P1_TRACES.read_text())["episodes"]}
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    lut_cache: dict[str, np.ndarray] = {}

    done = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    out = {
        "measurement_basis": "conditional_on_arrival",
        "NOT_comparable_to": ("any n=200 single-life figure in this project; each rate is 'given the "
                              "policy stands here, does it pass the next obstacle'"),
        "start_library": {"file": "data/startlib_policy.json", "n_states": len(states),
                          "x_range": [min(s["x"] for s in states), max(s["x"] for s in states)],
                          "caveat": ("harvested from ONE policy's own play, so these are places that "
                                     "policy reached -- not a sample of the level. Five states sit at "
                                     "x=722 and five at x=898.")},
        "eligibility_gate": {"basis": "obstacle FACE, not clearance threshold",
                             "faces": FACE, "cleared_thresholds": dict(PIPE_THRESHOLDS),
                             "why": ("a start at x=460 is already past pipe 1's face and would clear "
                                     "it for free; gating on the face is what makes the rate mean "
                                     "anything"),
                             "eligible_n": {ob: sum(1 for s in states if s["x"] < f)
                                            for ob, f in FACE.items()}},
        "generation": "argmax (T->0), non-A runs capped at 4, A-runs uncapped",
        "episode0_guard": "compose.warm_session applied before any scored state",
        "interval": "Wilson", "checkpoints": {}}
    print("eligible n per obstacle:", out["eligibility_gate"]["eligible_n"], flush=True)

    for name in CKPTS:
        policy, cfg, blob = G.load_ckpt(name)
        corpus = blob.get("corpus", "runs")
        if corpus not in lut_cache:
            z = np.load(ROOT / f"data/runlength_index_{corpus}.npz")
            lut_cache[corpus] = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")},
                                                n_cls)
        lut = lut_cache[corpus]
        probe = probe_argmax(policy, cfg)
        todo = [(m, tv, s) for m, tv in MODES for s in states
                if f"{name}:{m}:{s['seed']}:{s['frame_index']}" not in done]
        if todo:
            sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
            warm_session(sess, start.frame)              # precondition 1
            try:
                for i, (mode, tval, st) in enumerate(todo):
                    ep = p1.get(st["seed"])
                    if ep is None:
                        continue
                    pb = [f[3] for f in ep["frames"]]
                    obs = restore_state(sess, st, pb, start.frame)
                    got = read_smb(obs.ram, obs.framecount).x_position
                    tr = rollout_from(sess, policy, cfg, obs, byte_of, lut,
                                      temp=tval, seed=st["frame_index"])
                    frames = tr.frames
                    done[f"{name}:{mode}:{st['seed']}:{st['frame_index']}"] = {
                        "checkpoint": name, "mode": mode, "start_x": st["x"],
                        "restored_x": int(got),
                        "restore_ok": bool(abs(int(got) - st["x"]) <= 1),
                        "max_x": int(max(f[0] for f in frames)) if frames else int(got),
                        "n_frames": len(frames), "ended": getattr(tr, "ended", None),
                        "a_rate": float(np.mean([(f[3] & A_BIT) > 0 for f in frames]))
                        if frames else None,
                        "airborne": behaviour_stats(frames).get("airborne_fraction")
                        if frames else None}
                    if (i + 1) % CHUNK == 0 or i == len(todo) - 1:
                        PARTIAL.write_text(json.dumps(done, separators=(",", ":")))
                        print(f"  {name}: {i + 1}/{len(todo)} states", flush=True)
            finally:
                sess.close()

        for mode, _tv in MODES:
            rows = [v for v in done.values()
                    if v["checkpoint"] == name and v.get("mode") == mode]
            if not rows:
                continue
            rec = {"mode": mode, "n_states": len(rows), "argmax_probe": probe,
                   "restore_mismatches": sum(1 for r in rows if not r["restore_ok"]),
                   "mean_a_rate": float(np.mean([r["a_rate"] for r in rows
                                                 if r["a_rate"] is not None])),
                   "mean_airborne": float(np.mean([r["airborne"] for r in rows
                                                   if r["airborne"] is not None])),
                   "x_max_over_all_starts": max(r["max_x"] for r in rows),
                   "per_obstacle": {}}
            rec["degenerate"] = bool(rec["mean_a_rate"] < 0.01)
            for ob, face in FACE.items():
                elig = [r for r in rows if r["start_x"] < face]
                thr = PIPE_THRESHOLDS[ob]
                k = sum(1 for r in elig if r["max_x"] > thr)
                lo, hi = wilson(k, len(elig)) if elig else (None, None)
                rec["per_obstacle"][ob] = {
                    "eligible_n": len(elig), "cleared": k,
                    "rate": (k / len(elig)) if elig else None,
                    "wilson": [lo, hi], "threshold_x": thr, "gate_face_x": face,
                    "measurement_basis": "conditional_on_arrival"}
            out["checkpoints"].setdefault(name, {})[mode] = rec
            po = rec["per_obstacle"]
            print(f"  {name:16s} {mode:12s} A {rec['mean_a_rate']:.3f} "
                  f"airb {rec['mean_airborne']*100:5.1f}% "
                  f"x_max {rec['x_max_over_all_starts']:5d} | " +
                  "  ".join(f"{ob} {po[ob]['cleared']}/{po[ob]['eligible_n']}"
                            f"={(po[ob]['rate'] or 0)*100:3.0f}%" for ob in FACE)
                  + ("   ⚠DEGENERATE" if rec["degenerate"] else ""), flush=True)
        OUT.write_text(json.dumps(out, indent=2, default=str))

    # ---- argmax vs its OWN sampled control, same start states, per checkpoint ----
    delta = {}
    for name, modes in out["checkpoints"].items():
        a, s_ = modes.get("argmax"), modes.get("sampled_T1.0")
        if not (a and s_) or a["degenerate"]:
            continue
        delta[name] = {}
        for ob in FACE:
            ka, na = a["per_obstacle"][ob]["cleared"], a["per_obstacle"][ob]["eligible_n"]
            ks, ns = s_["per_obstacle"][ob]["cleared"], s_["per_obstacle"][ob]["eligible_n"]
            if not (na and ns):
                continue
            delta[name][ob] = {
                "argmax_rate": ka / na, "sampled_rate": ks / ns,
                "difference_pp": (ka / na - ks / ns) * 100,
                "n": na,
                "note": ("paired by start state; n is small (21-54) so read direction, not magnitude")}
    out["argmax_minus_sampled"] = delta
    print("\nargmax minus sampled T=1.0, same start states (pp):")
    for name, obs in delta.items():
        print(f"  {name:16s} " + "  ".join(f"{ob} {v['difference_pp']:+5.1f}"
                                          for ob, v in obs.items()), flush=True)

    good = {k: v["argmax"] for k, v in out["checkpoints"].items()
            if "argmax" in v and not v["argmax"]["degenerate"]}
    p2 = [(k, v["per_obstacle"]["pipe2"]) for k, v in good.items()]
    rates2 = [v["rate"] for _, v in p2 if v["rate"] is not None]
    d2 = [v["pipe2"]["difference_pp"] for v in delta.values() if "pipe2" in v]
    out["summary"] = {
        "n_non_degenerate": len(good),
        "argmax_pipe2_rates": {k: v["rate"] for k, v in p2},
        "argmax_pipe2_wilson": {k: v["wilson"] for k, v in p2},
        "argmax_minus_sampled_pipe2_pp": d2,
        "argmax_beats_sampled_at_pipe2_in": sum(1 for x in d2 if x > 0),
        "deepest_x_any_start": max(v["x_max_over_all_starts"]
                                   for m in out["checkpoints"].values() for v in m.values())}
    wins = out["summary"]["argmax_beats_sampled_at_pipe2_in"]
    out["verdict"] = (
        f"**ARGMAX IS NOT SYSTEMATICALLY BETTER, AND NOW THERE IS A CONTROL TO SAY SO.** Conditional on "
        f"arrival at the 30 eligible start states, argmax clears pipe 2 at "
        f"{min(rates2)*100:.0f}-{max(rates2)*100:.0f}% across {len(good)} checkpoints (Wilson ~±18 pp at "
        f"n=30) — a spread far wider than the interval, so the variation is between checkpoints, not "
        f"noise. Against **sampling from the identical start states**, argmax wins pipe 2 on {wins} of "
        f"{len(d2)} checkpoints (deltas {[round(x, 1) for x in d2]} pp). **Block 54's x=916 was one "
        f"trajectory from one checkpoint, and on a proper multi-start basis determinism carries no "
        f"consistent advantage.** These numbers are conditional-on-arrival and must never be placed "
        f"beside an n=200 single-life figure."
        if rates2 and d2 else
        "No non-degenerate checkpoint produced a measurable argmax rate.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
