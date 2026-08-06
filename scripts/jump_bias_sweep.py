"""§2: does biasing the policy toward jumping close the gap to the high-jump script?

The measured deficit: a script running **A 0.85** gets further through 1-1 than the policy (past pipe 3
57.5% vs 47.5%, p=0.044). At **matched** rates the policy wins by ~50 pp at pipe 2. So state-conditioning is
worth a great deal and the policy's chosen jump *rate* is worse than a simple higher one. **The policy
under-jumps.**

The intervention is a **logit bonus on every vocabulary class whose combo contains A**, applied before
sampling. Not temperature — temperature scales all logits together and block 58's ladder showed that fails
(lowering A from 0.544 to 0.284 monotonically *worsened* clearance).

**The dose is calibrated, not guessed.** `LEDGER` records that an offline-fitted logit offset overshot badly
because the arm visits its own states; the fix is live bisection. So each target A rate is reached by
bisecting the bonus against short live rollouts, and the *realised* rate is reported beside every dose.

**⚠ Reported as a dose–response including the degenerate end.** `sustain_loss` once produced an always-jump
policy, so `a_hold` median/p99/max and airborne fraction are reported at every dose — the degeneracy is
visible in those, never in clearance alone.

**⚠ Correction to the directive's premise, carried into the build:** it gives `P_84_cnn32_seed4` as
"A ≈0.505". Measured at T=0.7 with `STALL=6500`, that arm's A rate is **0.338**. The gap to the 0.85 script
is therefore larger than stated, not smaller. The 1,000-step arms used here sit at **A 0.477–0.500**.
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
from scripts.compose import warm_session  # noqa: E402
from scripts.scaleup_eval import resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT, a_hold_onsets, hold_stats  # noqa: E402
from tasdata.bc.script_baseline import behaviour_stats, vs_script  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/jump_bias_sweep.json"
TRACED = ROOT / "data/traces"

ARMS = ["PK32_84_s0", "PK32_84_s1", "PK32_84_s2"]      # 1,000 steps -- the peak, not 15,000
TEMP = 0.7
N_EVAL = 200
CAP_NON_A = 4
TARGETS = [None, 0.60, 0.70, 0.80, 0.85, 0.90]          # None = unbiased, whatever it lands at
CAL_EPISODES = 12
CAL_ITERS = 12
WALLS = {"goomba_320": 320, "pipe1_470": 470, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562, "flagpole_3266": 3266}
EXPERT_AIRBORNE = 0.611
ARM_BUDGET_S = 15 * 60


def rollout(session, policy, cfg, start, seed, lut, byte_of, a_mask, bonus, *, temp,
            max_frames=None):
    """`capped` generation with a logit bonus added to every A-containing class."""
    s = cfg.frame_size
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    win[:] = _resize_gray(obs.rgb, (s, s))
    held, remaining = None, 0
    best = since = frames = 0
    cap = max_frames or RB.CAP_FRAMES
    while frames < cap:
        if remaining <= 0:
            with torch.no_grad():
                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
            if bonus:
                lg = lg + bonus * a_mask          # bonus on the A-containing classes only
            e = np.exp((lg - lg.max()) / float(temp))
            p = e / e.sum()
            c = int(rng.choice(len(p), p=p))
            b, L = int(byte_of[c]), max(1, int(lut[c]))
            if not (b & A_BIT):
                L = min(L, CAP_NON_A)
            held, remaining = b, L
        remaining -= 1
        obs = session.step(held)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (s, s))
        t.record(obs, held)
        frames += 1
        r = read_smb(obs.ram, obs.framecount)
        if r.player_state in (0x06, 0x0B):
            t.record_death(obs)
            return t
        if r.x_position > best:
            best, since = r.x_position, 0
        else:
            since += 1
            if since > RB.STALL:
                t.ended = "stuck"
                return t
    return t


def realised_a(traces):
    b = np.asarray([f[3] for t in traces for f in t.frames], dtype=np.int64)
    return float(((b & A_BIT) > 0).mean()) if b.size else 0.0


def calibrate(sess_get, policy, cfg, start, lut, byte_of, a_mask, target, dl):
    """Bisect the logit bonus against LIVE rollouts. Offline fitting overshoots -- see LEDGER."""
    lo, hi = 0.0, 8.0
    trace_log = []
    s = sess_get()
    try:
        for it in range(CAL_ITERS):
            if dl.remaining() < 120:
                break
            mid = (lo + hi) / 2
            tr = [rollout(s, policy, cfg, start, 10_000 + it * 100 + k, lut, byte_of, a_mask,
                          mid, temp=TEMP, max_frames=900)
                  for k in range(CAL_EPISODES)]
            got = realised_a(tr)
            trace_log.append({"iter": it, "bonus": mid, "realised_A": got})
            if abs(got - target) < 0.012:
                return mid, got, trace_log
            if got < target:
                lo = mid
            else:
                hi = mid
    finally:
        s.close()
    return (lo + hi) / 2, (trace_log[-1]["realised_A"] if trace_log else None), trace_log


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 180 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    a_mask = np.array([1.0 if (int(byte_of[c]) & A_BIT) else 0.0 for c in range(n_cls)])
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("calibration", {})
    out.setdefault("skipped", [])
    out.update({
        "intervention": ("logit bonus added to every vocabulary class whose combo contains A, before "
                         "sampling; NOT temperature -- block 58's ladder showed temperature fails"),
        "checkpoints": ARMS, "steps": 1000, "temperature": TEMP, "n_eval": N_EVAL,
        "terminator": RB.describe(), "measurement_basis": "single_life_from_level_start",
        "targets_realised_A": TARGETS,
        "calibration_method": ("live bisection on short rollouts; LEDGER records that an offline-fitted "
                               "logit offset overshot (0.349 realised against a 0.219 target) because "
                               "the arm visits its own states"),
        "directive_premise_correction": (
            "the directive gives P_84_cnn32_seed4 as 'A ~0.505'; measured at T=0.7 with STALL=6500 it is "
            "0.338. The 1,000-step arms here sit at 0.477-0.500. The gap to the 0.85 script is larger "
            "than stated."),
        "degeneracy_watch": ("a_hold median/p99/max and airborne reported at every dose; sustain_loss "
                             "once produced an always-jump policy and clearance alone would hide it"),
        "expert_airborne": EXPERT_AIRBORNE})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    for arm in ARMS:
        if not (ROOT / f"data/bc_scaleup/{arm}.pt").exists():
            continue
        policy, cfg, blob = G.load_ckpt(arm)
        for target in TARGETS:
            tag = "unbiased" if target is None else f"A{target:.2f}"
            key = f"{arm}/{tag}"
            if key in out["arms"]:
                continue
            if not dl.can_afford(200):
                out["skipped"].append({"arm": key, "reason": "deadline"})
                print(f"{dl.stamp()} SKIP {key}", flush=True)
                continue
            if target is None:
                bonus, cal_got, cal_log = 0.0, None, []
            else:
                ck = f"{arm}/{tag}"
                if ck in out["calibration"]:
                    bonus = out["calibration"][ck]["bonus"]
                    cal_got = out["calibration"][ck]["realised_A_during_calibration"]
                    cal_log = out["calibration"][ck]["trace"]
                else:
                    bonus, cal_got, cal_log = calibrate(sess_get, policy, cfg, start, lut,
                                                        byte_of, a_mask, target, dl)
                    out["calibration"][ck] = {"target": target, "bonus": bonus,
                                              "realised_A_during_calibration": cal_got,
                                              "trace": cal_log}
                    save()
                    print(f"  {dl.stamp()} calibrated {key}: bonus {bonus:.3f} -> A {cal_got}",
                          flush=True)
            tp = TRACED / f"jb_{arm}_{tag}_{N_EVAL}.json"
            try:
                with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                    s = sess_get()
                    try:
                        traces = resumable(tp, N_EVAL,
                                           lambda i: rollout(s, policy, cfg, start, i, lut,
                                                             byte_of, a_mask, bonus, temp=TEMP))
                    finally:
                        s.close()
            except TimedOut as e:
                out["skipped"].append({"arm": key, "reason": str(e)})
                save()
                continue
            rec = score(key, traces)
            xs = [max(f[0] for f in t.frames) for t in traces]
            hv = [h for t in traces for h in a_hold_onsets(t.frames, (0, 10 ** 9))]
            hs = hold_stats(hv)
            if hv:
                hs = {**hs, "p99": float(np.percentile(hv, 99)), "max": float(max(hv))}
            rec.update({
                "checkpoint": arm, "target_A": target, "logit_bonus": bonus,
                "realised_A": rec["button_marginals"]["rates"]["A"],
                "a_hold_anywhere": hs,
                "x_p90": float(np.percentile(xs, 90)),
                "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                  "rate": float(np.mean([x > v for x in xs]))}
                              for w, v in WALLS.items()},
                "flagpole_episodes": int(sum(
                    1 for t in traces if any(len(f) > 4 and f[4] == 0x05 for f in t.frames))),
                "airborne_minus_expert_pp": (rec["behaviour"]["airborne_fraction"]
                                             - EXPERT_AIRBORNE) * 100,
                "terminator": RB.describe()})
            out["arms"][key] = rec
            save()
            pw = rec["past_wall"]
            print(f"  {dl.stamp()} {key:24s} bonus {bonus:5.2f} A {rec['realised_A']:.3f} "
                  f"airb {rec['behaviour']['airborne_fraction']*100:5.1f}% "
                  f"p2 {rec['clearance']['pipe2']['rate']*100:5.1f}% "
                  f"p3 {pw['pipe3_735']['rate']*100:5.1f}% p4 {pw['pipe4_975']['rate']*100:5.1f}% "
                  f"x_med {rec['x_median']:4.0f} hold p99 {hs.get('p99', 0):5.0f} "
                  f"max {hs.get('max', 0):5.0f} flag {rec['flagpole_episodes']}", flush=True)

    # ---------------- dose-response ----------------
    curve = {}
    for target in TARGETS:
        tag = "unbiased" if target is None else f"A{target:.2f}"
        rows = [out["arms"][f"{a}/{tag}"] for a in ARMS if f"{a}/{tag}" in out["arms"]]
        if not rows:
            continue
        g = lambda f: [r["past_wall"][f]["rate"] * 100 for r in rows]  # noqa: E731
        curve[tag] = {
            "target_A": target, "n_seeds": len(rows),
            "realised_A": [r["realised_A"] for r in rows],
            "realised_A_mean": float(np.mean([r["realised_A"] for r in rows])),
            "logit_bonus": [r["logit_bonus"] for r in rows],
            "pipe2": [r["clearance"]["pipe2"]["rate"] * 100 for r in rows],
            "pipe2_mean": float(np.mean([r["clearance"]["pipe2"]["rate"] * 100 for r in rows])),
            "past_pipe3": g("pipe3_735"), "past_pipe3_mean": float(np.mean(g("pipe3_735"))),
            "past_pipe4": g("pipe4_975"), "past_pipe4_mean": float(np.mean(g("pipe4_975"))),
            "x_median": [r["x_median"] for r in rows],
            "x_median_mean": float(np.mean([r["x_median"] for r in rows])),
            "x_max": [r["x_max"] for r in rows],
            "airborne": [r["behaviour"]["airborne_fraction"] for r in rows],
            "airborne_mean": float(np.mean([r["behaviour"]["airborne_fraction"] for r in rows])),
            "a_hold_p99": [r["a_hold_anywhere"].get("p99") for r in rows],
            "a_hold_max": [r["a_hold_anywhere"].get("max") for r in rows],
            "flagpole": [r["flagpole_episodes"] for r in rows],
            "vs_script_best_fixed_rate_pipe3": [
                r["vs_script_best_fixed_rate"]["per_obstacle"]["pipe3"]["advantage_pp"]
                for r in rows]}
    out["dose_response"] = curve

    if curve:
        base = curve.get("unbiased")
        best = max(curve.values(), key=lambda v: v["past_pipe3_mean"])
        ordered = [curve[k] for k in curve]
        p3s = [v["past_pipe3_mean"] for v in ordered]
        rising = all(p3s[i] <= p3s[i + 1] + 1e-9 for i in range(len(p3s) - 1))
        out["binary_question"] = {
            "unbiased_past_pipe3": base["past_pipe3_mean"] if base else None,
            "best_dose_past_pipe3": best["past_pipe3_mean"],
            "best_dose_target_A": best["target_A"],
            "gain_pp": (best["past_pipe3_mean"] - base["past_pipe3_mean"]) if base else None,
            "high_jump_script_past_pipe3": 57.5,
            "closes_gap": bool(base and best["past_pipe3_mean"] >= 57.5),
            "monotone_rising": bool(rising)}
        bq = out["binary_question"]
        if bq["gain_pp"] is not None and bq["gain_pp"] > 5:
            out["verdict"] = (
                f"**BIASING TOWARD JUMPING HELPS: past pipe 3 rises {base['past_pipe3_mean']:.1f}% -> "
                f"{best['past_pipe3_mean']:.1f}% at target A={best['target_A']} "
                f"({bq['gain_pp']:+.1f} pp).** The high-jump script sits at 57.5%, so this "
                f"{'CLOSES' if bq['closes_gap'] else 'does not fully close'} the gap. **The deficit was a "
                f"marginal and it is free to fix at generation time; the policy's state-conditioning was "
                f"never the problem.**")
        elif bq["gain_pp"] is not None and bq["gain_pp"] < -5:
            out["verdict"] = (
                f"**BIASING TOWARD JUMPING HURTS ({bq['gain_pp']:+.1f} pp at best).** The policy's A rate "
                f"is already near its own optimum and the script wins for a different reason — which "
                f"becomes the question.")
        else:
            out["verdict"] = (
                f"**CLEARANCE IS FLAT IN THE JUMP BIAS: past pipe 3 moves {bq['gain_pp']:+.1f} pp across "
                f"realised A {curve['unbiased']['realised_A_mean']:.2f} to "
                f"{max(v['realised_A_mean'] for v in curve.values()):.2f}.** The script's advantage is "
                f"**not** its jump rate, and something else explains it.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out.get("verdict", "no arms"))
    print(f"\n{'dose':>10s}{'A':>7s}{'airb':>7s}{'p2':>7s}{'>p3':>7s}{'>p4':>7s}"
          f"{'x_med':>7s}{'holdp99':>9s}{'holdmax':>9s}{'flag':>6s}")
    for tag, v in curve.items():
        print(f"{tag:>10s}{v['realised_A_mean']:>7.3f}{v['airborne_mean']*100:>7.1f}"
              f"{v['pipe2_mean']:>7.1f}{v['past_pipe3_mean']:>7.1f}{v['past_pipe4_mean']:>7.1f}"
              f"{v['x_median_mean']:>7.0f}"
              f"{np.mean([h for h in v['a_hold_p99'] if h]):>9.0f}"
              f"{np.mean([h for h in v['a_hold_max'] if h]):>9.0f}"
              f"{sum(v['flagpole']):>6d}")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
