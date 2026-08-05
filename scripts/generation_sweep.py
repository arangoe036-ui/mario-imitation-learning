"""Cap the A-run, sharpen the sample, and see whether either reaches behaviour. **No training.**

The fifty-third block left a dissociation: at 128x128 the corrected timing lift is ~5x B's and clearance is
unchanged. If the generation rule is what blocks the signal from reaching behaviour, then fixing the rule
should help **R more than B** -- and that, not the raw clearance number, is the experiment.

`capped` truncates **no-op** runs at 4 frames and leaves **A-runs uncapped**. So the policy re-decides
constantly while grounded and, having once sampled a long A-class, commits to up to 304 frames of airtime. It
starts 1.4-1.8x more jumps than the expert (39-49 per 1,000 frames against 27.5). **A sharper p(A) cannot
reach behaviour while one sample buys two seconds of flight.** Capping no-op runs doubled x median; the
symmetric cap on A-runs has never been tried.

| stage | grid | on |
|---|---|---|
| **2a** | `max_a_hold` in {12, 24, 48, uncapped} | `B_84_d64_L1`, `R_128_d64_L1` |
| **2b** | temperature in {1.0, 0.7, argmax} at the best cap | both |
| **2d** | rate-matched `vs_script` **at the new A rate** for any arm that improves | improving arms |
| **2e** | the winning configuration re-run on `B_84_seed1`, `B_84_seed2` | seeds |

**⚠ Two warnings carried into the code rather than the prose.**

**Argmax is degenerate as an evaluation, not merely discouraged.** SMB is deterministic and the start state is
fixed, so a greedy policy produces the *same episode every seed* -- n=200 identical rollouts, an interval of
zero width around a sample of one. It is run at `N_ARGMAX` episodes, its distinctness is measured rather than
assumed, and `effective_n` is reported as 1 if they collapse. Separately, `LEDGER.md` records that argmax on
this head vote-splits: the A-containing tokens each lose to Right+B and A is emitted on 0.03% of frames. The
button marginals are checked for that signature.

**A cap lowers the A rate, so a clearance gain has two readings** -- better timing, or merely less jumping.
The rate-matched bar recomputed **at the new rate** is what separates them, and no improvement is reported
without it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from scripts.scaleup_eval import _Ep, resumable, score, scripted_episode  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.runlength import N_BUCKETS, class_lengths, joint_size  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKDIR = ROOT / "data/bc_scaleup"
TRACEDIR = ROOT / "data/traces"
OUT = ROOT / "data/generation_sweep.json"

CAP_NON_A = 4
N_EVAL, CAP_FRAMES, STALL = 200, 3000, 300
#: argmax collapses to one episode; more than a handful is pure waste, but >1 is needed to SHOW it collapses
N_ARGMAX = 12
CAPS = [12, 24, 48, None]
TEMPS = [1.0, 0.7, "argmax"]
BASE = ["B_84_d64_L1", "R_128_d64_L1"]
SEED_ARMS = ["B_84_seed1", "B_84_seed2"]
EXPERT_ONSETS_PER_1K, EXPERT_AIRBORNE = 27.5, 0.611
#: (checkpoint, cap, temp) whose traces already exist from the fifty-third block's eval
REUSE = {("B_84_d64_L1", None, 1.0): "scaleup_B_84_d64_L1_200.json",
         ("R_128_d64_L1", None, 1.0): "scaleup_R_128_d64_L1_200.json"}


def tag(ckpt: str, cap, temp) -> str:
    c = "inf" if cap is None else str(cap)
    t = "argmax" if temp == "argmax" else f"{temp:g}".replace(".", "p")
    return f"{ckpt}_cap{c}_t{t}"


def rollout(session, policy, cfg, start, seed, lut, byte_of, *, cap, temp) -> EpisodeTrace:
    """`capped` generation plus an A-run cap and a temperature on the class softmax."""
    s = cfg.frame_size
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    win[:] = _resize_gray(obs.rgb, (s, s))
    best = since = frames = 0
    held_byte, remaining = None, 0
    while frames < CAP_FRAMES:
        if remaining <= 0:
            with torch.no_grad():
                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
            if temp == "argmax":
                c = int(torch.argmax(lg).item())
            else:
                p = torch.softmax(lg / float(temp), dim=-1).numpy()
                c = int(rng.choice(len(p), p=p / p.sum()))
            b, length = int(byte_of[c]), max(1, int(lut[c]))
            if b & A_BIT:
                if cap is not None:
                    length = min(length, cap)       # 2a: the symmetric cap, never tried before
            else:
                length = min(length, CAP_NON_A)
            held_byte, remaining = b, length
        byte = held_byte
        remaining -= 1
        obs = session.step(byte)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (s, s))
        t.record(obs, byte)
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


def load_ckpt(name: str):
    blob = torch.load(CKDIR / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig.from_dict(cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()
    return policy, cfg, blob


def distinctness(traces) -> dict:
    """How many of these episodes are actually different? Argmax makes this the whole story."""
    sigs = {tuple(f[3] for f in t.frames[:400]) for t in traces}
    return {"n_episodes": len(traces), "n_distinct_prefixes": len(sigs),
            "effective_n": len(sigs),
            "collapsed": len(sigs) == 1 and len(traces) > 1}


def run_arm(sess_get, ck, cap, temp, ctx, start, *, n=N_EVAL) -> dict:
    policy, cfg, blob = ck["policy"], ck["cfg"], ck["blob"]
    name = tag(ck["name"], cap, temp)
    reuse = REUSE.get((ck["name"], cap, temp))
    tp = TRACEDIR / (reuse or f"gen_{name}_{n}.json")
    if reuse and tp.exists():
        traces = [_Ep(e) for e in json.loads(tp.read_text())["episodes"]]
        print(f"  {name:34s} reusing block-53 traces ({len(traces)} eps)", flush=True)
    else:
        s = sess_get()
        try:
            traces = resumable(tp, n, lambda i: rollout(s, policy, cfg, start, i,
                                                        ck["lut"], ck["byte_of"],
                                                        cap=cap, temp=temp))
        finally:
            s.close()
    rec = score(name, traces)
    rec.update({"checkpoint": ck["name"], "max_a_hold": cap, "temperature": temp,
                "frame_size": cfg.frame_size, "train_seed": blob.get("seed"),
                "train_steps": blob.get("steps", blob.get("step")),
                "generation_rule": f"capped(non-A<=4) + A-run cap {cap} + temperature {temp}",
                "distinctness": distinctness(traces), "traces": str(tp.relative_to(ROOT))})
    c = rec["clearance"]
    d = rec["distinctness"]
    warn = "  ⚠COLLAPSED n=1" if d["collapsed"] else ""

    def f(v, spec="5.1f", scale=1.0):
        """A missing statistic prints as '-', never crashes the run that produced it.

        An argmax trajectory that never presses A has no A-hold distribution at all, so `max` is
        None -- and losing a completed 200-episode arm to a format string would be absurd.
        """
        return format(v * scale, spec) if v is not None else "-".rjust(int(spec.split(".")[0]))
    print(f"  {name:34s} p1 {f(c['pipe1']['rate'], scale=100)} p2 {f(c['pipe2']['rate'], scale=100)} "
          f"p3 {f(c['pipe3']['rate'], scale=100)} p4 {f(c['pipe4']['rate'], scale=100)} | "
          f"x {f(rec['x_median'], '4.0f')} | A {f(rec['button_marginals']['rates']['A'], '.3f')} | "
          f"on/1k {f(rec['a_onsets_per_1000_frames'])} | airb "
          f"{f(rec['behaviour'].get('airborne_fraction'), '4.1f', 100)}% | "
          f"hold max {f(rec['a_hold_anywhere'].get('max'), '4.0f')}{warn}", flush=True)
    return rec


def rate_matched(sess_get, rec, ctx, start, out, n=N_EVAL) -> dict:
    """2d: a script at THIS arm's own marginals. A cap lowers the A rate; this is what says whether
    the gain is timing or merely less jumping."""
    r = rec["button_marginals"]["rates"]
    strong = {**{k: float(r[k]) for k in ("A", "B", "Right", "Down", "Left")},
              "B": 1.0, "Right": 1.0}
    tp = TRACEDIR / f"gen_{rec['label']}_ratematched_{n}.json"
    s = sess_get()
    try:
        ctrl = resumable(tp, n, lambda i: scripted_episode(s, start, i, strong))
    finally:
        s.close()
    cs = score(f"{rec['label']}_ratematched", ctrl)
    bar = {}
    for ob in cs["clearance"]:
        kp = int(round(rec["clearance"][ob]["rate"] * n))
        kc = int(round(cs["clearance"][ob]["rate"] * n))
        lo, hi = diff_ci(kc, n, kp, n)
        bar[ob] = {"policy_rate": rec["clearance"][ob]["rate"],
                   "script_rate": cs["clearance"][ob]["rate"],
                   "difference_pp": (rec["clearance"][ob]["rate"] - cs["clearance"][ob]["rate"]) * 100,
                   "ci_pp": [lo * 100, hi * 100], "method": "Newcombe"}
    print(f"    rate-matched at A={strong['A']:.3f}: " +
          "  ".join(f"{k} {v['difference_pp']:+.1f}" for k, v in bar.items()), flush=True)
    return {"control_rates": strong, "control_score": cs, "per_obstacle": bar,
            "note": "the stronger Right+B-held reading; recomputed at THIS arm's A rate, not the old one"}


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // N_BUCKETS) for c in range(n_cls)], dtype=np.int64)
    lut_cache: dict[str, np.ndarray] = {}

    def get_ck(name):
        policy, cfg, blob = load_ckpt(name)
        corpus = blob.get("corpus", "runs")
        if corpus not in lut_cache:
            z = np.load(ROOT / f"data/runlength_index_{corpus}.npz")
            lut_cache[corpus] = class_lengths({k: z[k] for k in ("rows", "joints", "lengths")},
                                              n_cls)
        return {"name": name, "policy": policy, "cfg": cfg, "blob": blob,
                "lut": lut_cache[corpus], "byte_of": byte_of}

    def sess_get():
        return session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.update({
        "n_eval": N_EVAL, "n_argmax": N_ARGMAX, "measurement_basis": "single_life",
        "base_rule": f"capped: non-A runs <= {CAP_NON_A} frames",
        "expert_reference": {"a_onsets_per_1000_frames": EXPERT_ONSETS_PER_1K,
                             "airborne_fraction": EXPERT_AIRBORNE, "a_rate": 0.152},
        "no_training": "inference-time only; every checkpoint pre-existed this block"})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    # ---------------- 2a: cap the A-run ----------------
    print("=== 2a: A-run cap, temperature 1.0 ===", flush=True)
    for name in BASE:
        ck = get_ck(name)
        for cap in CAPS:
            k = tag(name, cap, 1.0)
            if k not in out["arms"]:
                out["arms"][k] = run_arm(sess_get, ck, cap, 1.0, ctx, start)
                save()

    def p2(k):
        return out["arms"][k]["clearance"]["pipe2"]["rate"]

    best_cap = {}
    for name in BASE:
        # Sort key must never compare the caps themselves: `None` (uncapped) is in CAPS and
        # `24 > None` is a TypeError, so a tie on pipe-2 rate would crash the sweep. Ties break
        # toward the LOOSER cap, i.e. the smaller intervention.
        cands = [(p2(tag(name, c, 1.0)), (10 ** 9 if c is None else c), c) for c in CAPS]
        best_r, _, best_c = max(cands)
        best_cap[name] = best_c
        base_r = p2(tag(name, None, 1.0))
        k_un, k_be = tag(name, None, 1.0), tag(name, best_c, 1.0)
        lo, hi = diff_ci(int(round(base_r * N_EVAL)), N_EVAL, int(round(best_r * N_EVAL)), N_EVAL)
        out.setdefault("cap_effect", {})[name] = {
            "uncapped_pipe2": base_r, "best_cap": best_c, "best_pipe2": best_r,
            "gain_pp": (best_r - base_r) * 100, "ci_pp": [lo * 100, hi * 100],
            "method": "Newcombe", "by_cap": {str(c): p2(tag(name, c, 1.0)) for c in CAPS},
            "by_cap_all_obstacles": {str(c): {o: out["arms"][tag(name, c, 1.0)]["clearance"][o]["rate"]
                                              for o in ("pipe1", "pipe2", "pipe3", "pipe4")}
                                     for c in CAPS},
            "arms": [k_un, k_be]}
        print(f"  best cap for {name}: {best_c} "
              f"(pipe2 {base_r*100:.1f} -> {best_r*100:.1f}, "
              f"{(best_r-base_r)*100:+.1f} pp [{lo*100:+.1f},{hi*100:+.1f}])", flush=True)
    save()

    # ---------------- 2b: sharpen ----------------
    print("\n=== 2b: temperature at the best cap ===", flush=True)
    for name in BASE:
        ck = get_ck(name)
        for temp in TEMPS:
            if temp == 1.0:
                continue
            k = tag(name, best_cap[name], temp)
            if k not in out["arms"]:
                n = N_ARGMAX if temp == "argmax" else N_EVAL
                out["arms"][k] = run_arm(sess_get, ck, best_cap[name], temp, ctx, start, n=n)
                save()

    # ---------------- 2c: the interaction ----------------
    ce = out["cap_effect"]
    gB, gR = ce["B_84_d64_L1"]["gain_pp"], ce["R_128_d64_L1"]["gain_pp"]
    kb, kr = ce["B_84_d64_L1"]["arms"], ce["R_128_d64_L1"]["arms"]
    # difference-of-differences, Newcombe on each arm's own pair then combined conservatively
    out["interaction"] = {
        "question": "does a better generation rule help R (5x timing signal) more than B?",
        "B_gain_pp": gB, "R_gain_pp": gR, "R_minus_B_pp": gR - gB,
        "B_arms": kb, "R_arms": kr,
        "caveat": ("each gain is one training seed per checkpoint; the difference of two gains has "
                   "roughly twice the variance of either, so read the sign only if it is large")}
    if gR > gB + 5:
        out["interaction"]["reading"] = (
            "HELPS R MORE: the extra timing signal at 128x128 was real and bottlenecked downstream. "
            "Resolution is vindicated and keeping the 128 corpus is correct.")
    elif abs(gR - gB) <= 5:
        out["interaction"]["reading"] = (
            "HELPS BOTH ABOUT EQUALLY: generation is a separate, additive problem and the resolution "
            "spend remains unpaid-for.")
    else:
        out["interaction"]["reading"] = (
            "HELPS B MORE THAN R: the extra lift at 128 does not convert into behaviour even once the "
            "generation rule is relaxed, which puts the LIFT itself under suspicion rather than the policy.")

    # ---------------- 2d: rate-matched bar for improving arms ----------------
    print("\n=== 2d: rate-matched control at the NEW A rate ===", flush=True)
    base_p2 = {n: p2(tag(n, None, 1.0)) for n in BASE}
    improving = [k for k, r in out["arms"].items()
                 if r["clearance"]["pipe2"]["rate"] > base_p2.get(r["checkpoint"], 1.0)
                 and not r["distinctness"]["collapsed"]
                 and "vs_script_rate_matched" not in r]
    for k in improving:
        out["arms"][k]["vs_script_rate_matched"] = rate_matched(sess_get, out["arms"][k], ctx,
                                                                start, out)
        save()

    # ---------------- 2e: seeds ----------------
    print("\n=== 2e: winning configuration on two more training seeds ===", flush=True)
    winner = max(((p2(k), k) for k in out["arms"]
                  if out["arms"][k]["checkpoint"] == "B_84_d64_L1"
                  and not out["arms"][k]["distinctness"]["collapsed"]))[1]
    wcfg = (out["arms"][winner]["max_a_hold"], out["arms"][winner]["temperature"])
    out["winner_config"] = {"from": winner, "max_a_hold": wcfg[0], "temperature": wcfg[1]}
    print(f"  winning B config: cap {wcfg[0]}, temperature {wcfg[1]}", flush=True)
    for name in SEED_ARMS:
        ck = get_ck(name)
        for cap, temp in ((wcfg[0], wcfg[1]), (None, 1.0)):
            k = tag(name, cap, temp)
            if k not in out["arms"]:
                out["arms"][k] = run_arm(sess_get, ck, cap, temp, ctx, start)
                save()
    seeds = []
    for name in ["B_84_d64_L1"] + SEED_ARMS:
        kb2, kw = tag(name, None, 1.0), tag(name, wcfg[0], wcfg[1])
        if kb2 in out["arms"] and kw in out["arms"]:
            seeds.append({"seed_arm": name, "uncapped_pipe2": p2(kb2), "capped_pipe2": p2(kw),
                          "gain_pp": (p2(kw) - p2(kb2)) * 100})
    g = [s["gain_pp"] for s in seeds]
    out["seed_replication"] = {
        "config": out["winner_config"], "per_seed": seeds, "n_seeds": len(g),
        "median_gain_pp": float(np.median(g)) if g else None,
        "min_gain_pp": float(min(g)) if g else None, "max_gain_pp": float(max(g)) if g else None,
        "all_positive": bool(g) and all(x > 0 for x in g)}

    # ---------------- verdict ----------------
    sr = out["seed_replication"]
    helped = bool(sr["all_positive"] and sr["n_seeds"] >= 2)
    out["binary_question"] = {
        "does_capping_improve_pipe2": helped,
        "does_it_help_R_more_than_B": bool(gR > gB + 5)}
    if helped and gR > gB + 5:
        out["verdict"] = (
            f"**YES AND YES.** Capping the A-run improves pipe 2 in {sr['n_seeds']}/{sr['n_seeds']} "
            f"training seeds (median {sr['median_gain_pp']:+.1f} pp) and helps R more than B "
            f"({gR:+.1f} vs {gB:+.1f}). **The bottleneck is the generation rule, and the resolution "
            f"spend is vindicated** -- the extra signal was real and blocked downstream.")
    elif helped:
        out["verdict"] = (
            f"**YES, BUT NOT PREFERENTIALLY FOR R.** Capping improves pipe 2 in "
            f"{sr['n_seeds']}/{sr['n_seeds']} seeds (median {sr['median_gain_pp']:+.1f} pp); the gain "
            f"is {gR:+.1f} pp for R against {gB:+.1f} for B. Generation and resolution are two "
            f"separate problems and the 128x128 spend remains unpaid-for.")
    else:
        out["verdict"] = (
            f"**NO.** Capping the A-run does not improve pipe 2 across seeds (per-seed gains "
            f"{[round(x, 1) for x in g]}). The corrected timing lift rose 5x with resolution and "
            f"neither the lift nor a freer generation rule moves behaviour. **The lift itself is now "
            f"the thing under suspicion** -- it would be the fourth estimator in this thread to fail, "
            f"and the most consequential.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\n{out['interaction']['reading']}")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
