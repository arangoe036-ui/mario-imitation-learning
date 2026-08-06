"""§3e: the missing baseline — a RUN-LENGTH script. Does learning add anything beyond the encoding?

Block 60 reported "the policy beats a rate-matched script by 80 pp at pipe 2". **That comparison is
representational, not evidence of state-conditioning.** The matched script draws A independently per frame at
0.338, so:

| | |
|---|---|
| P(a 12-frame hold) — pipe 2's requirement | 0.338¹² = **2.2 × 10⁻⁶** |
| mean geometric hold | **1.51 frames** |
| P(a 14-frame hold) — pipe 3's median requirement | **2.5 × 10⁻⁷** |

**It physically cannot hold A long enough. Its 0% past pipe 3 is arithmetic, not a measurement of skill.**
This is the p^L problem already in `LEDGER.md`.

The honest control matches the policy **on the action representation** rather than on per-frame button rates:
sample **(combo, hold-length) tokens** from the policy's own *token* marginals, with **Right+B forced on** and
A-holds drawn from **the policy's own hold distribution**. That single arm also subsumes the Right/B
hypothesis from block 60, whose "matched Right+B held" arm still sampled A per frame and so isolated nothing.

**If the policy still beats this, the +80 pp is real learning. If it does not, the project's central positive
result is a statement about run-length tokens** — still a genuine finding, and one we would then be reporting
correctly rather than accidentally.

The token marginals are measured from the policy's own traces, so the control is matched to the arm it is
compared against rather than to an older checkpoint.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.scaleup_eval import resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/runlength_script_control.json"
TRACED = ROOT / "data/traces"

POLICY_ARMS = [f"PK32_84_s{i}" for i in range(3)]
N = 200
LOCOMOTION = 0x82           # Right+B forced on, as the strongest reading
WALLS = {"goomba_320": 320, "pipe1_470": 470, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562}
ARM_BUDGET_S = 15 * 60


def token_stats(traces):
    """The policy's own (combo, hold) distribution, measured from its traces."""
    runs = collections.Counter()
    a_holds, non_a_holds = [], []
    for t in traces:
        bs = [f[3] for f in t.frames]
        i = 0
        while i < len(bs):
            j = i
            while j < len(bs) and bs[j] == bs[i]:
                j += 1
            runs[int(bs[i])] += 1
            (a_holds if (bs[i] & A_BIT) else non_a_holds).append(j - i)
            i = j
    tot = sum(runs.values())
    return {"combo_probs": {int(k): v / tot for k, v in runs.items()},
            "a_hold_dist": a_holds, "non_a_hold_dist": non_a_holds,
            "n_runs": tot,
            "mean_a_hold": float(np.mean(a_holds)) if a_holds else 0.0}


def rl_script_episode(sess, start, seed, stats):
    """Sample (combo, hold) tokens from the policy's own token marginals. Right+B forced on."""
    rng = np.random.default_rng(seed)
    combos = list(stats["combo_probs"].keys())
    probs = np.array([stats["combo_probs"][c] for c in combos], dtype=float)
    probs = probs / probs.sum()
    a_hold = np.asarray(stats["a_hold_dist"]) if stats["a_hold_dist"] else np.array([1])
    n_hold = np.asarray(stats["non_a_hold_dist"]) if stats["non_a_hold_dist"] else np.array([1])
    t = EpisodeTrace(seed=seed)
    obs = sess.reset(start.frame)
    best = since = frames = 0
    while frames < RB.CAP_FRAMES:
        c = int(rng.choice(combos, p=probs))
        byte = c | LOCOMOTION
        hold = int(rng.choice(a_hold if (c & A_BIT) else n_hold))
        for _ in range(max(1, hold)):
            obs = sess.step(byte)
            t.record(obs, byte)
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
            if frames >= RB.CAP_FRAMES:
                break
    return t


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    from scripts.scaleup_eval import _Ep

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("skipped", [])
    out.update({"n": N, "terminator": RB.describe(),
                "measurement_basis": "single_life_from_level_start",
                "control": ("run-length script: (combo, hold) tokens drawn from the POLICY'S OWN token "
                            "marginals, Right+B forced on, A-holds from the policy's own hold "
                            "distribution"),
                "why": ("the per-frame matched script cannot hold A for 12 frames (0.338^12 = 2.2e-6), "
                        "so its 0% past pipe 3 is arithmetic; this control removes that confound and "
                        "subsumes block 60's Right/B hypothesis")})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    for arm in POLICY_ARMS:
        key = f"rlscript_matched_to_{arm}"
        if key in out["arms"]:
            continue
        src = TRACED / f"jb_{arm}_unbiased_{N}.json"
        if not src.exists():
            continue
        pol_traces = [_Ep(e) for e in json.loads(src.read_text())["episodes"]]
        stats = token_stats(pol_traces)
        if not dl.can_afford(150):
            out["skipped"].append({"arm": key, "reason": "deadline"})
            continue
        tp = TRACED / f"rlscript_{arm}_{N}.json"
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                s = sess_get()
                try:
                    traces = resumable(tp, N, lambda i: rl_script_episode(s, start, i, stats))
                finally:
                    s.close()
        except TimedOut as e:
            out["skipped"].append({"arm": key, "reason": str(e)})
            save()
            continue
        rec = score(key, traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        pol = score(f"policy_{arm}", pol_traces)
        pxs = [max(f[0] for f in t.frames) for t in pol_traces]
        rec.update({
            "matched_to": arm, "terminator": RB.describe(),
            "token_stats": {"n_runs": stats["n_runs"], "mean_a_hold": stats["mean_a_hold"],
                            "n_distinct_combos": len(stats["combo_probs"])},
            "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                              "rate": float(np.mean([x > v for x in xs]))}
                          for w, v in WALLS.items()},
            "policy_past_wall": {w: {"k": int(sum(1 for x in pxs if x > v)), "n": len(pxs),
                                     "rate": float(np.mean([x > v for x in pxs]))}
                                 for w, v in WALLS.items()}})
        out["arms"][key] = rec
        save()
        pw, pp = rec["past_wall"], rec["policy_past_wall"]
        print(f"  {dl.stamp()} {arm}: RL-script p2 {pw['pipe2_630']['rate']*100:5.1f}% "
              f"p3 {pw['pipe3_735']['rate']*100:5.1f}% p4 {pw['pipe4_975']['rate']*100:5.1f}% "
              f"| policy p2 {pp['pipe2_630']['rate']*100:5.1f}% "
              f"p3 {pp['pipe3_735']['rate']*100:5.1f}% p4 {pp['pipe4_975']['rate']*100:5.1f}% "
              f"| mean A-hold {stats['mean_a_hold']:.1f}", flush=True)

    rows = list(out["arms"].values())
    if rows:
        comp = {}
        for w in WALLS:
            sc = [r["past_wall"][w]["rate"] * 100 for r in rows]
            po = [r["policy_past_wall"][w]["rate"] * 100 for r in rows]
            ks = int(round(np.mean(sc) / 100 * N * len(sc)))
            kp = int(round(np.mean(po) / 100 * N * len(po)))
            lo, hi = diff_ci(ks, N * len(sc), kp, N * len(po))
            comp[w] = {"rl_script": sc, "rl_script_mean": float(np.mean(sc)),
                       "policy": po, "policy_mean": float(np.mean(po)),
                       "policy_minus_script_pp": float(np.mean(po) - np.mean(sc)),
                       "ci_pp_pooled_episodes": [lo * 100, hi * 100],
                       "ci_caveat": "pooled over episodes; the seed is the unit -- read per-seed"}
        out["comparison"] = comp
        p2 = comp["pipe2_630"]["policy_minus_script_pp"]
        p3 = comp["pipe3_735"]["policy_minus_script_pp"]
        out["verdict"] = (
            f"**Against a RUN-LENGTH script matched on the action representation, the policy is "
            f"{p2:+.1f} pp at pipe 2 and {p3:+.1f} pp past pipe 3.** Block 60's +80 pp was measured "
            f"against a per-frame script that physically could not hold A for 12 frames, so that "
            f"figure is a statement about the ENCODING. "
            + ("**The policy still beats the honest control, so there is real learning beyond the "
               "encoding.**" if p2 > 5 else
               "**The policy does NOT beat the honest control, so the project's central positive "
               "result is a statement about run-length tokens rather than about learning.**"))
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out.get("verdict", "no arms"))
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
