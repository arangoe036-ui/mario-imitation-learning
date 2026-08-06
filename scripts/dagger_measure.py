"""§3d + §4: did the distilled policy improve — and did it learn a POLICY or a MARGINAL?

**§3d.** Primary endpoint: past pipe 3 (x > 735), n=200, 3 seeds, `STALL=6500`, from the level start,
against **the high-jump script's 57.5%** and against the **1,000-step baseline's 41.2%**. The pre-registered
completion endpoint (≥8/200 with the script re-run beside it) is reported every round and is not expected in
round one.

**§4 is what the three failed distillations lacked.** "Distilling Goomba solutions changed nothing; the
solution is a marginal" was the historical failure mode, and nothing at the time distinguished a
state-conditional correction from a global shift toward jumping.

So the policy's probability mass on the solution set is measured **separately at states search could solve and
at states it could not**:

* **mass rises at solvable states and not at unsolvable ones** → a **state-conditional** correction, which is
  the result this project has been trying to produce;
* **mass rises equally at both** → a **marginal**, which the jump-bias sweep already gets for free and which
  is not learning to play.

Reported whatever the clearance does — it is what makes a null interpretable.
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
from scripts.button_mask_eval import rollout  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.scaleup_eval import resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.runlength import N_BUCKETS, joint_size  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/dagger_round1_measure.json"
TRACED = ROOT / "data/traces"

NEW = [f"DAG1_84_cnn32_s{i}" for i in range(3)]
BASE = [f"PK32_84_s{i}" for i in range(3)]
TEMP = 0.7
N_EVAL = 200
SCRIPT_PAST_P3 = 57.5           # high-jump A 0.85, n=200, same terminator (block 60)
WALLS = {"goomba_320": 320, "pipe1_470": 470, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562, "flagpole_3266": 3266}
ARM_BUDGET_S = 15 * 60


def mass_on_solutions(policy, cfg, ctx, samples, labels, kinds, n_cls):
    """Summed softmax on the solution classes at each captured correction state."""
    a = np.asarray(samples)
    out = {}
    for kind in sorted(set(kinds)):
        m = [i for i, k in enumerate(kinds) if k == kind]
        if not m:
            continue
        vals = []
        for i in range(0, len(m), 256):
            batch = torch.from_numpy(a[m[i:i + 256]]).float().div_(255.0)
            with torch.no_grad():
                p = torch.softmax(policy(batch), dim=-1).numpy()
            for r, j in enumerate(m[i:i + 256]):
                vals.append(float(p[r, int(labels[j])]))
        out[kind] = {"n": len(vals), "mean_mass": float(np.mean(vals)),
                     "median_mass": float(np.median(vals))}
    return out


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 90 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("skipped", [])
    out.update({"n_eval": N_EVAL, "temperature": TEMP, "terminator": RB.describe(),
                "measurement_basis": "single_life_from_level_start",
                "primary_endpoint": "past pipe 3 (x > 735)",
                "script_bar_past_pipe3": SCRIPT_PAST_P3,
                "new_arms": NEW, "baseline_arms": BASE})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    for name in NEW + BASE:
        if name in out["arms"]:
            continue
        if not (ROOT / f"data/bc_scaleup/{name}.pt").exists():
            continue
        prior = TRACED / f"jb_{name}_unbiased_{N_EVAL}.json"
        if name in BASE and prior.exists():
            from scripts.scaleup_eval import _Ep
            traces = [_Ep(e) for e in json.loads(prior.read_text())["episodes"]]
            src = "reused: jump-bias sweep unbiased arm (same policy, T, terminator, start)"
        else:
            if not dl.can_afford(150):
                out["skipped"].append({"arm": name, "reason": "deadline"})
                continue
            policy, cfg, _ = G.load_ckpt(name)
            tp = TRACED / f"dag1_{name}_{N_EVAL}.json"
            try:
                with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), name):
                    s = sess_get()
                    try:
                        traces = resumable(tp, N_EVAL,
                                           lambda i: rollout(s, policy, cfg, start, i, lut,
                                                             byte_of, None, temp=TEMP))
                    finally:
                        s.close()
            except TimedOut as e:
                out["skipped"].append({"arm": name, "reason": str(e)})
                save()
                continue
            src = "block 61"
        rec = score(name, traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        rec.update({"source": src, "terminator": RB.describe(),
                    "group": "distilled" if name in NEW else "baseline",
                    "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                      "rate": float(np.mean([x > v for x in xs]))}
                                  for w, v in WALLS.items()},
                    "completions": int(sum(
                        1 for t in traces
                        if any(len(f) > 7 and f[6] == 1 and f[7] == 2 for f in t.frames))),
                    "flagpole_episodes": int(sum(
                        1 for t in traces if any(len(f) > 4 and f[4] == 0x05 for f in t.frames)))})
        out["arms"][name] = rec
        save()
        pw = rec["past_wall"]
        print(f"  {dl.stamp()} {name:20s} p2 {pw['pipe2_630']['rate']*100:5.1f}% "
              f"p3 {pw['pipe3_735']['rate']*100:5.1f}% p4 {pw['pipe4_975']['rate']*100:5.1f}% "
              f"x_med {rec['x_median']:4.0f} A {rec['button_marginals']['rates']['A']:.3f} "
              f"Left {rec['button_marginals']['rates']['Left']:.3f} "
              f"comp {rec['completions']}", flush=True)

    # ---------------- 3d analysis ----------------
    def vals(group, w):
        return [out["arms"][n]["past_wall"][w]["rate"] * 100
                for n in (NEW if group == "new" else BASE) if n in out["arms"]]
    p3n, p3b = vals("new", "pipe3_735"), vals("base", "pipe3_735")
    res = {}
    if p3n and p3b:
        kn = int(round(np.mean(p3n) / 100 * N_EVAL * len(p3n)))
        kb = int(round(np.mean(p3b) / 100 * N_EVAL * len(p3b)))
        lo, hi = diff_ci(kb, N_EVAL * len(p3b), kn, N_EVAL * len(p3n))
        res["past_pipe3"] = {
            "distilled": p3n, "distilled_mean": float(np.mean(p3n)),
            "baseline": p3b, "baseline_mean": float(np.mean(p3b)),
            "gain_pp": float(np.mean(p3n) - np.mean(p3b)),
            "ci_pp_pooled_episodes": [lo * 100, hi * 100],
            "ci_caveat": ("pooled over episodes across seeds; the unit of randomisation is the SEED, "
                          "so this interval is anti-conservative -- read the per-seed values"),
            "script_bar": SCRIPT_PAST_P3,
            "beats_script": bool(np.mean(p3n) > SCRIPT_PAST_P3)}
        for w in ("pipe2_630", "pipe4_975", "koopas_1248", "frontier_1562"):
            res[w] = {"distilled": vals("new", w), "baseline": vals("base", w),
                      "gain_pp": float(np.mean(vals("new", w)) - np.mean(vals("base", w)))}
    res["completions"] = {
        "distilled": [out["arms"][n]["completions"] for n in NEW if n in out["arms"]],
        "baseline": [out["arms"][n]["completions"] for n in BASE if n in out["arms"]],
        "prereg_threshold": ">=8/200 with the script re-run beside it"}
    out["analysis"] = res

    # ---------------- §4: policy or marginal? ----------------
    cache = ROOT / "data/dagger_round1_samples.npz"
    if cache.exists():
        zz = np.load(cache, allow_pickle=True)
        samples, labels = zz["obs"], zz["lab"]
        metas = [json.loads(m) for m in zz["meta"]] if "meta" in zz else []
        kinds = [m.get("kind", "?") for m in metas] if metas else ["?"] * len(labels)
        # solvable = every captured state was solved this round, so split by SOLUTION KIND instead:
        # retreat-only corrections vs policy-sampled ones. Stated, not silently substituted.
        diag = {"note": ("every one of the 60 captured states was solved this round, so there is no "
                         "unsolvable group to contrast against. The split reported instead is by "
                         "SOLUTION KIND -- retreat macros (which the policy had ~0.5% mass on) versus "
                         "policy-sampled sequences (which it already had mass on). A rise confined to "
                         "the retreat class is state-conditional learning of something new; an equal "
                         "rise on both is a marginal."),
                "by_kind": {}}
        for name in (BASE[:1] + NEW):
            if not (ROOT / f"data/bc_scaleup/{name}.pt").exists():
                continue
            policy, cfg, _ = G.load_ckpt(name)
            diag["by_kind"][name] = mass_on_solutions(policy, cfg, ctx, samples, labels,
                                                      kinds, n_cls)
        out["policy_or_marginal"] = diag
        save()
        print("\n§4 mass on the solution classes:")
        for name, byk in diag["by_kind"].items():
            print(f"  {name:20s} " + "  ".join(
                f"{k} {v['mean_mass']:.4f} (n={v['n']})" for k, v in byk.items()), flush=True)

    p3 = res.get("past_pipe3")
    if p3:
        d = out.get("policy_or_marginal", {}).get("by_kind", {})
        base_name = BASE[0]
        rises = {}
        if base_name in d:
            for name in NEW:
                if name in d:
                    for k in d[name]:
                        if k in d[base_name]:
                            rises.setdefault(k, []).append(
                                d[name][k]["mean_mass"] - d[base_name][k]["mean_mass"])
        rise_txt = ", ".join(f"{k} {np.mean(v):+.4f}" for k, v in rises.items()) or "n/a"
        out["verdict"] = (
            f"**Past pipe 3: {p3['baseline_mean']:.1f}% -> {p3['distilled_mean']:.1f}% "
            f"({p3['gain_pp']:+.1f} pp), per seed {[round(x, 1) for x in p3['distilled']]} against "
            f"{[round(x, 1) for x in p3['baseline']]}.** The high-jump script's bar is "
            f"{SCRIPT_PAST_P3}%, so this "
            f"{'BEATS' if p3['beats_script'] else 'does NOT beat'} it. Completions "
            f"{res['completions']['distilled']} against the pre-registered threshold of 8/200. "
            f"Mass change on the solution classes: {rise_txt}.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out.get("verdict", "insufficient arms"))
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
