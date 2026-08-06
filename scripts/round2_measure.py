"""§3d + §4 + §5 at TEN PAIRED SEEDS: is the correction state-conditional, and does it beat the baseline?

Round one's headlines all sat at or above their own design floor: at three paired seeds the smallest
attainable two-sided p on a paired sign-flip permutation is **2/2³ = 0.250**. At ten it is **2/2¹⁰ =
0.00195**. Training costs 20 seconds, so every arm here is ten seeds.

Three things measured on the same rollouts:

* **§3d** — past pipe 3 and every other wall, round 2 against the 1,000-step baseline, paired by seed.
* **§4** — the diagnostic that separates conditional from marginal *directly*: the **Left rate INSIDE the
  correction windows versus OUTSIDE them**. Round one's mass table was near-circular (retreat mass rose 8×
  while 98% of the mix was retreats); what actually carried "marginal" was the 11× global Left rate. Rising
  only inside the windows is state-conditional; rising everywhere is still a marginal.
* **§5** — the run-length script control re-run at ten paired seeds, because it is now the project's central
  positive claim and at three seeds it sat at its floor with one seed negative at pipe 3.

Tests are **paired sign-flip permutation over seeds**, exact, with the attainable floor stated.
"""
from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.button_mask_eval import rollout  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.runlength_script_control import rl_script_episode, token_stats  # noqa: E402
from scripts.scaleup_eval import _Ep, resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/round2_measure.json"
TRACED = ROOT / "data/traces"

N_SEEDS = 10
N_EVAL = 200
TEMP = 0.7
LEFT_BIT = NES_BUTTON_BITS["Left"]
WALLS = {"goomba_320": 320, "pipe1_470": 470, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562}
#: §4: x-windows where corrections were collected, from the round-1 failure histogram
CORRECTION_CENTRES = [288, 720, 912, 1216, 1504, 800]
WINDOW = 96
ARM_BUDGET_S = 12 * 60


def sign_flip_p(diffs):
    """Exact two-sided paired sign-flip permutation over seeds."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs(float(np.mean(d * np.array(signs)))) >= obs - 1e-12:
            cnt += 1
    return cnt / (2 ** n), 2 ** n, 2.0 / (2 ** n)


def left_inside_outside(traces):
    """§4: the Left rate inside correction windows versus outside, and per wall."""
    ins = out_ = ins_l = out_l = 0
    per_wall = collections.defaultdict(lambda: [0, 0])
    for t in traces:
        for f in t.frames:
            x, b = f[0], f[3]
            near = None
            for c in CORRECTION_CENTRES:
                if abs(x - c) <= WINDOW:
                    near = c
                    break
            left = 1 if (b & LEFT_BIT) else 0
            if near is not None:
                ins += 1
                ins_l += left
                per_wall[near][0] += 1
                per_wall[near][1] += left
            else:
                out_ += 1
                out_l += left
    return {"inside_frames": ins, "outside_frames": out_,
            "left_rate_inside": (ins_l / ins) if ins else None,
            "left_rate_outside": (out_l / out_) if out_ else None,
            "per_wall_left_rate": {str(c): (v[1] / v[0]) if v[0] else None
                                   for c, v in sorted(per_wall.items())}}


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 200 * 60)
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
    out.update({"n_seeds": N_SEEDS, "n_eval": N_EVAL, "temperature": TEMP,
                "terminator": RB.describe(),
                "measurement_basis": "single_life_from_level_start",
                "test": "paired sign-flip permutation over seeds; floor 2/2^n",
                "correction_windows": {"centres": CORRECTION_CENTRES, "half_width": WINDOW}})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    def eval_policy(name, tag):
        key = f"{tag}/{name}"
        if key in out["arms"]:
            return
        if not (ROOT / f"data/bc_scaleup/{name}.pt").exists():
            return
        cached = TRACED / f"jb_{name}_unbiased_{N_EVAL}.json"
        if tag == "baseline" and cached.exists():
            traces = [_Ep(e) for e in json.loads(cached.read_text())["episodes"]]
            src = "reused: jump-bias unbiased arm"
        else:
            if not dl.can_afford(150):
                out["skipped"].append({"arm": key, "reason": "deadline"})
                return
            policy, cfg, _ = G.load_ckpt(name)
            tp = TRACED / f"r2_{name}_{N_EVAL}.json"
            try:
                with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                    s = sess_get()
                    try:
                        traces = resumable(tp, N_EVAL,
                                           lambda i: rollout(s, policy, cfg, start, i, lut,
                                                             byte_of, None, temp=TEMP))
                    finally:
                        s.close()
            except TimedOut as e:
                out["skipped"].append({"arm": key, "reason": str(e)})
                save()
                return
            src = "block 62"
        rec = score(key, traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        rec.update({"group": tag, "checkpoint": name, "source": src,
                    "terminator": RB.describe(),
                    "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                      "rate": float(np.mean([x > v for x in xs]))}
                                  for w, v in WALLS.items()},
                    "left_windows": left_inside_outside(traces),
                    "completions": int(sum(
                        1 for t in traces
                        if any(len(f) > 7 and f[6] == 1 and f[7] == 2 for f in t.frames)))})
        out["arms"][key] = rec
        save()
        lw = rec["left_windows"]
        print(f"  {dl.stamp()} {key:26s} p3 {rec['past_wall']['pipe3_735']['rate']*100:5.1f}% "
              f"p4 {rec['past_wall']['pipe4_975']['rate']*100:5.1f}% "
              f"Left in {lw['left_rate_inside']:.3f} out {lw['left_rate_outside']:.3f} "
              f"comp {rec['completions']}", flush=True)

    def eval_rl_script(name):
        key = f"rlscript/{name}"
        if key in out["arms"]:
            return
        src = TRACED / f"jb_{name}_unbiased_{N_EVAL}.json"
        if not src.exists():
            src = TRACED / f"r2_{name}_{N_EVAL}.json"
        if not src.exists():
            return
        if not dl.can_afford(150):
            out["skipped"].append({"arm": key, "reason": "deadline"})
            return
        pol = [_Ep(e) for e in json.loads(src.read_text())["episodes"]]
        stats = token_stats(pol)
        tp = TRACED / f"r2rl_{name}_{N_EVAL}.json"
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                s = sess_get()
                try:
                    traces = resumable(tp, N_EVAL, lambda i: rl_script_episode(s, start, i, stats))
                finally:
                    s.close()
        except TimedOut as e:
            out["skipped"].append({"arm": key, "reason": str(e)})
            save()
            return
        rec = score(key, traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        rec.update({"group": "rlscript", "matched_to": name, "terminator": RB.describe(),
                    "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                      "rate": float(np.mean([x > v for x in xs]))}
                                  for w, v in WALLS.items()},
                    "mean_a_hold": stats["mean_a_hold"]})
        out["arms"][key] = rec
        save()
        print(f"  {dl.stamp()} {key:26s} p2 {rec['past_wall']['pipe2_630']['rate']*100:5.1f}% "
              f"p3 {rec['past_wall']['pipe3_735']['rate']*100:5.1f}% "
              f"A-hold {stats['mean_a_hold']:.1f}", flush=True)

    # baselines first, then round 2, then the controls -- so a deadline cut loses the least
    for i in range(N_SEEDS):
        eval_policy(f"PK32_84_s{i}", "baseline")
    for i in range(N_SEEDS):
        eval_policy(f"DAG2_84_cnn32_s{i}", "round2")
    for i in range(N_SEEDS):
        eval_rl_script(f"PK32_84_s{i}")

    # ---------------- analysis ----------------
    def paired(tagA, tagB, namesA, namesB, field, wall=None):
        a, b = [], []
        for na, nb in zip(namesA, namesB):
            ra, rb = out["arms"].get(f"{tagA}/{na}"), out["arms"].get(f"{tagB}/{nb}")
            if ra and rb:
                if wall:
                    a.append(ra["past_wall"][wall]["rate"] * 100)
                    b.append(rb["past_wall"][wall]["rate"] * 100)
                else:
                    a.append(ra[field])
                    b.append(rb[field])
        return a, b

    base_names = [f"PK32_84_s{i}" for i in range(N_SEEDS)]
    r2_names = [f"DAG2_84_cnn32_s{i}" for i in range(N_SEEDS)]
    res = {}
    for wall in WALLS:
        a, b = paired("round2", "baseline", r2_names, base_names, None, wall)
        if len(a) >= 3:
            d = [x - y for x, y in zip(a, b)]
            p, nperm, floor = sign_flip_p(d)
            res[wall] = {"round2": a, "baseline": b, "diffs": d,
                         "mean_diff": float(np.mean(d)), "n_pairs": len(d),
                         "perm_p": p, "n_permutations": nperm, "floor": floor,
                         "n_positive": int(sum(1 for x in d if x > 0))}
    out["section3d"] = res

    # §4
    diag = {}
    for tag, names in (("baseline", base_names), ("round2", r2_names)):
        ins = [out["arms"][f"{tag}/{n}"]["left_windows"]["left_rate_inside"]
               for n in names if f"{tag}/{n}" in out["arms"]]
        outs = [out["arms"][f"{tag}/{n}"]["left_windows"]["left_rate_outside"]
                for n in names if f"{tag}/{n}" in out["arms"]]
        diag[tag] = {"left_inside": ins, "left_outside": outs,
                     "inside_mean": float(np.mean(ins)) if ins else None,
                     "outside_mean": float(np.mean(outs)) if outs else None}
    if diag["baseline"]["inside_mean"] is not None and diag["round2"]["inside_mean"] is not None:
        n = min(len(diag["baseline"]["left_inside"]), len(diag["round2"]["left_inside"]))
        d_in = [diag["round2"]["left_inside"][i] - diag["baseline"]["left_inside"][i]
                for i in range(n)]
        d_out = [diag["round2"]["left_outside"][i] - diag["baseline"]["left_outside"][i]
                 for i in range(n)]
        p_in, _, floor = sign_flip_p(d_in)
        p_out, _, _ = sign_flip_p(d_out)
        diag["rise_inside"] = {"diffs": d_in, "mean": float(np.mean(d_in)), "perm_p": p_in}
        diag["rise_outside"] = {"diffs": d_out, "mean": float(np.mean(d_out)), "perm_p": p_out}
        diag["floor"] = floor
        diag["ratio_inside_over_outside"] = (float(np.mean(d_in) / np.mean(d_out))
                                            if abs(np.mean(d_out)) > 1e-9 else None)
        diag["conditional"] = bool(np.mean(d_in) > 0 and p_in < 0.05
                                   and (abs(np.mean(d_out)) < 0.25 * abs(np.mean(d_in))))
        # Three outcomes, not two: conditional / marginal / NEITHER. Round 1 was marginal
        # (Left 0.05 -> 0.55 everywhere). If neither rate moves significantly the corrections
        # simply did not take, which is a different failure and needs a different response.
        diag["marginal"] = bool(p_out < 0.05 and np.mean(d_out) > 0)
        diag["neither"] = bool(p_in >= 0.05 and p_out >= 0.05)
        diag["reading"] = ("state-conditional" if diag["conditional"]
                           else "marginal" if diag["marginal"]
                           else "NEITHER -- the corrections did not take at all")
    out["section4"] = diag

    # §5
    a, b = paired("baseline", "rlscript", base_names, base_names, None, "pipe2_630")
    rl = {}
    for wall in WALLS:
        a, b = paired("baseline", "rlscript", base_names, base_names, None, wall)
        if len(a) >= 3:
            d = [x - y for x, y in zip(a, b)]
            p, nperm, floor = sign_flip_p(d)
            rl[wall] = {"policy": a, "rl_script": b, "diffs": d,
                        "mean_diff": float(np.mean(d)), "n_pairs": len(d),
                        "perm_p": p, "floor": floor,
                        "n_positive": int(sum(1 for x in d if x > 0))}
    out["section5"] = rl

    p3 = res.get("pipe3_735")
    s4 = out["section4"]
    r5 = rl.get("pipe2_630")
    parts = []
    if p3:
        parts.append(
            f"**Past pipe 3: round 2 {np.mean(p3['round2']):.1f}% vs baseline "
            f"{np.mean(p3['baseline']):.1f}% ({p3['mean_diff']:+.1f} pp, {p3['n_positive']}/"
            f"{p3['n_pairs']} seeds up, paired sign-flip p={p3['perm_p']:.4f}, floor "
            f"{p3['floor']:.4f}).**")
    if "rise_inside" in s4:
        parts.append(
            f"**§4: Left rate rose {s4['rise_inside']['mean']:+.4f} INSIDE the correction windows "
            f"(p={s4['rise_inside']['perm_p']:.4f}) and {s4['rise_outside']['mean']:+.4f} OUTSIDE "
            f"(p={s4['rise_outside']['perm_p']:.4f}) — "
            + f"**{s4.get('reading', '?')}**. Round 1 was a marginal (Left 0.05 -> 0.55 "
              f"everywhere); at a balanced mix neither rate moves.")
    if r5:
        parts.append(
            f"**§5 at ten paired seeds: policy − run-length script at pipe 2 = "
            f"{r5['mean_diff']:+.1f} pp ({r5['n_positive']}/{r5['n_pairs']} seeds, "
            f"p={r5['perm_p']:.4f}, floor {r5['floor']:.4f}).**")
    out["verdict"] = " ".join(parts) if parts else "insufficient arms"
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
