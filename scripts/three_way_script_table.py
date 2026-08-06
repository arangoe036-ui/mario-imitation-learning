"""§3: policy vs MATCHED script vs HIGH-JUMP script, one block, one terminator, one start.

The write-up turns on this and it has never been produced in one place. Two true statements that must never
be made alone:

1. **At its own button rates the policy beats a script by ~50 pp at pipe 2** — state-conditioning is worth a
   great deal.
2. **A script that jumps far more often beats the policy on the surface** — the policy's chosen jump *rate*
   is worse than a simple higher one.

The A 0.85 arm is **not** a control for this policy and must never be called "a coin flip" or "rate-matched":
it runs A 0.85 against the policy's **0.338** and Down 0.086 against 0.027. It is also **the selected arm of
a per-obstacle best-of-N envelope** — `script_baseline.build` keeps `row["rate"] > cur["rate"]` per obstacle
— which is a deliberately hard bar and fine as *a* bar, but it is a maximum over arms, not a typical script.

**⚠ Correction carried into the build:** the directive gives the policy as "A ≈0.505, Down 0.0148". Measured
at T=0.7 with `STALL=6500` from the level start, `P_84_cnn32_seed4` runs **A 0.338, Down 0.027, Left 0.040,
Right 0.628, B 0.657**. The matched script is built from the measured values, and the gap to 0.85 is
therefore *larger* than the directive states.

Both readings of "matched" are run, because the ledger records that they answer different questions:
**all-five-matched**, and the stronger **Right+B held** variant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.scaleup_eval import resumable, score, scripted_episode  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/three_way_script_table.json"
TRACED = ROOT / "data/traces"

POLICY_ARM = "P_84_cnn32_seed4"
#: measured from mask_P_84_cnn32_seed4_t0.7_unmasked_200 -- same policy, temperature, terminator, start
POLICY_RATES = {"A": 0.338, "B": 0.657, "Right": 0.628, "Down": 0.027, "Left": 0.040}
HIGH_JUMP = {"A": 0.85, "Left": 0.135, "Down": 0.086, "B": 1.0, "Right": 1.0}
N = 200
WALLS = {"goomba_320": 320, "pipe1_470": 470, "pipe2_630": 630, "pipe3_735": 735,
         "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562}
ARM_BUDGET_S = 20 * 60


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    arms = {
        "matched_all_five": dict(POLICY_RATES),
        "matched_rightB_held": {**POLICY_RATES, "B": 1.0, "Right": 1.0},
        "high_jump_A0.85": dict(HIGH_JUMP),
    }
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("skipped", [])
    out.update({
        "n": N, "terminator": RB.describe(),
        "measurement_basis": "single_life_from_level_start",
        "policy_arm": POLICY_ARM, "policy_measured_rates": POLICY_RATES,
        "script_arms": arms,
        "high_jump_is_not_a_control": (
            "A 0.85 vs the policy's 0.338 and Down 0.086 vs 0.027; it is also the SELECTED arm of a "
            "per-obstacle best-of-N envelope (script_baseline.build keeps the per-obstacle maximum "
            "over arms). A hard bar, but never 'a coin flip' and never 'rate-matched'"),
        "directive_premise_correction": (
            "the directive gives the policy as A ~0.505 / Down 0.0148; measured at T=0.7, STALL=6500, "
            "from the level start it is A 0.338 / Down 0.027")})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    for name, rates in arms.items():
        if name in out["arms"]:
            continue
        if not dl.can_afford(180):
            out["skipped"].append({"arm": name, "reason": "deadline"})
            continue
        tp = TRACED / f"threeway_{name}_{N}.json"
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), name):
                s = sess_get()
                try:
                    traces = resumable(tp, N, lambda i: scripted_episode(s, start, i, rates))
                finally:
                    s.close()
        except TimedOut as e:
            out["skipped"].append({"arm": name, "reason": str(e)})
            save()
            continue
        rec = score(name, traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        rec.update({
            "rates": rates, "kind": "script", "terminator": RB.describe(),
            "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                              "rate": float(np.mean([x > v for x in xs]))}
                          for w, v in WALLS.items()},
            "flagpole_episodes": int(sum(
                1 for t in traces if any(len(f) > 4 and f[4] == 0x05 for f in t.frames))),
            "traces": str(tp.relative_to(ROOT))})
        out["arms"][name] = rec
        save()
        pw = rec["past_wall"]
        print(f"  {dl.stamp()} {name:22s} A {rec['button_marginals']['rates']['A']:.3f} "
              f"p2 {pw['pipe2_630']['rate']*100:5.1f}% p3 {pw['pipe3_735']['rate']*100:5.1f}% "
              f"p4 {pw['pipe4_975']['rate']*100:5.1f}% x_med {rec['x_median']:4.0f} "
              f"flag {rec['flagpole_episodes']}", flush=True)

    # ---- the policy row, reused from block 59's route audit conditions ----
    pol_trace = TRACED / "mask_P_84_cnn32_seed4_t0.7_unmasked_200.json"
    if pol_trace.exists() and "policy" not in out["arms"]:
        from scripts.scaleup_eval import _Ep
        traces = [_Ep(e) for e in json.loads(pol_trace.read_text())["episodes"]]
        rec = score("policy", traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        rec.update({"kind": "policy", "checkpoint": POLICY_ARM, "terminator": RB.describe(),
                    "source": "reused: mask study, unmasked arm -- same policy, T, terminator, start",
                    "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                      "rate": float(np.mean([x > v for x in xs]))}
                                  for w, v in WALLS.items()},
                    "flagpole_episodes": int(sum(
                        1 for t in traces
                        if any(len(f) > 4 and f[4] == 0x05 for f in t.frames)))})
        out["arms"]["policy"] = rec
        save()

    # ---------------- the three-way table ----------------
    order = ["policy", "matched_all_five", "matched_rightB_held", "high_jump_A0.85"]
    table = {}
    for w in WALLS:
        row = {}
        for a in order:
            r = out["arms"].get(a)
            if r:
                row[a] = r["past_wall"][w]["rate"] * 100
        pol = out["arms"].get("policy")
        if pol:
            for a in order[1:]:
                r = out["arms"].get(a)
                if not r:
                    continue
                kp = pol["past_wall"][w]["k"]
                ks = r["past_wall"][w]["k"]
                lo, hi = diff_ci(ks, N, kp, N)
                row[f"policy_minus_{a}_pp"] = (pol["past_wall"][w]["rate"]
                                               - r["past_wall"][w]["rate"]) * 100
                row[f"policy_minus_{a}_ci"] = [lo * 100, hi * 100]
        table[w] = row
    out["three_way_table"] = table

    pol = out["arms"].get("policy")
    if pol:
        m = out["arms"].get("matched_rightB_held") or out["arms"].get("matched_all_five")
        h = out["arms"].get("high_jump_A0.85")
        p2m = table["pipe2_630"].get(
            "policy_minus_matched_rightB_held_pp",
            table["pipe2_630"].get("policy_minus_matched_all_five_pp"))
        p3h = table["pipe3_735"].get("policy_minus_high_jump_A0.85_pp")
        out["verdict"] = (
            f"**BOTH STATEMENTS ARE TRUE AND NEITHER MAY STAND ALONE.** Against a script at its OWN "
            f"measured marginals the policy is **{p2m:+.1f} pp at pipe 2** — state-conditioning is worth a "
            f"great deal. Against the high-jump A 0.85 arm it is **{p3h:+.1f} pp past pipe 3** — a script "
            f"that jumps far more often gets further. **The deficit is a MARGINAL, not an absence of "
            f"skill.** The A 0.85 arm runs A 0.85 against the policy's 0.338 and is the selected arm of a "
            f"per-obstacle best-of-N envelope: a hard bar, never 'a coin flip', never 'rate-matched'.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out.get("verdict", "policy row missing"))
    print(f"\n{'wall':>16s}" + "".join(f"{a[:14]:>16s}" for a in order))
    for w, row in table.items():
        print(f"{w:>16s}" + "".join(
            f"{row.get(a, float('nan')):>16.1f}" for a in order))
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
