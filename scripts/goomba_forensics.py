"""What does the policy do in the frames before it dies at the Goomba?

The sweep could not answer this: it started from curated grounded states just before the obstacle, found
1,005 of 1,152 configurations winning, and therefore measured an obstacle whose hard part had been removed.
This reads the policy's **own** episodes instead — the ones where it actually died.

The framing correction that makes the question narrow: the rate-matched script clears the Goomba at 83.0%
while airborne **88.3%** of frames, but **the expert is airborne 61.1% — less than the policy's 66.1%.** So
88.3% is not the solution, it is a degenerate exploit 27 points above the expert. The policy is *closer to
the expert* than the script is and still fails, which means the deficit is timing or approach state, not
airborne fraction.

Four questions, all from retained traces:

1. does it press A at all in the approach (x 200–260)?
2. if so, where — against the sweep's winning trigger band around x=252, and against the expert's own
   A-onset positions, which are also read here
3. if not, what is it doing: grounded and idle, grounded and holding Right, or airborne and descending
4. what distinguishes the 130 episodes that cleared

Medians carry max and p99 throughout.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci, wilson  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "data/traces/variant_capped_200.json"
OUT = ROOT / "data/goomba_forensics.json"
RIGHT = NES_BUTTON_BITS["Right"]

DEATH_BAND = (272, 320)
APPROACH = (200, 260)          # before the enemy; the sweep's winning triggers were 244-288
WIN_TRIGGERS = (244, 288)
CLEAR_X = 320
FLOOR = 432                    # absolute y of the 1-1 floor


def stats(v) -> dict:
    a = np.asarray(list(v), dtype=float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "min": float(a.min())}


def a_onsets_in(frames, lo, hi) -> list[int]:
    """x positions where A turns on, within [lo, hi]."""
    out = []
    prev = False
    for f in frames:
        a = bool(f[3] & A_BIT)
        if a and not prev and lo <= f[0] <= hi:
            out.append(int(f[0]))
        prev = a
    return out


def describe_episode(e) -> dict:
    fr = e["frames"]
    mx = max(f[0] for f in fr)
    onsets = a_onsets_in(fr, *APPROACH)
    # the fatal / final stretch: last 30 frames before the end
    tail = fr[-30:]
    grounded_tail = [f[5] for f in tail if len(f) >= 6]
    right_tail = [bool(f[3] & RIGHT) for f in tail]
    idle_tail = [f[3] == 0 for f in tail]
    # y trend over the last 10 frames: y grows downward, so rising y = descending Mario
    ys = [f[1] for f in fr[-10:]]
    descending = bool(len(ys) >= 2 and ys[-1] > ys[0])
    # state at the last frame
    last = fr[-1]
    return {
        "seed": e["seed"], "max_x": int(mx), "cleared": bool(mx > CLEAR_X),
        "died_in_band": bool(e.get("death")
                             and DEATH_BAND[0] <= e["death"]["x"] < DEATH_BAND[1]),
        "n_onsets_approach": len(onsets), "onsets_x": onsets,
        "first_onset_x": onsets[0] if onsets else None,
        "grounded_frac_tail": float(np.mean(grounded_tail)) if grounded_tail else None,
        "right_frac_tail": float(np.mean(right_tail)),
        "idle_frac_tail": float(np.mean(idle_tail)),
        "descending_at_end": descending,
        "final_grounded": bool(last[5]) if len(last) >= 6 else None,
        "final_y_above_floor": FLOOR - int(last[1]),
        "speed_at_end": int(last[2]),
    }


def expert_goomba_onsets(ctx) -> dict:
    """Where the expert presses A near the Goomba, on 1-1 surface frames."""
    xs = []
    for run in ctx.expert_train:
        tr = np.asarray(run.trace)
        w, s, pg = column(tr, "world"), column(tr, "stage"), column(tr, "pregame")
        x, ps = column(tr, "x_position"), column(tr, "player_state")
        a = np.asarray(run.actions, dtype=np.uint8)
        n = min(len(x), len(a))
        m = (w[:n] == 1) & (s[:n] == 1) & (pg[:n] == 1) & (ps[:n] == 8)
        idx = np.flatnonzero(m)
        prev = False
        for i in idx:
            on = bool(a[i] & A_BIT)
            if on and not prev and 180 <= x[i] <= 320:
                xs.append(int(x[i]))
            prev = on
    return {"onset_x": stats(xs), "histogram_16px": dict(sorted(Counter(
        v // 16 * 16 for v in xs).items())), "n_runs": len(ctx.expert_train)}


def main() -> None:
    ctx = O.Ctx()
    eps = json.loads(TRACES.read_text())["episodes"]
    rows = [describe_episode(e) for e in eps]
    died = [r for r in rows if r["died_in_band"]]
    cleared = [r for r in rows if r["cleared"]]
    print(f"{len(eps)} episodes: {len(died)} died in x {DEATH_BAND[0]}-{DEATH_BAND[1] - 1}, "
          f"{len(cleared)} cleared x>{CLEAR_X}\n", flush=True)

    exp = expert_goomba_onsets(ctx)
    print(f"EXPERT A-onsets in x 180-320: n={exp['onset_x']['n']} "
          f"median x {exp['onset_x'].get('median')} "
          f"(min {exp['onset_x'].get('min')}, max {exp['onset_x'].get('max')})")
    print(f"  histogram: {exp['histogram_16px']}\n", flush=True)

    # Q1 -- does it press A at all in the approach?
    d_any = sum(1 for r in died if r["n_onsets_approach"] > 0)
    c_any = sum(1 for r in cleared if r["n_onsets_approach"] > 0)
    lo, hi = diff_ci(d_any, len(died), c_any, len(cleared)) if died and cleared else (0, 0)
    print(f"Q1 pressed A in x {APPROACH[0]}-{APPROACH[1]}:")
    print(f"   died    {d_any}/{len(died)} = {d_any / max(len(died), 1) * 100:.1f}% "
          f"{[round(v * 100, 1) for v in wilson(d_any, len(died))]}")
    print(f"   cleared {c_any}/{len(cleared)} = {c_any / max(len(cleared), 1) * 100:.1f}% "
          f"{[round(v * 100, 1) for v in wilson(c_any, len(cleared))]}")
    print(f"   difference {(c_any / max(len(cleared), 1) - d_any / max(len(died), 1)) * 100:+.1f} pp "
          f"[{lo * 100:+.1f}, {hi * 100:+.1f}]\n", flush=True)

    # Q2 -- where, relative to the winning band
    d_first = [r["first_onset_x"] for r in died if r["first_onset_x"] is not None]
    c_first = [r["first_onset_x"] for r in cleared if r["first_onset_x"] is not None]
    in_band = lambda v: WIN_TRIGGERS[0] <= v <= WIN_TRIGGERS[1]
    print(f"Q2 first A-onset x (winning trigger band {WIN_TRIGGERS[0]}-{WIN_TRIGGERS[1]}):")
    print(f"   died    {stats(d_first)}")
    print(f"   cleared {stats(c_first)}")
    print(f"   in band: died {sum(1 for v in d_first if in_band(v))}/{len(d_first)}, "
          f"cleared {sum(1 for v in c_first if in_band(v))}/{len(c_first)}\n", flush=True)

    # Q3 -- if it does not jump, what is it doing
    d_nojump = [r for r in died if r["n_onsets_approach"] == 0]
    print(f"Q3 of the {len(d_nojump)} deaths with NO approach jump — final 30 frames:")
    if d_nojump:
        print(f"   grounded fraction  {stats([r['grounded_frac_tail'] for r in d_nojump])}")
        print(f"   holding Right      {stats([r['right_frac_tail'] for r in d_nojump])}")
        print(f"   idle (no buttons)  {stats([r['idle_frac_tail'] for r in d_nojump])}")
        print(f"   descending at end  {sum(1 for r in d_nojump if r['descending_at_end'])}"
              f"/{len(d_nojump)}")
        print(f"   grounded at the last frame "
              f"{sum(1 for r in d_nojump if r['final_grounded'])}/{len(d_nojump)}")
    print(flush=True)

    # Q4 -- what distinguishes the clearers
    print("Q4 clearers vs deaths, approach and final state:")
    for name, key in (("onsets in approach", "n_onsets_approach"),
                      ("grounded frac (final 30)", "grounded_frac_tail"),
                      ("Right frac (final 30)", "right_frac_tail"),
                      ("idle frac (final 30)", "idle_frac_tail")):
        dv = stats([r[key] for r in died if r[key] is not None])
        cv = stats([r[key] for r in cleared if r[key] is not None])
        print(f"   {name:26s} died med {dv.get('median')} p99 {dv.get('p99')} max {dv.get('max')} "
              f"| cleared med {cv.get('median')} p99 {cv.get('p99')} max {cv.get('max')}")

    out = {
        "death_band": list(DEATH_BAND), "approach": list(APPROACH),
        "winning_trigger_band": list(WIN_TRIGGERS), "clear_x": CLEAR_X,
        "n_episodes": len(eps), "n_died_in_band": len(died), "n_cleared": len(cleared),
        "expert_onsets_near_goomba": exp,
        "q1_pressed_A_in_approach": {
            "died": {"k": d_any, "n": len(died), "rate": d_any / max(len(died), 1),
                     "ci": list(wilson(d_any, len(died)))},
            "cleared": {"k": c_any, "n": len(cleared), "rate": c_any / max(len(cleared), 1),
                        "ci": list(wilson(c_any, len(cleared)))},
            "difference_pp": (c_any / max(len(cleared), 1) - d_any / max(len(died), 1)) * 100,
            "ci_pp": [lo * 100, hi * 100]},
        "q2_first_onset_x": {"died": stats(d_first), "cleared": stats(c_first),
                             "in_band_died": sum(1 for v in d_first if in_band(v)),
                             "in_band_cleared": sum(1 for v in c_first if in_band(v))},
        "q3_no_jump_deaths": {
            "n": len(d_nojump),
            "grounded_frac_tail": stats([r["grounded_frac_tail"] for r in d_nojump]),
            "right_frac_tail": stats([r["right_frac_tail"] for r in d_nojump]),
            "idle_frac_tail": stats([r["idle_frac_tail"] for r in d_nojump]),
            "descending_at_end": sum(1 for r in d_nojump if r["descending_at_end"]),
            "grounded_at_last_frame": sum(1 for r in d_nojump if r["final_grounded"])},
        "q4_contrast": {k: {"died": stats([r[k] for r in died if r[k] is not None]),
                            "cleared": stats([r[k] for r in cleared if r[k] is not None])}
                        for k in ("n_onsets_approach", "grounded_frac_tail",
                                  "right_frac_tail", "idle_frac_tail")},
        "rows": rows,
    }
    presses = d_any / max(len(died), 1)
    out["verdict"] = (
        f"TIMING FAILURE: {d_any} of {len(died)} deaths ({presses * 100:.1f}%) pressed A during the "
        f"approach, so the policy does act and acts wrongly. A sweep that starts from the states the "
        f"policy actually arrives in could address it."
        if presses >= 0.5 else
        f"FAILURE TO ACT: only {d_any} of {len(died)} deaths ({presses * 100:.1f}%) pressed A anywhere in "
        f"x {APPROACH[0]}-{APPROACH[1]}. The policy walks into the enemy without jumping, in a state where "
        f"the expert jumps. That is not a timing problem and not a demonstration problem -- it points at "
        f"the observation or the generation rule.")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
