"""Audit the pipe-4 distillation result. No emulator; a read over two retained trace files.

`pipe4_distil.py` printed "CLEARANCE MOVED WITHOUT THE HOLD", which is wrong in a way worth fixing
in code rather than in prose. Its A-hold comparison reported `median 4.0 -> None`, and it treated
`None` as "did not rise". `None` is **missing data**: the distilled policy never reached x=880, so
there were zero A-hold onsets in the 880-924 window and the transfer question is unanswerable *in
that window* rather than answered negatively.

This script does three things the original did not:

1. **Separates "did not rise" from "could not be measured."** A window with no arrivals yields
   `measurable: false` and is never scored as a decrease.
2. **Measures the hold where a paired comparison is actually possible** -- windows both arms reach.
   That is the strongest available form of the transfer test.
3. **Names the mechanism** by comparing each arm's button marginals against the expert's own press
   rates, which is where the regression is visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.pipe4_metrics import a_hold_onsets, arm_metrics, hold_stats  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARMS = {"baseline": ROOT / "data/traces/p1_200.json",
        "distilled": ROOT / "data/traces/pipe4_distil_200.json"}
OUT = ROOT / "data/pipe4_transfer_audit.json"
A_BIT = NES_BUTTON_BITS["A"]

#: Windows to test the hold in. The pipe-4 window is the one the directive asked about; the others
#: exist because a window neither arm reaches cannot answer anything.
WINDOWS = {"pipe1_approach_300_470": (300, 470), "pipe2_560_640": (560, 640),
           "pipe4_880_924": (880, 924)}


def all_holds(episodes) -> np.ndarray:
    out = []
    for e in episodes:
        a = [(f[3] & A_BIT) > 0 for f in e["frames"]]
        i = 0
        while i < len(a):
            if a[i] and (i == 0 or not a[i - 1]):
                j = i
                while j < len(a) and a[j]:
                    j += 1
                out.append(j - i)
                i = j
            else:
                i += 1
    return np.asarray(out, dtype=int)


def main() -> None:
    ctx = O.Ctx()
    expert = {k: round(float(v), 3) for k, v in ctx.target_rates.items()}
    out = {"expert_press_rates": expert, "arms": {}, "windows": {},
           "measurement_basis": "single_life", "n_per_arm": 200,
           "note": "paired: identical episode function, identical seeds 0-199"}

    eps = {}
    for tag, path in ARMS.items():
        d = json.loads(path.read_text())
        eps[tag] = d["episodes"]
        bits = np.asarray([f[3] for e in d["episodes"] for f in e["frames"]], dtype=np.int64)
        h = all_holds(d["episodes"])
        m = arm_metrics([e["frames"] for e in d["episodes"]])
        m.pop("rows")
        out["arms"][tag] = {
            "checkpoint": d.get("checkpoint"),
            "frames": int(len(bits)),
            "press_rates": {n: round(float(((bits & NES_BUTTON_BITS[n]) > 0).mean()), 3)
                            for n in NES_BUTTON_ORDER},
            "a_rate_over_expert": round(float(((bits & A_BIT) > 0).mean()) / expert["A"], 2),
            "a_holds_anywhere": {**hold_stats(h.tolist()),
                                 "frames_in_a_hold_pct": round(float(h.sum()) / len(bits) * 100, 1)},
            "frontier": m,
            "ended": {k: sum(1 for e in d["episodes"] if e["ended"] == k)
                      for k in ("died", "stuck")},
        }

    for name, (lo, hi) in WINDOWS.items():
        row = {"x_range": [lo, hi]}
        for tag in ARMS:
            reach = sum(1 for e in eps[tag] if max(f[0] for f in e["frames"]) >= lo)
            h = [x for e in eps[tag] for x in a_hold_onsets(e["frames"], (lo, hi))]
            row[tag] = {"episodes_reaching_lo": reach, "measurable": bool(h), **hold_stats(h)}
        both = row["baseline"]["measurable"] and row["distilled"]["measurable"]
        row["paired_comparison_possible"] = both
        if both:
            row["hold_direction"] = (
                "rose" if row["distilled"]["median"] > row["baseline"]["median"]
                else "fell" if row["distilled"]["median"] < row["baseline"]["median"] else "flat")
        else:
            row["hold_direction"] = "unmeasurable -- one arm never reached this window"
        out["windows"][name] = row

    b, g = out["arms"]["baseline"], out["arms"]["distilled"]
    measurable = [n for n, r in out["windows"].items() if r["paired_comparison_possible"]]
    fell = [n for n in measurable if out["windows"][n]["hold_direction"] == "fell"]
    out["verdict"] = {
        "answer_to_the_one_question": "NEITHER",
        "hold_at_pipe4": ("unmeasurable: the distilled policy never reached x=880, so there were "
                          "zero A-hold onsets in 880-924. This is missing data, not a decrease."),
        "hold_where_measurable": (
            f"fell in {len(fell)} of {len(measurable)} windows both arms reach: " +
            "; ".join(f"{n} median {out['windows'][n]['baseline']['median']}->"
                      f"{out['windows'][n]['distilled']['median']}, >=12 "
                      f"{(out['windows'][n]['baseline']['frac_ge_required'] or 0) * 100:.1f}%->"
                      f"{(out['windows'][n]['distilled']['frac_ge_required'] or 0) * 100:.1f}%"
                      for n in measurable)),
        "stuck_at_pipe4": ("0 of 200 versus 29 of 200, but only because 0 of 200 arrived at x=880. "
                           "The count fell for the wrong reason and must not be read as progress."),
        "mechanism": (
            f"The baseline checkpoint presses A on {b['press_rates']['A'] * 100:.1f}% of frames, "
            f"{b['a_rate_over_expert']}x the expert's {expert['A'] * 100:.1f}%, and spends "
            f"{b['a_holds_anywhere']['frames_in_a_hold_pct']}% of all frames inside an A-hold. "
            f"Fine-tuning on demonstrations whose A-rate is near the expert's pulled the marginal "
            f"to {g['press_rates']['A'] * 100:.1f}% "
            f"({g['a_rate_over_expert']}x expert), and holds shortened everywhere: median "
            f"{b['a_holds_anywhere']['median']}->{g['a_holds_anywhere']['median']}, "
            f">=12 {b['a_holds_anywhere']['frac_ge_required'] * 100:.1f}%->"
            f"{g['a_holds_anywhere']['frac_ge_required'] * 100:.1f}%. Down and Left were also "
            f"erased ({b['press_rates']['Down']}->{g['press_rates']['Down']} and "
            f"{b['press_rates']['Left']}->{g['press_rates']['Left']}), because the scripted "
            f"demonstrations contain neither. x_median fell "
            f"{b['frontier']['x_median']:.0f}->{g['frontier']['x_median']:.0f}."),
        "consequence_for_prior_results": (
            "The frontier map -- 29 stuck at pipe 4, 38 clearing it, x_median 723 -- is a property "
            "of a policy that holds A on 85% of frames. That is the always-jump degeneracy the "
            "ledger already records as a known objective flaw, still present in the checkpoint "
            "every recent figure rests on."),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    for k, v in out["verdict"].items():
        print(f"\n[{k}]\n{v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
