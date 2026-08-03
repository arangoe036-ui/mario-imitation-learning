"""The central question: does imitating the expert better make a better player?

Pools every checkpoint that has both an imitation-fidelity measurement (A-onset recall,
exact match) and a task-performance measurement (pipe 1 cleared, x median) taken under the
*same* protocol, and tests whether they anti-correlate.

Only checkpoints measured after the double-normalisation fix are eligible; earlier numbers
used a different calibration and are not comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.bc.overnight_lib import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["data/overnight.jsonl", "data/followup.jsonl", "data/chain_position.jsonl"]
OUT_JSON = ROOT / "data/fidelity_vs_performance.json"
OUT_PNG = ROOT / "data/plots/fidelity_vs_performance.png"


def collect() -> list[dict]:
    pts = []
    for src in SOURCES:
        p = ROOT / src
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            off, live = r.get("offline"), r.get("live")
            if not (isinstance(off, dict) and isinstance(live, dict)):
                continue
            one = live.get("1-1") or {}
            if "pipe1_rate" not in one or "onset_recall" not in off:
                continue
            pts.append({
                "tag": r.get("tag") or r.get("arm") or r.get("kind"),
                "source": src,
                "a_onset_recall": off["onset_recall"]["A"],
                "exact_match": off.get("exact_match"),
                "pipe1_rate": one["pipe1_rate"],
                "pipe1_k": one.get("pipe1_k"), "pipe1_n": one.get("n"),
                "x_median": one.get("x_median"),
                "max_a_hold": one.get("longest_a_hold_max"),
            })
    # De-duplicate on tag, keeping the largest evaluation.
    best: dict[str, dict] = {}
    for p in pts:
        k = str(p["tag"])
        if k not in best or (p["pipe1_n"] or 0) > (best[k]["pipe1_n"] or 0):
            best[k] = p
    return list(best.values())


def main() -> None:
    pts = collect()
    if len(pts) < 4:
        print(f"only {len(pts)} comparable checkpoints; need more before this means anything")
        return
    rec = np.array([p["a_onset_recall"] for p in pts])
    pipe = np.array([p["pipe1_rate"] for p in pts])
    ex = np.array([p["exact_match"] or np.nan for p in pts])

    def pearson(a, b):
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 3:
            return None
        a, b = a[m], b[m]
        return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-12))

    r_rec = pearson(rec, pipe)
    r_ex = pearson(ex, pipe)
    print(f"{len(pts)} comparable checkpoints")
    print(f"  corr(A-onset recall, pipe1) = {r_rec}")
    print(f"  corr(exact match,   pipe1) = {r_ex}")
    for p in sorted(pts, key=lambda q: -q["pipe1_rate"]):
        print(f"  {str(p['tag'])[:34]:34s} recall {p['a_onset_recall'] * 100:5.1f}%  "
              f"exact {(p['exact_match'] or 0) * 100:5.1f}%  "
              f"pipe1 {p['pipe1_rate'] * 100:5.1f}% (n={p['pipe1_n']})")

    verdict = ("ANTI-CORRELATED: imitating the expert more closely makes a worse player"
               if (r_rec is not None and r_rec < -0.3) else
               ("POSITIVELY correlated: fidelity and performance rise together"
                if (r_rec is not None and r_rec > 0.3) else
                "NO clear relationship at this sample size"))
    print(f"\n{verdict}")

    OUT_JSON.write_text(json.dumps(
        {"n": len(pts), "corr_recall_pipe1": r_rec, "corr_exact_pipe1": r_ex,
         "verdict": verdict, "points": pts}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, xs, name in ((axes[0], rec * 100, "A-onset recall (%)"),
                             (axes[1], ex * 100, "exact match (%)")):
            ys = pipe * 100
            lo = [(y - wilson(p["pipe1_k"] or 0, p["pipe1_n"] or 1)[0] * 100)
                  for y, p in zip(ys, pts)]
            hi = [(wilson(p["pipe1_k"] or 0, p["pipe1_n"] or 1)[1] * 100 - y)
                  for y, p in zip(ys, pts)]
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color="#4878a8", capsize=3)
            for x, y, p in zip(xs, ys, pts):
                ax.annotate(str(p["tag"])[:18], (x, y), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel(name)
            ax.set_ylabel("pipe 1 cleared (%)")
            ax.grid(alpha=0.25)
        axes[0].set_title(f"imitation fidelity vs task performance  (r={r_rec:.2f})",
                          fontsize=10)
        axes[1].set_title(f"exact match vs task performance  (r={r_ex:.2f})"
                          if r_ex is not None else "exact match vs performance", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=150)
        print(f"wrote {OUT_PNG}")
    except Exception as exc:
        print(f"plot skipped: {exc}")


if __name__ == "__main__":
    main()
