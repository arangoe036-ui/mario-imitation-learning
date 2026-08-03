"""Fold the overnight JSONL results into FINDINGS.md, negative results included.

Run after the overnight job finishes. Idempotent: it replaces the generated block rather
than appending a second copy.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARK = "<!-- overnight-generated -->"


def main() -> None:
    rows = [json.loads(x) for x in (ROOT / "data/overnight.jsonl").read_text().splitlines()
            if x.strip()]
    done = {r["task"]: r["result"] for r in rows if r["kind"] == "task_done"}
    failed = [r for r in rows if r["kind"] == "task_failed"]
    L = [MARK, "", "# Overnight results", ""]

    r2 = [r for r in rows if r["kind"] == "tier2_round"]
    if r2:
        L += ["## Stage 3 arm A: self-imitation rounds", "",
              "| tag | expert:self | accept % | A-onset recall | pipe1 1-1 (95% CI) |",
              "| --- | --- | --- | --- | --- |"]
        for r in r2:
            one = r["live"].get("1-1", {})
            ci = one.get("pipe1_ci", [0, 0])
            L.append(f"| {r.get('tag')} | {r.get('ratio', '-')} | "
                     f"{(r.get('acceptance_rate') or 0) * 100:.0f} | "
                     f"{r['offline']['onset_recall']['A'] * 100:.1f}% | "
                     f"{one.get('pipe1_rate', 0) * 100:.1f}% "
                     f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}] |")
        L.append("")

    t3 = done.get("tier3_oracle_margin")
    if t3:
        L += ["## Oracle, third and final attempt (margin-calibrated)", "",
              f"**{t3.get('verdict')}**", "",
              f"Matched the expert's {t3.get('expert_rate', 0) * 100:.1f}% jump rate at "
              f"margin M={t3.get('matched', {}).get('margin')}, giving onset agreement "
              f"{t3.get('matched', {}).get('agreement_at_onsets', 0) * 100:.1f}%.", ""]

    t4 = done.get("tier4_glitchless_vs_glitchy")
    if t4:
        L += ["## Glitchless vs glitch-heavy", "", f"> {t4.get('caveat')}", ""]
        arms = [r for r in rows if r["kind"] == "tier4_arm"]
        if arms:
            L += ["| arm | seed | A-onset recall | pipe1 1-1 |", "| --- | --- | --- | --- |"]
            for r in arms:
                L.append(f"| {r['arm']} | {r['seed']} | "
                         f"{r['offline']['onset_recall']['A'] * 100:.1f}% | "
                         f"{r['live'].get('1-1', {}).get('pipe1_rate', 0) * 100:.1f}% |")
            L.append("")

    t5 = [r for r in rows if r["kind"] == "tier5_point"]
    if t5:
        L += ["## Data scaling", "", "| fraction | frames | A-onset recall | pipe1 1-1 |",
              "| --- | --- | --- | --- |"]
        for r in sorted(t5, key=lambda x: x["fraction"]):
            L.append(f"| {r['fraction']:.0%} | {r['frames']:,} | "
                     f"{r['offline']['onset_recall']['A'] * 100:.1f}% | "
                     f"{r['live'].get('1-1', {}).get('pipe1_rate', 0) * 100:.1f}% |")
        L.append("")

    t6 = done.get("tier6_two_one_wall")
    if t6:
        L += ["## The 2-1 wall", "",
              f"Running right and holding reaches x={t6.get('run_right_max_x')}; running "
              f"and jumping reaches x={t6.get('run_and_jump_max_x')}; died running right: "
              f"{t6.get('died_running_right')}. **{t6.get('interpretation')}**", ""]

    if failed:
        L += ["## What broke", ""]
        L += [f"- **{f['task']}** — `{f['error']}`" for f in failed]
        L.append("")

    text = (ROOT / "FINDINGS.md").read_text()
    text = text.split(MARK)[0].rstrip() + "\n\n---\n\n" + "\n".join(L) + "\n"
    (ROOT / "FINDINGS.md").write_text(text)
    print(f"appended {len(L)} lines to FINDINGS.md")


if __name__ == "__main__":
    main()
