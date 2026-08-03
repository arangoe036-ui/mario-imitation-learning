"""Turn ``stage2_results.jsonl`` into a readable summary.

The sweep appends records as it goes, so this reads whatever exists -- a half-finished
night still produces a valid report, which is the point of streaming the results.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_records(path: Path | str) -> list[dict]:
    """Read a JSONL log, skipping any truncated final line."""
    records: list[dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a partially written last line while the sweep is running
    return records


def _fmt(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _live_cell(live: dict, key: str, stat: str = "median") -> str:
    if not live or "error" in live:
        return "err"
    return _fmt(live.get(key, {}).get(stat))


def build_summary(records: list[dict]) -> str:
    smoke = [r for r in records if r.get("kind") == "smoke"]
    baselines = [r for r in records if r.get("kind") == "baseline"]
    tables = [r for r in records if r.get("kind") == "baseline_table"]
    evals = [r for r in records if r.get("kind") == "eval"]
    done = [r for r in records if r.get("kind") == "config_done"]
    failed = [r for r in records if r.get("kind") == "config_failed"]

    lines: list[str] = ["# Stage 2 — behavioural cloning baseline", ""]

    # -- what ran ---------------------------------------------------------- #
    env = next((r.get("environment") for r in records if r.get("environment")), {})
    lines += [
        "## What ran",
        "",
        f"- torch {env.get('torch', '?')} on `{env.get('device', '?')}`",
        f"- {len(evals)} evaluation points across "
        f"{len({e['config'] for e in evals})} configs",
        f"- {len(done)} configs finished, {len(failed)} failed",
        "",
    ]

    if smoke:
        last = smoke[-1]
        if last.get("ok"):
            checks = last.get("checks", {})
            loss = checks.get("loss_decreases", {})
            live = checks.get("live_play", {})
            ck = checks.get("checkpoint_roundtrip", {})
            lines += [
                "## Smoke test (the gate)",
                "",
                "PASSED. All four checks:",
                "",
                f"1. data: {checks.get('data', {}).get('frames_used')} of "
                f"{checks.get('data', {}).get('train_frames_available', 0):,} train frames, memory-mapped",
                f"2. loss decreased {_fmt(loss.get('first10_mean'), 4)} -> "
                f"{_fmt(loss.get('last10_mean'), 4)} over {loss.get('steps')} steps",
                f"3. checkpoint saved and reloaded, max|Δlogit| = "
                f"{ck.get('max_abs_diff')}",
                f"4. one live episode completed: {live.get('frames')} frames, "
                f"level {live.get('furthest_level')}, x={live.get('furthest_x')} "
                f"({live.get('ended')})",
                "",
            ]
        else:
            lines += [
                "## Smoke test (the gate)",
                "",
                f"**FAILED — long run not started.** {last.get('error')}",
                "",
            ]

    # -- baselines ---------------------------------------------------------- #
    if tables:
        lines += [
            "## Baselines",
            "",
            "| baseline | val accuracy | val macro accuracy | predicts |",
            "| --- | --- | --- | --- |",
        ]
        for s in tables[-1]["scores"]:
            lines.append(
                f"| {s['name']} | {s['val_accuracy'] * 100:.2f}% | "
                f"{s['val_macro_accuracy'] * 100:.2f}% | {s['predicts']} |"
            )
        lines.append("")

    if baselines:
        lines += [
            "### Baselines in live play",
            "",
            "| baseline | progress (median) | levels (median) | deaths (median) | frames survived (median) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for b in baselines:
            live = b.get("live", {})
            lines.append(
                f"| {b['name']} | {_live_cell(live, 'total_progress')} | "
                f"{_live_cell(live, 'levels_reached')} | {_live_cell(live, 'deaths')} | "
                f"{_live_cell(live, 'frames_survived')} |"
            )
        lines.append("")

    # -- learned configs ---------------------------------------------------- #
    if evals:
        by_config: dict[str, list[dict]] = {}
        for e in evals:
            by_config.setdefault(e["config"], []).append(e)

        lines += [
            "## Configs, at their best evaluation point",
            "",
            "| config | params | blind | best step | val loss | val acc | macro acc | "
            "live progress (median) | live levels (median) | deaths (median) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, points in sorted(by_config.items()):
            best = max(points, key=lambda p: p["val"].get("accuracy", 0))
            live = best.get("live", {})
            params = best.get("val", {}).get("parameters")
            lines.append(
                f"| {name} | {params or '—'} | {'yes' if best.get('blind') else 'no'} | "
                f"{best['step']} | {_fmt(best['val'].get('loss'), 4)} | "
                f"{best['val'].get('accuracy', 0) * 100:.2f}% | "
                f"{best['val'].get('macro_accuracy', 0) * 100:.2f}% | "
                f"{_live_cell(live, 'total_progress')} | "
                f"{_live_cell(live, 'levels_reached')} | "
                f"{_live_cell(live, 'deaths')} |"
            )
        lines.append("")

        # Curves, so a partial night is still legible.
        lines += ["## Curves (val accuracy / live median progress by step)", ""]
        for name, points in sorted(by_config.items()):
            points = sorted(points, key=lambda p: p["step"])
            acc = " ".join(
                f"{p['step']}:{p['val'].get('accuracy', 0) * 100:.1f}%" for p in points
            )
            prog = " ".join(
                f"{p['step']}:{_live_cell(p.get('live', {}), 'total_progress')}"
                for p in points
            )
            lines += [f"- **{name}** accuracy — {acc}", f"  progress — {prog}"]
        lines.append("")

        # RARE token behaviour, requested explicitly.
        lines += ["## RARE token", ""]
        for name, points in sorted(by_config.items()):
            best = max(points, key=lambda p: p["val"].get("accuracy", 0))
            names = best.get("vocab_names") or []
            counts = best.get("val", {}).get("prediction_counts") or []
            labels = best.get("val", {}).get("label_counts") or []
            rare_idx = next(
                (i for i, n in enumerate(names) if n.startswith("RARE")), None
            )
            if rare_idx is None or not counts:
                continue
            total = sum(counts) or 1
            lines.append(
                f"- **{name}**: RARE predicted on {counts[rare_idx]:,} of {total:,} "
                f"val frames ({counts[rare_idx] * 100 / total:.4f}%); "
                f"true RARE labels {labels[rare_idx] if labels else '?'}"
            )
        lines.append("")

    if failed:
        lines += ["## What broke", ""]
        for f in failed:
            lines += [
                f"- **{f['config']}** after {_fmt(f.get('wall_seconds'))}s: "
                f"`{f.get('error')}`"
            ]
        lines.append("")
    elif done:
        lines += ["## What broke", "", "Nothing — every config completed.", ""]

    return "\n".join(lines)


def write_summary(results_path: Path | str, out_path: Path | str) -> Path:
    records = read_records(results_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_summary(records))
    return out
