"""Audit every run's declared category against what its replay actually did.

The declared category comes from the TASVideos branch name. It is a claim, not a
measurement, and it has already been wrong once: the runs labelled ``warps-glitchless``
are glitchless *warps* runs, not warpless ones -- the qualifier describes the style and
the route word describes the route, and reading the label as a whole misled the
glitchless-vs-glitchy experiment into thinking warpless-glitchless data existed.

This checks the label against two measured facts: the route the replay took, and how many
distinct levels it cleared. It cannot check "glitchless", because nothing in the pipeline
measures glitch use -- which is itself the finding worth recording.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Levels a route is expected to visit, measured rather than assumed.
ROUTE_LEVELS = {"warpless": 32, "warps": 8, "all-items": 32}


def main() -> None:
    split = json.loads((ROOT / "data/split.json").read_text())["splits"]
    member = {n: b for b, ns in split.items() for n in ns}

    rows, mismatches, unverifiable = [], [], []
    for m in sorted(glob.glob(str(ROOT / "data/runs/*/manifest.json"))):
        name = os.path.basename(os.path.dirname(m))
        j = json.loads(Path(m).read_text())
        declared = str(j.get("category", ""))
        route = str(j.get("measured_route", ""))
        levels = j.get("measured_levels")
        entry = {
            "run": name, "declared": declared, "measured_route": route,
            "measured_levels": levels, "n_frames": j.get("n_frames"),
            "split": member.get(name, "not in split"), "synced": bool(j.get("synced")),
            "furthest_level": j.get("furthest_level"),
        }

        # The route word inside the declared label, ignoring style qualifiers.
        declared_route = declared.split("-")[0] if declared else ""
        if declared.startswith("all-items"):
            declared_route = "all-items"
        problems = []
        if route and declared_route and declared_route not in route and route != declared:
            problems.append(f"declared route '{declared_route}' != measured '{route}'")
        expected = ROUTE_LEVELS.get(route)
        if expected and levels is not None and levels != expected:
            problems.append(f"{levels} levels measured, {expected} expected for '{route}'")
        if "glitchless" in declared:
            unverifiable.append(entry | {
                "note": "the 'glitchless' claim is not measurable by this pipeline; "
                        "only the route part of the label was checked"})
        if problems:
            entry["problems"] = problems
            mismatches.append(entry)
        rows.append(entry)

    print(f"audited {len(rows)} runs\n")
    print(f"{'run':22s} {'declared':22s} {'measured route':15s} {'levels':>6s}  split")
    print("-" * 84)
    for e in rows:
        flag = "  <-- MISMATCH" if "problems" in e else ""
        print(f"{e['run']:22s} {e['declared']:22s} {e['measured_route']:15s} "
              f"{str(e['measured_levels']):>6s}  {e['split']}{flag}")

    print(f"\n{len(mismatches)} mismatch(es):")
    for e in mismatches:
        print(f"  {e['run']}: {'; '.join(e['problems'])}")

    print(f"\n{len(unverifiable)} run(s) carry an unverifiable 'glitchless' claim:")
    for e in unverifiable:
        print(f"  {e['run']}: declared '{e['declared']}', measured route "
              f"'{e['measured_route']}' ({e['measured_levels']} levels), split={e['split']}")

    out = ROOT / "data/category_audit.json"
    out.write_text(json.dumps(
        {"n_runs": len(rows), "mismatches": mismatches, "unverifiable_glitchless":
         unverifiable, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
