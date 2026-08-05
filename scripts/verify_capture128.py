"""Did the 128x128 re-capture actually capture the same runs, or something that merely has the right shape?

§3 asks for the verifier re-run on the new capture. `batch` already runs `verify_smb` per run and writes
`sync.json`, so re-reading those is necessary but weak: **silent failure #2 in this project was 17,868 frames
of attract mode with a correct frame count**, which a shape check and a frame total both pass.

Five checks, and only the first is the one the directive literally asked for:

1. **`sync.json`** -- every hard check passed, `SYNC: PASS`, and the level span.
2. **RAM trace byte-identity against the 84x84 corpus.** The RAM trace is resolution-independent, so a
   correct re-capture of the same movie must reproduce it *exactly*. This is the check that would have caught
   failure #2, and it is only available because the old corpus was kept.
3. **`actions.npy` identity** -- same inputs applied.
4. **Shape and row count** -- `(n, 128, 128)`, `n` equal to the 84x84 run's.
5. **Frames are not constant.** Per-run standard deviation across time on a sample of pixels: an attract-mode
   or frozen capture has near-zero temporal variance no matter how many rows it has.

Runs present in one corpus and not the other are reported, not skipped -- a missing run silently reduces the
training set and would confound resolution with data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "data/runs"
NEW = ROOT / "data/runs128"
OUT = ROOT / "data/verify_capture128.json"
SIZE = 128
#: pixels sampled for the temporal-variance check; a full 25 GB read is not needed to see a frozen screen
N_PIX, N_ROWS = 64, 400


def temporal_variance(path: Path, rng) -> float:
    """Mean per-pixel std across time on a random sample of rows and pixels."""
    a = np.load(path, mmap_mode="r")
    n = a.shape[0]
    rows = np.unique(rng.integers(0, n, size=min(N_ROWS, n)))
    ys = rng.integers(0, a.shape[1], size=N_PIX)
    xs = rng.integers(0, a.shape[2], size=N_PIX)
    sample = np.asarray(a[rows][:, ys, xs], dtype=np.float32)
    return float(sample.std(axis=0).mean())


def main() -> None:
    rng = np.random.default_rng(0)
    old_names = {p.name for p in OLD.iterdir() if (p / "frames.npy").exists()}
    new_names = {p.name for p in NEW.iterdir() if (p / "frames.npy").exists()} if NEW.exists() \
        else set()
    out = {"old_corpus": str(OLD.relative_to(ROOT)), "new_corpus": str(NEW.relative_to(ROOT)),
           "n_old": len(old_names), "n_new": len(new_names),
           "missing_from_new": sorted(old_names - new_names),
           "extra_in_new": sorted(new_names - old_names),
           "checks": ["sync_json_pass", "trace_identical", "actions_identical",
                      "shape_and_rows", "frames_not_constant"],
           "runs": {}}
    print(f"old {len(old_names)} runs | new {len(new_names)} runs | "
          f"missing from new: {len(old_names - new_names)}\n")
    hdr = (f"{'run':22s} {'sync':>6s} {'trace':>7s} {'acts':>6s} {'shape':>16s} "
           f"{'rows':>8s} {'tvar':>7s} {'levels':>9s}")
    print(hdr)
    print("-" * len(hdr))
    n_ok = 0
    for name in sorted(new_names):
        r: dict = {}
        d = NEW / name
        syncp = d / "sync.json"
        sync = json.loads(syncp.read_text()) if syncp.exists() else {}
        checks = sync.get("checks") or []
        # Advisory checks are context, not failures -- `verify_smb` marks them so explicitly.
        failed = [c["name"] for c in checks
                  if not c.get("passed", True) and not c.get("advisory", False)]
        r["sync_json_pass"] = bool(syncp.exists() and sync.get("synced") and not failed)
        r["synced"] = sync.get("synced")
        r["sync_failed_checks"] = failed
        r["sync_advisory_failures"] = [c["name"] for c in checks
                                       if not c.get("passed", True) and c.get("advisory", False)]
        r["n_checks"] = len(checks)
        lv = sync.get("levels_reached") or []
        r["levels"] = len(lv)
        r["first_last_level"] = [lv[0], lv[-1]] if lv else [None, None]
        r["reason"] = sync.get("reason")

        a_new = np.load(d / "frames.npy", mmap_mode="r")
        r["shape"] = list(a_new.shape)
        r["shape_and_rows"] = (len(a_new.shape) == 3 and a_new.shape[1] == SIZE
                               and a_new.shape[2] == SIZE)
        if name in old_names:
            a_old = np.load(OLD / name / "frames.npy", mmap_mode="r")
            r["rows_new"], r["rows_old"] = int(a_new.shape[0]), int(a_old.shape[0])
            r["shape_and_rows"] = bool(r["shape_and_rows"]
                                       and a_new.shape[0] == a_old.shape[0])
            t_old, t_new = OLD / name / "trace.npy", d / "trace.npy"
            r["trace_identical"] = bool(
                t_old.exists() and t_new.exists()
                and np.array_equal(np.load(t_old), np.load(t_new)))
            ac_o, ac_n = OLD / name / "actions.npy", d / "actions.npy"
            r["actions_identical"] = bool(
                ac_o.exists() and ac_n.exists()
                and np.array_equal(np.load(ac_o), np.load(ac_n)))
        else:
            r.update({"rows_new": int(a_new.shape[0]), "rows_old": None,
                      "trace_identical": None, "actions_identical": None,
                      "not_in_old_corpus": True})
        tv = temporal_variance(d / "frames.npy", rng)
        r["temporal_pixel_std"] = round(tv, 3)
        r["frames_not_constant"] = tv > 1.0
        # **Absolute sync is the WRONG criterion.** 9 of the 34 movies desync in the ORIGINAL 84x84
        # corpus too (README: "34 runs captured, 25 verified in sync") -- that is a property of those
        # movies, not of this capture, and the frozen split only draws on synced runs. What must hold
        # is that the new capture *agrees with the old one* run for run.
        if name in old_names:
            osync = json.loads((OLD / name / "sync.json").read_text()).get("synced") \
                if (OLD / name / "sync.json").exists() else None
            r["old_synced"] = osync
            r["sync_agrees_with_old"] = (r["synced"] == osync)
        else:
            r["old_synced"] = None
            r["sync_agrees_with_old"] = None
        r["all_pass"] = bool(r["shape_and_rows"] and r["frames_not_constant"]
                             and r["trace_identical"] is not False
                             and r["actions_identical"] is not False
                             and r["sync_agrees_with_old"] is not False)
        n_ok += bool(r["all_pass"])
        out["runs"][name] = r

        def m(v):
            return "-" if v is None else ("PASS" if v else "FAIL")
        print(f"{name:22s} {m(r['sync_json_pass']):>6s} {m(r['trace_identical']):>7s} "
              f"{m(r['actions_identical']):>6s} {str(tuple(r['shape'])):>16s} "
              f"{r['rows_new']:>8,} {tv:>7.1f} {str(r['levels']):>9s}", flush=True)

    out["n_all_pass"] = n_ok
    out["n_trace_mismatch"] = sum(1 for v in out["runs"].values()
                                  if v.get("trace_identical") is False)
    out["n_constant_frames"] = sum(1 for v in out["runs"].values()
                                   if not v.get("frames_not_constant"))
    out["n_synced_new"] = sum(1 for v in out["runs"].values() if v.get("synced"))
    out["n_sync_disagreements"] = sum(1 for v in out["runs"].values()
                                      if v.get("sync_agrees_with_old") is False)
    # The frozen split is what training actually reads; it must be present and synced.
    split = json.loads((ROOT / "data/split.json").read_text())["splits"]
    train = list(split["train"])
    out["train_split"] = {
        "n": len(train),
        "present_in_new": sum(1 for n_ in train if n_ in new_names),
        "synced_in_new": sum(1 for n_ in train if out["runs"].get(n_, {}).get("synced")),
        "trace_identical": sum(1 for n_ in train
                               if out["runs"].get(n_, {}).get("trace_identical")),
        "missing": [n_ for n_ in train if n_ not in new_names]}
    ts = out["train_split"]
    complete = not out["missing_from_new"]
    all_pass = n_ok == len(new_names) and len(new_names) > 0
    train_ok = (ts["present_in_new"] == ts["n"] == ts["synced_in_new"] == ts["trace_identical"])
    out["capture_usable"] = bool(complete and all_pass and train_ok)
    if complete and all_pass and train_ok:
        out["verdict"] = (
            f"**CAPTURE VERIFIES.** {len(new_names)} runs at 128x128; RAM trace and applied actions "
            f"byte-identical to the 84x84 corpus on every run, row counts equal, none frozen, and sync "
            f"status agrees run-for-run ({out['n_synced_new']} synced, the same "
            f"{out['n_synced_new']} as the old corpus -- the 9 desyncs are a property of those movies, "
            f"not of this capture). **All {ts['n']} runs of the frozen train split are present, synced, "
            f"and trace-identical**, so R and RT train on exactly the data B did.")
    elif not complete:
        out["verdict"] = (
            f"**CAPTURE INCOMPLETE: {len(out['missing_from_new'])} of {len(old_names)} runs missing** "
            f"({', '.join(out['missing_from_new'][:5])}). Training an arm on a smaller corpus would "
            f"confound resolution with which runs were seen, so the 128 arms must not run until these "
            f"are captured.")
    else:
        out["verdict"] = (
            f"**CAPTURE PRESENT BUT {len(new_names) - n_ok} RUN(S) FAILED A CHECK** "
            f"(trace mismatches {out['n_trace_mismatch']}, frozen {out['n_constant_frames']}). "
            f"A re-capture is where silent failure #2 lived; do not train on it until resolved.")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
