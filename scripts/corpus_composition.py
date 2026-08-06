"""§1 precondition: what is the training index actually made of, by level and by in-control status?

The directive's shares are computed over **in-control frames** (`pregame==1 & player_state==8`). It asks
explicitly whether the dataset's own filter already excludes non-in-control frames, and says to state the
answer rather than assume it.

**It does not.** Neither `FrameStackDataset` nor `runlength.build_index` references `pregame` or
`player_state` — `build_index` walks every observation row and keeps it if the action token differs from the
previous frame's. So the 77,916 run-length training samples include pregame frames, level transitions and
death animations.

That makes two separate questions, and only the second one bears on §1:

1. what share of the *corpus's frames* are in-control — the directive's table;
2. what share of the *training samples the model actually draws* are 1-1, and of those how many are
   in-control.

Both are measured here, per level, so the 1-1 exposure figure used to size C1 and C2 is the one the sampler
sees rather than a frame count that no training step ever touches.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.data import FrameStackDataset  # noqa: E402
from tasdata.bc.runlength import RunLengthDataset  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/corpus_composition.json"
IDXCACHE = ROOT / "data/level_index_map.npz"


def main() -> None:
    ctx = O.Ctx()
    base = FrameStackDataset(ctx.expert_train, ctx.vocab, stack=4, label_mode="token")
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    idx = {k: z[k] for k in ("rows", "joints", "lengths")}
    ds = RunLengthDataset(base, idx)

    # per training sample: which level, and was the labelled frame in control?
    rows = idx["rows"]
    n = len(rows)
    lvl = np.empty(n, dtype=object)
    inctl = np.zeros(n, dtype=bool)
    # run boundaries in global row space
    offs = base.offsets
    for run_id, entry in enumerate(base.index):
        tr = np.asarray(ctx.expert_train[run_id].trace)
        w, s = column(tr, "world"), column(tr, "stage")
        pg, ps = column(tr, "pregame"), column(tr, "player_state")
        last = len(base.tokens[run_id]) - 1
        lo, hi = int(offs[run_id]), int(offs[run_id + 1])
        sel = np.flatnonzero((rows >= lo) & (rows < hi))
        if not len(sel):
            continue
        local = rows[sel] - lo
        fidx = entry.frame_indices
        m = np.minimum(fidx[local] + base.label_offset, last)
        m = np.minimum(m, len(w) - 1)
        lvl[sel] = [f"{int(a)}-{int(b)}" for a, b in zip(w[m], s[m])]
        inctl[sel] = (pg[m] == 1) & (ps[m] == 8)

    by_lvl = collections.Counter(lvl)
    by_lvl_ctl = collections.Counter(lvl[inctl])
    n11 = int(by_lvl.get("1-1", 0))
    n11c = int(by_lvl_ctl.get("1-1", 0))

    # corpus frame-level shares, for comparison with the directive's table
    tot_frames = ctl_frames = ctl_11 = frames_11 = 0
    for run in ctx.expert_train:
        tr = np.asarray(run.trace)
        w, s = column(tr, "world"), column(tr, "stage")
        pg, ps = column(tr, "pregame"), column(tr, "player_state")
        c = (pg == 1) & (ps == 8)
        is11 = (w == 1) & (s == 1)
        tot_frames += len(w)
        ctl_frames += int(c.sum())
        frames_11 += int(is11.sum())
        ctl_11 += int((c & is11).sum())

    out = {
        "question": ("does the dataset's own filter exclude non-in-control frames, and what share of the "
                     "TRAINING SAMPLES is 1-1?"),
        "filter_answer": {
            "FrameStackDataset_filters_in_control": False,
            "build_index_filters_in_control": False,
            "evidence": ("neither module references `pregame` or `player_state`; build_index keeps every "
                         "observation row whose action token differs from the previous frame's"),
            "consequence": ("the 77,916 run-length training samples INCLUDE pregame frames, level "
                            "transitions and death animations")},
        "corpus_frames_expert_train_split": {
            "total_frames": tot_frames,
            "in_control_frames": ctl_frames,
            "in_control_share": ctl_frames / tot_frames,
            "frames_labelled_1_1": frames_11,
            "in_control_frames_1_1": ctl_11,
            "in_control_share_of_1_1_label": ctl_11 / max(1, frames_11),
            "note": ("the directive's table is over the WHOLE 34-run corpus (1,684,996 frames); this is "
                     "the 20-run expert-train split, which is what training actually reads")},
        "TRAINING_SAMPLES": {
            "total_run_samples": n,
            "samples_1_1": n11,
            "share_1_1": n11 / n,
            "samples_1_1_in_control": n11c,
            "share_1_1_in_control": n11c / n,
            "in_control_samples_total": int(inctl.sum()),
            "in_control_share_of_samples": float(inctl.mean()),
            "by_level_top12": dict(by_lvl.most_common(12)),
            "by_level_in_control_top12": dict(by_lvl_ctl.most_common(12))},
    }
    # exposure arithmetic at the measured optimum
    for steps in (1000, 15000):
        seen = steps * 64
        out.setdefault("exposure", {})[f"{steps}_steps"] = {
            "samples_drawn": seen,
            "epochs_over_full_index": seen / n,
            "expected_1_1_samples_seen": seen * (n11 / n),
            "epochs_over_1_1_if_restricted": seen / max(1, n11),
            "epochs_over_1_1_in_control_if_restricted": seen / max(1, n11c)}
    e = out["exposure"]["1000_steps"]
    out["verdict"] = (
        f"**The dataset does NOT filter to in-control frames** — neither `FrameStackDataset` nor "
        f"`build_index` references `pregame` or `player_state`, so {100 - out['TRAINING_SAMPLES']['in_control_share_of_samples'] * 100:.1f}% "
        f"of training samples are out-of-control frames (pregame, transitions, death animations). "
        f"**1-1 is {n11:,} of {n:,} training samples ({n11 / n * 100:.1f}%), of which {n11c:,} are "
        f"in-control ({n11c / n * 100:.1f}%).** At the measured optimum of 1,000 steps × batch 64 = "
        f"{e['samples_drawn']:,} draws, the model sees about **{e['expected_1_1_samples_seen']:.0f} 1-1 "
        f"samples — {e['epochs_over_1_1_if_restricted']:.1f} epochs' worth if restricted**, against "
        f"{e['epochs_over_full_index']:.2f} epochs over the full index.")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "by_level"}, indent=2, default=str)[:2600])
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT}")
    # cache the per-sample level map so the arms can restrict without recomputing
    np.savez_compressed(IDXCACHE, rows=rows,
                        level=np.array([str(x) for x in lvl]), in_control=inctl)
    print(f"wrote {IDXCACHE}")


if __name__ == "__main__":
    main()
