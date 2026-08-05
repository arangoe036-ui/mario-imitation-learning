"""Is the negative timing lift real, or an artifact of scoring inputs the model never decides on?

The fifty-first block reported the x-matched onset lift as **negative at pipes 1, 2 and 4** and concluded
the policy has no timing anywhere. **That estimator is confounded.** Under run-length encoding the model is
trained *only at run boundaries*; the baseline "non-onset frames at the same x" is overwhelmingly **mid-run**,
at a mean expert run length of 12.6. So it compared in-distribution onsets against off-distribution
neighbours, and **the sign of the difference is not interpretable.**

Corrected baseline: **onset boundaries versus NON-A RUN STARTS** — frames where the expert began a run whose
combo contains no A. That holds "this is a decision point" fixed, which is the one thing the previous version
did not.

Both forms are computed on the identical rows and reported side by side, so the size of the artifact is
visible rather than argued:

| baseline | what it measures |
|---|---|
| all non-onset frames | the fifty-first block's estimator — confounded, mostly mid-run |
| **non-A run starts only** | **the timing distinction, decision point held fixed** |

**Definitions are token-level, not raw-byte**, because p(A) is a sum over the head's A-containing *classes*
and the model's own notion of "the action changed" is a token change. The raw-byte onset count is reported
beside the token count as a cross-check against the fifty-first block's table.

**The estimator is stratified, never pooled** — within each 16-px bin, mean p(A) at onsets minus mean p(A) at
the baseline set; then a weighted average over bins with the onset count as the weight. Pooling re-imports the
positional gradient, which is the defect that voided the earlier +0.067.

**Intervals bootstrap over ONSETS**, the correct cluster: each onset contributes exactly one frame, and the
baseline means are held fixed.

Forward passes only: expert observations are on disk as `frames.npy`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.runlength import N_BUCKETS, joint_size  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "data/bc_phase1/runlength.pt"
OUT = ROOT / "data/onset_lift_boundary_matched.json"
BIN = 16
BOOT = 4000

#: obstacle -> (window lo, hi). Identical to the fifty-first block's windows, so the two runs are comparable.
WINDOWS = {
    "goomba_288": (180, 320),
    "pipe1_432": (370, 480),
    "pipe2_592": (530, 645),
    "pipe3_720": (660, 775),
    "pipe4_912": (850, 980),
    "koopas_1216": (1150, 1265),
    "gap_1380": (1320, 1430),
}
#: rows whose n_onsets is below this are screens; the fifty-first block's load-bearing rows are the first three
THIN = 20


def stratified_lift(xs, pa, ons, base, bin_px=BIN):
    """Onset-minus-baseline p(A), averaged over x bins, weighted by onset count.

    `base` is a boolean mask selecting the comparison set; it is *not* assumed to be `~ons`, which is the
    whole point of this script.
    """
    bins = {}
    for lo in range(int(xs.min()) // bin_px * bin_px, int(xs.max()) + bin_px, bin_px):
        m = (xs >= lo) & (xs < lo + bin_px)
        o, n = pa[m & ons], pa[m & base]
        if len(o) and len(n):
            bins[lo] = {"n_onset": int(len(o)), "n_base": int(len(n)),
                        "onset_mean": float(o.mean()), "base_mean": float(n.mean()),
                        "diff": float(o.mean() - n.mean())}
    if not bins:
        return None, {}, 0
    w = np.array([b["n_onset"] for b in bins.values()], dtype=float)
    d = np.array([b["diff"] for b in bins.values()], dtype=float)
    #: onsets that fall in a bin with no baseline frame contribute nothing and are counted
    used = int(w.sum())
    return float((w * d).sum() / w.sum()), bins, used


def boot_lift(xs, pa, ons, base, reps=BOOT, seed=0):
    """Bootstrap the stratified lift by resampling ONSETS; baseline bin means stay fixed."""
    rng = np.random.default_rng(seed)
    oi = np.flatnonzero(ons)
    if len(oi) < 3:
        return None
    base_bins = {}
    for lo in range(int(xs.min()) // BIN * BIN, int(xs.max()) + BIN, BIN):
        m = (xs >= lo) & (xs < lo + BIN)
        n = pa[m & base]
        if len(n):
            base_bins[lo] = float(n.mean())
    if not base_bins:
        return None
    vals = []
    for _ in range(reps):
        pick = rng.choice(oi, size=len(oi), replace=True)
        acc = {}
        for i in pick:
            b = int(xs[i]) // BIN * BIN
            if b in base_bins:
                acc.setdefault(b, []).append(pa[i])
        num = den = 0.0
        for b, v in acc.items():
            k = len(v)
            num += k * (float(np.mean(v)) - base_bins[b])
            den += k
        if den:
            vals.append(num / den)
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    n_cls = joint_size(ctx.vocab.size)
    a_mask = np.array([(ctx.vocab.decode_byte(c // N_BUCKETS) & A_BIT) > 0
                       for c in range(n_cls)])
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()
    ds = ctx.dataset(ctx.expert_train)

    lo_all = min(v[0] for v in WINDOWS.values())
    hi_all = max(v[1] for v in WINDOWS.values())
    #: (x, is_A_onset, is_run_start, token_has_A, raw_byte_onset, global row)
    recs = []
    for run_id, entry in enumerate(ds.index):
        tr = np.asarray(ctx.expert_train[run_id].trace)
        w, s, pg = column(tr, "world"), column(tr, "stage"), column(tr, "pregame")
        x, ps = column(tr, "x_position"), column(tr, "player_state")
        raw = ds.raw[run_id]
        tokens = ds.tokens[run_id]
        last = len(tokens) - 1
        # token-level A membership per movie frame, in exactly the head's terms
        has_a = np.array([(ctx.vocab.decode_byte(int(t)) & A_BIT) > 0 for t in tokens])
        base_off = int(ds.offsets[run_id])
        fidx = entry.frame_indices
        for row in range(entry.n_obs):
            m = min(int(fidx[row]) + ds.label_offset, last)
            if m >= len(x) or not (w[m] == 1 and s[m] == 1 and pg[m] == 1 and ps[m] == 8):
                continue
            if not (lo_all <= x[m] <= hi_all):
                continue
            p = max(m - 1, 0)
            recs.append((int(x[m]),
                         bool(has_a[m] and not has_a[p] and m > 0),
                         bool(m == 0 or tokens[p] != tokens[m]),
                         bool(has_a[m]),
                         bool((int(raw[m]) & A_BIT) and not (int(raw[p]) & A_BIT) and m > 0),
                         base_off + row))
    xs = np.asarray([r[0] for r in recs])
    ons = np.asarray([r[1] for r in recs])
    starts = np.asarray([r[2] for r in recs])
    tok_a = np.asarray([r[3] for r in recs])
    raw_ons = np.asarray([r[4] for r in recs])
    rows = [r[5] for r in recs]
    nonA_starts = starts & ~tok_a
    print(f"expert 1-1 surface frames in x {lo_all}-{hi_all}: {len(recs):,}")
    print(f"  token A-onsets {int(ons.sum())}   raw-byte A-onsets {int(raw_ons.sum())}"
          f"   (agree on {int((ons == raw_ons).sum())}/{len(recs)})")
    print(f"  run starts {int(starts.sum())} ({starts.mean():.1%} of frames)"
          f"   non-A run starts {int(nonA_starts.sum())}")
    print(f"  of the OLD baseline (all non-onset frames), mid-run fraction: "
          f"{float((~ons & ~starts).sum()) / float((~ons).sum()):.1%}\n", flush=True)

    pa = []
    for i in range(0, len(rows), 256):
        ob = torch.stack([ds[j][0] for j in rows[i:i + 256]])
        with torch.no_grad():
            p = torch.softmax(policy(ob), dim=-1).numpy()
        pa.extend(p[:, a_mask].sum(axis=1).tolist())
        if (i // 256) % 8 == 0:
            print(f"    forward {min(i + 256, len(rows)):,}/{len(rows):,}", flush=True)
    pa = np.asarray(pa)

    out = {"question": ("with the decision point held fixed, is the x-matched timing lift still negative "
                        "at pipe 2?"),
           "checkpoint": CKPT.name, "bin_px": BIN, "boot_reps": BOOT,
           "estimator": ("stratified: within each 16px bin, mean p(A) at expert A-onsets minus mean p(A) "
                         "over the baseline set; weighted average over bins by onset count"),
           "interval": "bootstrap over ONSETS (each onset contributes one frame), baseline means fixed",
           "definitions": {
               "onset": "token-level: the run beginning at this frame contains A, the previous run did not",
               "baseline_all_non_onset": "every frame in the window that is not an A-onset (CONFOUNDED)",
               "baseline_nonA_starts": "frames beginning a run whose combo contains no A (CORRECTED)"},
           "corpus": {
               "n_frames": len(recs), "n_token_onsets": int(ons.sum()),
               "n_raw_byte_onsets": int(raw_ons.sum()),
               "onset_definitions_agree": int((ons == raw_ons).sum()) == len(recs),
               "n_run_starts": int(starts.sum()), "n_nonA_run_starts": int(nonA_starts.sum()),
               "run_start_fraction": float(starts.mean()),
               "old_baseline_mid_run_fraction": float((~ons & ~starts).sum()) / float((~ons).sum())},
           "obstacles": {}}

    hdr = (f"{'obstacle':13s} {'ons':>4s} | {'OLD lift':>9s} {'95% CI':>18s} {'nbase':>6s} "
           f"| {'NEW lift':>9s} {'95% CI':>18s} {'nbase':>6s} | {'sign flip':>9s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, (lo, hi) in WINDOWS.items():
        m = (xs >= lo) & (xs <= hi)
        rec = {"window": [lo, hi], "n_frames": int(m.sum()), "n_onsets": int((ons & m).sum())}
        if not m.any() or (ons & m).sum() == 0:
            out["obstacles"][name] = rec
            print(f"{name:13s} {'   -':>4s} | no onsets")
            continue
        old_l, old_b, old_used = stratified_lift(xs[m], pa[m], ons[m], ~ons[m])
        new_l, new_b, new_used = stratified_lift(xs[m], pa[m], ons[m], nonA_starts[m])
        old_ci = boot_lift(xs[m], pa[m], ons[m], ~ons[m])
        new_ci = boot_lift(xs[m], pa[m], ons[m], nonA_starts[m])
        swing = [b["base_mean"] for b in old_b.values()]
        rec.update({
            "n_nonA_starts_in_window": int((nonA_starts & m).sum()),
            "thin": bool((ons & m).sum() < THIN),
            "baseline_all_non_onset": {
                "lift": old_l, "ci_onset_bootstrap": old_ci, "n_base": int((~ons & m).sum()),
                "onsets_used": old_used, "bins": old_b},
            "baseline_nonA_starts": {
                "lift": new_l, "ci_onset_bootstrap": new_ci,
                "n_base": int((nonA_starts & m).sum()), "onsets_used": new_used, "bins": new_b},
            "spatial_swing_old_baseline": (max(swing) - min(swing)) if swing else None,
            "sign_flipped_negative_to_positive": bool(
                old_l is not None and new_l is not None and old_l < 0 and new_l > 0)})
        out["obstacles"][name] = rec
        f_old = f"[{old_ci[0]:+.3f},{old_ci[1]:+.3f}]" if old_ci else "n<3"
        f_new = f"[{new_ci[0]:+.3f},{new_ci[1]:+.3f}]" if new_ci else "n<3"
        flip = "YES" if rec["sign_flipped_negative_to_positive"] else ""
        print(f"{name:13s} {rec['n_onsets']:4d} | {old_l:+9.3f} {f_old:>18s} "
              f"{int((~ons & m).sum()):6d} | "
              f"{(new_l if new_l is not None else float('nan')):+9.3f} {f_new:>18s} "
              f"{int((nonA_starts & m).sum()):6d} | {flip:>9s}", flush=True)

    p2 = out["obstacles"]["pipe2_592"]
    p1 = out["obstacles"]["pipe1_432"]
    g = out["obstacles"]["goomba_288"]

    def pair(rec):
        return (rec["baseline_all_non_onset"]["lift"], rec["baseline_nonA_starts"]["lift"],
                rec["baseline_nonA_starts"]["ci_onset_bootstrap"])
    p2_old, p2_new, p2_ci = pair(p2)
    p1_old, p1_new, p1_ci = pair(p1)
    g_old, g_new, g_ci = pair(g)

    #: "positive" requires the interval to exclude zero -- a positive point estimate whose CI spans zero
    #: does not void a prior negative finding, it only fails to confirm it.
    p2_pos = p2_new is not None and p2_ci is not None and p2_ci[0] > 0
    p2_neg = p2_new is not None and p2_ci is not None and p2_ci[1] < 0
    out["comparison"] = {
        "pipe2": {"old": p2_old, "new": p2_new, "new_ci": p2_ci,
                  "delta_new_minus_old": (p2_new - p2_old) if None not in (p2_new, p2_old) else None},
        "pipe1": {"old": p1_old, "new": p1_new, "new_ci": p1_ci,
                  "delta_new_minus_old": (p1_new - p1_old) if None not in (p1_new, p1_old) else None},
        "goomba": {"old": g_old, "new": g_new, "new_ci": g_ci,
                   "delta_new_minus_old": (g_new - g_old) if None not in (g_new, g_old) else None},
        "pipe2_positive_excluding_zero": bool(p2_pos),
        "pipe2_negative_excluding_zero": bool(p2_neg),
        "n_obstacles_sign_flipped": sum(
            1 for v in out["obstacles"].values()
            if v.get("sign_flipped_negative_to_positive")),
    }
    if p2_pos:
        out["verdict"] = (
            f"**THE FIFTY-FIRST BLOCK'S CONCLUSION IS VOID.** With the decision point held fixed, the "
            f"pipe-2 timing lift is {p2_new:+.3f} [{p2_ci[0]:+.3f}, {p2_ci[1]:+.3f}] -- POSITIVE and "
            f"excluding zero -- against {p2_old:+.3f} under the confounded baseline. The negative values "
            f"were an artifact of scoring mid-run frames the model never decides on. **'The policy has no "
            f"timing anywhere' comes off the board, and so does the case for changing the observation.** "
            f"STOP HERE: the scale-up is now an optimisation rather than a rescue, and the owner should "
            f"choose the order.")
    elif p2_neg:
        out["verdict"] = (
            f"**THE FINDING STANDS.** With the decision point held fixed -- onsets against non-A run "
            f"starts only -- the pipe-2 timing lift is {p2_new:+.3f} [{p2_ci[0]:+.3f}, {p2_ci[1]:+.3f}], "
            f"still negative and still excluding zero, against {p2_old:+.3f} under the confounded "
            f"baseline. The confound was real ({out['corpus']['old_baseline_mid_run_fraction']:.1%} of the "
            f"old baseline was mid-run) but it was not producing the sign. **The policy genuinely does not "
            f"discriminate when to jump where it succeeds.**")
    else:
        out["verdict"] = (
            f"**INCONCLUSIVE AT PIPE 2, AND THAT IS THE ANSWER.** The corrected lift is {p2_new:+.3f} "
            f"[{p2_ci[0]:+.3f}, {p2_ci[1]:+.3f}] -- the interval spans zero, so it neither confirms the "
            f"fifty-first block's negative nor establishes a positive. Against the confounded baseline's "
            f"{p2_old:+.3f}. **'No timing anywhere' is no longer supported at pipe 2, but the corrected "
            f"estimator does not replace it with a positive signal either** -- with "
            f"{p2['n_onsets']} onsets and {p2['n_nonA_starts_in_window']} baseline decision points, the "
            f"honest statement is that the timing lift at pipe 2 is not measurably different from zero.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
