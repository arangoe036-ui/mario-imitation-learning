"""§2: fix the probe's power, because the whole block rests on it.

Block 63 reported the on-top-versus-at-face distinction as "present in the features but not linearly
reachable" from a **linear AUC 0.651 (p=0.1725)** against an **MLP AUC 0.743 (p=0.0100)**. That is a
difference in **significance**, not a tested difference in AUC — the two were never compared — and with 11
positive states the gap is quite possibly not significant. **The claim is reasonable and not established.**

Two fixes:

1. **Bootstrap the AUC difference over STATES** and report its interval. If it spans zero, say so plainly.
2. **More positives.** 34 of the 200 round-1 failures were on-top and the probe used 11, because it only
   looked at the 60 states that were searched. Every one of the 200 failures has a recorded `fail_frame`,
   `fail_y` and byte prefix, so features can be captured at all of them: **~34 positives against ~166
   negatives.**

Protocol is unchanged and deliberately so: grouped cross-validation with folds over **states**, never frames,
and a **state-level** label permutation null. With 64 features and this many states a linear probe will
separate noise without one.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.probe_retreat_states import auc, features, logistic_cv, perm_p  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EPS = ROOT / "data/dagger_round1.partial.json"
OUT = ROOT / "data/probe_ontop_power.json"
CACHE = ROOT / "data/probe_ontop_features_all.npz"

ARM = "P_84_cnn32"
OFFSETS = [0, -6, -12]
GROUND_Y = 432
N_BOOT = 4000
N_PERM = 300
N_FOLDS = 8
ARM_BUDGET_S = 30 * 60


def wall_bin(x):
    for name, lo, hi in (("goomba_288", 240, 340), ("pipe1_432", 400, 500),
                         ("pipe2_592", 560, 660), ("pipe3_720", 660, 760),
                         ("pipe4_912", 860, 1000), ("koopas_1216", 1150, 1300),
                         ("frontier_1504", 1450, 1600), ("gap_1380", 1300, 1450)):
        if lo <= x < hi:
            return name
    return f"other_{int(x) // 200 * 200}"


def mlp_cv(X, y, groups, seed=0, folds=N_FOLDS, hid=32, epochs=400):
    rng = np.random.default_rng(seed)
    uq = np.unique(groups).copy()
    rng.shuffle(uq)
    out = np.zeros(len(y))
    for f in np.array_split(uq, folds):
        te = np.isin(groups, f)
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        B = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        torch.manual_seed(seed)
        m = torch.nn.Sequential(torch.nn.Linear(A.shape[1], hid), torch.nn.ReLU(),
                                torch.nn.Linear(hid, 1))
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)
        w = float((len(yt) - yt.sum()) / max(1.0, float(yt.sum())))
        for _ in range(epochs):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                m(A).squeeze(-1), yt, pos_weight=torch.tensor(w))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out[te] = m(B).squeeze(-1).numpy()
    return out


def boot_auc_diff(y, s_lin, s_mlp, groups, n=N_BOOT, seed=0):
    """Bootstrap AUC(MLP) - AUC(linear) by resampling STATES, keeping both scores paired."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a1, a2 = auc(yy, s_lin[idx]), auc(yy, s_mlp[idx])
        if a1 is not None and a2 is not None:
            vals.append(a2 - a1)
    v = np.asarray(vals)
    return (float(v.mean()), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            float((v <= 0).mean()))


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60 * 60)
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        X = z["X"]
        meta = [json.loads(m) for m in z["meta"]]
        print(f"reusing {len(X)} cached vectors", flush=True)
    else:
        ctx = O.Ctx()
        start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
        eps = json.loads(EPS.read_text())["episodes"]
        policy, cfg, _ = G.load_ckpt(ARM)
        s_ = cfg.frame_size
        X, meta = [], []
        sess = None
        try:
            with time_limit(min(ARM_BUDGET_S, dl.remaining() - 120), "capture"):
                sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
                warm_session(sess, start.frame)
                for i, ep in enumerate(eps):
                    if ep.get("completed"):
                        continue
                    if dl.remaining() < 180:
                        break
                    for off in OFFSETS:
                        f = max(0, ep["fail_frame"] + off)
                        obs = sess.reset(start.frame)
                        for b in ep["bytes"][:f]:
                            obs = sess.step(int(b))
                        win = np.zeros((cfg.stack, s_, s_), np.uint8)
                        win[:] = _resize_gray(obs.rgb, (s_, s_))
                        X.append(features(policy, cfg, win))
                        meta.append({"state": ep["seed"], "wall": wall_bin(ep["fail_x"]),
                                     "on_top": bool(ep["fail_y"] < GROUND_Y - 8),
                                     "fail_x": int(ep["fail_x"]), "offset": off})
                    if (i + 1) % 40 == 0:
                        print(f"  {dl.stamp()} {i + 1}/{len(eps)} episodes", flush=True)
        finally:
            if sess is not None:
                try:
                    sess.close()
                except BaseException:
                    pass
        X = np.asarray(X, dtype=np.float32)
        np.savez_compressed(CACHE, X=X, meta=np.array([json.dumps(m) for m in meta]))
        print(f"captured {len(X)} vectors", flush=True)
    if len(X) == 0:
        print("nothing captured")
        return

    groups = np.array([m["state"] for m in meta])
    y = np.array([1 if m["on_top"] else 0 for m in meta])
    n_pos = len({m["state"] for m in meta if m["on_top"]})
    n_neg = len({m["state"] for m in meta if not m["on_top"]})
    print(f"states: {n_pos} on-top, {n_neg} at-face", flush=True)

    s_lin = logistic_cv(X, y, groups, n_folds=N_FOLDS)
    s_mlp = mlp_cv(X, y, groups)
    a_lin, a_mlp = auc(y, s_lin), auc(y, s_mlp)
    p_lin, _ = perm_p(X, y, groups, a_lin, n_perm=N_PERM)

    # permutation null for the MLP, same state-level scheme
    rng = np.random.default_rng(1)
    uq = np.unique(groups)
    lab = {g: y[groups == g][0] for g in uq}
    null = []
    for _ in range(150):
        perm = rng.permutation([lab[g] for g in uq])
        mp = dict(zip(uq, perm))
        yy = np.array([mp[g] for g in groups])
        a = auc(yy, mlp_cv(X, yy, groups, seed=1))
        if a is not None:
            null.append(a)
    null = np.asarray(null)
    p_mlp = float((np.abs(null - 0.5) >= abs(a_mlp - 0.5) - 1e-12).mean())

    dmean, dci, p_le0 = boot_auc_diff(y, s_lin, s_mlp, groups)

    out = {"arm": ARM, "n_samples": int(len(X)), "n_states": int(len(uq)),
           "n_on_top_states": n_pos, "n_at_face_states": n_neg, "offsets": OFFSETS,
           "cv": f"grouped {N_FOLDS}-fold over STATES",
           "null": "state-level label permutation",
           "block63_values": {"linear_auc": 0.651, "linear_p": 0.1725,
                              "mlp_auc": 0.743, "mlp_p": 0.0100,
                              "n_on_top_states": 11},
           "linear": {"auc": a_lin, "perm_p": p_lin, "n_perm": N_PERM},
           "mlp": {"auc": a_mlp, "perm_p": p_mlp, "n_perm": len(null),
                   "null_auc_mean": float(null.mean())},
           "auc_difference_mlp_minus_linear": {
               "mean": dmean, "ci95_bootstrap_over_states": dci,
               "fraction_of_bootstrap_at_or_below_zero": p_le0,
               "excludes_zero": bool(dci[0] > 0),
               "note": ("this is the test block 63 did NOT run: a difference in significance is not a "
                        "tested difference in AUC")}}
    lin_sep = bool(a_lin is not None and p_lin < 0.05)
    mlp_sep = bool(a_mlp is not None and p_mlp < 0.05)
    gap = out["auc_difference_mlp_minus_linear"]["excludes_zero"]
    out["framing"] = ("the head IS the bottleneck" if (mlp_sep and not lin_sep and gap)
                      else "the head MAY be the bottleneck" if mlp_sep
                      else "no separation at all -- the premise of §1 is not supported")
    out["verdict"] = (
        f"At {n_pos} on-top states against {n_neg} at-face (block 63 had 11): **linear AUC "
        f"{a_lin:.3f} (p={p_lin:.4f}), MLP AUC {a_mlp:.3f} (p={p_mlp:.4f})**. The difference "
        f"MLP−linear is **{dmean:+.3f}, 95% CI [{dci[0]:+.3f}, {dci[1]:+.3f}]** bootstrapped over "
        f"states, so it **{'EXCLUDES' if gap else 'does NOT exclude'} zero**. "
        f"⇒ Report §1 as **\"{out['framing']}\"**.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
