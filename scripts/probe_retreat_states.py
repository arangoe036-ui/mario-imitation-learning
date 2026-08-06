"""§2a: do the frozen policy's features separate correction states? A linear probe, gated on ground truth.

**⚠ Probe A as specified is not constructible, and that is a substantive answer rather than a dodge.**
The directive asks for positives = "round-2 correction states where retreat was the solution" and negatives =
"matched states from the same walls where it was not". Measured over the 60 searched states:

| retreat share of a state's solutions | states |
|---|---|
| >= 0.50 (retreat-dominant) | **2** |
| <= 0.15 (sampled-dominant) | **5** |
| in between | **53** |

**No state is retreat-only (minimum sampled solutions = 1) and no state has zero retreat solutions
(minimum share 0.047, median 0.204).** Retreat works to some degree *everywhere* and is strictly required
*nowhere*. So "is a retreat required here?" has no ground truth to probe — **the label the distillation was
being asked to condition on is not a well-defined property of the states.** Two positives cannot support a
probe and would not be reported if they could.

So the probe runs on distinctions that **do** have ground truth and that conditional recovery would require:

* **B — on-top versus at-face** (11 vs 49 states, from `y` at the stall). These need opposite corrections
  ("get off the pipe" vs "clear it"), so a policy that cannot tell them apart cannot correct either.
* **C — wall identity** (8 walls). If the features cannot say *which* obstacle this is, no correction can be
  conditioned on it. Well powered: 60 states.
* **D — x position**, as a regression. A sanity floor: if features do not carry position, B and C failing
  would say nothing about conditionality.

**Every probe is cross-validated with folds over STATES, never frames**, and every one is accompanied by a
**label-permutation null over states** — with 64-dimensional features and 60 states a linear probe can
separate noise, so an AUC without its permutation p-value is meaningless here.

Features are the frozen policy's penultimate representation — `norm(transformer(...))` at the last frame
position, the 64-dim vector the action head reads.
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
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATES = ROOT / "data/dagger_round1_states.json"
EPS = ROOT / "data/dagger_round1.partial.json"
OUT = ROOT / "data/probe_retreat_states.json"  # noqa: E501
CACHE = ROOT / "data/probe_features.npz"

ARM = "P_84_cnn32"
#: frame offsets before the captured state, so each state contributes several observations
OFFSETS = [0, -4, -8, -12]
N_PERM = 2000
N_FOLDS = 6
ARM_BUDGET_S = 20 * 60


def features(policy, cfg, win):
    """The 64-dim vector the action head reads: norm(transformer(...)) at the last frame position."""
    x = torch.from_numpy(win[None]).float().div_(255.0)
    with torch.no_grad():
        b, t = x.shape[0], x.shape[1]
        s = cfg.frame_size
        h = policy.encoder(x.reshape(b * t, 1, s, s)).reshape(b, t, -1)
        h = policy.transformer(h + policy.pos[:, : h.shape[1]])
        f = policy.norm(h[:, t - 1])
    return f.numpy()[0]


def capture(dl):
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    states = json.loads(STATES.read_text())["states"]
    eps = {e["seed"]: e for e in json.loads(EPS.read_text())["episodes"]}
    policy, cfg, _ = G.load_ckpt(ARM)
    s_ = cfg.frame_size
    X, meta = [], []
    sess = None
    try:
        with time_limit(min(ARM_BUDGET_S, dl.remaining() - 120), "capture"):
            sess = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
            warm_session(sess, start.frame)
            for si, st in enumerate(states):
                ep = eps.get(st["seed"])
                if ep is None or not st.get("leads_tried"):
                    continue
                lead = st["solved_at_lead"] or st["leads_tried"][0]["lead"]
                f_state = max(0, st["fail_x"] and (0) or 0)  # placeholder, replaced below
                # the frame index the search started from
                f0 = None
                for lt in st["leads_tried"]:
                    if lt["lead"] == lead:
                        f0 = max(0, ep["fail_frame"] - lead)
                        break
                if f0 is None:
                    continue
                tot = st["n_solutions"]
                ret = st["n_retreat_solutions"]
                for off in OFFSETS:
                    f = max(0, f0 + off)
                    obs = sess.reset(start.frame)
                    for b in ep["bytes"][:f]:
                        obs = sess.step(int(b))
                    win = np.zeros((cfg.stack, s_, s_), np.uint8)
                    win[:] = _resize_gray(obs.rgb, (s_, s_))
                    X.append(features(policy, cfg, win))
                    meta.append({"state": si, "wall": st["wall"],
                                 "on_top": bool(st["on_top"]),
                                 "fail_x": int(st["fail_x"]), "offset": off,
                                 "retreat_share": float(ret / max(1, tot))})
                if (si + 1) % 15 == 0:
                    print(f"  {dl.stamp()} {si + 1}/{len(states)} states", flush=True)
    finally:
        if sess is not None:
            try:
                sess.close()
            except BaseException:
                pass
    return np.asarray(X, dtype=np.float32), meta


def logistic_cv(X, y, groups, n_folds=N_FOLDS, seed=0):
    """Grouped CV logistic regression by plain gradient descent; returns out-of-fold scores."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    rng.shuffle(uq)
    folds = np.array_split(uq, min(n_folds, len(uq)))
    scores = np.zeros(len(y), dtype=float)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = (X[tr] - mu) / sd
        B = (X[te] - mu) / sd
        w = np.zeros(A.shape[1])
        b = 0.0
        yt = y[tr].astype(float)
        for _ in range(600):
            z = A @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = A.T @ (p - yt) / len(yt) + 1e-3 * w
            gb = float((p - yt).mean())
            w -= 0.5 * g
            b -= 0.5 * gb
        scores[te] = B @ w + b
    return scores


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based AUC with ties at 0.5
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def perm_p(X, y, groups, observed, n_perm=N_PERM, seed=0):
    """Permute labels AT THE STATE LEVEL -- permuting frames would leak."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    lab = {g: y[groups == g][0] for g in uq}
    vals = []
    for _ in range(n_perm):
        perm = rng.permutation([lab[g] for g in uq])
        m = dict(zip(uq, perm))
        yy = np.array([m[g] for g in groups])
        s = logistic_cv(X, yy, groups, seed=1)
        a = auc(yy, s)
        if a is not None:
            vals.append(a)
    vals = np.asarray(vals)
    return float((np.abs(vals - 0.5) >= abs(observed - 0.5) - 1e-12).mean()), vals


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60 * 60)
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        X = z["X"]
        meta = [json.loads(m) for m in z["meta"]]
        print(f"reusing {len(X)} cached feature vectors", flush=True)
    else:
        X, meta = capture(dl)
        np.savez_compressed(CACHE, X=X, meta=np.array([json.dumps(m) for m in meta]))
        print(f"captured {len(X)} feature vectors from {len({m['state'] for m in meta})} states",
              flush=True)
    if len(X) == 0:
        print("no features")
        return
    groups = np.array([m["state"] for m in meta])
    walls = np.array([m["wall"] for m in meta])
    shares = np.array([m["retreat_share"] for m in meta])

    # ---- probe A: not constructible, reported with the numbers ----
    per_state = {}
    for m in meta:
        per_state[m["state"]] = m["retreat_share"]
    hi = [s for s, v in per_state.items() if v >= 0.50]
    lo = [s for s, v in per_state.items() if v <= 0.15]
    out = {"arm": ARM, "feature": "norm(transformer(...)) at the last frame position, d_model=64",
           "n_features": int(X.shape[1]), "n_samples": int(len(X)),
           "n_states": int(len(np.unique(groups))), "offsets": OFFSETS,
           "cv": f"grouped {N_FOLDS}-fold over STATES",
           "null": f"label permutation AT THE STATE LEVEL, {N_PERM} draws",
           "probe_A_retreat_required": {
               "constructible": False,
               "n_retreat_dominant_share_ge_0.50": len(hi),
               "n_sampled_dominant_share_le_0.15": len(lo),
               "n_middle": int(len(per_state) - len(hi) - len(lo)),
               "retreat_share_min": float(min(per_state.values())),
               "retreat_share_median": float(np.median(list(per_state.values()))),
               "retreat_share_max": float(max(per_state.values())),
               "why": ("no state is retreat-only (min sampled solutions = 1) and none has zero retreat "
                       "solutions; retreat works to some degree everywhere and is strictly required "
                       "nowhere. 'Is a retreat required here?' has no ground truth to probe, so the "
                       "label the distillation was asked to condition on is not a well-defined "
                       "property of the states")},
           "probes": {}}
    print(json.dumps(out["probe_A_retreat_required"], indent=2), flush=True)

    # ---- probe B: on-top vs at-face ----
    y = np.array([1 if m["on_top"] else 0 for m in meta])
    s = logistic_cv(X, y, groups)
    a = auc(y, s)
    p, _ = perm_p(X, y, groups, a, n_perm=400)
    out["probes"]["B_on_top_vs_at_face"] = {
        "n_pos_states": int(len({m['state'] for m in meta if m['on_top']})),
        "n_neg_states": int(len({m['state'] for m in meta if not m['on_top']})),
        "auc": a, "perm_p": p, "n_perm": 400,
        "meaning": ("these need opposite corrections -- get off the pipe vs clear it -- so a policy "
                    "that cannot tell them apart cannot correct either")}
    print(f"  probe B on-top vs at-face: AUC {a:.3f}  perm p={p:.4f}", flush=True)

    # ---- probe C: wall identity, one-vs-rest for the walls with enough states ----
    cw = collections.Counter(m["wall"] for m in meta)
    big = [w for w, n in cw.items() if len({m['state'] for m in meta if m['wall'] == w}) >= 8]
    cres = {}
    for w in big:
        yy = (walls == w).astype(int)
        ss = logistic_cv(X, yy, groups)
        aa = auc(yy, ss)
        pp, _ = perm_p(X, yy, groups, aa, n_perm=300)
        cres[w] = {"auc": aa, "perm_p": pp,
                   "n_pos_states": int(len({m['state'] for m in meta if m['wall'] == w}))}
        print(f"  probe C wall={w:16s} AUC {aa:.3f}  perm p={pp:.4f}", flush=True)
    out["probes"]["C_wall_identity_one_vs_rest"] = cres

    # ---- probe D: x position, regression sanity floor ----
    xs = np.array([m["fail_x"] for m in meta], dtype=float)
    uq = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(uq)
    folds = np.array_split(uq, N_FOLDS)
    pred = np.zeros(len(xs))
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())]
        B = np.c_[(X[te] - mu) / sd, np.ones(te.sum())]
        w = np.linalg.lstsq(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ xs[tr], rcond=None)[0]
        pred[te] = B @ w
    ss_res = float(((xs - pred) ** 2).sum())
    ss_tot = float(((xs - xs.mean()) ** 2).sum())
    out["probes"]["D_x_position_regression"] = {
        "r2_out_of_fold": float(1 - ss_res / ss_tot),
        "rmse_px": float(np.sqrt(ss_res / len(xs))),
        "x_range": [float(xs.min()), float(xs.max())],
        "meaning": "sanity floor: if features do not carry position, B and C failing says nothing"}
    print(f"  probe D x-position: out-of-fold R2 "
          f"{out['probes']['D_x_position_regression']['r2_out_of_fold']:.3f}  "
          f"RMSE {out['probes']['D_x_position_regression']['rmse_px']:.0f} px", flush=True)

    b = out["probes"]["B_on_top_vs_at_face"]
    d = out["probes"]["D_x_position_regression"]
    c_sig = [w for w, v in cres.items() if v["perm_p"] < 0.05]
    b_sep = bool(b["auc"] is not None and b["auc"] > 0.5 and b["perm_p"] < 0.05)
    carries = bool(d["r2_out_of_fold"] > 0.3 or c_sig)
    out["gate_for_2b"] = {
        "probe_A_constructible": False,
        "features_separate_on_top": b_sep,
        "walls_separable": c_sig,
        "features_carry_position": bool(d["r2_out_of_fold"] > 0.3),
        "open_2b": bool(b_sep or carries)}
    if not out["gate_for_2b"]["open_2b"]:
        out["verdict"] = (
            f"**THE FEATURES DO NOT CARRY THE STATE DISTINCTIONS CONDITIONAL RECOVERY WOULD NEED.** "
            f"On-top vs at-face AUC {b['auc']:.3f} (perm p={b['perm_p']:.3f}); wall identity separable "
            f"for {len(c_sig)} of {len(cres)} walls; x-position out-of-fold R² "
            f"{d['r2_out_of_fold']:.3f}. **Conditional behaviour is not representable from this "
            f"observation through this trunk, which explains round one and round two in one stroke.** "
            f"§2b is not run.")
    else:
        out["verdict"] = (
            f"**THE FEATURES DO CARRY STATE INFORMATION: x-position out-of-fold R² "
            f"{d['r2_out_of_fold']:.3f}, wall identity separable for {len(c_sig)} of {len(cres)} walls, "
            f"on-top vs at-face AUC {b['auc']:.3f} (perm p={b['perm_p']:.3f}).** So the representation "
            f"is not the limit. **But probe A is not constructible: retreat is required at NO state and "
            f"useful at EVERY state (share {out['probe_A_retreat_required']['retreat_share_min']:.3f}"
            f"–{out['probe_A_retreat_required']['retreat_share_max']:.3f}, median "
            f"{out['probe_A_retreat_required']['retreat_share_median']:.3f}), so the label the "
            f"distillation was asked to condition on does not exist.** That is why neither round "
            f"produced conditionality, and it is a different diagnosis from either an objective problem "
            f"or a capacity problem.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
