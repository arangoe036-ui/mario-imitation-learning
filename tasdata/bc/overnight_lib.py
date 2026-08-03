"""Shared helpers for the overnight run: calibration, offline metrics, live eval, training.

Two sampling rules matter here and they are not interchangeable:

* **Calibration** needs an unbiased sample of probabilities, and order is irrelevant --
  it only matches a press *rate*. A random subset is correct; the first N rows of the
  first run is not, and that bias is a live suspect in the 0.0% recall result.
* **Onset metrics** need temporally adjacent rows, because an onset is defined by
  comparing a frame to the one before it. Those must come from a contiguous slice.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from ..buttons import NES_BUTTON_ORDER
from .bernoulli import bce_with_onset_weights, calibrate_thresholds
from .model import BCPolicy, PolicyConfig
from .train import make_loader

A_INDEX = NES_BUTTON_ORDER.index("A")


# ---------------------------------------------------------------- intervals


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def boot_ci(values, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    meds = np.median(rng.choice(arr, size=(reps, arr.size), replace=True), axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def diff_ci(ka: int, na: int, kb: int, nb: int, z: float = 1.96) -> tuple[float, float]:
    """Newcombe interval for (B - A)."""
    la, ua = wilson(ka, na, z)
    lb, ub = wilson(kb, nb, z)
    pa, pb = ka / na, kb / nb
    lo = (pb - pa) - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2)
    hi = (pb - pa) + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)
    return (lo, hi)


# ---------------------------------------------------------------- policies


def load_policy(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()
    return policy, cfg, blob.get("thresholds")


def fresh_policy(cfg: PolicyConfig, seed: int = 0) -> BCPolicy:
    torch.manual_seed(seed)
    return BCPolicy(cfg)


def save_policy(path: Path, policy, cfg, thresholds: dict, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": policy.state_dict(), "policy_config": cfg,
                "thresholds": thresholds, **extra}, path)
    return path


# ---------------------------------------------------------------- inference


def _forward(policy, dataset, rows, batch: int = 256):
    """Probabilities and label bits for a fixed set of dataset rows, in order.

    ``FrameStackDataset`` already returns float32 in [0, 1]. Dividing by 255 again -- as an
    earlier version of this code and of ``stage3_train`` both did -- feeds the network a
    near-black image, and it responds with a constant: p(A) sat at 0.00710 with a spread of
    1e-5 across every frame, which is what produced the 0.0% A-onset recall.
    """
    loader = make_loader(Subset(dataset, rows), batch_size=batch, shuffle=False,
                         num_workers=0)
    probs, labels = [], []
    policy.eval()
    with torch.no_grad():
        for obs, _prev, bits, _onset in loader:
            probs.append(torch.sigmoid(policy(obs)).cpu().numpy())
            labels.append(bits.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def random_rows(dataset, n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    n = min(n, len(dataset))
    return sorted(rng.choice(len(dataset), size=n, replace=False).tolist())


def contiguous_rows(dataset, n: int, start: int = 0) -> list[int]:
    n = min(n, len(dataset) - start)
    return list(range(start, start + n))


def expert_target_rates(runs) -> dict[str, float]:
    """Per-button press rate over the expert corpus -- the rate calibration matches."""
    from ..buttons import NES_BUTTON_BITS

    total = 0
    counts = {n: 0 for n in NES_BUTTON_ORDER}
    for run in runs:
        a = np.asarray(run.actions, dtype=np.uint8)
        total += a.size
        for name in NES_BUTTON_ORDER:
            counts[name] += int(np.count_nonzero(a & NES_BUTTON_BITS[name]))
    return {n: (counts[n] / total if total else 0.0) for n in NES_BUTTON_ORDER}


def calibrate(policy, train_set, target_rates: dict[str, float], *, n_rows: int = 20_000,
              seed: int = 0):
    """Match each button's realized press rate to the expert's, on a random TRAIN slice."""
    rows = random_rows(train_set, n_rows, seed=seed)
    probs, _ = _forward(policy, train_set, rows)
    return calibrate_thresholds(probs, target_rates), probs


def onset_metrics(policy, val_set, thresholds_vec: np.ndarray, *, n_rows: int = 30_000,
                  start: int = 0) -> dict:
    """Per-button onset recall at the given thresholds, on a contiguous VAL slice.

    Contiguity is required: an onset is `held now and not held on the previous frame`,
    which is only meaningful if the previous row really is the previous frame.
    """
    rows = contiguous_rows(val_set, n_rows, start)
    probs, labels = _forward(policy, val_set, rows)
    prev = np.zeros_like(labels)
    prev[1:] = labels[:-1]
    onset = (labels > 0) & (prev == 0)
    pred = probs > thresholds_vec[None, :]

    recall, counts, realized, sep = {}, {}, {}, {}
    for j, name in enumerate(NES_BUTTON_ORDER):
        m = onset[:, j]
        counts[name] = int(m.sum())
        recall[name] = float(pred[m, j].mean()) if m.any() else 0.0
        realized[name] = float(pred[:, j].mean())
        off = labels[:, j] == 0
        sep[name] = (
            float(probs[m, j].mean() / max(probs[off, j].mean(), 1e-9)) if m.any() else 0.0
        )
    exact = float((pred == (labels > 0)).all(axis=1).mean())
    return {
        "onset_recall": recall, "onset_counts": counts, "realized_rate": realized,
        "separation": sep, "exact_match": exact, "n_rows": len(rows),
        "prob_at_onset_A": _describe(probs[onset[:, A_INDEX], A_INDEX]),
        "prob_elsewhere_A": _describe(probs[labels[:, A_INDEX] == 0, A_INDEX]),
    }


def _describe(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "mean": float(arr.mean()), "p10": float(np.percentile(arr, 10)),
            "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max())}


# ---------------------------------------------------------------- training


def train_policy(policy, dataset, *, steps: int, lr: float = 1e-4, onset_weight: float = 10.0,
                 batch_size: int = 128, seed: int = 0, weight_decay: float = 1e-4,
                 grad_clip: float = 1.0, log=print, log_every: int = 200):
    """Train (or fine-tune) a Bernoulli-head policy on CPU.

    CPU deliberately: probing MPS availability poisons every FCEUX child process launched
    afterwards in the same session into broken software OpenGL, and this run interleaves
    training with emulation.
    """
    policy = policy.to(torch.device("cpu"))
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                         seed=seed)
    step, running = 0, 0.0
    while step < steps:
        for obs, _prev, bits, onset in loader:
            loss = bce_with_onset_weights(policy(obs), bits.float(),
                                          onset.float(), onset_weight=onset_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()
            running += float(loss.detach())
            step += 1
            if step % log_every == 0:
                log(f"      step {step}/{steps} loss {running / log_every:.4f}")
                running = 0.0
            if step >= steps:
                break
    policy.eval()
    return policy


# ---------------------------------------------------------------- live eval


def eval_live(session, policy, thresholds_vec, starts, vocab, cfg, *, seeds: int = 200,
              expert_bytes=None, max_frames: int = 2500) -> dict:
    """Per-button-sampling live evaluation with intervals, per start point."""
    from .session_player import play_episode

    out = {}
    for start in starts:
        eps = []
        for s in range(seeds):
            try:
                eps.append(play_episode(session, policy, start, vocab, seed=s,
                                        selection="sample", thresholds=thresholds_vec,
                                        head_type=cfg.head_type, stack=cfg.stack,
                                        expert_bytes=expert_bytes, max_frames=max_frames))
            except Exception as exc:  # one bad episode must not lose the rest
                out.setdefault("errors", []).append(f"{start.label} seed {s}: {exc}"[:160])
        if not eps:
            continue
        xs = [e.furthest_x for e in eps]
        k1 = sum(e.cleared_pipe1 for e in eps)
        out[start.label] = {
            "n": len(eps),
            "pipe1_k": k1, "pipe1_rate": k1 / len(eps), "pipe1_ci": wilson(k1, len(eps)),
            "x_median": float(np.median(xs)), "x_median_ci": boot_ci(xs),
            "x_mean": float(np.mean(xs)), "x_p90": float(np.percentile(xs, 90)),
            "deaths_mean": float(np.mean([e.deaths for e in eps])),
            "a_presses_median": float(np.median([e.a_presses for e in eps])),
            "longest_a_hold_max": int(max(e.longest_a_hold for e in eps)),
            "furthest_level": max(e.furthest_level for e in eps),
            "hold_A_median": float(np.median(
                [e.hold_stats.get("A", {}).get("median", 0.0) for e in eps])),
            "hold_A_p90": float(np.mean(
                [e.hold_stats.get("A", {}).get("p90", 0.0) for e in eps])),
            "end_class": {c: sum(1 for e in eps if e.end_class == c)
                          for c in sorted({e.end_class for e in eps})},
            "end_class_pct": {c: round(100 * sum(1 for e in eps if e.end_class == c)
                                       / len(eps), 1)
                              for c in sorted({e.end_class for e in eps})},
        }
    return out
