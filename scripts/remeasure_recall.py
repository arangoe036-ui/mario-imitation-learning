"""Re-measure every A-onset recall number under one calibration method.

The frozen table mixed two: early numbers calibrated on the first N rows of the first
training run (a biased slice) and, in the Stage 3 path, through a double-normalised input
that collapsed the model's output to a constant. Neither is comparable with the other.

Everything here uses the same procedure:

* thresholds calibrated on a **random** subset of TRAIN against the expert's own per-button
  press rates;
* onset recall measured on a **contiguous** slice of VAL, because an onset is defined by
  comparison with the previous frame;
* observations passed through exactly as the dataset yields them (already float32 in [0,1]).

The categorical head has no per-button output, so its button probabilities are obtained by
marginalising the softmax: P(button) is the total probability of the tokens whose action
byte sets that bit. That is the only way to put a 25-way head and an 8-Bernoulli head on
one axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.bernoulli import calibrate_thresholds  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    A_INDEX,
    calibrate,
    contiguous_rows,
    load_policy,
    onset_metrics,
    random_rows,
)
from tasdata.bc.train import make_loader  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/recall_remeasured.json"

CHECKPOINTS = [
    ("stage2 categorical (small_lr3e-4)", ROOT / "data/bc/small_lr3e-4_step3000.pt",
     "categorical"),
    ("stage2 categorical (tiny_lr1e-3)", ROOT / "data/bc/tiny_lr1e-3_step3000.pt",
     "categorical"),
    ("stage2 blind (control)", ROOT / "data/bc/blind_lr3e-4_step3000.pt", "categorical"),
    ("stage2 bernoulli only (arm A)",
     ROOT / "data/bc3/A_bernoulli_only_step3000_recal.pt", "bernoulli"),
    ("stage2 bernoulli + reweight (arm B)",
     ROOT / "data/bc3/B_bernoulli_onset10x_step3000_recal.pt", "bernoulli"),
]


def button_matrix(vocab) -> np.ndarray:
    """(n_tokens, 8) indicator of which buttons each token's action byte presses."""
    m = np.zeros((len(vocab.token_to_byte), 8), dtype=np.float32)
    for t, byte in enumerate(vocab.token_to_byte):
        for j, name in enumerate(NES_BUTTON_ORDER):
            if int(byte) & NES_BUTTON_BITS[name]:
                m[t, j] = 1.0
    return m


def categorical_probs(policy, dataset, rows, mat, batch: int = 256):
    """Marginalise a softmax over tokens into per-button probabilities."""
    loader = make_loader(Subset(dataset, rows), batch_size=batch, shuffle=False,
                         num_workers=0)
    probs, labels = [], []
    policy.eval()
    with torch.no_grad():
        for obs, _prev, bits, _onset in loader:
            p = torch.softmax(policy(obs), dim=-1).numpy()
            probs.append(p @ mat)
            labels.append(bits.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def categorical_metrics(policy, ctx, mat, target_rates) -> dict:
    cal_rows = random_rows(ctx.dataset(ctx.expert_train), 20_000, seed=0)
    train_set = ctx.dataset(ctx.expert_train)
    cal_p, _ = categorical_probs(policy, train_set, cal_rows, mat)
    calibration = calibrate_thresholds(cal_p, target_rates)
    thr = calibration.vector.astype(np.float64)

    rows = contiguous_rows(ctx.val_set, 30_000, 0)
    p, y = categorical_probs(policy, ctx.val_set, rows, mat)
    prev = np.zeros_like(y)
    prev[1:] = y[:-1]
    onset = (y > 0) & (prev == 0)
    pred = p > thr[None, :]
    recall = {}
    for j, name in enumerate(NES_BUTTON_ORDER):
        m = onset[:, j]
        recall[name] = float(pred[m, j].mean()) if m.any() else 0.0
    return {
        "onset_recall": recall,
        "onset_counts": {n: int(onset[:, j].sum()) for j, n in enumerate(NES_BUTTON_ORDER)},
        "realized_rate": {n: float(pred[:, j].mean()) for j, n in enumerate(NES_BUTTON_ORDER)},
        "exact_match": float((pred == (y > 0)).all(axis=1).mean()),
        "thresholds": dict(calibration.thresholds),
        "prob_at_onset_A": {"median": float(np.median(p[onset[:, A_INDEX], A_INDEX]))
                            if onset[:, A_INDEX].any() else 0.0},
    }


def main() -> None:
    ctx = O.Ctx()
    mat = button_matrix(ctx.vocab)
    out = {"method": ("thresholds calibrated on a random TRAIN subset against expert press "
                      "rates; onset recall on a contiguous VAL slice; no double "
                      "normalisation; categorical heads marginalised to per-button "
                      "probabilities"),
           "expert_A_rate": ctx.target_rates["A"], "checkpoints": []}

    for label, path, head in CHECKPOINTS:
        if not path.exists():
            print(f"  {label}: MISSING {path.name}")
            out["checkpoints"].append({"label": label, "missing": str(path)})
            continue
        try:
            policy, cfg, _ = load_policy(path)
            if head == "categorical":
                m = categorical_metrics(policy, ctx, mat, ctx.target_rates)
            else:
                calibration, _ = calibrate(policy, ctx.dataset(ctx.expert_train),
                                           ctx.target_rates)
                m = onset_metrics(policy, ctx.val_set,
                                  calibration.vector.astype(np.float64))
                m["thresholds"] = dict(calibration.thresholds)
            row = {"label": label, "head": head, "checkpoint": path.name,
                   "A_onset_recall": m["onset_recall"]["A"],
                   "A_onsets": m["onset_counts"]["A"],
                   "A_threshold": m["thresholds"]["A"],
                   "A_realized_rate": m["realized_rate"]["A"],
                   "exact_match": m["exact_match"],
                   "onset_recall_all": m["onset_recall"]}
            out["checkpoints"].append(row)
            print(f"  {label:38s} A-onset recall {row['A_onset_recall'] * 100:5.1f}%  "
                  f"thr {row['A_threshold']:.2f}  realized {row['A_realized_rate']:.3f}  "
                  f"exact {row['exact_match'] * 100:.1f}%")
        except Exception as exc:
            out["checkpoints"].append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {label}: FAILED {exc}")

    # Arm A rounds already measured under the corrected method during the overnight run.
    rounds = []
    for line in (ROOT / "data/overnight.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "tier2_round":
            rounds.append({"label": r.get("tag"), "head": "bernoulli",
                           "A_onset_recall": r["offline"]["onset_recall"]["A"],
                           "A_threshold": r["thresholds"]["A"],
                           "pipe1_rate": r["live"].get("1-1", {}).get("pipe1_rate"),
                           "pipe1_n": r["live"].get("1-1", {}).get("n")})
    out["arm_a_rounds"] = rounds
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
