"""Fine-tuning and evaluation helpers for the Stage 3 self-imitation rounds.

Deliberately a fine-tune rather than a retrain from scratch. Each round adds a few tens of
thousands of self-generated frames to ~981k expert frames, so restarting would spend most
of its budget relearning what the previous round already knew, and the round-to-round
comparison the experiment depends on would be dominated by initialisation noise instead of
by the new data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..buttons import NES_BUTTON_ORDER
from .bernoulli import bce_with_onset_weights, calibrate_thresholds, evaluate_bernoulli
from .train import make_loader

#: Cap on validation rows scored per round; the full split is far larger than needed
#: for a stable onset-recall estimate and every round pays for it.
VAL_ROWS = 30_000


def _probs_and_labels(policy, dataset, *, max_rows: int = VAL_ROWS, batch: int = 256):
    """Run the policy over a dataset and return (probs, label bits, previous bits)."""
    loader = make_loader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    probs, labels, prevs = [], [], []
    seen = 0
    policy.eval()
    with torch.no_grad():
        for obs, prev, bits, _onset in loader:
            logits = policy(obs)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(bits.cpu().numpy())
            prevs.append(prev.cpu().numpy() if prev is not None else None)
            seen += len(obs)
            if seen >= max_rows:
                break
    p = np.concatenate(probs)[:max_rows]
    y = np.concatenate(labels)[:max_rows]
    # Previous-frame bits, derived from the labels themselves: onset of button j at row i
    # means y[i, j] and not y[i-1, j]. The dataset's prev channel is unused here because
    # the prev-action input was dropped after the copycat result.
    prev_bits = np.zeros_like(y)
    prev_bits[1:] = y[:-1]
    return p, y, prev_bits


def calibrate_and_eval(policy, cfg, val_set, train_set, *, target_rates=None):
    """Calibrate thresholds on a slice of TRAIN, then score on VAL.

    Thresholds are never fitted on the validation split -- doing so would let the number
    used to report performance also be the number tuned to it.
    """
    cal_p, cal_y, _ = _probs_and_labels(policy, train_set, max_rows=20_000)
    if target_rates is None:
        target_rates = {
            name: float(cal_y[:, j].mean()) for j, name in enumerate(NES_BUTTON_ORDER)
        }
    calibration = calibrate_thresholds(cal_p, target_rates)

    val_p, val_y, val_prev = _probs_and_labels(policy, val_set)
    report = evaluate_bernoulli(
        val_p, val_y, val_prev, target_rates, calibration=calibration
    )
    d = report.to_dict() if hasattr(report, "to_dict") else dict(report.__dict__)
    thresholds = calibration.vector.astype(np.float64)
    d.setdefault("onset_recall", {})
    return thresholds, d


def finetune(policy, cfg, tcfg, train_set, val_set, rnd: int, *, out_dir: Path | None = None):
    """Fine-tune the current policy on expert + accepted self data, and checkpoint it."""
    out_dir = Path(out_dir or Path(__file__).resolve().parents[2] / "data/bc_stage3")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # never probe MPS: it poisons every later FCEUX child
    policy = policy.to(device)
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    loader = make_loader(
        train_set, batch_size=tcfg.batch_size, shuffle=True, num_workers=0, seed=rnd
    )

    step = 0
    running = 0.0
    while step < tcfg.steps:
        for obs, _prev, bits, onset in loader:
            logits = policy(obs)
            loss = bce_with_onset_weights(
                logits, bits.float(), onset.float(), onset_weight=tcfg.onset_weight
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), tcfg.grad_clip)
            opt.step()
            running += float(loss)
            step += 1
            if step % 200 == 0:
                print(f"    step {step}/{tcfg.steps} loss {running / 200:.4f}")
                running = 0.0
            if step >= tcfg.steps:
                break

    path = out_dir / f"stage3_round{rnd}.pt"
    torch.save(
        {
            "model_state": policy.state_dict(),
            "policy_config": cfg,
            "step": tcfg.steps,
            "round": rnd,
            "thresholds": {n: 0.5 for n in NES_BUTTON_ORDER},  # replaced by calibration
        },
        path,
    )
    return path
