"""Per-button Bernoulli head: loss, threshold calibration, and diagnostics.

Why per-button
--------------
Under a 25-way softmax the four A-containing tokens (8.23 + 3.27 + 2.29 + 0.44 = 14.2%)
each individually lose to ``Right+B`` at 30.63%, so argmax emitted A on 0.03% of frames
and A-onset recall was exactly 0.00% -- despite the model carrying real signal (18.76%
mass on A at onsets vs 10.21% at rest). Predicting each button independently removes the
vote-splitting: A is judged against its own ~15% base rate, not against Right+B.

Thresholds
----------
0.5 is the wrong default and would reproduce the old failure exactly: mass on A at
onsets is only ~19%, so a 0.5 threshold never fires. Instead each button's threshold is
chosen so the *realized press rate* matches the expert's. That is a calibration, not a
tuning knob, and it is reported alongside every result.

Onset weighting
---------------
Transitions are ~8% of training frames, so ~92% of the gradient goes to "keep doing what
you are doing". Arm B upweights the loss on the specific button that changes, at the
frame it changes, which is the only place "decide to act" is learnable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..buttons import NES_BUTTON_ORDER, NES_BUTTON_ORDER_BITS

#: Thresholds are searched on this grid. Deliberately reaches well below 0.5.
THRESHOLD_GRID = np.concatenate(
    [np.arange(0.02, 0.50, 0.01), np.arange(0.50, 0.96, 0.05)]
)


def bce_with_onset_weights(
    logits: torch.Tensor,
    targets: torch.Tensor,
    onsets: torch.Tensor | None = None,
    *,
    onset_weight: float = 1.0,
) -> torch.Tensor:
    """BCE over 8 buttons, optionally upweighting the button that just turned on.

    ``onsets`` is a 0/1 mask the same shape as ``targets``: 1 where that button goes
    from released to held on this frame. Only the *changing* button is upweighted, not
    the whole frame -- an A-onset should not also inflate the loss on Right.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if onsets is not None and onset_weight != 1.0:
        weights = 1.0 + (onset_weight - 1.0) * onsets
        loss = loss * weights
        return loss.sum() / weights.sum().clamp(min=1.0)
    return loss.mean()


def bits_to_byte(bits: np.ndarray) -> np.ndarray:
    """``(..., 8)`` 0/1 array in canonical order -> controller byte."""
    out = np.zeros(bits.shape[:-1], dtype=np.uint8)
    for j, bit in enumerate(NES_BUTTON_ORDER_BITS):
        out |= (bits[..., j] > 0).astype(np.uint8) * bit
    return out


def byte_to_bits(byte: int) -> np.ndarray:
    return np.array(
        [1.0 if byte & bit else 0.0 for bit in NES_BUTTON_ORDER_BITS], dtype=np.float32
    )


def expert_button_rates(actions_list: list[np.ndarray]) -> dict[str, float]:
    """Fraction of frames each button is held, across the given action arrays."""
    total = sum(a.size for a in actions_list) or 1
    rates: dict[str, float] = {}
    for name, bit in zip(NES_BUTTON_ORDER, NES_BUTTON_ORDER_BITS):
        held = sum(int(((a.astype(np.int64) & bit) > 0).sum()) for a in actions_list)
        rates[name] = held / total
    return rates


@dataclass
class ThresholdCalibration:
    """Chosen threshold per button and what it actually produces."""

    thresholds: dict[str, float]
    realized_rate: dict[str, float]
    target_rate: dict[str, float]

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.thresholds[n] for n in NES_BUTTON_ORDER], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "realized_rate": self.realized_rate,
            "target_rate": self.target_rate,
        }

    def text(self) -> str:
        lines = [f"{'button':8s} {'thresh':>7s} {'realized':>9s} {'expert':>8s} {'ratio':>7s}"]
        for n in NES_BUTTON_ORDER:
            t = self.target_rate.get(n, 0.0)
            r = self.realized_rate.get(n, 0.0)
            ratio = (r / t) if t > 1e-9 else float("nan")
            lines.append(
                f"{n:8s} {self.thresholds[n]:7.3f} {r * 100:8.2f}% {t * 100:7.2f}% "
                f"{ratio:6.2f}x"
            )
        return "\n".join(lines)


def calibrate_thresholds(
    probs: np.ndarray, target_rates: dict[str, float]
) -> ThresholdCalibration:
    """Pick, per button, the threshold whose realized press rate matches the expert.

    ``probs`` is ``(n_frames, 8)`` of sigmoid outputs. Buttons the expert essentially
    never presses get a threshold of 1.01 so they can never fire.
    """
    thresholds: dict[str, float] = {}
    realized: dict[str, float] = {}
    for j, name in enumerate(NES_BUTTON_ORDER):
        target = float(target_rates.get(name, 0.0))
        column = probs[:, j]
        if target <= 1e-5:
            thresholds[name] = 1.01
            realized[name] = 0.0
            continue
        rates = np.array([(column > t).mean() for t in THRESHOLD_GRID])
        best = int(np.argmin(np.abs(rates - target)))
        thresholds[name] = float(THRESHOLD_GRID[best])
        realized[name] = float(rates[best])
    return ThresholdCalibration(thresholds, realized, dict(target_rates))


@dataclass
class BernoulliReport:
    """Everything worth knowing about a Bernoulli head's validation behaviour."""

    calibration: ThresholdCalibration
    onset_recall: dict[str, float] = field(default_factory=dict)
    onset_counts: dict[str, int] = field(default_factory=dict)
    separation: dict[str, float] = field(default_factory=dict)
    exact_match: float = 0.0
    hamming: float = 0.0
    novel_combo_rate: float = 0.0
    novel_combos: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "calibration": self.calibration.to_dict(),
            "onset_recall": self.onset_recall,
            "onset_counts": self.onset_counts,
            "separation": self.separation,
            "exact_match": self.exact_match,
            "hamming": self.hamming,
            "novel_combo_rate": self.novel_combo_rate,
            "novel_combos": self.novel_combos,
        }


def evaluate_bernoulli(
    probs: np.ndarray,
    labels: np.ndarray,
    prev_bits: np.ndarray,
    target_rates: dict[str, float],
    *,
    expert_bytes: set[int] | None = None,
    calibration: ThresholdCalibration | None = None,
) -> BernoulliReport:
    """Calibrate thresholds and measure onset recall, separation and novel combos.

    ``separation`` is mean P(button) at that button's onsets divided by mean P at frames
    where it is not held -- the quantity that was 1.84x for A under the categorical head
    and which onset reweighting is supposed to increase.
    """
    calibration = calibration or calibrate_thresholds(probs, target_rates)
    thresh = calibration.vector
    pred = (probs > thresh[None, :]).astype(np.float32)

    onset_recall: dict[str, float] = {}
    onset_counts: dict[str, int] = {}
    separation: dict[str, float] = {}
    for j, name in enumerate(NES_BUTTON_ORDER):
        held = labels[:, j] > 0
        was = prev_bits[:, j] > 0
        onset = held & ~was
        onset_counts[name] = int(onset.sum())
        onset_recall[name] = float(pred[onset, j].mean()) if onset.any() else 0.0
        rest = ~held
        separation[name] = (
            float(probs[onset, j].mean() / max(probs[rest, j].mean(), 1e-9))
            if onset.any() and rest.any()
            else 0.0
        )

    pred_bytes = bits_to_byte(pred)
    novel_rate = 0.0
    novel_list: list[str] = []
    if expert_bytes is not None:
        novel_mask = ~np.isin(pred_bytes, np.array(sorted(expert_bytes), dtype=np.uint8))
        novel_rate = float(novel_mask.mean())
        from ..buttons import describe_action

        seen: dict[int, int] = {}
        for b in pred_bytes[novel_mask]:
            seen[int(b)] = seen.get(int(b), 0) + 1
        novel_list = [
            f"{describe_action(b)} x{c}"
            for b, c in sorted(seen.items(), key=lambda kv: -kv[1])[:10]
        ]

    return BernoulliReport(
        calibration=calibration,
        onset_recall=onset_recall,
        onset_counts=onset_counts,
        separation=separation,
        exact_match=float((pred == labels).all(axis=1).mean()),
        hamming=float((pred == labels).mean()),
        novel_combo_rate=novel_rate,
        novel_combos=novel_list,
    )
