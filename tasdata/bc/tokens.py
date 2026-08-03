"""Action vocabulary for behavioural cloning.

The raw label is a controller byte, so there are 256 possible values but only 67
ever occur and 43 of those occur fewer than 100 times in 1.2M frames. Training a
67-way head on classes with single-digit support teaches noise, so the rare tail is
folded into one ``RARE`` token. That leaves 25 classes.

The vocabulary is derived from data and then *persisted*: training and live play
must agree on token ids, and a vocabulary silently rebuilt from a different run set
would relabel everything.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..buttons import NES_BUTTON_BITS, describe_action

#: Occurrence count below which an action byte is folded into ``RARE``.
DEFAULT_RARE_THRESHOLD = 100

RARE_NAME = "RARE"

#: Buttons masked out during live play. Start pauses the game and Select does
#: nothing in-game; both appear in under 500 of 1.2M training frames, so a policy
#: that emits them is only ever hurting itself. Masking is applied at the emulator
#: boundary, not in the label space, so the model is still scored on what it chose.
LIVE_MASK = ~(NES_BUTTON_BITS["Start"] | NES_BUTTON_BITS["Select"]) & 0xFF


@dataclass
class ActionVocab:
    """Bidirectional map between controller bytes and token ids."""

    #: Token id -> the action byte it represents (for RARE, its most common member).
    token_to_byte: list[int]
    #: Token id -> every action byte folded into it.
    token_members: list[list[int]]
    #: Token id -> human-readable name.
    names: list[str]
    #: Token id -> training-set occurrence count.
    counts: list[int]
    rare_token: int
    threshold: int
    #: 256-entry lookup: action byte -> token id.
    byte_to_token: np.ndarray = field(repr=False, default=None)

    @property
    def size(self) -> int:
        return len(self.token_to_byte)

    @property
    def n_rare_members(self) -> int:
        return len(self.token_members[self.rare_token])

    def encode(self, actions: np.ndarray) -> np.ndarray:
        """Map an array of action bytes to token ids."""
        return self.byte_to_token[actions.astype(np.uint8)]

    def decode_byte(self, token: int, *, mask_live: bool = True) -> int:
        """Token id -> a controller byte to actually press."""
        byte = self.token_to_byte[int(token)]
        return byte & LIVE_MASK if mask_live else byte

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "rare_token": self.rare_token,
            "size": self.size,
            "tokens": [
                {
                    "id": i,
                    "name": self.names[i],
                    "byte": self.token_to_byte[i],
                    "count": self.counts[i],
                    "members": self.token_members[i],
                }
                for i in range(self.size)
            ],
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> ActionVocab:
        data = json.loads(Path(path).read_text())
        tokens = sorted(data["tokens"], key=lambda t: t["id"])
        vocab = cls(
            token_to_byte=[t["byte"] for t in tokens],
            token_members=[t["members"] for t in tokens],
            names=[t["name"] for t in tokens],
            counts=[t["count"] for t in tokens],
            rare_token=data["rare_token"],
            threshold=data["threshold"],
        )
        vocab._build_lookup()
        return vocab

    def _build_lookup(self) -> None:
        table = np.full(256, self.rare_token, dtype=np.int64)
        for token, members in enumerate(self.token_members):
            for byte in members:
                table[byte] = token
        self.byte_to_token = table

    def summary(self) -> str:
        lines = [
            f"action vocabulary: {self.size} tokens "
            f"(threshold {self.threshold}, {self.n_rare_members} bytes folded into RARE)"
        ]
        total = sum(self.counts) or 1
        for i in range(self.size):
            lines.append(
                f"  {i:2d} {self.names[i]:26s} byte={self.token_to_byte[i]:#04x} "
                f"n={self.counts[i]:9,d} {self.counts[i] * 100 / total:6.3f}%"
            )
        return "\n".join(lines)


def build_vocab(
    action_arrays: list[np.ndarray], *, threshold: int = DEFAULT_RARE_THRESHOLD
) -> ActionVocab:
    """Build a vocabulary from training action bytes, folding the rare tail."""
    counts: Counter = Counter()
    for arr in action_arrays:
        values, freq = np.unique(arr.astype(np.uint8), return_counts=True)
        for v, c in zip(values.tolist(), freq.tolist()):
            counts[int(v)] += int(c)

    frequent = sorted(
        (b for b, c in counts.items() if c >= threshold),
        key=lambda b: (-counts[b], b),
    )
    rare = sorted((b for b, c in counts.items() if c < threshold), key=lambda b: -counts[b])

    token_to_byte = list(frequent)
    token_members = [[b] for b in frequent]
    names = [describe_action(b) for b in frequent]
    token_counts = [counts[b] for b in frequent]

    # RARE always exists, even if empty, so token ids are stable across run sets.
    rare_token = len(token_to_byte)
    # Its representative press is the most common member, so emitting RARE live does
    # something plausible rather than something arbitrary.
    token_to_byte.append(rare[0] if rare else 0)
    token_members.append(rare)
    names.append(
        f"{RARE_NAME}({len(rare)} combos"
        + (f", plays {describe_action(rare[0])}" if rare else "")
        + ")"
    )
    token_counts.append(sum(counts[b] for b in rare))

    vocab = ActionVocab(
        token_to_byte=token_to_byte,
        token_members=token_members,
        names=names,
        counts=token_counts,
        rare_token=rare_token,
        threshold=threshold,
    )
    vocab._build_lookup()
    return vocab
