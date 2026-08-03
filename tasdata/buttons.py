"""Button naming and the bk2 -> nes-py action-byte mapping.

BizHawk names NES buttons ``P1 Up``, ``P1 A``, ... and writes one mnemonic
character per button in the Input Log.  nes-py wants a single byte per frame
whose bits are laid out right/left/down/up/start/select/B/A from the MSB down.
This module owns that translation and nothing else.
"""

from __future__ import annotations

import re

import numpy as np

#: nes-py's controller byte layout (see nes_py.wrappers.JoypadSpace._button_map).
NES_BUTTON_BITS: dict[str, int] = {
    "Right": 0b1000_0000,
    "Left": 0b0100_0000,
    "Down": 0b0010_0000,
    "Up": 0b0001_0000,
    "Start": 0b0000_1000,
    "Select": 0b0000_0100,
    "B": 0b0000_0010,
    "A": 0b0000_0001,
}

#: Canonical per-player button order used for the parsed boolean array columns.
NES_BUTTON_ORDER: tuple[str, ...] = (
    "Up", "Down", "Left", "Right", "Start", "Select", "B", "A",
)

#: Bit masks in the canonical output order used by the Bernoulli head. Same order as
#: :data:`NES_BUTTON_ORDER` so index j of an 8-vector always means the same button.
NES_BUTTON_ORDER_BITS: tuple[int, ...] = tuple(
    NES_BUTTON_BITS[name] for name in NES_BUTTON_ORDER
)

#: Console-level (non-controller) lines BizHawk records alongside the pads.
CONSOLE_BUTTONS: frozenset[str] = frozenset({"Power", "Reset", "FDS Insert", "FDS Eject"})

_PLAYER_RE = re.compile(r"^P(\d+)\s+(.*)$")


def split_player(name: str) -> tuple[int | None, str]:
    """Split ``"P1 Right"`` into ``(1, "Right")``; console lines get player ``None``.

    >>> split_player("P2 Start")
    (2, 'Start')
    >>> split_player("Reset")
    (None, 'Reset')
    """
    m = _PLAYER_RE.match(name.strip())
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, name.strip()


def is_pressed(char: str) -> bool:
    """Interpret one Input Log character.

    BizHawk writes the button's mnemonic when held (``U``, ``R``, ``A``, ...) and
    ``.`` when not.  Some writers emit a space instead of a dot, and TAStudio can
    emit ``o`` for an unset-but-tracked frame.  Everything else counts as held.
    """
    return char not in (".", " ", "\t", "\0")


def actions_from_states(
    states: np.ndarray,
    button_names: list[str],
    player: int = 1,
) -> np.ndarray:
    """Reduce a per-frame boolean button matrix to nes-py action bytes.

    Args:
        states: bool array ``(n_frames, n_buttons)`` as produced by the parser.
        button_names: column names, e.g. ``["Power", "Reset", "P1 Up", ...]``.
        player: which controller port to extract (1-based).

    Returns:
        ``uint8`` array of length ``n_frames``.

    Raises:
        ValueError: if the requested player has no columns in ``states``.
    """
    if states.ndim != 2:
        raise ValueError(f"expected 2-D state array, got shape {states.shape}")
    if states.shape[1] != len(button_names):
        raise ValueError(
            f"state array has {states.shape[1]} columns but {len(button_names)} names"
        )

    columns: list[tuple[int, int]] = []  # (column index, bit mask)
    for idx, name in enumerate(button_names):
        who, button = split_player(name)
        if who != player:
            continue
        bit = NES_BUTTON_BITS.get(button)
        if bit is not None:
            columns.append((idx, bit))

    if not columns:
        raise ValueError(
            f"no P{player} buttons found in log key columns {button_names!r}"
        )

    actions = np.zeros(states.shape[0], dtype=np.uint8)
    for idx, bit in columns:
        actions[states[:, idx]] |= bit
    return actions


def console_button_frames(
    states: np.ndarray, button_names: list[str], button: str
) -> np.ndarray:
    """Frame indices on which a console-level button (``Reset``/``Power``) is held."""
    for idx, name in enumerate(button_names):
        who, btn = split_player(name)
        if who is None and btn.lower() == button.lower():
            return np.flatnonzero(states[:, idx])
    return np.empty(0, dtype=np.int64)


def describe_action(action: int) -> str:
    """Render an action byte as ``"R+A"`` for logs and error messages."""
    held = [name for name, bit in NES_BUTTON_BITS.items() if action & bit]
    return "+".join(held) if held else "-"
