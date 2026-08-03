"""Backend selection.

Two replay backends share one interface (constructor keywords and
:meth:`replay`), so the parser and verifier never need to know which produced a
run:

``fceux``
    Lets FCEUX replay the ``.fm2`` itself and captures the result. The movie was
    recorded in FCEUX, so this is the only backend that is in sync by
    construction. Requires the ``fceux`` binary and a window.

``nes-py``
    Feeds inputs to nes-py frame by frame. Pure pip install, no window, but not
    accurate enough to survive an SMB level transition. Retained as a regression
    check.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .replay import NesReplayer

#: Backend names accepted by :func:`get_replayer`.
BACKENDS: tuple[str, ...] = ("fceux", "nes-py")

#: The backend used when none is requested.
DEFAULT_BACKEND = "fceux"


@runtime_checkable
class Replayer(Protocol):
    """The interface both backends implement."""

    backend: str
    observation_shape: tuple[int, int]
    frame_skip: int

    def replay(self, movie, **kwargs): ...


def get_replayer(
    backend: str = DEFAULT_BACKEND,
    rom_path: str | None = None,
    *,
    extra_args: Sequence[str] = (),
    **kwargs,
):
    """Construct a replayer by backend name.

    ``extra_args`` is FCEUX-only and ignored by the nes-py backend, so callers can
    pass one keyword set to either.
    """
    name = backend.strip().lower().replace("_", "-")
    if name in ("nes-py", "nespy", "nes"):
        return NesReplayer(rom_path, **kwargs)
    if name == "fceux":
        from .fceux_backend import FceuxReplayer

        return FceuxReplayer(rom_path, extra_args=extra_args, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
