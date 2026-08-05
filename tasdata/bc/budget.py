"""Wall-clock budgets for unattended runs. **A single hung FCEUX must not eat the remaining hours.**

The one-instance `flock` on `~/.tasdata_fceux.lock` means a process that dies while holding the lock blocks
everything queued behind it. In an attended block that is a nuisance; in a five-hour unattended block it is
the difference between four results and none.

Two things here:

* `Deadline` — a shared clock for a whole run. `remaining()` lets a section decide for itself whether to
  start work it cannot finish, which is better than starting it and being killed halfway.
* `time_limit` — a per-arm cap. On expiry it raises `TimedOut`, so the caller can log, release the lock in
  its own `finally`, skip that arm and continue.

**`time_limit` uses SIGALRM, so it only fires in the main thread and only interrupts Python-level
execution.** A C-level block (a `read()` on a FIFO waiting for an emulator that will never answer) is
exactly the case that matters, and SIGALRM does interrupt a blocking syscall -- the read fails with EINTR and
the exception propagates. It will not interrupt a non-Python spin loop, which this codebase does not contain.
"""
from __future__ import annotations

import signal
import time
from contextlib import contextmanager


class TimedOut(RuntimeError):
    """Raised when a per-arm wall-clock cap expires."""


class Deadline:
    """A shared clock for one unattended run."""

    def __init__(self, total_seconds: float) -> None:
        self.t0 = time.time()
        self.total = float(total_seconds)

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def can_afford(self, seconds: float) -> bool:
        """Should a section that needs `seconds` be started at all?"""
        return self.remaining() > seconds

    def stamp(self) -> str:
        e = self.elapsed()
        return f"[t+{int(e // 3600)}h{int(e % 3600 // 60):02d}m, {self.remaining() / 60:.0f}m left]"


@contextmanager
def time_limit(seconds: float, label: str = ""):
    """Cap a block of work. Raises `TimedOut` on expiry; always restores the previous handler."""
    if seconds is None or seconds <= 0:
        yield
        return

    def _fire(signum, frame):
        raise TimedOut(f"{label or 'block'} exceeded {seconds:.0f}s")

    prev = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)
