"""One long-lived FCEUX process per evaluation run.

The problem this replaces
------------------------
Evaluation used to spawn a fresh FCEUX per episode. Hundreds of window creations
exhaust macOS's IOSurface client limit, at which point every launch aborts with
``exit=134``. That single cause produced all three symptoms: the emulator test
failures, the window flashing, and the slow evaluation (a process launch plus a movie
fast-forward per episode).

The fix
-------
Launch FCEUX once. Reset between episodes by loading a savestate, never by relaunching.
Close the process when the whole run finishes. A cross-process lock enforces a hard cap
of one FCEUX at a time so this cannot regress -- including across the separate eval
worker processes, which is where the parallel-emulator experiment went wrong.

Savestates
----------
FCEUX Lua can create anonymous in-memory savestates freely, and numbered slots 0-9 that
persist to ``~/.fceux/fcs``. It cannot write a savestate to an arbitrary path --
``savestate.object("/tmp/x.fcs")`` returns without error and creates nothing. So a
library of hundreds of states cannot live on disk as binaries.

Instead the library is built **once per process**: one fast-forward through an expert
movie captures every requested point into an in-memory savestate object, keyed by an
index that *is* persisted to disk (:mod:`tasdata.bc.statelib`). Rebuilding costs a single
movie replay (~20 s for 67k frames at max speed) amortised over every episode in the run,
instead of one replay per episode.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

try:  # POSIX-only. Absent on Windows, which cannot run FCEUX evaluation anyway --
    import fcntl  # but the module must still import so the suite can be collected.
except ModuleNotFoundError:  # pragma: no cover - platform-dependent
    fcntl = None

import numpy as np

from ..fceux_backend import (
    GD_HEADER,
    GD_HEIGHT,
    GD_LEN,
    GD_WIDTH,
    RAM_BYTES,
    FceuxError,
    _FifoReader,
    find_fceux,
)

#: Opcodes on the command channel. One byte plus a little-endian uint16 argument.
OP_STEP = 0
OP_RESET = 1
OP_QUIT = 2
#: Scratch savestates. The movie-frame library is built once at startup and covers only
#: positions the *expert* visits. Practising an obstacle needs states the *policy* visits, and
#: replaying a policy prefix costs its full length on every rollout. These two ops let a prefix be
#: replayed once, snapshotted, and then restored for free any number of times.
OP_SAVE_SCRATCH = 3
OP_LOAD_SCRATCH = 4

#: Cross-process guard. Only one FCEUX may exist at a time, ever.
LOCK_PATH = Path.home() / ".tasdata_fceux.lock"


class TooManyEmulators(FceuxError):
    """Another FCEUX session is already running in this or another process."""


class LockingUnavailable(FceuxError):
    """This platform has no `fcntl.flock`, so the one-emulator cap cannot be enforced."""


class EmulatorLock:
    """Exclusive, cross-process lock enforcing a hard cap of one FCEUX."""

    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        if fcntl is None:
            raise LockingUnavailable(
                "fcntl.flock is unavailable on this platform, so the one-FCEUX cap "
                "cannot be enforced. Running an emulator without it reintroduces the "
                "OpenGL race this lock exists to prevent. Evaluate on macOS or Linux."
            )
        self._fh = self.path.open("w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise TooManyEmulators(
                    "an FCEUX session is already running. Only one is permitted: "
                    "concurrent launches exhaust macOS IOSurface clients and every "
                    "subsequent launch aborts with exit=134."
                ) from exc
            raise
        self._fh.write(str(os.getpid()))
        self._fh.flush()

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


_LUA_SERVER = r"""
-- Persistent FCEUX server. Builds a savestate library once, then serves
-- reset/step commands forever until told to quit.
local OBS   = %(obs)s
local ACT   = %(act)s
local READY = %(ready)s
local FRAMES = %(frames)s          -- movie frames at which to capture a savestate

local function u32(n)
  return string.char(n %% 256, math.floor(n / 256) %% 256,
                     math.floor(n / 65536) %% 256, math.floor(n / 16777216) %% 256)
end

emu.speedmode("maximum")

local function press(byte)
  joypad.set(1, {
    right  = bit.band(byte, 0x80) ~= 0, left   = bit.band(byte, 0x40) ~= 0,
    down   = bit.band(byte, 0x20) ~= 0, up     = bit.band(byte, 0x10) ~= 0,
    start  = bit.band(byte, 0x08) ~= 0, select = bit.band(byte, 0x04) ~= 0,
    B      = bit.band(byte, 0x02) ~= 0, A      = bit.band(byte, 0x01) ~= 0,
  })
end

-- ---- build the savestate library in ONE pass through the movie ----------------
local want = {}
local order = {}
for f in string.gmatch(FRAMES, "(%%d+)") do
  local n = tonumber(f)
  want[n] = true
  order[#order + 1] = n
end
table.sort(order)

-- states are stored by ORDINAL (1..N), not by frame number: frame numbers reach
-- 67,117 which does not fit the uint16 command argument.
local states = {}
local scratch = {}      -- caller-managed in-memory snapshots, keyed by slot number
local captured = 0
local target = 1
while target <= #order do
  if emu.framecount() >= order[target] then
    local s = savestate.object()
    savestate.save(s)
    states[target] = s
    captured = captured + 1
    target = target + 1
  else
    emu.frameadvance()
  end
end
movie.stop()

-- Signal readiness with the number of states captured, then open the channels.
local rf = io.open(READY, "w")
rf:write(tostring(captured) .. "\n")
rf:close()

local obs = io.open(OBS, "wb")
local act = io.open(ACT, "rb")
if obs == nil or act == nil then os.exit(1) end

local function send()
  local ram = memory.readbyterange(0x0000, 0x800)
  local s = gui.gdscreenshot()
  obs:write("TF", u32(emu.framecount()), ram, u32(#s), s)
  obs:flush()
end

-- ---- serve ------------------------------------------------------------------
while true do
  local c = act:read(3)
  if c == nil then break end
  local op = string.byte(c, 1)
  local arg = string.byte(c, 2) + string.byte(c, 3) * 256
  if op == 2 then
    break
  elseif op == 3 then
    -- snapshot the live state into a scratch slot (no frameadvance: the caller is already
    -- standing where it wants to be, and advancing would move it)
    local s = savestate.object()
    savestate.save(s)
    scratch[arg] = s
    send()
  elseif op == 4 then
    local s = scratch[arg]
    if s ~= nil then savestate.load(s) end
    if movie.active() then movie.stop() end
    -- one null-input frame, exactly as op 1 does, so restored states are indistinguishable
    -- from reset() states to every caller
    press(0)
    emu.frameadvance()
    send()
  elseif op == 1 then
    local s = states[arg + 1]   -- arg is a 0-based ordinal
    if s ~= nil then savestate.load(s) end
    -- A savestate captured during movie playback restores the movie's *playback
    -- state* too, which would let the recorded inputs drive Mario instead of us.
    -- Stop it after every load, not just once at build time.
    if movie.active() then movie.stop() end
    press(0)
    emu.frameadvance()
    send()
  else
    press(arg)
    emu.frameadvance()
    send()
  end
end

obs:close()
os.exit(0)
"""


def _lua_string(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass
class Observation:
    """One frame returned by the session."""

    framecount: int
    ram: np.ndarray
    rgb: np.ndarray


class FceuxSession:
    """A persistent FCEUX with a savestate library, reset by state id.

    Use as a context manager. ``reset(frame)`` loads the savestate captured at that
    movie frame; ``step(byte)`` applies one input and returns the next observation.
    """

    def __init__(
        self,
        rom_path: Path | str,
        movie_path: Path | str,
        capture_frames: list[int],
        *,
        binary: str | Path = "fceux",
        opposite_directionals: bool = True,
        window_position: tuple[int, int] | None = None,
        startup_timeout: float = 600.0,
    ) -> None:
        self.rom_path = Path(rom_path)
        self.movie_path = Path(movie_path)
        self.capture_frames = sorted(set(int(f) for f in capture_frames))
        if not self.capture_frames:
            raise ValueError("need at least one capture frame")
        if len(self.capture_frames) > 65535:
            raise ValueError("capture frame ids must fit in a uint16 argument")
        self.binary = find_fceux(binary)
        self.opposite_directionals = opposite_directionals
        self.window_position = window_position
        self.startup_timeout = startup_timeout

        self._lock = EmulatorLock()
        self._proc: subprocess.Popen | None = None
        self._reader: _FifoReader | None = None
        self._act_fd: int | None = None
        self._workdir: Path | None = None
        self.n_states = 0
        self.build_seconds = 0.0
        self.steps_served = 0
        self.resets_served = 0
        self.scratch_saves = 0
        self.scratch_loads = 0

    # -- lifecycle ----------------------------------------------------------- #

    def __enter__(self) -> FceuxSession:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        self._lock.acquire()
        try:
            self._start_locked()
        except Exception:
            self._lock.release()
            raise

    def _start_locked(self) -> None:
        wd = Path(tempfile.mkdtemp(prefix="tasdata-session-"))
        self._workdir = wd
        obs, act, ready = wd / "obs.fifo", wd / "act.fifo", wd / "ready.txt"
        os.mkfifo(obs)
        os.mkfifo(act)
        lua = wd / "server.lua"
        lua.write_text(
            _LUA_SERVER
            % {
                "obs": _lua_string(str(obs)),
                "act": _lua_string(str(act)),
                "ready": _lua_string(str(ready)),
                "frames": _lua_string(" ".join(str(f) for f in self.capture_frames)),
            }
        )
        log = (wd / "fceux.log").open("wb")
        cmd = [
            str(self.binary),
            "--no-config", "1",
            "--sound", "0",
            "--opposite-directionals", "1" if self.opposite_directionals else "0",
            "--playmov", str(self.movie_path),
        ]
        if self.window_position is not None:
            cmd += [
                "--winposx", str(self.window_position[0]),
                "--winposy", str(self.window_position[1]),
            ]
        cmd += ["--loadlua", str(lua), str(self.rom_path)]

        started = time.perf_counter()
        self._proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        )
        # Wait for the library to be built before opening the channels: the fast-forward
        # takes ~20 s and we must not confuse it with a hang.
        deadline = started + self.startup_timeout
        while not ready.exists():
            if self._proc.poll() is not None:
                tail = (wd / "fceux.log").read_text(errors="replace")[-500:]
                raise FceuxError(
                    f"FCEUX exited during savestate build (code {self._proc.returncode})"
                    f"\n{tail}"
                )
            if time.perf_counter() > deadline:
                raise FceuxError(
                    f"savestate library not built within {self.startup_timeout:.0f}s"
                )
            time.sleep(0.2)
        self.n_states = int(ready.read_text().strip() or 0)
        self.build_seconds = time.perf_counter() - started

        self._reader = _FifoReader(obs, self._proc)
        self._act_fd = os.open(str(act), os.O_WRONLY)

    def close(self) -> None:
        try:
            if self._act_fd is not None:
                try:
                    os.write(self._act_fd, bytes([OP_QUIT, 0, 0]))
                except OSError:
                    pass
                try:
                    os.close(self._act_fd)
                except OSError:
                    pass
                self._act_fd = None
            if self._reader is not None:
                self._reader.close()
                self._reader = None
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
            self._proc = None
            if self._workdir is not None:
                shutil.rmtree(self._workdir, ignore_errors=True)
                self._workdir = None
        finally:
            self._lock.release()

    # -- protocol ------------------------------------------------------------ #

    def _command(self, op: int, arg: int = 0, timeout: float = 120.0) -> Observation:
        if self._act_fd is None or self._reader is None:
            raise FceuxError("session is not running")
        try:
            os.write(self._act_fd, bytes([op, arg & 0xFF, (arg >> 8) & 0xFF]))
        except (BrokenPipeError, OSError) as exc:
            raise FceuxError(f"emulator closed the command channel: {exc}") from exc
        header = self._reader.read_exact(6, timeout)
        if header is None or header[:2] != b"TF":
            raise FceuxError("emulator closed the observation stream")
        framecount = int.from_bytes(header[2:6], "little")
        ram_bytes = self._reader.read_exact(RAM_BYTES, timeout)
        if ram_bytes is None:
            raise FceuxError("stream ended inside RAM")
        screen_len = int.from_bytes(
            self._reader.read_exact(4, timeout) or b"\0\0\0\0", "little"
        )
        if screen_len != GD_LEN:
            raise FceuxError(f"unexpected screen payload {screen_len}")
        blob = self._reader.read_exact(screen_len, timeout)
        if blob is None:
            raise FceuxError("stream ended inside screen")
        px = np.frombuffer(blob[GD_HEADER:], dtype=np.uint8).reshape(
            GD_HEIGHT, GD_WIDTH, 4
        )
        return Observation(
            framecount=framecount,
            ram=np.frombuffer(ram_bytes, dtype=np.uint8),
            rgb=np.ascontiguousarray(px[:, :, 1:]),
        )

    def reset(self, capture_frame: int) -> Observation:
        """Load the savestate captured at ``capture_frame`` and return the state."""
        try:
            ordinal = self.capture_frames.index(int(capture_frame))
        except ValueError:
            raise KeyError(
                f"no savestate at movie frame {capture_frame}; "
                f"available: {len(self.capture_frames)} points"
            ) from None
        self.resets_served += 1
        return self._command(OP_RESET, ordinal)

    def save_scratch(self, slot: int) -> Observation:
        """Snapshot the live state into scratch ``slot``. No frame is advanced."""
        if not 0 <= slot <= 0xFFFF:
            raise KeyError(f"scratch slot {slot} out of range")
        self.scratch_saves += 1
        return self._command(OP_SAVE_SCRATCH, slot)

    def load_scratch(self, slot: int) -> Observation:
        """Restore scratch ``slot``, then advance one null-input frame as ``reset`` does.

        Restoring is O(1) instead of O(prefix length), which is what makes practising from
        policy-visited states affordable: replay the prefix once, snapshot, then reload per rollout.
        """
        if not 0 <= slot <= 0xFFFF:
            raise KeyError(f"scratch slot {slot} out of range")
        self.scratch_loads += 1
        return self._command(OP_LOAD_SCRATCH, slot)

    def reset_ordinal(self, ordinal: int) -> Observation:
        """Load the ``ordinal``-th savestate (0-based, sorted by movie frame)."""
        if not 0 <= ordinal < len(self.capture_frames):
            raise KeyError(f"ordinal {ordinal} out of range")
        self.resets_served += 1
        return self._command(OP_RESET, ordinal)

    def step(self, action_byte: int) -> Observation:
        self.steps_served += 1
        return self._command(OP_STEP, int(action_byte) & 0xFF)

    def stats(self) -> dict:
        return {
            "n_states": self.n_states,
            "build_seconds": round(self.build_seconds, 1),
            "steps_served": self.steps_served,
            "resets_served": self.resets_served,
        }
