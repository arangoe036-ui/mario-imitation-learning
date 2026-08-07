"""Render the headline clips as GIFs, from the EXACT generation rule that produced the numbers.

Two properties, each of which cost a failed attempt, recorded so nobody repeats them:

1. **The rollout is not reimplemented.** The session is wrapped in a recorder that tees every
   ``Observation.rgb`` to ffmpeg, and that wrapper is handed to the same ``button_mask_eval.rollout``
   the measured arms used. Identical generation, identical RNG, plus frames.

2. **A NAMED episode cannot be re-filmed at all.** ``session.reset()`` restores RAM, but SMB's own
   pseudo-random state advances with total frames elapsed and is not cleared by a level restart, so an
   episode's outcome depends on the session's whole history. Measured on seed 99 of
   ``peak_PK32_84_s0_t0.7_200`` (trace: x=3266): **x=312 run first on a fresh session, x=1537 after
   replaying episodes 0..98.** Prefix replay gets closer and still does not reproduce, because
   ``resumable`` may have resumed the original arm from a partial, making its true history
   unrecoverable. Block 58's determinism check compared repeats in the *same* position, which is why it
   passed -- a different question.

   **So we do not reproduce. We film every episode and keep the ones that actually do the thing.**
   The caption then describes the footage rather than a trace, and cannot be wrong.

Never call ``pick_device`` here: it probes MPS, which puts every subsequently spawned FCEUX child on
Qt's broken software OpenGL backend, irreversibly. CPU explicitly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.button_mask_eval import rollout  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.live import _FfmpegWriter  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402

CKDIR = ROOT / "data/bc_scaleup"
TRACES = ROOT / "data/traces"
OUT = ROOT.parent / "gifs"
TEMP = 0.7


class RecordingSession:
    """Tee every observation's native RGB frame to ffmpeg when armed. Generation is untouched."""

    def __init__(self, inner, writer=None):
        self._inner, self._w = inner, writer

    def arm(self, writer):
        self._w = writer

    def _tee(self, obs):
        if self._w is not None and obs is not None and getattr(obs, "rgb", None) is not None:
            self._w.write(obs.rgb)
        return obs

    def reset(self, *a, **k):
        return self._tee(self._inner.reset(*a, **k))

    def step(self, *a, **k):
        return self._tee(self._inner.step(*a, **k))

    def load_scratch(self, *a, **k):
        return self._tee(self._inner.load_scratch(*a, **k))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def load(name: str):
    blob = torch.load(CKDIR / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig.from_dict(cfg)
    p = BCPolicy(cfg)
    p.load_state_dict(blob["model_state"])
    p.eval()
    return p, cfg, blob


def trace_max_x(trace_name: str) -> dict[int, int]:
    p = TRACES / trace_name
    if not p.exists():
        return {}
    eps = json.loads(p.read_text())
    eps = eps.get("episodes", eps) if isinstance(eps, dict) else eps
    return {int(e.get("seed", i)): max((f[0] for f in e.get("frames") or []), default=0)
            for i, e in enumerate(eps)}


def to_gif(mp4: Path, gif: Path, fps: int = 15, width: int = 448) -> None:
    """Two-pass palette. flags=neighbor keeps the pixel art sharp instead of blurring it."""
    pal = mp4.with_suffix(".pal.png")
    vf = f"fps={fps},scale={width}:-1:flags=neighbor"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                    "-vf", f"{vf},palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(pal),
                    "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
                    str(gif)], check=True)
    pal.unlink(missing_ok=True)
    # GitHub only renders a GIF inline under ~10 MB; step down rather than ship one it will not show.
    for f2, w2 in ((12, 384), (10, 320)):
        if gif.stat().st_size <= 10 * 1024 ** 2:
            break
        to_gif(mp4, gif, fps=f2, width=w2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    lut = G.class_lengths({k: z[k] for k in ("rows", "joints", "lengths")}, n_cls)

    manifest = {
        "terminator": {"STALL": RB.STALL, "CAP_FRAMES": RB.CAP_FRAMES},
        "temperature": TEMP,
        "reproduction_note": (
            "A named episode cannot be re-filmed: SMB's pseudo-random state advances with total frames "
            "elapsed and is not reset by a level restart, and the original arm's session history is "
            "unrecoverable. So every episode was filmed live and the takes matching each claim were "
            "kept. episode_seed identifies the take within THIS filming run, not the published arm."),
        "clips": [],
    }

    mx = trace_max_x("peak_PK32_84_s0_t0.7_200.json")
    # An episode cannot be reproduced by seed: `resumable` may have resumed the original arm from a
    # partial, so its session history at episode N is unrecoverable. Do not try to reproduce a named
    # episode -- FILM EVERY EPISODE and keep the ones that actually do the thing. Honest by construction.
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tmp = OUT / "_takes"; tmp.mkdir(exist_ok=True)
    policy, cfg, _ = load("PK32_84_s0")
    inner = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    warm_session(inner, start.frame)
    sess = RecordingSession(inner)
    takes: list[tuple[int, int, Path]] = []
    try:
        for seed in range(N):
            f = tmp / f"ep{seed:03d}.mp4"
            w = _FfmpegWriter(f, 256, 240)
            sess.arm(w)
            t = rollout(sess, policy, cfg, start, seed, lut, byte_of, None, temp=TEMP)
            w.close()
            got = max((f2[0] for f2 in t.frames), default=0)
            takes.append((seed, got, f))
            if got >= 3100:
                print(f"  ep{seed}: COMPLETION max_x={got}", flush=True)
            elif seed % 25 == 0:
                print(f"  ..ep{seed} max_x={got}", flush=True)
    finally:
        inner.close()

    want = [("01_completion_1-1", 3100, 10 ** 9,
             "1-1 completed from the level start; the stage advances to 1-2"),
            ("02_koopas_cleared", 1300, 3099,
             "past the Koopas at x=1248 - the obstacle class where learning beats the matched script")]
    keep = set()
    for tag, lo, hi, caption in want:
        hits = sorted([t for t in takes if lo <= t[1] <= hi], key=lambda z: -z[1])
        if not hits:
            print(f"  no take for {tag} (lo={lo})"); continue
        seed, got, src = hits[0]
        dst = OUT / f"{tag}.mp4"
        src.replace(dst); keep.add(dst)
        manifest["clips"].append({"file": f"{tag}.gif", "checkpoint": "PK32_84_s0",
                                  "episode_seed": seed, "max_x": got,
                                  "episodes_filmed": len(takes), "caption": caption})
        print(f"KEEP {tag}: ep{seed} max_x={got}")
    for _, _, f in takes:
        f.unlink(missing_ok=True)
    tmp.rmdir()

    for c in manifest["clips"]:
        mp4 = OUT / c["file"].replace(".gif", ".mp4")
        if mp4.exists():
            to_gif(mp4, OUT / c["file"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("manifest:", OUT / "manifest.json")


if __name__ == "__main__":
    main()
