"""Run live evaluation in an isolated, CPU-only process.

Why this exists
---------------
Once a process *uses* the MPS (Metal) backend, any FCEUX child it spawns falls back to
Qt's software OpenGL path -- "known to be broken on macOS Tahoe" -- and segfaults.
Measured directly:

    before importing torch    exit=0    GL=Metal
    after importing torch     exit=0    GL=Metal
    after using MPS           exit=-11  GL=SOFTWARE

That is what made the first sweep's evaluation flaky (60 of 940 episodes truncated):
training ran on MPS and then launched the emulator from the same process. Training on
MPS is worth keeping -- it is twice as fast -- so evaluation moves out instead.

This module is the child. It loads a checkpoint onto the CPU, never touches MPS, runs
the episodes, and writes one JSON object to stdout. The policy is ~150k parameters, so
CPU inference is nowhere near the bottleneck; the emulator is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tasdata-bc-eval-worker")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--rom", required=True)
    p.add_argument(
        "--selection", default="greedy",
        choices=("greedy", "sticky", "temperature", "threshold", "sample"),
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sticky-p", type=float, default=0.25)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--levels", nargs="*", default=["1-1"])
    p.add_argument("--expert-movie", default=None)
    p.add_argument("--stall-frames", type=int, default=300)
    p.add_argument("--max-frames", type=int, default=3000)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--expert-bytes", default=None, help="json file: list of expert action bytes")
    p.add_argument("--workers", type=int, default=1, help="parallel emulators")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Import torch only here, and only ever ask it for the CPU.
    import torch

    from .live import LivePlayer
    from .tokens import ActionVocab
    from .train import load_checkpoint

    torch.set_num_threads(2)
    vocab = ActionVocab.load(args.vocab)
    policy, blob = load_checkpoint(args.checkpoint, device="cpu")
    pcfg = blob.get("policy_config", {})
    n_prev = int(pcfg.get("n_prev_actions", 0))
    head_type = str(pcfg.get("head_type", "categorical"))

    thresholds = None
    if head_type == "bernoulli":
        from ..buttons import NES_BUTTON_ORDER

        thr = blob.get("thresholds")
        if not thr:
            raise SystemExit("checkpoint has no calibrated thresholds")
        thresholds = [float(thr[n]) for n in NES_BUTTON_ORDER]

    expert_bytes = None
    if args.expert_bytes:
        expert_bytes = set(json.loads(Path(args.expert_bytes).read_text()))

    player = LivePlayer(
        args.rom,
        vocab,
        max_frames=args.max_frames,
        device="cpu",
        n_prev_actions=n_prev,
        expert_movie=args.expert_movie,
        stall_limit=args.stall_frames,
        head_type=head_type,
        thresholds=thresholds,
        expert_bytes=expert_bytes,
    )
    result = player.evaluate(
        policy,
        seeds=args.seeds,
        selection=args.selection,
        temperature=args.temperature,
        sticky_p=args.sticky_p,
        levels=tuple(args.levels),
        retries=args.retries,
        workers=args.workers,
    )
    json.dump(result, sys.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
