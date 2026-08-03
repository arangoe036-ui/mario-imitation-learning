"""Stage 2c: two Bernoulli arms, run as one sweep so they are directly comparable.

``arm A`` -- ``bernoulli_only``: the control. Per-button sigmoid outputs, plain BCE.
``arm B`` -- ``bernoulli_onset10x``: identical, plus 10x loss weight on the specific
button that turns on in that frame.

Same seed, same steps, same eval schedule, same split. The only difference between the
arms is ``onset_weight``, so any difference in behaviour is attributable to it.

Prediction under test (from the categorical diagnosis): A confidence was 19.25% during
holds versus 10.21% at rest, so once a jump starts it should sustain itself above
threshold. Arm A is expected to clear the pipe with badly-timed jumps; arm B to clear it
with better timing (higher onset separation, higher A-onset recall).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..buttons import NES_BUTTON_ORDER
from .bernoulli import expert_button_rates
from .sweep import Corpus, append_jsonl, environment, run_eval_isolated
from .train import TrainConfig, train

#: Steps to evaluate at. Early and log-spaced: everything peaked by 3,000 last time.
EVAL_STEPS: tuple[int, ...] = (100, 250, 500, 1000, 2000, 3000)

#: Selection rules for a Bernoulli head. ``threshold`` is deterministic and primary.
SELECTION_RULES: tuple[tuple[str, dict], ...] = (
    ("threshold", {"selection": "threshold"}),
    ("threshold+sticky0.25", {"selection": "sticky", "sticky_p": 0.25}),
    ("per-button-sample", {"selection": "sample"}),
)


def arm_configs(*, steps: int = 3000, num_workers: int = 2) -> list[TrainConfig]:
    common = dict(
        d_model=64, n_layers=1, n_heads=2, cnn_channels=(16, 32, 32), lr=1e-3,
        steps=steps, eval_steps=EVAL_STEPS, num_workers=num_workers,
        head_type="bernoulli", n_prev_actions=0, seed=0,
    )
    return [
        TrainConfig(name="A_bernoulli_only", onset_weight=1.0, **common),
        TrainConfig(name="B_bernoulli_onset10x", onset_weight=10.0, **common),
    ]


def expert_reference(corpus: Corpus) -> tuple[dict[str, float], set[int], Path]:
    """Expert press rates and the set of combinations that actually occur.

    Rates come from the *training* corpus, not val: the val split holds Right where
    train taps it (2.99% vs 7.95% action-change rate), so calibrating to val would
    target the wrong behaviour.
    """
    actions = [r.actions for r in corpus.train_runs]
    rates = expert_button_rates(actions)
    combos = set()
    for a in actions:
        combos.update(int(b) for b in np.unique(a))
    path = Path("data/expert_bytes.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(combos)))
    return rates, combos, path


def run_arms(
    corpus: Corpus,
    rom_path: str,
    configs: list[TrainConfig],
    *,
    out_dir: str = "data/bc3",
    results_path: str = "data/stage2c_results.jsonl",
    expert_movie: str = "data/movies/happylee_mars608-smb-warpless.fm2",
    levels: tuple[str, ...] = ("1-1", "2-1"),
    train_seeds: int = 5,
    final_seeds: int = 20,
    live_frames: int = 2500,
    stall_frames: int = 300,
    workers: int = 4,
    log=print,
) -> None:
    """Train both arms, evaluating all three selection rules at every checkpoint."""
    rates, combos, bytes_path = expert_reference(corpus)
    log("expert press rates (train corpus):")
    for n in NES_BUTTON_ORDER:
        log(f"    {n:8s} {rates[n] * 100:6.2f}%")
    log(f"  {len(combos)} distinct button combinations occur in expert data")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # MPS would poison every FCEUX child

    for config in configs:
        log(f"\n=== {config.name} (onset_weight={config.onset_weight}) ===")
        started = time.perf_counter()
        try:
            train_set = corpus.dataset_bernoulli("train", stack=config.stack)
            val_set = corpus.dataset_bernoulli("val", stack=config.stack)

            def on_eval(metrics, ckpt, policy, _cfg=config, _final=False):
                seeds = final_seeds if metrics["step"] == _cfg.steps else train_seeds
                by_rule = {}
                for name, kw in SELECTION_RULES:
                    by_rule[name] = run_eval_isolated(
                        ckpt, corpus.vocab_path, rom_path,
                        seeds=seeds, levels=levels, expert_movie=expert_movie,
                        stall_frames=stall_frames, max_frames=live_frames,
                        expert_bytes=str(bytes_path), workers=workers, **kw,
                    )
                append_jsonl(results_path, {
                    "kind": "arm_eval",
                    "arm": _cfg.name,
                    "onset_weight": _cfg.onset_weight,
                    "step": metrics["step"],
                    "seeds": seeds,
                    "timestamp": time.time(),
                    "train_config": _cfg.to_dict(),
                    "val": metrics,
                    "live_by_rule": by_rule,
                    "environment": environment("cpu"),
                })
                for name, res in by_rule.items():
                    if "error" in res:
                        log(f"      {name:22s} ERROR {res['error'][:60]}")
                        continue
                    p1 = res.get("cleared_pipe1_rate")
                    log(
                        f"      {name:22s} pipe1={'n/a' if p1 is None else format(p1 * 100, '.0f') + '%':>5s} "
                        f"x_med={res.get('furthest_x', {}).get('median')} "
                        f"Ahold={res.get('longest_a_hold', {}).get('max')} "
                        f"Apress={res.get('a_presses', {}).get('median')} "
                        f"novel={res.get('novel_combo_rate', {}).get('mean', 0) * 100:.1f}% "
                        f"flakes={res.get('retried_flakes')}"
                    )

            train(
                config, train_set, val_set, corpus.vocab,
                out_dir=out_dir, vocab_path=corpus.vocab_path, device=device,
                target_rates=rates, expert_bytes=combos, on_eval=on_eval, log=log,
            )
            append_jsonl(results_path, {
                "kind": "arm_done", "arm": config.name,
                "wall_seconds": round(time.perf_counter() - started, 1),
            })
        except Exception as exc:
            import traceback

            append_jsonl(results_path, {
                "kind": "arm_failed", "arm": config.name,
                "error": f"{type(exc).__name__}: {exc}"[:400],
                "traceback": traceback.format_exc(limit=6)[:1500],
            })
            log(f"  !! {config.name} FAILED: {type(exc).__name__}: {exc}")
