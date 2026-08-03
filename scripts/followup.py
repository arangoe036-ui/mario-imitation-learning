"""Follow-up run: failure taxonomy, the pipe-2 ceiling, the sustain fix, fixed-epoch scaling.

Waits for any existing emulator job to finish before touching FCEUX (the lock permits one).
Same conventions as the overnight run: results stream to JSONL, each task is isolated, a
failure is recorded and the rest continue.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    calibrate,
    eval_live,
    fresh_policy,
    load_policy,
    onset_metrics,
    random_rows,
    save_policy,
    train_policy,
    wilson,
)
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.train import make_loader  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / "data/followup.jsonl"
CKPTS = ROOT / "data/bc_followup"
ROM, MOVIE = O.ROM, O.MOVIE
STAGE2 = O.STAGE2_CKPT
ROUND3 = ROOT / "data/bc_overnight/round3_ratio1to1.pt"
A_INDEX = NES_BUTTON_ORDER.index("A")
_t0 = time.time()


def log(*a):
    print(f"[{(time.time() - _t0) / 60:6.1f}m]", *a, flush=True)


def emit(kind, **p):
    with JSONL.open("a") as fh:
        fh.write(json.dumps({"kind": kind, "t": datetime.now(timezone.utc).isoformat(
            timespec="seconds"), **p}, default=str) + "\n")


def task(name):
    def wrap(fn):
        def run(*a, **kw):
            log(f"=== START {name}")
            t = time.time()
            try:
                out = fn(*a, **kw)
                emit("task_done", task=name, seconds=round(time.time() - t, 1), result=out)
                log(f"=== DONE {name} ({(time.time() - t) / 60:.1f}m)")
                return out
            except Exception as exc:
                emit("task_failed", task=name, error=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc()[-1500:])
                log(f"=== FAILED {name}: {exc}")
                return None
        return run
    return wrap


def wait_for_emulator(timeout: float = 7200) -> None:
    """Block until no other job holds FCEUX."""
    pidfile = ROOT / "data/chain_position.pid"
    if pidfile.exists():
        pid = int(pidfile.read_text().strip())
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(30)
    log("emulator free")


# ------------------------------------------------------------------ 1. taxonomy

@task("failure_taxonomy")
def taxonomy(ctx) -> dict:
    out = {}
    targets = [("stage2_armB", STAGE2), ("arm_a_round3", ROUND3)]
    for label, path in targets:
        if not Path(path).exists():
            out[label] = {"missing": str(path)}
            continue
        policy, cfg, _ = load_policy(Path(path))
        cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
        thr = cal.vector.astype(np.float64)
        with FceuxSession(ROM, MOVIE, ctx.frames_needed()) as session:
            live = eval_live(session, policy, thr, ctx.starts, ctx.vocab, cfg,
                             seeds=200, expert_bytes=ctx.expert_bytes)
        out[label] = live
        for lvl in ("1-1", "2-1"):
            d = live.get(lvl, {})
            log(f"  {label} {lvl}: {d.get('end_class_pct')}  x_med {d.get('x_median')}")
        emit("taxonomy", checkpoint=label, live=live)
    return out


# ------------------------------------------------------------------ 2. pipe-2 ceiling

@task("pipe2_ceiling_emulator")
def pipe2(ctx) -> dict:
    """Ground truth: sweep A-hold length at the pipe and see what actually clears it."""
    from tasdata.ram import read_smb

    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    RUN = NES_BUTTON_BITS["Right"] | NES_BUTTON_BITS["B"]
    A = NES_BUTTON_BITS["A"]
    results = []
    with FceuxSession(ROM, MOVIE, ctx.frames_needed()) as session:
        for hold in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]:
            best = {"hold": hold, "max_x": 0, "cleared_pipe1": False, "cleared_pipe2": False}
            # Try several jump-trigger positions; the model does not get to pick perfectly.
            for trigger in (540, 550, 560, 570, 580):
                obs = session.reset(start.frame)
                jumped = 0
                maxx = 0
                for _ in range(1400):
                    st = read_smb(obs.ram, obs.framecount)
                    x = st.x_position
                    maxx = max(maxx, x)
                    byte = RUN
                    # jump at the first pipe unconditionally, then at the chosen trigger
                    if 395 <= x <= 420 and jumped == 0:
                        jumped = 1
                        hold_left = 12
                    if jumped == 1 and hold_left > 0:
                        byte |= A
                        hold_left -= 1
                    if x >= trigger and jumped <= 1:
                        jumped = 2
                        hold2 = hold
                    if jumped == 2 and hold2 > 0:
                        byte |= A
                        hold2 -= 1
                    obs = session.step(byte)
                if maxx > best["max_x"]:
                    best = {"hold": hold, "max_x": int(maxx), "trigger": trigger,
                            "cleared_pipe1": bool(maxx > 470),
                            "cleared_pipe2": bool(maxx > 630)}
            results.append(best)
            log(f"  A-hold {hold:3d} frames -> max x {best['max_x']:5d} "
                f"pipe2={'YES' if best['cleared_pipe2'] else 'no'}")
    ok = [r for r in results if r["cleared_pipe2"]]
    minimum = min((r["hold"] for r in ok), default=None)
    verdict = (f"pipe 2 requires an A-hold of at least {minimum} frames; the model's "
               f"maximum measured hold is 8, so it is UNREACHABLE by this policy"
               if minimum and minimum > 8 else
               (f"pipe 2 clears from a {minimum}-frame hold, within the model's 8-frame "
                f"maximum -- reachable, so the failure is timing/placement, not height"
                if minimum else
                "no tested hold cleared pipe 2 with this fixed approach"))
    log(f"  VERDICT: {verdict}")
    return {"sweep": results, "min_hold_clearing_pipe2": minimum, "verdict": verdict,
            "model_max_hold": 8}


# ------------------------------------------------------------------ 3. sustain

@task("sustain_diagnosis")
def sustain_diagnosis(ctx) -> dict:
    """Is p(A) collapsing on continuation frames while onsets stay high?"""
    policy, cfg, _ = load_policy(STAGE2)
    rows = list(range(0, min(40_000, len(ctx.val_set))))
    loader = make_loader(Subset(ctx.val_set, rows), batch_size=256, shuffle=False,
                         num_workers=0)
    P, Y = [], []
    with torch.no_grad():
        for obs, _p, bits, _o in loader:
            P.append(torch.sigmoid(policy(obs)).numpy())
            Y.append(bits.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    a, pa = Y[:, A_INDEX] > 0, P[:, A_INDEX]
    prev = np.zeros_like(a)
    prev[1:] = a[:-1]
    onset = a & ~prev
    sustain = a & prev
    idle = ~a & ~prev
    out = {
        "p_A_at_onset": float(pa[onset].mean()) if onset.any() else 0.0,
        "p_A_at_sustain": float(pa[sustain].mean()) if sustain.any() else 0.0,
        "p_A_when_idle": float(pa[idle].mean()) if idle.any() else 0.0,
        "n_onset": int(onset.sum()), "n_sustain": int(sustain.sum()),
        "sustain_over_onset": (float(pa[sustain].mean() / max(pa[onset].mean(), 1e-9))
                               if onset.any() and sustain.any() else 0.0),
    }
    out["verdict"] = (
        "CONFIRMED: p(A) on continuation frames is below p(A) at onsets, so the 10x onset "
        "weighting taught initiation at the expense of sustain"
        if out["p_A_at_sustain"] < out["p_A_at_onset"] else
        "NOT confirmed: continuation probability is at or above onset probability, so the "
        "short holds come from the sampling rule, not the training signal")
    log(f"  p(A) onset {out['p_A_at_onset']:.3f}  sustain {out['p_A_at_sustain']:.3f}  "
        f"idle {out['p_A_when_idle']:.3f}")
    log(f"  {out['verdict']}")
    return out


@task("sustain_arms")
def sustain_arms(ctx) -> dict:
    """(a) reweight sustain too, (b) onset 3x, (d) control. (c) is reported separately."""
    from tasdata.bc.bernoulli import bce_with_onset_weights

    train_set = ctx.dataset(ctx.expert_train)
    rows = random_rows(train_set, 300_000, seed=11)
    ds = Subset(train_set, rows)

    def sustain_loss(logits, bits, onset, *, onset_w, sustain_w):
        base = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, bits, reduction="none")
        w = torch.ones_like(base)
        w = w + (onset_w - 1.0) * onset
        if sustain_w != 1.0:
            # held now and not an onset == a continuation frame
            cont = (bits > 0).float() * (1.0 - onset)
            w = w + (sustain_w - 1.0) * cont
        return (base * w).mean()

    arms = {
        "d_control_onset10x": dict(onset_w=10.0, sustain_w=1.0),
        "a_sustain_and_onset": dict(onset_w=10.0, sustain_w=5.0),
        "b_onset3x": dict(onset_w=3.0, sustain_w=1.0),
    }
    out = []
    for tag, kw in arms.items():
        policy = fresh_policy(ctx.cfg, seed=0)
        policy = policy.to(torch.device("cpu"))
        policy.train()
        opt = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = make_loader(ds, batch_size=128, shuffle=True, num_workers=0, seed=0)
        step = 0
        while step < 2000:
            for obs, _p, bits, onset in loader:
                loss = sustain_loss(policy(obs), bits.float(), onset.float(), **kw)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 400 == 0:
                    log(f"    {tag} step {step}/2000 loss {float(loss.detach()):.4f}")
                if step >= 2000:
                    break
        policy.eval()
        save_policy(CKPTS / f"{tag}.pt", policy, ctx.cfg,
                    {n: 0.5 for n in NES_BUTTON_ORDER})
        row = O.full_eval(ctx, policy, ctx.cfg, tag, seeds=200)
        row["arm"] = tag
        one = row["live"].get("1-1", {})
        row["max_A_hold"] = one.get("longest_a_hold_max")
        out.append(row)
        emit("sustain_arm", **row)
        log(f"  {tag}: A recall {row['offline']['onset_recall']['A'] * 100:.1f}%  "
            f"pipe1 {one.get('pipe1_rate', 0) * 100:.1f}%  "
            f"pipe2 {one.get('pipe2_rate', 0) if 'pipe2_rate' in one else 'n/a'}  "
            f"maxAhold {row['max_A_hold']}  x_med {one.get('x_median')}")
    return {"arms": out}


# ------------------------------------------------------------------ 5. scaling, fixed epochs

@task("scaling_fixed_epochs")
def scaling(ctx) -> dict:
    """Same number of passes over the data, so larger subsets are not starved of steps."""
    full = ctx.dataset(ctx.expert_train)
    EPOCHS = 0.26          # what 2,000 steps of batch 128 amounted to on the 100% set
    out = []
    for frac in (0.10, 0.25, 0.50, 1.00):
        rows = random_rows(full, int(len(full) * frac), seed=7)
        steps = max(200, int(EPOCHS * len(rows) / 128))
        policy = fresh_policy(ctx.cfg, seed=0)
        policy = train_policy(policy, Subset(full, rows), steps=steps, lr=3e-4,
                              onset_weight=10.0, seed=0, log=log)
        row = O.full_eval(ctx, policy, ctx.cfg, f"epochmatched_{int(frac * 100)}pct",
                          seeds=200)
        row.update({"fraction": frac, "frames": len(rows), "steps": steps,
                    "epochs": EPOCHS})
        out.append(row)
        emit("scaling_fixed_epochs", **row)
        log(f"  {frac:.0%} {len(rows):,}f {steps} steps: A recall "
            f"{row['offline']['onset_recall']['A'] * 100:.1f}%  "
            f"pipe1 {row['live'].get('1-1', {}).get('pipe1_rate', 0) * 100:.1f}%")
    return {"points": out, "note": "constant epochs; steps scale with subset size"}


@task("oracle_rerun")
def oracle(ctx) -> dict:
    return O.tier3(ctx)


def main() -> None:
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    CKPTS.mkdir(parents=True, exist_ok=True)
    wait_for_emulator()
    ctx = O.Ctx()
    log("context ready")
    taxonomy(ctx)
    pipe2(ctx)
    sustain_diagnosis(ctx)
    sustain_arms(ctx)
    scaling(ctx)
    oracle(ctx)
    emit("run_end", minutes=round((time.time() - _t0) / 60, 1))
    log("ALL DONE")


if __name__ == "__main__":
    main()
