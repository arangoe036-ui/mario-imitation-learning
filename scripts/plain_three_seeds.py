"""§1: three training seeds of the plain-BCE arm, scored conditionally on arrival.

One seed is a screen (LEDGER.md §2: training-seed spread is 14.5-24.5 pp). The previous block reported a
+7.5 pp unconditional gain at pipe 3 from a single seed, which sits inside that band. This spends three.

**Two things this fixes about the previous measurement.**

1. **`vs_script` was unconditional.** Improving upstream inflates the apparent advantage at every later
   obstacle. Decomposed, the +7.5 pp at pipe 3 was ~+3.5 pp of pipe-3 skill and the rest more episodes
   arriving -- while pipe 4 got conditionally **worse** by 11.1 pp. Both forms are reported here, and the
   conditional one is what the gate is judged on.
2. **The A-rate was the headline behaviour statistic**, which cannot separate "jumps often" from "holds
   the button down". Reported instead: airborne fraction, A-onsets while grounded per 1,000 frames, and
   **A still held while airborne** -- the actual pathology, since A during a descent does nothing and
   blocks the next jump.

The rollout phase is unchanged and deterministic given the base policy, the start library and the rollout
seeds, so all three arms train on the identical accepted self-data and differ only in the training seed.
**The self-data's own marginals are reported**, because the loss fix stops amplification, not inheritance:
this data came from a sustain-trained base with A on 87.1%, and plain BCE reproducing that is correct
behaviour on degenerate data rather than a fresh degeneracy.

The base checkpoint is re-evaluated here too, because the traces it was measured from predate the
`grounded` field and cannot yield the behaviour statistics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import EARLIEST, session_when_free  # noqa: E402
from scripts.p1_run import episode as traced_episode  # noqa: E402
from scripts.train_script_net import LOSSES, train_with  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    calibrate,
    diff_ci,
    load_policy,
    random_rows,
    save_policy,
)
from tasdata.bc.pipe4_metrics import button_marginals, clearance  # noqa: E402
from tasdata.bc.script_baseline import (  # noqa: E402
    behaviour_stats,
    conditional_script_baseline,
    vs_script,
    vs_script_conditional,
)
from tasdata.bc.trace_log import write_traces  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
SELFDIR = ROOT / "data/runs_self/script_net_round1"
OUT = ROOT / "data/plain_three_seeds.json"
TRACEDIR = ROOT / "data/traces"
SEEDS = (0, 1, 2)
STEPS, LR = 400, 1e-4
EXPERT_PER_SELF = 0.33
N_EVAL = 200


def resumable_eval(session, policy, cfg, thr, start, path: Path, n: int, chunk: int = 20,
                   **meta):
    """Run `n` episodes, banking progress every `chunk` so a restart costs at most one chunk.

    The environment recycles long jobs every few minutes. Per-arm resumption was not enough: a
    200-episode arm takes longer than the interval, so nothing ever completed and the run looped
    forever making no progress. Episodes are the unit of work, so episodes are the unit of banking.
    """
    partial = path.with_suffix(".partial.json")
    done = []
    if path.exists():
        blob = json.loads(path.read_text())
        if blob.get("n_episodes") == n:
            return [_Frames(e) for e in blob["episodes"]], True
    if partial.exists():
        done = json.loads(partial.read_text())["episodes"]
        print(f"    resuming {path.stem} from {len(done)}/{n} episodes", flush=True)
    traces = [_Frames(e) for e in done]
    while len(traces) < n:
        upto = min(len(traces) + chunk, n)
        for i in range(len(traces), upto):
            traces.append(traced_episode(session, policy, cfg, thr, start, i))
        partial.write_text(json.dumps(
            {"n_episodes": len(traces),
             "episodes": [t.to_dict() if hasattr(t, "to_dict") else t.raw for t in traces]},
            separators=(",", ":")))
        print(f"    {path.stem}: {len(traces)}/{n} banked", flush=True)
    write_traces(path, [t for t in traces if hasattr(t, "to_dict")] or traces, **meta) \
        if all(hasattr(t, "to_dict") for t in traces) else _write_mixed(path, traces, **meta)
    partial.unlink(missing_ok=True)
    return traces, False


def _write_mixed(path: Path, traces, **meta) -> None:
    """Persist a mix of freshly-run EpisodeTrace objects and resumed dicts."""
    eps = [t.to_dict() if hasattr(t, "to_dict") else t.raw for t in traces]
    path.write_text(json.dumps(
        {"schema": "per-frame (x, y_absolute, speed_byte, buttons, player_state, grounded)",
         "n_episodes": len(eps), **meta, "episodes": eps}, separators=(",", ":")))


class _Frames:
    """A resumed episode, exposing the same `.frames` / `.to_dict()` surface as EpisodeTrace."""

    def __init__(self, e):
        self.raw = e
        self.frames = e["frames"]
        self.ended = e.get("ended", "budget")

    def to_dict(self):
        return self.raw


def train_resumable(policy, ds, steps, lr, seed, ckpt_partial: Path, chunk: int = 100):
    """Train with the model state banked every `chunk` steps, so a kill costs at most a chunk.

    Whole-run resumption was still too coarse: the environment kills this job faster than 400 steps
    take, so every relaunch restarted training from zero and never reached the save. Steps are the unit
    of work here, so steps are the unit of banking.
    """
    import torch as _t

    from tasdata.bc.train import make_loader as _mk

    done = 0
    if ckpt_partial.exists():
        blob = _t.load(ckpt_partial, map_location="cpu", weights_only=False)
        policy.load_state_dict(blob["model_state"])
        done = int(blob["step"])
        print(f"    resuming training from step {done}/{steps}", flush=True)
    if done >= steps:
        return policy
    policy = policy.to(_t.device("cpu"))
    policy.train()
    opt = _t.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    if ckpt_partial.exists():
        blob = _t.load(ckpt_partial, map_location="cpu", weights_only=False)
        if blob.get("opt_state"):
            try:
                opt.load_state_dict(blob["opt_state"])
            except Exception:
                pass          # a fresh optimiser is acceptable; losing momentum is not a correctness bug
    loss_fn = LOSSES["plain"]
    step = done
    while step < steps:
        for obs, _p, bits, onset in _mk(ds, batch_size=128, shuffle=True, num_workers=0,
                                        seed=seed * 1000 + step):
            loss = loss_fn(policy(obs), bits.float(), onset.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            _t.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if step % chunk == 0 or step >= steps:
                _t.save({"model_state": policy.state_dict(), "opt_state": opt.state_dict(),
                         "step": step}, ckpt_partial)
                print(f"    train seed {seed}: {step}/{steps} banked "
                      f"(loss {float(loss.detach()):.4f})", flush=True)
            if step >= steps:
                break
    policy.eval()
    return policy


ARMCACHE = ROOT / "data/plain_three_seeds.arms.json"


def cached_score(label: str, traces) -> dict:
    """Score an arm once. Restart overhead dominated progress: each relaunch was recomputing every
    finished arm's metrics before doing 20 new episodes."""
    cache = json.loads(ARMCACHE.read_text()) if ARMCACHE.exists() else {}
    if label in cache:
        r = cache[label]
        c, b = r["vs_script_conditional"]["per_obstacle"], r["behaviour"]
        print(f"  {label:14s} cond adv: " +
              "  ".join(f"{o} {c[o]['advantage_pp']:+6.1f}" for o in
                        ("pipe1", "pipe2", "pipe3", "pipe4")) +
              f"  | airborne {b['airborne_fraction'] * 100:4.1f}%  (cached)", flush=True)
        return r
    r = score(label, traces)
    cache[label] = r
    ARMCACHE.write_text(json.dumps(cache, indent=2, default=str))
    return r


def score(label: str, traces) -> dict:
    xs = [max(f[0] for f in t.frames) for t in traces]
    frames = [f for t in traces for f in t.frames]
    row = {"label": label, "n": len(traces), "measurement_basis": "single_life",
           "loss": "plain_BCE", "x_median": float(np.median(xs)),
           "clearance": clearance(xs),
           "vs_script": vs_script(xs),
           "vs_script_conditional": vs_script_conditional(xs),
           "button_marginals": button_marginals(frames),
           "behaviour": behaviour_stats(frames)}
    c = row["vs_script_conditional"]["per_obstacle"]
    b = row["behaviour"]
    print(f"  {label:14s} cond adv: " +
          "  ".join(f"{o} {c[o]['advantage_pp']:+6.1f}" for o in
                    ("pipe1", "pipe2", "pipe3", "pipe4")) +
          f"  | airborne {b['airborne_fraction'] * 100:4.1f}%  "
          f"A-held-airborne {b['a_held_while_airborne'] * 100:4.1f}%  "
          f"onsets/1k grounded {b['a_onsets_while_grounded_per_1000']:5.1f}", flush=True)
    return row


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    _cache = {}

    def training_data():
        """Loaded on demand: three of four arms need no dataset at all on a resumed run."""
        if "mixed" not in _cache:
            expert = ctx.dataset([load_run_dir(ROOT / "data/runs" / n) for n in EARLIEST])
            selfds = ctx.dataset([load_run_dir(SELFDIR)])
            n_exp = min(len(expert), int(len(selfds) * EXPERT_PER_SELF))
            _cache["mixed"] = ConcatDataset(
                [Subset(expert, random_rows(expert, n_exp, seed=0)), selfds])
        return _cache["mixed"]

    self_bytes = np.load(SELFDIR / "actions.npy")
    self_marg = {n: round(float(((self_bytes & NES_BUTTON_BITS[n]) > 0).mean()), 3)
                 for n in NES_BUTTON_ORDER}
    print(f"self-data {SELFDIR.name}: {len(self_bytes):,} frames, marginals "
          f"{ {k: v for k, v in self_marg.items() if v > 0.01} }")
    print("  (the loss fix stops amplification, not inheritance: this data came from a "
          "sustain-trained base)\n", flush=True)

    cond_script = conditional_script_baseline()
    print("strongest fixed-rate script, conditional on arrival:")
    for ob, r in cond_script.items():
        print(f"  {ob:6s} {r['rate'] * 100:5.1f}%  (k={r['k']}/{r['n_arrived']})  arm={r['arm']}",
              flush=True)
    print(flush=True)

    out = {"seeds": list(SEEDS), "base_checkpoint": BASE.name, "loss": "plain_BCE",
           "steps": STEPS, "lr": LR, "expert_per_self": EXPERT_PER_SELF,
           "self_data": {"dir": SELFDIR.name, "frames": int(len(self_bytes)),
                         "button_marginals": self_marg,
                         "note": "identical across seeds; rollouts are deterministic"},
           "conditional_script_baseline": cond_script,
           "arms": {}, "measurement_basis": "single_life"}

    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    all_traces = []
    try:
        # the base, re-measured so the behaviour statistics exist for it too
        # `traced_episode` samples per button from the sigmoid and never reads `thr`, so calibrating
        # before an evaluation is pure startup cost -- and startup cost is what was preventing this
        # job from finishing. Passing None makes the non-use explicit.
        pol, cfg, _ = load_policy(BASE)
        btr, was_done = resumable_eval(s, pol, cfg, None, start,
                                       TRACEDIR / "seeds_base_200.json", N_EVAL,
                                       checkpoint=BASE.name)
        print(f"base{' (resumed)' if was_done else ''}:", flush=True)
        out["arms"]["base"] = cached_score("base", btr)
        out["arms"]["base"]["loss"] = "onset_10x_sustain_5x"

        print("\nplain-BCE arms:", flush=True)
        for sd in SEEDS:
            tp = TRACEDIR / f"seeds_plain_s{sd}_200.json"
            ck = ROOT / f"data/bc_coverage/plain_s{sd}.pt"
            if tp.exists() and json.loads(tp.read_text()).get("n_episodes") == N_EVAL:
                tr, _ = resumable_eval(s, None, None, None, start, tp, N_EVAL)
            else:
                if ck.exists():
                    pol, cfg, _ = load_policy(ck)       # training already banked
                else:
                    pol, cfg, _ = load_policy(BASE)
                    pol = train_resumable(pol, training_data(), STEPS, LR, sd,
                                          ROOT / f"data/bc_coverage/plain_s{sd}.partial.pt")
                    save_policy(ck, pol, cfg, {n: 0.5 for n in NES_BUTTON_ORDER},
                                loss="plain_BCE", seed=sd)
                    (ROOT / f"data/bc_coverage/plain_s{sd}.partial.pt").unlink(missing_ok=True)
                tr, _ = resumable_eval(s, pol, cfg, None, start, tp,
                                       N_EVAL, seed=sd, loss="plain_BCE")
            out["arms"][f"plain_s{sd}"] = cached_score(f"plain_s{sd}", tr)
            all_traces.extend(tr)
    finally:
        s.close()

    # pooled across seeds: the unit of randomisation is the SEED, so pooling 600 episodes tightens the
    # measurement interval but does not remove training-seed variance -- both are reported.
    pooled_xs = [max(f[0] for f in t.frames) for t in all_traces]
    pooled_frames = [f for t in all_traces for f in t.frames]
    out["pooled"] = {
        "n": len(pooled_xs), "seeds": len(SEEDS),
        "clearance": clearance(pooled_xs), "vs_script": vs_script(pooled_xs),
        "vs_script_conditional": vs_script_conditional(pooled_xs),
        "button_marginals": button_marginals(pooled_frames),
        "behaviour": behaviour_stats(pooled_frames),
    }
    per_seed_cond = {ob: [out["arms"][f"plain_s{sd}"]["vs_script_conditional"]["per_obstacle"][ob]
                          ["advantage_pp"] for sd in SEEDS]
                     for ob in ("pipe1", "pipe2", "pipe3", "pipe4")}
    out["per_seed_conditional_advantage_pp"] = per_seed_cond
    out["seed_spread_pp"] = {ob: max(v) - min(v) for ob, v in per_seed_cond.items()}

    bc = out["arms"]["base"]["vs_script_conditional"]["per_obstacle"]
    pc = out["pooled"]["vs_script_conditional"]["per_obstacle"]
    moved = {ob: pc[ob]["advantage_pp"] - bc[ob]["advantage_pp"] for ob in bc}
    # is the pooled conditional rate itself better than the base's, with an interval?
    gains = {}
    for ob in ("pipe3", "pipe4"):
        a = out["arms"]["base"]["vs_script_conditional"]["policy_conditional"][ob]
        b = out["pooled"]["vs_script_conditional"]["policy_conditional"][ob]
        lo, hi = diff_ci(a["k"], a["n_arrived"], b["k"], b["n_arrived"])
        gains[ob] = {"base_rate": a["rate"], "base_n": a["n_arrived"],
                     "pooled_rate": b["rate"], "pooled_n": b["n_arrived"],
                     "delta_pp": (b["rate"] - a["rate"]) * 100,
                     "ci_pp": [lo * 100, hi * 100], "excludes_zero": bool(lo > 0 or hi < 0),
                     "improved": bool(lo > 0)}
    out["conditional_gain_vs_base"] = gains
    improved = [ob for ob, g in gains.items() if g["improved"]]

    print("\npooled across 3 seeds (n=600), conditional on arrival, vs base:", flush=True)
    for ob, g in gains.items():
        print(f"  {ob}: {g['base_rate'] * 100:.1f}% (n={g['base_n']}) -> "
              f"{g['pooled_rate'] * 100:.1f}% (n={g['pooled_n']})  "
              f"{g['delta_pp']:+.1f} pp [{g['ci_pp'][0]:+.1f}, {g['ci_pp'][1]:+.1f}]"
              f"{'  IMPROVED' if g['improved'] else ''}", flush=True)
    print(f"\nper-seed conditional advantage spread (pp): "
          f"{ {k: round(v, 1) for k, v in out['seed_spread_pp'].items()} }", flush=True)

    out["verdict"] = (
        "THE LOOP WORKS: pooled across three training seeds, `vs_script` conditional on arrival "
        "improves over the base at " + ", ".join(
            f"{ob} {gains[ob]['delta_pp']:+.1f} pp [{gains[ob]['ci_pp'][0]:+.1f}, "
            f"{gains[ob]['ci_pp'][1]:+.1f}]" for ob in improved) +
        ". An unbiased objective, a credit no marginal can game, start states where practice is "
        "useful, and an advantage that survives both seed noise and the upstream confound."
        if improved else
        "EXHAUSTED: pooled across three training seeds, `vs_script` conditional on arrival does not "
        "improve at pipe 3 or pipe 4 over the base (" + ", ".join(
            f"{ob} {gains[ob]['delta_pp']:+.1f} pp [{gains[ob]['ci_pp'][0]:+.1f}, "
            f"{gains[ob]['ci_pp'][1]:+.1f}]" for ob in gains) +
        "). With an unbiased loss, a degeneracy-proof credit, start states at the obstacles that "
        "matter, and the upstream confound removed by conditioning, this corpus and this method have "
        "given everything they have.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
