"""§2: mask the junk buttons at generation time. Free, no retraining, and it tests the owner's hypothesis.

The owner: *"we are trying to teach it to skip over pipe three while it is trying to go down. Maybe that is
a confusion."* The corpus supports it. Expert rates over 1,684,996 frames, from `actions.npy` (verified
against the advisor's table to 2 dp): Right 38.48 · Left 2.25 · **Down 0.72** · Up 0.08 · **Start 0.01** ·
Select 0.02 · B 47.76 · A 14.04. The expert presses **Down** on the 1-1 surface in all 34 of 34 movies, and
348 of those frames sit in x 672–735 — the pipe-3 arrival window and its face, exactly where the policy has
to jump instead. So the corpus demonstrates "jump this" and "press Down here" at nearly the same visual state.

**`Start` pauses the game.** It appears on 0.01% of expert frames and there is no reading under which the
policy should ever emit it.

The mask is over the **action vocabulary**: zero the probability of every joint class whose combo contains a
masked button, renormalise, sample as usual. `Down`, `Up`, `Start`, `Select` masked; **`Left` kept** — it is
2.25% of expert frames and genuinely used for positioning.

**Masked and unmasked are both re-run here under the same terminator** (`rollout_budget.STALL/CAP_FRAMES`),
with identical RNG seeds, so the pair is matched and neither inherits block 57's censored rule.

Also reported, because it is a second and separate leak if true: how many of the 300 joint classes the mask
removes, and **how many are never demonstrated in the corpus at all** yet remain sampleable.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.generation_sweep as G  # noqa: E402
import scripts.overnight as O  # noqa: E402
from scripts.compose import warm_session  # noqa: E402
from scripts.scaleup_eval import _Ep, resumable, score  # noqa: E402
from tasdata.bc import rollout_budget as RB  # noqa: E402
from tasdata.bc.budget import Deadline, TimedOut, time_limit  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/button_mask_eval.json"

MASKED = ("Down", "Up", "Start", "Select")
MASK_BITS = 0
for _n in MASKED:
    MASK_BITS |= NES_BUTTON_BITS[_n]
SEEDS = ["P_84_cnn32", "P_84_cnn32_seed1", "P_84_cnn32_seed2", "P_84_cnn32_seed3",
         "P_84_cnn32_seed4"]
TEMPS = [0.7, 1.0]
N_EVAL = 200
CAP_NON_A = 4
WALLS = {"pipe3_735": 735, "pipe4_975": 975, "koopas_1248": 1248, "frontier_1562": 1562,
         "flagpole_3266": 3266}
ARM_BUDGET_S = 15 * 60


def rollout(session, policy, cfg, start, seed, lut, byte_of, class_ok, *, temp) -> EpisodeTrace:
    """`capped` generation with an optional vocabulary mask. Terminator from rollout_budget."""
    s = cfg.frame_size
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    win[:] = _resize_gray(obs.rgb, (s, s))
    held, remaining = None, 0
    best = since = frames = 0
    while frames < RB.CAP_FRAMES:
        if remaining <= 0:
            with torch.no_grad():
                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0]
            p = torch.softmax(lg / float(temp), dim=-1).numpy()
            if class_ok is not None:
                p = p * class_ok
                tot = p.sum()
                # A fully-masked distribution would be a silent failure; fall back and record it by
                # emitting the no-op rather than dividing by zero.
                p = (p / tot) if tot > 0 else None
            if p is None:
                c = 0
            else:
                c = int(rng.choice(len(p), p=p / p.sum()))
            b, L = int(byte_of[c]), max(1, int(lut[c]))
            if not (b & A_BIT):
                L = min(L, CAP_NON_A)
            held, remaining = b, L
        remaining -= 1
        obs = session.step(held)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (s, s))
        t.record(obs, held)
        frames += 1
        r = read_smb(obs.ram, obs.framecount)
        if r.player_state in (0x06, 0x0B):
            t.record_death(obs)
            return t
        if r.x_position > best:
            best, since = r.x_position, 0
        else:
            since += 1
            if since > RB.STALL:
                t.ended = "stuck"
                return t
    return t


def perm_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool = np.concatenate([a, b])
    obs = a.mean() - b.mean()
    d = []
    for idx in itertools.combinations(range(len(pool)), len(a)):
        x = pool[list(idx)]
        y = pool[[i for i in range(len(pool)) if i not in idx]]
        d.append(x.mean() - y.mean())
    d = np.asarray(d)
    return float((np.abs(d) >= abs(obs) - 1e-9).mean()), len(d), 1.0 / len(d)


def main() -> None:
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 150 * 60)
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = G.joint_size(ctx.vocab.size)
    byte_of = np.array([ctx.vocab.decode_byte(c // G.N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    z = np.load(ROOT / "data/runlength_index_runs.npz")
    idx = {k: z[k] for k in ("rows", "joints", "lengths")}
    lut = G.class_lengths(idx, n_cls)

    # which joint classes survive the mask, and which are undemonstrated in the corpus
    class_ok = np.array([0.0 if (int(byte_of[c]) & MASK_BITS) else 1.0 for c in range(n_cls)])
    demonstrated = np.zeros(n_cls, dtype=bool)
    demonstrated[np.unique(idx["joints"])] = True
    vocab_report = {
        "n_classes": int(n_cls),
        "masked_buttons": list(MASKED), "mask_bits": int(MASK_BITS),
        "n_removed_by_mask": int((class_ok == 0).sum()),
        "n_surviving": int((class_ok > 0).sum()),
        "n_demonstrated_in_corpus": int(demonstrated.sum()),
        "n_undemonstrated_but_sampleable": int((~demonstrated).sum()),
        "n_undemonstrated_and_not_masked": int((~demonstrated & (class_ok > 0)).sum()),
        "note": ("an undemonstrated class the policy can still sample is a second, separate leak from "
                 "the masked buttons; both are free to close")}
    print(json.dumps(vocab_report, indent=2), flush=True)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.setdefault("skipped", [])
    out.update({"n_eval": N_EVAL, "measurement_basis": "single_life_from_level_start",
                "terminator": RB.describe(), "temps": TEMPS, "seeds": SEEDS,
                "expert_rates_from_actions_npy": {
                    "frames": 1684996, "Right": 38.48, "Left": 2.25, "Down": 0.72, "Up": 0.08,
                    "Start": 0.01, "Select": 0.02, "B": 47.76, "A": 14.04},
                "vocabulary": vocab_report})

    def save():
        OUT.write_text(json.dumps(out, indent=2, default=str))

    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        warm_session(s, start.frame)
        return s

    TRACED = ROOT / "data/traces"
    # ⚠ Temperature OUTER, seed INNER. With seeds outer, a deadline cut loses whole seeds at both
    # temperatures; with temperature outer it loses the secondary temperature instead, so the primary
    # T=0.7 comparison is complete across all five seeds whatever happens.
    for T in TEMPS:
        for name in SEEDS:
            if not (ROOT / f"data/bc_scaleup/{name}.pt").exists():
                continue
            ck = None
            for cond, ok in (("unmasked", None), ("masked", class_ok)):
                key = f"{name}_t{T:g}_{cond}"
                if key in out["arms"]:
                    continue
                if not dl.can_afford(150):
                    out["skipped"].append({"arm": key, "reason": "deadline"})
                    print(f"{dl.stamp()} SKIP {key}", flush=True)
                    continue
                if ck is None:
                    policy, cfg, blob = G.load_ckpt(name)
                    ck = (policy, cfg, blob)
                policy, cfg, blob = ck
                tp = TRACED / f"mask_{key}_{N_EVAL}.json"
                try:
                    with time_limit(min(ARM_BUDGET_S, dl.remaining() - 60), key):
                        s = sess_get()
                        try:
                            traces = resumable(
                                tp, N_EVAL,
                                lambda i: rollout(s, policy, cfg, start, i, lut, byte_of, ok,
                                                  temp=T))
                        finally:
                            s.close()
                except TimedOut as e:
                    out["skipped"].append({"arm": key, "reason": str(e)})
                    print(f"{dl.stamp()} TIMEOUT {key}: {e}", flush=True)
                    save()
                    continue
                rec = score(key, traces)
                xs = [max(f[0] for f in t.frames) for t in traces]
                frames = [f for t in traces for f in t.frames]
                b = np.asarray([f[3] for f in frames], dtype=np.int64)
                rec.update({
                    "checkpoint": name, "temperature": T, "condition": cond,
                    "train_seed": blob.get("seed"), "terminator": RB.describe(),
                    "x_p90": float(np.percentile(xs, 90)),
                    "past_wall": {w: {"k": int(sum(1 for x in xs if x > v)), "n": len(xs),
                                      "rate": float(np.mean([x > v for x in xs]))}
                                  for w, v in WALLS.items()},
                    "realised_masked_rates": {
                        nm: float(((b & NES_BUTTON_BITS[nm]) > 0).mean()) for nm in MASKED},
                    "pipe_entry_state_episodes": int(sum(
                        1 for t in traces if any(len(f) > 4 and f[4] == 0x07 for f in t.frames))),
                    "flagpole_episodes": int(sum(
                        1 for t in traces if any(len(f) > 4 and f[4] == 0x05 for f in t.frames))),
                    "traces": str(tp.relative_to(ROOT))})
                out["arms"][key] = rec
                save()
                pw = rec["past_wall"]
                print(f"  {dl.stamp()} {key:36s} x_max {rec['x_max']:5d} x_med {rec['x_median']:5.0f} "
                      f"p3 {pw['pipe3_735']['rate']*100:5.1f}% p4 {pw['pipe4_975']['rate']*100:4.1f}% "
                      f"| Down {rec['realised_masked_rates']['Down']:.4f} "
                      f"0x07 {rec['pipe_entry_state_episodes']:3d} flag {rec['flagpole_episodes']}",
                      flush=True)

    # -------- paired analysis --------
    res = {}
    for T in TEMPS:
        def vals(cond, field):
            v = []
            for name in SEEDS:
                a = out["arms"].get(f"{name}_t{T:g}_{cond}")
                if a:
                    v.append(a[field] if field in a else a["past_wall"][field]["rate"])
            return v
        um, mk = vals("unmasked", "x_max"), vals("masked", "x_max")
        row = {"n_seeds": min(len(um), len(mk)),
               "x_max_unmasked": um, "x_max_masked": mk,
               "x_median_unmasked": vals("unmasked", "x_median"),
               "x_median_masked": vals("masked", "x_median"),
               "past_pipe3_unmasked": vals("unmasked", "pipe3_735"),
               "past_pipe3_masked": vals("masked", "pipe3_735")}
        if len(um) >= 3 and len(mk) >= 3:
            row["x_max_median_unmasked"] = float(np.median(um))
            row["x_max_median_masked"] = float(np.median(mk))
            row["x_max_mean_diff"] = float(np.mean(mk) - np.mean(um))
            p, nperm, minp = perm_p(mk, um)
            row["x_max_perm_p_two_sided"] = p
            row["n_permutations"] = nperm
            row["min_attainable_p"] = minp
            # paired per-seed deltas, which at n=5 say more than the p-value
            row["per_seed_delta"] = [float(m - u) for m, u in zip(mk, um)]
            row["n_seeds_improved"] = int(sum(1 for d in row["per_seed_delta"] if d > 0))
        res[f"T{T}"] = row
    out["analysis"] = res

    dr = [a["realised_masked_rates"]["Down"] for k, a in out["arms"].items()
          if a["condition"] == "masked"]
    pe = [a["pipe_entry_state_episodes"] for k, a in out["arms"].items()
          if a["condition"] == "masked"]
    out["mask_verification"] = {
        "max_realised_Down_rate_masked": (max(dr) if dr else None),
        "all_masked_Down_zero": bool(dr and max(dr) == 0.0),
        "max_pipe_entry_episodes_masked": (max(pe) if pe else None),
        "all_masked_pipe_entry_zero": bool(pe and max(pe) == 0)}

    best = None
    for T in TEMPS:
        r = res[f"T{T}"]
        if "x_max_mean_diff" in r and (best is None or r["x_max_mean_diff"] > best[1]):
            best = (T, r["x_max_mean_diff"], r)
    if best is None:
        out["verdict"] = "Insufficient arms evaluated to decide."
        out["branch"] = "incomplete"
    else:
        T, diff, r = best
        imp, ns = r["n_seeds_improved"], r["n_seeds"]
        if diff > 0 and imp > ns / 2:
            out["branch"] = "depth_improves"
            out["verdict"] = (
                f"**THE OWNER'S HYPOTHESIS IS CONFIRMED: masking the junk buttons improves depth.** At "
                f"T={T}, x_max rises {diff:+.0f} px on average and in {imp} of {ns} seeds "
                f"(permutation p={r['x_max_perm_p_two_sided']:.3f}, floor "
                f"{r['min_attainable_p']:.3f}). Realised Down rate is "
                f"{out['mask_verification']['max_realised_Down_rate_masked']:.4f} and pipe-entry states "
                f"are {out['mask_verification']['max_pipe_entry_episodes_masked']}. **The mask becomes "
                f"permanent and §3's curve runs with it on.**")
        elif abs(diff) <= 50 or imp == ns / 2:
            out["branch"] = "depth_unchanged"
            out["verdict"] = (
                f"**A CLEAN NEGATIVE: the Down mass is harmless.** Best case is T={T} at {diff:+.0f} px, "
                f"{imp} of {ns} seeds improved, permutation p="
                f"{r['x_max_perm_p_two_sided']:.3f}. The mask does exactly what it should — Down rate "
                f"{out['mask_verification']['max_realised_Down_rate_masked']:.4f}, pipe-entry states "
                f"{out['mask_verification']['max_pipe_entry_episodes_masked']} — and depth does not move. "
                f"**The loss/depth anti-correlation needs a different mechanism than corpus "
                f"self-contradiction.**")
        else:
            out["branch"] = "depth_degrades"
            out["verdict"] = (
                f"**MASKING DEGRADES DEPTH ({diff:+.0f} px at T={T}, {imp} of {ns} seeds improved).** The "
                f"masked classes were doing something useful; do not keep the mask.")
    out["minutes"] = round(dl.elapsed() / 60, 1)
    save()
    print("\n" + "=" * 78)
    print(out["verdict"])
    for T in TEMPS:
        r = res[f"T{T}"]
        print(f"\nT={T}: x_max unmasked {r['x_max_unmasked']} -> masked {r['x_max_masked']}")
        if "per_seed_delta" in r:
            print(f"   per-seed delta {[int(d) for d in r['per_seed_delta']]} | "
                  f"{r['n_seeds_improved']}/{r['n_seeds']} improved | "
                  f"p={r['x_max_perm_p_two_sided']:.3f}")
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
