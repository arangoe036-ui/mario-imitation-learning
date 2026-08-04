"""Does the composed recipe's loss inflate button marginals above its training data?

Found while reporting §3. The script-net round trained on self-data with A on **0.871** of frames and
produced a policy pressing A on **0.970**, Down on 0.756 against the data's 0.314, and Left on 0.398
against 0.122. **Every marginal came out above the data's.** Plain supervised learning on i.i.d. targets
should reproduce the base rate, so the objective is suspect rather than the data.

The suspect is `scripts/compose.py::sustain_loss`, used by every "composed recipe" run:

    w = 1 + (ONSET_W - 1)*onset  +  (SUSTAIN_W - 1)*is_pressed_and_not_onset      # 10x and 5x

Every term up-weights **pressed** frames. Nothing up-weights released frames. So the weighted objective's
optimum is not the conditional base rate -- it is pushed toward pressing, on **every** button at once.
For a Bernoulli head with weight `a` on positives and `1` on negatives, the weighted optimum is
``a*p / (a*p + (1-p))``, so a 5x sustain weight turns a true p=0.5 into 0.833.

This probe needs no emulator. Two policies are trained from the same seed on the same expert data, one
with `sustain_loss` and one with plain BCE, and the **mean predicted probability** per button is compared
against the data's own press rate. Mean predicted probability is the right quantity because live play
samples per button from the sigmoid and ignores the calibrated thresholds.

Prediction stated before running: plain BCE lands near the expert's rates (A 0.152); `sustain_loss` lands
far above, and the closed form above says roughly where.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import ONSET_W, SUSTAIN_W, sustain_loss  # noqa: E402
from tasdata.bc.overnight_lib import fresh_policy, random_rows  # noqa: E402
from tasdata.bc.train import make_loader  # noqa: E402
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/loss_bias_probe.json"
STEPS, LR, BATCH = 400, 3e-4, 128
PROBE_ROWS = 8000


def plain_loss(logits, bits, onset):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, bits.float())


def weighted_optimum(p: float, a: float) -> float:
    """Optimum of a BCE weighted `a` on positives, `1` on negatives, for true rate `p`."""
    return a * p / (a * p + (1.0 - p))


def train_with(ctx, ds, loss_fn, steps: int, seed: int = 0):
    policy = fresh_policy(ctx.cfg, seed=seed)
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    loader = make_loader(ds, batch_size=BATCH, shuffle=True, num_workers=0, seed=seed)
    step = 0
    while step < steps:
        for obs, _p, bits, onset in loader:
            loss = loss_fn(policy(obs), bits.float(), onset.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"      step {step}/{steps} loss {float(loss):.4f}", flush=True)
            if step >= steps:
                break
    policy.eval()
    return policy


def mean_probs(policy, ds, rows) -> dict:
    ps = []
    with torch.no_grad():
        for obs, _p, _b, _o in make_loader(Subset(ds, rows), batch_size=256, shuffle=False,
                                           num_workers=0):
            ps.append(torch.sigmoid(policy(obs)).numpy())
    arr = np.concatenate(ps)
    return {n: float(arr[:, i].mean()) for i, n in enumerate(NES_BUTTON_ORDER)}


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    ds = ctx.dataset(ctx.expert_train)
    rows = random_rows(ds, PROBE_ROWS, seed=1)

    # the data's own press rate, measured on the same rows the probe scores
    bits = []
    with torch.no_grad():
        for _o, _p, b, _on in make_loader(Subset(ds, rows), batch_size=256, shuffle=False,
                                          num_workers=0):
            bits.append(b.numpy())
    data_rate = {n: float(np.concatenate(bits)[:, i].mean())
                 for i, n in enumerate(NES_BUTTON_ORDER)}
    print(f"expert data press rate on the probe rows: "
          f"{ {k: round(v, 3) for k, v in data_rate.items() if v > 0.001} }\n", flush=True)

    out = {"steps": STEPS, "lr": LR, "probe_rows": PROBE_ROWS,
           "onset_weight": ONSET_W, "sustain_weight": SUSTAIN_W,
           "data_press_rate": data_rate, "arms": {},
           "closed_form": {n: {"true_p": data_rate[n],
                               "optimum_at_sustain_weight": weighted_optimum(data_rate[n],
                                                                            SUSTAIN_W)}
                           for n in NES_BUTTON_ORDER if data_rate[n] > 0.001}}

    for label, fn in (("plain_bce", plain_loss), ("sustain_onset", sustain_loss)):
        print(f"  [{label}] {STEPS} steps", flush=True)
        policy = train_with(ctx, ds, fn, STEPS)
        mp = mean_probs(policy, ds, rows)
        out["arms"][label] = {"mean_predicted_prob": mp,
                             "over_data_rate": {n: (mp[n] / data_rate[n])
                                                if data_rate[n] > 0.001 else None
                                                for n in NES_BUTTON_ORDER}}
        print(f"    mean p: " +
              "  ".join(f"{n} {mp[n]:.3f}" for n in ("A", "B", "Right", "Down", "Left")),
              flush=True)

    pb, so = out["arms"]["plain_bce"], out["arms"]["sustain_onset"]
    print(f"\n{'button':8s} {'data':>8s} {'plain BCE':>10s} {'sustain':>9s} "
          f"{'closed form':>12s}")
    inflated = []
    for n in ("A", "B", "Right", "Down", "Left"):
        cf = out["closed_form"].get(n, {}).get("optimum_at_sustain_weight")
        print(f"{n:8s} {data_rate[n]:8.3f} {pb['mean_predicted_prob'][n]:10.3f} "
              f"{so['mean_predicted_prob'][n]:9.3f} {cf if cf is None else f'{cf:12.3f}'}")
        if so["mean_predicted_prob"][n] > pb["mean_predicted_prob"][n] + 0.02:
            inflated.append(n)
    out["buttons_inflated_by_sustain_loss"] = inflated
    out["verdict"] = (
        f"THE LOSS INFLATES THE MARGINAL on {inflated}: `sustain_loss` up-weights pressed frames "
        f"({ONSET_W}x onsets, {SUSTAIN_W}x sustained) and never up-weights released frames, so its "
        f"optimum is above the base rate by construction. Every 'composed recipe' run used it, which "
        f"is a mechanism for the always-jump degeneracy that is independent of the data and of the "
        f"acceptance filter."
        if inflated else
        "The loss does not measurably inflate the marginal at this budget; the inflation seen in the "
        "script-net round must come from the data or the acceptance filter instead.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
