# Overnight run

Started 2026-08-01T05:00:20+00:00, running 175 min. 6 tasks finished, 1 failed.

Regenerated every 2 minutes. Raw stream: `data/overnight.jsonl`.

## Tier 1 — the 0.0% A-onset recall

**WIRING BUG, not calibration drift: double normalization collapsed p(A) to a constant (std 7.10e-06 vs 0.147 when fed correctly). Recalibration is still adopted after every round as good practice, but it was not the cause.**

| checkpoint | recall, stored threshold | recall, recalibrated | p(A) at onsets (median) |
| --- | --- | --- | --- |
| stage2_armB | 56.7% | 50.0% | 0.330 |
| stage3_round1 | 1.6% | 48.9% | 0.356 |

## Tier 2 — arm A rounds

| tag | ratio | accept % | A-onset recall | pipe1 1-1 (95% CI) | x_med 1-1 | x_med 2-1 |
| --- | --- | --- | --- | --- | --- | --- |
| round1_contaminated_for_reference | - | 26 | 48.9% | 53.3% [40.9, 65.4] | 594 | 530 |
| stage2_armB_baseline | - | 0 | 50.0% | 59.5% [52.6, 66.1] | 594 | 531 |
| round2_ratio3to1 | 3:1 | 25 | 47.3% | 92.0% [87.4, 95.0] | 595 | 530 |
| round3_ratio3to1 | 3:1 | 25 | 44.1% | 96.5% [93.0, 98.3] | 595 | 530 |
| round2_ratio1to1 | 1:1 | 25 | 38.8% | 96.5% [93.0, 98.3] | 595 | 530 |
| round3_ratio1to1 | 1:1 | 25 | 39.6% | 99.0% [96.4, 99.7] | 595 | 530 |

## Tier 4 — glitchless vs glitch-heavy

> The corpus contains NO warpless-glitchless runs -- the only glitchless runs are 'warps-glitchless', and one of the two is in the val split and cannot be trained on. The comparison below is therefore glitchless-WARPS against a matched-frame subsample of glitch-heavy WARPS runs, which controls the route but rests the glitchless arm on 1 run(s). Run-level variance is not controlled.

| arm | seed | A-onset recall | pipe1 1-1 | x_med 2-1 |
| --- | --- | --- | --- | --- |
| glitchless | 0 | 18.7% | 74.0% | 456 |
| glitch_heavy | 0 | 32.1% | 90.5% | 531 |
| glitchless | 1 | 36.9% | 53.0% | 371 |
| glitch_heavy | 1 | 34.5% | 72.0% | 530 |
| glitchless | 2 | 29.1% | 62.0% | 371 |
| glitch_heavy | 2 | 29.9% | 96.0% | 531 |

## Tier 5 — data scaling

| fraction | frames | A-onset recall | pipe1 1-1 |
| --- | --- | --- | --- |
| 10% | 98,138 | 35.0% | 60.8% |
| 25% | 245,346 | 44.7% | 75.0% |
| 50% | 490,692 | 44.4% | 65.0% |
| 100% | 981,385 | 40.4% | 56.7% |

## Tier 6 — the 2-1 wall

Running right and holding reaches x=306; running and jumping reaches x=531. Died running right: False. **Interpretation: blocks (no deaths running right)**

## Failures

- **tier3_oracle_margin** after 221.3s — `TypeError: only length-1 arrays can be converted to Python scalars`

## Task log

| task | status | minutes |
| --- | --- | --- |
| tier1_calibration_drift | ok | 1.6 |
| tier2_arm_a_rounds | ok | 62.2 |
| tier3_oracle_margin | FAILED | 3.7 |
| tier4_glitchless_vs_glitchy | ok | 70.6 |
| tier5_scaling_curve | ok | 36.2 |
| tier6_two_one_wall | ok | 0.6 |
| tier7_plots | ok | 0.0 |
