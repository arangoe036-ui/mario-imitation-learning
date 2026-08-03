# Stage 2 — behavioural cloning baseline

## What ran

- torch 2.13.0 on `mps`
- 47 evaluation points across 5 configs
- 4 configs finished, 0 failed

## Smoke test (the gate)

PASSED. All four checks:

1. data: 1000 of 981,385 train frames, memory-mapped
2. loss decreased 2.3577 -> 1.7620 over 50 steps
3. checkpoint saved and reloaded, max|Δlogit| = 0.0
4. one live episode completed: 641 frames, level 1-1, x=435 (reached frame budget)

## Baselines

| baseline | val accuracy | val macro accuracy | predicts |
| --- | --- | --- | --- |
| always nothing | 36.48% | 6.25% | - |
| always Right+B | 46.59% | 6.25% | Right+B |
| sample marginal distribution | 31.51% | 5.93% | (stochastic) |
| always train-mode token (blind ceiling) | 36.48% | 6.25% | - |

### Baselines in live play

| baseline | progress (median) | levels (median) | deaths (median) | frames survived (median) |
| --- | --- | --- | --- | --- |
| baseline_always_nothing | 40.0 | 1.0 | 0.0 | 5041.0 |
| baseline_always_right_b | 314.0 | 1.0 | 5.0 | 5041.0 |
| baseline_marginal_sample | 594.0 | 1.0 | 0.0 | 5041.0 |

## Configs, at their best evaluation point

| config | params | blind | best step | val loss | val acc | macro acc | live progress (median) | live levels (median) | deaths (median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind_lr3e-4 | — | yes | 3000 | 1.2827 | 36.34% | 6.67% | 594.0 | 1.0 | 0.0 |
| small_lr1e-4 | — | no | 3000 | 0.7871 | 72.88% | 13.64% | 594.0 | 1.0 | 0.0 |
| small_lr3e-4 | — | no | 3000 | 0.7661 | 73.89% | 13.99% | 595.0 | 1.0 | 0.0 |
| tiny_lr1e-3 | — | no | 3000 | 0.8187 | 74.59% | 12.78% | 595.0 | 1.0 | 0.0 |
| tiny_lr3e-4 | — | no | 3000 | 0.8139 | 74.05% | 13.62% | 435.0 | 1.0 | 0.0 |

## Curves (val accuracy / live median progress by step)

- **blind_lr3e-4** accuracy — 3000:36.3% 6000:36.3% 9000:36.3% 12000:36.3% 15000:36.3% 18000:36.3% 21000:36.3% 24000:36.3% 27000:36.3% 30000:36.3%
  progress — 3000:594.0 6000:err 9000:0.0 12000:0.0 15000:0.0 18000:err 21000:err 24000:594.0 27000:594.0 30000:594.0
- **small_lr1e-4** accuracy — 3000:72.9% 6000:71.9% 9000:68.6% 12000:69.0% 15000:64.6% 18000:66.8% 21000:68.7% 24000:62.7% 27000:64.1% 30000:64.7%
  progress — 3000:594.0 6000:594.5 9000:594.0 12000:594.5 15000:594.0 18000:595.0 21000:595.0 24000:595.0 27000:595.0 30000:595.0
- **small_lr3e-4** accuracy — 3000:73.9% 6000:71.5% 9000:68.5% 12000:69.6% 15000:64.8% 18000:66.8% 21000:67.4% 24000:63.9% 27000:65.7% 30000:66.7%
  progress — 3000:595.0 6000:594.0 9000:595.0 12000:595.0 15000:594.0 18000:595.0 21000:594.0 24000:595.0 27000:595.0 30000:594.0
- **tiny_lr1e-3** accuracy — 3000:74.6% 6000:74.3% 9000:74.4% 12000:72.2% 15000:69.2% 18000:69.4% 21000:69.8% 24000:68.4% 27000:66.1% 30000:68.3%
  progress — 3000:595.0 6000:594.5 9000:594.0 12000:594.0 15000:515.5 18000:594.5 21000:594.0 24000:595.0 27000:595.0 30000:595.0
- **tiny_lr3e-4** accuracy — 3000:74.1% 6000:73.8% 9000:71.3% 12000:69.6% 15000:67.5% 18000:68.3% 21000:65.8%
  progress — 3000:435.0 6000:515.0 9000:595.0 12000:594.0 15000:594.5 18000:595.0 21000:595.0

## RARE token

- **blind_lr3e-4**: RARE predicted on 0 of 40,000 val frames (0.0000%); true RARE labels 4
- **small_lr1e-4**: RARE predicted on 0 of 40,000 val frames (0.0000%); true RARE labels 4
- **small_lr3e-4**: RARE predicted on 0 of 40,000 val frames (0.0000%); true RARE labels 4
- **tiny_lr1e-3**: RARE predicted on 0 of 40,000 val frames (0.0000%); true RARE labels 4
- **tiny_lr3e-4**: RARE predicted on 0 of 40,000 val frames (0.0000%); true RARE labels 4

## What broke

Nothing — every config completed.

---

_Sweep process 46889 exited at 2026-07-30 05:16:26; summary regenerated automatically._

---

## Stage 2, closed out (frozen)

The frozen table is regenerated from artifacts only, by
`scripts/print_stage2_table.py`, and kept at `data/stage2_frozen_table.txt`.
Live rows come from `data/stage2_final_session.jsonl`; offline rows from
`data/stage2c_results.jsonl` at step 3,000.

### Corrections made while closing out

Three defects in the evaluation *start points* were found and fixed. All three
inflated or distorted earlier live numbers, so the table above supersedes any
previously reported figure.

1. **Level starts were not level starts.** Making them the first frame passing the
   full grounded filter put the 1-1 start at x=2616, past both pipes, so
   "cleared pipe 1" was true before the episode began and every arm scored 100%.
2. **Seven W-1 starts were the previous world's castle walk.** The world counter
   increments during the end-of-W-4 walk, so the first frame *labelled* 2-1 was
   Mario at x=2430 finishing 1-4. Fixed by requiring x <= 100.
3. **The 1-1 start was before control handover.** Frame 42 has pregame=1,
   player_state=0x08 and x=40, but it is boot-time RAM transient with the title
   screen still up — 60 frames of Right+B from there leaves x at 0. Fixed by
   requiring the expert's x to actually *increase* within 10 frames. 1-1 now
   starts at frame 196.

**Consequence for the arm A/B comparison.** On the bad 1-1 start, arm B cleared
pipe 1 on 70% of episodes against arm A's 20%. On the corrected start at n=20 the
two read 45% and 40%, which was reported here as indistinguishable. That was an
underpowered reading, not a null result — 9 episodes against 8.

Rerun at **n=200 per arm** (`scripts/arm_ab_power.py`):

| start | arm A | arm B | difference (Newcombe 95%) |
| --- | --- | --- | --- |
| 1-1 pipe 1 | 29.5% [23.6, 36.2] | **59.5% [52.6, 66.1]** | **+30.0 pp [+20.4, +38.8]** |
| 2-1 x median | 530 | 531 | no difference |

The interval excludes zero. Arm B's offline advantage (A-onset recall 45.5% vs
31.5%) **does** transfer to live play; the n=20 comparison simply could not see a
30-point effect.

### Savestate library

532 start points (32 level starts + 500 filtered trajectory points), each carrying
a RAM hash and a **frame hash** — sha256 of the 84x84 observation the policy
actually receives. RAM equality does not cover PPU state and the model consumes
pixels, so both are asserted on rebuild: 532/532 identical across processes.

Two of 532 states share a frame hash with another while having distinct RAM: the
rendered pixels are identical, the game variables are not. That is expected and is
the reason both hashes are kept rather than either alone.
