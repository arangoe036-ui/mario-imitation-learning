# The claim, as it should be written

Approved wording for the write-up, block 63. **The two prior forms are void and must not appear anywhere:**
"+80 pp at pipe 2" (that was the action encoding, not learning) and "+6.3 pp at pipe 2" (three training seeds,
sitting at their own design floor).

---

## The one-sentence form

**Run-length action tokens buy the early obstacles; learning buys the late ones — the policy beats a
representation-matched script at the Koopa Troopas by 5.5 percentage points, in 10 of 10 training seeds
(paired sign-flip permutation p = 0.0020, the smallest value the design can return).**

## The paragraph form

A behaviourally-cloned policy over run-length action tokens was compared against a scripted control matched
**on the action representation** — button-combination and hold-duration tokens drawn from the policy's own
token marginals, with rightward movement and run held on. This control is the honest one: an earlier control
that sampled buttons independently per frame could not physically hold the jump button for the twelve
consecutive frames the second pipe requires (probability 0.338¹² ≈ 2 × 10⁻⁶), so its score was arithmetic
rather than a measurement of skill.

Against the matched control, across ten paired training seeds at 200 episodes each, the policy shows **no
establishable advantage at the early, static obstacles** — pipe 2 +2.3 pp (p = 0.164), pipe 3 +2.5 pp
(p = 0.180) — and a **clear advantage at the late, dynamic ones**: **the Koopa Troopas +5.5 pp, 10/10 seeds,
p = 0.0020**, and the frontier fall +1.8 pp, 9/10 seeds, p = 0.0039. Both survive Bonferroni correction across
all six measured obstacles (0.012 and 0.023).

## The mechanism — stated in advance of the data, not fitted to it

The early obstacles are **static geometry**: clearing a pipe requires a jump of a particular duration, and a
fixed distribution over hold-lengths can produce that as often as a policy can. The Koopa Troopas are **moving
enemies**: the correct moment to jump depends on where they are, which is on the screen and not in any fixed
token distribution. **Screen-conditioning can only pay where the required action depends on something that
varies, and that is exactly the division the data shows.**

## The numbers table

| obstacle | kind | policy | matched script | difference | seeds ahead | p | ×6 Bonferroni |
|---|---|---|---|---|---|---|---|
| Goomba 320 | static | 66.0% | 63.7% | +2.3 pp | 7/10 | 0.164 | ns |
| pipe 1 470 | static | 66.0% | 63.7% | +2.3 pp | 7/10 | 0.164 | ns |
| pipe 2 630 | static | 66.0% | 63.7% | +2.3 pp | 7/10 | 0.164 | ns |
| pipe 3 735 | static | 39.3% | 36.8% | +2.5 pp | 8/10 | 0.180 | ns |
| pipe 4 975 | static | 23.6% | 24.6% | −1.0 pp | 4/10 | 0.385 | ns |
| **Koopas 1248** | **moving** | **20.4%** | **14.8%** | **+5.5 pp** | **10/10** | **0.0020** | **0.012 ✓** |
| **frontier 1562** | **fall** | **3.9%** | **2.0%** | **+1.8 pp** | 9/10 | **0.0039** | **0.023 ✓** |

Measurement basis: single life from the 1-1 level start (x=40), terminator `STALL=6500 / CAP=12000`, n=200 per
seed, 10 paired seeds, exact paired sign-flip permutation over seeds (floor 2/2¹⁰ = 0.00195).

## What must be said alongside it

1. **A script that jumps far more often than the policy still gets further through the early level.** Past
   pipe 3 the A-0.85 script reaches 57.5% against the policy's 39.3%. That script is not rate-matched and is
   the best-performing arm of a per-obstacle envelope, so it is a hard bar rather than a typical opponent —
   but it is a real one and it is not closed.
2. **Raising the policy's own jump rate to match does not recover it** (best dose +1.8 pp past pipe 3 across a
   0.49→0.91 sweep), so the difference is not the jump rate in either direction.
3. **The search-and-distil line has not produced an improvement.** Search solves every failure state it is
   given (60/60, 14,675 corrections), and two rounds of distillation produced an unconditional habit and then
   nothing. Reported as a negative with its diagnosis, not omitted.
4. **1-1 has been completed from the level start**, 4 of 200 episodes, stage-advance verified — but a
   coin-flipping script also completes it (1 of 200, Fisher p = 0.372), so completion is a project milestone
   and **not** evidence of skill.

---

# Update, block 64 — the nonlinear-head result, and a correction to its motivation

## The correction comes first, because it changes how the negative reads

Block 63 reported the on-top-versus-at-face distinction as **"present in the trunk's features but not
linearly reachable"**, from a linear probe AUC of 0.651 (p = 0.17) against a small MLP's 0.743 (p = 0.010).

**That was a power artifact and it is withdrawn.** The probe used only 11 on-top states because it looked only
at the 60 states that had been searched. Re-run over all 200 recorded failures — **38 on-top against 162
at-face** — the numbers are:

| probe | AUC | permutation p |
|---|---|---|
| **linear** | **0.859** | **0.0000** |
| MLP (32 hidden) | 0.915 | 0.0000 |
| **difference** | **+0.056** | 95% CI **[+0.012, +0.117]** over states |

**The linear head reads the distinction well.** The MLP's advantage is real — the bootstrapped interval
excludes zero — but it is 0.056 of AUC, not the difference between "unreadable" and "readable".

**So the claim to retire is "the action head cannot read the one distinction that matters".** What is true is
that a nonlinear head decodes it slightly better.

## And the behavioural result: no effect

Ten paired seeds, 1,000 steps, n = 200 each, pre-specified primary outcome declared before running.

| | params | **on-top failures at pipe 4** (primary) | past pipe 3 | past pipe 2 |
|---|---|---|---|---|
| H0 `Linear(64,300)` | 325,964 | 8.5 per 200 | 39.3% | 66.0% |
| H1 `64→128→300` | 353,484 | 8.7 (**+0.2**, p = 0.918) | 38.5% (−0.8 pp) | 64.2% (−1.8 pp) |
| H2 `64→256→300` | 400,204 | 8.4 (−0.1, p = 1.000) | 38.8% (−0.5 pp) | 65.1% (−0.8 pp) |

**Nothing moves, on the pre-specified outcome or on any wall.** Completions 3 and 3 against the baseline's 4.

One nominal result recorded and **not** promoted: H1's on-top failures summed over *all* walls rose by 3.3 per
200 with 1 of 10 seeds lower (p = 0.031). That is one of several secondary comparisons, it does not survive
multiplicity, and its direction is opposite to the intervention's intent.

## What this adds to the claim

Nothing is removed from the positive result. **The Koopas finding stands unchanged: +5.5 pp over a
representation-matched script, 10 of 10 seeds, p = 0.0020, Bonferroni ×6 = 0.012.**

A fifth item joins the "must be said alongside it" list:

5. **The action head is not the bottleneck.** It decodes the on-top-versus-at-face distinction at AUC 0.859,
   and adding a nonlinear head — the cheapest capacity change available, +8% parameters, placed exactly where
   probing pointed — changes no measured outcome at ten paired seeds. **Together with the trunk probes (wall
   identity AUC 0.892–1.000, position R² 0.712) this means no future negative in this project can be blamed on
   the representation or on the read-out. Both have been measured and both are adequate.**

---

## Block 65 — the corpus is load-bearing, and the claim's boundary held under a new test

**The claim is unchanged:** run-length action tokens buy the early static obstacles; learning buys the late
moving ones — **Koopas +5.5 pp, 10/10 seeds, p=0.0020, Bonferroni ×6 = 0.012.**

Block 65 restricted the training corpus to the only level ever evaluated (1-1), on the hypothesis that the
model was seeing it less than once through. Two arms at ten paired seeds: **C1** drew every batch from 1-1
alone (2,323 samples, 27.6 epochs at 1,000 steps against the full corpus's 0.82); **C2** drew half of each
batch from 1-1.

**Both arms were worse, and worse in a specific place.** Early walls flat or slightly up and null
(pipe 2: C1 +2.5 pp, 7/10 seeds, p=0.242). Late walls down in both arms with the same sign and the same seed
count (pipe 4: C1 −3.2 pp 1/10 up p=0.035; C2 −4.0 pp 1/10 up p=0.022). Nothing survives Bonferroni over the
14 wall tests and nothing was pre-specified; the replication across two independent arms is what is claimed,
not any p-value. A steps ladder confirmed it: C1's pipe-4 clearance falls monotonically 17→18→11→5→**0%** as
its loss falls 3.933→**0.700**, a loss the full corpus never reaches in 60,000 steps while still clearing
pipe 4 in 25% of episodes. **Matched on loss rather than on steps, the restricted arms are worse at pipe 4 at
all ten matched rungs.**

**Sixth item on the "must be said alongside it" list:**

6. **The 91% of the corpus that is never evaluated is load-bearing.** Removing it degrades exactly the late,
   moving obstacles that the claim attributes to learning, and leaves the early, static ones — the ones the
   claim attributes to the token representation — untouched. The off-task levels act as regularisation.
   **This is the first time a negative in this project landed on the side of a line drawn in advance**, which
   is weak corroboration of the claim's boundary rather than of the claim's size.

**Also settled, and it must be said whenever "train it more" comes up:** the peak did **not** move. C1's best
rung is 1,000 steps, identical to the full corpus. What moved is the collapse, ~9× earlier in steps.

---

## Block 66 — the objective changed nothing, and one block-65 sentence is retracted

### RETRACTION, carried forward

**"The off-task levels were acting as regularisation" is WITHDRAWN.** A 34× smaller training set reaches 27.6
epochs in 1,000 steps, so "restricting the corpus" and "training for 27.6 epochs" are the same manipulation in
block 65 and cannot be separated by it. **Matched on epochs instead of steps the comparison inverts.**

**What block 65 establishes, and all it establishes:** at the operating point anyone would use — the peak —
restricting the corpus to 1-1 is worse at the late walls. **State the practical conclusion; do not state the
mechanism.** The sixth "must be said alongside it" item is corrected to read that way.

### The claim is unchanged

**Run-length action tokens buy the early static obstacles; learning buys the late moving ones — Koopas
+5.5 pp, 10/10 seeds, p=0.0020, Bonferroni ×6 = 0.012.**

### Seventh item on the "must be said alongside it" list

7. **The objective is not the constraint either, and the over-commitment story is retired.** Label smoothing
   at ε = 0.05 / 0.10 / 0.20 was worse than plain cross-entropy at **all six walls at all three strengths —
   18 of 18 cells negative in sign, none positive** — with the peak still at 1,000 steps and the collapse no
   slower (8 of 10 NLL-matched ladder rungs at or below the unsmoothed curve). Obstacle-window reweighting was
   null at 1.5×/2.0×/3.0× and **significantly harmful at 8.0× (−4.0 pp past pipe 4, 0/10 seeds up, p at the
   0.00195 floor, surviving Bonferroni)**. **Three failed predictions for over-commitment: augmented data,
   restricted corpus, and now the objective.**

### And a correction to the multiplicity discipline itself

**The Goomba (320), pipe 1 (470) and pipe 2 (630) are ONE measurement, not three.** Zero of 2,000 baseline
failures land between x=320 and x=630; across the 90 arms of block 66, **89 have byte-identical Goomba and
pipe-2 counts** and the one exception differs by a single episode. **There are four independent failure
regions, not seven.** Every Bonferroni family this project has quoted has been too large and therefore too
strict — the safe direction, but the corrected family should be used from here. **The live claim's
"Bonferroni ×6 = 0.012" is conservative and stands.**
