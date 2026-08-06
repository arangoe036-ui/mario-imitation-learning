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
