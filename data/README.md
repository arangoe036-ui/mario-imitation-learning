# The evidence base

Every number in this repository traces to a file in here. Nothing was computed in a notebook,
by hand, or in a terminal that wasn't saved.

## The convention

**One experiment, one script, one artifact.** `scripts/<name>.py` writes `data/<name>.json`.
83 of the 107 experiment scripts pair with a same-named artifact; the rest either write several
files or extend an earlier one. 145 result files in total.

This is why the directory looks the way it does. It is not a dump — it is 107 experiments, each
of which can be re-run on its own.

## Where the published claims come from

**The one positive — learning beats a blind baseline at the Koopas**
`runlength_script_control.json` · `rate_matched_control.json` · `plain_three_seeds.json` ·
`three_way_script_table.json` · `seed_variance_5.json` · `seed_variance_permutation.json`

The control is a run-length script matched on the policy's *own* token marginals — same action
representation, no vision. Ten paired seeds; the permutation test is exact.

**The negative — a perfect teacher leaves no recoveries to learn from**
`corpus_composition.json` · `stats_synced.json` · `split.json` · `depth_vs_steps.json` ·
`lift_by_training_length.json` · `peak_rebaseline.json` · `fidelity_vs_performance.json`

The corpus is 1,223,797 frames with zero deaths and zero recoveries. Live play peaks at
0.82 epochs while cross-entropy keeps falling, and imitation fidelity is uncorrelated with task
performance (r = −0.04).

**The reach ceiling — failures are positional, not gradual**
`reach_table.json` · `reach_walls.json` · `recheck_720.json` · `reach_curve_P_84_cnn32_seed1.json` ·
`reach_curve_P_84_cnn32_seed4.json` · `terrain_profile.json`

720 episodes from 72 saved start positions across two networks. The policy stops at the same
absolute x no matter where it begins.

**The trivial baselines that came first**
`p1_script_control.json` · `p1_control_ladder.json` · `script_baseline.json` · `single_life.json`

Right and B held with A flipped as a coin. It matches the policy through pipe 2, and it is the
reason the level-completion clip is captioned as proving nothing.

**Obstacle-level diagnosis**
`goomba_forensics.json` · `goomba_pa_probe.json` · `phase2_goomba.json` · `phase2_goomba_v2.json` ·
`pipe2_sweep.json` · `pipe2_pa_probe.json` · `pipe3_requirement.json` · `pipe3_reconcile.json` ·
`pipe4_build.json` · `pipe4_transfer_audit.json` · `partial_right.json` · `partial_right_dose.json`

**Interventions that did not work** — kept because a null is a result
`window_reweight_sweep.json` · `level_restricted_*.json` · `label_smoothing_*.json` ·
`nonlinear_head_eval.json` · `probe_ontop_power.json` · `dagger_*.json` · `objective_train.json` ·
`generation_sweep.json` · `temperature_ladder.json` · `vision_2x2.json`

**Audits that caught measurement defects** — see the README's defect table
`stall_rule_audit.json` · `hidden_area_check.json` · `graphics_integrity.json` ·
`route_audit.json` · `arrival_state_audit.json` · `category_audit.json` · `idle_audit.json` ·
`loss_provenance.json` · `mps_boundary.json` · `verify_capture128.json`

Each of these exists because a number disagreed with an independent measurement of the same
quantity and the disagreement was investigated.

**Withdrawn claims** — `claim_rewrite.md` records what was retracted and what replaced it.

## Subdirectories

- **`movies/`** — the TAS input files, with provenance in `movies/README.md`. These are the work
  of their authors, not of this project.
- **`plots/`** — published figures.

## Not in the repository

- **`traces/`** — 552 MB of raw per-episode dumps. Regenerable, and no published number cites
  one. Removed from history to keep the clone at ~30 MB.
- **Captured run data** (~11 GB) — rebuild with `tasdata batch --plan data/shortlist.json`.
- **Training checkpoints** — intermediate `.pt` files.
- **The ROM** — bring your own; see the main README.

⚠ `shortlist.json` contains stale absolute paths from a previous directory layout.
`shortlist_repaired7.json` is the corrected copy. Check that every `selected[].path` exists
before trusting a batch run — a missing movie prints one `FAIL` line and still reports overall
success.
