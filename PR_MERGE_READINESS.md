# PR merge-readiness assessment

**Branch:** `glw/hevi-imex-docs` → `main`
**Net diff vs main:** +39,478 / −2,047 across 188 files

## What's substantive (must keep)

### Source code

- `src/CompressibleEquations/acoustic_substepping.jl` — the substepper
  (rewritten over multiple sessions: from-scratch Baldauf-style core,
  Phase 4.5 Klemp 3-D damping fix, Phase 1+2A moisture infrastructure
  + γᵐRᵐ⁰ in PGF/Schur). All 71 dry tests pass.
- `src/CompressibleEquations/time_discretizations.jl` —
  `SplitExplicitTimeDiscretization`, `AcousticOuterScheme`/
  `WickerSkamarock3` abstraction, `stage_fractions`,
  `KlempDivergenceDamping`, `ProportionalSubsteps` /
  `MonolithicFirstStage` distributions.
- `src/CompressibleEquations/compressible_dynamics.jl` — substepper
  hookup, `convert_slow_tendencies!`, `prepare_acoustic_cache!`.
- `src/CompressibleEquations/CompressibleEquations.jl` — module exports
  including the new abstraction names.
- `src/Thermodynamics/reference_states.jl` — discrete-balance reference
  state work from Phase 0/1/2.
- `src/TimeSteppers/acoustic_runge_kutta_3.jl` — uses `stage_fractions`
  from the new abstraction.
- `src/TimeSteppers/acoustic_substep_helpers.jl` — small helper updates.
- `src/Breeze.jl` — re-exports `AcousticOuterScheme`, `WickerSkamarock3`,
  `KlempDivergenceDamping`, etc.
- `src/AtmosphereModels/dynamics_interface.jl` — small dispatch addition.
- `src/BoundaryConditions/BoundaryConditions.jl` — minor.

### Tests

- `test/acoustic_substepping.jl` — 71 tests covering:
  construction, `compute_acoustic_substeps`, model runs without NaN,
  IGW stability at WS-RK3, balanced state stays quiet (rest-atm
  bit-quiet), divergence damping, ExnerReferenceState, slow-tendency
  modes, no-reference path. All passing.

### Examples (substepped)

- `examples/baroclinic_wave.jl` — DRY BW at 2° / Δt=450s / 14 days,
  runs in ~2.2 min on one GPU. (Just reverted from the dry+moist
  merge attempt; moist deferred to follow-up.)
- `examples/inertia_gravity_wave.jl` — already uses substepping.

### Documentation

- `docs/src/compressible_dynamics.md` — theoretical chapter
  rewrite (WS-RK3 + Klemp damping, post-HEVI/IMEX scrub).
- `docs/src/breeze.bib` — Baldauf 2010, KlempSkamarockHa 2018,
  KnothWensch 2014 references.

## What's working/diagnostic and probably should NOT merge to main

### Validation directories (~200 ad-hoc scripts, ~36k lines)

- `validation/substepping/` — 60+ scripts numbered 01_*..59_* plus
  `out/` (818 MB of GIFs, reports, intermediate diagnostics).
  Genuinely useful during development; not appropriate for `main`.
- `test/substepper_validation/` — 30+ scripts (eigenvalue scans, sweep
  runners, proof tests, long-run BW tests). Some of these
  (e.g. `m1_moist_rest.jl`, `m2_moist_rest_qv_gradient.jl`) are
  M-tier acceptance tests that *should* be ported to proper
  `@test`-based regression tests.
- `validation/anelastic_compressible_comparison/` — paired anelastic-vs-
  compressible runs. Some are useful examples, others are throwaway.
- Loose `*.log` files and `*.jld2` data files (the latter mostly
  gitignored or excluded by my staging).

### Top-level diagnostic docs (could move or drop)

- `MOIST_SUBSTEPPER_STRATEGY.md` — the Phase 3 moist plan, partially
  superseded by `MOIST_BW_FUTURE_PLAN.md`.
- `PHASE_2A_RESULTS.md` — diagnostic study writeup. Keep as
  PR-description content or move under `docs/src/appendix/`.
- `SUBSTEPPER_INSTABILITY_SUMMARY.md`,
  `SUBSTEPPER_LONG_RUN_REPORT.md`,
  `split_explicit_substepping_foundations.md` — earlier-session
  diagnostic reports. Not needed in `main`.
- `validation/substepping/PRISTINE_SUBSTEPPER_PLAN.md` — long-running
  effort plan + standards. Useful as design doc; could move under
  `docs/src/design/` or drop.
- `_draft_first_pass.md` — drop.

## Cleanup options for the PR

### Option A — minimal cleanup (lowest risk)

Drop:
- `_draft_first_pass.md`
- the 4 PDFs and the `.zip` (already gitignored on main but committed
  in commit `9db0145`)
- `.claude/scheduled_tasks.lock`

Keep everything else, accept that `validation/` and
`test/substepper_validation/` are dev scratch space living in the
repo. Open a follow-up issue to harvest the M-tier tests into proper
regression tests and clean up the rest.

**Pros:** smallest behavioural change, ready to review now.
**Cons:** ~36k lines of throwaway content lands in `main`. Hard to
maintain over time; future contributors will be confused about which
scripts to trust.

### Option B — medium cleanup (recommended)

Drop:
- `validation/substepping/out/` (818 MB of run output)
- `validation/substepping/[0-5][0-9]*.jl` and similar — the bulk of
  the ad-hoc scripts
- `test/substepper_validation/long_runs/*.jl` ad-hoc scripts (keep
  only m1/m2/m3-tier as they're the future test surface)
- `.log` files
- Top-level WIP docs (`SUBSTEPPER_INSTABILITY_SUMMARY.md`,
  `_draft_first_pass.md`, `split_explicit_substepping_foundations.md`)

Keep:
- `validation/substepping/PRISTINE_SUBSTEPPER_PLAN.md` (move to
  `docs/src/appendix/substepper_program.md`)
- `MOIST_BW_FUTURE_PLAN.md` (top-level, references in PR description)
- `MOIST_SUBSTEPPER_STRATEGY.md` (drop, superseded by future plan)
- `PHASE_2A_RESULTS.md` (move to `docs/src/appendix/phase2a_results.md`
  or drop and put in PR description)
- `test/substepper_validation/long_runs/m1_moist_rest.jl`,
  `m2_moist_rest_qv_gradient.jl`, `m3_moist_acoustic_pulse.jl` and
  `M_TIER_BASELINES.md` — the future moist regression test surface

**Pros:** much cleaner main; the durable docs survive.
**Cons:** larger diff churn from the cleanup commits; need to verify
nothing breaks (no example or test references any of the dropped
scripts).

### Option C — minimal-and-focused (reset cleanup)

Squash the 4 new commits into 2 clean commits:

1. `Substepper Phase 1+2A: γᵐRᵐ⁰ in PGF + moisture snapshot
   infrastructure` (just the source code changes)
2. `Examples + docs: dry BW with substepping; moist plan deferred`

And rebase to drop the `8e3d427` validation-scripts commit entirely.
Force-push to the feature branch.

**Pros:** clean PR diff, easy to review, no dev artifacts in main.
**Cons:** loses the diagnostic-study scripts from the branch (they
were useful evidence; can be recovered from this branch in the
future via a `git tag` before force-push).

## Recommended path

**Option B**, with a `git tag` of `glw/hevi-imex-docs-pre-cleanup`
before the cleanup commits to preserve the diagnostic scripts for
reference.

## Pre-merge checklist

Regardless of which cleanup option:

- [ ] All 71 acoustic substepper tests pass: `Pkg.test("Breeze";
      test_args=`acoustic_substepping`)`
- [ ] `examples/baroclinic_wave.jl` runs end-to-end at 2° in <5 min
      on a GPU (verified: 2.2 min)
- [ ] `examples/inertia_gravity_wave.jl` runs cleanly
- [ ] `examples/moist_baroclinic_wave.jl` is removed cleanly (no
      remaining references in docs/tests; verified)
- [ ] CI green (need to push and watch)
- [ ] Doctests pass (haven't touched docstrings significantly; should
      be fine but worth a check)
- [ ] No PDF / ZIP / GIF binaries committed (gitignore added by
      origin commit `2bc4670`)
- [ ] No `.log` / `.jld2` files committed
- [ ] No `_draft_*.md` working docs committed
- [ ] PR description summarizes:
  - Dry substepper ready for production (CFL=0.7 verified, 14d clean)
  - Moist substepper deferred (10× CFL gap, structural fix needed)
  - `MOIST_BW_FUTURE_PLAN.md` describes the dry-density switch path

## Open items the user should decide

1. **Which cleanup option (A/B/C)?**
2. **Where to put the design docs** (top-level vs `docs/src/appendix/`
   vs drop)?
3. **Which validation scripts to harvest as proper tests?** Suggested
   minimum:
   - rest-atm bit-quiet (already in test/acoustic_substepping.jl as
     "Balanced state stays quiet")
   - 1-day dry BW smoke (new — verifies the substepper handles
     long-running production-Δt runs without NaN; could be GPU-only)
   - Phase 1 snapshot bit-identity (γᵐRᵐ⁰ collapses to γᵈRᵈ for dry)
4. **Branch retention**: tag `glw/hevi-imex-docs-pre-cleanup` before
   any history rewrite, so the diagnostic scripts remain accessible.
