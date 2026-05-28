# Running review: acoustic substepping no-fill PR

Last reviewed: 2026-05-21, initial baseline.

## Current status

- No implementation changes detected yet.
- Current untracked files are planning/review notes:
  - `acoustic_substepping_no_fill_plan.md`
  - `acoustic_substepping_no_fill_plan_review.md`

## Review focus

I will review incoming implementation diffs against the no-fill acoustic
substepping plan, with special attention to:

- No substep-loop `fill_halo_regions!` calls in the optimized path.
- A temporary A/B path or test toggle only for validation, not a public
  long-term filled-halo algorithm.
- Use of Oceananigans topology-aware operators where available.
- Correct handling of `Bounded`, `Periodic`, and connected local topologies.
- Correct active launch ranges by field location.
- Previous-`ρθ′` storage fused into `_build_predictors_and_vertical_rhs!`,
  not `_post_solve_recovery!`.
- No serial dependency on active halo columns.
- Distributed design preserving or extending model-owned stage-entry fields,
  not only substepper fields.
- Tridiagonal solve range remaining physical-domain only for serial, but
  explicitly addressed before distributed active halos.
- Tests for damping, sponge, mixed topology, and backward timestepping.

## Initial suggestions

- Keep the first implementation commit behavior-preserving if possible:
  introduce launch/operator helpers while leaving the filled-halo path active.
- Add small CPU correctness tests before larger GPU work. The fastest useful
  tests are one-stage or one-step comparisons between the old filled-halo path
  and the new no-fill path on small grids.

## Poll log

### 2026-05-21 poll 1

- No source or test implementation changes detected yet.
- No review findings.

### 2026-05-21 poll 2

- No source or test implementation changes detected yet.
- Suggested validation tests for the first implementation stage:
  - A tiny CPU comparison on `(Periodic, Periodic, Bounded)` with fixed
    `substeps = 6`, comparing filled-halo and no-fill modes after one RK
    acoustic stage.
  - The same comparison on `(Bounded, Periodic, Bounded)` to exercise
    topology-aware x operators at physical walls.
  - A `NoDivergenceDamping()` case and a default
    `ThermalDivergenceDamping()` case, because previous-`ρθ′` storage is only
    strongly exercised by damping.
  - A backward-step smoke test with `Δt < 0` and no sponge.
  - A test that no optimized-path `fill_halo_regions!` calls occur inside the
    acoustic substep loop. This can start as a structural test if runtime
    instrumentation is too invasive.

### 2026-05-21 poll 3

- No source or test implementation changes detected yet.
- No new review findings.

### 2026-05-21 poll 4

- No source, test, benchmark, or docs implementation changes detected yet.
- No new review findings.

### 2026-05-21 poll 6

New files detected:

- `src/CompressibleEquations/acoustic_kernel_parameters.jl`
- `src/CompressibleEquations/acoustic_operators.jl`

Findings:

1. `src/CompressibleEquations/acoustic_kernel_parameters.jl` is not included
   from `src/CompressibleEquations/CompressibleEquations.jl`, and neither is
   `src/CompressibleEquations/acoustic_operators.jl`. If these are intended
   to be active in this stage, add includes before `acoustic_substepping.jl`
   and make sure all imports stay explicit.

2. `acoustic_kernel_size(::Type{FullyConnected}, N, H)` and the left/right
   connected methods are wrong when `H == 0`. For example,
   `FullyConnected, H = 0` gives `2:N-1`, not `1:N`. Stage 0/serial helpers
   are supposed to be behavior-preserving with `active_halo_width = 0` on all
   local topologies, including distributed smoke tests. Add an `H == 0`
   guard/path so every topology returns `1:N` when the active halo width is
   zero.

3. The `acoustic_xface_parameters` and `acoustic_yface_parameters` helpers
   currently return the center-cell range in the normal direction. That is not
   sufficient for no-fill behavior at `Bounded` normal faces if the kernel is
   responsible for writing boundary faces. This is the same issue that came up
   in the `compute_velocities!` PR: face-field launch ranges need field
   location awareness, probably using `length(Face(), TX(), Nx)` or equivalent
   logic for the physical extent before adding connected active halos.

4. The docstring claim that topology-aware operators "handle the wall faces"
   in `acoustic_xface_parameters` / `acoustic_yface_parameters` is too strong.
   Operators can handle stencils at an edge that is launched, but they cannot
   write an unlaunched east/north boundary face. Either the boundary face is
   intentionally not part of the acoustic update and is set elsewhere, or the
   launch range has to include it.

5. In `acoustic_operators.jl`, the overloads with `f::Function` are narrower
   than the usual Oceananigans operator pattern. Consider dropping the
   `::Function` annotation so callable structs and other function-like kernel
   arguments are supported. This is probably not blocking for the current
   acoustic helpers, but keeping the operator wrapper generic is safer.

6. Please add small helper tests before wiring these into the acoustic loop:
   - `acoustic_kernel_size(T, N, 0) == 1:N` for `Bounded`, `Periodic`,
     `LeftConnected`, `RightConnected`, and `FullyConnected`.
   - connected-topology ranges with `H > 1` match the Oceananigans
     split-explicit formulas.
  - x/y face parameter helpers include the expected bounded normal face
    extent if they are meant to write boundary faces.

### 2026-05-21 poll 7

New changes:

- `src/CompressibleEquations/CompressibleEquations.jl` now includes
  `acoustic_kernel_parameters.jl` and `acoustic_operators.jl`.
- `_explicit_horizontal_step!` now uses `∂xTᶠᶜᶜ` / `∂yTᶜᶠᶜ`.

Findings:

1. This is no longer a no-behavior-change refactor. The active
   `_explicit_horizontal_step!` path has switched to topology-aware
   derivatives immediately, while the substep loop still appears to be the
   filled-halo loop. The plan called for a temporary filled-halo/no-fill
   selector so Stage 0 can prove the refactor before changing behavior. Please
   either add that selector or explicitly move this commit into the
   behavior-changing serial stage and add comparison tests now.

2. The `H == 0` connected-topology bug in `acoustic_kernel_size` still appears
   unresolved. `FullyConnected, H = 0` still returns `2:N-1`. This can break
   any behavior-preserving test on local connected topologies, even before
   distributed active halos are implemented.

3. The face-range concern is still unresolved. `acoustic_xface_parameters` and
   `acoustic_yface_parameters` still return center-cell normal ranges, while
   their docstrings imply wall faces are handled. Operators cannot write faces
   that were never launched. Please either make the face ranges truly
   location-aware or document why acoustic perturbation boundary faces are set
   elsewhere and do not need to be launched.

4. `acoustic_substepping.jl` imports `δxTᶜᵃᵃ, δyTᵃᶜᵃ` but the current diff
   does not use them there. If they remain unused, ExplicitImports/Aqua may
   complain. If predictor changes are coming next, this is fine temporarily.

5. I attempted a quick `julia --project -e 'using Breeze'` load check, but the
   Julia launcher failed before loading the project because it could not create
   its config lockfile in this sandbox. I could not verify imports or
   precompilation locally yet.

### 2026-05-21 poll 8

New changes:

- `_build_predictors_and_vertical_rhs!` now accepts `ρθ′ˢ⁻`.
- Predictor horizontal divergence and damping now use topology-aware
  differences/derivatives.

Findings:

1. Serious bug: the call site for `_build_predictors_and_vertical_rhs!` was
   not updated for the new `ρθ′ˢ⁻` argument. The kernel signature is now:

   ```julia
   _build_predictors_and_vertical_rhs!(ρw′_rhs, ρ′★, ρθ′★, ρθ′ˢ⁻,
                                       ρ′, ρθ′, ρw′, ρu′, ρv′, ...)
   ```

   but the launch still passes `density_perturbation` immediately after
   `density_potential_temperature_predictor`. That shifts every later
   argument by one position. Please pass
   `substepper.previous_density_potential_temperature_perturbation` in the
   new slot before `substepper.density_perturbation`.

2. The old parent-array copy is still present in the substep loop:

   ```julia
   parent(substepper.previous_density_potential_temperature_perturbation) .=
       parent(substepper.density_potential_temperature_perturbation)
   ```

   Once the predictor kernel writes `ρθ′ˢ⁻`, this copy should be removed from
   the optimized/no-fill path. Keeping both defeats part of the optimization
   and could mask whether the fused snapshot is actually correct.

3. The behavior-changing concern remains. The active loop now uses
   topology-aware operators in several kernels while the filled-halo loop is
   still structurally active. This may be mathematically equivalent in many
   cases, but it is not the staged A/B selector described in the plan. Please
   add tests that compare the current filled-halo baseline against this
   operator-refactored path before removing substep halo fills.

4. The `H == 0` connected-topology launch range issue remains unresolved in
   `acoustic_kernel_parameters.jl`.

### 2026-05-21 poll 9

New changes:

- `_explicit_horizontal_step!` reordered `apply_pressure_gradient` to the
  first positional argument.
- `_accumulate_momentum_perturbations!` kernel was added.
- `test/acoustic_substepping.jl` was updated for the direct
  `_explicit_horizontal_step!` test.

Blocking findings:

1. Production launch for `_explicit_horizontal_step!` still uses the old
   argument order in `acoustic_rk3_substep_loop!`:

   ```julia
   launch!(arch, grid, :xyz, _explicit_horizontal_step!,
           substepper.momentum_perturbation.u,
           ...
           apply_pressure_gradient)
   ```

   The kernel now expects `apply_pressure_gradient` first. The unit test was
   updated, but the actual acoustic loop was not. This will pass the wrong
   object as the boolean flag and shift every kernel argument.

2. Production launch for `_build_predictors_and_vertical_rhs!` still has the
   shifted-argument bug from poll 8. It still does not pass
   `substepper.previous_density_potential_temperature_perturbation` in the new
   `ρθ′ˢ⁻` slot.

3. `accumulate_momentum_perturbations!(substepper)` is still called in the
   acoustic loop, but the old wrapper function appears to have been replaced
   by the `_accumulate_momentum_perturbations!` kernel. There is no visible
   wrapper or launch for the new kernel, so the acoustic loop likely has a
   missing method now.

4. New imports appear unused at this point:
   - `convert_to_device`
   - `configure_kernel`
   - `@unroll`

   If these are for the next commit, fine temporarily, but source QA may fail
   if they remain unused.

5. The direct unit test update is not enough to catch the production
   call-site mismatches. Please add or run a test that actually executes
   `acoustic_rk3_substep_loop!` or `time_step!` with
   `SplitExplicitTimeDiscretization`.

### 2026-05-21 poll 10

- Blocking findings from poll 9 are still present:
  - `_explicit_horizontal_step!` production launch still uses the old argument
    order.
  - `_build_predictors_and_vertical_rhs!` production launch still omits the
    new `ρθ′ˢ⁻` argument.
  - `accumulate_momentum_perturbations!(substepper)` is still called even
    though the old wrapper appears to have been replaced by a kernel.
- No new findings beyond those blockers.

### 2026-05-21 poll 11

Progress:

- The production loop now uses configured kernels and converted argument
  tuples.
- The `_explicit_horizontal_step!` argument-order issue appears fixed in the
  configured-kernel call.
- The predictor argument list now includes
  `substepper.previous_density_potential_temperature_perturbation`.
- The old `accumulate_momentum_perturbations!(substepper)` call is replaced by
  `accumulate_kernel!(...)`.
- The old per-substep parent copy of previous `ρθ′` is gone from the new loop.

Remaining findings:

1. The implementation has jumped directly into the behavior-changing serial
   no-fill path. That can be fine, but it means the next required work is
   correctness coverage. Please add a test that executes the actual acoustic
   loop or `time_step!`, not just direct kernel unit tests.

2. `acoustic_kernel_size(..., H = 0)` is still wrong for connected topology
   methods. Please fix before using these helpers in any distributed smoke
   test or local connected topology.

3. `acoustic_xface_parameters`, `acoustic_yface_parameters`, and
   `acoustic_zface_parameters` exist but the new loop uses `center_params` for
   `_explicit_horizontal_step!`, `_thermal_divergence_damping!`, and
   `_accumulate_momentum_perturbations!`. If this is intentional because
   topology-aware operators ignore physical wall faces and the stage-end fills
   repair diagnostic velocity boundaries, document that. Otherwise use the
   location-specific helpers.

4. `KernelAbstractions.Extras.LoopInfo: @unroll` appears imported but unused.
   Remove unless it is used in the next commit.

5. The first tests should cover at least:
   - one small `time_step!` or `acoustic_rk3_substep_loop!` run with
     `SplitExplicitTimeDiscretization(substeps = 6)`;
   - default `ThermalDivergenceDamping()` to exercise fused previous-`ρθ′`;
   - `NoDivergenceDamping()` to prove the no-op damping setup path;
   - `(Bounded, Periodic, Bounded)` to exercise topology-aware x-boundaries.

### 2026-05-21 poll 12

Progress:

- The previous production call-site blockers appear fixed in the new
  configured-kernel loop.
- The substep-loop `fill_halo_regions!` calls are gone from the new loop.
- The previous-`ρθ′` parent copy is gone from the new loop.
- `initialize_stage_perturbations!` no longer zeroes predictor/previous
  workspaces or fills perturbation halos.

Remaining findings:

1. This is now fully in the serial no-fill stage, so loop-level tests are
   mandatory before further refactoring. The only visible test change is still
   the direct `_explicit_horizontal_step!` call signature update. That will not
   catch configured-kernel argument tuples, no-op damping setup, fused
   previous-`ρθ′`, or removal of perturbation halo fills.

2. Please add a minimal `time_step!` or `acoustic_rk3_substep_loop!` test on a
   small grid with `SplitExplicitTimeDiscretization(substeps = 6)`. Run it for
   both default `ThermalDivergenceDamping()` and `NoDivergenceDamping()`.

3. `initialize_stage_perturbations!` now relies on every workspace being
   overwritten before read. That seems plausible for the physical serial
   range, but it needs a test that exercises at least two substeps with
   damping enabled so stale `previous_density_potential_temperature_perturbation`
   cannot hide.

4. The `H == 0` connected-topology bug in `acoustic_kernel_size` remains.

5. The branch skipped the behavior-preserving A/B stage. That is acceptable
   only if the comparison tests are added now; otherwise it will be hard to
   distinguish operator-refactor errors from no-fill algorithm errors.

### 2026-05-21 poll 14

Progress:

- `acoustic_kernel_size(..., H = 0)` now has guards for connected topologies
  and returns `1:N`. That resolves the connected-topology stage-0 range bug.

Remaining findings:

- I still do not see new loop-level tests in `test/acoustic_substepping.jl`;
  the visible test diff remains the direct `_explicit_horizontal_step!`
  signature update. The serial no-fill loop needs at least one small
  `time_step!` / `acoustic_rk3_substep_loop!` execution test before more
  refactoring.

### 2026-05-21 post-commit review of `1eb70b6`

Progress:

- The first serial implementation is now committed as
  `1eb70b6 Phase 1: serial no-fill acoustic substepping`.
- `git diff --check main...HEAD` is clean.
- The committed source diff does remove the acoustic substep-loop
  `fill_halo_regions!` calls and preconfigures the repeated kernels with
  device-converted argument tuples inside `GC.@preserve`.
- The `H == 0` connected-topology range issue has been fixed.

Findings:

1. `RUNNING_REVIEW.md` and `acoustic_substepping_no_fill_plan_review.md` are
   included in the commit. I would keep `acoustic_substepping_no_fill_plan.md`
   if the PR is meant to carry the plan, but the live review and review
   scratch file should probably not be part of the implementation PR.

2. Test coverage is still the biggest correctness risk. The only committed
   test change I see is updating the direct `_explicit_horizontal_step!`
   unit-test call signature. Existing `time_step!` tests may catch gross
   failures, but this PR should add a targeted no-fill regression that
   exercises the configured-kernel loop, fused `ρθ′ˢ⁻` snapshot, no-op damping
   setup path, and removed perturbation halo fills.

3. Please add at least two tiny CPU runs before opening the PR:
   - `(Periodic, Periodic, Bounded)` with
     `SplitExplicitTimeDiscretization(substeps = 6)` and default
     `ThermalDivergenceDamping()`;
   - `(Bounded, Periodic, Bounded)` with
     `SplitExplicitTimeDiscretization(substeps = 6,
                                      damping = NoDivergenceDamping())`.
   These should call `time_step!` or `acoustic_rk3_substep_loop!`, not only the
   individual kernels.

4. `acoustic_operators.jl` still defines function-form derivatives as
   `f::Function`. The Oceananigans topology-aware operators do not require that
   annotation. Dropping it keeps these wrappers compatible with callable
   structs and avoids an avoidable dispatch restriction inside kernels.

5. The Oceananigans source does use a `filled_halos` selector for the free
   surface implementation. This PR intentionally skips that path, which is
   fine, but the lack of a filled-halo/no-fill A/B switch makes the targeted
   loop-level tests above more important.

### 2026-05-21 poll 18

Progress:

- A new no-fill topology smoke test was added to
  `test/acoustic_substepping.jl`.
- It covers the configured-kernel loop with `substeps = 6` for
  `(Periodic, Periodic, Bounded)`, `(Bounded, Periodic, Bounded)`, and
  `(Bounded, Bounded, Bounded)`, and it includes both default thermal damping
  and `NoDivergenceDamping()`.

Finding:

- This is useful smoke coverage, but it is still a rest-state test:
  `θ = 300`, `u = 0`, `qᵗ = 0`, and `ρ = ref.density`. That means the
  topology-aware horizontal operators and the damping gradient mostly see
  zeros, so the test proves the new loop runs but does not really stress the
  no-fill algorithm. Please add at least one nonzero horizontal perturbation,
  for example a small smooth `θ` or `ρ` perturbation with horizontal structure,
  or a small interior velocity perturbation. The key is to make `(ρθ)′`,
  pressure gradients, and/or momentum divergence nonzero so the topology-aware
  stencils at periodic and bounded edges are actually exercised.
