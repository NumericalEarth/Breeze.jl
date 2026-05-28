# Review: acoustic substepping no-fill optimization plan

Review of `acoustic_substepping_no_fill_plan.md` against the Oceananigans
split-explicit free-surface implementation (`SplitExplicitFreeSurfaces/step_split_explicit_free_surface.jl`,
`split_explicit_free_surface.jl`, `Operators/topology_aware_operators.jl`)
and the current Breeze acoustic substepping code
(`src/CompressibleEquations/acoustic_substepping.jl`).

## Overall

The plan is structurally sound and correctly identifies the relevant
Oceananigans pattern. The PR split is well-conceived, the test matrix is
appropriate, and the H100 benchmarking plan is solid. The refinements
below are recommended before coding begins. The most important ones are
(a) a quantitative stencil-reach analysis to size the halo budget,
(b) picking a specific approach for the topology-aware operator
selection, and (c) confirming a few implementation details against the
current acoustic loop.

## What the plan gets right

1. **Matches Oceananigans's structure.** `maybe_extend_halos` +
   `maybe_augmented_kernel_parameters` + per-side `split_explicit_kernel_size`
   dispatched on local-topology types (`split_explicit_free_surface.jl:392-413`)
   is the right pattern to import.
2. **Dispatch on local topology, not runtime branches.** Correct —
   Oceananigans uses `split_explicit_kernel_size(::Type{FullyConnected}, ...)`
   etc.
3. **Halo widening at materialization time.** Oceananigans does this via
   `with_halo` before any field allocation
   (`split_explicit_free_surface.jl:360-388`), and the substepper fields
   and the model state must share that grid.
4. **`configure_kernel` + one-shot `convert_to_device` + `GC.@preserve`.**
   Closely follows `step_split_explicit_free_surface.jl:111-133`.
5. **`Nsubsteps + 2` as a starting point + reassess.** Correctly flagged
   for reassessment.
6. **Keeping the dual path only long-term.** A single algorithm with
   `halo_width = 0` for serial is simpler than Oceananigans's
   `Val(filled_halos)` discriminator.

## Refinements

### 1. The halo-budget arithmetic needs more than `Nsubsteps + 2`

Oceananigans's `Nsubsteps + 2` is for the 2-D barotropic system where
each substep's horizontal stencil reach is 1 cell (`δxᶜᵃᵃ` of `U` to
update `η`, then `∂xᵣ` of `η` to update `U` — two single-cell stencils,
alternating, not compounding into a single field). Breeze's per-substep
horizontal stencil reach is wider:

- `_explicit_horizontal_step!`: writes `ρu′[i, j, k]`, reads
  `ρθ′[i±1, j, k]` (via `∂xᶠᶜᶜ` of `linearized_pressure_perturbation`)
  and `p[i±1, j, k]`. Reach **1**.
- `_build_predictors_and_vertical_rhs!`: writes `ρ′★[i, j, k]`, reads
  `ρu′[i±1, j, k]`, `ρv′[i, j±1, k]` (via `div_xyᶜᶜᶜ` and
  `theta_face_*_flux`). Reach **1**.
- `_post_solve_recovery!`: writes `ρθ′[i, j, k]` from `ρθ′★[i, j, k]` +
  column quantities. Reach **0** horizontally.
- `_thermal_divergence_damping!`: writes `ρu′[i, j, k]`, reads
  `ρθ′[i±1, j, k] − ρθ′ˢ⁻[i±1, j, k]` (via `∂xᶠᶜᶜ(dρθ′, ...)`) and
  `θᴸ[i±1, j, k]`. Reach **1**.

Chaining within a single substep: the loop writes (`ρu′, ρv′`), reads
them to write (`ρ′★, ρθ′★`), reads them to compute the tridiag RHS,
writes `ρw′`, reads it for `_post_solve_recovery!` to write
(`ρ′, ρθ′`), then reads `ρθ′` for damping that updates
(`ρu′, ρv′`) again. **Per substep, valid `(ρu′, ρv′)` halos need to be
one cell wider than the valid `(ρ′, ρθ′)` halos**, which in turn need to
be one cell wider for the next substep's `_explicit_horizontal_step!` to
be valid at the same range. Each substep therefore consumes about 1–2
halo cells of reach for the perturbation momenta and about 1 cell for
`ρθ′`.

Concrete budget for `Nsubsteps` substeps so that all substepper kernels
are valid over `1:Nx` after the loop:

- momentum perturbation halo width ≥ `2 · Nsubsteps + 1` (perturbation
  momenta touched twice per substep with reach 1)
- `ρθ′, ρ′` halo width ≥ `2 · Nsubsteps`
- linearization caches (`Πᴸ, θᴸ, γRᵐᴸ`), pressure, density, slow
  tendencies: width ≥ `2 · Nsubsteps + 2` (read at the widest reach by
  `_explicit_horizontal_step!`)

Write that derivation explicitly into the plan and budget the halo as
`H = c · Nsubsteps + small_constant` with `c ≈ 2`. With `Nsubsteps = 24`
that's `H ≈ 50`, significantly larger than Oceananigans's
`Nsubsteps + 2 = 26`. This matters because Oceananigans already warns
when `H ≥ N` per local dimension
(`split_explicit_free_surface.jl:372-381`) — at the local rank sizes
needed for distributed scaling, this becomes a real constraint. The
break-even between compute and exchange is sensitive to `c`.

If `c · Nsubsteps` is unaffordable on the target rank size, the
alternative is **block subcycling**: exchange every `B` substeps with
`H = c · B + small`. The plan mentions this; elevate it from "later" to a
design parameter you may need from day one of stage 2.

### 2. Where to put the topology dispatch

Oceananigans uses two layers:

- The kernel-range layer (`split_explicit_kernel_size` per topology
  type), which controls *how far* to launch.
- The operator layer (`δxTᶜᵃᵃ`, `∂xᵣTᶠᶜᶠ`, etc., in
  `Operators/topology_aware_operators.jl:30-90`), which controls *what
  to compute at the edge*.

These are independent. Breeze needs both. The plan says "small inline
acoustic operator wrappers" — **do not roll Breeze-specific wrappers
around already-topology-aware Oceananigans operators**. The Oceananigans
operators (`δxTᶜᵃᵃ`, `δyTᵃᶜᵃ`, etc.) already dispatch on `Bounded`,
`Periodic`, `LeftConnected`, `RightConnected`, `FullyConnected`, etc. —
exactly what Breeze wants. Use them directly in the kernels. The only
Breeze-specific wrappers needed are derivative operators that are not
yet upstream (e.g. a topology-aware `∂xᶠᶜᶜ` that wraps `δxTᶠᵃᵃ` ×
`Δx⁻¹ᶠᶜᶜ`). If those do not exist, upstream them.

Oceananigans's switch is done via
`@inline x_difference_operator(::Val{false}) = δxTᶜᵃᵃ` /
`Val{true}) = δxᶜᵃᵃ` (`step_split_explicit_free_surface.jl:11-19`). That
is a single algorithm path discriminated by a `Val` argument, with
identical kernels for both modes. Even if Breeze keeps only the no-fill
path, writing the kernels in that discriminated form for the first PR
makes A/B testing against the existing path trivial — the test that
"old fill-each-substep ≈ new no-fill" turns into "kernel with `Val(true)`
≈ kernel with `Val(false)`," same kernel object. Recommend adding this
rather than the plan's current "use one no-fill acoustic algorithm"
up-front.

### 3. `BatchedTridiagonalSolver.solve!` is a real upstream blocker

Confirmed by reading
`Oceananigans/src/Solvers/batched_tridiagonal_solver.jl:108-131`:
`solve!` calls `launch!(arch, grid, :xy, ...)` with a hardcoded launch
config. There is no `KernelParameters` plumbed through. For the
**serial** no-fill path with `halo_width = 0`, the `:xy` launch matches
`1:Nx, 1:Ny` and is fine. For the **distributed** path with wide active
halos, the tridiag must launch over the augmented range. Consequences:

- Plan PR 5 ("upstream `KernelParameters` support for the batched
  tridiagonal acoustic solve") is only needed by PR 6 (distributed).
  Plan ordering is fine.
- Do the Oceananigans upstream PR **early** so it merges and gets a
  release before it is needed downstream. This is a small additive
  change (parameterize the `launch_config` argument, fall back to the
  current symbol).

### 4. The "save previous ρθ′" copy

The current code does this between substeps
(`acoustic_substepping.jl:1355-1356`):

```julia
parent(substepper.previous_density_potential_temperature_perturbation) .=
    parent(substepper.density_potential_temperature_perturbation)
```

The plan suggests fusing this into `_post_solve_recovery!`. **Fuse into
`_build_predictors_and_vertical_rhs!` instead**:

- The predictor kernel already reads `ρθ′[i, j, k]` to compute the
  predictor; it can write `ρθ′ˢ⁻[i, j, k] = ρθ′[i, j, k]` in the same
  line.
- That gives the correct semantics for **every** substep including the
  first (where `_post_solve_recovery!` of a *previous* substep does not
  exist).
- The damping kernel reads `ρθ′ˢ⁻` after `_post_solve_recovery!` has
  run, so the ordering still works.

Fusing into `_post_solve_recovery!` would require special-casing the
very first substep (stage init writes the initial `ρθ′`, then substep 1's
column kernel runs and immediately needs `ρθ′ˢ⁻` for damping).

### 5. Stage-initialization halo fills

`initialize_stage_perturbations!` currently does five
`fill_halo_regions!` calls (`acoustic_substepping.jl:873-877`). With
wide active halos, none of these are needed *if* the launch range for
`_initialize_perturbation_with_rewind!` covers the full active halo
region AND the input fields (`Uᴸ_outer.*`, `model.dynamics.density`,
etc.) already have valid wide halos. Two prerequisites worth stating in
the plan:

- the launch range for `_initialize_perturbation_with_rewind!` must be
  the same wide active range as the substep kernels;
- the model-side `fill_halo_regions!` that runs once at stage entry
  (before the substep loop) must fill the wide active halo of the model
  state, not just the model's natural halo width. This means either (a)
  the model grid uses the same widened halos as the substepper grid (the
  cleanest), or (b) explicit wider-halo fills are called.

The plan mentions both options under "Decide where halo widening
happens" but should commit to (a). Oceananigans pulls the same trick
(`SplitExplicitFreeSurface{<:Any, <:DistributedField}` uses
`free_surface.displacement.grid` distinct from `model.grid` only for
fields the free surface owns; the underlying baroclinic fields share the
extended grid via the materialize path). The Breeze acoustic loop reads
model-owned `pᴸ, ρᴸ`, `Gⁿρu, Gⁿρv, Gⁿρ`, `Gˢρθ`, `Gˢρw`, and (live)
`model.momentum.*` (only at recovery, also through `Uᴸ_outer`). All of
those need wide halos. A substepper-only extended grid will not work.

### 6. Vertical operators

The plan says "Vertical operators are a separate category." Be explicit:
`ℑbzᵃᵃᶠ` (`acoustic_substepping.jl:574-585`) already handles
`peripheral_node` correctly, so vertical boundaries are taken care of.
No changes needed for the first distributed target.

### 7. `accumulate_momentum_perturbations!` parent broadcasts

The three `parent(...) .+=` calls (`acoustic_substepping.jl:1190-1198`)
launch over the entire parent array, halos included. Replacing them with
a fused kernel over the active face range is a clear win and reduces
three launches per substep to one. Fuse into another kernel that already
touches those cells — most naturally `_thermal_divergence_damping!`, or
into the bottom of `_explicit_horizontal_step!` of the next substep
(less clean). A standalone kernel launched over
`acoustic_xface`/`yface`/`zface` ranges with the three fields as args is
probably simplest.

### 8. PR sequence and what to test first

The PR sequence is reasonable, but swap #1 and #2:

- **PR 1**: Add launch-parameter helpers and topology-aware operator
  wrappers, with kernels written in the discriminated `Val{filled_halos}`
  form. *No behavior change yet* — both paths still call
  `fill_halo_regions!` between substeps. Tests pass identically. This
  proves the operator+launch refactor is bit-identical at
  `halo_width = 0`.
- **PR 2**: Add the benchmark hooks and baseline case.
- **PR 3**: Flip the substep loop to no-fill (`Val(false)` mode), remove
  the substep-loop `fill_halo_regions!` calls and parent broadcasts.
  Tests now exercise the no-fill path against the old fill path (still
  selectable via a constructor option or test toggle).
- **PR 4**: Hoist configured kernels and `convert_to_device`.
- **PR 5**: Upstream `KernelParameters` for the batched tridiag.
- **PR 6**: Distributed halo widening and connected-topology launch
  ranges.
- **PR 7**: Multi-GPU correctness tests and H100 profiling.

Reason for swapping 1 and 2: getting the launch helpers in first means
the benchmark scaffolding can use them, and the benchmark gives you a
controlled regression for everything that comes after.

### 9. Smaller notes

- The plan does not mention `assemble_slow_vertical_momentum_tendency!`
  (`acoustic_substepping.jl:752-777`). It is launched once per stage but
  writes `Gˢρw` which is then read inside the substep loop at
  active-halo indices. Its launch range must also be widened. Same for
  `refresh_linearization_basic_state!` (`Πᴸ, θᴸ, γRᵐᴸ` need wide-halo
  values).
- `linearization_exner`, `linearization_potential_temperature`,
  `linearization_gamma_R_mixture` currently call `fill_halo_regions!`
  once per stage (`acoustic_substepping.jl:420-422`). With wider halos,
  these become the only mechanism by which connected ranks share
  boundary-state data. Keep these as explicit fills (at the new wider
  halo width).
- The plan's tests do not mention **backward integration** (`Δt < 0`).
  The recent PR #718 added backward acoustic substepping; the no-fill
  optimization needs to preserve that. Add `Δt < 0` to the test sweep.
- The plan benchmarks Float32 first. Check that all the cached
  arithmetic (`δτᵐ⁺^2`, `γRᵐᴸ * Πᴸ`, etc.) is well-conditioned in
  Float32 with the larger H100 grids — particularly the off-centered CN
  tridiag, where `θᶜᶜᶠ * Δz⁻¹` cancellations could matter.
- For the active-halo width, watch out for **immersed boundaries** down
  the road: not present now, but if added, the `peripheral_node` masking
  inside the active halo region will need active-cells-map support
  (Oceananigans's `get_active_cells_map`). Not for this PR series, but
  worth flagging.

## Bottom line

The plan is well-aligned with the Oceananigans split-explicit pattern
and the PR sequence is the right shape. Before starting code, add:

1. An explicit per-substep stencil-reach analysis driving a halo-budget
   formula `H ≈ 2 · Nsubsteps + small`, with block subcycling as a
   fallback design parameter.
2. A commitment to use Oceananigans's existing `δxT`/`∂xT`
   topology-aware operators directly, plus a list of which derivative
   variants need to be upstreamed.
3. A note on which kernels
   (`assemble_slow_vertical_momentum_tendency!`,
   `refresh_linearization_basic_state!`,
   `_initialize_perturbation_with_rewind!`) also need widened launch
   ranges.
4. A switch to fusing `previous_ρθ′` into
   `_build_predictors_and_vertical_rhs!` rather than
   `_post_solve_recovery!`.
5. A reordered PR sequence putting the launch-helper refactor before the
   benchmarking scaffold, written in the `Val{filled_halos}`-discriminated
   form to make A/B comparison trivial during the transition.
