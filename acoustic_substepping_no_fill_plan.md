# Acoustic substepping no-fill optimization plan

This note plans one staged draft PR for optimizing `CompressibleDynamics` with
`SplitExplicitTimeDiscretization` by removing halo fills from the acoustic
substep loop. The implementation can proceed in staged commits, but the PR
should not merge after only the serial path: changing the acoustic loop to
no-fill semantics without completing distributed support risks breaking
distributed grids.

The staged workflow is:

1. Refactor launch ranges and operator selection with no behavior change.
2. Build and validate the serial no-fill path on the PR branch.
3. Benchmark and profile the serial stage on an H100 while the PR remains
   draft.
4. Extend the same PR to distributed grids with active connected halo columns,
   widened storage, and communication/computation overlap.
5. Benchmark and profile the distributed path before marking the PR ready.

The Oceananigans split-explicit free-surface implementation is the closest
model for the distributed stage. In Oceananigans, local partition topology
selects whether the kernel launch range is augmented, while topology-aware
operators handle physical boundaries. Breeze should follow that structure, but
does not need to maintain a second long-term "filled halos" acoustic path after
the transition is validated.

## Current acoustic loop

The current loop in `src/CompressibleEquations/acoustic_substepping.jl` does
the following work each acoustic substep:

- Explicit horizontal momentum update for the perturbation momenta.
- Halo fill for horizontal perturbation momenta.
- Copy the previous thermodynamic-density perturbation for damping.
- Build cell-centered predictors and the vertical tridiagonal right-hand side.
- Solve the implicit vertical acoustic system with `BatchedTridiagonalSolver`.
- Recover density and thermodynamic-density perturbations.
- Halo fill density and thermodynamic-density perturbations.
- Apply optional divergence damping.
- Halo fill horizontal perturbation momenta again.
- Accumulate perturbation momenta for the time-averaged transport velocity.

The performance target is to remove the halo fills and bulk broadcasts from
this inner loop, then reduce per-substep launch overhead and argument
conversion overhead.

Velocity diagnosis is not part of the inner loop. `model.velocities` are used
before the acoustic loop when slow tendencies are assembled, and full-state
velocities are diagnosed after the loop for the next RK stage. The recent
`compute_velocities!` refactor is therefore only a stage-boundary analogy, not
the central substep optimization.

## Core design

Use one no-fill acoustic algorithm in the ready-to-merge PR. During staged
development, it is useful to keep a temporary internal selector equivalent to
Oceananigans' `Val(filled_halos)` pattern so the same kernels can be tested in
the old fill-each-substep mode and the new no-fill mode. That selector should
exist for validation, not as a long-term public algorithm mode.

The serial and distributed cases should differ mainly through launch ranges
and storage extent. The implementation should introduce acoustic launch
helpers, for example:

- `acoustic_center_parameters(grid, active_halo_width)`
- `acoustic_xface_parameters(grid, active_halo_width)`
- `acoustic_yface_parameters(grid, active_halo_width)`
- `acoustic_zface_parameters(grid, active_halo_width)`
- `acoustic_column_parameters(grid, active_halo_width)`

The names are placeholders. The important contract is that launch ranges are
chosen by field location and local topology.

For serial grids, `active_halo_width == 0`. Kernels should launch over
physical center cells and over the required physical face extent for kernels
that write face fields. For example, a kernel that writes an `XFaceField` on a
`Bounded` x topology may need to include the normal boundary face. It should
not compute through unused halo regions in serial.

For distributed grids, `active_halo_width > 0` only in locally connected
horizontal directions. The range should be augmented on connected sides and
not augmented on physical bounded sides. This follows the Oceananigans pattern:

- ordinary or physical bounded side: `1:N`
- fully connected side: `-H+2:N+H-1`
- right-connected side: `1:N+H-1`
- left-connected side: `-H+2:N`

Use dispatch on Oceananigans local topology types rather than runtime `if`
branches where possible.

## Topology-aware operators

All horizontal stencil operations inside the acoustic loop need to be valid
without a fresh halo fill after every substep. This means the acoustic kernels
must stop relying on ordinary halo-reading horizontal operators at physical
boundaries.

There are two independent layers:

- The launch-range layer controls how far kernels run.
- The operator layer controls what happens at physical and connected edges.

Use Oceananigans' existing topology-aware operators directly where they exist,
for example `δxT...` and `δyT...` variants. Do not wrap these in
Breeze-specific abstractions unless Breeze needs an operator that is not yet
available upstream. For missing derivatives such as a topology-aware
`∂xᶠᶜᶜ`, prefer adding or upstreaming the small derivative wrapper built from
the topology-aware difference and metric factors.

The operator audit starts with:

- `_explicit_horizontal_step!`: horizontal pressure-gradient terms.
- `_build_predictors_and_vertical_rhs!`: horizontal mass divergence and
  horizontal thermodynamic-flux divergence.
- `_thermal_divergence_damping!`: horizontal gradients of the divergence
  proxy and face interpolation of the stage-entry thermodynamic state.

Topology-aware operators are needed in serial and distributed runs:

- For `Bounded` directions, no halo exists outside the physical domain, so
  boundary behavior must be handled by the operator or by explicitly written
  boundary faces.
- For `Periodic` directions in serial, the operator must wrap correctly rather
  than depending on a refreshed perturbation halo every substep.
- For distributed connected directions, topology-aware operators are still
  needed at global bounded edges, while active connected halo columns provide
  neighbor-rank data inside the augmented range.

Vertical operators are a separate category. The current
`ℑbzᵃᵃᶠ` helper already handles top and bottom boundaries with
`peripheral_node`, so no vertical-boundary operator change is expected for the
first distributed target. A z-partitioned distributed tridiagonal solve is out
of scope for this PR.

## Active halo budget

Do not assume the Oceananigans free-surface budget `Nsubsteps + 2` is large
enough for acoustic substepping. The acoustic loop has a wider chained
horizontal stencil reach than the 2-D free-surface loop.

Per substep:

- `_explicit_horizontal_step!` writes `(ρu)′, (ρv)′` and reads neighboring
  `ρθ′`, pressure, and linearization fields through horizontal gradients.
  Horizontal reach is 1.
- `_build_predictors_and_vertical_rhs!` writes `ρ′★`, `ρθ′★`, and the vertical
  RHS, and reads neighboring `(ρu)′`, `(ρv)′` through horizontal divergence and
  thermodynamic flux divergence. Horizontal reach is 1.
- `_post_solve_recovery!` has no horizontal stencil reach.
- `_thermal_divergence_damping!` writes `(ρu)′`, `(ρv)′` and reads neighboring
  `ρθ′`, previous `ρθ′`, and `θᴸ` through horizontal gradients and face
  interpolation. Horizontal reach is 1.

The chained dependency means a valid momentum-perturbation active halo needs
to be one cell wider than the scalar perturbation active halo, and the scalar
perturbation active halo must remain wide enough for the next substep's
pressure-gradient update.

Initial distributed budget:

- `(ρu)′`, `(ρv)′`, `(ρw)′` perturbation storage: at least
  `2 * B + 1` active horizontal halo cells for a block of `B` no-fill
  acoustic substeps.
- `ρ′` and `ρθ′` perturbation storage: at least `2 * B` active horizontal
  halo cells.
- Stage-entry fields and caches read by acoustic kernels, including `Πᴸ`,
  `θᴸ`, `γRᵐᴸ`, pressure, density, momenta, and slow tendencies: at least
  `2 * B + 2` active horizontal halo cells.

For a monolithic no-fill stage with `B = Nsubsteps`, this gives
`H ≈ 2 * Nsubsteps + O(1)`. With `Nsubsteps = 24`, the active halo budget is
roughly 50 cells, not the Oceananigans free-surface value of 26. This can be
unaffordable when local rank sizes are small.

Therefore block subcycling is not merely a later fallback. Treat block size
`B` as a distributed design parameter from the start:

- `B = Nsubsteps` gives one halo exchange per stage and the largest active
  halo.
- smaller `B` exchanges every `B` substeps and reduces memory and extra
  active-halo compute.
- the H100/multi-GPU benchmark must identify the break-even between fewer
  exchanges and extra 3-D active-halo work.

## PR implementation stages

The PR should be implemented in staged commits and kept draft until the
distributed path is working. Each stage should leave the branch in a testable
state, even if later stages are required before merge.

### Stage 0: no-behavior-change refactor and guardrails

Before changing the algorithm:

1. Add or identify a compact serial correctness case for
   `SplitExplicitTimeDiscretization`.
2. Add or identify a distributed smoke test that exercises the current
   acoustic path, so accidental distributed breakage is visible.
3. Add acoustic launch-parameter helpers for center, face, and column kernels
   with `active_halo_width = 0`.
4. Refactor relevant kernels to use a temporary internal operator selector:
   - filled-halo mode uses the current ordinary operators and keeps the
     current substep-loop halo fills.
   - no-fill mode uses topology-aware operators.
   - tests can compare both modes through the same kernels.
5. Keep the current fill-each-substep behavior as the active path for this
   stage.
6. Add benchmark hooks if needed to time:
   - the full outer timestep,
   - one RK stage,
   - the acoustic substep loop,
   - halo-fill calls inside the acoustic loop.

This stage should not change numerical results. It proves the launch/operator
refactor before the algorithm is flipped to no-fill.

### Stage 1: serial no-fill path

The serial stage removes substep-loop halo fills without computing through
extra halo cells.

Implementation steps:

1. Flip the serial acoustic loop to no-fill mode with `active_halo_width = 0`.
2. Replace `:xyz` and `:xy` launches inside the acoustic loop with the new
   launch helpers where field location matters. Use location-specific face
   ranges where a kernel writes a face field.
3. Remove the substep-loop `fill_halo_regions!` calls for perturbation
   momenta, density perturbation, and thermodynamic-density perturbation.
4. Fuse the previous-`ρθ′` copy into `_build_predictors_and_vertical_rhs!`,
   not `_post_solve_recovery!`. The predictor kernel already reads the old
   `ρθ′`, so it can write `ρθ′ˢ⁻[i, j, k] = ρθ′[i, j, k]` before the recovery
   kernel overwrites `ρθ′`. This preserves first-substep semantics without a
   special case.
5. Replace `accumulate_momentum_perturbations!` parent broadcasts with a fused
   kernel over active face ranges. A standalone one-launch kernel for all
   three momentum components is the simplest first implementation; later
   fusion into a neighboring kernel can be benchmarked.
6. Revisit `initialize_stage_perturbations!`.
   - The zero fills for predictor fields and previous perturbation fields are
     not needed if their active ranges are fully written before read.
   - The time-averaged velocity accumulators still need initialization, but it
     should be done by a kernel over the active accumulation ranges or by a
     first-substep assignment path.
7. Update `finalize_time_averaged_velocity!` to use field-location-aware
   launch ranges. This is outside the substep loop, but it keeps the no-fill
   contract consistent at stage boundaries.
8. Hoist kernel configuration and device argument conversion out of the
   substep loop:
   - Use `configure_kernel` for repeated kernels.
   - Build stable argument tuples once.
   - Use `convert_to_device(arch, args)` once outside the substep loop.
   - Wrap converted argument use in `GC.@preserve`, following the
     Oceananigans split-explicit free-surface implementation.
9. Keep stage-boundary halo fills for the full prognostic model state while
   serial correctness is being established. The first goal is no halo fills
   inside the acoustic loop, not eliminating every halo fill in the timestep.

For the serial stage, the existing physical-domain `BatchedTridiagonalSolver`
launch may be sufficient because no active horizontal halo columns are solved.
Explicit `KernelParameters` support in the vertical solve becomes a hard
requirement in the distributed stage.

Serial validation:

- One RK stage comparison against the baseline on small grids.
- Several full timesteps compared against the baseline with tolerances based
  on floating point precision.
- Rest-state preservation with `SplitExplicitTimeDiscretization`.
- Bounded, periodic, and mixed topologies, especially `(Periodic, Periodic,
  Bounded)`, `(Bounded, Periodic, Bounded)`, and `(Bounded, Bounded, Bounded)`.
- `NoDivergenceDamping`, default `ThermalDivergenceDamping`, and upper sponge.
- Fixed substep counts such as 6, 12, and 24.
- Adaptive substeps for the serial path.
- Backward integration with `Δt < 0`, since split-explicit supports backward
  timestepping except for intentionally irreversible sponge behavior.

After this stage, push the branch and open a draft PR if it is not already
open. The PR should remain draft because distributed support is not complete.

### Stage 2: serial H100 benchmark and profile

Benchmark the serial stage on the draft PR branch before distributed work
adds more moving parts.

Use the existing `benchmarking/` framework, but add a compressible
split-explicit acoustic benchmark case or options that force:

- `dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(...))`
- fixed `substeps` values
- optional damping and sponge sweeps
- representative horizontal and vertical resolutions

Serial benchmark variants:

1. Baseline current acoustic loop.
2. Preconverted/configured kernels only, if isolated in a commit.
3. No-fill topology-aware operators only, if isolated in a commit.
4. No-fill plus fused copy/accumulation kernels.
5. Combined serial optimized path.

Suggested single-GPU H100 sweeps:

- Grid sizes: `64x64x32`, `128x128x64`, `256x256x128`, and at least one
  larger case that saturates the H100.
- Substeps: 6, 12, 24, 48.
- Topologies: periodic-periodic-bounded and bounded-periodic-bounded.
- Damping: none, default thermal divergence damping, and upper sponge.
- Precision: Float32 first, Float64 second.

Metrics to report:

- Wall time per outer timestep.
- Wall time per RK stage.
- Wall time per acoustic substep loop.
- Number of kernel launches per acoustic substep and per outer timestep.
- Number and total time of `fill_halo_regions!` calls.
- Effective cell-updates/s and acoustic-substep-updates/s.
- GPU memory bandwidth and achieved occupancy for the dominant kernels.
- Allocation count on CPU setup paths.

Use `CUDA.@sync` around timed regions. For microbenchmarks, prefer CUDA events
or `CUDA.@elapsed` after warmup. Avoid including JIT compilation in reported
timings.

Nsight Systems should confirm that no `fill_halo_regions!` work appears inside
the acoustic substep loop for the optimized serial path. Nsight Compute should
profile:

- `_explicit_horizontal_step!`
- `_build_predictors_and_vertical_rhs!`
- the vertical tridiagonal solve kernel
- `_post_solve_recovery!`
- `_thermal_divergence_damping!` when damping is enabled
- the new accumulation/finalization kernels

This benchmark stage should decide whether to keep all serial optimizations in
the PR or split any low-value/high-risk cleanup out before the distributed work
continues.

### Stage 3: distributed no-fill path

The distributed stage extends the same no-fill kernels to compute active
connected halo columns. This is a longer project than the serial stage because
it requires storage changes and communication scheduling, not only operator
and launch changes.

Initial distributed scope:

- Horizontal partitioning only.
- Full vertical columns on each rank.
- Fixed integer `substeps` or fixed block size `B` first.
- No z-partitioned tridiagonal solve.

The fixed-substep or fixed-block restriction is intentional. If
`substeps === nothing`, the adaptive acoustic substep count is chosen from the
timestep and grid spacing. For wide-halo no-fill substepping, the active halo
width must be known when fields are allocated. Distributed adaptive support
should come later via either:

- a user-configured maximum acoustic halo width, or
- block subcycling with a fixed maximum block size.

Distributed implementation steps:

1. Choose the distributed active halo budget from the stencil-reach analysis:
   `H ≈ 2 * B + O(1)`, where `B` is the number of substeps advanced between
   halo exchanges.
2. Commit to materialization-time halo extension for fields that participate
   in acoustic substepping. This is the primary distributed design. A
   substepper-only extended grid is not sufficient because the acoustic loop
   reads model-owned pressure, density, momenta, thermodynamic density, and
   slow-tendency fields. Extended acoustic caches are a fallback only if
   model-grid halo extension proves infeasible, and would need their own
   correctness and synchronization plan.
3. Build substepper fields on storage that includes the active connected halo
   range.
4. Ensure every field read inside active connected halo columns has valid data
   there. This includes perturbation fields, linearization caches, pressure,
   density, thermodynamic-density fields, momenta, slow tendencies, and
   `Gˢρw`.
5. Widen all once-per-stage acoustic setup kernels and fills that feed the
   substep loop:
   - `refresh_linearization_basic_state!` for `Πᴸ`, `θᴸ`, and `γRᵐᴸ`;
   - `assemble_slow_vertical_momentum_tendency!` for `Gˢρw`;
   - `_initialize_perturbation_with_rewind!` for all perturbations;
   - stage-entry halo fills for model state, linearization caches, and slow
     tendencies.
6. Fill connected wide halos once before each no-fill block. If block
   subcycling is used, exchange every `B` substeps.
7. Use local connected topology to choose augmented kernel ranges. Do not
   augment across global physical bounded boundaries.
8. Extend the vertical tridiagonal solve to the augmented horizontal column
   range. This likely requires `KernelParameters` support in
   `BatchedTridiagonalSolver` or an acoustic-specific solve wrapper.
9. Start the Oceananigans upstream PR for `BatchedTridiagonalSolver`
   `KernelParameters` support early. This should be an additive change that
   defaults to the current `:xy` launch configuration.
10. Overlap communication and computation.
    - Start wide-halo communication as early as possible after stage-entry
      fields and caches are ready.
    - Compute interior columns while connected halo exchange is in flight.
    - Complete boundary/active-halo columns after communication completes.
    - Profile whether the overlap hides communication enough to offset the
      extra 3-D active-halo compute.
11. Compare single-rank and multi-rank results on the same global grid to
    check that the distributed active-halo algorithm matches the serial
    no-fill algorithm in the physical domain.

Distributed validation:

- Same global grid, one rank versus multiple horizontal ranks.
- Periodic horizontal partitioning.
- Globally bounded horizontal partitioning, verifying first and last ranks do
  not compute past physical walls.
- Mixed local topology cases: left-connected, right-connected, and fully
  connected partitions.
- Conservation checks for mass and bounded normal momentum behavior.
- Backward integration with `Δt < 0`, excluding irreversible sponge
  expectations.
- H100 or multi-GPU profiling that reports communication frequency,
  communication/computation overlap, and extra active-halo compute cost.

### Stage 4: final benchmark and PR readiness

The PR is ready only after both serial and distributed paths pass correctness
tests and the performance evidence is documented.

Final benchmark variants:

1. Baseline current acoustic loop.
2. Combined serial optimized path.
3. Distributed active-halo path without overlap, if available as a diagnostic.
4. Distributed active-halo path with overlap.
5. Distributed block-subcycling variants for several block sizes `B`.

Expected performance signatures:

- Serial no-fill should reduce substep-loop wall time mostly through removing
  halo-fill launches and synchronization points.
- Preconversion/configured kernels should reduce CPU overhead and launch-side
  gaps, especially for small and moderate grids.
- Distributed no-fill should improve more strongly as substeps and rank count
  increase, because communication frequency drops from every substep to every
  block or every stage.
- Distributed wide-halo compute may be slower for small local domains, large
  substep counts, or large block sizes if the extra 3-D column work dominates
  communication savings. The H100 benchmarks need to identify that break-even.

## Veracity and risk review

High-confidence facts:

- The current acoustic substep loop fills perturbation halos inside the
  substep loop.
- Velocities are not diagnosed or consumed inside the acoustic substep loop;
  they matter before the loop for slow tendencies and after the loop for the
  next RK stage and transport velocities.
- Topology-aware horizontal operators are required for physical bounded
  boundaries and useful for serial periodic no-fill behavior.
- Oceananigans split-explicit free surface uses local connected topology and
  augmented `KernelParameters` to compute through active halos.
- `BatchedTridiagonalSolver.solve!` currently launches over `:xy`; serial
  no-fill can likely use that, but distributed active halos need an augmented
  column solve.
- A distributed active-halo acoustic path is materially harder than the serial
  path because the acoustic solve is 3-D and includes a vertical tridiagonal
  solve over every active horizontal column.

Key risks:

- Halo budget: the acoustic stencil reach likely requires
  `H ≈ 2 * B + O(1)`, not `Nsubsteps + 2`. This can erase the distributed
  performance benefit if local rank sizes are too small.
- Missing operator coverage: Breeze may need topology-aware variants for
  derivatives, differences, interpolations, and area-weighted flux operators
  used by the acoustic kernels.
- Face extents: kernels that write staggered fields may currently miss
  physical boundary faces if they continue to use center-cell launch ranges.
- Boundary conditions: no-fill semantics must preserve impenetrable walls,
  open boundaries, sponge behavior, and divergence damping.
- Previous-value storage: eliminating the parent-array copy for
  thermodynamic-density perturbations must preserve damping semantics exactly;
  fusing into the predictor kernel is the lowest-risk route.
- Tridiagonal solver range: serial can likely keep the physical-column solve,
  but distributed active halos need solver launches and scratch storage over
  augmented horizontal ranges.
- Extended storage: distributed kernels read model-owned stage-entry fields
  and slow tendencies, not only substepper fields. The primary plan is to
  build participating model and substepper fields on widened halos at
  materialization time. Extended acoustic caches are a higher-risk fallback.
- Stage-entry setup: `refresh_linearization_basic_state!`,
  `assemble_slow_vertical_momentum_tendency!`, and
  `_initialize_perturbation_with_rewind!` must all cover the same active
  range as the substep loop in distributed runs.
- Communication overlap: the distributed performance case depends on hiding
  communication behind useful work. Wide halos alone may be slower for small
  local domains.
- PR size: one PR is necessary to avoid merging a distributed regression, but
  the implementation should still be reviewed in coherent staged commits.

The old fill-each-substep implementation should be kept only long enough to
validate the no-fill path and produce baseline benchmarks. It should not become
a long-term maintained acoustic mode unless the no-fill path exposes an
unavoidable correctness issue.
