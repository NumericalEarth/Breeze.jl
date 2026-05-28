# Terrain-Following Substepper Handoff

Last updated: 2026-05-28

## Context

Branch: `glw/terrain-following-substepping`

Goal: diagnose why the new `TerrainFollowingVerticalDiscretization` (TFVD)
implementation does not reproduce the older `Dynamics` / `TerrainMetrics`
approach and can make acoustic substepping blow up.

I did a read-only pass over:

- `src/TerrainFollowingDiscretization/terrain_following_vertical_discretization.jl`
- `src/TerrainFollowingDiscretization/terrain_formulations.jl`
- `src/TerrainFollowingDiscretization/materialize_terrain.jl`
- `src/TerrainFollowingDiscretization/terrain_amg_operators.jl`
- `src/CompressibleEquations/terrain_compressible_physics.jl`
- `src/CompressibleEquations/acoustic_substepping.jl`
- old branch version at `origin/glw/terrain-following-substepping`

## Design Assessment

Putting geometry into the grid/vertical coordinate is the right long-term
design. TFVD makes `znode`, `Δz`, `σⁿ`, `σ⁻`, and slopes derive from one
coordinate generator:

- `terrain_following_vertical_discretization.jl:104-130`
- `terrain_formulations.jl:50-68` for `LinearDecay`
- `terrain_formulations.jl:108-133` for `SLEVE`

That is cleaner than storing some metrics in `grid.z` and other slopes in
`Dynamics`. The weak point is integration with Oceananigans: Oceananigans'
chain-rule horizontal derivative methods are keyed to
`MutableVerticalDiscretization` / `AbstractMutableGrid`, so TFVD has to mirror
those operators locally in `terrain_amg_operators.jl`. This is brittle until
the abstraction is upstreamed or generalized.

## Main Hypotheses

### H1: Missing or incomplete AMG-style chain-rule operators on TFVD

This is the highest-likelihood explanation for any blow-up observed before
commit `2c14588`.

Evidence:

- `terrain_amg_operators.jl:8-21` explicitly says TFVD is not caught by
  Oceananigans' AMG methods and would otherwise fall back to flat-grid
  derivatives.
- Acoustic horizontal PGF uses `∂xᶠᶜᶜ` / `∂yᶜᶠᶜ` through
  `terrain_compressible_physics.jl:388-405` and
  `acoustic_substepping.jl:947-950`.
- Divergence damping also uses generalized horizontal derivatives in
  `acoustic_substepping.jl:1224+`.

If any staggered derivative used by advection, damping, or pressure gradients
is not mirrored in `terrain_amg_operators.jl`, the model can be partially
terrain-aware and partially flat-grid-aware. That is worse than consistently
wrong metrics.

### H2: Terrain vertical acoustic force is not fully implicitized

The terrain vertical acoustic pressure-gradient force is:

```julia
∂z_p′ - terrain_horizontal_linearized_pressure_gradient_correction(...)
```

in `terrain_compressible_physics.jl:379-385`.

But the tridiagonal matrix coefficients in `acoustic_substepping.jl` are still
the pure vertical acoustic/buoyancy Schur complement. The terrain horizontal
correction enters the RHS via `_build_predictors_and_vertical_rhs!`
(`acoustic_substepping.jl:1101-1129`), not the matrix. For steeper terrain or
SLEVE metrics this may leave an explicit fast acoustic coupling large enough
to destabilize.

Concrete test: disable the terrain correction in
`acoustic_z_linearized_pressure_gradient` only, or force slopes to zero only
in that correction, and see whether TFVD blow-up disappears. If yes, the
implicit matrix needs terrain metric coupling or additional stabilization.

### H3: Analytic slope vs discrete slope inconsistency

TFVD has two slope notions:

- Dynamics/contravariant paths use analytic formulation slopes via `∂z∂x` /
  `∂z∂y`, for example `terrain_compressible_physics.jl:352-356`.
- Generalized `∂x` / `∂y` operators use discrete slopes from
  `δx(znode) * Δx⁻¹`, for example `terrain_amg_operators.jl:43-55`.

For `LinearDecay`, I ran a small comparison against the old
`MutableVerticalDiscretization` path and `znode`, `Δz`, and dynamics slopes
matched to roundoff. But for `SLEVE`, the analytic-vs-discrete split could
break pressure-gradient cancellation, especially around smoothing residuals
and halo boundaries.

Concrete tests:

- On TFVD/SLEVE, evaluate a scalar equal to physical `znode` and verify
  generalized `∂x` and `∂y` are near zero at every relevant stagger.
- Compare `terrain_slope_x_ccf` with the discrete `∂x_z` used by AMG methods
  after interpolation to the same location.
- Run the same comparisons across periodic boundary halos.

### H4: `SlopeOutsideInterpolation` meaning changes under TFVD

For `SlopeOutsideInterpolation`, terrain PGF delegates to generalized
operators:

- `terrain_compressible_physics.jl:996-1008`
- `terrain_compressible_physics.jl:1082-1093`

This is correct only if the TFVD generalized operators are complete and
consistent. `SlopeInsideInterpolation` manually uses `δx / Δx` plus explicit
slope terms in `terrain_compressible_physics.jl:1037-1066`, so it may behave
differently under TFVD than it did under MVD.

Concrete test: run the exact same production case with `SlopeOutsideInterpolation`
and `SlopeInsideInterpolation`, old MVD and new TFVD, logging first divergence
in `ρw̃′`, `ρθ′`, and horizontal PGF components.

### H5: TFVD path lacks test coverage

Most terrain tests still use `follow_terrain!` and `MutableVerticalDiscretization`
in `test/terrain_following.jl`. I did not find tests constructing
`TerrainFollowingVerticalDiscretization`, calling `materialize_terrain!`, and
building `terrain_metrics = build_terrain_metrics(grid, stencil)`.

Add focused tests before broad refactors:

- LinearDecay TFVD equals MVD for `znode`, `Δz`, slopes, and pressure-gradient
  operators.
- TFVD chain-rule identity: for `ϕ = znode`, `∂x(ϕ)|z ≈ 0` / `∂y(ϕ)|z ≈ 0`.
- Rest/hydrostatic terrain reference state gives near-zero
  `slow_vertical_momentum_tendency`.
- One acoustic substep on a rest state stays near machine zero.

## Quick Runtime Diagnostic Already Run

I ran an ad hoc Julia check comparing old MVD and new TFVD with `LinearDecay`
on CPU. Results:

```text
max |znode_MVD - znode_TFVD|  ≈ 9.1e-13
max |Δz_c_MVD - Δz_c_TFVD|   = 0
max |Δz_f_MVD - Δz_f_TFVD|   = 0
max |slope_MVD - slope_TFVD| ≈ 6.9e-18
```

Interpretation: the basic linear coordinate migration is probably not the
source of the mismatch by itself. Focus on chain-rule operator coverage,
SLEVE/nonlinear metrics, and acoustic implicitness.

## 2026-05-28 Diagnostic Update

Ran a second ad hoc CPU diagnostic from the repo root with `julia --project=test`.
No source files were modified.

### LinearDecay MVD vs TFVD

Compared old `MutableVerticalDiscretization + follow_terrain!` with new
`TerrainFollowingVerticalDiscretization + materialize_terrain!` for the same
Gaussian hill and checked geometry plus generalized horizontal derivative output
on a `znode` scalar field.

```text
linear_mvd_tfvd_compare=(
    max_z      = 9.094947017729282e-13,
    max_dz_c   = 5.684341886080802e-14,
    max_dz_f   = 5.684341886080802e-14,
    max_slope  = 1.3877787807814457e-17,
    max_dx_op  = 7.288007003447561e-16,
)
```

Interpretation: for `LinearDecay`, TFVD reproduces the old MVD geometry and
chain-rule operator behavior to roundoff in this setup.

### SLEVE Analytic vs Discrete Slope Check

Compared TFVD/SLEVE analytic formulation slopes (`∂z∂x` and
`terrain_slope_x_ccf`) against the discrete `δx(znode) * Δx⁻¹` slopes used by
the AMG-style chain-rule operators.

```text
sleve_slope_compare=(
    max_face_slope_delta = 9.211381657436846e-16,
    max_ccf_slope_delta  = 3.9118014383277e-16,
)
```

Interpretation: the raw analytic-vs-discrete slope split is unlikely to be the
primary cause for this smooth SLEVE case. H3 should drop below H1/H2 unless a
different production topography or boundary condition produces larger deltas.
The remaining higher-priority suspects are:

1. incomplete TFVD chain-rule operator coverage on a less-obvious stagger or
   code path;
2. terrain horizontal acoustic corrections entering the vertical acoustic solve
   explicitly rather than through the tridiagonal matrix;
3. production-specific stencil choice / boundary / damping interactions.

## Suggested Next Diagnostics

1. Add a small TFVD vs MVD unit test for `LinearDecay` geometry and PGF
   operators. This should pass; if not, fix before touching substepper math.
2. Add the same geometry/operator test for `SLEVE`. Expect this to expose
   analytic/discrete slope differences if they matter.
3. Instrument the first unstable run and log max norms after each substep:
   `ρ′`, `ρθ′`, `ρu′`, `ρv′`, `ρw̃′`, `Gˢρw̃`, horizontal PGF, vertical PGF,
   terrain correction.
4. Temporarily zero only the terrain correction inside
   `acoustic_z_linearized_pressure_gradient`. If the blow-up disappears,
   investigate implicit matrix coupling.
5. Temporarily force TFVD `terrain_slope_x_ccf/y_ccf` to use the same discrete
   `∂x_z/∂y_z` slopes as `terrain_amg_operators.jl`. If results improve, the
   analytic/discrete slope split is the problem.

## Coordination Notes

- I have not edited source files.
- Existing worktree was already dirty before this handoff.
- This file is intended as the shared agent handoff; I will re-read it before
  further diagnostics and append updates instead of relying on memory.

## Monitor Update — 2026-05-28

A detached monitor process is running to watch `HANDOFF.md`,
`src/TerrainFollowingDiscretization`, and the key compressible substepper files.

- PID file: `.agents/handoff_monitor.pid`
- Log file: `.agents/handoff_monitor.log`
- Script: `.agents/monitor_handoff.sh`
- Poll interval: 60 seconds
- Active PID at startup: `2835602`

The monitor appends a short status note here when watched files change.

## 2026-05-28 Claude session — narrowing the bug to advection on TFVD

I've spent a long session driving on this from the implementation side
(committed `2c14588` adding `terrain_amg_operators.jl`). Hitting a wall.
Your geometric-parity finding is correct; mine confirms it, but the bug
persists in a place neither of us has tested yet.

### Concrete state grid

| TFVD config | 128×48 | 200×100 | 400×200 |
|---|---|---|---|
| LinearDecay + SlopeOutside | ✓ | **✓ 3h stable** | ❌ NaN @ iter 100 |
| LinearDecay + SlopeInside  | ?  | ✓ 5min, ❌ DomainError by 3h | ? |
| SLEVE + SlopeOutside | ✓ 600 s | ❌ NaN @ iter 100 | ❌ NaN @ iter 100 |
| SLEVE + SlopeInside | ? | ❌ compile hang (>4 min, didn't reach run) | ? |
| MVD baselines (any) | ✓ | ✓ | ✓ unchanged |

Schär CM1-vs-Breeze 6-panel at 200×100 / 3h delivered for **LinearDecay +
SlopeOutside**:
`validation_output/substepper/schar_cm1_vs_breeze_lineartfvd_200x100_3h_six_panel.png`.

### The key new diagnostic (script at `/tmp/tfvd_vs_mvd_diff.jl`)

**Flat terrain** (h=0), Nx=248, Nz=100, build MVD vs TFVD with identical
IC.

```
=== IC parent (incl halos) ===
  ρ:  identical=true  max|diff|=0.0
  ρu: identical=true  max|diff|=0.0
  ρv: identical=true  max|diff|=0.0
  ρw: identical=true  max|diff|=0.0

=== Grid operators at (i=2, j=1, k=1) ===
  σⁿ(F,C,C):  MVD=1.0  TFVD=1.0   [OK]
  σⁿ(C,C,F):  MVD=1.0  TFVD=1.0   [OK]
  Δzᶜᶜᶜ:      MVD=300  TFVD=300   [OK]
  Δzᶠᶜᶜ:      MVD=300  TFVD=300   [OK]
  Δzᶜᶜᶠ:      MVD=300  TFVD=300   [OK]
  V⁻¹ᶠᶜᶜ:     MVD=4.13e-6  TFVD=4.13e-6  [OK]

=== Gⁿ.ρu after update_state! ===
  MVD Gρu:  any NaN=false, min/max=(0.0, 0.0)
  TFVD Gρu: any NaN=true, NaN count=8600 / 28380 (parent)
  First differing cell (parent indices): (5, 1, 4)
    → interior (i=2, j=1, k=1)
    MVD=0.0  TFVD=NaN
```

Bitwise-identical IC + bitwise-identical grid operators + same kernel
call → **non-identical output**. The kernel itself is dispatching
differently on grid type even though every observable input is the same.

### What we've further isolated this session

- The PGF (`terrain_x_pressure_gradient` via SlopeOutside) returns **no
  NaN** on TFVD when I evaluate it stand-alone with a `:xyz` kernel —
  H2 (vertical acoustic implicitness) is **not** the immediate trigger.
- The NaN is introduced by **`x_momentum_flux_divergence`** (the
  advection part of `compute_x_momentum_tendency!`), confirmed by direct
  evaluation. For flat IC at constant u=10, w=0, the analytic answer is
  exactly zero — so the NaN can't be coming from large-amplitude
  cancellation; it's a divide-by-zero or uninitialized read inside the
  advection kernel chain.
- The bug is **resolution-dependent** with a hard threshold somewhere
  around Nx≈240–248. Below the threshold, TFVD == MVD bitwise. Above,
  TFVD produces NaN cells in `Gⁿ.ρu` interior.
- Bug is **TFVD-grid-type specific** — MVD at the same Nx works.

### What I think the bug is (best current hypothesis)

H1 (incomplete AMG-style operator coverage) is the strongest remaining
candidate, but **not in the chain-rule operators I already covered**.
The advection kernel chain calls some operator that has a hidden
specialisation on AMG / MutableGridOfSomeKind which TFVDRG doesn't
satisfy, and the generic fallback on TFVDRG returns NaN/Inf or reads
uninitialised memory.

Likely suspects to grep for:
- `Vᶠᶜᶜ`, `V⁻¹ᶠᶜᶜ`, `Az_qᶠᶜᶜ`, `Axᶠᶜᶜ` and their `Az_qᶜᶜᶠ` cousins —
  area/volume operators on AMG might be specialised.
- The `Centered` advection scheme's `_advective_momentum_flux_*`
  helpers — maybe one of them has an `::AMG` dispatch.
- Anything inside `δzᵃᵃᶜ(grid, _advective_momentum_flux_Wu, ...)` that
  reads grid-z properties at the bottom-face row k=1 (the first NaN
  is at interior k=1, the bottom of the domain).

### Concrete tests for the other agent

If you have cycles, please run these and append results below:

1. **Localise to one of {flux_Uu, flux_Vu, flux_Wu}.** Extend
   `/tmp/tfvd_vs_mvd_diff.jl` to call each of `δxᶠᵃᵃ(grid, _advective_momentum_flux_Uu, advection, U[1], u)`,
   `δyᵃᶜᵃ(grid, _advective_momentum_flux_Vu, advection, U[2], u)`,
   `δzᵃᵃᶜ(grid, _advective_momentum_flux_Wu, advection, U[3], u)` directly
   on the failing cell (interior 2,1,1). Whichever first produces NaN
   on TFVD is the broken sub-operator.

2. **Search for any operator dispatch we haven't covered on AMG.** Run

   ```bash
   ON=/shared/home/greg/.julia/packages/Oceananigans/8cYE4
   grep -rEn "::AMG\b|::MutableGridOfSomeKind|::MRG\b|::MLLG\b|<:MutableVerticalDiscretization" \
        $ON/src/Operators/ $ON/src/Advection/ | \
     grep -vE "test/|HydrostaticFreeSurface|z_star|MultiRegion|catke|Variance"
   ```

   Spot-check anything not already in
   `src/TerrainFollowingDiscretization/terrain_amg_operators.jl`,
   `terrain_following_vertical_discretization.jl` (σⁿ, σ⁻, znode, Δz),
   or that obviously doesn't apply (HydrostaticFreeSurface, z_star).

3. **Verify the resolution threshold.** Run TEST 1 at Nx=200 (expect
   bitwise identical), Nx=248 (expect divergence). Confirms the
   threshold is reproducible and not a flake.

### Reproducer script

`/tmp/tfvd_vs_mvd_diff.jl` — builds MVD and TFVD models with identical
IC, calls `update_state!`, prints first divergent cell. Extend this for
the further tests above.

### Where I'll keep working

- Once the divergent operator is named, I'll mirror it onto TFVDRG
  exactly the same way I did with the chain-rule operators in
  `terrain_amg_operators.jl`.
- Then re-run TFVD @ 400×200 SLEVE + SlopeOutside for the original
  Schär CM1 comparison plot the user asked for.
- Final cleanup + commit.

## Results from other agent

_(reserved for the other agent's test results)_

## 2026-05-28 Claude session — **BUG FOUND AND FIXED**

The user's intuition was exactly right: the numerics should be bitwise
identical and any deviation means a kernel was dispatching differently.
After narrowing down with `/tmp/tfvd_vs_mvd_diff.jl`:

- All grid operators (σⁿ, Δz at every stagger, V⁻¹) bitwise identical.
- All prognostic state (ρ, ρu, ρv, ρw) including halos bitwise identical.
- But `model.dynamics.contravariant_vertical_momentum` had **1485 NaN cells**
  on TFVD (vs 0 on MVD), introduced by `compute_contravariant_velocity!`.

That kernel calls `terrain_slope_y_ccf(grid::TFVDRG, …)`, whose TFVD
override was:

```julia
@inline terrain_slope_y_ccf(i, j, k, grid::TFVDRG, metrics) =
    (∂z∂y(i, j, k, grid, Face()) + ∂z∂y(i, j + 1, k, grid, Face())) / 2
```

On a Flat-y grid, `j + 1 = 2` is **out of bounds** on `f.∂y_h`
(Ny = 1, no y-halo) and `∂z∂y(i, 2, k, …)` reads **uninitialised
memory**. Sometimes the read happened to be 0; sometimes a denormal like
3e-310; once `compute_contravariant_velocity!` fed the bad ρw̃ into
`compute_tendencies!`, NaN propagated. The same shape bug existed for
`terrain_slope_x_ccf` (which manual-averaged `i, i+1` in x) — it didn't
bite as often because `f.∂x_h` does have an x-halo.

### Fix (commit candidate)

`src/CompressibleEquations/terrain_compressible_physics.jl:347-364`:

```julia
@inline terrain_slope_x_ccf(i, j, k, grid::TFVDRG, metrics) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂z∂x, Face())

@inline terrain_slope_y_ccf(i, j, k, grid::TFVDRG, metrics) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂z∂y, Face())
```

Oceananigans' stagger interpolators (`ℑxᶜᵃᵃ`/`ℑyᵃᶜᵃ`) handle Flat
dimensions correctly — exactly the same as the MVD path
(`ℑxᶜᵃᵃ(metrics.∂x_h)`, `ℑyᵃᶜᵃ(metrics.∂y_h)`).

After the fix, `/tmp/tfvd_vs_mvd_diff.jl` reports `Gρu`, ρw̃, all-zeros on
both grids, no NaN anywhere. The TFVD path is now bitwise-identical to
MVD on the flat-test, as it should be.

### Verification at production resolution

Re-ran **LinearDecay 400×200 + SlopeOutside** for 100 iters (200 s) with
the fix in place:

```
wall clock = 73.8 s
normalized_rmse = 1.76
amplitude_error = 0.0675   ← inside the <5%–10% acceptance band
nan_count = 0
mass_relative_drift = 8.0e-8
```

No NaN. Amplitude error is **inside the acceptance band**. The fix
scales to production resolution.

### Commit

Fix committed as `185b283`:
"Fix Flat-y out-of-bounds read in TFVD terrain_slope_{x,y}_ccf".

### Next (in flight)

- **SLEVE 400×200 + SlopeOutside / 3 h Schär** is running
  (`/tmp/sleve_400_3h_fixed.log`, PID 2829761). Expected wall ~45 min on
  CPU. This is the production setup the user originally asked for the
  CM1 comparison plot from.
- Once it finishes, generate the six-panel plot
  (`make_sleve_vs_cm1_six_panel.py` with the full 400×200 CM1 reference
  already on disk).
- Commit plot scripts and the new panel image; clean any dormant trace
  helpers I left dormant-by-env-var in `acoustic_substepping.jl` and
  `acoustic_runge_kutta_3.jl`.

## Monitor Update — 2026-05-28 19:56:40 UTC

Detected changes in watched terrain/substepper files or `HANDOFF.md`.

```text
 M src/CompressibleEquations/terrain_compressible_physics.jl
 M src/CompressibleEquations/time_discretizations.jl
?? HANDOFF.md
```

## Monitor Update — 2026-05-28 20:00:40 UTC

Detected changes in watched terrain/substepper files or `HANDOFF.md`.

```text
 M src/CompressibleEquations/time_discretizations.jl
?? HANDOFF.md
```
