# Follow-up: `SlopeInsideInterpolation` × `TerrainFollowingVerticalDiscretization` blowup

## Summary

After commit `2c14588` ("Add AMG-equivalent chain-rule operators for
TerrainFollowingVerticalDiscretization grids"), the TFVD grid runs the full
Schär mountain-wave validation cleanly to 300 iterations (Nx=128, Nz=48, 600 s)
when paired with `SlopeOutsideInterpolation`. The instability that previously
blocked the work is **isolated to the single combination**
`TFVD × SlopeInsideInterpolation`.

| Configuration                            | Result                                |
|------------------------------------------|---------------------------------------|
| MVD + SlopeInsideInterpolation (default) | stable (the existing BTF prod path)   |
| MVD + SlopeOutsideInterpolation          | stable                                |
| TFVD + SlopeInsideInterpolation          | NaN at iteration ≈ 60–100             |
| TFVD + SlopeOutsideInterpolation         | stable; SLEVE MP4 produced            |

## Why the chain-rule operators were necessary but not sufficient

`src/TerrainFollowingDiscretization/terrain_amg_operators.jl` mirrors
Oceananigans' `AbstractMutableGrid`-dispatched chain-rule horizontal-derivative
methods onto `TFVDRG`. Without it, `∂xᶠᶜᶜ(grid::TFVDRG, ϕ)` falls back to the
flat-grid `δxᶠᶜᶜ(ϕ)·Δx⁻¹` and the chain-rule term `−(∂z/∂x)_ζ·(∂ϕ/∂z)` is lost.

`SlopeOutsideInterpolation` reaches the chain rule **through** `∂xᶠᶜᶜ(grid, ϕ)`,
so it needs these operators. `SlopeInsideInterpolation` builds the chain rule by
hand from `δxᶠᶜᶜ(ϕ) − ℑz(ℑx(slope·∂z(ϕ)))` and never touches `∂xᶠᶜᶜ`, so the
operators do not change its behavior.

## What is different in `SlopeInside × TFVD`

The slope used by `SlopeInside` is `terrain_slope_x_ccf(i, j, k, grid, metrics)`,
which on a TFVD grid is overridden to use the formulation's analytic
`∂z∂x`. The continuous formula is identical to the BTF/MVD discrete formula
(at machine precision); the one-step state matches to ~1e-13. But the run
diverges over ~60–100 iterations, while MVD with the same `SlopeInside` stencil
remains stable.

Open hypotheses (none confirmed):

1. A stagger/halo asymmetry: TFVD's override averages `∂z∂x` at faces i and
   i+1, while BTF's uses `ℑxᶜᵃᵃ(metrics.∂x_h)`. The two are formally equivalent
   on a periodic grid with filled halos, but `materialize_terrain!` uses
   `fill_halo_regions!` on temporary fields whereas `follow_terrain!` writes
   directly into `metrics.∂x_h` after halo fill — a roundoff-level divergence
   path could exist at i = Nx that compounds under the substep loop.
2. A subtle difference between the analytic
   `∂z∂x = ∂x_h(i, j, 1) · (1 − ζ / z_top)` (TFVD)
   and the BTF stored-array form
   `metrics.∂x_h(i, j, 1) · (1 − ζ / z_top)`. Both are bit-identical to 1e-17
   at step 1; whether they remain so under the model's halo-fill / property-
   access patterns has not been verified.
3. Some non-`∂x`-mediated AMG dispatch that MVD takes but TFVD does not. Ruled
   out for σⁿ/σ⁻/∂t_σ/znode/Δz; remaining candidates would have to live in
   a path the substepper hits but the static diagnostic missed.

## Next steps if someone resumes this

- Instrument the SlopeInside PGF (e.g.
  `terrain_x_linearized_pressure_gradient` and
  `terrain_x_full_dry_pressure_gradient`) to dump every term at a fixed
  `(i, j, k)` for both BTF/MVD and LinearDecay/TFVD over the first 100
  iterations. The first divergence beyond roundoff identifies the bad
  component.
- Compare `parent(metrics.∂x_h)` vs `parent(grid.z.formulation.∂x_h)` halo
  regions in detail at i=Nx for the periodic boundary.
- If a divergence is found in `terrain_slope_x_ccf` at the boundary, the fix
  is most likely in `_fill_terrain_slopes!` (`materialize_terrain.jl`).

## Production status

- **Shipped:** `SlopeOutsideInterpolation` is the current TFVD production
  stencil. SLEVE Schär run produced at default 128×48 / 600 s, full 300 steps,
  no NaN. Artifacts under
  `validation_output/substepper/terrain_schar_mountain_wave/`.
- **Untouched:** `MVD + SlopeInsideInterpolation` (the existing BTF production
  path) is bit-for-bit unchanged.

## Memo for the upstream PR (option 2)

Independent of this bug, the right long-term fix for the chain-rule dispatch
is upstream: introduce `AbstractMutableVerticalCoordinate` in Oceananigans and
re-key `AbstractMutableGrid` on it, so downstream coordinate types (TFVD,
SLEVE, future) can inherit AMG status without each adding ~40 method
definitions. The file `terrain_amg_operators.jl` would then collapse to a
single supertype declaration on `TerrainFollowingVerticalDiscretization`.
