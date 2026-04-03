# MPAS to Breeze/Oceananigans operator translation

## Grid staggering (identical: Arakawa C-grid)

| Variable | MPAS location | Oceananigans location | Field type |
|----------|---------------|----------------------|------------|
| u (zonal momentum) | edge-normal | (Face, Center, Center) | `XFaceField` |
| v (meridional momentum) | edge-normal | (Center, Face, Center) | `YFaceField` |
| w (vertical momentum) | cell center, half-level | (Center, Center, Face) | `ZFaceField` |
| ρ, ρθ, p, π, θ | cell center | (Center, Center, Center) | `CenterField` |

MPAS uses edge-normal velocity on an unstructured Voronoi mesh.
Breeze uses component velocities (u, v) on a structured lat-lon grid.
The C-grid staggering is equivalent.

## Divergence operators

### Full 3D divergence

**MPAS** (in acoustic step, for density):
```fortran
! Horizontal flux divergence of ru_p
do i = 1, nEdgesOnCell(iCell)
    flux = sign * dts * dvEdge * ru_p(k,iEdge) * invAreaCell
    rs(k) = rs(k) - flux
end do
! Vertical divergence of rw_p
rho_pp(k) = rs(k) - cofrz(k) * (rw_p(k+1) - rw_p(k))
```

**Breeze/Oceananigans**:
```julia
divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
# = V⁻¹ [δx(Ax·ρu) + δy(Ay·ρv) + δz(Az·ρw)]
```

Oceananigans' `divᶜᶜᶜ` is area-weighted and handles LatitudeLongitudeGrid
automatically through the metric operators `Axᶠᶜᶜ`, `Ayᶜᶠᶜ`, `Azᶜᶜᶠ`, `Vᶜᶜᶜ`.

### Horizontal-only divergence

**MPAS**: Same loop over cell edges, omitting vertical.

**Breeze/Oceananigans**:
```julia
div_xyᶜᶜᶜ(i, j, k, grid, u, v)
# = V⁻¹ [δx(Ax·u) + δy(Ay·v)]
```

Note: `V = Az·Δz` so `div_xy = (1/Az·Δz)[δx(Ax·u) + δy(Ay·v)]`. This differs
from a pure 2D divergence `(1/Az)[δx(Ax·u) + δy(Ay·v)]` by a factor of `1/Δz`.
In the acoustic step, use the area-weighted form directly (as in the existing
`_compute_π′_forcing!` kernel) rather than calling `div_xyᶜᶜᶜ`.

### Vertical divergence (manual)

**MPAS**: `cofrz(k) * (rw_p(k+1) - rw_p(k))` where `cofrz = dtseps * rdzw`

**Breeze/Oceananigans**: No dedicated operator. Compute manually:
```julia
Az_top = Azᶜᶜᶠ(i, j, k+1, grid)
Az_bot = Azᶜᶜᶠ(i, j, k, grid)
V = Vᶜᶜᶜ(i, j, k, grid)
vert_div = (Az_top * ρw[i,j,k+1] - Az_bot * ρw[i,j,k]) / V
```

Or simpler for uniform horizontal area:
```julia
vert_div = (ρw[i,j,k+1] - ρw[i,j,k]) / Δzᶜᶜᶜ(i,j,k,grid)
```

## Pressure gradient operators

### Horizontal pressure gradient at u-face

**MPAS** (from `rtheta_pp`):
```fortran
pgrad = c2 * 0.5*(exner(k,cell1)+exner(k,cell2)) * (rtheta_pp(k,cell2)-rtheta_pp(k,cell1)) &
        * invDcEdge(iEdge) / (0.5*(zz(k,cell2)+zz(k,cell1)))
```

**Breeze** (existing, from Exner perturbation):
```julia
∂x_π′ = δxTᶠᵃᵃ(i, j, k, grid, π′) / Δxᶠᶜᶜ(i, j, k, grid)
pgf_x = -cₚᵈ * θᵥᶠ * ∂x_π′
```

For MPAS-style substepping using `rtheta_pp` instead of `π'`:
```julia
## Horizontal PGF from ρθ perturbation (MPAS form)
Π_face = ℑxᶠᵃᵃ(i, j, k, grid, Π)  # Exner at u-face
∂x_ρθ_pp = δxᶠᵃᵃ(i, j, k, grid, ρθ_pp) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
pgf_x = -c2 * Π_face * ∂x_ρθ_pp * cqu  # c2 = cₚ·Rᵈ/cᵥ
```

### Horizontal pressure gradient at v-face

Same pattern with y-operators:
```julia
Π_face = ℑyᵃᶠᵃ(i, j, k, grid, Π)
∂y_ρθ_pp = δyᵃᶠᵃ(i, j, k, grid, ρθ_pp) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
pgf_y = -c2 * Π_face * ∂y_ρθ_pp * cqv
```

### Vertical pressure gradient at w-face

**MPAS** (`cofwz` term in w equation):
```fortran
cofwz(k) * (zz(k)*ts(k) - zz(k-1)*ts(k-1))
```
where `cofwz = dtseps * c2 * zz_face * rdzu * cqw * pi_face`

**Breeze** (no terrain, `zz = 1`):
```julia
## Vertical PGF from ρθ perturbation
Δzᶠ = Δzᶜᶜᶠ(i, j, k, grid)
Π_face = ℑzᵃᵃᶠ(i, j, k, grid, Π)
δz_ρθ_pp = (ρθ_pp[i,j,k] - ρθ_pp[i,j,k-1]) / Δzᶠ
pgf_z = -c2 * Π_face * δz_ρθ_pp * cqw
```

## Interpolation operators

| MPAS | Breeze/Oceananigans | Notes |
|------|---------------------|-------|
| `fzm(k)*f(k) + fzp(k)*f(k-1)` | `ℑzᵃᵃᶠ(i,j,k,grid,f)` | Center→face (uniform: simple average) |
| `0.5*(f(cell1)+f(cell2))` | `ℑxᶠᵃᵃ(i,j,k,grid,f)` | Center→x-face |
| — | `ℑyᵃᶠᵃ(i,j,k,grid,f)` | Center→y-face |
| `fzm(k)*f(k) + fzp(k)*f(k-1)` at w-level | `ℑzᵃᵃᶠ(i,j,k,grid,f)` | For non-uniform Δz, `fzm/fzp` are asymmetric weights. Oceananigans uses simple average (correct for uniform Δz). |

**Note on `fzm`/`fzp`**: MPAS uses `fzm(k) = dzw(k-1)/(dzw(k-1)+dzw(k))` and
`fzp(k) = 1 - fzm(k)` for non-uniform vertical spacing. Oceananigans' `ℑzᵃᵃᶠ`
uses simple averaging `0.5*(f[k-1]+f[k])`. For uniform vertical grids these are
identical. For stretched grids, we'd need distance-weighted interpolation.

## MPAS coefficient → Breeze translation

All MPAS coefficients incorporate `dtseps = 0.5 * Δτ * (1 + ε)` where ε is the
off-centering parameter. In Breeze, factor this as `αΔτ` where `α = 0.5*(1+ε)`.

| MPAS coefficient | Physical meaning | Breeze equivalent |
|-----------------|------------------|-------------------|
| `cofrz(k)` = `dtseps * rdzw(k)` | ρ tendency from vertical ρw divergence | `αΔτ / Δzᶜᶜᶜ` |
| `coftz(k)` = `dtseps * θₘ_face(k)` | ρθ tendency from vertical ρw flux | `αΔτ * ℑzᵃᵃᶠ(θₘ)` |
| `cofwz(k)` = `dtseps * c2 * zz_face * rdzu * cqw * Π_face` | Vertical PGF in w equation from ρθ pert | `αΔτ * c2 * Π_face / Δzᶜᶜᶠ` (no terrain: zz=1, cqw=1 dry) |
| `cofwr(k)` = `0.5 * dtseps * g * zz_face` | Buoyancy in w equation from ρ pert | `αΔτ * 0.5 * g` (no terrain: zz=1) |
| `cofwt(k)` = `0.5 * dtseps * rcv * g * zz * ρ_base/(1+q) * Π/(ρθ * Π_base)` | EOS buoyancy correction from θ pert | `αΔτ * 0.5 * (Rᵈ/cᵥ) * g * ρ_base * Π / (ρθ_total * Π_base)` |
| `resm` = `(1-ε)/(1+ε)` | Old-time weight in off-centered scheme | Same formula |

## Tridiagonal system translation

**MPAS tridiagonal for rw_p** (at w-faces, k=2,...,Nz):

```
a_tri(k) · rw_p(k-1) + b_tri(k) · rw_p(k) + c_tri(k) · rw_p(k+1) = rhs(k)
```

where (without terrain, zz=1):
```
a_tri(k) = -cofwz(k) * coftz(k-1) * rdzw(k-1)
           + cofwr(k) * cofrz(k-1)
           - cofwt(k-1) * coftz(k-1) * rdzw(k-1)

b_tri(k) = 1
           + cofwz(k) * (coftz(k)*rdzw(k) + coftz(k)*rdzw(k-1))
           - coftz(k) * (cofwt(k)*rdzw(k) - cofwt(k-1)*rdzw(k-1))
           + cofwr(k) * (cofrz(k) - cofrz(k-1))

c_tri(k) = -cofwz(k) * coftz(k+1) * rdzw(k)
           - cofwr(k) * cofrz(k)
           + cofwt(k) * coftz(k+1) * rdzw(k)
```

**Breeze translation** (using Oceananigans operators in the build kernel):
```julia
## At w-face k:
Δzᶜ_above = Δzᶜᶜᶜ(i, j, k, grid)      # cell height above face
Δzᶜ_below = Δzᶜᶜᶜ(i, j, k-1, grid)    # cell height below face
rdzw_above = 1 / Δzᶜ_above
rdzw_below = 1 / Δzᶜ_below

## Read precomputed coefficients at cell centers k and k-1
## (cofwz, cofwr, cofwt at the face; coftz, cofrz at cells)
```

The tridiagonal is built per-column and solved with `BatchedTridiagonalSolver`.
Note: Oceananigans' solver uses **shifted convention**: `lower[k]` is the
coefficient of `x[k]` in row `k+1`, not `x[k-1]` in row `k`.

## Key implementation notes

### 1. Use velocity form for acoustic loop, not momentum form

MPAS uses `ru_p = ρd·u·zz` (coupled momentum perturbation). For Breeze, we can
either work with velocity perturbations (u_p, v_p, w_p) like WRF/CM1, or coupled
momentum (ρu_p, ρv_p, ρw_p). The velocity form is simpler and matches CM1.

### 2. Perturbation variables reset each RK stage

Following MPAS: at the start of each RK stage, set all acoustic perturbation
variables to zero (`u_p = 0, π'_p = 0, ρθ_pp = 0, ρ_pp = 0`). The perturbations
measure the acoustic response within one stage only.

### 3. Full 3D divergence in the pressure equation

The pressure/theta equation uses the FULL velocity divergence (horizontal from
new u,v + vertical from implicit w solve). Use the area-weighted form:
```julia
div = (Ax_east * u_east - Ax_west * u_west
     + Ay_north * v_north - Ay_south * v_south) / V
     + (w_top - w_bot) / Δz  # or area-weighted vertical
```

### 4. Terrain-following terms (future)

Without terrain: `zz = 1` everywhere, `zxu = 0` (no slope). All MPAS terms
involving `zz`, `zxu`, `zb_cell`, `zb3_cell` reduce to trivial values.
Terrain support requires these Jacobian terms from the coordinate transform.

### 5. Existing infrastructure to reuse

From `src/CompressibleEquations/acoustic_substepping.jl`:
- `ExnerReferenceState` with discrete Π₀ (exact hydrostatic balance)
- `AcousticSubstepper` struct (fields, solver, filtering)
- `_prepare_exner_cache!` (compute θᵥ, S, π')
- `_convert_slow_tendencies!` (momentum→velocity, compute buoyancy)
- `BatchedTridiagonalSolver` for the vertically implicit solve
- Polar filter (`add_polar_filter!`)
- Time-averaged velocities (`averaged_velocities`)
