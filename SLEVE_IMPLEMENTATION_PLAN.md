# SLEVE terrain-following coordinate — implementation plan

## 1. Motivation

The Schär mountain-wave validation shows a spurious **low vertical wavenumber,
high horizontal wavenumber** signal in Breeze's `w`, visible in any horizontal
cross-section between z ≈ 2–20 km, that CM1 does not have. It is the canonical
failure the Schär test was designed to expose.

**Diagnosis (data-backed).** Breeze uses **Basic Terrain-Following (BTF)** — the
Gal-Chen & Somerville (1975) coordinate with terrain influence decaying
*linearly* to the model top, `z = ζ + h(x)·(1 − ζ/z_top)`. The Schär ridges
(λ = 4 km) are short-wavelength features in the **evanescent** regime
(λ < 2πU/N = 6.3 km), so the physical response must decay within ~1 km of the
surface. But the linear decay keeps the ridge corrugation imprinted on the
coordinate surfaces aloft (at z = 10 km it is still 67% of its surface
amplitude), and the metric/PGF terms on those corrugated surfaces inject a
spurious, vertically-coherent, high-`k_x` `w`.

Measured sub-6.3 km fraction of `w` variance vs height:

| z (km) | Breeze (BTF) | CM1 |
|---|---|---|
| 2  | 10.8% | 19.9% |
| 6  | 5.1%  | 7.1%  |
| 10 | 6.4%  | 1.7%  |
| 14 | 3.2%  | 0.6%  |

CM1 evanesces the short scales (20% → <1% by 14 km); Breeze retains them.

**Fix.** Implement **SLEVE** (Smooth LEvel VErtical coordinate, Schär et al.
2002): split the terrain into large- and small-scale parts that decay with
different scale heights, so the small-scale ridge corrugation vanishes within a
few km, leaving smooth coordinate surfaces aloft.

## 2. Background — BTF vs SLEVE

Both are the same coordinate family,
```
z(x, y, ζ) = ζ + Σₙ hₙ(x, y) · bₙ(ζ),     b(0) = 1,  b(z_top) = 0
```
differing only in the decay functions and number of terrain components:

- **LinearDecay (BTF, Gal-Chen & Somerville 1975):** one component,
  `b(ζ) = 1 − ζ/z_top`.
- **SLEVE (Schär et al. 2002):** two components `h = h₁ + h₂` (large + small
  scale, via a horizontal smoothing split), with
  `bₙ(ζ) = sinh((z_top − ζ)/sₙ) / sinh(z_top/sₙ)`, where `s₁` (large-scale
  decay height) is large and `s₂` (small-scale) is short.

Derived metrics (both formulations):
```
Jacobian:  σ(ζ) = ∂z/∂ζ = 1 + Σₙ hₙ · bₙ′(ζ),   bₙ′(ζ) = −cosh((z_top−ζ)/sₙ)/(sₙ·sinh(z_top/sₙ))
slope:     (∂z/∂x)_ζ    = Σₙ ∂x_hₙ · bₙ(ζ)
```

## 3. Design

### 3.1 One discretization type, formulation as a field

```julia
struct TerrainFollowingVerticalDiscretization{C,D,E,F,FM} <: AbstractVerticalCoordinate
    cᵃᵃᶠ :: C            # reference ζ faces     ┐ reference coordinate (1-D),
    cᵃᵃᶜ :: D            # reference ζ centers   │ same role as in every
    Δᵃᵃᶠ :: E            # reference Δ (face)     │ vertical coordinate
    Δᵃᵃᶜ :: F            # reference Δ (center)   ┘
    formulation :: FM    # LinearDecay() | SLEVE(...) — the generator
end
```

**Key principle: store the *generator*, derive σ and the slope.** Unlike
`MutableVerticalDiscretization` (which stores σ as fields and has no concept of
a horizontal slope), this type stores the terrain components + decay law and
*computes* σ and `(∂z/∂x)` in the operators. This is what (a) makes it a genuinely
distinct type rather than `MutableVerticalDiscretization` + a tag, (b) keeps σ
and the slope derived from the *same* decay function so they cannot drift out
of consistency, and (c) collapses BTF and SLEVE into two `formulation`s of one
type.

### 3.2 Formulations (the generators)

```julia
struct LinearDecay{T2,T1}
    h     :: T2          # terrain                (2-D, [i,j])
    ∂x_h  :: T2; ∂y_h :: T2                       # slopes (2-D)
    b     :: T1          # decay profile  1 − ζ/z_top   (1-D, [k])
    b′    :: T1          # decay derivative  −1/z_top    (1-D, [k])
end

struct SLEVE{FT,T2,T1}
    large_scale_height :: FT     # s₁  (slow decay)
    small_scale_height :: FT     # s₂  (fast decay)
    h₁ :: T2; h₂ :: T2                            # large/small terrain (2-D)
    ∂x_h₁ :: T2; ∂x_h₂ :: T2; ∂y_h₁ :: T2; ∂y_h₂ :: T2   # slopes (2-D)
    b₁ :: T1; b₂ :: T1                            # decay profiles  bₙ(ζ)  (1-D, [k])
    b₁′ :: T1; b₂′ :: T1                          # decay derivatives      (1-D, [k])
end

# User-facing skeleton constructors (terrain fields = nothing until materialized):
LinearDecay() = LinearDecay(nothing, nothing, nothing, nothing, nothing)
SLEVE(; large_scale_height, small_scale_height) =
    SLEVE(large_scale_height, small_scale_height, ntuple(_->nothing, 10)...)
```

The decay profiles `bₙ(ζ_k)` are **horizontally uniform → 1-D arrays in `k`**.
So σ and slope are a 2-D × 1-D product per point — no 3-D σ storage, no `sinh`
in the hot loop.

### 3.3 Operators (compute from the generator)

These dispatch on the grid's vertical-coordinate type. We confirmed
Oceananigans already funnels `zspacing`/`znode`/`Vᶜᶜᶜ` through a single
`σⁿ(i,j,k,grid,ℓx,ℓy,ℓz)` accessor, so only `σⁿ` (4 stagger methods) and a new
`∂z∂x`/`∂z∂y` need defining:

```julia
const TFVDGrid = AbstractGrid{<:Any,<:Any,<:Any,<:Any,<:TerrainFollowingVerticalDiscretization}

@inline σⁿ(i,j,k, grid::TFVDGrid, ::C,::C, ℓz) = _sigma(i,j,k, grid, grid.z.formulation, C(),C(),ℓz)
# ... ::F,::C / ::C,::F / ::F,::F

@inline function _sigma(i,j,k, grid, f::SLEVE, ℓx,ℓy,ℓz)
    h₁ = terrain_at(i,j, grid, f.h₁, ℓx,ℓy);  h₂ = terrain_at(i,j, grid, f.h₂, ℓx,ℓy)
    @inbounds return 1 + h₁*f.b₁′[vk(k,ℓz)] + h₂*f.b₂′[vk(k,ℓz)]
end
@inline function _sigma(i,j,k, grid, f::LinearDecay, ℓx,ℓy,ℓz)
    h = terrain_at(i,j, grid, f.h, ℓx,ℓy)
    @inbounds return 1 + h*f.b′[vk(k,ℓz)]
end

@inline ∂z∂x(i,j,k, grid::TFVDGrid, ℓz) = _slope_x(i,j,k, grid, grid.z.formulation, ℓz)
@inline _slope_x(i,j,k,grid,f::SLEVE,ℓz)      = @inbounds f.∂x_h₁[i,j]*f.b₁[vk(k,ℓz)] + f.∂x_h₂[i,j]*f.b₂[vk(k,ℓz)]
@inline _slope_x(i,j,k,grid,f::LinearDecay,ℓz)= @inbounds f.∂x_h[i,j]*f.b[vk(k,ℓz)]
# ∂z∂y analogous
```
(`vk` selects the face/center vertical index, `terrain_at` interpolates the 2-D
terrain to the requested horizontal stagger.)

### 3.4 `materialize_terrain!` — the fill step

Topography needs the assembled horizontal grid (`x,y`), which does not exist
when the vertical coordinate is constructed — so a post-construction
materialize step is fundamental. It replaces today's `follow_terrain!`
(which mutated `grid.z` *and* returned a separate Dynamics-side `TerrainMetrics`).
The new step does one job: fill the coordinate's own formulation fields
in place. Skeleton → materialize, matching Breeze's standard pattern.

```julia
function materialize_terrain!(grid, topography)
    f = grid.z.formulation
    h = CenterField(grid, indices=(:,:,1)); set!(h, topography); fill_halo_regions!(h)
    materialize_formulation!(f, h, grid)   # dispatch on LinearDecay / SLEVE
    return grid
end

# LinearDecay: store h, its slopes, and b(ζ_k)=1−ζ/z_top, b′=−1/z_top.
# SLEVE:       h₁ = smooth(h) (horizontal low-pass), h₂ = h − h₁;
#              store h₁,h₂, their slopes, and bₙ(ζ_k), bₙ′(ζ_k) from sinh.
#              assert σ > 0 everywhere (no grid folding) for the given terrain + sₙ.
```

### 3.5 `TerrainFollowingGrid` — convenience constructor

Hide the two phases so users never call the fill step manually:
```julia
function TerrainFollowingGrid(arch=CPU(); size, x, y, z_faces,
                              topography, formulation = SLEVE(...))
    z    = TerrainFollowingVerticalDiscretization(z_faces; formulation)  # skeleton
    grid = RectilinearGrid(arch; size, x, y, z, topology=(Periodic,Flat,Bounded))
    materialize_terrain!(grid, topography)                              # fill once x,y exist
    return grid
end
```

## 4. Migration — remove the Dynamics-side split

Today the terrain geometry is split: σ/η in `grid.z` (`MutableVerticalDiscretization`),
but the slope `(∂z/∂x)` and the decay law in `CompressibleDynamics.terrain_metrics`
(`TerrainMetrics`), with the decay law encoded twice. After this change:

- `CompressibleDynamics` no longer holds `terrain_metrics`; the geometry lives
  entirely in `grid.z`.
- The four hardcoded `(1 − ζ/z_top)` sites in `terrain_compressible_physics.jl`
  (≈ lines 130–134, 341–350) and `terrain_metrics.jl:terrain_slope_x/y` call the
  grid operator `∂z∂x(i,j,k,grid,ℓz)` instead — coordinate-agnostic.
- `_set_btf_sigma!` and the old `follow_terrain!`/`TerrainMetrics` are removed;
  `BasicTerrainFollowing` → `LinearDecay` formulation.

The Dynamics becomes formulation-agnostic: flat, LinearDecay, and SLEVE all flow
through the same operator calls.

## 5. Operator-dispatch surface — verify before committing

The `σⁿ` accessor is the intended single chokepoint for the vertical metric.
**Spike first:** grep Oceananigans + Breeze for any code that reads
`grid.z.σᶜᶜⁿ` (or `.ηⁿ`, `.σᶜᶜ⁻`, `.∂t_σ`) *directly*, bypassing `σⁿ`/`σ⁻`/`∂t_σ`.
If `σⁿ` is the only consumer, the operator cost is ~4 `σⁿ` methods + `∂z∂x/∂z∂y`.
If not, each direct reader needs a dispatch path. This determines the true blast
radius and must be settled before implementation.

## 6. Validation

1. **Discrete metric identity / rest state** — a hydrostatic atmosphere at rest
   on a SLEVE grid must produce no spurious `w` (extend
   `terrain_hydrostatic_rest_acceptance` to a SLEVE grid). This is the core
   correctness gate.
2. **The diagnostic from this investigation** — re-run Schär 6 h on a SLEVE grid
   and confirm the sub-6.3 km `w` energy now **decays** with height
   (Breeze ~3–6% → CM1-like <1% by 14 km) and the columnar high-`k_x` signal is
   gone from 2–20 km cross-sections.
3. **Below-sponge `w` L2 vs CM1** should drop materially — this is the real
   remaining gate gap (currently ~1.1 even with a working sponge).
4. **Regression** — `LinearDecay` must reproduce the existing BTF results
   bit-for-bit (same coordinate, refactored code path).

## 7. Risks / open questions

- **σⁿ chokepoint completeness** (§5) — the one thing to verify first.
- **Grid folding** — SLEVE requires `σ > 0` everywhere; add a constructor check
  for the given terrain and `s₁, s₂` (Schär's constraint `s₂ < s₁` and small
  enough `h₂` relative to `s₂`).
- **Oceananigans coupling** — `TerrainFollowingVerticalDiscretization <:
  AbstractVerticalCoordinate` lives in Breeze; confirm Oceananigans' grid
  constructors accept a Breeze-defined coordinate (or whether the type must be
  upstreamed / the grid built via an extension point).
- **`h₁/h₂` split operator** — choose the horizontal smoother (e.g. a few passes
  of a 1-2-1 filter, or a spectral/Gaussian low-pass) and the cutoff that places
  the 4-km ridges in `h₂`.

## 8. Phasing

1. **Spike (≈1 day):** §5 chokepoint grep + a throwaway `LinearDecay`
   `TerrainFollowingVerticalDiscretization` that reproduces BTF through the new
   operators (validates the dispatch path with zero physics change).
2. **LinearDecay path (≈1 day):** full type + `materialize_terrain!` +
   `TerrainFollowingGrid`; migrate the Dynamics to the operator; pass the BTF
   regression (§6.4).
3. **SLEVE formulation (≈1–2 days):** decay profiles, `h₁/h₂` split, folding
   check; pass the rest-state identity (§6.1).
4. **Validate (≈1 day):** Schär 6 h on SLEVE; confirm the sub-6.3 km decay and
   below-sponge L2 improvement vs CM1 (§6.2–6.3).

Keep `LinearDecay` as the default `formulation` so existing setups are unchanged;
SLEVE is opt-in.
