#####
##### Terrain-decay formulations for TerrainFollowingVerticalDiscretization.
#####
##### Each formulation is the *generator* of the coordinate map
#####   z(x,y,r) = r + Σₙ hₙ(x,y) · bₙ(r)
##### supplying the terrain components hₙ (and their slopes) plus the decay
##### functions bₙ(r) and bₙ′(r). The grid operators (σⁿ, znode, ∂z∂x) call
##### the methods below, so σ = 1 + Σ hₙ bₙ′ and the slope Σ ∂ₓhₙ bₙ are
##### derived from the *same* bₙ — they cannot drift apart.
#####
##### Skeleton instances (terrain fields = `nothing`) are built before the grid
##### exists; `materialize_terrain!` fills them once the horizontal nodes are
##### known.
#####

abstract type AbstractTerrainFormulation end

#####
##### LinearDecay — Gal-Chen & Somerville (1975). One component, linear decay.
#####   b(r) = 1 − r/z_top,   b′(r) = −1/z_top
#####

"""
$(TYPEDEF)

Gal-Chen & Somerville (1975) terrain-following formulation: a single decay
basis ``b(r) = 1 - r/z_{top}`` that linearly attenuates the terrain from the
surface to the model top.
"""
struct LinearDecay{FT, H, SX, SY} <: AbstractTerrainFormulation
    z_top :: FT
    h     :: H      # terrain height (Center, Center)
    ∂x_h  :: SX     # ∂h/∂x (Face, Center)
    ∂y_h  :: SY     # ∂h/∂y (Center, Face)
end

LinearDecay() = LinearDecay(nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, f::LinearDecay) =
    LinearDecay(f.z_top, Adapt.adapt(to, f.h), Adapt.adapt(to, f.∂x_h), Adapt.adapt(to, f.∂y_h))

Oceananigans.Architectures.on_architecture(arch, f::LinearDecay) =
    LinearDecay(f.z_top,
                Oceananigans.Architectures.on_architecture(arch, f.h),
                Oceananigans.Architectures.on_architecture(arch, f.∂x_h),
                Oceananigans.Architectures.on_architecture(arch, f.∂y_h))

@inline b_linear(r, z_top)  = 1 - r / z_top
@inline b′_linear(z_top)    = -1 / z_top

# h interpolated to the (ℓx, ℓy) horizontal stagger. The `::Nothing` cases
# arise when one of the horizontal directions is Flat: znode/node may be
# called with `ℓy=nothing` (or `ℓx=nothing`) so the function still has to
# dispatch. Treat the Flat direction as Center (no interpolation in that
# direction since the grid is degenerate there).
@inline terrain_at_stagger(i, j, grid, h, ::Center, ::Center)  = @inbounds h[i, j, 1]
@inline terrain_at_stagger(i, j, grid, h, ::Face,   ::Center)  = ℑxᶠᵃᵃ(i, j, 1, grid, h)
@inline terrain_at_stagger(i, j, grid, h, ::Center, ::Face)    = ℑyᵃᶠᵃ(i, j, 1, grid, h)
@inline terrain_at_stagger(i, j, grid, h, ::Face,   ::Face)    = ℑxyᶠᶠᵃ(i, j, 1, grid, h)
@inline terrain_at_stagger(i, j, grid, h, ::Center, ::Nothing) = @inbounds h[i, j, 1]
@inline terrain_at_stagger(i, j, grid, h, ::Face,   ::Nothing) = ℑxᶠᵃᵃ(i, j, 1, grid, h)
@inline terrain_at_stagger(i, j, grid, h, ::Nothing, ::Center) = @inbounds h[i, j, 1]
@inline terrain_at_stagger(i, j, grid, h, ::Nothing, ::Face)   = ℑyᵃᶠᵃ(i, j, 1, grid, h)
@inline terrain_at_stagger(i, j, grid, h, ::Nothing, ::Nothing) = @inbounds h[i, j, 1]

@inline function terrain_following_σ(i, j, k, grid, f::LinearDecay, ℓx, ℓy, ℓz)
    h = terrain_at_stagger(i, j, grid, f.h, ℓx, ℓy)
    return 1 + h * b′_linear(f.z_top)
end

@inline function terrain_following_Δz_surface(i, j, k, grid, f::LinearDecay, ℓx, ℓy, ℓz)
    r = rnode(k, grid, ℓz)
    h = terrain_at_stagger(i, j, grid, f.h, ℓx, ℓy)
    return h * b_linear(r, f.z_top)
end

@inline function terrain_following_∂z∂x(i, j, k, grid, f::LinearDecay, ℓz)
    r = rnode(k, grid, ℓz)
    @inbounds return f.∂x_h[i, j, 1] * b_linear(r, f.z_top)
end

@inline function terrain_following_∂z∂y(i, j, k, grid, f::LinearDecay, ℓz)
    r = rnode(k, grid, ℓz)
    @inbounds return f.∂y_h[i, j, 1] * b_linear(r, f.z_top)
end

#####
##### TwoLevelDecay — Schär et al. (2002). Large/small split, sinh decay.
#####   bₙ(r) = sinh((z_top−r)/sₙ)/sinh(z_top/sₙ)
#####   bₙ′(r) = −cosh((z_top−r)/sₙ)/(sₙ·sinh(z_top/sₙ))
#####

"""
$(TYPEDEF)

Schär et al. (2002) "Smooth LEvel VErtical" (SLEVE) terrain-following
formulation. Splits the terrain into a smoothed large-scale component ``h_1``
(decay length `large_scale_height`) and the residual small-scale component
``h_2`` (decay length `small_scale_height`). Each is attenuated with a
hyperbolic-sine basis ``b_n(r) = \\sinh((z_{top}-r)/s_n) / \\sinh(z_{top}/s_n)``,
so the small-scale features decay quickly while the large-scale envelope is
preserved aloft.

Constructed via the kwarg form `TwoLevelDecay(; large_scale_height,
small_scale_height)`.
"""
struct TwoLevelDecay{ZT, FT, H, SX, SY, B} <: AbstractTerrainFormulation
    z_top              :: ZT   # Nothing (skeleton) or FT (after allocation)
    large_scale_height :: FT   # s₁ (slow decay)
    small_scale_height :: FT   # s₂ (fast decay)
    h₁ :: H; h₂ :: H           # large/small terrain (Center, Center)
    ∂x_h₁ :: SX; ∂x_h₂ :: SX   # (Face, Center)
    ∂y_h₁ :: SY; ∂y_h₂ :: SY   # (Center, Face)
    basis :: B                 # TwoLevelBasis (materialized) or Nothing (skeleton)
end

TwoLevelDecay(; large_scale_height, small_scale_height) =
    TwoLevelDecay(nothing, large_scale_height, small_scale_height,
          nothing, nothing, nothing, nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, f::TwoLevelDecay) =
    TwoLevelDecay(f.z_top, f.large_scale_height, f.small_scale_height,
          Adapt.adapt(to, f.h₁), Adapt.adapt(to, f.h₂),
          Adapt.adapt(to, f.∂x_h₁), Adapt.adapt(to, f.∂x_h₂),
          Adapt.adapt(to, f.∂y_h₁), Adapt.adapt(to, f.∂y_h₂),
          Adapt.adapt(to, f.basis))

Oceananigans.Architectures.on_architecture(arch, f::TwoLevelDecay) =
    TwoLevelDecay(f.z_top, f.large_scale_height, f.small_scale_height,
          Oceananigans.Architectures.on_architecture(arch, f.h₁),
          Oceananigans.Architectures.on_architecture(arch, f.h₂),
          Oceananigans.Architectures.on_architecture(arch, f.∂x_h₁),
          Oceananigans.Architectures.on_architecture(arch, f.∂x_h₂),
          Oceananigans.Architectures.on_architecture(arch, f.∂y_h₁),
          Oceananigans.Architectures.on_architecture(arch, f.∂y_h₂),
          Oceananigans.Architectures.on_architecture(arch, f.basis))

# The normalization sinh(z_top / s) depends only on the static formulation
# parameters. The four-argument forms accept it precomputed so a caller that
# evaluates the basis repeatedly at the same s — the Newton inversion in
# `lagrangian_particles.jl` — can hoist it out of its loop. The three-argument
# forms below are defined in terms of them, so there is one source of truth.
@inline b_two_level_normalization(z_top, s) = sinh(z_top / s)

@inline b_two_level(r, z_top, s, normalization)  = sinh((z_top - r) / s) / normalization
@inline b′_two_level(r, z_top, s, normalization) = -cosh((z_top - r) / s) / (s * normalization)

@inline b_two_level(r, z_top, s)  = b_two_level(r, z_top, s, b_two_level_normalization(z_top, s))
@inline b′_two_level(r, z_top, s) = b′_two_level(r, z_top, s, b_two_level_normalization(z_top, s))

#####
##### Precomputed SLEVE decay bases.
#####
##### bₙ(r) and bₙ′(r) depend on the reference coordinate r alone (plus the static
##### formulation parameters), but σ → Δz and znode evaluate them on every operator
##### access. For TwoLevelDecay each evaluation is a sinh/cosh pair, which is
##### catastrophic on the GPU (≈20–30× the LinearDecay per-step cost). We
##### materialize bₙ, bₙ′ once at the Center and Face z-locations (1D in z, indexed
##### [1, 1, k] over the same halo'd k-range as rnode) so the hot path is a memory
##### read instead of a transcendental.
#####

struct TwoLevelBasis{A}
    b₁ᶜ :: A; b₁ᶠ :: A     # b₁(r) at Center, Face   (large scale, s₁)
    b₂ᶜ :: A; b₂ᶠ :: A     # b₂(r) at Center, Face   (small scale, s₂)
    ∂b₁ᶜ :: A; ∂b₁ᶠ :: A   # b₁′(r) at Center, Face
    ∂b₂ᶜ :: A; ∂b₂ᶠ :: A   # b₂′(r) at Center, Face
end

Adapt.adapt_structure(to, c::TwoLevelBasis) =
    TwoLevelBasis(Adapt.adapt(to, c.b₁ᶜ), Adapt.adapt(to, c.b₁ᶠ),
                  Adapt.adapt(to, c.b₂ᶜ), Adapt.adapt(to, c.b₂ᶠ),
                  Adapt.adapt(to, c.∂b₁ᶜ), Adapt.adapt(to, c.∂b₁ᶠ),
                  Adapt.adapt(to, c.∂b₂ᶜ), Adapt.adapt(to, c.∂b₂ᶠ))

Oceananigans.Architectures.on_architecture(arch, c::TwoLevelBasis) =
    TwoLevelBasis(Oceananigans.Architectures.on_architecture(arch, c.b₁ᶜ),
                  Oceananigans.Architectures.on_architecture(arch, c.b₁ᶠ),
                  Oceananigans.Architectures.on_architecture(arch, c.b₂ᶜ),
                  Oceananigans.Architectures.on_architecture(arch, c.b₂ᶠ),
                  Oceananigans.Architectures.on_architecture(arch, c.∂b₁ᶜ),
                  Oceananigans.Architectures.on_architecture(arch, c.∂b₁ᶠ),
                  Oceananigans.Architectures.on_architecture(arch, c.∂b₂ᶜ),
                  Oceananigans.Architectures.on_architecture(arch, c.∂b₂ᶠ))

# Read the precomputed basis at vertical index k and z-location ℓz (Center/Face).
@inline sleve_b₁(c::TwoLevelBasis, k, ::Center)  = @inbounds c.b₁ᶜ[1, 1, k]
@inline sleve_b₁(c::TwoLevelBasis, k, ::Face)    = @inbounds c.b₁ᶠ[1, 1, k]
@inline sleve_b₂(c::TwoLevelBasis, k, ::Center)  = @inbounds c.b₂ᶜ[1, 1, k]
@inline sleve_b₂(c::TwoLevelBasis, k, ::Face)    = @inbounds c.b₂ᶠ[1, 1, k]
@inline sleve_∂b₁(c::TwoLevelBasis, k, ::Center) = @inbounds c.∂b₁ᶜ[1, 1, k]
@inline sleve_∂b₁(c::TwoLevelBasis, k, ::Face)   = @inbounds c.∂b₁ᶠ[1, 1, k]
@inline sleve_∂b₂(c::TwoLevelBasis, k, ::Center) = @inbounds c.∂b₂ᶜ[1, 1, k]
@inline sleve_∂b₂(c::TwoLevelBasis, k, ::Face)   = @inbounds c.∂b₂ᶠ[1, 1, k]

@inline function terrain_following_σ(i, j, k, grid, f::TwoLevelDecay, ℓx, ℓy, ℓz)
    h₁ = terrain_at_stagger(i, j, grid, f.h₁, ℓx, ℓy)
    h₂ = terrain_at_stagger(i, j, grid, f.h₂, ℓx, ℓy)
    return 1 + h₁ * sleve_∂b₁(f.basis, k, ℓz) + h₂ * sleve_∂b₂(f.basis, k, ℓz)
end

@inline function terrain_following_Δz_surface(i, j, k, grid, f::TwoLevelDecay, ℓx, ℓy, ℓz)
    h₁ = terrain_at_stagger(i, j, grid, f.h₁, ℓx, ℓy)
    h₂ = terrain_at_stagger(i, j, grid, f.h₂, ℓx, ℓy)
    return h₁ * sleve_b₁(f.basis, k, ℓz) + h₂ * sleve_b₂(f.basis, k, ℓz)
end

@inline function terrain_following_∂z∂x(i, j, k, grid, f::TwoLevelDecay, ℓz)
    @inbounds return f.∂x_h₁[i, j, 1] * sleve_b₁(f.basis, k, ℓz) +
                     f.∂x_h₂[i, j, 1] * sleve_b₂(f.basis, k, ℓz)
end

@inline function terrain_following_∂z∂y(i, j, k, grid, f::TwoLevelDecay, ℓz)
    @inbounds return f.∂y_h₁[i, j, 1] * sleve_b₁(f.basis, k, ℓz) +
                     f.∂y_h₂[i, j, 1] * sleve_b₂(f.basis, k, ℓz)
end
