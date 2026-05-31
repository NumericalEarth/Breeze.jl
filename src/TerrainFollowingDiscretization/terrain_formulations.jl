#####
##### Terrain-decay formulations for TerrainFollowingVerticalDiscretization.
#####
##### Each formulation is the *generator* of the coordinate map
#####   z(x,y,ζ) = ζ + Σₙ hₙ(x,y) · bₙ(ζ)
##### supplying the terrain components hₙ (and their slopes) plus the decay
##### functions bₙ(ζ) and bₙ′(ζ). The grid operators (σⁿ, znode, ∂z∂x) call
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
#####   b(ζ) = 1 − ζ/z_top,   b′(ζ) = −1/z_top
#####

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

@inline _b_linear(ζ, z_top)  = 1 - ζ / z_top
@inline _b′_linear(z_top)    = -1 / z_top

# h interpolated to the (ℓx, ℓy) horizontal stagger. The `::Nothing` cases
# arise when one of the horizontal directions is Flat: znode/node may be
# called with `ℓy=nothing` (or `ℓx=nothing`) so the function still has to
# dispatch. Treat the Flat direction as Center (no interpolation in that
# direction since the grid is degenerate there).
@inline _h(i, j, grid, h, ::Center, ::Center)  = @inbounds h[i, j, 1]
@inline _h(i, j, grid, h, ::Face,   ::Center)  = ℑxᶠᵃᵃ(i, j, 1, grid, h)
@inline _h(i, j, grid, h, ::Center, ::Face)    = ℑyᵃᶠᵃ(i, j, 1, grid, h)
@inline _h(i, j, grid, h, ::Face,   ::Face)    = ℑxyᶠᶠᵃ(i, j, 1, grid, h)
@inline _h(i, j, grid, h, ::Center, ::Nothing) = @inbounds h[i, j, 1]
@inline _h(i, j, grid, h, ::Face,   ::Nothing) = ℑxᶠᵃᵃ(i, j, 1, grid, h)
@inline _h(i, j, grid, h, ::Nothing, ::Center) = @inbounds h[i, j, 1]
@inline _h(i, j, grid, h, ::Nothing, ::Face)   = ℑyᵃᶠᵃ(i, j, 1, grid, h)
@inline _h(i, j, grid, h, ::Nothing, ::Nothing) = @inbounds h[i, j, 1]

@inline function terrain_following_σ(i, j, k, grid, f::LinearDecay, ℓx, ℓy, ℓz)
    h = _h(i, j, grid, f.h, ℓx, ℓy)
    return 1 + h * _b′_linear(f.z_top)
end

@inline function terrain_following_Δz_surface(i, j, k, grid, f::LinearDecay, ℓx, ℓy, ℓz)
    ζ = rnode(k, grid, ℓz)
    h = _h(i, j, grid, f.h, ℓx, ℓy)
    return h * _b_linear(ζ, f.z_top)
end

@inline function terrain_following_∂z∂x(i, j, k, grid, f::LinearDecay, ℓz)
    ζ = rnode(k, grid, ℓz)
    @inbounds return f.∂x_h[i, j, 1] * _b_linear(ζ, f.z_top)
end

@inline function terrain_following_∂z∂y(i, j, k, grid, f::LinearDecay, ℓz)
    ζ = rnode(k, grid, ℓz)
    @inbounds return f.∂y_h[i, j, 1] * _b_linear(ζ, f.z_top)
end

#####
##### SLEVE — Schär et al. (2002). Large/small split, sinh decay.
#####   bₙ(ζ) = sinh((z_top−ζ)/sₙ)/sinh(z_top/sₙ)
#####   bₙ′(ζ) = −cosh((z_top−ζ)/sₙ)/(sₙ·sinh(z_top/sₙ))
#####

struct SLEVE{ZT, FT, H, SX, SY} <: AbstractTerrainFormulation
    z_top              :: ZT   # Nothing (skeleton) or FT (after allocation)
    large_scale_height :: FT   # s₁ (slow decay)
    small_scale_height :: FT   # s₂ (fast decay)
    h₁ :: H; h₂ :: H           # large/small terrain (Center, Center)
    ∂x_h₁ :: SX; ∂x_h₂ :: SX   # (Face, Center)
    ∂y_h₁ :: SY; ∂y_h₂ :: SY   # (Center, Face)
end

SLEVE(; large_scale_height, small_scale_height) =
    SLEVE(nothing, large_scale_height, small_scale_height,
          nothing, nothing, nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, f::SLEVE) =
    SLEVE(f.z_top, f.large_scale_height, f.small_scale_height,
          Adapt.adapt(to, f.h₁), Adapt.adapt(to, f.h₂),
          Adapt.adapt(to, f.∂x_h₁), Adapt.adapt(to, f.∂x_h₂),
          Adapt.adapt(to, f.∂y_h₁), Adapt.adapt(to, f.∂y_h₂))

Oceananigans.Architectures.on_architecture(arch, f::SLEVE) =
    SLEVE(f.z_top, f.large_scale_height, f.small_scale_height,
          Oceananigans.Architectures.on_architecture(arch, f.h₁),
          Oceananigans.Architectures.on_architecture(arch, f.h₂),
          Oceananigans.Architectures.on_architecture(arch, f.∂x_h₁),
          Oceananigans.Architectures.on_architecture(arch, f.∂x_h₂),
          Oceananigans.Architectures.on_architecture(arch, f.∂y_h₁),
          Oceananigans.Architectures.on_architecture(arch, f.∂y_h₂))

@inline _b_sleve(ζ, z_top, s)  = sinh((z_top - ζ) / s) / sinh(z_top / s)
@inline _b′_sleve(ζ, z_top, s) = -cosh((z_top - ζ) / s) / (s * sinh(z_top / s))

@inline function terrain_following_σ(i, j, k, grid, f::SLEVE, ℓx, ℓy, ℓz)
    ζ  = rnode(k, grid, ℓz)
    h₁ = _h(i, j, grid, f.h₁, ℓx, ℓy)
    h₂ = _h(i, j, grid, f.h₂, ℓx, ℓy)
    return 1 + h₁ * _b′_sleve(ζ, f.z_top, f.large_scale_height) +
               h₂ * _b′_sleve(ζ, f.z_top, f.small_scale_height)
end

@inline function terrain_following_Δz_surface(i, j, k, grid, f::SLEVE, ℓx, ℓy, ℓz)
    ζ  = rnode(k, grid, ℓz)
    h₁ = _h(i, j, grid, f.h₁, ℓx, ℓy)
    h₂ = _h(i, j, grid, f.h₂, ℓx, ℓy)
    return h₁ * _b_sleve(ζ, f.z_top, f.large_scale_height) +
           h₂ * _b_sleve(ζ, f.z_top, f.small_scale_height)
end

@inline function terrain_following_∂z∂x(i, j, k, grid, f::SLEVE, ℓz)
    ζ = rnode(k, grid, ℓz)
    @inbounds return f.∂x_h₁[i, j, 1] * _b_sleve(ζ, f.z_top, f.large_scale_height) +
                     f.∂x_h₂[i, j, 1] * _b_sleve(ζ, f.z_top, f.small_scale_height)
end

@inline function terrain_following_∂z∂y(i, j, k, grid, f::SLEVE, ℓz)
    ζ = rnode(k, grid, ℓz)
    @inbounds return f.∂y_h₁[i, j, 1] * _b_sleve(ζ, f.z_top, f.large_scale_height) +
                     f.∂y_h₂[i, j, 1] * _b_sleve(ζ, f.z_top, f.small_scale_height)
end
