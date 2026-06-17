#####
##### Terrain-aware horizontal derivatives on TerrainFollowingGrid grids
#####
##### Oceananigans applies the chain-rule horizontal-derivative correction
#####
#####   (∂ϕ/∂x)_z = (∂ϕ/∂x)_r − (∂z/∂x)_r · (∂ϕ/∂z)
#####
##### only on `AbstractMutableGrid` (alias `AMG`), which is keyed on the concrete
##### `MutableVerticalDiscretization`. Our `TerrainFollowingVerticalDiscretization`
##### is *not* a subtype of `MutableVerticalDiscretization`, so the AMG methods do
##### not catch and the substepper / advection silently fall back to a flat-grid
##### derivative — which is the source of the slow-blowup TFVD instability.
#####
##### Until `AbstractMutableGrid` is widened upstream to admit other mutable
##### coordinates (the "option 2" PR to Oceananigans), this file mirrors the AMG
##### `∂x_z*` / `∂y_z*` / chain-rule `∂x*` / `∂y*` / number-disambiguation methods
##### onto `TerrainFollowingGrid`. The implementations are verbatim copies of the AMG bodies in
##### `Oceananigans/.../mutable_immersed_grid.jl`: they use δx(znode)·Δx⁻¹ for
##### the metric slope so that the chain rule cancels discretely for z-only
##### fields (using the analytic ∂z∂x from the formulation here instead would
##### leave a non-cancelling roundoff).
#####

using Oceananigans.Operators: δxᶠᶜᶜ, δxᶜᶜᶜ, δxᶠᶜᶠ, δxᶜᶠᶜ, δxᶠᶠᶜ, δxᶜᶜᶠ,
                              δyᶜᶠᶜ, δyᶜᶜᶜ, δyᶜᶠᶠ, δyᶠᶜᶜ, δyᶠᶠᶜ, δyᶜᶜᶠ,
                              Δx⁻¹ᶠᶜᶜ, Δx⁻¹ᶜᶜᶜ, Δx⁻¹ᶠᶜᶠ, Δx⁻¹ᶜᶠᶜ, Δx⁻¹ᶠᶠᶜ, Δx⁻¹ᶜᶜᶠ,
                              Δy⁻¹ᶜᶠᶜ, Δy⁻¹ᶜᶜᶜ, Δy⁻¹ᶜᶠᶠ, Δy⁻¹ᶠᶜᶜ, Δy⁻¹ᶠᶠᶜ, Δy⁻¹ᶜᶜᶠ,
                              ∂zᶜᶜᶠ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ, ∂zᶠᶠᶠ, ∂zᶜᶜᶜ,
                              ℑxzᶠᵃᶜ, ℑxzᶜᵃᶜ, ℑxzᶠᵃᶠ,
                              ℑyzᵃᶠᶜ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶠ

const _C = Center
const _F = Face

#####
##### Coordinate slopes ∂z/∂x|_r and ∂z/∂y|_r at every stagger
#####
##### Use the discrete δx(znode)·Δx⁻¹ rather than the analytic ∂z∂x from the
##### formulation: that is what the chain-rule bodies below assume, and it
##### makes (∂z/∂x)_z = 0 hold to machine precision.
#####

@inline Oceananigans.Operators.∂x_zᶠᶜᶜ(i, j, k, grid::TerrainFollowingGrid) = δxᶠᶜᶜ(i, j, k, grid, znode, _C(), _C(), _C()) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline Oceananigans.Operators.∂x_zᶜᶜᶜ(i, j, k, grid::TerrainFollowingGrid) = δxᶜᶜᶜ(i, j, k, grid, znode, _F(), _C(), _C()) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline Oceananigans.Operators.∂x_zᶠᶜᶠ(i, j, k, grid::TerrainFollowingGrid) = δxᶠᶜᶠ(i, j, k, grid, znode, _C(), _C(), _F()) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline Oceananigans.Operators.∂x_zᶜᶠᶜ(i, j, k, grid::TerrainFollowingGrid) = δxᶜᶠᶜ(i, j, k, grid, znode, _F(), _F(), _C()) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline Oceananigans.Operators.∂x_zᶠᶠᶜ(i, j, k, grid::TerrainFollowingGrid) = δxᶠᶠᶜ(i, j, k, grid, znode, _C(), _F(), _C()) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
@inline Oceananigans.Operators.∂x_zᶜᶜᶠ(i, j, k, grid::TerrainFollowingGrid) = δxᶜᶜᶠ(i, j, k, grid, znode, _F(), _C(), _F()) * Δx⁻¹ᶜᶜᶠ(i, j, k, grid)

@inline Oceananigans.Operators.∂y_zᶜᶠᶜ(i, j, k, grid::TerrainFollowingGrid) = δyᶜᶠᶜ(i, j, k, grid, znode, _C(), _C(), _C()) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline Oceananigans.Operators.∂y_zᶜᶜᶜ(i, j, k, grid::TerrainFollowingGrid) = δyᶜᶜᶜ(i, j, k, grid, znode, _C(), _F(), _C()) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline Oceananigans.Operators.∂y_zᶜᶠᶠ(i, j, k, grid::TerrainFollowingGrid) = δyᶜᶠᶠ(i, j, k, grid, znode, _C(), _C(), _F()) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
@inline Oceananigans.Operators.∂y_zᶠᶜᶜ(i, j, k, grid::TerrainFollowingGrid) = δyᶠᶜᶜ(i, j, k, grid, znode, _F(), _F(), _C()) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline Oceananigans.Operators.∂y_zᶠᶠᶜ(i, j, k, grid::TerrainFollowingGrid) = δyᶠᶠᶜ(i, j, k, grid, znode, _F(), _C(), _C()) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
@inline Oceananigans.Operators.∂y_zᶜᶜᶠ(i, j, k, grid::TerrainFollowingGrid) = δyᶜᶜᶠ(i, j, k, grid, znode, _C(), _F(), _F()) * Δy⁻¹ᶜᶜᶠ(i, j, k, grid)

#####
##### Number-argument disambiguation (derivative of a constant is zero)
#####

@inline Oceananigans.Operators.∂xᶠᶜᶜ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)
@inline Oceananigans.Operators.∂xᶜᶜᶜ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)
@inline Oceananigans.Operators.∂xᶠᶜᶠ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)
@inline Oceananigans.Operators.∂xᶜᶠᶜ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)
@inline Oceananigans.Operators.∂xᶠᶠᶜ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)

@inline Oceananigans.Operators.∂yᶜᶠᶜ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)
@inline Oceananigans.Operators.∂yᶜᶜᶜ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)
@inline Oceananigans.Operators.∂yᶜᶠᶠ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)
@inline Oceananigans.Operators.∂yᶠᶜᶜ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)
@inline Oceananigans.Operators.∂yᶠᶠᶜ(i, j, k, grid::TerrainFollowingGrid, c::Number) = zero(grid)

#####
##### Chain-rule-correct x-derivatives: (∂ϕ/∂x)_z = (∂ϕ/∂x)_r − (∂z/∂x)_r·(∂ϕ/∂z)
#####

@inline function Oceananigans.Operators.∂xᶠᶜᶜ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, ϕ)
    ∂x_z = Oceananigans.Operators.∂x_zᶠᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶠᶜᶜ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, f, args...)
    ∂x_z = Oceananigans.Operators.∂x_zᶠᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶜᶜᶜ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂x_at_r = δxᶜᶜᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, ϕ)
    ∂x_z = Oceananigans.Operators.∂x_zᶜᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶜᶜᶜ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂x_at_r = δxᶜᶜᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, f, args...)
    ∂x_z = Oceananigans.Operators.∂x_zᶜᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶠᶜᶠ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂x_at_r = δxᶠᶜᶠ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ϕ)
    ∂x_z = Oceananigans.Operators.∂x_zᶠᶜᶠ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶠᶜᶠ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂x_at_r = δxᶠᶜᶠ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶠ(i, j, k, grid, ∂zᶜᶜᶜ, f, args...)
    ∂x_z = Oceananigans.Operators.∂x_zᶠᶜᶠ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶜᶠᶜ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂x_at_r = δxᶜᶠᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶠᶠ, ϕ)
    ∂x_z = Oceananigans.Operators.∂x_zᶜᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶜᶠᶜ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂x_at_r = δxᶜᶠᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶠᶠ, f, args...)
    ∂x_z = Oceananigans.Operators.∂x_zᶜᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶠᶠᶜ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂x_at_r = δxᶠᶠᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶠᶠ, ϕ)
    ∂x_z = Oceananigans.Operators.∂x_zᶠᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂xᶠᶠᶜ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂x_at_r = δxᶠᶠᶜ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶠᶠ, f, args...)
    ∂x_z = Oceananigans.Operators.∂x_zᶠᶠᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end

#####
##### Chain-rule-correct y-derivatives: (∂ϕ/∂y)_z = (∂ϕ/∂y)_r − (∂z/∂y)_r·(∂ϕ/∂z)
#####

@inline function Oceananigans.Operators.∂yᶜᶠᶜ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂y_at_r = δyᶜᶠᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, ϕ)
    ∂y_z = Oceananigans.Operators.∂y_zᶜᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶜᶠᶜ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂y_at_r = δyᶜᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, f, args...)
    ∂y_z = Oceananigans.Operators.∂y_zᶜᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶜᶜᶜ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂y_at_r = δyᶜᶜᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, ϕ)
    ∂y_z = Oceananigans.Operators.∂y_zᶜᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶜᶜᶜ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂y_at_r = δyᶜᶜᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, f, args...)
    ∂y_z = Oceananigans.Operators.∂y_zᶜᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶜᶠᶠ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂y_at_r = δyᶜᶠᶠ(i, j, k, grid, ϕ) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶠ(i, j, k, grid, ∂zᶜᶜᶜ, ϕ)
    ∂y_z = Oceananigans.Operators.∂y_zᶜᶠᶠ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶜᶠᶠ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂y_at_r = δyᶜᶠᶠ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶠ(i, j, k, grid, ∂zᶜᶜᶜ, f, args...)
    ∂y_z = Oceananigans.Operators.∂y_zᶜᶠᶠ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶠᶜᶜ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂y_at_r = δyᶠᶜᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶠᶠᶠ, ϕ)
    ∂y_z = Oceananigans.Operators.∂y_zᶠᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶠᶜᶜ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂y_at_r = δyᶠᶜᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶠᶠᶠ, f, args...)
    ∂y_z = Oceananigans.Operators.∂y_zᶠᶜᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶠᶠᶜ(i, j, k, grid::TerrainFollowingGrid, ϕ)
    ∂y_at_r = δyᶠᶠᶜ(i, j, k, grid, ϕ) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶠᶜᶠ, ϕ)
    ∂y_z = Oceananigans.Operators.∂y_zᶠᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end

@inline function Oceananigans.Operators.∂yᶠᶠᶜ(i, j, k, grid::TerrainFollowingGrid, f::Function, args...)
    ∂y_at_r = δyᶠᶠᶜ(i, j, k, grid, f, args...) * Δy⁻¹ᶠᶠᶜ(i, j, k, grid)
    ∂z_ϕ = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶠᶜᶠ, f, args...)
    ∂y_z = Oceananigans.Operators.∂y_zᶠᶠᶜ(i, j, k, grid)
    return ∂y_at_r - ∂y_z * ∂z_ϕ
end
