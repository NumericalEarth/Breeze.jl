#####
##### Grid-scaled ∇⁴ hyperdiffusion of HydrostaticDynamics along coordinate surfaces
#####
##### For ψ ∈ {u, v, θ}, layer by layer,
#####
#####     ∂ₜψ = … - (1 / 64 τ₄) 𝓛(𝓛ψ) ,        𝓛ψ = m⁻¹ [δx(mˣ δxψ) + δy(mʸ δyψ)] ,
#####
##### with δ the undivided difference between neighbors along a horizontal index, m the mass of
##### the cell of ψ (kg) and mˣ, mʸ its averages on the faces of that cell. Where the mass is
##### uniform 𝓛 is the undivided second difference δx² + δy², so a wave damps at the rate
#####
#####     (1 / τ₄) {[sin²(kΔx/2) + sin²(lΔy/2)] / 2}² :
#####
##### 1 / τ₄ for the oblique checkerboard (2Δ in both directions), which is what defines τ₄, and
##### 1 / (4 τ₄) for a 2Δ wave along one axis.
#####
##### 𝓛 is a difference of fluxes over the mass, and it is self-adjoint in the mass-weighted sum,
##### Σ m a 𝓛b = Σ m b 𝓛a = - Σ (mˣ δxa δxb + mʸ δya δyb), provided that no flux crosses the edge
##### of the sum or that the quantities vanish there. On one grid this gives, for θ,
#####
#####     conservation     Σ V ∂ₜ(ρθ) = 0 ,                     the mass-weighted Θ integral;
#####     dissipation      Σ m θ ∂ₜθ = - (1 / 64 τ₄) Σ m (𝓛θ)² ≤ 0 ,
#####
##### and the same dissipation for u and v with the masses of their cells. The mass-weighted sum
##### of a velocity component is conserved along walls but not across them: the component normal
##### to a wall is held at zero on it. A uniform ψ is not changed, because its differences vanish
##### exactly. The tendency of ρθ is ρ times that of θ: the layer mass is not diffused.
#####
##### Walls. Through the face of a wall (a `Bounded` direction) neither flux passes: mˣ δxψ = 0 and
##### mˣ δx(𝓛ψ) = 0 there, for θ and for the velocity component along the wall. The component
##### normal to a wall vanishes on it, and so does its 𝓛. The stencil therefore never reads
##### beyond a wall.
#####
##### Yin-Yang. Each panel applies the operator to all of its cells and reads the other panel's
##### interpolated values in its halos, so the two panels disagree on the fluxes through the curve
##### that divides the sphere between them: the owned Θ integral changes at the seam budget of
##### these fluxes, which the `HydrostaticConservationFixer` returns with that of the advective
##### fluxes when it is added to a simulation. The dissipation identity holds on a panel only up
##### to the same exchange with its halos.
#####

struct GridScaleHyperdiffusion{FT}
    damping_time :: FT # τ₄: e-folding time of the oblique checkerboard, in s
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a grid-scaled ∇⁴ hyperdiffusion of the velocities `u` and `v` and of the potential
temperature `θ` of [`HydrostaticDynamics`](@ref) along its coordinate surfaces,

    ∂ₜψ = … - (1 / 64 τ₄) 𝓛(𝓛ψ),        𝓛ψ = m⁻¹ [δx(mˣ δxψ) + δy(mʸ δyψ)],

with `δ` the undivided difference between neighboring points, `m` the mass of the cell of `ψ`
and `mˣ`, `mʸ` its averages on the faces of that cell. The coefficient scales with the grid: the
grid scale damps in the same time wherever the cells are small, as near the walls of a
latitude-longitude grid.

`damping_time` is `τ₄`, in seconds: the e-folding time of the oblique checkerboard, the wave of
two grid lengths in both horizontal directions. A wave of two grid lengths along one axis damps
in `4 τ₄`, and a wave of wavenumbers `k` and `l` at the rate
`{[sin²(kΔx/2) + sin²(lΔy/2)] / 2}² / τ₄`.

The potential temperature is diffused in flux form, `∂ₜ(ρθ) = ρ ∂ₜθ`, so the integral of `ρθ`
over a grid is conserved and the mass-weighted variance of `θ` cannot grow, nor can the
mass-weighted sums of the squares of `u` and `v`. A uniform field is unchanged. No flux of `θ`
or of the velocity along it passes through a wall, on which the velocity normal to it is zero.
On a `YinYangGrid` the integral over the sphere is conserved only with the conservation fixer,
which returns what the panels' fluxes exchange through the seam.

The kinetic energy that the hyperdiffusion removes is not returned as heat.

```jldoctest
using Breeze
using Breeze.HydrostaticEquations: GridScaleHyperdiffusion

GridScaleHyperdiffusion(Float64; damping_time=43200)

# output
GridScaleHyperdiffusion{Float64}(damping_time=43200.0 s): 2Δ checkerboard in 12.0 h, 2Δ axis wave in 48.0 h
```
"""
GridScaleHyperdiffusion(FT::DataType=Oceananigans.defaults.FloatType; damping_time=43200) =
    GridScaleHyperdiffusion{FT}(convert(FT, damping_time))

function Base.summary(hyperdiffusion::GridScaleHyperdiffusion{FT}) where FT
    τ₄ = hyperdiffusion.damping_time
    return string("GridScaleHyperdiffusion{", FT, "}(damping_time=", τ₄, " s): 2Δ checkerboard in ",
                  τ₄ / 3600, " h, 2Δ axis wave in ", 4 * τ₄ / 3600, " h")
end

Base.show(io::IO, hyperdiffusion::GridScaleHyperdiffusion) = print(io, summary(hyperdiffusion))

materialize_hyperdiffusion(::Nothing, grid) = nothing

function materialize_hyperdiffusion(hyperdiffusion::GridScaleHyperdiffusion, grid)
    hyperdiffusion.damping_time > 0 ||
        throw(ArgumentError("The damping time of GridScaleHyperdiffusion must be positive, got " *
                            "$(hyperdiffusion.damping_time)."))

    return GridScaleHyperdiffusion(eltype(grid); damping_time=hyperdiffusion.damping_time)
end

# 1 / (64 τ₄): the symbol of 𝓛² is (4 + 4)² = 64 for the oblique checkerboard
@inline hyperdiffusion_coefficient(hyperdiffusion::GridScaleHyperdiffusion) = 1 / (64 * hyperdiffusion.damping_time)

#####
##### Walls: the faces through which nothing passes
#####

const XBoundedGrid = AbstractGrid{<:Any, Bounded}
const YBoundedGrid = AbstractGrid{<:Any, <:Any, Bounded}

@inline interior_x_face(i, grid) = true
@inline interior_y_face(j, grid) = true
@inline interior_x_face(i, grid::XBoundedGrid) = (i > 1) & (i < size(grid, 1) + 1)
@inline interior_y_face(j, grid::YBoundedGrid) = (j > 1) & (j < size(grid, 2) + 1)

#####
##### Masses of the cells of θ, u and v, and of the faces between them
#####

@inline cell_mass(i, j, k, grid, ρ) = @inbounds ρ[i, j, k] * Vᶜᶜᶜ(i, j, k, grid) # (Center, Center, Center)
@inline cell_mass(i, j, k, grid::PressureLevelGrid, ρ) = Azᶜᶜᶜ(i, j, k, grid) * layer_mass(i, j, k, grid) # Az Δp / g, exactly
@inline x_face_mass(i, j, k, grid, ρ) = ℑxᶠᵃᵃ(i, j, k, grid, cell_mass, ρ)       # (Face, Center, Center)
@inline y_face_mass(i, j, k, grid, ρ) = ℑyᵃᶠᵃ(i, j, k, grid, cell_mass, ρ)       # (Center, Face, Center)
@inline corner_mass(i, j, k, grid, ρ) = ℑyᵃᶠᵃ(i, j, k, grid, x_face_mass, ρ)     # (Face, Face, Center)

#####
##### 𝓛 of a quantity at cell centers
#####
##### `c` is a field, or a function `c(i, j, k, grid, args...)` of the same location, which is
##### how 𝓛 is applied to 𝓛.
#####

@inline center_x_flux(i, j, k, grid, ρ, c, args...) =
    ifelse(interior_x_face(i, grid), x_face_mass(i, j, k, grid, ρ) * δxᶠᵃᵃ(i, j, k, grid, c, args...), zero(grid))

@inline center_y_flux(i, j, k, grid, ρ, c, args...) =
    ifelse(interior_y_face(j, grid), y_face_mass(i, j, k, grid, ρ) * δyᵃᶠᵃ(i, j, k, grid, c, args...), zero(grid))

@inline center_laplacian(i, j, k, grid, ρ, c, args...) =
    (δxᶜᵃᵃ(i, j, k, grid, center_x_flux, ρ, c, args...) +
     δyᵃᶜᵃ(i, j, k, grid, center_y_flux, ρ, c, args...)) / cell_mass(i, j, k, grid, ρ)

#####
##### 𝓛 of the x-velocity, at (Face, Center, Center)
#####
##### Its neighbors along x meet at cell centers and those along y at the corners. On a wall
##### normal to x the velocity vanishes, and its 𝓛 with it.
#####

@inline x_velocity_x_flux(i, j, k, grid, ρ, u, args...) = cell_mass(i, j, k, grid, ρ) * δxᶜᵃᵃ(i, j, k, grid, u, args...)

@inline x_velocity_y_flux(i, j, k, grid, ρ, u, args...) =
    ifelse(interior_y_face(j, grid), corner_mass(i, j, k, grid, ρ) * δyᵃᶠᵃ(i, j, k, grid, u, args...), zero(grid))

@inline x_velocity_laplacian(i, j, k, grid, ρ, u, args...) =
    ifelse(interior_x_face(i, grid),
           (δxᶠᵃᵃ(i, j, k, grid, x_velocity_x_flux, ρ, u, args...) +
            δyᵃᶜᵃ(i, j, k, grid, x_velocity_y_flux, ρ, u, args...)) / x_face_mass(i, j, k, grid, ρ),
           zero(grid))

#####
##### 𝓛 of the y-velocity, at (Center, Face, Center)
#####

@inline y_velocity_x_flux(i, j, k, grid, ρ, v, args...) =
    ifelse(interior_x_face(i, grid), corner_mass(i, j, k, grid, ρ) * δxᶠᵃᵃ(i, j, k, grid, v, args...), zero(grid))

@inline y_velocity_y_flux(i, j, k, grid, ρ, v, args...) = cell_mass(i, j, k, grid, ρ) * δyᵃᶜᵃ(i, j, k, grid, v, args...)

@inline y_velocity_laplacian(i, j, k, grid, ρ, v, args...) =
    ifelse(interior_y_face(j, grid),
           (δxᶜᵃᵃ(i, j, k, grid, y_velocity_x_flux, ρ, v, args...) +
            δyᵃᶠᵃ(i, j, k, grid, y_velocity_y_flux, ρ, v, args...)) / y_face_mass(i, j, k, grid, ρ),
           zero(grid))

#####
##### Tendencies
#####

@inline potential_temperature_hyperdiffusion(i, j, k, grid, ::Nothing, ρ, θ) = zero(grid)
@inline x_velocity_hyperdiffusion(i, j, k, grid, ::Nothing, ρ, u) = zero(grid)
@inline y_velocity_hyperdiffusion(i, j, k, grid, ::Nothing, ρ, v) = zero(grid)

# Tendency of ρθ: ρ times that of θ, which is the divergence of the flux mˣ δx(𝓛θ) over the volume
@inline potential_temperature_hyperdiffusion(i, j, k, grid, hyperdiffusion::GridScaleHyperdiffusion, ρ, θ) =
    @inbounds - hyperdiffusion_coefficient(hyperdiffusion) * ρ[i, j, k] *
                center_laplacian(i, j, k, grid, ρ, center_laplacian, ρ, θ)

@inline x_velocity_hyperdiffusion(i, j, k, grid, hyperdiffusion::GridScaleHyperdiffusion, ρ, u) =
    - hyperdiffusion_coefficient(hyperdiffusion) * x_velocity_laplacian(i, j, k, grid, ρ, x_velocity_laplacian, ρ, u)

@inline y_velocity_hyperdiffusion(i, j, k, grid, hyperdiffusion::GridScaleHyperdiffusion, ρ, v) =
    - hyperdiffusion_coefficient(hyperdiffusion) * y_velocity_laplacian(i, j, k, grid, ρ, y_velocity_laplacian, ρ, v)

# The horizontal flux divergence of ρθ that the hyperdiffusion adds to that of advection: what a
# conservation fixer integrates over the owned cells of a Yin-Yang grid
@inline hyperdiffusion_flux_divergence(i, j, k, grid, hyperdiffusion, ρ, θ) =
    - potential_temperature_hyperdiffusion(i, j, k, grid, hyperdiffusion, ρ, θ)

#####
##### Dissipation
#####

# Rates per unit volume of the grid at which the hyperdiffusion removes ½ ρ θ² (kg K² m⁻³ s⁻¹)
# and the kinetic energies ½ ρ u² and ½ ρ v² (W m⁻³): (1 / 64 τ₄) m (𝓛ψ)² / V. None is negative.
@inline potential_temperature_variance_dissipation(i, j, k, grid, hyperdiffusion, ρ, θ) =
    @inbounds hyperdiffusion_coefficient(hyperdiffusion) * ρ[i, j, k] * center_laplacian(i, j, k, grid, ρ, θ)^2

@inline x_kinetic_energy_dissipation(i, j, k, grid, hyperdiffusion, ρ, u) =
    hyperdiffusion_coefficient(hyperdiffusion) * x_face_mass(i, j, k, grid, ρ) / Vᶠᶜᶜ(i, j, k, grid) *
    x_velocity_laplacian(i, j, k, grid, ρ, u)^2

@inline y_kinetic_energy_dissipation(i, j, k, grid, hyperdiffusion, ρ, v) =
    hyperdiffusion_coefficient(hyperdiffusion) * y_face_mass(i, j, k, grid, ρ) / Vᶜᶠᶜ(i, j, k, grid) *
    y_velocity_laplacian(i, j, k, grid, ρ, v)^2
