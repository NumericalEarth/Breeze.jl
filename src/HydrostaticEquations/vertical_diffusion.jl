#####
##### Vertical diffusion of HydrostaticDynamics, in mass coordinates
#####
##### The flux of ψ ∈ {θ, u, v} through the z-face `k`, between the layers `k - 1` and `k`, is
#####
#####     F = - C (ψ[k] - ψ[k-1]) ,        C = ρᵃ K / Δh ,
#####
##### with K the diffusivity on the face (m² s⁻¹), ρᵃ = p / (Rᵈ T) the density of the air there
##### and Δh the height between the two layers, both diagnosed from the hydrostatic column. A layer
##### holds the mass μ = Δp / g per unit area, so
#####
#####     μ ∂ₜψ = C⁺ (ψ[k+1] - ψ[k]) - C⁻ (ψ[k] - ψ[k-1]) ,
#####
##### with no flux through the ground or the model top: the fluxes of the boundary conditions
##### enter the tendencies of the lowest and the highest layer with the explicit stages, as for a
##### model without diffusion. The column sums Σ μ θ = Σ ρθ Δz and Σ μ u are therefore changed by
##### the boundary fluxes only.
#####
##### The equation is integrated with a backward Euler step over the time step, after its explicit
##### stages: one tridiagonal system per column, whose matrix is symmetric and diagonally dominant
##### for any K ≥ 0 and any time step.
#####

struct VerticalDiffusion{K, N, B, C, S}
    diffusivity :: K     # K of θ on z-faces, m² s⁻¹
    viscosity :: N       # K of u and v on z-faces, m² s⁻¹
    boundary_layer :: B  # scheme that diagnoses the two before every solve, or nothing: they are prescribed
    conductances :: C    # (scalar, momentum): ρᵃ K / Δh on z-faces, kg m⁻² s⁻¹
    scratch :: S         # upper diagonal of the eliminated tridiagonal systems
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the vertical diffusion of the potential temperature `θ` and of the velocities `u` and `v`
of [`HydrostaticDynamics`](@ref) in each column, in mass coordinates: a layer of mass `Δp / g`
per unit area exchanges with its neighbors through fluxes `- ρᵃ K ∂ψ/∂h` on the coordinate
surfaces, with `h` the height and `ρᵃ` the density of the air, both diagnosed from the
hydrostatic column.

It is integrated implicitly, with one backward Euler step per time step after the explicit
stages, by a tridiagonal solve in each column. No flux passes through the ground or the model
top: boundary fluxes are those of the boundary conditions of `u`, `v`, `ρθ` and `ρE`, which enter
the lowest layer with the explicit stages. The column integrals of `ρθ` and of the momentum are
therefore changed by the boundary fluxes only. What is diffused is `θ`, so it is the integral
of `ρθ` that is conserved and not the enthalpy `cᵖᵈ T Δp / g` of the column, which changes
wherever `θ` moves between layers of different pressure. The kinetic energy that the diffusion
of the velocities removes is not returned as heat.

Keyword arguments
=================

- `diffusivity`: `K` of `θ` on the z-faces, in m² s⁻¹: a number, a function of the grid's
  coordinates, or an array. It is held in `model.dynamics.vertical_diffusion.diffusivity`, a field
  on the z-faces that may be set at any time, and so serves as an interface for a scheme
  outside the model. Its values must be finite and not negative: this is checked when the model
  is built and before every solve, which throws a `DomainError` otherwise. Default 0.
- `viscosity`: `K` of `u` and `v`, likewise, in `model.dynamics.vertical_diffusion.viscosity`.
  Default 0.
- `boundary_layer`: `nothing` (default), for which the two fields are used as they are, or a
  scheme that diagnoses them from the column before every solve, such as
  [`RichardsonNumberBoundaryLayer`](@ref).

```jldoctest
using Breeze
using Breeze.HydrostaticEquations: VerticalDiffusion

VerticalDiffusion(diffusivity=10, viscosity=10)

# output
VerticalDiffusion(diffusivity=10, viscosity=10, boundary_layer=nothing)
```
"""
VerticalDiffusion(; diffusivity=0, viscosity=0, boundary_layer=nothing) =
    VerticalDiffusion(diffusivity, viscosity, boundary_layer, nothing, nothing)

Adapt.adapt_structure(to, diffusion::VerticalDiffusion) =
    VerticalDiffusion(adapt(to, diffusion.diffusivity),
                      adapt(to, diffusion.viscosity),
                      adapt(to, diffusion.boundary_layer),
                      adapt(to, diffusion.conductances),
                      adapt(to, diffusion.scratch))

@regional VerticalDiffusion

Base.summary(diffusion::VerticalDiffusion) =
    string("VerticalDiffusion(diffusivity=", prettysummary(diffusion.diffusivity),
           ", viscosity=", prettysummary(diffusion.viscosity),
           ", boundary_layer=", summary_of_boundary_layer(diffusion.boundary_layer), ")")

Base.show(io::IO, diffusion::VerticalDiffusion) = print(io, summary(diffusion))

summary_of_boundary_layer(::Nothing) = "nothing"
summary_of_boundary_layer(boundary_layer) = summary(boundary_layer)

materialize_vertical_diffusion(::Nothing, grid) = nothing

function materialize_vertical_diffusion(diffusion::VerticalDiffusion, grid)
    diffusivity = ZFaceField(grid)
    viscosity = ZFaceField(grid)
    set!(diffusivity, diffusion.diffusivity)
    set!(viscosity, diffusion.viscosity)

    (is_admissible_diffusivity(diffusivity) && is_admissible_diffusivity(viscosity)) ||
        throw(ArgumentError("The diffusivity and the viscosity of VerticalDiffusion must be finite and may " *
                            "not be negative."))

    conductances = (scalar = ZFaceField(grid), momentum = ZFaceField(grid))
    scratch = CenterField(grid)

    return VerticalDiffusion(diffusivity, viscosity,
                             materialize_boundary_layer(diffusion.boundary_layer, grid),
                             conductances, scratch)
end

# Whether every value of a diffusivity (or conductance) is a finite number that is not negative.
# A NaN fails the first comparison and an infinity the second.
is_admissible_diffusivity(K) = minimum(K) ≥ 0 && isfinite(maximum(K))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Throw a `DomainError` unless the diffusivity and the viscosity of `diffusion` are finite and not
negative everywhere. The two are fields that a scheme outside the model may set at any time, so
they are checked where they are used: before every implicit solve, whose matrix is diagonally
dominant only for such values.
"""
function validate_diffusivities(diffusion::VerticalDiffusion)
    for (name, K) in ((:diffusivity, diffusion.diffusivity), (:viscosity, diffusion.viscosity))
        is_admissible_diffusivity(K) ||
            throw(DomainError(name, "The $name of VerticalDiffusion has a value that is negative, infinite or " *
                                    "not a number. It lies between $(minimum(K)) and $(maximum(K))."))
    end
    return nothing
end

#####
##### A dry boundary-layer K-profile: a mixing length and the gradient Richardson number
#####

struct RichardsonNumberBoundaryLayer{FT}
    von_karman_constant :: FT         # κᵛᵏ
    asymptotic_mixing_length :: FT    # λ, in m
    stable_coefficient :: FT          # b′ of the stable stability function
    unstable_coefficient :: FT        # b of the unstable stability function
    free_convection_coefficient :: FT # c of the unstable stability function
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a dry boundary-layer scheme for [`VerticalDiffusion`](@ref) that diagnoses one eddy
diffusivity for `θ`, `u` and `v` on every coordinate surface of the column from the local
gradients, in the manner of [Louis (1979)](@cite Louis1979):

    K = ℓ² S F(Ri),        ℓ = κᵛᵏ h / (1 + κᵛᵏ h / λ),        Ri = N² / S²,

with `h` the height of the surface above the ground, `S` the shear of the horizontal wind
between the two layers, `N² = (g / θ) ∂θ/∂h` and `ℓ` the mixing length of
[Blackadar (1962)](@cite Blackadar1962). The stability functions are those of
[Louis (1979)](@cite Louis1979),

    F = 1 / (1 + b′ Ri)²                    for Ri ≥ 0,
    F = 1 - b Ri / (1 + c |Ri|^½)           for Ri < 0,

so that without shear an unstable layer mixes at the free-convection limit
`K = ℓ² (b / c) (-N²)^½`.

This is a test-scale scheme, not a default and not a choice for climate integrations. It
departs from Louis's in three ways that matter for one: `c` is a constant, where Louis's depends
on the height and on the roughness length; heat and momentum share one diffusivity; and there is
neither a non-local (counter-gradient) transport nor entrainment at the top of the mixed layer,
so a convective layer keeps a slightly unstable gradient and grows by encroachment only.

Keyword arguments
=================

- `von_karman_constant`: `κᵛᵏ`. Default 0.4.
- `asymptotic_mixing_length`: `λ` in m. Default 150.
- `stable_coefficient`: `b′`. Default 4.7.
- `unstable_coefficient`: `b`. Default 9.4.
- `free_convection_coefficient`: `c`. Default 5.

```jldoctest
using Breeze
using Breeze.HydrostaticEquations: RichardsonNumberBoundaryLayer

RichardsonNumberBoundaryLayer(Float64)

# output
RichardsonNumberBoundaryLayer{Float64}(κᵛᵏ=0.4, λ=150.0 m, b′=4.7, b=9.4, c=5.0)
```
"""
function RichardsonNumberBoundaryLayer(FT::DataType = Oceananigans.defaults.FloatType;
                                       von_karman_constant = 0.4,
                                       asymptotic_mixing_length = 150,
                                       stable_coefficient = 4.7,
                                       unstable_coefficient = 9.4,
                                       free_convection_coefficient = 5)

    (von_karman_constant > 0 && asymptotic_mixing_length > 0 && stable_coefficient ≥ 0 &&
     unstable_coefficient ≥ 0 && free_convection_coefficient > 0) ||
        throw(ArgumentError("The parameters of RichardsonNumberBoundaryLayer must be positive " *
                            "(the two coefficients b′ and b may be zero)."))

    return RichardsonNumberBoundaryLayer{FT}(von_karman_constant, asymptotic_mixing_length, stable_coefficient,
                                             unstable_coefficient, free_convection_coefficient)
end

Base.summary(scheme::RichardsonNumberBoundaryLayer{FT}) where FT =
    string("RichardsonNumberBoundaryLayer{", FT, "}(κᵛᵏ=", scheme.von_karman_constant,
           ", λ=", scheme.asymptotic_mixing_length, " m, b′=", scheme.stable_coefficient,
           ", b=", scheme.unstable_coefficient, ", c=", scheme.free_convection_coefficient, ")")

Base.show(io::IO, scheme::RichardsonNumberBoundaryLayer) = print(io, summary(scheme))

materialize_boundary_layer(::Nothing, grid) = nothing

materialize_boundary_layer(scheme::RichardsonNumberBoundaryLayer, grid) =
    RichardsonNumberBoundaryLayer(eltype(grid);
                                  von_karman_constant = scheme.von_karman_constant,
                                  asymptotic_mixing_length = scheme.asymptotic_mixing_length,
                                  stable_coefficient = scheme.stable_coefficient,
                                  unstable_coefficient = scheme.unstable_coefficient,
                                  free_convection_coefficient = scheme.free_convection_coefficient)

# K = ℓ² S F(Ri) at the height `h` above the ground, from the squares S² and N² of the shear and
# of the buoyancy frequency. Written without Ri so that it stays finite without shear:
#
#   stable     ℓ² S / (1 + b′ N² / S²)² = ℓ² S⁵ / (S² + b′ N²)²
#   unstable   ℓ² S [1 - b Ri / (1 + c |Ri|^½)] = ℓ² [S + b (-N²) / (S + c (-N²)^½)]
#
# Both vanish with the gradients, and a neutral layer at rest does not mix.
@inline function eddy_diffusivity(scheme::RichardsonNumberBoundaryLayer, h, S², N²)
    κᵛᵏ = scheme.von_karman_constant
    ℓ = κᵛᵏ * h / (1 + κᵛᵏ * h / scheme.asymptotic_mixing_length)
    S = sqrt(S²)

    stable_denominator = (S² + scheme.stable_coefficient * max(0, N²))^2
    stable = ifelse(stable_denominator > 0, S^5 / ifelse(stable_denominator > 0, stable_denominator, one(S)), zero(S))

    Nᶜ = sqrt(max(0, -N²))
    unstable_denominator = S + scheme.free_convection_coefficient * Nᶜ
    convective = ifelse(unstable_denominator > 0,
                        Nᶜ^2 / ifelse(unstable_denominator > 0, unstable_denominator, one(S)), zero(S))
    unstable = S + scheme.unstable_coefficient * convective

    return ℓ^2 * ifelse(N² ≥ 0, stable, unstable)
end
