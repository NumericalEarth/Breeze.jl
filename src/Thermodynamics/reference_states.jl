using Oceananigans: Oceananigans, Center, Field, set!, fill_halo_regions!, ∂z, znodes
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, GradientBoundaryCondition
using Oceananigans.Operators: ℑzᵃᵃᶠ

using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar

#####
##### Reference state computations for Boussinesq and Anelastic models
#####

struct ReferenceState{FT, P, R}
    surface_pressure :: FT # base pressure: reference pressure at z=0
    potential_temperature :: FT  # constant reference potential temperature
    standard_pressure :: FT # pˢᵗ: reference pressure for potential temperature (default 1e5)
    pressure :: P
    density :: R
end

Adapt.adapt_structure(to, ref::ReferenceState) =
    ReferenceState(adapt(to, ref.surface_pressure),
                   adapt(to, ref.potential_temperature),
                   adapt(to, ref.standard_pressure),
                   adapt(to, ref.pressure),
                   adapt(to, ref.density))

Base.eltype(::ReferenceState{FT}) where FT = FT

function Base.summary(ref::ReferenceState)
    FT = eltype(ref)
    return string("ReferenceState{$FT}(p₀=", prettysummary(ref.surface_pressure),
                  ", θ₀=", prettysummary(ref.potential_temperature),
                  ", pˢᵗ=", prettysummary(ref.standard_pressure), ")")
end

Base.show(io::IO, ref::ReferenceState) = print(io, summary(ref))

#####
##### How to compute the reference state
#####

"""
    surface_density(reference_state)

Return the density at z=0 by interpolating the reference density field to the surface.
"""
function surface_density(ref::ReferenceState)
    ρ = ref.density
    grid = ρ.grid
    return @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, ρ)
end

"""
    surface_density(p₀, T₀, constants)

Compute the surface air density from surface pressure `p₀`, surface temperature `T₀`,
and thermodynamic `constants` using the ideal gas law for dry air.
"""
@inline function surface_density(p₀, T₀, constants)
    Rᵈ = dry_air_gas_constant(constants)
    return p₀ / (Rᵈ * T₀)
end

"""
    surface_density(p₀, θ₀, pˢᵗ, constants)

Compute the surface air density from surface pressure `p₀`, potential temperature `θ₀`,
standard pressure `pˢᵗ`, and thermodynamic `constants` using the ideal gas law for dry air.

The temperature is computed from potential temperature using the Exner function:
`T₀ = Π₀ * θ₀` where `Π₀ = (p₀ / pˢᵗ)^(Rᵈ/cᵖᵈ)`.
"""
@inline function surface_density(p₀, θ₀, pˢᵗ, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    Π₀ = (p₀ / pˢᵗ)^(Rᵈ / cᵖᵈ)
    T₀ = Π₀ * θ₀
    return p₀ / (Rᵈ * T₀)
end

"""
$(TYPEDSIGNATURES)

Compute the reference pressure at height `z` that associated with the reference pressure `p₀` and
potential temperature `θ₀`. The reference pressure is defined as the pressure of dry air at the
reference pressure and temperature.
"""
@inline function adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(constants)
    g = constants.gravitational_acceleration
    return p₀ * (1 - g * z / (cᵖᵈ * θ₀))^(cᵖᵈ / Rᵈ)
end

"""
$(TYPEDSIGNATURES)

Compute the reference density at height `z` that associated with the reference pressure `p₀`,
potential temperature `θ₀`, and standard pressure `pˢᵗ`. The reference density is defined as
the density of dry air at the reference pressure and temperature.
"""
@inline function adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    pᵣ = adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants)
    ρ₀ = surface_density(p₀, θ₀, pˢᵗ, constants)
    return ρ₀ * (pᵣ / p₀)^(1 - Rᵈ / cᵖᵈ)
end

"""
$(TYPEDSIGNATURES)

Return a `ReferenceState` on `grid`, with [`ThermodynamicConstants`](@ref) `constants`
that includes the adiabatic hydrostatic reference pressure and reference density
for a `surface_pressure` and `potential_temperature`.

Arguments
=========
- `grid`: The grid.
- `constants :: ThermodynamicConstants`: By default, `ThermodynamicConstants(eltype(grid))`.

Keyword arguments
=================
- `surface_pressure`: By default, 101325.
- `potential_temperature`: By default, 288.
- `standard_pressure`: Reference pressure for potential temperature (pˢᵗ). By default, 1e5.
"""
function ReferenceState(grid, constants=ThermodynamicConstants(eltype(grid));
                        surface_pressure = 101325,
                        potential_temperature = 288,
                        standard_pressure = 1e5)

    FT = eltype(grid)
    p₀ = convert(FT, surface_pressure)
    θ₀ = convert(FT, potential_temperature)
    pˢᵗ = convert(FT, standard_pressure)
    g = constants.gravitational_acceleration
    loc = (nothing, nothing, Center())

    # Use GradientBoundaryCondition at both top and bottom boundaries to ensure
    # correct discrete hydrostatic balance: ρ = -∂z(p)/g. The gradient is set
    # using the analytical hydrostatic density at each boundary cell center.
    z = znodes(grid, Center())
    z_bottom = first(z)
    z_top = last(z)

    ρ_bottom = adiabatic_hydrostatic_density(z_bottom, p₀, θ₀, pˢᵗ, constants)
    ρ_top = adiabatic_hydrostatic_density(z_top, p₀, θ₀, pˢᵗ, constants)
    ∂p∂z_bottom = -ρ_bottom * g
    ∂p∂z_top = -ρ_top * g

    p_bcs = FieldBoundaryConditions(grid, loc,
        bottom = GradientBoundaryCondition(∂p∂z_bottom),
        top = GradientBoundaryCondition(∂p∂z_top))
    pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
    set!(pᵣ, z -> adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants))
    fill_halo_regions!(pᵣ)

    # Compute density from discrete pressure gradient for discrete hydrostatic balance.
    # Use gradient BC based on analytical density at each boundary.
    ρ_bcs = FieldBoundaryConditions(grid, loc,
        bottom = GradientBoundaryCondition(zero(FT)),
        top = GradientBoundaryCondition(zero(FT)))
    ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)
    set!(ρᵣ, - ∂z(pᵣ) / g)
    fill_halo_regions!(ρᵣ)

    return ReferenceState(p₀, θ₀, pˢᵗ, pᵣ, ρᵣ)
end
