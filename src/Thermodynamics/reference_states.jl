using Oceananigans: Oceananigans, Center, Field, set!, fill_halo_regions!
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, ValueBoundaryCondition

using Adapt: Adapt, adapt

#####
##### Reference state computations for Boussinesq and Anelastic models
#####

struct ReferenceState{FT, F}
    surface_pressure :: FT # base pressure: reference pressure at z=0
    potential_temperature :: FT  # constant reference potential temperature
    standard_pressure :: FT # pˢᵗ: reference pressure for potential temperature (default 1e5)
    pressure :: F
    density :: F
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

@inline function surface_density(p₀, θ₀, constants)
    Rᵈ = dry_air_gas_constant(constants)
    return p₀ / (Rᵈ * θ₀)
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

Compute the reference density at height `z` that associated with the reference pressure and
potential temperature `θ₀`. The reference density is defined as the density of dry air at the
reference pressure and temperature.
"""
@inline function adiabatic_hydrostatic_density(z, p₀, θ₀, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    pᵣ = adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants)
    ρ₀ = surface_density(p₀, θ₀, constants)
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
    loc = (nothing, nothing, Center())

    ρ₀ = surface_density(p₀, θ₀, constants)
    ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ₀))
    ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)
    set!(ρᵣ, z -> adiabatic_hydrostatic_density(z, p₀, θ₀, constants))
    fill_halo_regions!(ρᵣ)

    p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
    pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
    set!(pᵣ, z -> adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants))
    fill_halo_regions!(pᵣ)

    return ReferenceState(p₀, θ₀, pˢᵗ, pᵣ, ρᵣ)
end
