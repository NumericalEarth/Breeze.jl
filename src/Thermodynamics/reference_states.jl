using Oceananigans: Oceananigans, Center, Field, set!, fill_halo_regions!
using Adapt: Adapt, adapt
                   
#####
##### Reference state computations for Boussinesq and Anelastic models
#####

struct ReferenceState{FT, F}
    base_pressure :: FT # base pressure: reference pressure at z=0
    potential_temperature :: FT  # constant reference potential temperature
    pressure :: F
    density :: F
end

Adapt.adapt_structure(to, ref::ReferenceState) =
    ReferenceState(adapt(to, ref.base_pressure),
                   adapt(to, ref.potential_temperature),
                   adapt(to, ref.pressure),
                   adapt(to, ref.density))

Base.eltype(::ReferenceState{FT}) where FT = FT

function Base.summary(ref::ReferenceState)
    FT = eltype(ref)
    return string("ReferenceState{$FT}(p₀=", prettysummary(ref.base_pressure),
                  ", θᵣ=", prettysummary(ref.potential_temperature), ")")
end

Base.show(io::IO, ref::ReferenceState) = print(io, summary(ref))

#####
##### How to compute the reference state
#####

@inline function base_density(p₀, θᵣ, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    return p₀ / (Rᵈ * θᵣ)
end

"""
    adiabatic_hydrostatic_pressure(z, p₀, θᵣ, thermo)

Compute the reference pressure at height `z` that associated with the reference pressure and
potential temperature. The reference pressure is defined as the pressure of dry air at the
reference pressure and temperature.
"""
@inline function adiabatic_hydrostatic_pressure(z, p₀, θᵣ, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    g = thermo.gravitational_acceleration
    return p₀ * (1 - g * z / (cᵖᵈ * θᵣ))^(cᵖᵈ / Rᵈ)
end

"""
    adiabatic_hydrostatic_density(z, p₀, θᵣ, thermo)

Compute the reference density at height `z` that associated with the reference pressure and
potential temperature. The reference density is defined as the density of dry air at the
reference pressure and temperature.
"""
@inline function adiabatic_hydrostatic_density(z, p₀, θᵣ, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    pᵣ = adiabatic_hydrostatic_pressure(z, p₀, θᵣ, thermo)
    ρ₀ = base_density(p₀, θᵣ, thermo)
    return ρ₀ * (pᵣ / p₀)^(1 - Rᵈ / cᵖᵈ)
end

function ReferenceState(grid, thermo;
                        base_pressure = 101325,
                        potential_temperature = 288)

    FT = eltype(grid)
    p₀ = convert(FT, base_pressure)
    θᵣ = convert(FT, potential_temperature)

    pᵣ = Field{Nothing, Nothing, Center}(grid)
    ρᵣ = Field{Nothing, Nothing, Center}(grid)
    set!(pᵣ, z -> adiabatic_hydrostatic_pressure(z, p₀, θᵣ, thermo))
    set!(ρᵣ, z -> adiabatic_hydrostatic_density(z, p₀, θᵣ, thermo))
    fill_halo_regions!(pᵣ)
    fill_halo_regions!(ρᵣ)

    return ReferenceState(p₀, θᵣ, pᵣ, ρᵣ)
end
