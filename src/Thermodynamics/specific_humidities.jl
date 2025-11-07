struct MoistureMassFractions{FT}
    vapor :: FT
    liquid :: FT
    ice :: FT
end

@inline total_specific_humidity(q::MoistureMassFractions) = q.vapor + q.liquid + q.ice
@inline dry_air_mass_fraction(q::MoistureMassFractions) = 1 - total_specific_humidity(q)

"""
    mixture_gas_constant(q::MoistureMassFractions, thermo)

Compute the gas constant of moist air given the specific humidity `q` and 
thermodynamic parameters `thermo`.

The mixture gas constant is calculated as a weighted average of the dry air
and water vapor gas thermo:

```math
Rᵐ = qᵈ * Rᵈ + qᵛ * Rᵛ
```

where:
- `Rᵈ` is the dry air gas constant
- `Rᵛ` is the water vapor gas constant  
- `qᵈ` is the mass fraction of dry air
- `qᵛ` is the mass fraction of water vapor

# Arguments
- `qᵈ`: Mass fraction of dry air (dimensionless)
- `qᵛ`: Mass fraction of water vapor (dimensionless)
- `thermo`: `ThermodynamicConstants` instance containing gas thermo

# Returns
- Gas constant of the moist air mixture in J/(kg·K)
"""
@inline function mixture_gas_constant(q::MoistureMassFractions, thermo::TC)
    qᵈ = dry_air_mass_fraction(q)
    qᵛ = q.vapor
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    return qᵈ * Rᵈ + qᵛ * Rᵛ
end

"""
    mixture_heat_capacity(qᵈ, qᵛ, thermo)

Compute the heat capacity of state air given the total specific humidity q
and assuming that condensate mass ratio qᶜ ≪ q, where qℓ is the mass ratio of
liquid condensate.
"""
@inline function mixture_heat_capacity(q::MoistureMassFractions, thermo::TC)
    qᵈ = dry_air_mass_fraction(q)
    qᵛ = q.vapor
    cᵖᵈ = thermo.dry_air.heat_capacity
    cᵖᵛ = thermo.vapor.heat_capacity
    return qᵈ * cᵖᵈ + qᵛ * cᵖᵛ
end

# TODO: deprecate these
"""
    mixture_gas_constant(q, thermo)

Compute the gas constant of moist air given the specific humidity `q` and
thermodynamic parameters `thermo`.

The mixture gas constant is calculated as a weighted average of the dry air
and water vapor gas constants:

```math
R_m = R_d (1 - q) + R_v q
```

where:
- `R_d` is the dry air gas constant
- `R_v` is the water vapor gas constant
- `q` is the specific humidity (mass fraction of water vapor)

# Arguments
- `q`: Specific humidity (dimensionless)
- `thermo`: `ThermodynamicConstants` instance containing gas constants

# Returns
- Gas constant of the moist air mixture in J/(kg·K)
"""
@inline function mixture_gas_constant(q, thermo::TC)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    return Rᵈ * (1 - q) + Rᵛ * q
end

"""
    mixture_heat_capacity(q, thermo)

Compute the heat capacity of state air given the total specific humidity q
and assuming that condensate mass ratio qᶜ ≪ q, where qℓ is the mass ratio of
liquid condensate.
"""
@inline function mixture_heat_capacity(q, thermo::TC)
    cᵖᵈ = thermo.dry_air.heat_capacity
    cᵖᵛ = thermo.vapor.heat_capacity
    return cᵖᵈ * (1 - q) + cᵖᵛ * q
end
