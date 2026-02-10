using Oceananigans: Oceananigans, Center, Field, set!, fill_halo_regions!
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, ValueBoundaryCondition
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: znode
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!

using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar
using KernelAbstractions: @kernel, @index

#####
##### Reference state computations for Boussinesq and Anelastic models
#####

struct ReferenceState{FT, P, D, T, QV, QL, QI}
    surface_pressure :: FT # base pressure: reference pressure at z=0
    potential_temperature :: FT  # constant reference potential temperature
    standard_pressure :: FT # pˢᵗ: reference pressure for potential temperature (default 1e5)
    pressure :: P
    density :: D
    temperature :: T
    vapor_mass_fraction :: QV
    liquid_mass_fraction :: QL
    ice_mass_fraction :: QI
end

Adapt.adapt_structure(to, ref::ReferenceState) =
    ReferenceState(adapt(to, ref.surface_pressure),
                   adapt(to, ref.potential_temperature),
                   adapt(to, ref.standard_pressure),
                   adapt(to, ref.pressure),
                   adapt(to, ref.density),
                   adapt(to, ref.temperature),
                   adapt(to, ref.vapor_mass_fraction),
                   adapt(to, ref.liquid_mass_fraction),
                   adapt(to, ref.ice_mass_fraction))

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

#####
##### Hydrostatic reference profiles from temperature and moisture
#####

@kernel function _compute_hydrostatic_reference!(pᵣ, ρᵣ, T, qᵛ, qˡ, qⁱ, grid, Nz, p₀, Rᵈ, Rᵛ, g)
    _ = @index(Global)
    c = Center()

    # Use first-level thermodynamic values as surface approximation
    @inbounds begin
        T¹  = T[1, 1, 1]
        qᵛ¹ = qᵛ[1, 1, 1]
        qˡ¹ = qˡ[1, 1, 1]
        qⁱ¹ = qⁱ[1, 1, 1]
    end
    qᵈ¹ = 1 - qᵛ¹ - qˡ¹ - qⁱ¹
    Rᵐ¹ = qᵈ¹ * Rᵈ + qᵛ¹ * Rᵛ

    # Initialize at z = 0 (surface)
    z⁻   = zero(T¹)
    RᵐT⁻ = Rᵐ¹ * T¹
    p⁻   = p₀

    # Integrate upward: d(ln p)/dz = -g / (Rᵐ T)
    for k in 1:Nz
        zᵏ = znode(1, 1, k, grid, c, c, c)
        @inbounds begin
            Tᵏ  = T[1, 1, k]
            qᵛᵏ = qᵛ[1, 1, k]
            qˡᵏ = qˡ[1, 1, k]
            qⁱᵏ = qⁱ[1, 1, k]
        end
        qᵈᵏ = 1 - qᵛᵏ - qˡᵏ - qⁱᵏ
        Rᵐᵏ = qᵈᵏ * Rᵈ + qᵛᵏ * Rᵛ
        RᵐTᵏ = Rᵐᵏ * Tᵏ

        Δz = zᵏ - z⁻
        pᵏ = p⁻ * exp(-g * Δz / ((RᵐT⁻ + RᵐTᵏ) / 2))

        @inbounds pᵣ[1, 1, k] = pᵏ
        @inbounds ρᵣ[1, 1, k] = pᵏ / RᵐTᵏ

        z⁻   = zᵏ
        RᵐT⁻ = RᵐTᵏ
        p⁻   = pᵏ
    end
end

"""
    compute_hydrostatic_reference!(ref::ReferenceState, constants)

Compute the hydrostatic reference pressure and density profiles from the
temperature and moisture mass fraction profiles stored in `ref`.

The integration uses the mixture gas constant `Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ`
(where `qᵈ = 1 - qᵛ - qˡ - qⁱ`) and the ideal gas law `ρ = p / (Rᵐ T)`.
"""
function compute_hydrostatic_reference!(ref::ReferenceState, constants)
    grid = ref.pressure.grid
    arch = architecture(grid)
    Nz = grid.Nz

    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    g = constants.gravitational_acceleration
    p₀ = ref.surface_pressure

    launch!(arch, grid, tuple(1),
            _compute_hydrostatic_reference!,
            ref.pressure, ref.density,
            ref.temperature, ref.vapor_mass_fraction,
            ref.liquid_mass_fraction, ref.ice_mass_fraction,
            grid, Nz, p₀, Rᵈ, Rᵛ, g)

    fill_halo_regions!(ref.pressure)
    fill_halo_regions!(ref.density)

    return nothing
end

#####
##### Constructor
#####

"""
$(TYPEDSIGNATURES)

Return a `ReferenceState` on `grid`, with [`ThermodynamicConstants`](@ref) `constants`
that includes the hydrostatic reference pressure and reference density.

The reference state is initialized with a dry adiabatic temperature profile
and the given moisture profiles (zero by default). The pressure and density
are then computed by hydrostatic integration using the mixture gas constant
`Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ` and the ideal gas law `ρ = p / (Rᵐ T)`.

Arguments
=========
- `grid`: The grid.
- `constants :: ThermodynamicConstants`: By default, `ThermodynamicConstants(eltype(grid))`.

Keyword arguments
=================
- `surface_pressure`: By default, 101325.
- `potential_temperature`: By default, 288.
- `standard_pressure`: Reference pressure for potential temperature (pˢᵗ). By default, 1e5.
- `vapor_mass_fraction`: Initial qᵛ profile. Can be a `Number`, `Function(z)`, or `Field`. Default: `nothing` (`ZeroField`).
- `liquid_mass_fraction`: Initial qˡ profile. Default: `nothing` (`ZeroField`).
- `ice_mass_fraction`: Initial qⁱ profile. Default: `nothing` (`ZeroField`).

Pass `=0` to allocate an actual `Field` initialized to zero — required for later use
with [`compute_reference_state!`](@ref) or `set_to_mean!`.
"""
function ReferenceState(grid, constants=ThermodynamicConstants(eltype(grid));
                        surface_pressure = 101325,
                        potential_temperature = 288,
                        standard_pressure = 1e5,
                        vapor_mass_fraction = nothing,
                        liquid_mass_fraction = nothing,
                        ice_mass_fraction = nothing)

    FT = eltype(grid)
    p₀ = convert(FT, surface_pressure)
    θ₀ = convert(FT, potential_temperature)
    pˢᵗ = convert(FT, standard_pressure)
    loc = (nothing, nothing, Center())

    # Output fields: pressure and density
    ρ₀ = surface_density(p₀, θ₀, pˢᵗ, constants)
    ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ₀))
    ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)

    p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
    pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)

    # Input fields: temperature and moisture mass fractions
    Tᵣ = Field{Nothing, Nothing, Center}(grid)

    # Moisture mass fractions: ZeroField by default, actual Field when specified
    qᵛᵣ = reference_moisture_field(vapor_mass_fraction, grid)
    qˡᵣ = reference_moisture_field(liquid_mass_fraction, grid)
    qⁱᵣ = reference_moisture_field(ice_mass_fraction, grid)

    # Analytical dry adiabatic profiles for pressure and density.
    # These satisfy ρᵣ = pᵣ / (Rᵈ Tᵣ) exactly where Tᵣ = θ₀ (pᵣ/pˢᵗ)^κ,
    # ensuring zero buoyancy for the default initial condition θ = θ₀.
    set!(pᵣ, z -> adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants))
    set!(ρᵣ, z -> adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants))
    fill_halo_regions!(pᵣ)
    fill_halo_regions!(ρᵣ)

    # Temperature from the Exner function (consistent with pᵣ and ρᵣ)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    set!(Tᵣ, z -> θ₀ * (adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants) / pˢᵗ)^κ)
    fill_halo_regions!(Tᵣ)

    ref = ReferenceState(p₀, θ₀, pˢᵗ, pᵣ, ρᵣ, Tᵣ, qᵛᵣ, qˡᵣ, qⁱᵣ)

    return ref
end

#####
##### Helpers for moisture mass fraction fields
#####

reference_moisture_field(::Nothing, grid) = ZeroField(eltype(grid))

function reference_moisture_field(value, grid)
    field = Field{Nothing, Nothing, Center}(grid)
    set!(field, value)
    fill_halo_regions!(field)
    return field
end

# set! and fill_halo_regions! are no-ops for ZeroField
set_reference_field!(field, value) = (set!(field, value); fill_halo_regions!(field); nothing)
set_reference_field!(::ZeroField, value) = nothing

#####
##### Recompute reference state from profiles
#####

"""
    compute_reference_state!(reference_state, T̄, q̄ᵛ, q̄ˡ, q̄ⁱ, constants)

Recompute the reference pressure and density profiles by setting the
reference temperature to `T̄` and moisture mass fractions to `q̄ᵛ`, `q̄ˡ`, `q̄ⁱ`,
then integrating the hydrostatic equation using the mixture gas constant
`Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ` and ideal gas law `ρ = p / (Rᵐ T)`.

`T̄`, `q̄ᵛ`, `q̄ˡ`, `q̄ⁱ` can be `Number`s, `Function(z)`s, or `Field`s.

This function is useful for:
- Initialization: setting the reference state to match a non-constant-θ initial condition
- Runtime: calling from a callback to keep the reference state close to the evolving mean state
"""
function compute_reference_state!(ref::ReferenceState, T̄, q̄ᵛ, q̄ˡ, q̄ⁱ, constants)
    set!(ref.temperature, T̄)
    fill_halo_regions!(ref.temperature)
    set_reference_field!(ref.vapor_mass_fraction, q̄ᵛ)
    set_reference_field!(ref.liquid_mass_fraction, q̄ˡ)
    set_reference_field!(ref.ice_mass_fraction, q̄ⁱ)
    compute_hydrostatic_reference!(ref, constants)
    return nothing
end

"""
    compute_reference_state!(reference_state, T̄, q̄ᵗ, constants)

Convenience method that assumes all moisture is vapor (no condensate in the
reference state). Equivalent to `compute_reference_state!(reference_state, T̄, q̄ᵗ, 0, 0, constants)`.
"""
function compute_reference_state!(ref::ReferenceState, T̄, q̄ᵗ, constants)
    compute_reference_state!(ref, T̄, q̄ᵗ, 0, 0, constants)
end
