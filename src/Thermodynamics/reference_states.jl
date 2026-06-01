using Oceananigans: Oceananigans, Center, CenterField, Field, set!, fill_halo_regions!
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, ValueBoundaryCondition
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶠ
using Oceananigans.Utils: launch!

using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar
using KernelAbstractions: @kernel, @index

#####
##### Reference state computations for Boussinesq and Anelastic models
#####

struct ReferenceState{FT, P, D, T, QV, QL, QI}
    surface_pressure :: FT # base pressure: reference pressure at z=0
    potential_temperature :: FT  # reference potential temperature at z=0
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

Compute the reference pressure at height `z` that associated with the reference pressure `p₀`,
potential temperature `θ₀`, and standard pressure `pˢᵗ`. The reference pressure is defined as
the pressure of dry air at the reference pressure and temperature.
"""
@inline function adiabatic_hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(constants)
    g = constants.gravitational_acceleration
    T₀ = θ₀ * (p₀ / pˢᵗ)^(Rᵈ / cᵖᵈ)
    return p₀ * (1 - g * z / (cᵖᵈ * T₀))^(cᵖᵈ / Rᵈ)
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
    pᵣ = adiabatic_hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)
    ρ₀ = surface_density(p₀, θ₀, pˢᵗ, constants)
    return ρ₀ * (pᵣ / p₀)^(1 - Rᵈ / cᵖᵈ)
end

"""
$(TYPEDSIGNATURES)

Return the density that keeps pressure unchanged when applying a potential-temperature
perturbation at fixed composition.

`pressure_balanced_density(ρ_background, θ_background, θ_initial)` applies the dry-air /
vapor-only relation, for which holding ``ρ θ`` fixed avoids seeding an acoustic pressure
perturbation.

For fixed-composition states with nonzero liquid or ice condensate, use
`pressure_balanced_density(ρ_background, θ_background, θ_initial, q, pᵣ, pˢᵗ, constants)`
instead. The condensate-aware method evaluates the full liquid-ice potential-temperature
equation of state before balancing density.

# Examples
```jldoctest
using Breeze.Thermodynamics: pressure_balanced_density

ρ_background = 1.0
θ_background = 300.0
θ_initial = 303.0
pressure_balanced_density(ρ_background, θ_background, θ_initial)

# output
0.9900990099009901
```
"""
@inline pressure_balanced_density(ρ_background, θ_background, θ_initial) =
    ρ_background * θ_background / θ_initial

@inline function pressure_balanced_density(ρ_background, θ_background, θ_initial,
                                          q::MoistureMassFractions, pᵣ, pˢᵗ, constants)
    T_background = temperature(LiquidIcePotentialTemperatureState(θ_background, q, pˢᵗ, pᵣ), constants)
    T_initial = temperature(LiquidIcePotentialTemperatureState(θ_initial, q, pˢᵗ, pᵣ), constants)
    return ρ_background * T_background / T_initial
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
$(TYPEDSIGNATURES)

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

# Dry hydrostatic balance is linear in the Exner function Π = (p / pˢᵗ)^κ,
# so integrating Π(z) directly avoids repeatedly evaluating the nonlinear EOS.
function converged_hydrostatic_integral(z, Π₀, dΠdz;
                                        tolerance = sqrt(eps(float(typeof(Π₀)))),
                                        initial_steps = 16,
                                        max_steps = 1 << 16)
    z == 0 && return Π₀

    integrate(nsteps) = begin
        dz = z / nsteps
        Π = Π₀
        for i in 1:nsteps
            zᵢ = (i - 0.5) * dz
            Π += dΠdz(zᵢ) * dz
        end
        return Π
    end

    nsteps = initial_steps
    Π_coarse = integrate(nsteps)
    while nsteps < max_steps
        nsteps *= 2
        Π_fine = integrate(nsteps)
        abs(Π_fine - Π_coarse) ≤ tolerance * abs(Π_fine) && return Π_fine
        Π_coarse = Π_fine
    end

    return Π_coarse
end

"""
    numerically_integrated_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants)

Compute the dry hydrostatic pressure at height ``z`` by numerically integrating
``∂p/∂z = -g ρ`` from ``z=0``, where ``ρ = p/(Rᵈ T)`` and ``T = θ(z) (p/pˢᵗ)^κ``.

This function handles non-uniform potential temperature profiles ``θ(z)`` for which
the closed-form adiabatic solution does not apply. The integration is carried out in
the dry Exner function ``Π = (p / pˢᵗ)^κ``, which satisfies the linear equation
``∂Π/∂z = -g / (cᵖᵈ θ(z))``.
"""
function numerically_integrated_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    g = constants.gravitational_acceleration
    Π₀ = (p₀ / pˢᵗ)^κ
    @inline dΠdz(zᵢ) = -g / (cᵖᵈ * θ_func(zᵢ))
    Π = converged_hydrostatic_integral(z, Π₀, dΠdz)
    return pˢᵗ * Π^(1 / κ)
end

"""
    numerically_integrated_hydrostatic_density(z, p₀, θ_func, pˢᵗ, constants)

Compute the dry hydrostatic density at height `z` from the numerically integrated pressure
and the given potential temperature profile `θ_func(z)`.
"""
function numerically_integrated_hydrostatic_density(z, p₀, θ_func, pˢᵗ, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    p = numerically_integrated_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants)
    θ = θ_func(z)
    T = θ * (p / pˢᵗ)^κ
    return p / (Rᵈ * T)
end

#####
##### Dispatch: select closed-form (constant θ₀) or numerical (θᵣ(z)) hydrostatic profiles
#####

# Closed-form for constant potential temperature
hydrostatic_pressure(z, p₀, θ₀::Number, pˢᵗ, constants) =
    adiabatic_hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)

hydrostatic_density(z, p₀, θ₀::Number, pˢᵗ, constants) =
    adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

function hydrostatic_temperature(z, p₀, θ₀::Number, pˢᵗ, constants)
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    p = adiabatic_hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)
    return θ₀ * (p / pˢᵗ)^κ
end

# Numerical integration for θᵣ(z) profiles
hydrostatic_pressure(z, p₀, θᵣ::Function, pˢᵗ, constants) =
    numerically_integrated_hydrostatic_pressure(z, p₀, θᵣ, pˢᵗ, constants)

hydrostatic_density(z, p₀, θᵣ::Function, pˢᵗ, constants) =
    numerically_integrated_hydrostatic_density(z, p₀, θᵣ, pˢᵗ, constants)

function hydrostatic_temperature(z, p₀, θᵣ::Function, pˢᵗ, constants)
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    p = numerically_integrated_hydrostatic_pressure(z, p₀, θᵣ, pˢᵗ, constants)
    return θᵣ(z) * (p / pˢᵗ)^κ
end

# Evaluate a profile (Number or Function) at a given height.
# Used both here and in terrain_compressible_physics.jl for reference state construction.
"""
    evaluate_profile(profile, z)

Evaluate a vertical profile at height `z`. If `profile` is a `Number`, returns it unchanged.
If `profile` is a `Function`, calls `profile(z)`.
"""
@inline evaluate_profile(value::Number, z) = value
@inline evaluate_profile(f::Function, z) = f(z)

# Surface value extraction. For 3-arg functions (lat, lon, z) used by the
# LatitudeLongitudeGrid reference state path, evaluate at the equator surface.
_surface_value(x::Number) = x
_surface_value(f::Function) = _nargs(f) == 1 ? f(0) : f(0, 0, 0)

"""
$(TYPEDSIGNATURES)

Return a `ReferenceState` on `grid`, with [`ThermodynamicConstants`](@ref) `constants`
that includes the hydrostatic reference pressure and reference density.

The reference state is initialized with a dry adiabatic temperature profile
and the given moisture profiles (zero by default). The pressure and density
are then computed by hydrostatic integration using the mixture gas constant
``Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ`` and the ideal gas law ``ρ = p / (Rᵐ T)``.

Arguments
=========
- `grid`: The grid.
- `constants :: ThermodynamicConstants`: By default, `ThermodynamicConstants(eltype(grid))`.

Keyword arguments
=================
- `surface_pressure`: By default, 101325.
- `potential_temperature`: A constant value (default 288) or a function ``θ(z)`` giving
  the potential temperature profile. When a constant is provided, closed-form adiabatic
  hydrostatic profiles are used. When a function is provided, the hydrostatic profiles
  are computed by numerical integration of ``∂p/∂z = -g ρ``.
- `standard_pressure`: Reference pressure for potential temperature (``pˢᵗ``). By default, 1e5.
- `discrete_hydrostatic_balance`: If `true`, recompute the reference pressure from the
  reference density using discrete integration, so that `∂z(p_ref) + g * ℑz(ρ_ref) = 0`
  exactly at the discrete level. By default, `false`.

  !!! note "Discrete vs continuous hydrostatic balance"
      With discrete balance, reference subtraction becomes a no-op (the subtracted terms
      cancel to machine precision). For split-explicit compressible dynamics, **continuous**
      balance (default) is preferred: both the actual and reference states share similar
      ``O(Δz^2)`` truncation errors that cancel in the perturbation PG, leaving only the
      tiny truncation error of the physical perturbation ``∂(p - p_{ref})/∂z``.
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
                        discrete_hydrostatic_balance = false,
                        vapor_mass_fraction = nothing,
                        liquid_mass_fraction = nothing,
                        ice_mass_fraction = nothing)

    FT = eltype(grid)
    p₀ = convert(FT, surface_pressure)
    pˢᵗ = convert(FT, standard_pressure)
    loc = (nothing, nothing, Center())

    # Moisture mass fractions: ZeroField by default, actual Field when specified
    qᵛᵣ = reference_moisture_field(vapor_mass_fraction, grid)
    qˡᵣ = reference_moisture_field(liquid_mass_fraction, grid)
    qⁱᵣ = reference_moisture_field(ice_mass_fraction, grid)

    θᵣ = potential_temperature
    θ₀ = convert(FT, _surface_value(θᵣ))
    ρ₀ = surface_density(p₀, θ₀, pˢᵗ, constants)

    ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ₀))
    ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)
    set!(ρᵣ, z -> hydrostatic_density(z, p₀, θᵣ, pˢᵗ, constants))
    fill_halo_regions!(ρᵣ)

    p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
    pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
    set!(pᵣ, z -> hydrostatic_pressure(z, p₀, θᵣ, pˢᵗ, constants))
    fill_halo_regions!(pᵣ)

    if discrete_hydrostatic_balance
        g = constants.gravitational_acceleration
        enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, grid, g)
    end

    Tᵣ = Field{Nothing, Nothing, Center}(grid)
    set!(Tᵣ, z -> hydrostatic_temperature(z, p₀, θᵣ, pˢᵗ, constants))
    fill_halo_regions!(Tᵣ)

    return ReferenceState(p₀, θ₀, pˢᵗ, pᵣ, ρᵣ, Tᵣ, qᵛᵣ, qˡᵣ, qⁱᵣ)
end

#####
##### ExnerReferenceState: built in Exner coordinates for split-explicit compressible dynamics
#####

"""
    ExnerReferenceState

A dry reference state built in Exner coordinates, ensuring that the discrete Exner
hydrostatic balance

```math
cᵖᵈ θᵣ^{face} \\frac{π₀[k] - π₀[k-1]}{Δz} = -g
```

holds _exactly_ at every interior z-face. This is essential for the Exner pressure
acoustic substepping formulation, where the vertical pressure gradient is computed
as ``cᵖᵈ θᵥ ∂π'/∂z`` and the hydrostatic part must cancel to machine precision.

Unlike [`ReferenceState`](@ref) which builds pressure first and derives Exner,
this type builds the Exner function π₀ first by discrete integration and then
derives pressure and density from it. This matches CM1's approach where `pi0`
is the fundamental reference variable.

Fields
======

- `surface_pressure`: Reference pressure at z=0 (Pa)
- `surface_potential_temperature`: Reference potential temperature at z=0 (K)
- `standard_pressure`: pˢᵗ for potential temperature definition (Pa)
- `pressure`: Reference pressure field ``p₀ = pˢᵗ π₀^{cᵖᵈ/Rᵈ}`` (derived from π₀)
- `density`: Reference density field ``ρ₀ = p₀/(Rᵈ T₀)`` (derived from π₀ and θᵣ)
- `exner_function`: Reference Exner function π₀ (built by discrete integration)
"""
struct ExnerReferenceState{FT, FP, FD, FE}
    surface_pressure :: FT
    surface_potential_temperature :: FT
    standard_pressure :: FT
    pressure :: FP
    density :: FD
    exner_function :: FE
end

Adapt.adapt_structure(to, ref::ExnerReferenceState) =
    ExnerReferenceState(adapt(to, ref.surface_pressure),
                        adapt(to, ref.surface_potential_temperature),
                        adapt(to, ref.standard_pressure),
                        adapt(to, ref.pressure),
                        adapt(to, ref.density),
                        adapt(to, ref.exner_function))

Base.eltype(::ExnerReferenceState{FT}) where FT = FT

function Base.summary(ref::ExnerReferenceState)
    FT = eltype(ref)
    return string("ExnerReferenceState{$FT}(p₀=", prettysummary(ref.surface_pressure),
                  ", θ₀=", prettysummary(ref.surface_potential_temperature),
                  ", pˢᵗ=", prettysummary(ref.standard_pressure), ")")
end

Base.show(io::IO, ref::ExnerReferenceState) = print(io, summary(ref))

@kernel function _compute_isothermal_reference!(π₀, pᵣ, ρᵣ, θᵣ, grid, Nz, p₀, pˢᵗ, κ, Rᵈ, g, T₀)
    _ = @index(Global)

    # Discrete-balance recurrence. Enforces
    #
    #   (p[k] − p[k − 1]) / Δz_face[k] + g · (ρ[k] + ρ[k − 1]) / 2 = 0
    #
    # exactly, where ρ[k] = p[k] / (Rᵈ T₀), so the substepper's slow
    # vertical-momentum tendency is zero on a rest atmosphere to
    # floating-point precision. The previous MPAS-form analytic
    # solution `p(z) = p₀ exp(-g z / (Rᵈ T₀))` satisfies the
    # *continuous* hydrostatic equation but leaves a discrete-operator
    # residual ~1e-3 N/m³, which seeds an acoustic instability at
    # production Δt.
    #
    # For constant T₀, the recurrence has the closed form
    #   p[k] = p[k − 1] · (1 − a) / (1 + a),  a = g Δz_face / (2 Rᵈ T₀).

    # Anchor at the first cell center via the continuous analytic
    # solution `p(z) = p₀ exp(-gz / (Rᵈ T₀))`. Face k = 1 is a
    # boundary face (ρw[1] = 0 by impenetrability) and the substepper
    # does NOT apply the discrete-balance check there, so we are free
    # to anchor however we like; the continuous formula keeps
    # `surface_pressure = p₀` semantics intact and matches what users
    # set with `θ = θᵇᵍ(z)`.
    z₁ = znode(1, 1, 1, grid, Center(), Center(), Center())
    p¹ = p₀ * exp(-g * z₁ / (Rᵈ * T₀))
    ρ¹ = p¹ / (Rᵈ * T₀)
    Π¹ = (p¹ / pˢᵗ)^κ
    @inbounds begin
        pᵣ[1, 1, 1] = p¹
        π₀[1, 1, 1] = Π¹
        ρᵣ[1, 1, 1] = ρ¹
        θᵣ[1, 1, 1] = T₀ / Π¹
    end
    p⁻ = p¹
    ρ⁻ = ρ¹

    # Discrete-balance recurrence for k = 2..Nz. Each iteration gives
    # `(p[k] − p[k − 1]) / Δz_face[k] + g · (ρ[k] + ρ[k − 1]) / 2 = 0`
    # at machine precision, so the substepper's slow vertical-momentum
    # tendency is zero on a rest atmosphere to ulp.
    for k in 2:Nz
        Δz_face = Δzᶜᶜᶠ(1, 1, k, grid)
        a = g * Δz_face / (2 * Rᵈ * T₀)
        pᵏ = (p⁻ - g * Δz_face * ρ⁻ / 2) / (1 + a)
        ρᵏ = pᵏ / (Rᵈ * T₀)
        Πᵏ = (pᵏ / pˢᵗ)^κ
        @inbounds begin
            pᵣ[1, 1, k] = pᵏ
            π₀[1, 1, k] = Πᵏ
            ρᵣ[1, 1, k] = ρᵏ
            θᵣ[1, 1, k] = T₀ / Πᵏ
        end
        p⁻ = pᵏ
        ρ⁻ = ρᵏ
    end
end

# Discrete-balance Exner integration for one column (i, j) of θ̄ and an
# optional vapor profile qᵛ. Enforces
#   (p[k] − p[k − 1]) / Δz_face[k] + g · (ρ[k] + ρ[k − 1]) / 2 = 0
# at every interior face k = 2..Nz, with the level-local moist EOS
# ρ[k] = p[k] / (Rᵐ[k] T[k]), T[k] = θ̄[k] · Π[k], Π[k] = (p/pˢᵗ)^κᵐ[k],
# κᵐ = Rᵐ/cᵖᵐ, Rᵐ = (1-qᵛ) Rᵈ + qᵛ Rᵛ, cᵖᵐ = (1-qᵛ) cᵖᵈ + qᵛ cᵖᵛ. The dry
# case is recovered exactly when qᵛ ≡ 0.
#
# The previous MPAS-style up-then-down Π integration satisfies the
# *continuous* hydrostatic equation but leaves a discrete-operator
# residual ~1e-3 N/m³ that seeds an acoustic instability at production Δt.
@inline function _compute_exner_column!(π₀, pᵣ, ρᵣ, θ₀, qᵛ, i, j, grid, Nz, p₀, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
    # Anchor at first cell center via the continuous Π recurrence (one
    # half-step from surface). Face k = 1 is the impenetrability boundary
    # face — no discrete-balance constraint applies — so the anchor is free.
    @inbounds begin
        qᵛ¹ = qᵛ[i, j, 1]
        qᵈ¹ = 1 - qᵛ¹
        Rᵐ¹ = qᵈ¹ * Rᵈ + qᵛ¹ * Rᵛ
        cᵖᵐ¹ = qᵈ¹ * cᵖᵈ + qᵛ¹ * cᵖᵛ
        κ¹ = Rᵐ¹ / cᵖᵐ¹
        π₀_surface = (p₀ / pˢᵗ)^κ¹

        Δzᶜ₁ = Δzᶜᶜᶜ(i, j, 1, grid)
        θ¹ = θ₀[i, j, 1]
        Π¹ = π₀_surface - g * Δzᶜ₁ / (2 * cᵖᵐ¹ * θ¹)
        p¹ = pˢᵗ * Π¹^(1/κ¹)
        ρ¹ = p¹ / (Rᵐ¹ * θ¹ * Π¹)

        π₀[i, j, 1] = Π¹
        pᵣ[i, j, 1] = p¹
        ρᵣ[i, j, 1] = ρ¹
    end
    p⁻ = p¹
    ρ⁻ = ρ¹

    # Discrete-balance recurrence for k = 2..Nz. The residual
    #   F(p) = (p − p⁻) / Δz_face + g · (ρ(p) + ρ⁻) / 2
    # with ρ(p) = p^(1−κᵏ) · pˢᵗ^κᵏ / (Rᵐᵏ θ̄[k]) is monotone increasing in
    # p, so Newton converges in O(few) iterations from the continuous-Π guess.
    for k in 2:Nz
        Δz_face = Δzᶜᶜᶠ(i, j, k, grid)
        @inbounds begin
            θᵏ = θ₀[i, j, k]
            θᵏ⁻ = θ₀[i, j, k - 1]
            qᵛᵏ = qᵛ[i, j, k]
        end
        qᵈᵏ = 1 - qᵛᵏ
        Rᵐᵏ = qᵈᵏ * Rᵈ + qᵛᵏ * Rᵛ
        cᵖᵐᵏ = qᵈᵏ * cᵖᵈ + qᵛᵏ * cᵖᵛ
        κᵏ = Rᵐᵏ / cᵖᵐᵏ

        # Initial guess: continuous Π integration (one face step).
        θ_face = (θᵏ + θᵏ⁻) / 2
        Πᵏ_init = @inbounds(π₀[i, j, k - 1]) - g * Δz_face / (cᵖᵐᵏ * θ_face)
        pᵏ = pˢᵗ * Πᵏ_init^(1/κᵏ)

        Aᵏ = g * pˢᵗ^κᵏ / (2 * Rᵐᵏ * θᵏ)
        Cᵏ = p⁻ / Δz_face - g * ρ⁻ / 2
        # Newton iterations on F(p) = p / Δz_face + Aᵏ · p^(1−κᵏ) − Cᵏ.
        for _ in 1:5
            ρp = pᵏ^(-κᵏ)               # = p^(1-κᵏ) / p
            f  = pᵏ / Δz_face + Aᵏ * pᵏ * ρp - Cᵏ
            f′ = 1 / Δz_face + Aᵏ * (1 - κᵏ) * ρp
            pᵏ = pᵏ - f / f′
        end
        Πᵏ = (pᵏ / pˢᵗ)^κᵏ
        ρᵏ = pᵏ / (Rᵐᵏ * θᵏ * Πᵏ)
        @inbounds begin
            π₀[i, j, k] = Πᵏ
            pᵣ[i, j, k] = pᵏ
            ρᵣ[i, j, k] = ρᵏ
        end
        p⁻ = pᵏ
        ρ⁻ = ρᵏ
    end
end

@kernel function _compute_exner_reference!(π₀, pᵣ, ρᵣ, θ₀, qᵛ, grid, Nz, p₀, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
    _ = @index(Global)
    _compute_exner_column!(π₀, pᵣ, ρᵣ, θ₀, qᵛ, 1, 1, grid, Nz, p₀, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
end

"""
$(TYPEDSIGNATURES)

Construct an `ExnerReferenceState` by discrete Exner integration on `grid`.

Two modes are supported, controlled by which keyword is provided:

**Isentropic** (`potential_temperature`): Constant or horizontally-varying θ₀.
Each column is built by Newton iteration on the discrete hydrostatic balance
``(p_k - p_{k-1})/Δz_{face} + g(ρ_k + ρ_{k-1})/2 = 0`` so the substepper's
slow vertical-momentum tendency vanishes to ulp on a rest atmosphere. The
same column kernel handles both the 1D path (`θ₀` constant or z-dependent)
and the 3D path (`θ₀(x, y, z)`); `vapor_mass_fraction` is supported in both.
When provided, the level-local moist gas constants ``Rᵐ = (1-qᵛ)Rᵈ + qᵛRᵛ``,
``cᵖᵐ = (1-qᵛ)cᵖᵈ + qᵛcᵖᵛ`` are used; the dry case is recovered exactly
when ``qᵛ ≡ 0``.

**Isothermal** (`reference_temperature`): Constant T₀ (MPAS baroclinic wave
convention). Uses the analytic isothermal solution:
``p₀(z) = pˢ \\exp(-g z / (Rᵈ T₀))``, ``Π₀ = (p₀/pˢᵗ)^κ``,
``ρ₀ = p₀/(Rᵈ T₀)``, ``θ₀ = T₀/Π₀``.
This matches MPAS init_atm_cases.F lines 813-817 exactly.

Arguments
=========
- `grid`: The grid
- `constants`: Thermodynamic constants (default: `ThermodynamicConstants(eltype(grid))`)

Keyword Arguments
=================
- `surface_pressure`: Pressure at z=0 (default: 101325 Pa)
- `potential_temperature`: Constant value or function `θᵣ(z)` for isentropic reference (default: 288 K)
- `reference_temperature`: Constant T₀ for isothermal reference (default: `nothing`).
  When provided, overrides `potential_temperature`.
- `standard_pressure`: pˢᵗ for potential temperature definition (default: 1e5 Pa)
- `vapor_mass_fraction`: Optional vapor mass fraction for a moist reference state.
  A number or function `qᵛ(z)` builds a 1D column; a multi-argument function
  `qᵛ(x, y, z)` (or `qᵛ(φ, z)` on a `LatitudeLongitudeGrid`) builds a 3D field.
"""
function ExnerReferenceState(grid, constants=ThermodynamicConstants(eltype(grid));
                             surface_pressure = 101325,
                             potential_temperature = 288,
                             reference_temperature = nothing,
                             standard_pressure = 1e5,
                             vapor_mass_fraction = nothing)

    FT = eltype(grid)
    arch = architecture(grid)
    p₀ = convert(FT, surface_pressure)
    pˢᵗ = convert(FT, standard_pressure)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    κ = Rᵈ / cᵖᵈ
    g = constants.gravitational_acceleration
    Nz = size(grid, 3)

    if reference_temperature !== nothing
        vapor_mass_fraction === nothing ||
            throw(ArgumentError("`vapor_mass_fraction` is not supported with `reference_temperature`."))

        # ── Isothermal base state (MPAS baroclinic wave convention) ──
        # Analytic solution: p(z) = p₀ exp(-gz/(Rᵈ T₀)), Π = (p/pˢᵗ)^κ,
        # ρ = p/(Rᵈ T₀), θ = T₀/Π.  (MPAS init_atm_cases.F lines 813-817)
        T₀ = convert(FT, reference_temperature)
        loc = (nothing, nothing, Center())

        θᵣ = Field{Nothing, Nothing, Center}(grid)
        πᵣ = Field{Nothing, Nothing, Center}(grid)
        pᵣ = Field{Nothing, Nothing, Center}(grid)
        ρ₀_surface = p₀ / (Rᵈ * T₀)
        p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
        pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
        ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ₀_surface))
        ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)

        launch!(arch, grid, tuple(1), _compute_isothermal_reference!,
                πᵣ, pᵣ, ρᵣ, θᵣ, grid, Nz, p₀, pˢᵗ, κ, Rᵈ, g, T₀)
    else
        # ── Isentropic base state (constant or z-dependent θ₀) ──
        # Detect whether θ₀ depends on horizontal coordinates (3D reference).
        needs_3d = _needs_3d_reference(potential_temperature)

        if needs_3d
            # 3D reference: per-column discrete-balance integration for
            # horizontally varying θ₀(x, y, z). Same kernel as the 1D path,
            # indexed by (i, j); dry case (qᵛ = ZeroField) and moist case
            # share a single code path.
            θᵣ = CenterField(grid)
            set!(θᵣ, potential_temperature)
            fill_halo_regions!(θᵣ)

            πᵣ = CenterField(grid)
            pᵣ = CenterField(grid)
            ρᵣ = CenterField(grid)

            qᵛᵣ = vapor_mass_fraction === nothing ? ZeroField(FT) :
                  let qf = CenterField(grid)
                      set!(qf, vapor_mass_fraction)
                      fill_halo_regions!(qf)
                      qf
                  end

            launch!(arch, grid, :xy, _compute_exner_reference_3d!,
                    πᵣ, pᵣ, ρᵣ, θᵣ, qᵛᵣ, grid, Nz, p₀, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
        else
            # 1D reference: single column, broadcast to all (i,j). The unified
            # kernel handles both dry (qᵛ = ZeroField) and moist cases via the
            # moist gas constants Rᵐ, cᵖᵐ, κᵐ — these reduce to Rᵈ, cᵖᵈ, κ
            # when qᵛ = 0.
            loc = (nothing, nothing, Center())
            θᵣ = Field{Nothing, Nothing, Center}(grid)
            set!(θᵣ, potential_temperature)
            fill_halo_regions!(θᵣ)

            πᵣ = Field{Nothing, Nothing, Center}(grid)
            θ₀_surface = convert(FT, _surface_value(potential_temperature))

            qᵛᵣ = reference_moisture_field(vapor_mass_fraction, grid)
            qᵛ_surface = @allowscalar qᵛᵣ[1, 1, 1]
            Rᵐ_surface = (1 - qᵛ_surface) * Rᵈ + qᵛ_surface * Rᵛ
            cᵖᵐ_surface = (1 - qᵛ_surface) * cᵖᵈ + qᵛ_surface * cᵖᵛ
            π₀_surface = (p₀ / pˢᵗ)^(Rᵐ_surface / cᵖᵐ_surface)
            ρ₀_surface = p₀ / (Rᵐ_surface * θ₀_surface * π₀_surface)

            p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
            pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
            ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ₀_surface))
            ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)

            launch!(arch, grid, tuple(1), _compute_exner_reference!,
                    πᵣ, pᵣ, ρᵣ, θᵣ, qᵛᵣ, grid, Nz, p₀, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
        end
    end

    fill_halo_regions!(θᵣ)
    fill_halo_regions!(πᵣ)
    fill_halo_regions!(pᵣ)
    fill_halo_regions!(ρᵣ)

    # Surface θ₀ for the struct: for isothermal, θ_surface = T₀/Π_surface = T₀
    θ₀_val = if reference_temperature !== nothing
        convert(FT, reference_temperature)  # T₀ (at surface Π≈1, θ≈T)
    else
        convert(FT, _surface_value(potential_temperature))
    end

    return ExnerReferenceState(p₀, θ₀_val, pˢᵗ, pᵣ, ρᵣ, πᵣ)
end

# Detect if θ₀ needs a 3D (per-column) reference.
# Functions with ≥2 methods or ≥2 arguments → 3D.
_needs_3d_reference(::Number) = false
_needs_3d_reference(f::Function) = _nargs(f) > 1
_nargs(f) = maximum(m.nargs for m in methods(f)) - 1  # subtract 1 for the function itself

@kernel function _compute_exner_reference_3d!(π₀, pᵣ, ρᵣ, θ₀, qᵛ, grid, Nz, p₀, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
    i, j = @index(Global, NTuple)
    _compute_exner_column!(π₀, pᵣ, ρᵣ, θ₀, qᵛ, i, j, grid, Nz, p₀, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
end

# ExnerReferenceState has the same surface_density interface as ReferenceState
function surface_density(ref::ExnerReferenceState)
    ρ = ref.density
    grid = ρ.grid
    return @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, ρ)
end

@kernel function _enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, grid, Nz, g)
    _ = @index(Global)

    # Integrate upward from k=2, enforcing balance at each interior face.
    # pᵣ[1] is kept at its analytic value (closest to the surface boundary condition).
    for k in 2:Nz
        Δzᶠ = Δzᶜᶜᶠ(1, 1, k, grid)
        @inbounds ρᶠ = (ρᵣ[1, 1, k] + ρᵣ[1, 1, k - 1]) / 2
        @inbounds pᵣ[1, 1, k] = pᵣ[1, 1, k - 1] - g * Δzᶠ * ρᶠ
    end
end

"""
    enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, grid, g)

Recompute the reference pressure `pᵣ` from the reference density `ρᵣ` by discrete
upward integration, ensuring that the discrete hydrostatic balance

```math
\\frac{p_{ref}[k] - p_{ref}[k-1]}{Δz} + g \\frac{ρ_{ref}[k] + ρ_{ref}[k-1]}{2} = 0
```

holds exactly at every interior z-face. This guarantees that reference-state subtraction
in the pressure gradient and buoyancy cancels to machine precision, eliminating the
``O(Δz^2)`` truncation error that would otherwise dominate the momentum tendency
for nearly-hydrostatic flows.
"""
function enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, grid, g)
    arch = architecture(grid)
    Nz = size(grid, 3)
    launch!(arch, grid, tuple(1), _enforce_discrete_hydrostatic_balance!, pᵣ, ρᵣ, grid, Nz, g)
    fill_halo_regions!(pᵣ)
    return nothing
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
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Convenience method that assumes all moisture is vapor (no condensate in the
reference state). Equivalent to `compute_reference_state!(reference_state, T̄, q̄ᵗ, 0, 0, constants)`.
"""
function compute_reference_state!(ref::ReferenceState, T̄, q̄ᵗ, constants)
    compute_reference_state!(ref, T̄, q̄ᵗ, 0, 0, constants)
end
