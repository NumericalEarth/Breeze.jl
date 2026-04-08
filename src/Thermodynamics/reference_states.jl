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

"""
    numerically_integrated_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants)

Compute the dry hydrostatic pressure at height ``z`` by numerically integrating
``∂p/∂z = -g ρ`` from ``z=0``, where ``ρ = p/(Rᵈ T)`` and ``T = θ(z) (p/pˢᵗ)^κ``.

This function handles non-uniform potential temperature profiles ``θ(z)`` for which
the closed-form adiabatic solution does not apply.
Uses 1000 midpoint integration steps.
"""
function numerically_integrated_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants)
    z == 0 && return p₀
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    g = constants.gravitational_acceleration
    nsteps = 1000
    dz = z / nsteps
    p = p₀
    for i in 1:nsteps
        zᵢ = (i - 0.5) * dz
        θᵢ = θ_func(zᵢ)
        Tᵢ = θᵢ * (p / pˢᵗ)^κ
        ρᵢ = p / (Rᵈ * Tᵢ)
        p = p - g * ρᵢ * dz
    end
    return p
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
    adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants)

hydrostatic_density(z, p₀, θ₀::Number, pˢᵗ, constants) =
    adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

function hydrostatic_temperature(z, p₀, θ₀::Number, pˢᵗ, constants)
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    p = adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants)
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

    # MPAS isothermal base state (init_atm_cases.F lines 813-817):
    #   ppb(k) = p0 * exp(-g * z_center / (Rᵈ * T₀))
    #   pb(k)  = (ppb(k) / p0)^κ
    #   rb(k)  = ppb(k) / (Rᵈ * T₀)
    #   tb(k)  = T₀ / pb(k)
    for k in 1:Nz
        @inbounds begin
            z_center = znode(1, 1, k, grid, Center(), Center(), Center())
            p_base = p₀ * exp(-g * z_center / (Rᵈ * T₀))
            Π_base = (p_base / pˢᵗ)^κ
            pᵣ[1, 1, k] = p_base
            π₀[1, 1, k] = Π_base
            ρᵣ[1, 1, k] = p_base / (Rᵈ * T₀)
            θᵣ[1, 1, k] = T₀ / Π_base
        end
    end
end

@kernel function _compute_exner_reference!(π₀, pᵣ, ρᵣ, θ₀, grid, Nz, π₀_surface, pˢᵗ, cᵖᵈ, κ, Rᵈ, g)
    _ = @index(Global)

    # MPAS-style integration: surface → top (up), then top → centers (down).
    # This matches MPAS init_atm_cases.F lines 1747-1772.
    #
    # Step 1: Integrate UP from surface (Π=π₀_surface) to model top.
    #   - Half-step from surface (z=0) to center k=1
    #   - Interior steps between adjacent centers (using face spacing)
    #   - Half-step from center Nz to model top

    @inbounds begin
        Δzᶜ₁ = Δzᶜᶜᶜ(1, 1, 1, grid)  # cell thickness at k=1 (= MPAS dzw(1))
        pi_top = π₀_surface - g * Δzᶜ₁ / (2 * cᵖᵈ * θ₀[1, 1, 1])
    end

    for k in 2:Nz
        Δzᶠₖ = Δzᶜᶜᶠ(1, 1, k, grid)  # center-to-center at face k (= MPAS dzu(k))
        @inbounds θ_face = (θ₀[1, 1, k] + θ₀[1, 1, k-1]) / 2
        pi_top = pi_top - g * Δzᶠₖ / (cᵖᵈ * θ_face)
    end

    @inbounds begin
        Δzᶜₙ = Δzᶜᶜᶜ(1, 1, Nz, grid)
        pi_top = pi_top - g * Δzᶜₙ / (2 * cᵖᵈ * θ₀[1, 1, Nz])
    end

    # Step 2: Integrate DOWN from top to get Π at each cell center.
    #   - Half-step from top to center Nz
    #   - Interior steps between adjacent centers (downward)
    @inbounds begin
        π₀[1, 1, Nz] = pi_top + g * Δzᶜₙ / (2 * cᵖᵈ * θ₀[1, 1, Nz])
    end

    for k in (Nz - 1):-1:1
        Δzᶠₖ₊₁ = Δzᶜᶜᶠ(1, 1, k + 1, grid)
        @inbounds θ_face = (θ₀[1, 1, k] + θ₀[1, 1, k + 1]) / 2
        @inbounds π₀[1, 1, k] = π₀[1, 1, k + 1] + g * Δzᶠₖ₊₁ / (cᵖᵈ * θ_face)
    end

    # Step 3: Derive pressure and density from π₀ (same as MPAS).
    # ρ = Π^(cᵥ/Rᵈ) × p₀ / (Rᵈ × θ)  (= MPAS rb = pb^(1/rcv) / ((rgas/p0)*tb))
    for k in 1:Nz
        @inbounds begin
            πᵏ = π₀[1, 1, k]
            pᵏ = pˢᵗ * πᵏ^(1/κ)
            Tᵏ = θ₀[1, 1, k] * πᵏ
            pᵣ[1, 1, k] = pᵏ
            ρᵣ[1, 1, k] = pᵏ / (Rᵈ * Tᵏ)
        end
    end
end

"""
$(TYPEDSIGNATURES)

Construct an `ExnerReferenceState` by discrete Exner integration on `grid`.

Two modes are supported, controlled by which keyword is provided:

**Isentropic** (`potential_temperature`): Constant or z-dependent θ₀.
The Exner function is built by MPAS-style up-then-down integration using
``ΔΠ = -g Δz / (cᵖᵈ θ₀^{face})``. Density: ``ρ₀ = p₀ / (Rᵈ θ₀ Π₀)``.

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
"""
function ExnerReferenceState(grid, constants=ThermodynamicConstants(eltype(grid));
                             surface_pressure = 101325,
                             potential_temperature = 288,
                             reference_temperature = nothing,
                             standard_pressure = 1e5)

    FT = eltype(grid)
    arch = architecture(grid)
    p₀ = convert(FT, surface_pressure)
    pˢᵗ = convert(FT, standard_pressure)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    g = constants.gravitational_acceleration
    Nz = size(grid, 3)

    if reference_temperature !== nothing
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
            # 3D reference: per-column integration for latitude-dependent θ₀(φ,z)
            θᵣ = CenterField(grid)
            set!(θᵣ, potential_temperature)
            fill_halo_regions!(θᵣ)

            πᵣ = CenterField(grid)
            pᵣ = CenterField(grid)
            ρᵣ = CenterField(grid)

            launch!(arch, grid, :xy, _compute_exner_reference_3d!,
                    πᵣ, pᵣ, ρᵣ, θᵣ, grid, p₀, pˢᵗ, cᵖᵈ, κ, Rᵈ, g)
        else
            # 1D reference: single column, broadcast to all (i,j)
            loc = (nothing, nothing, Center())
            θᵣ = Field{Nothing, Nothing, Center}(grid)
            set!(θᵣ, potential_temperature)
            fill_halo_regions!(θᵣ)

            πᵣ = Field{Nothing, Nothing, Center}(grid)
            π₀_surface = (p₀ / pˢᵗ)^κ
            θ₀_surface = convert(FT, _surface_value(potential_temperature))
            ρ₀_surface = p₀ / (Rᵈ * θ₀_surface * π₀_surface)

            p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
            pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
            ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ₀_surface))
            ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)

            launch!(arch, grid, tuple(1), _compute_exner_reference!,
                    πᵣ, pᵣ, ρᵣ, θᵣ, grid, Nz, π₀_surface, pˢᵗ, cᵖᵈ, κ, Rᵈ, g)
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

@kernel function _compute_exner_reference_3d!(π₀, pᵣ, ρᵣ, θ₀, grid, p₀, pˢᵗ, cᵖᵈ, κ, Rᵈ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    # MPAS-style up-then-down integration (same logic as 1D kernel).
    π₀_surface = (p₀ / pˢᵗ)^κ

    @inbounds begin
        # Step 1: UP — surface → center 1 → ... → center Nz → top
        Δzᶜ₁ = Δzᶜᶜᶜ(i, j, 1, grid)
        pi_top = π₀_surface - g * Δzᶜ₁ / (2 * cᵖᵈ * θ₀[i, j, 1])
    end

    for k in 2:Nz
        Δzᶠₖ = Δzᶜᶜᶠ(i, j, k, grid)
        @inbounds θ_face = (θ₀[i, j, k] + θ₀[i, j, k - 1]) / 2
        pi_top = pi_top - g * Δzᶠₖ / (cᵖᵈ * θ_face)
    end

    @inbounds begin
        Δzᶜₙ = Δzᶜᶜᶜ(i, j, Nz, grid)
        pi_top = pi_top - g * Δzᶜₙ / (2 * cᵖᵈ * θ₀[i, j, Nz])
    end

    # Step 2: DOWN — top → center Nz → ... → center 1
    @inbounds π₀[i, j, Nz] = pi_top + g * Δzᶜₙ / (2 * cᵖᵈ * θ₀[i, j, Nz])

    for k in (Nz - 1):-1:1
        Δzᶠₖ₊₁ = Δzᶜᶜᶠ(i, j, k + 1, grid)
        @inbounds θ_face = (θ₀[i, j, k] + θ₀[i, j, k + 1]) / 2
        @inbounds π₀[i, j, k] = π₀[i, j, k + 1] + g * Δzᶠₖ₊₁ / (cᵖᵈ * θ_face)
    end

    # Step 3: Derive p, ρ from π₀
    for k in 1:Nz
        @inbounds begin
            πᵏ = π₀[i, j, k]
            pᵏ = pˢᵗ * πᵏ^(1/κ)
            Tᵏ = θ₀[i, j, k] * πᵏ
            pᵣ[i, j, k] = pᵏ
            ρᵣ[i, j, k] = pᵏ / (Rᵈ * Tᵏ)
        end
    end
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
