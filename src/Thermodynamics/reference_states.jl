using Oceananigans: Oceananigans, Center, Field, set!, fill_halo_regions!
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, ValueBoundaryCondition
using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶠ

using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar

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
    general_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants)

Compute the hydrostatic pressure at height `z` by numerically integrating
`dp/dz = -gρ` from `z=0`, where `ρ = p/(Rᵈ T)` and `T = θ(z) (p/pˢᵗ)^κ`.
Uses 1000 midpoint Euler steps for accuracy.
"""
function general_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants)
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
    general_hydrostatic_density(z, p₀, θ_func, pˢᵗ, constants)

Compute the hydrostatic density at height `z` from the numerically integrated pressure
and the given potential temperature profile `θ_func(z)`.
"""
function general_hydrostatic_density(z, p₀, θ_func, pˢᵗ, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    p = general_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants)
    θ = θ_func(z)
    T = θ * (p / pˢᵗ)^κ
    return p / (Rᵈ * T)
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
- `potential_temperature`: A constant value (default 288) or a function `θ(z)` giving
  the potential temperature profile. When a constant is provided, closed-form adiabatic
  hydrostatic profiles are used. When a function is provided, the hydrostatic profiles
  are computed by numerical integration of `dp/dz = -gρ`.
- `standard_pressure`: Reference pressure for potential temperature (pˢᵗ). By default, 1e5.
- `discrete_hydrostatic_balance`: If `true`, recompute the reference pressure from the
  reference density using discrete integration, so that `∂z(p_ref) + g * ℑz(ρ_ref) = 0`
  exactly at the discrete level. By default, `false`.

  !!! note "Discrete vs continuous hydrostatic balance"
      With discrete balance, reference subtraction becomes a no-op (the subtracted terms
      cancel to machine precision). For split-explicit compressible dynamics, **continuous**
      balance (default) is preferred: both the actual and reference states share similar
      ``O(Δz^2)`` truncation errors that cancel in the perturbation PG, leaving only the
      tiny truncation error of the physical perturbation ``∂(p - p_{ref})/∂z``.
"""
function ReferenceState(grid, constants=ThermodynamicConstants(eltype(grid));
                        surface_pressure = 101325,
                        potential_temperature = 288,
                        standard_pressure = 1e5,
                        discrete_hydrostatic_balance = false)

    FT = eltype(grid)
    p₀ = convert(FT, surface_pressure)
    pˢᵗ = convert(FT, standard_pressure)
    loc = (nothing, nothing, Center())

    if potential_temperature isa Function
        θ_func = potential_temperature
        θ₀_value = convert(FT, θ_func(0))

        ρ_surface = surface_density(p₀, θ₀_value, pˢᵗ, constants)
        ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ_surface))
        ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)
        set!(ρᵣ, z -> general_hydrostatic_density(z, p₀, θ_func, pˢᵗ, constants))
        fill_halo_regions!(ρᵣ)

        p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
        pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
        set!(pᵣ, z -> general_hydrostatic_pressure(z, p₀, θ_func, pˢᵗ, constants))
        fill_halo_regions!(pᵣ)

        if discrete_hydrostatic_balance
            g = constants.gravitational_acceleration
            enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, g)
        end
    else
        θ₀ = convert(FT, potential_temperature)
        θ₀_value = θ₀

        ρ₀ = surface_density(p₀, θ₀, pˢᵗ, constants)
        ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ₀))
        ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)
        set!(ρᵣ, z -> adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants))
        fill_halo_regions!(ρᵣ)

        p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
        pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
        set!(pᵣ, z -> adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants))
        fill_halo_regions!(pᵣ)

        if discrete_hydrostatic_balance
            g = constants.gravitational_acceleration
            enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, g)
        end
    end

    return ReferenceState(p₀, θ₀_value, pˢᵗ, pᵣ, ρᵣ)
end

#####
##### ExnerReferenceState: built in Exner coordinates for split-explicit compressible dynamics
#####

"""
    ExnerReferenceState

A reference state built in Exner coordinates, ensuring that the discrete Exner
hydrostatic balance

```math
cᵖ θ₀^{face} \\frac{π₀[k] - π₀[k-1]}{Δz} = -g
```

holds EXACTLY at every interior z-face. This is essential for the Exner pressure
acoustic substepping formulation, where the vertical pressure gradient is computed
as ``cᵖ θᵥ ∂π'/∂z`` and the hydrostatic part must cancel to machine precision.

Unlike [`ReferenceState`](@ref) which builds pressure first and derives Exner,
this type builds the Exner function π₀ first by discrete integration and then
derives pressure and density from it. This matches CM1's approach where `pi0`
is the fundamental reference variable.

Fields
======

- `surface_pressure`: Reference pressure at z=0 (Pa)
- `potential_temperature`: Reference potential temperature (constant or surface value)
- `standard_pressure`: pˢᵗ for potential temperature definition (Pa)
- `pressure`: Reference pressure field p₀ = pˢᵗ π₀^(cᵖ/R) (derived from π₀)
- `density`: Reference density field ρ₀ = p₀/(R T₀) (derived from π₀ and θ₀)
- `exner`: Reference Exner function π₀ (built by discrete integration)
- `potential_temperature_field`: Reference potential temperature θ₀(z) (z-only field)
"""
struct ExnerReferenceState{FT, FP, FD, FE, FT2}
    surface_pressure :: FT
    potential_temperature :: FT
    standard_pressure :: FT
    pressure :: FP
    density :: FD
    exner :: FE
    potential_temperature_field :: FT2
end

Adapt.adapt_structure(to, ref::ExnerReferenceState) =
    ExnerReferenceState(adapt(to, ref.surface_pressure),
                        adapt(to, ref.potential_temperature),
                        adapt(to, ref.standard_pressure),
                        adapt(to, ref.pressure),
                        adapt(to, ref.density),
                        adapt(to, ref.exner),
                        adapt(to, ref.potential_temperature_field))

Base.eltype(::ExnerReferenceState{FT}) where FT = FT

function Base.summary(ref::ExnerReferenceState)
    FT = eltype(ref)
    return string("ExnerReferenceState{$FT}(p₀=", prettysummary(ref.surface_pressure),
                  ", θ₀=", prettysummary(ref.potential_temperature),
                  ", pˢᵗ=", prettysummary(ref.standard_pressure), ")")
end

Base.show(io::IO, ref::ExnerReferenceState) = print(io, summary(ref))

"""
$(TYPEDSIGNATURES)

Construct an `ExnerReferenceState` by discrete Exner integration on `grid`.

The Exner function π₀ is built by integrating upward from the surface:
```math
π₀[k] = π₀[k-1] - \\frac{g \\, Δz}{cᵖ \\, θ₀^{face}[k]}
```
where ``θ₀^{face}`` is the face-averaged reference potential temperature.
This ensures the discrete Exner hydrostatic balance is exact.

Pressure and density are then derived from π₀:
- ``p₀ = pˢᵗ \\, π₀^{cᵖ/R}``
- ``T₀ = θ₀ \\, π₀``
- ``ρ₀ = p₀ / (R \\, T₀)``

Arguments
=========
- `grid`: The grid
- `constants`: Thermodynamic constants (default: `ThermodynamicConstants(eltype(grid))`)

Keyword Arguments
=================
- `surface_pressure`: Pressure at z=0 (default: 101325 Pa)
- `potential_temperature`: Constant θ₀ or function θ₀(z) (default: 288 K)
- `standard_pressure`: pˢᵗ for potential temperature definition (default: 1e5 Pa)
"""
function ExnerReferenceState(grid, constants=ThermodynamicConstants(eltype(grid));
                             surface_pressure = 101325,
                             potential_temperature = 288,
                             standard_pressure = 1e5)

    FT = eltype(grid)
    p₀ = convert(FT, surface_pressure)
    pˢᵗ = convert(FT, standard_pressure)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    g = constants.gravitational_acceleration

    loc = (nothing, nothing, Center())

    # Build θ₀ field
    θ₀_field = Field{Nothing, Nothing, Center}(grid)
    if potential_temperature isa Function
        θ_func = potential_temperature
        θ₀_value = convert(FT, θ_func(0))
        set!(θ₀_field, z -> θ_func(z))
    else
        θ₀_value = convert(FT, potential_temperature)
        set!(θ₀_field, z -> θ₀_value)
    end
    fill_halo_regions!(θ₀_field)

    # Build π₀ by discrete upward integration
    π₀_field = Field{Nothing, Nothing, Center}(grid)
    Nz = size(grid, 3)

    # Bottom value from surface pressure
    π₀_surface = (p₀ / pˢᵗ)^κ
    @allowscalar π₀_field[1, 1, 1] = π₀_surface

    # Integrate upward: π₀[k] = π₀[k-1] - g Δz / (cᵖ avg(θ₀))
    @allowscalar for k in 2:Nz
        Δz_face = Δzᶜᶜᶠ(1, 1, k, grid)
        θ₀_face = (θ₀_field[1, 1, k] + θ₀_field[1, 1, k-1]) / 2
        π₀_field[1, 1, k] = π₀_field[1, 1, k-1] - g * Δz_face / (cᵖᵈ * θ₀_face)
    end
    fill_halo_regions!(π₀_field)

    # Derive pressure: p₀ = pˢᵗ π₀^(cᵖ/R)
    p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
    pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
    @allowscalar for k in 1:Nz
        pᵣ[1, 1, k] = pˢᵗ * π₀_field[1, 1, k]^(1/κ)
    end
    fill_halo_regions!(pᵣ)

    # Derive density: ρ₀ = p₀ / (Rᵈ T₀) where T₀ = θ₀ π₀
    ρ_surface = p₀ / (Rᵈ * θ₀_value * π₀_surface)
    ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ_surface))
    ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)
    @allowscalar for k in 1:Nz
        T₀ = θ₀_field[1, 1, k] * π₀_field[1, 1, k]
        ρᵣ[1, 1, k] = pᵣ[1, 1, k] / (Rᵈ * T₀)
    end
    fill_halo_regions!(ρᵣ)

    return ExnerReferenceState(p₀, θ₀_value, pˢᵗ, pᵣ, ρᵣ, π₀_field, θ₀_field)
end

# ExnerReferenceState has the same surface_density interface as ReferenceState
function surface_density(ref::ExnerReferenceState)
    ρ = ref.density
    grid = ρ.grid
    return @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, ρ)
end

"""
    enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, g)

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
function enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, g)
    grid = pᵣ.grid
    Nz = size(grid, 3)

    # Integrate upward from k=1, enforcing balance at each interior face k=2,...,Nz.
    # pᵣ[1] is kept at its analytic value (closest to the surface boundary condition).
    @allowscalar for k in 2:Nz
        Δz_face = Δzᶜᶜᶠ(1, 1, k, grid)
        ρ_avg = (ρᵣ[1, 1, k] + ρᵣ[1, 1, k - 1]) / 2
        pᵣ[1, 1, k] = pᵣ[1, 1, k - 1] - g * Δz_face * ρ_avg
    end

    fill_halo_regions!(pᵣ)

    return nothing
end
