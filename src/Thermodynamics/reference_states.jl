using Oceananigans: Oceananigans, Center, Field, set!, fill_halo_regions!
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, ValueBoundaryCondition
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: znode
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

#####
##### Dispatch: select closed-form (constant θ₀) or numerical (θ₀(z)) hydrostatic profiles
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

# Numerical integration for θ₀(z) profiles
hydrostatic_pressure(z, p₀, θ₀::Function, pˢᵗ, constants) =
    general_hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)

hydrostatic_density(z, p₀, θ₀::Function, pˢᵗ, constants) =
    general_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

function hydrostatic_temperature(z, p₀, θ₀::Function, pˢᵗ, constants)
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    p = general_hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)
    return θ₀(z) * (p / pˢᵗ)^κ
end

# Surface value extraction
_surface_value(x::Number) = x
_surface_value(f::Function) = f(0)

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

    θ₀ = potential_temperature
    θ₀_surface = convert(FT, _surface_value(θ₀))
    ρ_surface = surface_density(p₀, θ₀_surface, pˢᵗ, constants)

    ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ_surface))
    ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)
    set!(ρᵣ, z -> hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants))
    fill_halo_regions!(ρᵣ)

    p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
    pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
    set!(pᵣ, z -> hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants))
    fill_halo_regions!(pᵣ)

    if discrete_hydrostatic_balance
        g = constants.gravitational_acceleration
        enforce_discrete_hydrostatic_balance!(pᵣ, ρᵣ, g)
    end

    Tᵣ = Field{Nothing, Nothing, Center}(grid)
    set!(Tᵣ, z -> hydrostatic_temperature(z, p₀, θ₀, pˢᵗ, constants))
    fill_halo_regions!(Tᵣ)

    return ReferenceState(p₀, θ₀_surface, pˢᵗ, pᵣ, ρᵣ, Tᵣ, qᵛᵣ, qˡᵣ, qⁱᵣ)
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
- `surface_potential_temperature`: Reference potential temperature at z=0 (K)
- `standard_pressure`: pˢᵗ for potential temperature definition (Pa)
- `pressure`: Reference pressure field p₀ = pˢᵗ π₀^(cᵖ/R) (derived from π₀)
- `density`: Reference density field ρ₀ = p₀/(R T₀) (derived from π₀ and θ₀)
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

@kernel function _compute_exner_reference!(π₀, pᵣ, ρᵣ, θ₀, grid, Nz, π₀_surface, pˢᵗ, cᵖᵈ, κ, Rᵈ, g)
    _ = @index(Global)

    # Initialize π₀ at the bottom from the surface pressure
    @inbounds π₀[1, 1, 1] = π₀_surface

    # Integrate upward: π₀[k] = π₀[k-1] - g Δz / (cᵖ avg(θ₀))
    for k in 2:Nz
        Δz_face = Δzᶜᶜᶠ(1, 1, k, grid)
        @inbounds θ₀_face = (θ₀[1, 1, k] + θ₀[1, 1, k-1]) / 2
        @inbounds π₀[1, 1, k] = π₀[1, 1, k-1] - g * Δz_face / (cᵖᵈ * θ₀_face)
    end

    # Derive pressure and density from π₀
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
    arch = architecture(grid)
    p₀ = convert(FT, surface_pressure)
    pˢᵗ = convert(FT, standard_pressure)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    g = constants.gravitational_acceleration
    Nz = size(grid, 3)
    loc = (nothing, nothing, Center())

    # Build θ₀ field (temporary, used only during construction)
    θ₀ = Field{Nothing, Nothing, Center}(grid)
    set!(θ₀, potential_temperature)
    fill_halo_regions!(θ₀)

    # Surface values for boundary conditions and display
    π₀_surface = (p₀ / pˢᵗ)^κ
    θ₀_surface = @allowscalar θ₀[1, 1, 1]
    ρ_surface = p₀ / (Rᵈ * θ₀_surface * π₀_surface)

    # Allocate output fields
    π₀ = Field{Nothing, Nothing, Center}(grid)
    p_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(p₀))
    pᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=p_bcs)
    ρ_bcs = FieldBoundaryConditions(grid, loc, bottom=ValueBoundaryCondition(ρ_surface))
    ρᵣ = Field{Nothing, Nothing, Center}(grid, boundary_conditions=ρ_bcs)

    # Build π₀ by discrete upward integration, then derive p₀ and ρ₀
    launch!(arch, grid, tuple(1), _compute_exner_reference!,
            π₀, pᵣ, ρᵣ, θ₀, grid, Nz, π₀_surface, pˢᵗ, cᵖᵈ, κ, Rᵈ, g)

    fill_halo_regions!(π₀)
    fill_halo_regions!(pᵣ)
    fill_halo_regions!(ρᵣ)

    return ExnerReferenceState(p₀, θ₀_surface, pˢᵗ, pᵣ, ρᵣ, π₀)
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
