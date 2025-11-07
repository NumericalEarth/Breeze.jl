module MoistAirBuoyancies

using ..Thermodynamics: PotentialTemperatureState, MassRatios, exner_function, reference_density

export MoistAirBuoyancy
export UnsaturatedMoistAirBuoyancy
export TemperatureField
export CondensateField
export SaturationField

using Oceananigans: Oceananigans, Center, Field, KernelFunctionOperation
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂zᶜᶜᶠ

using Adapt: Adapt, adapt

import Oceananigans.BuoyancyFormulations: AbstractBuoyancyFormulation, buoyancy_perturbationᶜᶜᶜ,
                                          ∂z_b, required_tracers

using ..Thermodynamics:
    ThermodynamicConstants,
    ReferenceStateConstants,
    reference_specific_volume,
    mixture_heat_capacity,
    mixture_gas_constant,
    reference_pressure

import ..Thermodynamics:
    base_density,
    saturation_specific_humidity,
    condensate_specific_humidity

struct MoistAirBuoyancy{FT, AT} <: AbstractBuoyancyFormulation{Nothing}
    reference_constants :: ReferenceStateConstants{FT}
    thermodynamics :: AT
end

"""
    MoistAirBuoyancy(FT=Oceananigans.defaults.FloatType;
                     thermodynamics = ThermodynamicConstants(FT),
                     reference_constants = ReferenceStateConstants{FT}(101325, 290))

Return a MoistAirBuoyancy formulation that can be provided as input to an `AtmosphereModel`
or an `Oceananigans.NonhydrostaticModel`.

!!! note "Required tracers"
    `MoistAirBuoyancy` requires tracers `q` and `θ` to be included in the model.

Example
=======

```jldoctest
julia> using Breeze, Oceananigans

julia> buoyancy = MoistAirBuoyancy()
MoistAirBuoyancy
├── reference_constants: Breeze.Thermodynamics.ReferenceStateConstants{Float64}
└── thermodynamics: ThermodynamicConstants

julia> model = NonhydrostaticModel(; grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 2, 3)),
                                     buoyancy, tracers = (:θ, :q))
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: (θ, q)
├── closure: Nothing
├── buoyancy: MoistAirBuoyancy with ĝ = NegativeZDirection()
└── coriolis: Nothing
```
"""
function MoistAirBuoyancy(FT=Oceananigans.defaults.FloatType;
                          thermodynamics = ThermodynamicConstants(FT),
                          reference_constants = ReferenceStateConstants{FT}(101325, 290))

    AT = typeof(thermodynamics)
    return MoistAirBuoyancy{FT, AT}(reference_constants, thermodynamics)
end

Base.summary(b::MoistAirBuoyancy) = "MoistAirBuoyancy"

function Base.show(io::IO, b::MoistAirBuoyancy)
    print(io, summary(b), "\n",
        "├── reference_constants: ", summary(b.reference_constants), "\n",
        "└── thermodynamics: ", summary(b.thermodynamics))
end

required_tracers(::MoistAirBuoyancy) = (:θ, :q)

#####
#####
#####

const c = Center()

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, mb::MoistAirBuoyancy, tracers)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    θ = @inbounds tracers.θ[i, j, k]
    qᵗ = @inbounds tracers.q[i, j, k]
    q = MassRatios(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰 = PotentialTemperatureState(θ, q, z, mb.reference_constants)

    # Perform saturation adjustment
    T = temperature(𝒰, mb.thermodynamics)

    # Compute specific volume
    pᵣ = reference_pressure(z, mb.reference_constants, mb.thermodynamics)
    Rᵐ = mixture_gas_constant(q, mb.thermodynamics)
    α = Rᵐ * T / pᵣ

    # Compute reference specific volume
    αᵣ = reference_specific_volume(z, mb.reference_constants, mb.thermodynamics)
    g = mb.thermodynamics.gravitational_acceleration

    # Formulation in terms of base density:
    # ρ₀ = base_density(mb.reference_constants, mb.thermodynamics)
    # return ρ₀ * g * (α - αᵣ)

    return g * (α - αᵣ) / αᵣ
end

@inline ∂z_b(i, j, k, grid, mb::MoistAirBuoyancy, tracers) =
    ∂zᶜᶜᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, mb, tracers)

#####
##### Saturation adjustment
#####

# Solve
# θ = T/Π ( 1 - ℒ qˡ / (cᵖᵐ T))
# for temperature T with qˡ = max(0, q - qᵛ⁺).
# root of: f(T) = T - Π θ - ℒ qˡ / cᵖᵐ

"""
    temperature(state::PotentialTemperatureState, ref, thermo)

Return the temperature ``T`` that satisfies saturation adjustment, that is, the
temperature for which

```math
θ = [1 - ℒ qˡ / (cᵖᵐ T)] T / Π ,
```

with ``qˡ = \\max(0, qᵗ - qᵛ⁺)`` the condensate specific humidity, where ``qᵗ`` is the
total specific humidity, ``qᵛ⁺`` is the saturation specific humidity.

The saturation adjustment temperature is obtained by solving ``r(T)``, where
```math
r(T) ≡ T - θ Π - ℒ qˡ / (cᵖᵐ T) .
```

Solution of ``r(T) = 0`` is found via the [secant method](https://en.wikipedia.org/wiki/Secant_method).
"""
@inline function temperature(state::PotentialTemperatureState{FT}, thermo) where FT
    θ = state.potential_temperature
    θ == 0 && return zero(FT)

    # Generate guess for unsaturated conditions
    Π = exner_function(state, thermo)
    T₁ = Π * state.potential_temperature
    qˡ₁ = condensate_specific_humidity(T₁, state, thermo)
    qˡ₁ <= 0 && return T₁

    # If we made it this far, we have condensation
    r₁ = saturation_adjustment_residual(T₁, Π, qˡ₁, state, thermo)

    ℒᵛ = thermo.liquid.latent_heat
    cᵖᵐ = mixture_heat_capacity(state.mass_ratios, thermo)
    T₂ = T₁ + ℒᵛ * qˡ₁ / cᵖᵐ
    qˡ₂ = condensate_specific_humidity(T₂, state, thermo)
    r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)

    # Saturation adjustment
    R = sqrt(max(T₂, T₁))
    ϵ = convert(FT, 1e-9)
    δ = ϵ * R
    iter = 0

    while abs(r₂ - r₁) > δ
        # Compute slope
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        # Store previous values
        r₁ = r₂
        T₁ = T₂

        # Update
        T₂ -= r₂ * ΔTΔr
        qˡ₂ = condensate_specific_humidity(T₂, state, thermo)
        r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)
        iter += 1
    end

    return T₂
end

@inline function saturation_adjustment_residual(T, Π, qˡ, state::PotentialTemperatureState, thermo)
    ℒᵛ₀ = thermo.liquid.latent_heat
    cᵖᵐ = mixture_heat_capacity(state.mass_ratios, thermo)
    θ = state.potential_temperature
    return T^2 - ℒᵛ₀ * qˡ / cᵖᵐ - Π * θ * T
end

#####
##### Diagnostics
#####

const c = Center()

# Temperature
@inline function temperature(i, j, k, grid::AbstractGrid, mb::MoistAirBuoyancy, θ, q)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    θi = @inbounds θ[i, j, k]
    qᵗ = @inbounds q[i, j, k]
    q = MassRatios(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰 = PotentialTemperatureState(θi, q, z, mb.reference_constants)
    return temperature(𝒰, mb.thermodynamics)
end

struct TemperatureKernelFunction end

@inline (::TemperatureKernelFunction)(i, j, k, grid, buoyancy, θ, q) =
    temperature(i, j, k, grid, buoyancy, θ, q)

function TemperatureField(model)
    func = TemperatureKernelFunction()
    grid = model.grid
    buoyancy = model.buoyancy.formulation
    θ = model.tracers.θ
    q = model.tracers.q
    op = KernelFunctionOperation{Center, Center, Center}(func, grid, buoyancy, θ, q)
    return Field(op)
end

# Saturation specific humidity
@inline function saturation_specific_humidity(i, j, k, grid, mb::MoistAirBuoyancy, T, phase)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    Ti = @inbounds T[i, j, k]
    ρ = reference_density(z, mb.reference_constants, mb.thermodynamics)
    return saturation_specific_humidity(Ti, ρ, mb.thermodynamics, phase)
end

struct PhaseTransitionConstantsKernel{T, P}
    condensed_phase :: P
    temperature :: T
end

Adapt.adapt_structure(to, sk::PhaseTransitionConstantsKernel) =
    PhaseTransitionConstantsKernel(adapt(to, sk.condensed_phase),
                     adapt(to, sk.temperature))

@inline function (kernel::PhaseTransitionConstantsKernel)(i, j, k, grid, buoyancy)
    T = kernel.temperature
    return saturation_specific_humidity(i, j, k, grid, buoyancy, T, kernel.condensed_phase)
end

function SaturationField(model,
                         T = TemperatureField(model);
                         condensed_phase = model.buoyancy.formulation.thermodynamics.liquid)
    func = PhaseTransitionConstantsKernel(condensed_phase, T)
    grid = model.grid
    buoyancy = model.buoyancy.formulation
    op = KernelFunctionOperation{Center, Center, Center}(func, grid, buoyancy)
    return Field(op)
end

# Condensate
struct CondensateKernel{T}
    temperature :: T
end

Adapt.adapt_structure(to, ck::CondensateKernel) = CondensateKernel(adapt(to, ck.temperature))

@inline function condensate_specific_humidity(i, j, k, grid, mb::MoistAirBuoyancy, T, q)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    Ti = @inbounds T[i, j, k]
    qᵗ = @inbounds q[i, j, k]
    q = MassRatios(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰 = PotentialTemperatureState(Ti, q, z, mb.reference_constants)
    qˡ = condensate_specific_humidity(Ti, 𝒰, mb.thermodynamics)
    return qˡ
end

@inline function (kernel::CondensateKernel)(i, j, k, grid, buoyancy, q)
    T = kernel.temperature
    return condensate_specific_humidity(i, j, k, grid, buoyancy, T, q)
end

function CondensateField(model, T=TemperatureField(model))
    func = CondensateKernel(T)
    grid = model.grid
    buoyancy = model.buoyancy.formulation
    q = model.tracers.q
    op = KernelFunctionOperation{Center, Center, Center}(func, grid, buoyancy, q)
    return Field(op)
end

end # module