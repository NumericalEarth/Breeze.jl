module MoistAirBuoyancies

export
    MoistAirBuoyancy,
    TemperatureField,
    CondensateField,
    SaturationField

using ..Thermodynamics:
    PotentialTemperatureState,
    MoistureMassFractions,
    total_specific_humidity,
    dry_air_gas_constant,
    vapor_gas_constant,
    with_moisture,
    saturation_vapor_pressure,
    density,
    exner_function

using Oceananigans: Oceananigans, Center, Field, KernelFunctionOperation
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂zᶜᶜᶠ

using Adapt: Adapt, adapt

import Oceananigans.BuoyancyFormulations: AbstractBuoyancyFormulation, buoyancy_perturbationᶜᶜᶜ,
                                          ∂z_b, required_tracers

using ..Thermodynamics:
    ThermodynamicConstants,
    ReferenceState,
    mixture_heat_capacity,
    mixture_gas_constant

import ..Thermodynamics:
    saturation_specific_humidity,
    condensate_specific_humidity

struct MoistAirBuoyancy{RS, AT} <: AbstractBuoyancyFormulation{Nothing}
    reference_state :: RS
    thermodynamics :: AT
end

"""
    MoistAirBuoyancy(grid;
                     base_pressure = 101325,
                     reference_potential_temperature = 288,
                     thermodynamics = ThermodynamicConstants(FT))

Return a MoistAirBuoyancy formulation that can be provided as input to an
`Oceananigans.NonhydrostaticModel`.

!!! note "Required tracers"
    `MoistAirBuoyancy` requires tracers `θ` and `qᵗ`.

Example
=======

```jldoctest mab
using Breeze, Oceananigans

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 3e3))
buoyancy = MoistAirBuoyancy(grid)

# output
MoistAirBuoyancy:
├── reference_state: ReferenceState{Float64}(p₀=101325.0, θᵣ=288.0)
└── thermodynamics: ThermodynamicConstants{Float64}
```

To build a model with MoistAirBuoyancy, we include potential temperature and total specific humidity
tracers `θ` and `qᵗ` to the model.

```jldoctest mab
model = NonhydrostaticModel(; grid, buoyancy, tracers = (:θ, :qᵗ))
                                     
# output
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: (θ, qᵗ)
├── closure: Nothing
├── buoyancy: MoistAirBuoyancy with ĝ = NegativeZDirection()
└── coriolis: Nothing
```
"""
function MoistAirBuoyancy(grid;
                          base_pressure = 101325,
                          reference_potential_temperature = 288,
                          thermodynamics = ThermodynamicConstants(eltype(grid)))

    reference_state = ReferenceState(grid, thermodynamics;
                                     base_pressure,
                                     potential_temperature = reference_potential_temperature)
                          
    return MoistAirBuoyancy(reference_state, thermodynamics)
end

Base.summary(b::MoistAirBuoyancy) = "MoistAirBuoyancy"

function Base.show(io::IO, b::MoistAirBuoyancy)
    print(io, summary(b), ":\n",
        "├── reference_state: ", summary(b.reference_state), "\n",
        "└── thermodynamics: ", summary(b.thermodynamics))
end

required_tracers(::MoistAirBuoyancy) = (:θ, :qᵗ)

const c = Center()


@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, mb::MoistAirBuoyancy, tracers)
    @inbounds begin
        pᵣ = mb.reference_state.pressure[i, j, k]
        ρᵣ = mb.reference_state.density[i, j, k]
        θ = tracers.θ[i, j, k]
        qᵗ = tracers.qᵗ[i, j, k]
    end

    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    p₀ = mb.reference_state.base_pressure
    q = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰 = PotentialTemperatureState(θ, q, z, p₀, pᵣ, ρᵣ)

    # Perform saturation adjustment
    T = temperature(𝒰, mb.thermodynamics)

    # Compute specific volume
    Rᵐ = mixture_gas_constant(q, mb.thermodynamics)
    α = Rᵐ * T / pᵣ

    g = mb.thermodynamics.gravitational_acceleration

    # b = g * (α - αᵣ) / αᵣ
    return g * (ρᵣ * α - 1)
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
r(T) ≡ T - θ Π - ℒ qˡ / cᵖᵐ .
```

Solution of ``r(T) = 0`` is found via the [secant method](https://en.wikipedia.org/wiki/Secant_method).
"""
@inline function temperature(𝒰₀::PotentialTemperatureState{FT}, thermo) where FT
    θ = 𝒰₀.potential_temperature
    θ == 0 && return zero(FT)

    # Generate guess for unsaturated conditions; if dry, return T₁
    qᵗ = total_specific_humidity(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰₁ = with_moisture(𝒰₀, q₁)
    Π₁ = exner_function(𝒰₀, thermo)
    T₁ = Π₁ * θ

    pᵣ = 𝒰₀.reference_pressure
    ρ₁ = density(pᵣ, T₁, q₁, thermo)
    qᵛ⁺₁ = saturation_specific_humidity(T₁, ρ₁, thermo, thermo.liquid)
    qᵗ <= qᵛ⁺₁ && return T₁

    # If we made it this far, the state is saturated.
    # T₁ then provides a lower bound.
    # We generate a second guess using the liquid fraction
    # associated with T₁, which should also represent an underestimate.
    ℒˡ = thermo.liquid.reference_latent_heat
    q₁ = 𝒰₁.moisture_fractions
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    T₂ = T₁ + ℒˡ * q₁.liquid / cᵖᵐ
    𝒰₂ = adjust_state(𝒰₁, T₂, thermo)

    # Initialize saturation adjustment
    r₁ = saturation_adjustment_residual(T₁, 𝒰₁, thermo)
    r₂ = saturation_adjustment_residual(T₂, 𝒰₂, thermo)
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
        𝒰₁ = 𝒰₂

        T₂ -= r₂ * ΔTΔr
        𝒰₂ = adjust_state(𝒰₂, T₂, thermo)
        r₂ = saturation_adjustment_residual(T₂, 𝒰₂, thermo)

        iter += 1
    end

    return T₂
end

# This estimate assumes that the specific humidity is itself the saturation
# specific humidity, which is needed to compute density.
# See Pressel et al 2015, equation 37
function adjustment_saturation_specific_humidity(T, 𝒰, thermo)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, thermo.liquid)
    pᵣ = 𝒰.reference_pressure
    qᵗ = total_specific_humidity(𝒰)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    ϵ = Rᵈ / Rᵛ
    return ϵ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀, T, thermo)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, 𝒰₀, thermo)
    qᵗ = total_specific_humidity(𝒰₀)
    qˡ = max(0, qᵗ - qᵛ⁺)
    q₁ = MoistureMassFractions(qᵛ⁺, qˡ, zero(qˡ))
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T, 𝒰, thermo)
    Π = exner_function(𝒰, thermo)
    q = 𝒰.moisture_fractions
    θ = 𝒰.potential_temperature
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    qˡ = q.liquid
    θ = 𝒰.potential_temperature
    return T - ℒˡᵣ * qˡ / cᵖᵐ - Π * θ
end

#####
##### Diagnostics
#####

const c = Center()

# Temperature
@inline function temperature(i, j, k, grid::AbstractGrid, mb::MoistAirBuoyancy, θ, qᵗ)
    @inbounds begin
        θi = θ[i, j, k]
        qᵗi = qᵗ[i, j, k]
        pᵣ = mb.reference_state.pressure[i, j, k]
        ρᵣ = mb.reference_state.density[i, j, k]
    end
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    p₀ = mb.reference_state.base_pressure
    q = MoistureMassFractions(qᵗi, zero(qᵗi), zero(qᵗi))
    𝒰 = PotentialTemperatureState(θi, q, z, p₀, pᵣ, ρᵣ)
    return temperature(𝒰, mb.thermodynamics)
end

struct TemperatureKernelFunction end

@inline (::TemperatureKernelFunction)(i, j, k, grid, buoyancy, θ, qᵗ) =
    temperature(i, j, k, grid, buoyancy, θ, qᵗ)

function TemperatureField(model)
    func = TemperatureKernelFunction()
    grid = model.grid
    buoyancy = model.buoyancy.formulation
    θ = model.tracers.θ
    qᵗ = model.tracers.qᵗ
    op = KernelFunctionOperation{Center, Center, Center}(func, grid, buoyancy, θ, qᵗ)
    return Field(op)
end

# Saturation specific humidity
@inline function saturation_specific_humidity(i, j, k, grid, mb::MoistAirBuoyancy, T, phase)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    @inbounds begin
        Ti = T[i, j, k]
        ρᵣ = mb.reference_state.density[i, j, k]
    end
    return saturation_specific_humidity(Ti, ρᵣ, mb.thermodynamics, phase)
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

@inline function condensate_specific_humidity(i, j, k, grid, mb::MoistAirBuoyancy, T, qᵗ)
    @inbounds begin
        Ti = T[i, j, k]
        qᵗi = qᵗ[i, j, k]
        pᵣ = mb.reference_state.pressure[i, j, k]
        ρᵣ = mb.reference_state.density[i, j, k]
    end
    q₀ = MoistureMassFractions(qᵗi, zero(qᵗi), zero(qᵗi))
    ρ = density(pᵣ, Ti, q₀, mb.thermodynamics)
    qᵛ⁺ = saturation_specific_humidity(Ti, ρ, mb.thermodynamics, mb.thermodynamics.liquid)
    qˡ = max(0, qᵗi - qᵛ⁺)
    return qˡ
end

@inline function (kernel::CondensateKernel)(i, j, k, grid, buoyancy, qᵗ)
    T = kernel.temperature
    return condensate_specific_humidity(i, j, k, grid, buoyancy, T, qᵗ)
end

function CondensateField(model, T=TemperatureField(model))
    func = CondensateKernel(T)
    grid = model.grid
    buoyancy = model.buoyancy.formulation
    qᵗ = model.tracers.qᵗ
    op = KernelFunctionOperation{Center, Center, Center}(func, grid, buoyancy, qᵗ)
    return Field(op)
end

end # module