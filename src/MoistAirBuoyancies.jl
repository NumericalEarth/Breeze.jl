module MoistAirBuoyancies

export
    MoistAirBuoyancy,
    TemperatureField,
    CondensateField,
    SaturationField

using ..Thermodynamics: PotentialTemperatureState, MoistureMassFractions, exner_function

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
r(T) ≡ T - θ Π - ℒ qˡ / (cᵖᵐ T) .
```

Solution of ``r(T) = 0`` is found via the [secant method](https://en.wikipedia.org/wiki/Secant_method).
"""
@inline function temperature(state::PotentialTemperatureState{FT}, thermo) where FT
    θ = state.potential_temperature
    θ == 0 && return zero(FT)

    # Generate guess for unsaturated conditions
    Π = exner_function(state, thermo)
    T₁ = Π * θ
    qˡ₁ = condensate_specific_humidity(T₁, state, thermo)
    qˡ₁ <= 0 && return T₁

    # If we made it this far, we have condensation
    r₁ = saturation_adjustment_residual(T₁, Π, qˡ₁, state, thermo)

    ℒˡ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(state.moisture_fractions, thermo)
    T₂ = T₁ + ℒˡ * qˡ₁ / cᵖᵐ
    qˡ₂ = condensate_specific_humidity(T₂, state, thermo)
    r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)

    # Saturation adjustment
    R = sqrt(max(T₂, T₁))
    ϵ = convert(FT, 1e-6)
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
    ℒᵛ₀ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(state.moisture_fractions, thermo)
    θ = state.potential_temperature
    return T - ℒᵛ₀ * qˡ / cᵖᵐ - Π * θ
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
    q = MoistureMassFractions(qᵗi, zero(qᵗi), zero(qᵗi))
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    p₀ = mb.reference_state.base_pressure
    𝒰 = PotentialTemperatureState(Ti, q, z, p₀, pᵣ, ρᵣ)
    qˡ = condensate_specific_humidity(Ti, 𝒰, mb.thermodynamics)
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