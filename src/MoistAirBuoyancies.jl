module MoistAirBuoyancies

export MoistAirBuoyancy
export UnsaturatedMoistAirBuoyancy
export TemperatureField
export CondensateField
export SaturationField

using Oceananigans
using Oceananigans: AbstractModel
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂zᶜᶜᶠ

using Adapt

import Oceananigans.BuoyancyFormulations: AbstractBuoyancyFormulation,
                                          buoyancy_perturbationᶜᶜᶜ,
                                          ∂z_b,
                                          required_tracers

using ..Thermodynamics:
    ThermodynamicConstants,
    ReferenceState,
    SpecificHumidities,
    reference_specific_volume,
    PotentialTemperatureState

using ..Microphysics:
    SaturationAdjustmentMicrophysics,
    adjust_temperature_and_humidities

import ..Thermodynamics:
    base_density,
    saturation_specific_humidity

import ..Microphysics:
    temperature,
    specific_volume,
    adjusted_condensate_specific_humidity

struct MoistAirBuoyancy{FT, AT, M} <: AbstractBuoyancyFormulation{Nothing}
    reference_state :: ReferenceState{FT}
    thermodynamics :: AT
    microphysics :: M
end

"""
    MoistAirBuoyancy(FT=Oceananigans.defaults.FloatType;
                     thermodynamics = ThermodynamicConstants(FT),
                     reference_state = ReferenceState{FT}(101325, 290))
                     microphysics = nothing)

Return a MoistAirBuoyancy formulation that can be provided as input to an
[`AtmosphereModel`](@ref Breeze.AtmosphereModels.AtmosphereModel) or an
`Oceananigans.NonhydrostaticModel`.

!!! note "Required tracers"
    `MoistAirBuoyancy` requires tracers `q` and `θ` to be included in the model.

Example
=======

```jldoctest
julia> using Breeze, Oceananigans

julia> buoyancy = MoistAirBuoyancy()
MoistAirBuoyancy
├── reference_state: Breeze.Thermodynamics.ReferenceState{Float64}
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
                          reference_state = ReferenceState{FT}(101325, 290),
                          microphysics = SaturationAdjustmentMicrophysics())

    AT = typeof(thermodynamics)
    MT = typeof(microphysics)
    return MoistAirBuoyancy{FT, AT, MT}(reference_state, thermodynamics, microphysics)
end

Base.summary(b::MoistAirBuoyancy) = "MoistAirBuoyancy"

function Base.show(io::IO, b::MoistAirBuoyancy)
    print(io, summary(b), "\n",
        "├── reference_state: ", summary(b.reference_state), "\n",
        "├── thermodynamics: ", summary(b.thermodynamics), "\n",
        "└── microphysics: ", summary(b.microphysics))
end

required_tracers(::MoistAirBuoyancy) = (:θ, :q)
reference_density(z, mb::MoistAirBuoyancy) = reference_density(z, mb.reference_state, mb.thermodynamics)
base_density(mb::MoistAirBuoyancy) = base_density(mb.reference_state, mb.thermodynamics)

#####
##### buoyancy
#####

const c = Center()

# Nothing microphysics: no condensates
function compute_temperature_and_humidities(::Nothing, θ, qᵗ, z, mb)
    q = SpecificHumidities(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰 = PotentialTemperatureState(θ, q, z, mb.reference_state)
    T = temperature(𝒰, mb.thermodynamics)
    return T, q
end

function compute_temperature_and_humidities(::SaturationAdjustmentMicrophysics, θ, qᵗ, z, mb)
    q = SpecificHumidities(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰 = PotentialTemperatureState(θ, q, z, mb.reference_state)
    T, q = adjust_temperature_and_humidities(𝒰, mb.thermodynamics)
    return T, q
end

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, mb::MoistAirBuoyancy, tracers)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    θ = @inbounds tracers.θ[i, j, k]
    qᵗ = @inbounds tracers.q[i, j, k]

    T, q = compute_temperature_and_humidities(mb.microphysics, θ, qᵗ, z, mb)
    α = specific_volume(T, q, z, mb.reference_state, mb.thermodynamics)

    # Compute buoyancy
    αʳ = reference_specific_volume(z, mb.reference_state, mb.thermodynamics)
    g = mb.thermodynamics.gravitational_acceleration

    return g * (α - αʳ) / αʳ
end

@inline ∂z_b(i, j, k, grid, mb::MoistAirBuoyancy, tracers) =
    ∂zᶜᶜᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, mb, tracers)

#####
##### Temperature
#####

function temperature(i, j, k, grid::AbstractGrid, mb::MoistAirBuoyancy, θ, q)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    θi = @inbounds θ[i, j, k]
    qi = @inbounds q[i, j, k]
    q = SpecificHumidities(qi, zero(qi), zero(qi))
    𝒰 = PotentialTemperatureState(θi, q, z, mb.reference_state)    
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

#####
##### Saturation specific humidity
#####

@inline function saturation_specific_humidity(i, j, k, grid, mb::MoistAirBuoyancy, T, condensed_phase)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    Ti = @inbounds T[i, j, k]
    return saturation_specific_humidity(Ti, z, mb.reference_state, mb.thermodynamics, condensed_phase)
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

#####
##### Condensate
#####

struct CondensateKernel{T}
    temperature :: T
end

Adapt.adapt_structure(to, ck::CondensateKernel) = CondensateKernel(adapt(to, ck.temperature))

@inline function condensate_specific_humidity(i, j, k, grid, mb::MoistAirBuoyancy, T, q)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    Ti = @inbounds T[i, j, k]
    qᵗ = @inbounds q[i, j, k]
    qˡ = adjusted_condensate_specific_humidity(Ti, qᵗ, z, mb.reference_state, mb.thermodynamics)
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
