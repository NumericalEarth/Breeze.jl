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
    AtmosphereThermodynamics,
    ReferenceStateConstants,
    BoussinesqThermodynamicState,
    reference_specific_volume

import ..Thermodynamics:
    base_density,
    saturation_specific_humidity,
    condensate_specific_humidity

import ..Microphysics:
    HeightReferenceThermodynamicState,
    temperature,
    specific_volume

struct MoistAirBuoyancy{FT, AT, M} <: AbstractBuoyancyFormulation{Nothing}
    reference_constants :: ReferenceStateConstants{FT}
    thermodynamics :: AT
    microphysics :: M
end

"""
    MoistAirBuoyancy(FT=Oceananigans.defaults.FloatType;
                     thermodynamics = AtmosphereThermodynamics(FT),
                     reference_constants = ReferenceStateConstants{FT}(101325, 290))
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
├── reference_constants: Breeze.Thermodynamics.ReferenceStateConstants{Float64}
└── thermodynamics: AtmosphereThermodynamics

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
                          thermodynamics = AtmosphereThermodynamics(FT),
                          reference_constants = ReferenceStateConstants{FT}(101325, 290),
                          microphysics = nothing)

    AT = typeof(thermodynamics)
    MT = typeof(microphysics)
    return MoistAirBuoyancy{FT, AT, MT}(reference_constants, thermodynamics, microphysics)
end

Base.summary(b::MoistAirBuoyancy) = "MoistAirBuoyancy"

function Base.show(io::IO, b::MoistAirBuoyancy)
    print(io, summary(b), "\n",
        "├── reference_constants: ", summary(b.reference_constants), "\n",
        "└── thermodynamics: ", summary(b.thermodynamics))
end

required_tracers(::MoistAirBuoyancy) = (:θ, :q)
reference_density(z, mb::MoistAirBuoyancy) = reference_density(z, mb.reference_constants, mb.thermodynamics)
base_density(mb::MoistAirBuoyancy) = base_density(mb.reference_constants, mb.thermodynamics)

#####
##### buoyancy
#####

const c = Center()

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, mb::MoistAirBuoyancy, tracers)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    θ = @inbounds tracers.θ[i, j, k]
    q = @inbounds tracers.q[i, j, k]
    𝒰 = BoussinesqThermodynamicState(θ, q, z)

    # Compute temperature:
    α = specific_volume(𝒰, mb.microphysics, mb.reference_constants, mb.thermodynamics)

    # Compute buoyancy
    αʳ = reference_specific_volume(z, mb.reference_constants, mb.thermodynamics)
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
    𝒰 = BoussinesqThermodynamicState(θi, qi, z)
    return temperature(𝒰, mb.microphysics, mb.reference_constants, mb.thermodynamics)
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
    return saturation_specific_humidity(Ti, z, mb.reference_constants, mb.thermodynamics, condensed_phase)
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
    qi = @inbounds q[i, j, k]
    qˡ = condensate_specific_humidity(Ti, qi, z, mb.reference_constants, mb.thermodynamics)
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
