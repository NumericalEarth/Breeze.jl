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
    mixture_heat_capacity,
    mixture_gas_constant,
    reference_specific_volume,
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
                     thermodynamics = AtmosphereThermodynamics(FT),
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
reference_density(z, mb::MoistAirBuoyancy) = reference_density(z, mb.reference_constants, mb.thermodynamics)
base_density(mb::MoistAirBuoyancy) = base_density(mb.reference_constants, mb.thermodynamics)

#####
#####
#####

const c = Center()

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, mb::MoistAirBuoyancy, tracers)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    θ = @inbounds tracers.θ[i, j, k]
    q = @inbounds tracers.q[i, j, k]
    𝒰 = HeightReferenceThermodynamicState(θ, q, z)

    # Perform Saturation adjustment
    α = specific_volume(𝒰, mb.reference_constants, mb.thermodynamics)

    # Compute reference specific volume
    αʳ = reference_specific_volume(z, mb.reference_constants, mb.thermodynamics)
    g = mb.thermodynamics.gravitational_acceleration

    # Formulation in terms of base density:
    # ρ₀ = base_density(mb.reference_constants, mb.thermodynamics)
    # return ρ₀ * g * (α - αʳ)

    return g * (α - αʳ) / αʳ
end

@inline ∂z_b(i, j, k, grid, mb::MoistAirBuoyancy, tracers) =
    ∂zᶜᶜᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, mb, tracers)

const c = Center()

#####
##### Temperature
#####

function temperature(i, j, k, grid::AbstractGrid, mb::MoistAirBuoyancy, θ, q)
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    θi = @inbounds θ[i, j, k]
    qi = @inbounds q[i, j, k]
    𝒰 = HeightReferenceThermodynamicState(θi, qi, z)
    return temperature(𝒰, mb.reference_constants, mb.thermodynamics)
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

#####
##### Saturation adjustment
#####

# Organizing information about the state is a WIP
struct HeightReferenceThermodynamicState{FT}
    θ :: FT
    q :: FT
    z :: FT
end

Base.summary(at::HeightReferenceThermodynamicState{FT}) where FT = "HeightReferenceThermodynamicState{$FT}"

function Base.show(io::IO, hrts::HeightReferenceThermodynamicState)
    print(io, summary(hrts), ":", '\n',
        "├── θ: ", hrts.θ, "\n",
        "├── q: ", hrts.q, "\n",
        "└── z: ", hrts.z)
end

condensate_specific_humidity(T, state::HeightReferenceThermodynamicState, ref, thermo) =
    condensate_specific_humidity(T, state.q, state.z, ref, thermo)

# Solve
# θ = T/Π ( 1 - ℒ qˡ / (cᵖᵐ T))
# for temperature T with qˡ = max(0, q - qᵛ★).
# root of: f(T) = T - Π θ - ℒ qˡ / cᵖᵐ
@inline function temperature(state::HeightReferenceThermodynamicState{FT}, ref, thermo) where FT
    state.θ == 0 && return zero(FT)

    # Generate guess for unsaturated conditions
    Π = exner_function(state, ref, thermo)
    T₁ = Π * state.θ
    qˡ₁ = condensate_specific_humidity(T₁, state, ref, thermo)
    qˡ₁ <= 0 && return T₁

    # If we made it this far, we have condensation
    r₁ = saturation_adjustment_residual(T₁, Π, qˡ₁, state, thermo)

    ℒᵛ = thermo.liquid.latent_heat
    cᵖᵐ = mixture_heat_capacity(state.q, thermo)
    T₂ = T₁ + ℒᵛ * qˡ₁ / cᵖᵐ
    qˡ₂ = condensate_specific_humidity(T₂, state, ref, thermo)
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
        qˡ₂ = condensate_specific_humidity(T₂, state, ref, thermo)
        r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)
        iter += 1
    end

    return T₂
end

@inline function saturation_adjustment_residual(T, Π, qˡ, state::HeightReferenceThermodynamicState, thermo)
    ℒᵛ = thermo.liquid.latent_heat
    cᵖᵐ = mixture_heat_capacity(state.q, thermo)
    return T - ℒᵛ * qˡ / cᵖᵐ - Π * state.θ
end

@inline function specific_volume(state::HeightReferenceThermodynamicState, ref, thermo)
    T = temperature(state, ref, thermo)
    Rᵐ = mixture_gas_constant(state.q, thermo)
    pᵣ = reference_pressure(state.z, ref, thermo)
    return Rᵐ * T / pᵣ
end

@inline function exner_function(state::HeightReferenceThermodynamicState, ref, thermo)
    Rᵐ = mixture_gas_constant(state.q, thermo)
    cᵖᵐ = mixture_heat_capacity(state.q, thermo)
    inv_ϰᵐ = Rᵐ / cᵖᵐ
    pᵣ = reference_pressure(state.z, ref, thermo)
    p₀ = ref.base_pressure
    return (pᵣ / p₀)^inv_ϰᵐ
end

#####
##### Reference implementation of an "unsaturated" moist air buoyancy model,
##### which assumes unsaturated air
#####

struct UnsaturatedMoistAirBuoyancy{FT} <: AbstractBuoyancyFormulation{Nothing}
    expansion_coefficient :: FT
    reference_potential_temperature :: FT
    gas_constant_ratio :: FT
end

function UnsaturatedMoistAirBuoyancy(FT=Oceananigans.defaults.FloatType;
                                     expansion_coefficient = 3.27e-2,
                                     reference_potential_temperature = 0,
                                     gas_constant_ratio = 1.61)

    return UnsaturatedMoistAirBuoyancy{FT}(expansion_coefficient,
                                           reference_potential_temperature,
                                           gas_constant_ratio)
end

required_tracers(::UnsaturatedMoistAirBuoyancy) = (:θ, :q)

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, mb::UnsaturatedMoistAirBuoyancy, tracers)
    β = mb.expansion_coefficient
    θ₀ = mb.reference_potential_temperature
    ϵᵥ = mb.gas_constant_ratio
    δ = ϵᵥ - 1
    θ = @inbounds tracers.θ[i, j, k]
    q = @inbounds tracers.q[i, j, k]
    θᵥ = θ * (1 + δ * q)
    return β * (θᵥ - θ₀)
end

end # module
