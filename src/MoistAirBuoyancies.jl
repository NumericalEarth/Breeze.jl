module MoistAirBuoyancies

export
    MoistAirBuoyancy,
    TemperatureField,
    CondensateField,
    SaturationField

using ..Thermodynamics:
    PotentialTemperatureState,
    MoistureMassFractions,
    total_moisture_mass_fraction,
    dry_air_gas_constant,
    vapor_gas_constant,
    with_moisture,
    saturation_vapor_pressure,
    density,
    exner_function

using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Oceananigans, Center, Field, KernelFunctionOperation
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂zᶜᶜᶠ

using Adapt: Adapt, adapt

import Oceananigans.BuoyancyFormulations:
    AbstractBuoyancyFormulation,
    buoyancy_perturbationᶜᶜᶜ,
    ∂z_b,
    required_tracers

import ..Thermodynamics: saturation_specific_humidity

using ..Thermodynamics:
    ThermodynamicConstants,
    ReferenceState,
    mixture_heat_capacity,
    mixture_gas_constant

struct MoistAirBuoyancy{RS, AT} <: AbstractBuoyancyFormulation{Nothing}
    reference_state :: RS
    thermodynamics :: AT
end

Adapt.adapt_structure(to, mb::MoistAirBuoyancy) =
    MoistAirBuoyancy(adapt(to, mb.reference_state),
                     adapt(to, mb.thermodynamics))

"""
$(TYPEDSIGNATURES)

Return a `MoistAirBuoyancy` formulation that can be provided as input to an
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
├── reference_state: ReferenceState{Float64}(p₀=101325.0, θ₀=288.0)
└── thermodynamics: ThermodynamicConstants{Float64}
```

To build a model with `MoistAirBuoyancy`, we include potential temperature and total specific humidity
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
    q = MoistureMassFractions(qᵗ)
    𝒰 = PotentialTemperatureState(θ, q, z, p₀, pᵣ, ρᵣ)

    # Perform saturation adjustment
    T = compute_boussinesq_adjustment_temperature(𝒰, mb.thermodynamics)

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
$(TYPEDSIGNATURES)

Return the temperature ``T`` corresponding to thermodynamic equilibrium between the
specific humidity and liquid mass fractions of the input thermodynamic state `𝒰₀`,
wherein the specific humidity is equal to or less than the saturation specific humidity
at the given conditions and affiliated with theromdynamic constants `thermo`.

The saturation equilibrium temperature satisfies the nonlinear relation

```math
θ = [1 - ℒˡᵣ qˡ / (cᵖᵐ T)] T / Π ,
```

with ``ℒˡᵣ`` the latent heat at the reference temperature ``Tᵣ``, ``cᵖᵐ`` the mixture
specific heat, ``Π`` the Exner function, ``qˡ = \\max(0, qᵗ - qᵛ⁺)``
the condensate specific humidity, ``qᵗ`` is the
total specific humidity, and ``qᵛ⁺`` is the saturation specific humidity.

The saturation equilibrium temperature is thus obtained by solving ``r(T) = 0``, where
```math
r(T) ≡ T - θ Π - ℒˡᵣ qˡ / cᵖᵐ .
```

Solution of ``r(T) = 0`` is found via the [secant method](https://en.wikipedia.org/wiki/Secant_method).
"""
@inline function compute_boussinesq_adjustment_temperature(𝒰₀::PotentialTemperatureState{FT}, thermo) where FT
    θ = 𝒰₀.potential_temperature
    θ == 0 && return zero(FT)

    # Generate guess for unsaturated conditions; if dry, return T₁
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ)
    𝒰₁ = with_moisture(𝒰₀, q₁)
    Π₁ = exner_function(𝒰₀, thermo)
    T₁ = Π₁ * θ

    pᵣ = 𝒰₀.reference_pressure
    ρ₁ = density(pᵣ, T₁, q₁, thermo)
    qᵛ⁺₁ = saturation_specific_humidity(T₁, ρ₁, thermo, thermo.liquid)
    qᵗ <= qᵛ⁺₁ && return T₁

    # If we made it this far, the state is saturated.
    # T₁ then provides a lower bound, and our state 𝒰₁
    # has to be modified to consistently include the liquid mass fraction.
    # Subsequent computations will assume that the specific humidity
    # is given by the saturation specific humidity, eg ``qᵛ = qᵛ⁺``.
    qᵛ⁺₁ = adjustment_saturation_specific_humidity(T₁, 𝒰₁, thermo)
    qˡ₁ = qᵗ - qᵛ⁺₁
    q₁ = MoistureMassFractions(qᵛ⁺₁, qˡ₁)
    𝒰₁ = with_moisture(𝒰₀, q₁)

    # We generate a second guess simply by adding 1 K to T₁...

    # NOTE: We could also generate a second guess to start a secant iteration
    # by applying the potential temperature assuming a liquid fraction
    # associated with T₁. This should represent an _overestimate_,
    # since ``qᵛ⁺₁(T₁)`` underestimates the saturation specific humidity,
    # and therefore qˡ₁ is overestimated. This is similar to an approach
    # used in Pressel et al 2015. However, it doesn't work for large liquid fractions.
    T₂ = T₁ + 1

    #=
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    T₂ = T₁ + ℒˡᵣ * qˡ₁ / cᵖᵐ
    =#

    𝒰₂ = adjust_state(𝒰₁, T₂, thermo)

    # Initialize saturation adjustment
    r₁ = saturation_adjustment_residual(T₁, 𝒰₁, thermo)
    r₂ = saturation_adjustment_residual(T₂, 𝒰₂, thermo)
    δ = convert(FT, 1e-3)
    iter = 0

    while abs(T₂ - T₁) > δ
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
# specific humidity, eg ``qᵛ = qᵛ⁺``. Knowledge of the specific humidity
# is needed to compute the mixture gas constant, and thus density,
# which in turn is needed to compute the _saturation_ specific humidity.
# This consideration culminates in a new expression for saturation specific humidity
# used below, and also written in Pressel et al 2015, equation 37.
# (There is an error in the description below it, but the equation 37 is correct.)
@inline function adjustment_saturation_specific_humidity(T, 𝒰, thermo)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, thermo.liquid)
    pᵣ = 𝒰.reference_pressure
    qᵗ = total_moisture_mass_fraction(𝒰)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀, T, thermo)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, 𝒰₀, thermo)
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    qˡ = max(0, qᵗ - qᵛ⁺)
    q₁ = MoistureMassFractions(qᵛ⁺, qˡ)
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T, 𝒰, thermo)
    Π = exner_function(𝒰, thermo)
    q = 𝒰.moisture_mass_fractions
    θ = 𝒰.potential_temperature
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    qˡ = q.liquid
    θ = 𝒰.potential_temperature
    return T - Π * θ - ℒˡᵣ * qˡ / cᵖᵐ
end

#####
##### Diagnostics
#####

const c = Center()

# Temperature
@inline function temperature(i, j, k, grid::AbstractGrid, mb::MoistAirBuoyancy, θ, qᵗ)
    @inbounds begin
        θᵢ = θ[i, j, k]
        qᵗᵢ = qᵗ[i, j, k]
        pᵣ = mb.reference_state.pressure[i, j, k]
        ρᵣ = mb.reference_state.density[i, j, k]
    end
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    p₀ = mb.reference_state.base_pressure
    q = MoistureMassFractions(qᵗᵢ)
    𝒰 = PotentialTemperatureState(θᵢ, q, z, p₀, pᵣ, ρᵣ)
    return compute_boussinesq_adjustment_temperature(𝒰, mb.thermodynamics)
end

struct TemperatureKernelFunction end
const TemperatureOperation = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:TemperatureKernelFunction}
const TemperatureField = Field{Center, Center, Center, <:TemperatureOperation}

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
@inline function saturation_specific_humidity(i, j, k, grid, mb::MoistAirBuoyancy, T, qᵗ, phase)
    @inbounds begin
        Tᵢ = T[i, j, k]
        qᵗᵢ = qᵗ[i, j, k]
        pᵣ = mb.reference_state.pressure[i, j, k]
    end
    q = MoistureMassFractions(qᵗᵢ)
    ρ = density(pᵣ, Tᵢ, q, mb.thermodynamics)
    return saturation_specific_humidity(Tᵢ, ρ, mb.thermodynamics, phase)
end

struct PhaseTransitionConstantsKernel{T, P}
    condensed_phase :: P
    temperature :: T
end

Adapt.adapt_structure(to, sk::PhaseTransitionConstantsKernel) =
    PhaseTransitionConstantsKernel(adapt(to, sk.condensed_phase),
                                   adapt(to, sk.temperature))

@inline function (kernel::PhaseTransitionConstantsKernel)(i, j, k, grid, buoyancy, qᵗ)
    T = kernel.temperature
    return saturation_specific_humidity(i, j, k, grid, buoyancy, T, qᵗ, kernel.condensed_phase)
end

function SaturationField(model, T = TemperatureField(model);
                         condensed_phase = model.buoyancy.formulation.thermodynamics.liquid)
    func = PhaseTransitionConstantsKernel(condensed_phase, T)
    grid = model.grid
    buoyancy = model.buoyancy.formulation
    qᵗ = model.tracers.qᵗ
    op = KernelFunctionOperation{Center, Center, Center}(func, grid, buoyancy, qᵗ)
    return Field(op)
end

# Condensate
struct CondensateKernel{T}
    temperature :: T
end

Adapt.adapt_structure(to, ck::CondensateKernel) = CondensateKernel(adapt(to, ck.temperature))

@inline function liquid_mass_fraction(i, j, k, grid, mb::MoistAirBuoyancy, T, θ, qᵗ)
    @inbounds begin
        Tᵢ = T[i, j, k]
        θᵢ = θ[i, j, k]
        qᵗᵢ = qᵗ[i, j, k]
        pᵣ = mb.reference_state.pressure[i, j, k]
        ρᵣ = mb.reference_state.density[i, j, k]
    end

    # First assume non-saturation.
    z = Oceananigans.Grids.znode(i, j, k, grid, c, c, c)
    p₀ = mb.reference_state.base_pressure
    q = MoistureMassFractions(qᵗᵢ)
    𝒰 = PotentialTemperatureState(Tᵢ, q, z, p₀, pᵣ, ρᵣ)
    Π = exner_function(𝒰, mb.thermodynamics)
    Tᵢ <= Π * θᵢ + 10 * eps(Tᵢ) && return zero(qᵗᵢ)

    # Next assume a saturation value
    qᵛ⁺ = adjustment_saturation_specific_humidity(Tᵢ, 𝒰, mb.thermodynamics)
    return max(0, qᵗᵢ - qᵛ⁺)
end

@inline function (kernel::CondensateKernel)(i, j, k, grid, buoyancy, θ, qᵗ)
    T = kernel.temperature
    return liquid_mass_fraction(i, j, k, grid, buoyancy, T, θ, qᵗ)
end

function CondensateField(model, T=TemperatureField(model))
    func = CondensateKernel(T)
    grid = model.grid
    buoyancy = model.buoyancy.formulation
    qᵗ = model.tracers.qᵗ
    θ = model.tracers.θ
    op = KernelFunctionOperation{Center, Center, Center}(func, grid, buoyancy, θ, qᵗ)
    return Field(op)
end

end # module
