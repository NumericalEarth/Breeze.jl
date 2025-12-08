#####
##### Moist potential temperatures (virtual, liquid-ice, equivalent, and stability-equivalent)
#####

# Abstract type hierarchy for moist potential temperature flavors
abstract type AbstractMoistPotentialTemperatureFlavor end

# Virtual potential temperature flavors
abstract type AbstractVirtualFlavor <: AbstractMoistPotentialTemperatureFlavor end
struct SpecificVirtual <: AbstractVirtualFlavor end
struct VirtualDensity <: AbstractVirtualFlavor end

# Liquid-ice potential temperature flavors
abstract type AbstractLiquidIceFlavor <: AbstractMoistPotentialTemperatureFlavor end
struct SpecificLiquidIce <: AbstractLiquidIceFlavor end
struct LiquidIceDensity <: AbstractLiquidIceFlavor end

# Equivalent potential temperature flavors  
abstract type AbstractEquivalentFlavor <: AbstractMoistPotentialTemperatureFlavor end
struct SpecificEquivalent <: AbstractEquivalentFlavor end
struct EquivalentDensity <: AbstractEquivalentFlavor end

# Stability-equivalent potential temperature flavors (θᵇ)
abstract type AbstractStabilityEquivalentFlavor <: AbstractMoistPotentialTemperatureFlavor end
struct SpecificStabilityEquivalent <: AbstractStabilityEquivalentFlavor end
struct StabilityEquivalentDensity <: AbstractStabilityEquivalentFlavor end

const SpecificPotentialTemperature = Union{
    SpecificVirtual,
    SpecificLiquidIce,
    SpecificEquivalent,
    SpecificStabilityEquivalent
}

const PotentialTemperatureDensity = Union{
    VirtualDensity,
    LiquidIceDensity,
    EquivalentDensity,
    StabilityEquivalentDensity
}

struct MoistPotentialTemperatureKernelFunction{F, R, μ, M, MF, TMP, TH}
    flavor :: F
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::MoistPotentialTemperatureKernelFunction) =
    MoistPotentialTemperatureKernelFunction(adapt(to, k.flavor),
                                            adapt(to, k.reference_state),
                                            adapt(to, k.microphysics),
                                            adapt(to, k.microphysical_fields),
                                            adapt(to, k.specific_moisture),
                                            adapt(to, k.temperature),
                                            adapt(to, k.thermodynamic_constants))

# Type aliases for the user interface
const C = Center
const VirtualPotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractVirtualFlavor}}

const LiquidIcePotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractLiquidIceFlavor}}

const EquivalentPotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractEquivalentFlavor}}

const StabilityEquivalentPotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractStabilityEquivalentFlavor}}

"""
    VirtualPotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing virtual potential temperature ``θᵛ``.

Virtual potential temperature is the temperature that dry air would need to have
in order to have the same density as moist air at the same pressure. It accounts
for the effect of water vapor on air density:

```math
θᵛ = θᵈ \\left( qᵈ + ε qᵛ \\right)
```

where ``θᵈ`` is dry potential temperature, ``qᵛ``, ``qˡ``, ``qⁱ`` are the
specific humidities of vapor, liquid, and ice respectively, and
``ε = Rᵛ / Rᵈ ≈ 1.608`` is the ratio between the vapor and dry air gas constants.

See [Emanuel1994](@citet) for a derivation and discussion of virtual temperature
and its utility in atmospheric thermodynamics.

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θᵛ``, or `:density` to return ``ρ θᵛ``.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

θᵛ = VirtualPotentialTemperature(model)
Field(θᵛ)

# output
1×1×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×14 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:11) with eltype Float64 with indices 0:2×0:2×-2:11
    └── max=301.823, min=301.803, mean=301.813
```

# References

* [Emanuel1994](@cite)
"""
function VirtualPotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        SpecificVirtual()
    elseif flavor_symbol === :density
        VirtualDensity()
    else
        msg = "`flavor` must be :specific or :density, received :$flavor_symbol"
        throw(ArgumentError(msg))
    end

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   model.formulation.reference_state,
                                                   model.microphysics,
                                                   model.microphysical_fields,
                                                   model.specific_moisture,
                                                   model.temperature,
                                                   model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

"""
    LiquidIcePotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing liquid-ice potential temperature ``θˡⁱ``.

Liquid-ice potential temperature is a conserved quantity under moist adiabatic processes
that accounts for the latent heat associated with liquid water and ice:

```math
θˡⁱ = \\frac{T - (ℒˡᵣ qˡ + ℒⁱᵣ qⁱ) / cᵖᵐ}{Π}
```

where ``Π = (p/p₀)^{Rᵐ/cᵖᵐ}`` is the Exner function using mixture properties,
``ℒˡᵣ`` and ``ℒⁱᵣ`` are the reference latent heats of vaporization and sublimation,
``qˡ`` and ``qⁱ`` are the liquid and ice specific humidities, and
``cᵖᵐ`` is the moist air heat capacity.

This is the prognostic thermodynamic variable used in `LiquidIcePotentialTemperatureThermodynamics`.

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θˡⁱ``, or `:density` to return ``ρ θˡⁱ``.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300)

θˡⁱ = LiquidIcePotentialTemperature(model)
Field(θˡⁱ)

# output
1×1×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×14 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:11) with eltype Float64 with indices 0:2×0:2×-2:11
    └── max=300.0, min=300.0, mean=300.0
```
"""
function LiquidIcePotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        SpecificLiquidIce()
    elseif flavor_symbol === :density
        LiquidIceDensity()
    else
        msg = "`flavor` must be :specific or :density, received :$flavor_symbol"
        throw(ArgumentError(msg))
    end

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   model.formulation.reference_state,
                                                   model.microphysics,
                                                   model.microphysical_fields,
                                                   model.specific_moisture,
                                                   model.temperature,
                                                   model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

"""
    EquivalentPotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing equivalent potential temperature ``θᵉ``.

Equivalent potential temperature is conserved during moist adiabatic processes
(including condensation and evaporation) and is useful for identifying air masses
and diagnosing moist instabilities. It is the temperature that a parcel would have
if all its moisture were condensed out and the resulting latent heat used to warm
the parcel, followed by adiabatic expansion to a reference pressure.

We use the formulation from [BryanFritsch2002](@citet), which provides an accurate
approximation:

```math
θᵉ = T \\left( \\frac{p₀}{pᵈ} \\right)^{Rᵈ / cᵖᵐ}
      \\exp \\left( \\frac{ℒˡ qᵛ}{cᵖᵐ T} \\right)
```

where ``T`` is temperature, ``pᵈ`` is dry air pressure, ``p₀`` is the reference pressure,
``ℒˡ`` is the latent heat of vaporization, ``qᵛ`` is the vapor specific humidity,
and ``cᵖᵐ`` is the heat capacity of the moist air mixture.

The formulation follows equation (34) of [BryanFritsch2002](@cite), 
adapted from the derivation in [DurranKlemp1982](@citet).

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θᵉ``, or `:density` to return ``ρ θᵉ``.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

θᵉ = EquivalentPotentialTemperature(model)
Field(θᵉ)

# output
1×1×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×14 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:11) with eltype Float64 with indices 0:2×0:2×-2:11
    └── max=326.469, min=325.564, mean=326.012
```

# References

* [BryanFritsch2002](@cite)
* [DurranKlemp1982](@cite)
"""
function EquivalentPotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        SpecificEquivalent()
    elseif flavor_symbol === :density
        EquivalentDensity()
    else
        msg = "`flavor` must be :specific or :density, received :$flavor_symbol"
        throw(ArgumentError(msg))
    end

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   model.formulation.reference_state,
                                                   model.microphysics,
                                                   model.microphysical_fields,
                                                   model.specific_moisture,
                                                   model.temperature,
                                                   model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

"""
    StabilityEquivalentPotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing stability-equivalent potential temperature ``θᵇ``.

Stability-equivalent potential temperature is a moist-conservative variable suitable for
computing the moist Brunt-Väisälä frequency. It follows from the derivation in
[DurranKlemp1982](@citet), who show that the moist Brunt-Väisälä frequency ``Nᵐ`` is
correctly expressed in terms of the vertical gradient of a moist-conservative variable.

The formulation is based on equation (17) in [DurranKlemp1982](@cite):

```math
θᵇ = θᵉ \\left( \\frac{T}{Tᵣ} \\right)^{cˡ qᵗ / cᵖᵐ}
```

where ``θᵉ`` is the equivalent potential temperature, ``T`` is temperature, ``Tᵣ`` is
the energy reference temperature, ``cˡ`` is the heat capacity of liquid water,
``qᵗ`` is the total moisture specific humidity, and ``cᵖᵐ`` is the moist air heat capacity.

This quantity is conserved along moist adiabats and is appropriate for use in stability
calculations in saturated atmospheres.

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θᵇ``, or `:density` to return ``ρ θᵇ``.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

θᵇ = StabilityEquivalentPotentialTemperature(model)
Field(θᵇ)

# output
1×1×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×14 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:11) with eltype Float64 with indices 0:2×0:2×-2:11
    └── max=326.469, min=325.564, mean=326.012
```

# References

* [DurranKlemp1982](@cite)
"""
function StabilityEquivalentPotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        SpecificStabilityEquivalent()
    elseif flavor_symbol === :density
        StabilityEquivalentDensity()
    else
        msg = "`flavor` must be :specific or :density, received :$flavor_symbol"
        throw(ArgumentError(msg))
    end

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   model.formulation.reference_state,
                                                   model.microphysics,
                                                   model.microphysical_fields,
                                                   model.specific_moisture,
                                                   model.temperature,
                                                   model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

#####
##### Unified kernel function
#####

function (d::MoistPotentialTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    constants = d.thermodynamic_constants
    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    qᵛ = q.vapor
    qˡ = q.liquid
    qⁱ = q.ice

    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Πᵐ = (pᵣ / p₀)^(Rᵐ / cᵖᵐ)

    if d.flavor isa AbstractLiquidIceFlavor || d.flavor isa AbstractVirtualFlavor
        # Liquid-ice potential temperature
        θˡⁱ = (T - (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ) / Πᵐ

        if d.flavor isa AbstractLiquidIceFlavor
            θ★ = θˡⁱ

        elseif d.flavor isa AbstractVirtualFlavor
            θ★ = θˡⁱ * (1 + Rᵛ / Rᵈ * qᵛ)

        end

    elseif d.flavor isa AbstractEquivalentFlavor
        # Saturation specific humidity over a liquid surface
        surface = PlanarLiquidSurface()
        ℋ = relative_humidity(pᵣ, T, q, constants, surface)
        γ = - qᵛ * Rᵛ / cᵖᵐ

        # Latent heat of vaporization at temperature T
        ℒˡ = liquid_latent_heat(T, constants)

        # Equation 4.5.11 in Emanuel 1994
        # See also equation 17 in Durran & Klemp 1982
        # TODO: many things here... 
        # - Equation 4.5.11 (and Emmanuel 1994's whole development) via moist entropy uses mixing ratios.
        # - I have actually just guessed about these expressions, which we must form in terms
        #   of mass fractions.
        # - Not to mention that "specific entropy" should be entropy per
        #   unit total mass, rather than per unit dry air mass, as in Emmanuel.
        # - When this is verified, the math should be written in the documentation.
        θᵉ = (pᵣ / p₀)^(Rᵐ / cᵖᵐ) * ℋ^γ * exp(ℒˡ * qᵛ / (cᵖᵐ * T))

        if d.flavor isa AbstractStabilityEquivalentFlavor
            # Equation 16, Durran & Klemp 1982
            θ★ = θᵉ + (T / Tᵣ)^(cˡ * qᵗ / cᵖᵐ)

        elseif d.flavor isa AbstractEquivalentFlavor
            θ★ = θᵉ

        end
    end

    # Return specific or density-weighted value
    if d.flavor isa SpecificPotentialTemperature
        return θ★
    elseif d.flavor isa PotentialTemperatureDensity
        return ρᵣ * θ★
    end
end
