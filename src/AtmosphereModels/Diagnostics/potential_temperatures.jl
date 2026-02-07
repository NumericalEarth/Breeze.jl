using Breeze.Thermodynamics:
    Thermodynamics,
    vapor_gas_constant,
    dry_air_gas_constant,
    liquid_latent_heat

#####
##### Potential temperatures (mixture, virtual, liquid-ice, equivalent, and stability-equivalent)
#####

# Plain (mixture) potential temperature flavors (θ = T / Π)
abstract type AbstractPlainFlavor end
struct SpecificPlain <: AbstractPlainFlavor end
struct PlainDensity <: AbstractPlainFlavor end

# Virtual potential temperature flavors
abstract type AbstractVirtualFlavor end
struct SpecificVirtual <: AbstractVirtualFlavor end
struct VirtualDensity <: AbstractVirtualFlavor end

# Liquid-ice potential temperature flavors
abstract type AbstractLiquidIceFlavor end
struct SpecificLiquidIce <: AbstractLiquidIceFlavor end
struct LiquidIceDensity <: AbstractLiquidIceFlavor end

# Equivalent potential temperature flavors
abstract type AbstractEquivalentFlavor end
struct SpecificEquivalent <: AbstractEquivalentFlavor end
struct EquivalentDensity <: AbstractEquivalentFlavor end

# Stability-equivalent potential temperature flavors (θᵇ)
# This is a subtype of AbstractEquivalentFlavor because the computation builds on θᵉ
abstract type AbstractStabilityEquivalentFlavor <: AbstractEquivalentFlavor end
struct SpecificStabilityEquivalent <: AbstractStabilityEquivalentFlavor end
struct StabilityEquivalentDensity <: AbstractStabilityEquivalentFlavor end

const SpecificPotentialTemperatureType = Union{
    SpecificPlain,
    SpecificVirtual,
    SpecificLiquidIce,
    SpecificEquivalent,
    SpecificStabilityEquivalent
}

const PotentialTemperatureDensityType = Union{
    PlainDensity,
    VirtualDensity,
    LiquidIceDensity,
    EquivalentDensity,
    StabilityEquivalentDensity
}

#####
##### Kernel function for potential temperature diagnostics
#####
##### The kernel function stores the three fields needed for computing potential temperature:
##### - pressure: the pressure field used in the Exner function
##### - density: the density field used for moisture fractions and density-weighted outputs
##### - standard_pressure: the reference pressure for potential temperature (typically 10⁵ Pa)
#####
##### For anelastic dynamics, these come from the reference state (pᵣ, ρᵣ, pˢᵗ).
##### For compressible dynamics, these are the actual pressure and density fields.
#####

struct MoistPotentialTemperatureKernelFunction{F, P, D, FT, μ, M, MF, TMP, TH}
    flavor :: F
    pressure :: P
    density :: D
    standard_pressure :: FT
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::MoistPotentialTemperatureKernelFunction) =
    MoistPotentialTemperatureKernelFunction(adapt(to, k.flavor),
                                            adapt(to, k.pressure),
                                            adapt(to, k.density),
                                            k.standard_pressure,
                                            adapt(to, k.microphysics),
                                            adapt(to, k.microphysical_fields),
                                            adapt(to, k.specific_moisture),
                                            adapt(to, k.temperature),
                                            adapt(to, k.thermodynamic_constants))

# Type aliases for the user interface
const C = Center

const PotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractPlainFlavor}}

const VirtualPotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractVirtualFlavor}}

const LiquidIcePotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractLiquidIceFlavor}}

const EquivalentPotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractEquivalentFlavor}}

const StabilityEquivalentPotentialTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any,
    <:MoistPotentialTemperatureKernelFunction{<:AbstractStabilityEquivalentFlavor}}

#####
##### Dynamics interface for potential temperature diagnostics
#####
##### Extract pressure, density, and standard_pressure from the dynamics.
##### For anelastic dynamics, use the reference state.
##### For compressible dynamics (defined in CompressibleEquations module), use actual fields.
#####

"""
$(TYPEDSIGNATURES)

Return the pressure field used for computing potential temperature.
For anelastic dynamics, this is the reference pressure.
"""
dynamics_pressure_for_potential_temperature(dynamics) = dynamics.reference_state.pressure

"""
    dynamics_density_for_potential_temperature(dynamics)

Return the density field used for computing potential temperature.
For anelastic dynamics, this is the reference density.
"""
dynamics_density_for_potential_temperature(dynamics) = dynamics.reference_state.density

"""
    dynamics_standard_pressure(dynamics)

Return the standard pressure for potential temperature calculations.
"""
dynamics_standard_pressure(dynamics) = dynamics.reference_state.standard_pressure

# Note: CompressibleDynamics methods are defined in CompressibleEquations/compressible_dynamics.jl
# since CompressibleEquations is loaded after AtmosphereModels

"""
    PotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing the (mixture) potential temperature ``θ``.

The potential temperature is defined as the temperature a parcel would have if
adiabatically brought to a reference pressure ``p₀``:

```math
θ = \\frac{T}{Π}
```

where ``T`` is temperature and ``Π = (p/p₀)^{Rᵐ/cᵖᵐ}`` is the mixture Exner function,
computed using the moist air gas constant ``Rᵐ`` and heat capacity ``cᵖᵐ``.

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θ``, or `:density` to return ``ρ θ``.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

θ = PotentialTemperature(model)
Field(θ)

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
function PotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        SpecificPlain()
    elseif flavor_symbol === :density
        PlainDensity()
    else
        msg = "`flavor` must be :specific or :density, received :$flavor_symbol"
        throw(ArgumentError(msg))
    end

    p = dynamics_pressure_for_potential_temperature(model.dynamics)
    ρ = dynamics_density_for_potential_temperature(model.dynamics)
    pˢᵗ = dynamics_standard_pressure(model.dynamics)

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   p, ρ, pˢᵗ,
                                                   model.microphysics,
                                                   model.microphysical_fields,
                                                   model.specific_moisture,
                                                   model.temperature,
                                                   model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

"""
    VirtualPotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing virtual potential temperature ``θᵛ``.

Virtual potential temperature is the temperature that dry air would need to have
in order to have the same density as moist air at the same pressure. To define virtual
potential temperature, we first note the definition of virtual _temperature_:

```math
Tᵛ = T \\left( 1 + δᵛ qᵛ - qˡ - qⁱ \\right)
```

where ``δᵛ ≡ Rᵛ / Rᵈ - 1``. This follows from the ideal gas law for a mixture,
``p = ρ Rᵐ T``, the mixture gas constant
``Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ = Rᵈ \\left( 1 + δᵛ qᵛ - qˡ - qⁱ \\right)``,
and the definition of virtual temperature, ``p = ρ Rᵈ Tᵛ``, which leads to

```math
Tᵛ = T \frac{Rᵐ}{Rᵈ} = T \\left( 1 + δᵛ qᵛ - qˡ - qⁱ \\right)
```

The virtual potential temperature is defined analogously,

```math
θᵛ = θ \\left( 1 + δᵛ qᵛ - qˡ - qⁱ \\right)
```

where ``θ`` is potential temperature. Note that
``Rᵛ / Rᵈ ≈ 1.608`` for water vapor and a dry air mixture typical to Earth's atmosphere,
and that ``δᵛ ≈ 0.608``.

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
    └── max=301.824, min=301.824, mean=301.824
```
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

    p = dynamics_pressure_for_potential_temperature(model.dynamics)
    ρ = dynamics_density_for_potential_temperature(model.dynamics)
    pˢᵗ = dynamics_standard_pressure(model.dynamics)

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   p, ρ, pˢᵗ,
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
θˡⁱ = θ \\left(1 - \\frac{ℒˡᵣ qˡ + ℒⁱᵣ qⁱ}{cᵖᵐ T}\\right)
```

where ``θ`` is the mixture potential temperature, ``ℒˡᵣ`` and ``ℒⁱᵣ`` are the reference
latent heats for liquid and ice, and ``qˡ``, ``qⁱ`` are the liquid and ice mass fractions.

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θˡⁱ``, or `:density` to return ``ρ θˡⁱ``.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

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

    p = dynamics_pressure_for_potential_temperature(model.dynamics)
    ρ = dynamics_density_for_potential_temperature(model.dynamics)
    pˢᵗ = dynamics_standard_pressure(model.dynamics)

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   p, ρ, pˢᵗ,
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
and tracking convective processes. Following [Emanuel1994](@cite Emanuel1994) equation 4.5.11:

```math
θᵉ = T \\left(\\frac{p₀}{p}\\right)^{Rᵈ/cᵖᵐ} \\exp\\left(\\frac{ℒˡ qᵛ}{cᵖᵐ T}\\right) ℋ^γ
```

where ``ℒˡ`` is the latent heat of vaporization, ``qᵛ`` is the vapor mass fraction,
``ℋ`` is the relative humidity, and ``γ = -Rᵛ qᵛ / cᵖᵐ``.

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
    └── max=325.849, min=325.849, mean=325.849
```

# References

* Emanuel, K. A. (1994). Atmospheric Convection. Oxford University Press.
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

    p = dynamics_pressure_for_potential_temperature(model.dynamics)
    ρ = dynamics_density_for_potential_temperature(model.dynamics)
    pˢᵗ = dynamics_standard_pressure(model.dynamics)

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   p, ρ, pˢᵗ,
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
computing the moist Brunt-Väisälä frequency. It follows from the derivation in the paper
by [Durran and Klemp (1982)](@cite DurranKlemp1982), who show that the moist Brunt-Väisälä
frequency ``Nᵐ`` is correctly expressed in terms of the vertical gradient of a
moist-conservative variable.

The formulation is based on equation (17) by [Durran and Klemp (1982)](@cite DurranKlemp1982):

```math
θᵇ = θᵉ \\left( \\frac{T}{Tᵣ} \\right)^{cˡ qˡ / cᵖᵐ}
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
    └── max=326.162, min=325.849, mean=326.005
```

# References

* Durran, D. R. and Klemp, J. B. (1982). On the effects of moisture on the Brunt-Väisälä frequency.
    Journal of the Atmospheric Sciences 39, 2152–2158.
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

    p = dynamics_pressure_for_potential_temperature(model.dynamics)
    ρ = dynamics_density_for_potential_temperature(model.dynamics)
    pˢᵗ = dynamics_standard_pressure(model.dynamics)

    func = MoistPotentialTemperatureKernelFunction(flavor,
                                                   p, ρ, pˢᵗ,
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
        p = d.pressure[i, j, k]
        ρ = d.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        pˢᵗ = d.standard_pressure
        T = d.temperature[i, j, k]
    end

    constants = d.thermodynamic_constants
    q = grid_moisture_fractions(i, j, k, grid, d.microphysics, ρ, qᵗ, d.microphysical_fields)
    qᵛ = q.vapor
    qˡ = q.liquid
    qⁱ = q.ice

    # Extract thermodynamic constants
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat

    # Plain properties
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Πᵐ = (p / pˢᵗ)^(Rᵐ / cᵖᵐ)

    # Plain potential temperature (used as a base for several others)
    θ = T / Πᵐ

    θ★ = if d.flavor isa AbstractPlainFlavor
        θ

    elseif d.flavor isa AbstractLiquidIceFlavor
        # Liquid-ice potential temperature
        θ * (1 - (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / (cᵖᵐ * T))

    elseif d.flavor isa AbstractVirtualFlavor
        δ = Rᵛ / Rᵈ - 1
        θ * (1 + δ * qᵛ - qˡ - qⁱ)

    elseif d.flavor isa AbstractEquivalentFlavor
        # Saturation specific humidity over a liquid surface
        surface = PlanarLiquidSurface()
        ℋ = relative_humidity(T, p, q, constants, surface)
        γ = - Rᵛ * qᵛ / cᵖᵐ

        # Latent heat of vaporization at temperature T
        ℒˡ = liquid_latent_heat(T, constants)

        # Equation 4.5.11 in Emanuel 1994
        # See also equation 17 in Durran & Klemp 1982
        # TODO: many things here...
        # - Equation 4.5.11 (and Emmanuel 1994's whole development) via moist entropy uses mixing ratios.
        # - I have (mostly) guessed about these expressions, which we must form in terms
        #   of mass fractions.
        # - When this is verified, the math should be written in the documentation.
        # - Could this also be written θᵉ = θ * exp(ℒˡ * qᵛ / (cᵖᵐ * T)) * ℋ^γ ?
        θᵉ = T * (pˢᵗ / p)^(Rᵈ / cᵖᵐ) * exp(ℒˡ * qᵛ / (cᵖᵐ * T)) * ℋ^γ

        if d.flavor isa AbstractStabilityEquivalentFlavor
            # Equation 16, Durran & Klemp 1982
            Tᵣ = constants.energy_reference_temperature
            cˡ = constants.liquid.heat_capacity
            θᵉ * (T / Tᵣ)^(cˡ * qˡ / cᵖᵐ)
        else # d.flavor isa AbstractEquivalentFlavor (but not stability)
            θᵉ
        end
    end

    # Return specific or density-weighted value
    if d.flavor isa SpecificPotentialTemperatureType
        return θ★
    elseif d.flavor isa PotentialTemperatureDensityType
        return ρ * θ★
    end
end
