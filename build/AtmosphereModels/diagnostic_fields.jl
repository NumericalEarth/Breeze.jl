using ..Thermodynamics:
    Thermodynamics,
    MoistureMassFractions,
    dry_air_mass_fraction,
    total_specific_moisture,
    vapor_gas_constant,
    dry_air_gas_constant,
    saturation_vapor_pressure,
    PlanarLiquidSurface

using Oceananigans: Center, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Adapt: Adapt, adapt

struct Specific end
struct Density end

#####
##### Dry potential temperature
#####

struct DryPotentialTemperatureKernelFunction{R, TMP, TH}
    reference_state :: R
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::DryPotentialTemperatureKernelFunction) =
    DryPotentialTemperatureKernelFunction(adapt(to, k.reference_state),
                                          adapt(to, k.temperature),
                                          adapt(to, k.thermodynamic_constants))

const DryPotentialTemperature = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:DryPotentialTemperatureKernelFunction}

"""
    DryPotentialTemperature(model)

Return a `KernelFunctionOperation` representing dry potential temperature ``θ``.

Dry potential temperature is the temperature that an unsaturated parcel of air
would attain if adiabatically brought to a reference pressure ``p_0``:

```math
θ = T \\left( \\frac{p_0}{p} \\right)^{R^d / c_p^d}
```

where ``T`` is temperature, ``p`` is pressure, ``p_0`` is the reference pressure,
``R^d`` is the dry air gas constant, and ``c_p^d`` is the specific heat capacity
of dry air at constant pressure.

See [Stull1988](@citet) for a thorough discussion of potential temperature.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300)

θ = DryPotentialTemperature(model)
compute!(Field(θ))

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

# References

* [Stull1988](@cite)
"""
function DryPotentialTemperature(model::AtmosphereModel)
    func = DryPotentialTemperatureKernelFunction(model.formulation.reference_state,
                                                  model.temperature,
                                                  model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

function (d::DryPotentialTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    thermo = d.thermodynamic_constants
    Rᵈ = dry_air_gas_constant(thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity

    # Dry potential temperature: θ = T * (p₀ / p)^(Rᵈ / cᵖᵈ)
    Π = (pᵣ / p₀)^(Rᵈ / cᵖᵈ)
    return T / Π
end

#####
##### Virtual potential temperature
#####

struct VirtualPotentialTemperatureKernelFunction{F, R, μ, M, MF, TMP, TH}
    flavor :: F
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::VirtualPotentialTemperatureKernelFunction) =
    VirtualPotentialTemperatureKernelFunction(adapt(to, k.flavor),
                                              adapt(to, k.reference_state),
                                              adapt(to, k.microphysics),
                                              adapt(to, k.microphysical_fields),
                                              adapt(to, k.specific_moisture),
                                              adapt(to, k.temperature),
                                              adapt(to, k.thermodynamic_constants))

const VirtualPotentialTemperature = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:VirtualPotentialTemperatureKernelFunction}

"""
    VirtualPotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing virtual potential temperature ``θ_v``.

Virtual potential temperature is the temperature that dry air would need to have
in order to have the same density as moist air at the same pressure. It accounts
for the effect of water vapor on air density:

```math
θ_v = θ \\left( 1 + \\epsilon q^v - q^l - q^i \\right)
```

where ``θ`` is dry potential temperature, ``q^v``, ``q^l``, ``q^i`` are the
specific humidities of vapor, liquid, and ice respectively, and
``ε = R^v / R^d - 1 ≈ 0.608`` is a constant related to the ratio of gas constants.

See [Emanuel1994](@citet) for a derivation and discussion of virtual temperature
and its utility in atmospheric thermodynamics.

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θ_v``, or `:density` to return ``ρ θ_v``.

# Examples

Virtual potential temperature is larger than dry potential temperature when moisture is present:

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

θᵥ = VirtualPotentialTemperature(model)
θ = DryPotentialTemperature(model)
θᵥ_field = compute!(Field(θᵥ))
θ_field = compute!(Field(θ))
minimum(θᵥ_field) > minimum(θ_field)

# output
true
```

# References

* [Emanuel1994](@cite)
"""
function VirtualPotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        Specific()
    elseif flavor_symbol === :density
        Density()
    else
        error("Unknown $flavor_symbol")
    end

    func = VirtualPotentialTemperatureKernelFunction(flavor,
                                                     model.formulation.reference_state,
                                                     model.microphysics,
                                                     model.microphysical_fields,
                                                     model.specific_moisture,
                                                     model.temperature,
                                                     model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

function (d::VirtualPotentialTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    thermo = d.thermodynamic_constants
    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)

    # Compute dry potential temperature
    Rᵈ = dry_air_gas_constant(thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Π = (pᵣ / p₀)^(Rᵈ / cᵖᵈ)
    θ = T / Π

    # Virtual correction factor: ε = Rᵛ/Rᵈ - 1
    Rᵛ = vapor_gas_constant(thermo)
    ε = Rᵛ / Rᵈ - 1
    qᵛ = q.vapor
    qˡ = q.liquid
    qⁱ = q.ice

    θᵥ = θ * (1 + ε * qᵛ - qˡ - qⁱ)

    if d.flavor isa Specific
        return θᵥ
    elseif d.flavor isa Density
        return ρᵣ * θᵥ
    end
end

#####
##### Equivalent potential temperature
#####

struct EquivalentPotentialTemperatureKernelFunction{F, R, μ, M, MF, TMP, TH}
    flavor :: F
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::EquivalentPotentialTemperatureKernelFunction) =
    EquivalentPotentialTemperatureKernelFunction(adapt(to, k.flavor),
                                                 adapt(to, k.reference_state),
                                                 adapt(to, k.microphysics),
                                                 adapt(to, k.microphysical_fields),
                                                 adapt(to, k.specific_moisture),
                                                 adapt(to, k.temperature),
                                                 adapt(to, k.thermodynamic_constants))

const EquivalentPotentialTemperature = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:EquivalentPotentialTemperatureKernelFunction}

"""
    EquivalentPotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing equivalent potential temperature ``θ_e``.

Equivalent potential temperature is conserved during moist adiabatic processes
(including condensation and evaporation) and is useful for identifying air masses
and diagnosing moist instabilities. It is the temperature that a parcel would have
if all its moisture were condensed out and the resulting latent heat used to warm
the parcel, followed by adiabatic expansion to a reference pressure.

We use the formulation from [BryanFritsch2002](@citet), which provides an accurate
approximation:

```math
θ_e = T \\left( \\frac{p_0}{p_d} \\right)^{R^d / c_{pm}}
      \\exp \\left( \\frac{ℒ_v q^v}{c_{pm} T} \\right)
```

where ``T`` is temperature, ``p_d`` is dry air pressure, ``p_0`` is the reference pressure,
``ℒ_v`` is the latent heat of vaporization, ``q^v`` is the vapor specific humidity,
and ``c_{pm}`` is the heat capacity of the moist air mixture.

The formulation follows equation (39) of [BryanFritsch2002](@cite), which was
adapted from the derivation in [DurranKlemp1982](@citet).

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θ_e``, or `:density` to return ``ρ θ_e``.

# Examples

Equivalent potential temperature is larger than dry potential temperature when moisture
is present (due to the latent heat of condensation):

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

θₑ = EquivalentPotentialTemperature(model)
θ = DryPotentialTemperature(model)
θₑ_field = compute!(Field(θₑ))
θ_field = compute!(Field(θ))
minimum(θₑ_field) > minimum(θ_field)

# output
true
```

# References

* [BryanFritsch2002](@cite)
* [DurranKlemp1982](@cite)
"""
function EquivalentPotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        Specific()
    elseif flavor_symbol === :density
        Density()
    else
        error("Unknown $flavor_symbol")
    end

    func = EquivalentPotentialTemperatureKernelFunction(flavor,
                                                        model.formulation.reference_state,
                                                        model.microphysics,
                                                        model.microphysical_fields,
                                                        model.specific_moisture,
                                                        model.temperature,
                                                        model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

function (d::EquivalentPotentialTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    thermo = d.thermodynamic_constants
    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)

    # Heat capacity and gas constants
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q, thermo)

    # Dry air pressure: pᵈ = pᵣ - pᵛ, where pᵛ = ρ qᵛ Rᵛ T
    # Using the approximation pᵈ ≈ pᵣ * (1 - qᵛ * Rᵛ / Rᵐ)
    qᵛ = q.vapor
    Rᵐ = Thermodynamics.mixture_gas_constant(q, thermo)
    pᵈ = pᵣ * (1 - qᵛ * Rᵛ / Rᵐ)

    # Latent heat of vaporization at temperature T
    # ℒᵛ = ℒᵛᵣ + (cᵖᵛ - cˡ)(T - Tᵣ)
    ℒᵛᵣ = thermo.liquid.reference_latent_heat
    cᵖᵛ = thermo.vapor.heat_capacity
    cˡ = thermo.liquid.heat_capacity
    Tᵣ = thermo.energy_reference_temperature
    ℒᵛ = ℒᵛᵣ + (cᵖᵛ - cˡ) * (T - Tᵣ)

    # Equivalent potential temperature following Bryan & Fritsch (2002) Eq. (39)
    # θₑ = T * (p₀ / pᵈ)^(Rᵈ / cᵖᵐ) * exp(ℒᵛ * qᵛ / (cᵖᵐ * T))
    θₑ = T * (p₀ / pᵈ)^(Rᵈ / cᵖᵐ) * exp(ℒᵛ * qᵛ / (cᵖᵐ * T))

    if d.flavor isa Specific
        return θₑ
    elseif d.flavor isa Density
        return ρᵣ * θₑ
    end
end

struct LiquidIcePotentialTemperatureKernelFunction{F, R, μ, M, MF, TMP, TH}
    flavor :: F
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end

Adapt.adapt_structure(to, k::LiquidIcePotentialTemperatureKernelFunction) =
    LiquidIcePotentialTemperatureKernelFunction(adapt(to, k.flavor),
                                                adapt(to, k.reference_state),
                                                adapt(to, k.microphysics),
                                                adapt(to, k.microphysical_fields),
                                                adapt(to, k.specific_moisture),
                                                adapt(to, k.temperature),
                                                adapt(to, k.thermodynamic_constants))

const LiquidIcePotentialTemperature = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:LiquidIcePotentialTemperatureKernelFunction}
const LiquidIcePotentialTemperatureField = Field{Center, Center, Center, <:LiquidIcePotentialTemperature}

"""
    LiquidIcePotentialTemperature(model, flavor=:specific)

Return a `KernelFunctionOperation` representing liquid-ice potential temperature ``θ_{li}``.

Liquid-ice potential temperature is a conserved quantity under moist adiabatic processes
that accounts for the latent heat associated with liquid water and ice:

```math
θ_{li} = \\frac{T - (ℒ^l_r q^l + ℒ^i_r q^i) / c_p^m}{Π}
```

where ``Π = (p/p_0)^{R^m/c_p^m}`` is the Exner function using mixture properties,
``ℒ^l_r`` and ``ℒ^i_r`` are the reference latent heats of vaporization and sublimation,
``q^l`` and ``q^i`` are the liquid and ice specific humidities, and
``c_p^m`` is the moist air heat capacity.

This is the prognostic thermodynamic variable used in `LiquidIcePotentialTemperatureThermodynamics`.

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``θ_{li}``, or `:density` to return ``ρ θ_{li}``.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300)

θₗᵢ = LiquidIcePotentialTemperature(model)
θₗᵢ_field = compute!(Field(θₗᵢ))
all(interior(θₗᵢ_field) .≈ 300)

# output
true
```
"""
function LiquidIcePotentialTemperature(model::AtmosphereModel, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        Specific()
    elseif flavor_symbol === :density
        Density()
    else
        error("Unknown $flavor_symbol")
    end

    func = LiquidIcePotentialTemperatureKernelFunction(flavor, 
                                                       model.formulation.reference_state,
                                                       model.microphysics,
                                                       model.microphysical_fields,
                                                       model.specific_moisture,
                                                       model.temperature,
                                                       model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

"""
    LiquidIcePotentialTemperatureField(model, flavor=:specific)

Return a `Field` containing liquid-ice potential temperature.

See [`LiquidIcePotentialTemperature`](@ref) for details on the formulation.
"""
LiquidIcePotentialTemperatureField(model, flavor_symbol=:specific) =
    LiquidIcePotentialTemperature(model, flavor_symbol) |> Field

function (d::LiquidIcePotentialTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    Rᵐ = Thermodynamics.mixture_gas_constant(q, d.thermodynamic_constants)
    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q, d.thermodynamic_constants)
    Π = (pᵣ / p₀)^(Rᵐ / cᵖᵐ)

    ℒˡᵣ = d.thermodynamic_constants.liquid.reference_latent_heat
    ℒⁱᵣ = d.thermodynamic_constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    θ = (T - (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ) / Π

    if d.flavor isa Specific
        return θ
    elseif d.flavor isa Density
        return ρᵣ * θ
    end
end

#####
##### Static energy
#####

struct StaticEnergyKernelFunction{F, R, μ, M, MF, TMP, TH}
    flavor :: F
    reference_state :: R
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: TMP
    thermodynamic_constants :: TH
end
 
Adapt.adapt_structure(to, k::StaticEnergyKernelFunction) =
    StaticEnergyKernelFunction(adapt(to, k.flavor),
                               adapt(to, k.reference_state),
                               adapt(to, k.microphysics),
                               adapt(to, k.microphysical_fields),
                               adapt(to, k.specific_moisture),
                               adapt(to, k.temperature),
                               adapt(to, k.thermodynamic_constants))

const StaticEnergy = KernelFunctionOperation{Center, Center, Center, <:Any, <:Any, <:StaticEnergyKernelFunction}
const StaticEnergyField = Field{Center, Center, Center, <:StaticEnergy}

"""
    StaticEnergy(model, flavor=:specific)

Return a `KernelFunctionOperation` representing moist static energy ``e``.

Moist static energy is a conserved quantity in adiabatic, frictionless flow that
combines sensible heat, gravitational potential energy, and latent heat:

```math
e = c_p^m T + g z - ℒ^l_r q^l - ℒ^i_r q^i
```

where ``c_p^m`` is the moist air heat capacity, ``T`` is temperature,
``g`` is gravitational acceleration, ``z`` is height, and
``ℒ^l_r q^l + ℒ^i_r q^i`` is the latent heat content of condensate.

This is the prognostic thermodynamic variable used in `StaticEnergyThermodynamics`.

# Arguments

- `model`: An `AtmosphereModel` instance.
- `flavor`: Either `:specific` (default) to return ``e``, or `:density` to return ``ρ e``.

# Examples

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300)

e = StaticEnergy(model)
e_field = compute!(Field(e))
minimum(e_field) > 0  # static energy is positive

# output
true
```
"""
function StaticEnergy(model, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        Specific()
    elseif flavor_symbol === :density
        Density()
    else
        error("Unknown $flavor_symbol")
    end

    func = StaticEnergyKernelFunction(flavor,
                                      model.formulation.reference_state,
                                      model.microphysics,
                                      model.microphysical_fields,
                                      model.specific_moisture,
                                      model.temperature,
                                      model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

"""
    StaticEnergyField(model, flavor=:specific)

Return a `Field` containing moist static energy.

See [`StaticEnergy`](@ref) for details on the formulation.
"""
StaticEnergyField(model, flavor_symbol=:specific) =
    StaticEnergy(model, flavor_symbol) |> Field

function (d::StaticEnergyKernelFunction)(i, j, k, grid)
    @inbounds begin
        ρᵣ = d.reference_state.density[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
        p₀ = d.reference_state.base_pressure
        T = d.temperature[i, j, k]
    end

    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    cᵖᵐ = Thermodynamics.mixture_heat_capacity(q, d.thermodynamic_constants)

    g = d.thermodynamic_constants.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)

    ℒˡᵣ = d.thermodynamic_constants.liquid.reference_latent_heat
    ℒⁱᵣ = d.thermodynamic_constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    # Moist static energy
    e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ

    if d.flavor isa Specific
        return e
    elseif d.flavor isa Density
        return ρᵣ * e
    end
end
