using Adapt: Adapt, adapt
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, Center

using Breeze.Thermodynamics:
    saturation_specific_humidity,
    dry_air_gas_constant,
    vapor_gas_constant,
    density,
    saturation_vapor_pressure

import Oceananigans.Utils: prettysummary

struct SaturationSpecificHumidityKernelFunction{μ, FL, M, MF, T, R, TH}
    flavor :: FL
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: T
    reference_state :: R
    thermodynamic_constants :: TH
end

prettysummary(kf::SaturationSpecificHumidityKernelFunction) = "$(kf.flavor) SaturationSpecificHumidityKernelFunction"

Adapt.adapt_structure(to, k::SaturationSpecificHumidityKernelFunction) =
    SaturationSpecificHumidityKernelFunction(adapt(to, k.flavor),
                                             adapt(to, k.microphysics),
                                             adapt(to, k.microphysical_fields),
                                             adapt(to, k.specific_moisture),
                                             adapt(to, k.temperature),
                                             adapt(to, k.reference_state),
                                             adapt(to, k.thermodynamic_constants))

const C = Center
const SaturationSpecificHumidity = KernelFunctionOperation{C, C, C, <:Any, <:Any, <:SaturationSpecificHumidityKernelFunction}

struct Prognostic end
struct Equilibrium end
struct TotalMoisture end

"""
$(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` representing the specified flavor
of *saturation specific humidity* ``qᵛ⁺`` which correpsonds to `model.microphysics`.
If `model.microphysics` is not a saturation adjustment scheme, then
a warm phase scheme is assumed which computes the saturation specific humidity
over a planar liquid surface.

## Flavor options

* `:prognostic`

  Return the *saturation specific humidity* corresponding to the `model`'s prognostic state.
  This is the same as the equilibrium saturation specific humidity for saturated conditions
  and a model that uses saturation adjustment microphysics.

* `:equilibrium`

  Return the *saturation specific humidity* in saturated conditions, using the
  `model.specific_moisture`. This is equivalent to the `:total_moisture` flavor
  under saturated conditions with no condensate; or in other words, if `model.specific_moisture` happens
  to be equal to the saturation specific humidity.

* `:total_moisture`

  Return *saturation specific humidity* in the case that the total specific moisture is
  equal to the saturation specific humidity and there is no condensate.
  This is useful for manufacturing perfectly saturated initial conditions.

## Examples

```jldoctest ssh
using Breeze
grid = RectilinearGrid(size=(1, 1, 128), extent=(1e3, 1e3, 1e3))
microphysics = SaturationAdjustment()
model = AtmosphereModel(grid; microphysics)
set!(model, θ=300)
qᵛ⁺ = SaturationSpecificHumidity(model, :prognostic)

# output
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×128 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: Breeze.Microphysics.Prognostic() SaturationSpecificHumidityKernelFunction
└── arguments: ()
```

As `SaturationSpecificHumidity` it may be wrapped in `Field` to store the result
of its computation. For example, a `Field` representing the equilibrium saturation specific
humidity may be formed via,

```jldoctest ssh
qᵛ = SaturationSpecificHumidity(model, :equilibrium) |> Field

# output
1×1×128 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×128 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×134 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:131) with eltype Float64 with indices 0:2×0:2×-2:131
    └── max=0.0361828, min=0.0224965, mean=0.028878
```

We also provide a constructor and type alias for the `Field` itself.
For example, to build a `Field` representing the saturation specific humidity
in the case that the total specific moisture is exactly at saturation,

```jldoctest ssh
qᵗ = SaturationSpecificHumidityField(model, :total_moisture)

# output
1×1×128 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×128 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×134 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:131) with eltype Float64 with indices 0:2×0:2×-2:131
    └── max=0.0561539, min=0.0353807, mean=0.0451121
```
"""
function SaturationSpecificHumidity(model, flavor_symbol=:prognostic)

    flavor = if flavor_symbol == :prognostic
        Prognostic()
    elseif flavor_symbol == :equilibrium
        Equilibrium()
    elseif flavor_symbol == :total_moisture
        TotalMoisture()
    else
        valid_flavors = (:prognostic, :equilibrium, :total_moisture)
        throw(ArgumentError("Flavor $flavor_symbol is not one of the valid flavors $valid_flavors"))
    end

    microphysics = if model.microphysics isa SaturationAdjustment
        model.microphysics
    else
        SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    end

    func = SaturationSpecificHumidityKernelFunction(flavor,
                                                    microphysics,
                                                    model.microphysical_fields,
                                                    model.specific_moisture,
                                                    model.temperature,
                                                    model.formulation.reference_state,
                                                    model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

@inline function saturation_total_specific_moisture(T, pᵣ, constants, equil)
    surface = equilibrated_surface(equil, T)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    δᵈᵛ = Rᵈ / Rᵛ - 1
    return pᵛ⁺ / (pᵣ + δᵈᵛ * pᵛ⁺)
end

const AdjustmentSH = SaturationSpecificHumidityKernelFunction{<:SaturationAdjustment}

function (d::AdjustmentSH)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        T = d.temperature[i, j, k]
    end

    constants = d.thermodynamic_constants
    equil = d.microphysics.equilibrium

    if d.flavor isa Prognostic
        qᵗ = @inbounds d.specific_moisture[i, j, k]
        q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
        ρ = density(pᵣ, T, q, constants)
        surface = equilibrated_surface(equil, T)
        return saturation_specific_humidity(T, ρ, constants, surface)

    elseif d.flavor isa Equilibrium
        qᵗ = @inbounds d.specific_moisture[i, j, k]
        return equilibrium_saturation_specific_humidity(T, pᵣ, qᵗ, constants, equil)

    elseif d.flavor isa TotalMoisture
        return saturation_total_specific_moisture(T, pᵣ, constants, equil)

    end
end

const SaturationSpecificHumidityField = Field{C, C, C, <:SaturationSpecificHumidity}
SaturationSpecificHumidityField(model, flavor_symbol=:prognostic) = Field(SaturationSpecificHumidity(model, flavor_symbol))

#####
##### Relative Humidity
#####

struct RelativeHumidityKernelFunction{μ, M, MF, T, R, TH}
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: T
    reference_state :: R
    thermodynamic_constants :: TH
end

prettysummary(kf::RelativeHumidityKernelFunction) = "RelativeHumidityKernelFunction"

Adapt.adapt_structure(to, k::RelativeHumidityKernelFunction) =
    RelativeHumidityKernelFunction(adapt(to, k.microphysics),
                                   adapt(to, k.microphysical_fields),
                                   adapt(to, k.specific_moisture),
                                   adapt(to, k.temperature),
                                   adapt(to, k.reference_state),
                                   adapt(to, k.thermodynamic_constants))

const RelativeHumidityOp = KernelFunctionOperation{C, C, C, <:Any, <:Any, <:RelativeHumidityKernelFunction}

"""
$(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` representing the *relative humidity* ``ℋ``,
defined as the ratio of vapor pressure to saturation vapor pressure:

```math
ℋ = \\frac{pᵛ}{pᵛ⁺}
```

where ``pᵛ`` is the vapor pressure (partial pressure of water vapor) computed from
the ideal gas law ``pᵛ = ρ qᵛ Rᵛ T``, and ``pᵛ⁺`` is the saturation vapor pressure.

For unsaturated conditions, ``ℋ < 1``. For saturated conditions with saturation
adjustment microphysics, ``ℋ = 1`` (or very close to it due to numerical precision).

## Examples

```jldoctest rh
using Breeze
grid = RectilinearGrid(size=(1, 1, 128), extent=(1e3, 1e3, 1e3))
microphysics = SaturationAdjustment()
model = AtmosphereModel(grid; microphysics)
set!(model, θ=300, qᵗ=0.005)  # subsaturated
ℋ = RelativeHumidity(model)

# output
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×128 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: RelativeHumidityKernelFunction
└── arguments: ()
```

As with other diagnostics, `RelativeHumidity` may be wrapped in `Field` to store the result:

```jldoctest rh
ℋ_field = RelativeHumidity(model) |> Field

# output
1×1×128 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×128 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×134 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:131) with eltype Float64 with indices 0:2×0:2×-2:131
    └── max=0.2296, min=0.145879, mean=0.184014
```

We also provide a convenience constructor for the Field:

```jldoctest rh
ℋ_field = RelativeHumidityField(model)

# output
1×1×128 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×128 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×134 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:131) with eltype Float64 with indices 0:2×0:2×-2:131
    └── max=0.2296, min=0.145879, mean=0.184014
```
"""
function RelativeHumidity(model)
    microphysics = if model.microphysics isa SaturationAdjustment
        model.microphysics
    else
        SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    end

    func = RelativeHumidityKernelFunction(microphysics,
                                          model.microphysical_fields,
                                          model.specific_moisture,
                                          model.temperature,
                                          model.formulation.reference_state,
                                          model.thermodynamic_constants)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

const AdjustmentRH = RelativeHumidityKernelFunction{<:SaturationAdjustment}

function (d::AdjustmentRH)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        T = d.temperature[i, j, k]
        qᵗ = d.specific_moisture[i, j, k]
    end

    constants = d.thermodynamic_constants
    equil = d.microphysics.equilibrium

    # Compute moisture fractions (vapor, liquid, ice)
    q = compute_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)
    
    # Vapor specific humidity
    qᵛ = q.vapor

    # Compute actual density from equation of state
    ρ = density(pᵣ, T, q, constants)

    # Vapor pressure from ideal gas law: pᵛ = ρᵛ Rᵛ T = ρ qᵛ Rᵛ T
    Rᵛ = vapor_gas_constant(constants)
    pᵛ = ρ * qᵛ * Rᵛ * T

    # Saturation vapor pressure
    surface = equilibrated_surface(equil, T)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)

    # Relative humidity ℋ = pᵛ / pᵛ⁺
    return pᵛ / max(pᵛ⁺, eps(typeof(pᵛ⁺)))
end

const RelativeHumidityField = Field{C, C, C, <:RelativeHumidityOp}
RelativeHumidityField(model) = Field(RelativeHumidity(model))
