using Adapt: Adapt, adapt

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, Center
using Oceananigans.Utils: Utils

using Breeze.Thermodynamics:
    saturation_specific_humidity,
    vapor_gas_constant,
    density,
    saturation_vapor_pressure,
    equilibrated_surface

# Import diagnostics from AtmosphereModels.Diagnostics
using ..AtmosphereModels.Diagnostics:
    Diagnostics,
    SaturationSpecificHumidity,
    SaturationSpecificHumidityField,
    DewpointTemperature,
    DewpointTemperatureField,
    microphysics_phase_equilibrium

# Extend microphysics_phase_equilibrium for SaturationAdjustment
@inline Diagnostics.microphysics_phase_equilibrium(μ::SaturationAdjustment) = μ.equilibrium

const C = Center

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

Utils.prettysummary(::RelativeHumidityKernelFunction) = "RelativeHumidityKernelFunction"

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
the ideal gas law
```math
pᵛ = ρ qᵛ Rᵛ T
```
and ``pᵛ⁺`` is the saturation vapor pressure.

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
    └── max=0.214947, min=0.136946, mean=0.172492
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
    └── max=0.214947, min=0.136946, mean=0.172492
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
                                          model.dynamics.reference_state,
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
    equil = microphysics_phase_equilibrium(d.microphysics)

    # Compute moisture fractions (vapor, liquid, ice)
    q = grid_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵗ, d.microphysical_fields)

    # Vapor specific humidity
    qᵛ = q.vapor

    # Compute actual density from equation of state
    ρ = density(T, pᵣ, q, constants)

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
