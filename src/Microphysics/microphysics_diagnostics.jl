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
    specific_prognostic_moisture :: MF
    temperature :: T
    reference_state :: R
    thermodynamic_constants :: TH
end

Utils.prettysummary(::RelativeHumidityKernelFunction) = "RelativeHumidityKernelFunction"

Adapt.adapt_structure(to, k::RelativeHumidityKernelFunction) =
    RelativeHumidityKernelFunction(adapt(to, k.microphysics),
                                   adapt(to, k.microphysical_fields),
                                   adapt(to, k.specific_prognostic_moisture),
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
    └── max=0.214949, min=0.137169, mean=0.172626
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
    └── max=0.214949, min=0.137169, mean=0.172626
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
                                          specific_prognostic_moisture(model),
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
        # qᵛᵉ: vapor (non-equilibrium) or equilibrium moisture (saturation adjustment)
        qᵛᵉ = d.specific_prognostic_moisture[i, j, k]
    end

    constants = d.thermodynamic_constants
    equil = microphysics_phase_equilibrium(d.microphysics)

    # Compute moisture fractions (vapor, liquid, ice)
    q = grid_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qᵛᵉ, d.microphysical_fields)

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

#####
##### Number Concentration
#####

"""
    NumberConcentrationKernelFunction{P, M, Q, R}

Kernel callable for the lazy total number concentration ``ρnˣ`` (m⁻³) of a
one-moment microphysics species, computed from the prognostic mass density
``ρqˣ`` and the species' assumed Marshall–Palmer size distribution as
``ρnˣ = n_0 \\, λ^{-1}``.

# Fields
- `pdf`: size distribution (`ParticlePDFIceRain` or `ParticlePDFSnow`)
- `mass`: mass(radius) parameters (`ParticleMass`)
- `ρq`: prognostic mass density field for the species [kg/m³]
- `reference_density`: air density field [kg/m³]
"""
struct NumberConcentrationKernelFunction{P, M, Q, R}
    pdf :: P
    mass :: M
    ρq :: Q
    reference_density :: R
end

Utils.prettysummary(::NumberConcentrationKernelFunction) = "NumberConcentrationKernelFunction"

Adapt.adapt_structure(to, k::NumberConcentrationKernelFunction) =
    NumberConcentrationKernelFunction(adapt(to, k.pdf),
                                      adapt(to, k.mass),
                                      adapt(to, k.ρq),
                                      adapt(to, k.reference_density))

const NumberConcentrationOp = KernelFunctionOperation{C, C, C, <:Any, <:Any, <:NumberConcentrationKernelFunction}

"""
$(TYPEDSIGNATURES)

Lazy diagnostic returning the total number concentration ``ρnˣ`` (m⁻³) for the
requested `species`.

For `OneMomentCloudMicrophysics`, `species ∈ (:rain, :snow)` returns a
`KernelFunctionOperation` that computes ``n_0 \\, λ^{-1}`` from the prognostic
``ρqˣ`` and the scheme's size distribution. Snow's intercept ``n_0`` depends on
``(q, ρ)`` per [Kaul et al. (2015)](@cite Kaul2015) — so this diagnostic stays
consistent with the scheme's actual DSD without re-encoding species-specific
physics at every call site.

For `TwoMomentCloudMicrophysics`, returns the prognostic ``ρnˣ`` field directly
(e.g., `:rain` → `ρnʳ`, `:cloud_liquid` → `ρnᶜˡ`).

Returns `nothing` if the species is not carried by the model (e.g., `:hail` for
a 1-mom scheme without hail). Errors for microphysics schemes that do not
define a DSD-based number concentration (e.g., `SaturationAdjustment`).

See [`NumberConcentrationField`](@ref) for a convenience constructor that
wraps the result in a `Field`.
"""
NumberConcentration(model, species::Symbol) =
    number_concentration(model, model.microphysics, Val(species))

# Default fallback: unsupported microphysics scheme.
number_concentration(model, microphysics, ::Val{species}) where {species} =
    error("NumberConcentration is not defined for microphysics scheme of type ",
          typeof(microphysics), " (species = :", species, "). ",
          "Supported schemes: OneMomentCloudMicrophysics (species ∈ (:rain, :snow)) ",
          "and TwoMomentCloudMicrophysics.")

"""
$(TYPEDSIGNATURES)

Convenience constructor that wraps [`NumberConcentration`](@ref) in a `Field`.
Returns `nothing` when the requested species is not carried by the model.
"""
function NumberConcentrationField(model, species::Symbol)
    op = NumberConcentration(model, species)
    op === nothing && return nothing
    return Field(op)
end
