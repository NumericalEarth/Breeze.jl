# Imports are provided by the Diagnostics module

struct DewpointTemperatureKernelFunction{μ, M, MF, T, R, TH, FT}
    microphysics :: μ
    microphysical_fields :: M
    specific_moisture :: MF
    temperature :: T
    reference_state :: R
    thermodynamic_constants :: TH
    tolerance :: FT
    maxiter :: Int
end

Oceananigans.Utils.prettysummary(kf::DewpointTemperatureKernelFunction) = "DewpointTemperatureKernelFunction"

Adapt.adapt_structure(to, k::DewpointTemperatureKernelFunction) =
    DewpointTemperatureKernelFunction(adapt(to, k.microphysics),
                                      adapt(to, k.microphysical_fields),
                                      adapt(to, k.specific_moisture),
                                      adapt(to, k.temperature),
                                      adapt(to, k.reference_state),
                                      adapt(to, k.thermodynamic_constants),
                                      k.tolerance,
                                      k.maxiter)

const DewpointTemperature = KernelFunctionOperation{C, C, C, <:Any, <:Any, <:DewpointTemperatureKernelFunction}

"""
$(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` representing the dewpoint temperature ``T⁺``.

The dewpoint temperature is the temperature at which the air would become saturated
at its current vapor pressure. It is computed by solving the implicit equation:

```math
pᵛ⁺(T⁺) = pᵛ
```

using secant iteration, where ``pᵛ`` is the actual vapor pressure and ``pᵛ⁺``
is the saturation vapor pressure.

For saturated air, the dewpoint temperature equals the actual temperature.

The keyword arguments `tolerance` (default `1e-4`) and `maxiter` (default `10`) control
the secant iteration convergence.

# Example

```jldoctest dewpoint
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid; microphysics=SaturationAdjustment())
set!(model, θ=300, qᵗ=0.01)

T⁺ = DewpointTemperature(model)

# output
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: DewpointTemperatureKernelFunction
└── arguments: ()
```

The result may be wrapped in a `Field` to store the computed values:

```jldoctest dewpoint
T⁺_field = Field(T⁺)

# output
1×1×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×14 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:11) with eltype Float64 with indices 0:2×0:2×-2:11
    └── max=289.062, min=287.475, mean=288.27
```
"""
function DewpointTemperature(model; tolerance=1e-4, maxiter=10)
    func = DewpointTemperatureKernelFunction(model.microphysics,
                                             model.microphysical_fields,
                                             specific_prognostic_moisture(model),
                                             model.temperature,
                                             model.dynamics.reference_state,
                                             model.thermodynamic_constants,
                                             tolerance,
                                             maxiter)

    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

#####
##### Kernel function implementation
#####

function (d::DewpointTemperatureKernelFunction)(i, j, k, grid)
    @inbounds begin
        pᵣ = d.reference_state.pressure[i, j, k]
        ρᵣ = d.reference_state.density[i, j, k]
        qₘ = d.specific_moisture[i, j, k]
        T = d.temperature[i, j, k]
    end

    constants = d.thermodynamic_constants
    equilibrium = microphysics_phase_equilibrium(d.microphysics)
    surface = equilibrated_surface(equilibrium, T)

    # Get vapor specific humidity from microphysics partition
    q = grid_moisture_fractions(i, j, k, grid, d.microphysics, ρᵣ, qₘ, d.microphysical_fields)
    qᵛ = q.vapor

    # Compute density and vapor pressure
    ρ = Thermodynamics.density(T, pᵣ, q, constants)
    pᵛ = Thermodynamics.vapor_pressure(T, ρ, qᵛ, constants)

    # Compute dewpoint temperature
    return Thermodynamics.dewpoint_temperature(pᵛ, T, constants, surface;
                                               tolerance = d.tolerance,
                                               maxiter = d.maxiter)
end

const DewpointTemperatureField = Field{C, C, C, <:DewpointTemperature}
DewpointTemperatureField(model; kw...) = Field(DewpointTemperature(model; kw...))
