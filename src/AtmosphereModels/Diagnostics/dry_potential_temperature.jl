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

