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

```jldoctest
using Breeze

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1e3))
model = AtmosphereModel(grid)
set!(model, θ=300, qᵗ=0.01)

θᵥ = VirtualPotentialTemperature(model)
Field(θᵥ)

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

