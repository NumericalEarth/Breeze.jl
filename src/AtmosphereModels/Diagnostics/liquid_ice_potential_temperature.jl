#####
##### Liquid-ice potential temperature
#####

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
Field(θₗᵢ)

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

