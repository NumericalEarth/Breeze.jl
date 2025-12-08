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

"""
    StaticEnergy(model, flavor=:specific)

Return a `KernelFunctionOperation` representing moist static energy ``e``.

Moist static energy is a conserved quantity in adiabatic, frictionless flow that
combines sensible heat, gravitational potential energy, and latent heat:

```math
e = cᵖᵐ T + g z - ℒˡᵣ qˡ - ℒⁱᵣ qⁱ
```

where ``cᵖᵐ`` is the moist air heat capacity, ``T`` is temperature,
``g`` is gravitational acceleration, ``z`` is height, and
``ℒˡᵣ qˡ + ℒⁱᵣ qⁱ`` is the latent heat content of condensate.

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
Field(e)

# output
1×1×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 3×3×14 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, -2:11) with eltype Float64 with indices 0:2×0:2×-2:11
    └── max=3.01883e5, min=3.01526e5, mean=3.01704e5
```
"""
function StaticEnergy(model, flavor_symbol=:specific)

    flavor = if flavor_symbol === :specific
        Specific()
    elseif flavor_symbol === :density
        Density()
    else
        msg = "`flavor` must be :specific or :density, received :$flavor_symbol"
        throw(ArgumentError(msg))
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

