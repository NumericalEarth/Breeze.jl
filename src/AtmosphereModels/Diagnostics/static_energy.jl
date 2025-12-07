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

