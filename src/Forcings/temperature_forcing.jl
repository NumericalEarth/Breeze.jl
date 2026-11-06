using Breeze.AtmosphereModels: AtmosphereModels, materialize_atmosphere_model_forcing, grid_moisture_fractions
using Breeze.Thermodynamics: mixture_heat_capacity
using Oceananigans.Utils: prettysummary
using Adapt: Adapt

#####
##### TemperatureForcing
#####

struct TemperatureForcing{F, M, MF, Q, D, C}
    temperature_tendency :: F
    microphysics :: M
    microphysical_fields :: MF
    specific_prognostic_moisture :: Q
    density :: D
    thermodynamic_constants :: C
end

"""
$(TYPEDSIGNATURES)

Forcing by a prescribed temperature tendency ``∂ₜT`` (K s⁻¹), such as the large-scale heating
of a single-column case. Supply it under the specific energy key `E`, as in
`forcing = (; E = TemperatureForcing(∂ₜT))`. `∂ₜT` can be anything accepted as a forcing: a
`Field`, a function, a `Returns`, an `Oceananigans.Forcing`. At kernel time it is converted to an
energy tendency with the *local* mixture heat capacity,

```math
F_E = cᵖᵐ ∂ₜT ,
```

where ``cᵖᵐ`` is evaluated from the moisture mass fractions the tendency kernels see (vapor,
liquid and ice from the microphysics scheme). Breeze then multiplies by the density and routes the
result onto the thermodynamic variable:
[`LiquidIcePotentialTemperatureFormulation`](@ref Breeze.PotentialTemperatureFormulations.LiquidIcePotentialTemperatureFormulation)
divides an energy tendency by ``cᵖᵐ Π``, so ``θ`` changes at ``∂ₜT / Π``, the rate at which
``T = Π θ + (ℒˡᵣ qˡ + ℒⁱᵣ qⁱ) / cᵖᵐ`` changes at fixed moisture and pressure by ``∂ₜT``.
[`StaticEnergyFormulation`](@ref Breeze.StaticEnergyFormulations.StaticEnergyFormulation) adds
it to ``ρs`` unconverted, and ``s = cᵖᵐ T + g z - ℒˡᵣ qˡ - ℒⁱᵣ qⁱ`` changes by ``cᵖᵐ ∂ₜT``.

Converting with the dry heat capacity ``cᵖᵈ`` instead, as in `E = cᵖᵈ * ∂ₜT`, heats humid air
too little by the factor ``cᵖᵈ / cᵖᵐ``: about 1.7% at 20 g kg⁻¹ of vapor.

```jldoctest
using Breeze

grid = RectilinearGrid(size=4, z=(0, 1000), topology=(Flat, Flat, Bounded))
cooling(z, t) = -2 / 86400 # radiative cooling, K s⁻¹
forcing = TemperatureForcing(cooling)
model = AtmosphereModel(grid; forcing=(; E=forcing))
forcing

# output
TemperatureForcing with temperature_tendency: cooling (generic function with 1 method)
```
"""
TemperatureForcing(temperature_tendency) = TemperatureForcing(temperature_tendency, nothing, nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, forcing::TemperatureForcing) =
    TemperatureForcing(Adapt.adapt(to, forcing.temperature_tendency),
                       Adapt.adapt(to, forcing.microphysics),
                       Adapt.adapt(to, forcing.microphysical_fields),
                       Adapt.adapt(to, forcing.specific_prognostic_moisture),
                       Adapt.adapt(to, forcing.density),
                       Adapt.adapt(to, forcing.thermodynamic_constants))

Base.summary(forcing::TemperatureForcing) =
    string("TemperatureForcing with temperature_tendency: ", prettysummary(forcing.temperature_tendency))

Base.show(io::IO, forcing::TemperatureForcing) = print(io, summary(forcing))

#####
##### Kernel: the specific energy tendency cᵖᵐ ∂ₜT (the ρ-multiply happens in SpecificForcing)
#####

# The moisture mass fractions are those of `grid_thermodynamic_state` in the tendency kernels:
# total density, prognostic moisture and microphysical fields at (i, j, k). The cᵖᵐ here and the
# kernel's division by cᵖᵐ Π therefore cancel to rounding.
@inline function (forcing::TemperatureForcing)(i, j, k, grid, clock, fields)
    ∂ₜT = forcing.temperature_tendency(i, j, k, grid, clock, fields)
    @inbounds ρ = forcing.density[i, j, k]
    @inbounds qᵛᵉ = forcing.specific_prognostic_moisture[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, forcing.microphysics, ρ, qᵛᵉ, forcing.microphysical_fields)
    cᵖᵐ = mixture_heat_capacity(q, forcing.thermodynamic_constants)
    return cᵖᵐ * ∂ₜT
end

#####
##### Materialization: capture the moisture state and the thermodynamic constants
#####

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::TemperatureForcing,
                                                               field, name, model_field_names,
                                                               context::NamedTuple)
    if name !== :E
        msg = string("TemperatureForcing returns the specific energy tendency cᵖᵐ ∂ₜT and ",
                     "must be supplied under the specific energy key `E`, not `", name, "`. ",
                     "Breeze applies the density factor ρ automatically via SpecificForcing and ",
                     "converts the energy tendency for the thermodynamic variable.")
        throw(ArgumentError(msg))
    end

    temperature_tendency = materialize_atmosphere_model_forcing(forcing.temperature_tendency, field, name,
                                                                model_field_names, context)

    return TemperatureForcing(temperature_tendency,
                              context.microphysics,
                              context.microphysical_fields,
                              context.specific_prognostic_moisture,
                              context.total_density,
                              context.thermodynamic_constants)
end

AtmosphereModels.compute_forcing!(forcing::TemperatureForcing) = AtmosphereModels.compute_forcing!(forcing.temperature_tendency)
