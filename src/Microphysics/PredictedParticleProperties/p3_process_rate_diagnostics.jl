using Oceananigans.Fields: Field

#####
##### Process-rate diagnostics
#####
#
# `MicrophysicalProcessRate(model, name)` for P3 evaluates `compute_p3_process_rates` at every
# cell from the same inputs the tendency kernel assembles (`p3_grid_process_inputs`), and
# returns one field (or the sum of several fields) of the resulting `P3ProcessRates`. The rate
# names are a type parameter of the kernel function, so the field access is resolved at
# compile time.

"""
$(TYPEDEF)

Kernel function for the sum of the P3 process rates `names` (a tuple of fields of
`P3ProcessRates`), evaluated at the current model state. See [`MicrophysicalProcessRate`](@ref Breeze.AtmosphereModels.MicrophysicalProcessRate).
"""
struct P3ProcessRateKernelFunction{names, P, M, F, D, C, R, U}
    microphysics :: P
    microphysical_fields :: M
    formulation :: F
    dynamics :: D
    thermodynamic_constants :: C
    density :: R
    velocities :: U
end

function P3ProcessRateKernelFunction{names}(microphysics, microphysical_fields, formulation, dynamics,
                                           thermodynamic_constants, density, velocities) where names
    P, M, F = typeof(microphysics), typeof(microphysical_fields), typeof(formulation)
    D, C = typeof(dynamics), typeof(thermodynamic_constants)
    R, U = typeof(density), typeof(velocities)
    return P3ProcessRateKernelFunction{names, P, M, F, D, C, R, U}(microphysics, microphysical_fields,
                                                                  formulation, dynamics,
                                                                  thermodynamic_constants,
                                                                  density, velocities)
end

Utils.prettysummary(::P3ProcessRateKernelFunction{names}) where names =
    string("P3ProcessRateKernelFunction", names)

Adapt.adapt_structure(to, k::P3ProcessRateKernelFunction{names}) where names =
    P3ProcessRateKernelFunction{names}(Adapt.adapt(to, k.microphysics),
                                      Adapt.adapt(to, k.microphysical_fields),
                                      Adapt.adapt(to, k.formulation),
                                      Adapt.adapt(to, k.dynamics),
                                      Adapt.adapt(to, k.thermodynamic_constants),
                                      Adapt.adapt(to, k.density),
                                      Adapt.adapt(to, k.velocities))

@inline function (kernel::P3ProcessRateKernelFunction{names})(i, j, k, grid) where names
    p3 = kernel.microphysics
    constants = kernel.thermodynamic_constants
    inputs = p3_grid_process_inputs(i, j, k, grid, p3, kernel.microphysical_fields,
                                    kernel.formulation, kernel.dynamics, constants,
                                    kernel.density, kernel.velocities)

    rates = compute_p3_process_rates(p3, inputs.ρ, inputs.ℳ, inputs.𝒰, constants,
                                     inputs.properties, inputs.surface_temperature,
                                     inputs.temperature_tendency, inputs.pressure_tendency)

    return sum_process_rates(rates, names)
end

@inline sum_process_rates(rates, names::Tuple{Symbol}) = getfield(rates, first(names))
@inline sum_process_rates(rates, names::Tuple) =
    getfield(rates, first(names)) + sum_process_rates(rates, Base.tail(names))

function P3ProcessRate(model, p3, names)
    for name in names
        name ∈ fieldnames(P3ProcessRates) ||
            throw(ArgumentError(string(":", name, " is not a P3 process rate. Available rates are :",
                                       join(fieldnames(P3ProcessRates), ", :"), ".")))
    end

    kernel = P3ProcessRateKernelFunction{names}(p3, model.microphysical_fields,
                                                model.formulation, model.dynamics,
                                                model.thermodynamic_constants,
                                                AM.total_density(model.dynamics),
                                                model.velocities)

    return KernelFunctionOperation{Center, Center, Center}(kernel, model.grid)
end

"""
$(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` for the P3 process rate `name`, which may be any field of
`P3ProcessRates`. The most commonly visualized are

| `name`              | process                                          | units        | sign           |
|:--------------------|:-------------------------------------------------|:-------------|:---------------|
| `:condensation`     | net vapor ↔ cloud liquid (condensation − evaporation) | kg kg⁻¹ s⁻¹ | + condensation |
| `:autoconversion`   | cloud liquid → rain by collision–coalescence     | kg kg⁻¹ s⁻¹  | ≥ 0            |
| `:accretion`        | cloud liquid → rain, collected by rain           | kg kg⁻¹ s⁻¹  | ≥ 0            |
| `:rain_evaporation` | rain → vapor                                     | kg kg⁻¹ s⁻¹  | ≥ 0            |
| `:rain_condensation`| vapor → rain                                     | kg kg⁻¹ s⁻¹  | ≥ 0            |
| `:deposition`       | net vapor ↔ ice (deposition − sublimation)       | kg kg⁻¹ s⁻¹  | + deposition   |
| `:cloud_riming`, `:rain_riming` | liquid → ice by riming               | kg kg⁻¹ s⁻¹  | ≥ 0            |

Number rates (for example `:rain_self_collection`) are in kg⁻¹ s⁻¹.

The rates are the instantaneous, sink-limited rates that P3 evaluates from the current model
state to form its tendencies: P3 has no operator-split or substepped update, so these are
exactly the rates the time stepper integrates at the start of the next stage. As in the
tendency evaluation, the diffusional-growth driver uses the adiabatic temperature and
pressure tendency of the local vertical velocity, and the surface temperature that gates
rime splintering is the one stored by the most recent tendency evaluation.
"""
AM.MicrophysicalProcessRate(model, p3::P3, ::Val{name}) where name = P3ProcessRate(model, p3, (name,))

"""
$(TYPEDSIGNATURES)

Return a `Field` of the P3 liquid precipitation production rate, autoconversion plus
accretion (kg kg⁻¹ s⁻¹), consistently with the one- and two-moment schemes. This is the
rate at which cloud liquid becomes rain, not a flux; see
[`precipitation_flux`](@ref Breeze.AtmosphereModels.precipitation_flux) for the rate at which
precipitation falls through a level.
"""
AM.precipitation_rate(model, p3::P3, ::Val{:liquid}) = Field(P3ProcessRate(model, p3, (:autoconversion, :accretion)))
AM.precipitation_rate(model, ::P3, ::Val{:ice}) = nothing
