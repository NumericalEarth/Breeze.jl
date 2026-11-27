#####
##### Static stability: how `TKEBasedTurbulenceClosure` diagnoses N² at the cell interfaces.
#####
##### `DryStaticStability` is the gradient ∂z_b of the buoyancy of the dynamics — condensate
##### loading included, latent heating excluded. `MoistStaticStability` switches, where the air
##### is saturated, to the static stability of a saturated displacement (Durran & Klemp 1982)
##### in the model's thermodynamics, in which the parcel condenses as it rises and its latent
##### heating offsets part of the environmental stratification. `SaturationWeightedStaticStability` blends the two with a weight that
##### an external subgrid cloud model supplies. All are evaluated once per stage into `closure_fields.N²`.
#####

"""
$(TYPEDEF)

The static stability of [`TKEBasedTurbulenceClosure`](@ref) as the vertical gradient of the
buoyancy of the dynamics, ``N² = ∂_z b = g ∂_z \\ln θᵨ`` with ``θᵨ`` the density potential
temperature: the stratification a parcel feels when displaced without phase change. Condensate
loading is included; the latent heating of a saturated displacement is not. The default is
[`MoistStaticStability`](@ref), which reduces to this where the air is subsaturated.
"""
struct DryStaticStability end

Base.summary(::DryStaticStability) = "DryStaticStability"
Base.show(io::IO, ss::DryStaticStability) = print(io, summary(ss))

@inline static_stabilityᶜᶜᶠ(i, j, k, grid, ::DryStaticStability, buoyancy, tracers) =
    ∂z_b(i, j, k, grid, buoyancy, tracers)

"""
$(TYPEDEF)

Static stability using the dry buoyancy gradient in subsaturated air and otherwise the static stability of a
saturated displacement [as Durran and Klemp (1982)](@cite DurranKlemp1982) define it, in the model's thermodynamics
(`saturated_static_stabilityᶜᶜᶠ`).
The saturation test excludes precipitating water, while its density loading remains in the
buoyancy frequency. Latent heat follows the microphysics' phase equilibrium. This is the
default for [`TKEBasedTurbulenceClosure`](@ref) without a subgrid distribution.
"""
struct MoistStaticStability end

Base.summary(::MoistStaticStability) = "MoistStaticStability"
Base.show(io::IO, ss::MoistStaticStability) = print(io, summary(ss))

@inline function static_stabilityᶜᶜᶠ(i, j, k, grid, ::MoistStaticStability, buoyancy, tracers)
    N²ᵈ = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
    saturated = saturatedᶜᶜᶠ(i, j, k, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
    return ifelse(saturated, N²⁺, N²ᵈ)
end

"""
$(TYPEDEF)

Static stability as a blend of the dry buoyancy gradient ``N²ᵈ`` of [`DryStaticStability`](@ref)
and the static stability ``N²⁺`` of a saturated displacement (see [`MoistStaticStability`](@ref)),

```math
N² = (1 - aᴺ) N²ᵈ + aᴺ N²⁺,
```

with the weight ``aᴺ``, the effective saturated fraction — the fraction of a displacement's change
in saturation deficit that condenses — from a subgrid cloud model: ``aᴺ = 0`` is unsaturated air,
``aᴺ = 1`` saturated air, and a model whose ``aᴺ`` varies smoothly through saturation removes the jump
that [`MoistStaticStability`](@ref) makes where the grid-mean air saturates. This is the
moist-buoyancy weighting of [Cuijpers and Duynkerke (1993)](@cite CuijpersDuynkerke1993).

This slot is the only owner of ``N²``: every other component reads `closure_fields.N²`. An external
`fraction` supplies ``aᴺ`` by extending
[`saturated_stability_weightᶜᶜᶠ`](@ref Breeze.TurbulenceClosures.saturated_stability_weightᶜᶜᶠ);
without a method for the model, computing ``N²`` throws a `MethodError`. The closure's own subgrid
distribution is the cloud model [`CondensateFluxFraction`](@ref), ``aᴺ = η aʷ``, which depends on
the diffusivities and so is solved together with them in the closure update.
`SaturationWeightedStaticStability()`, without a model, takes the subgrid model of the microphysics
([`MicrophysicsSaturatedFraction`](@ref Breeze.TurbulenceClosures.MicrophysicsSaturatedFraction)), so that
the condensation and the turbulence see one subgrid cloud with one set of coefficients.
``N²`` is stored in `closure_fields.N²`, which the mixing length, the Richardson number, the
buoyancy flux and the implicit sinks of the TKE equation all read.

```jldoctest
using Breeze

struct HalfSaturated end

closure = TKEBasedTurbulenceClosure(static_stability = SaturationWeightedStaticStability(HalfSaturated()))
closure.static_stability

# output
SaturationWeightedStaticStability(HalfSaturated())
```

Fields
======

$(TYPEDFIELDS)
"""
struct SaturationWeightedStaticStability{M}
    "the subgrid cloud model that supplies the weight ``aᴺ`` (effective saturated fraction) of the saturated static stability"
    fraction :: M
end

"""
$(TYPEDEF)

The `fraction` of a [`SaturationWeightedStaticStability`](@ref) constructed without one: the weight
is that of the subgrid model the microphysics holds, found by
[`microphysics_subgrid_model`](@ref Breeze.TurbulenceClosures.microphysics_subgrid_model).

When that model is a source of supersaturation statistics (an
`AtmosphereModels.AbstractSupersaturationStatistics`, as in
`SubgridCondensation(statistics)` of P3), the weight is its saturated fraction ``f⁺``,
`AtmosphereModels.saturated_fraction(supersaturation_statistics(statistics, ...))`, the
probability that the vapor excess is positive, averaged from the two adjacent centers to the
interface. For `TurbulenceClosureStatistics` it is the cloud fraction of the closure's
distribution, of the previous update. Another model supplies the weight through
[`saturated_stability_weightᶜᶜᶠ`](@ref Breeze.TurbulenceClosures.saturated_stability_weightᶜᶜᶠ).
"""
struct MicrophysicsSaturatedFraction end

SaturationWeightedStaticStability() = SaturationWeightedStaticStability(MicrophysicsSaturatedFraction())

Base.show(io::IO, ::MicrophysicsSaturatedFraction) = print(io, "MicrophysicsSaturatedFraction()")

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The subgrid model that `microphysics` holds, whose weight
[`SaturationWeightedStaticStability`](@ref)`()` uses: for P3 with
`SubgridCondensation(statistics)`, its `statistics`. `nothing` for a microphysics scheme without
one (including P3 with `GridMeanCondensation()`), for which there is no weight.
"""
@inline microphysics_subgrid_model(microphysics) = nothing
@inline microphysics_subgrid_model(p3::PredictedParticlePropertiesMicrophysics) = condensation_statistics(p3.condensation)
@inline condensation_statistics(condensation) = nothing
@inline condensation_statistics(condensation::SubgridCondensation) = condensation.statistics

# The cloud model may hold traced coefficients when the closure is differentiated.
Adapt.adapt_structure(to, ss::SaturationWeightedStaticStability) = SaturationWeightedStaticStability(adapt(to, ss.fraction))

Base.summary(::SaturationWeightedStaticStability) = "SaturationWeightedStaticStability"
Base.show(io::IO, ss::SaturationWeightedStaticStability) = print(io, summary(ss), "(", ss.fraction, ")")

"""
The weight ``aᴺ`` (the effective saturated fraction) of the saturated static stability in [`SaturationWeightedStaticStability`](@ref) at the
cell interface `(i, j, k)`, (Center, Center, Face). A subgrid cloud model extends this function for
its own type, with the arguments `(i, j, k, grid, fraction, buoyancy, tracers, closure_fields)`.
`buoyancy` is the
`AtmosphereModelBuoyancy` of the model (its dynamics, formulation, thermodynamic constants,
microphysics and microphysical fields), `tracers` holds the temperature `T`, the prognostic
moisture `qᵛᵉ` and the specific prognostic tracers, and `closure_fields` are the
[`TKEClosureFields`](@ref Breeze.TurbulenceClosures.TKEClosureFields).

The static stability is the first field computed in an update, so `closure_fields.ℓᵐ` and the
diffusivities hold the values of the previous update (zero before the first): a weight that reads
the mixing length reads a lagged one. This is what keeps the dependence explicit, since the mixing
length itself is computed from ``N²``.

``aᴺ`` should lie in ``[0, 1]`` and must be finite; it is not clipped.
"""
function saturated_stability_weightᶜᶜᶠ end

@inline saturated_stability_weightᶜᶜᶠ(i, j, k, grid, ::MicrophysicsSaturatedFraction, buoyancy, tracers, closure_fields) =
    saturated_stability_weightᶜᶜᶠ(i, j, k, grid, microphysics_subgrid_model(buoyancy.microphysics), buoyancy, tracers, closure_fields)

#####
##### The saturated fraction of the microphysics' supersaturation statistics
#####
#
# When the subgrid model of the microphysics is a source of supersaturation statistics
# (`AtmosphereModels.AbstractSupersaturationStatistics`), `MicrophysicsSaturatedFraction` takes the
# weight from the statistics P3 condenses with: aᴺ is the saturated fraction f⁺ of
# `AtmosphereModels.saturated_fraction(supersaturation_statistics(...))`, averaged from the two
# adjacent centers to the interface. The statistics need the closure and the model fields, which the
# static stability does not receive; they are gathered when the static stability is launched
# (`saturated_fraction_model_fields`) and attached to the fraction in the kernel
# (`with_saturated_fraction_fields`). A statistics source of the closure's own distribution reads
# the distribution of the previous update, as a weight that reads the mixing length does.

# The fraction attached to the arguments of `supersaturation_statistics`
struct StatisticsSaturatedFraction{S, F}
    statistics :: S
    fields :: F
end

Adapt.adapt_structure(to, f::StatisticsSaturatedFraction) =
    StatisticsSaturatedFraction(adapt(to, f.statistics), adapt(to, f.fields))

const MicrophysicsSaturatedFractionStability = SaturationWeightedStaticStability{MicrophysicsSaturatedFraction}

# The model fields of `supersaturation_statistics` for the subgrid model of the microphysics, where the static stability
# reads them, or `nothing`. The statistics read the closure and its fields too, which the kernel of the static stability
# receives itself (`with_saturated_fraction_fields`): each reaches the kernel once.
saturated_fraction_model_fields(static_stability, model) = nothing
saturated_fraction_model_fields(::MicrophysicsSaturatedFractionStability, model) =
    statistics_model_fields(microphysics_subgrid_model(model.microphysics), model)
statistics_model_fields(subgrid_model, model) = nothing
statistics_model_fields(::AtmosphereModels.AbstractSupersaturationStatistics, model) =
    AtmosphereModels.supersaturation_statistics_model_fields(model)

@inline with_saturated_fraction_fields(static_stability, buoyancy, closure, closure_fields, model_fields) = static_stability
@inline with_saturated_fraction_fields(static_stability::MicrophysicsSaturatedFractionStability, buoyancy, closure, closure_fields,
                                       ::Nothing) = static_stability

@inline with_saturated_fraction_fields(::MicrophysicsSaturatedFractionStability, buoyancy, closure, closure_fields, model_fields) =
    SaturationWeightedStaticStability(StatisticsSaturatedFraction(microphysics_subgrid_model(buoyancy.microphysics),
                                                                  (closure, closure_fields, model_fields)))

@inline function cell_saturated_fraction(i, j, k, grid, fraction::StatisticsSaturatedFraction)
    statistics = AtmosphereModels.supersaturation_statistics(fraction.statistics, i, j, k, grid, fraction.fields...)
    return convert(grid_float_type(grid), AtmosphereModels.saturated_fraction(statistics))
end

@inline function saturated_stability_weightᶜᶜᶠ(i, j, k, grid, fraction::StatisticsSaturatedFraction, buoyancy, tracers, closure_fields)
    Nz = size(grid, 3)
    f⁺ᵇ = cell_saturated_fraction(i, j, max(k - 1, 1), grid, fraction)
    f⁺ᵗ = cell_saturated_fraction(i, j, min(k, Nz), grid, fraction)
    return (f⁺ᵇ + f⁺ᵗ) / 2
end

@inline function static_stabilityᶜᶜᶠ(i, j, k, grid, ss::SaturationWeightedStaticStability, buoyancy, tracers, closure_fields)
    N²ᵈ = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
    aᴺ = saturated_stability_weightᶜᶜᶠ(i, j, k, grid, ss.fraction, buoyancy, tracers, closure_fields)
    return N²ᵈ + aᴺ * (N²⁺ - N²ᵈ)
end

# A static stability that reads nothing of the closure fields is defined without them
@inline static_stabilityᶜᶜᶠ(i, j, k, grid, static_stability, buoyancy, tracers, closure_fields) =
    static_stabilityᶜᶜᶠ(i, j, k, grid, static_stability, buoyancy, tracers)

#####
##### Pointwise thermodynamics for the saturated branch
#####

# Temperature, pressure, total density and moisture mass fractions at cell `(i, j, k)`
@inline function moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    dynamics = buoyancy.dynamics
    p_field = dynamics_pressure(dynamics)
    ρ_field = total_density(dynamics)

    @inbounds begin
        Tᵢ = T[i, j, k]
        p = p_field[i, j, k]
        ρ = ρ_field[i, j, k]
        qᵛᵉᵢ = qᵛᵉ[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, buoyancy.microphysics, ρ, qᵛᵉᵢ, buoyancy.microphysical_fields)
    return Tᵢ, p, ρ, q
end

# The anelastic reference density normalizes prognostics but does not give their
# vapor partial pressure. Saturation requires the thermodynamic state density ρᵗʰ.
@inline function local_saturation_specific_humidity(i, j, k, buoyancy, Tᵢ, q)
    constants = buoyancy.thermodynamic_constants
    ρᵗʰ = thermodynamic_state_density(i, j, k, buoyancy.dynamics, Tᵢ, q, constants)
    equilibrium = microphysics_phase_equilibrium(buoyancy.microphysics)
    return saturation_specific_humidity(Tᵢ, ρᵗʰ, constants, equilibrium)
end

@inline function saturation_specific_humidityᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    return local_saturation_specific_humidity(i, j, k, buoyancy, Tᵢ, q)
end

@inline function nonprecipitating_waterᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    qʳ = precipitating_mass_fraction(i, j, k, buoyancy.microphysics, buoyancy.microphysical_fields)
    return q.vapor + q.liquid + q.ice - qʳ
end

#####
##### The saturated branch at the face
#####

@inline function saturatedᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    qʷ = ℑzᵃᵃᶠ(i, j, k, grid, nonprecipitating_waterᶜᶜᶜ, buoyancy, T, qᵛᵉ)
    qᵛ⁺ = ℑzᵃᵃᶠ(i, j, k, grid, saturation_specific_humidityᶜᶜᶜ, buoyancy, T, qᵛᵉ)
    return qʷ ≥ qᵛ⁺
end

# What a saturated displacement changes at center k, from the model's thermodynamics as the energetics of the applied
# fluxes form it (`buoyancy_derivatives`, `saturation_linearization`): the buoyancy of the condensate of the distributed
# water relative to that of vapor, b_qᶜ - b_qᵛ, the coefficients (c_q, c_θ) of its saturation excess, and its water qʷ,
# liquid-ice potential temperature θʷ and condensate qᶜ. The distributed water is that of `distributed_water`: the vapor
# and the cloud liquid of a microphysics with prognostic cloud liquid, whose other condensate is held, and otherwise all
# the water, so that ĉᵖ Π θʷ = cᵖᵐ T - ℒˡ₀ (qˡ - held liquid) - ℒⁱ₀ (qⁱ - held ice).
@inline function saturated_displacementᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    constants = buoyancy.thermodynamic_constants
    microphysics = buoyancy.microphysics
    pˢᵗ = standard_pressure(buoyancy.dynamics)
    κ̂ = AtmosphereModels.liquid_ice_exner_exponent(buoyancy.formulation)
    equilibrium = condensate_phase_equilibrium(microphysics)

    ρqᶜˡ = cloud_liquid_density(buoyancy.microphysical_fields)
    qᶜ = distributed_condensate(ρqᶜˡ, i, j, k, ρ, q)
    qʷ = @inbounds qᵛᵉ[i, j, k] + distributed_cloud_liquid(ρqᶜˡ, i, j, k, ρ)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    Π = (p / pˢᵗ)^exner_exponent(κ̂, q, constants)
    cᵖᵐT = mixture_heat_capacity(q, constants) * Tᵢ
    θˡⁱ = (cᵖᵐT - ℒˡ₀ * q.liquid - ℒⁱ₀ * q.ice) / (ĉᵖ * Π)
    θʷ = distributed_water_potential_temperature(ρqᶜˡ, i, j, k, θˡⁱ, ρ, q, ĉᵖ * Π, ℒˡ₀, ℒⁱ₀)

    g = constants.gravitational_acceleration
    b = AtmosphereModels.buoyancy_forceᶜᶜᶜ(i, j, k, grid, buoyancy.dynamics, T, qᵛᵉ, microphysics,
                                          buoyancy.microphysical_fields, constants) / ρ
    _, b_qᵛ, b_qˡ, b_qⁱ = buoyancy_derivatives(θˡⁱ, p, q, g - b, constants, κ̂, pˢᵗ)
    λ, _ = distributed_condensate_liquid_fraction(equilibrium, Tᵢ)
    b_qᶜ = λ * b_qˡ + (1 - λ) * b_qⁱ

    thermodynamics = (; constants, phase_equilibrium = equilibrium)
    linearization = saturation_linearization(thermodynamics, Tᵢ, p, q, θʷ, κ̂, pˢᵗ)
    c_q = linearization.v_q / linearization.D
    c_θ = linearization.v_θ / linearization.D
    return (; b_qᶜ, b_qᵛ, c_q, c_θ, qʷ, θʷ, qᶜ)
end

# The cloud liquid of the distributed water, which the moisture density leaves out: none where that is the total water
@inline distributed_cloud_liquid(::Nothing, i, j, k, ρ) = zero(ρ)
@inline distributed_cloud_liquid(ρqᶜˡ, i, j, k, ρ) = @inbounds ρqᶜˡ[i, j, k] / ρ

# θʷ: θˡⁱ where all the water is distributed, and otherwise θˡⁱ with the latent heat of the held condensate returned
@inline distributed_water_potential_temperature(::Nothing, i, j, k, θˡⁱ, ρ, q, ĉᵖΠ, ℒˡ₀, ℒⁱ₀) = θˡⁱ
@inline function distributed_water_potential_temperature(ρqᶜˡ, i, j, k, θˡⁱ, ρ, q, ĉᵖΠ, ℒˡ₀, ℒⁱ₀)
    qᶜˡ = @inbounds ρqᶜˡ[i, j, k] / ρ
    return θˡⁱ + (ℒˡ₀ * (q.liquid - qᶜˡ) + ℒⁱ₀ * q.ice) / ĉᵖΠ
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The static stability of a saturated displacement at the face `k`, whose condensate follows the saturation excess
``s = c_q qʷ - c_θ θʷ`` of its distributed water at fixed pressure:

```math
N²⁺ = ∂_z b + (b_{qᶜ} - b_{qᵛ}) (c_q ∂_z qʷ - c_θ ∂_z θʷ - ∂_z qᶜ),
```

with ``∂_z b`` the buoyancy gradient of the dynamics, ``b_{qᶜ} - b_{qᵛ}`` the buoyancy of the condensate of the
distributed water relative to that of vapor at fixed pressure and ``(c_q, c_θ)`` the linearized saturation excess,
from the model's thermodynamics as the energetics of the applied fluxes form them (`buoyancy_derivatives`,
`saturation_linearization`), averaged from the two centers. It is the energetics' saturated stratification with the
buoyancy gradient of the dynamics in place of the gradient of the mixed variables.
"""
@inline function saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    N²ᵇ = ∂z_b(i, j, k, grid, buoyancy, (; T, qᵛᵉ))
    s⁺ = saturated_displacementᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    s⁻ = saturated_displacementᶜᶜᶜ(i, j, k-1, grid, buoyancy, T, qᵛᵉ)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    bᶜᵛ = (s⁺.b_qᶜ + s⁻.b_qᶜ - (s⁺.b_qᵛ + s⁻.b_qᵛ)) / 2
    c_q = (s⁺.c_q + s⁻.c_q) / 2
    c_θ = (s⁺.c_θ + s⁻.c_θ) / 2
    ∂z_s = (c_q * (s⁺.qʷ - s⁻.qʷ) - c_θ * (s⁺.θʷ - s⁻.θʷ)) / Δz
    ∂z_qᶜ = (s⁺.qᶜ - s⁻.qᶜ) / Δz
    return N²ᵇ + bᶜᵛ * (∂z_s - ∂z_qᶜ)
end
