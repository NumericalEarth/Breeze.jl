#####
##### Static stability: how `TKEBasedTurbulenceClosure` diagnoses N² at the cell interfaces.
#####
##### `DryStaticStability` is the gradient ∂z_b of the buoyancy of the dynamics — condensate
##### loading included, latent heating excluded. `MoistStaticStability` switches, where the air
##### is saturated, to the buoyancy frequency of a saturated displacement (Durran & Klemp 1982),
##### in which the parcel condenses as it rises and its latent heating offsets part of the
##### environmental stratification. `SaturationWeightedStaticStability` blends the two with a weight that
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

Static stability using the dry buoyancy gradient in subsaturated air and the saturated
buoyancy frequency of [Durran and Klemp (1982)](@cite DurranKlemp1982) otherwise.
The saturation test excludes precipitating water, while its density loading remains in the
buoyancy frequency. Latent heat follows the microphysics' phase equilibrium. This is the
default for [`TKEBasedTurbulenceClosure`](@ref).
"""
struct MoistStaticStability end

Base.summary(::MoistStaticStability) = "MoistStaticStability"
Base.show(io::IO, ss::MoistStaticStability) = print(io, summary(ss))

@inline function static_stabilityᶜᶜᶠ(i, j, k, grid, ::MoistStaticStability, buoyancy, tracers)
    N²ᵈ = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²ˢ = saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
    saturated = saturatedᶜᶜᶠ(i, j, k, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
    return ifelse(saturated, N²ˢ, N²ᵈ)
end

"""
$(TYPEDEF)

Static stability as a blend of the dry buoyancy gradient ``N²ᵈ`` of [`DryStaticStability`](@ref)
and the saturated buoyancy frequency ``N²ˢ`` of [Durran and Klemp (1982)](@cite DurranKlemp1982),

```math
N² = (1 - aᴺ) N²ᵈ + aᴺ N²ˢ,
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
distribution is the cloud model [`CondensateFluxFraction`](@ref), ``aᴺ = w′qˡ′ / w′s′``, which depends on
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
"""
struct MicrophysicsSaturatedFraction end

SaturationWeightedStaticStability() = SaturationWeightedStaticStability(MicrophysicsSaturatedFraction())

Base.show(io::IO, ::MicrophysicsSaturatedFraction) = print(io, "MicrophysicsSaturatedFraction()")

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The subgrid model that `microphysics` holds, whose weight
[`SaturationWeightedStaticStability`](@ref)`()` uses: for P3, the model in its `subgrid` slot (for
example an external subgrid supersaturation model). `nothing` for a microphysics scheme without
one, for which there is no weight.
"""
@inline microphysics_subgrid_model(microphysics) = nothing
@inline microphysics_subgrid_model(p3::PredictedParticlePropertiesMicrophysics) = p3.subgrid

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

@inline function static_stabilityᶜᶜᶠ(i, j, k, grid, ss::SaturationWeightedStaticStability, buoyancy, tracers, closure_fields)
    N²ᵈ = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²ˢ = saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
    aᴺ = saturated_stability_weightᶜᶜᶠ(i, j, k, grid, ss.fraction, buoyancy, tracers, closure_fields)
    return N²ᵈ + aᴺ * (N²ˢ - N²ᵈ)
end

# A static stability that reads nothing of the closure fields is defined without them
@inline static_stabilityᶜᶜᶠ(i, j, k, grid, static_stability, buoyancy, tracers, closure_fields) =
    static_stabilityᶜᶜᶠ(i, j, k, grid, static_stability, buoyancy, tracers)

#####
##### Pointwise thermodynamics for the saturated branch
#####

# The latent heat over `surface` at temperature `T`, ℒ(T) = ℒ₀ + Δcᵖ T, linear in T with the
# vapor-condensate heat capacity difference; over a mixed-phase surface both coefficients
# interpolate between liquid and ice.
@inline latent_heat(T, constants, surface) =
    absolute_zero_latent_heat(constants, surface) + specific_heat_difference(constants, surface) * T

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

# Mixing ratios, per unit mass of dry air: qᵈ excludes all water, including rain.
@inline function saturation_mixing_ratioᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    qˢ = local_saturation_specific_humidity(i, j, k, buoyancy, Tᵢ, q)
    return qˢ / (1 - q.vapor - q.liquid - q.ice)
end

@inline function total_water_mixing_ratioᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    qᵗ = q.vapor + q.liquid + q.ice
    return qᵗ / (1 - qᵗ)
end

# ln θ of the dry potential temperature θ = T (pˢᵗ / p)^{Rᵈ / cᵖᵈ}
@inline function log_dry_potential_temperatureᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    constants = buoyancy.thermodynamic_constants
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    pˢᵗ = standard_pressure(buoyancy.dynamics)
    return log(Tᵢ) + Rᵈ / cᵖᵈ * log(pˢᵗ / p)
end

#####
##### The saturated branch at the face
#####

@inline function saturatedᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    qʷ = ℑzᵃᵃᶠ(i, j, k, grid, nonprecipitating_waterᶜᶜᶜ, buoyancy, T, qᵛᵉ)
    qˢ = ℑzᵃᵃᶠ(i, j, k, grid, saturation_specific_humidityᶜᶜᶜ, buoyancy, T, qᵛᵉ)
    return qʷ ≥ qˢ
end

@inline function saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛᵉ)
    constants = buoyancy.thermodynamic_constants
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    ϵ = Rᵈ / Rᵛ

    Tᶠ = ℑzᵃᵃᶠ(i, j, k, grid, T)
    rˢ = ℑzᵃᵃᶠ(i, j, k, grid, saturation_mixing_ratioᶜᶜᶜ, buoyancy, T, qᵛᵉ)
    equilibrium = microphysics_phase_equilibrium(buoyancy.microphysics)
    ℒ = latent_heat(Tᶠ, constants, equilibrated_surface(equilibrium, Tᶠ))

    ∂z_lnθ = ∂zᶜᶜᶠ(i, j, k, grid, log_dry_potential_temperatureᶜᶜᶜ, buoyancy, T, qᵛᵉ)
    ∂z_rˢ = ∂zᶜᶜᶠ(i, j, k, grid, saturation_mixing_ratioᶜᶜᶜ, buoyancy, T, qᵛᵉ)
    # All water contributes to density loading, even precipitation that does not
    # equilibrate with vapor during the parcel displacement.
    ∂z_rᵗ = ∂zᶜᶜᶠ(i, j, k, grid, total_water_mixing_ratioᶜᶜᶜ, buoyancy, T, qᵛᵉ)

    # The ratio of the saturated to the dry adiabatic lapse rate, Γₛ / Γ_d
    A = (1 + ℒ * rˢ / (Rᵈ * Tᶠ)) / (1 + ϵ * ℒ^2 * rˢ / (cᵖᵈ * Rᵈ * Tᶠ^2))

    return g * (A * (∂z_lnθ + ℒ / (cᵖᵈ * Tᶠ) * ∂z_rˢ) - ∂z_rᵗ)
end
