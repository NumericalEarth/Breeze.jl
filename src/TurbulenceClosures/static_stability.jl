#####
##### Static stability: how `TKEBasedTurbulenceClosure` diagnoses N² at the cell interfaces.
#####
##### `DryStaticStability` is the gradient ∂z_b of the buoyancy of the dynamics — condensate
##### loading included, latent heating excluded. `MoistStaticStability` switches, where the air
##### is saturated, to the buoyancy frequency of a saturated displacement (Durran & Klemp 1982),
##### in which the parcel condenses as it rises and its latent heating offsets part of the
##### environmental stratification. Both are evaluated once per stage into `closure_fields.N²`.
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

"""
$(TYPEDSIGNATURES)

The static stability ``N²`` at (Center, Center, Face) for `DryStaticStability`: the buoyancy
gradient `∂z_b` of the model.
"""
@inline static_stabilityᶜᶜᶠ(i, j, k, grid, ::DryStaticStability, buoyancy, tracers) =
    ∂z_b(i, j, k, grid, buoyancy, tracers)

"""
$(TYPEDEF)

The static stability of [`TKEBasedTurbulenceClosure`](@ref), and its default, with saturation taken
into account: the dry buoyancy gradient ``∂_z b`` ([`DryStaticStability`](@ref)) where the air is
subsaturated, and where it is saturated the buoyancy frequency of a saturated displacement of
[Durran and Klemp (1982)](@cite DurranKlemp1982),

```math
N²_s = g \\left[ \\frac{1 + ℒ rˢ / (Rᵈ T)}{1 + ϵ ℒ² rˢ / (cᵖᵈ Rᵈ T²)}
               \\left( ∂_z \\ln θ + \\frac{ℒ}{cᵖᵈ T} ∂_z rˢ \\right) - ∂_z rʷ \\right],
```

in which a parcel displaced upward condenses and its latent heating offsets part of the
environmental stratification. Here ``θ`` is the dry potential temperature of the air, ``rˢ`` the
saturation mixing ratio, ``rʷ`` the mixing ratio of nonprecipitating water — vapor, cloud liquid
and cloud ice, so that precipitation falling through subsaturated air does not make it saturated —
``ϵ = Rᵈ / Rᵛ``, and ``ℒ`` the latent heat. The mixing ratios are formed exactly from Breeze's
mass fractions with the dry-air mass fraction, ``r = q / (1 - qʷ)``.

The phase equilibrium of the microphysics (through `microphysics_phase_equilibrium`) supplies
both the saturation test and the parcel's latent heating: over a mixed-phase surface the
saturation vapor pressure and ``ℒ`` interpolate between liquid and ice with the equilibrium's
liquid fraction, so the liquid, ice and mixed-phase branches are one expression. The saturation
test compares the nonprecipitating water and the saturation specific humidity interpolated to
the face, ``qʷ ≥ qˢ``.
"""
struct MoistStaticStability end

Base.summary(::MoistStaticStability) = "MoistStaticStability"
Base.show(io::IO, ss::MoistStaticStability) = print(io, summary(ss))

@inline function static_stabilityᶜᶜᶠ(i, j, k, grid, ::MoistStaticStability, buoyancy, tracers)
    N²ᵈ = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²ˢ = saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, tracers.T, tracers.qᵛ)
    saturated = saturatedᶜᶜᶠ(i, j, k, grid, buoyancy, tracers.T, tracers.qᵛ)
    return ifelse(saturated, N²ˢ, N²ᵈ)
end

#####
##### Pointwise thermodynamics for the saturated branch
#####

# The latent heat over `surface` at temperature `T`, ℒ(T) = ℒ₀ + Δcᵖ T, linear in T with the
# vapor-condensate heat capacity difference; over a mixed-phase surface both coefficients
# interpolate between liquid and ice.
@inline latent_heat(T, constants, surface) =
    absolute_zero_latent_heat(constants, surface) + specific_heat_difference(constants, surface) * T

# Temperature, pressure, total density and moisture mass fractions at cell `(i, j, k)`
@inline function moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    dynamics = buoyancy.dynamics
    p_field = dynamics_pressure(dynamics)
    ρ_field = total_density(dynamics)

    @inbounds begin
        Tᵢ = T[i, j, k]
        p = p_field[i, j, k]
        ρ = ρ_field[i, j, k]
        qᵛᵢ = qᵛ[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, buoyancy.microphysics, ρ, qᵛᵢ, buoyancy.microphysical_fields)
    return Tᵢ, p, ρ, q
end

@inline function saturation_specific_humidityᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    equilibrium = microphysics_phase_equilibrium(buoyancy.microphysics)
    return saturation_specific_humidity(Tᵢ, ρ, buoyancy.thermodynamic_constants, equilibrium)
end

@inline function nonprecipitating_waterᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    return q.vapor + q.liquid + q.ice
end

# Mixing ratios, per unit mass of dry air: r = q / qᵈ with qᵈ = 1 - qʷ
@inline function saturation_mixing_ratioᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    qˢ = saturation_specific_humidityᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    qʷ = nonprecipitating_waterᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    return qˢ / (1 - qʷ)
end

@inline function nonprecipitating_water_mixing_ratioᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    qʷ = nonprecipitating_waterᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    return qʷ / (1 - qʷ)
end

# ln θ of the dry potential temperature θ = T (pˢᵗ / p)^{Rᵈ / cᵖᵈ}
@inline function log_dry_potential_temperatureᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    Tᵢ, p, ρ, q = moist_stateᶜᶜᶜ(i, j, k, grid, buoyancy, T, qᵛ)
    constants = buoyancy.thermodynamic_constants
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    pˢᵗ = standard_pressure(buoyancy.dynamics)
    return log(Tᵢ) + Rᵈ / cᵖᵈ * log(pˢᵗ / p)
end

#####
##### The saturated branch at the face
#####

"""
$(TYPEDSIGNATURES)

Whether the air at face `(i, j, k)` is saturated: the nonprecipitating water and the saturation
specific humidity, both interpolated from the adjacent cell centers, satisfy ``qʷ ≥ qˢ``.
"""
@inline function saturatedᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛ)
    qʷ = ℑzᵃᵃᶠ(i, j, k, grid, nonprecipitating_waterᶜᶜᶜ, buoyancy, T, qᵛ)
    qˢ = ℑzᵃᵃᶠ(i, j, k, grid, saturation_specific_humidityᶜᶜᶜ, buoyancy, T, qᵛ)
    return qʷ ≥ qˢ
end

"""
$(TYPEDSIGNATURES)

The buoyancy frequency of a saturated displacement at face `(i, j, k)`, the expression of
[Durran and Klemp (1982)](@cite DurranKlemp1982) given in [`MoistStaticStability`](@ref), with
the gradients as differences between the adjacent cell centers and the coefficients at the face.
"""
@inline function saturated_static_stabilityᶜᶜᶠ(i, j, k, grid, buoyancy, T, qᵛ)
    constants = buoyancy.thermodynamic_constants
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    ϵ = Rᵈ / Rᵛ

    Tᶠ = ℑzᵃᵃᶠ(i, j, k, grid, T)
    rˢ = ℑzᵃᵃᶠ(i, j, k, grid, saturation_mixing_ratioᶜᶜᶜ, buoyancy, T, qᵛ)
    equilibrium = microphysics_phase_equilibrium(buoyancy.microphysics)
    ℒ = latent_heat(Tᶠ, constants, equilibrated_surface(equilibrium, Tᶠ))

    ∂z_lnθ = ∂zᶜᶜᶠ(i, j, k, grid, log_dry_potential_temperatureᶜᶜᶜ, buoyancy, T, qᵛ)
    ∂z_rˢ = ∂zᶜᶜᶠ(i, j, k, grid, saturation_mixing_ratioᶜᶜᶜ, buoyancy, T, qᵛ)
    ∂z_rʷ = ∂zᶜᶜᶠ(i, j, k, grid, nonprecipitating_water_mixing_ratioᶜᶜᶜ, buoyancy, T, qᵛ)

    # The ratio of the saturated to the dry adiabatic lapse rate, Γₛ / Γ_d
    A = (1 + ℒ * rˢ / (Rᵈ * Tᶠ)) / (1 + ϵ * ℒ^2 * rˢ / (cᵖᵈ * Rᵈ * Tᶠ^2))

    return g * (A * (∂z_lnθ + ℒ / (cᵖᵈ * Tᶠ) * ∂z_rˢ) - ∂z_rʷ)
end
