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
# vapor partial pressure. Saturation requires the local density from the moist EOS.
@inline function local_saturation_specific_humidity(i, j, k, buoyancy, Tᵢ, q)
    constants = buoyancy.thermodynamic_constants
    ρᵐ = gas_phase_density(i, j, k, buoyancy.dynamics, Tᵢ, q, constants)
    equilibrium = microphysics_phase_equilibrium(buoyancy.microphysics)
    return saturation_specific_humidity(Tᵢ, ρᵐ, constants, equilibrium)
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
