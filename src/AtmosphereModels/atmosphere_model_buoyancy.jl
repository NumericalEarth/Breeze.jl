using Oceananigans.TurbulenceClosures: TurbulenceClosures as OceanTurbulenceClosures
using Oceananigans.BuoyancyFormulations: BuoyancyFormulations as OceanBuoyancyFormulations
using Oceananigans.Operators: ∂zᶜᶜᶠ

"""
$(TYPEDEF)

Wrapper struct for computing buoyancy for [`AtmosphereModel`](@ref)
in the context of a turbulence closure. Used to interface with Oceananigans
turbulence closures that require buoyancy gradients. The microphysics and its fields
are carried so that the moisture mass fractions — condensate included — can be
recovered at any grid point through `grid_moisture_fractions`.
"""
struct AtmosphereModelBuoyancy{D, F, T, M, MF}
    dynamics :: D
    formulation :: F
    thermodynamic_constants :: T
    microphysics :: M
    microphysical_fields :: MF
end

Adapt.adapt_structure(to, b::AtmosphereModelBuoyancy) =
    AtmosphereModelBuoyancy(adapt(to, b.dynamics),
                            adapt(to, b.formulation),
                            adapt(to, b.thermodynamic_constants),
                            adapt(to, b.microphysics),
                            adapt(to, b.microphysical_fields))

#####
##### Buoyancy interface for AtmosphereModel
#####

OceanTurbulenceClosures.buoyancy_force(model::AtmosphereModel) =
    AtmosphereModelBuoyancy(model.dynamics, model.formulation, model.thermodynamic_constants,
                            model.microphysics, model.microphysical_fields)

# buoyancy_tracers returns tracers needed for:
# 1. Buoyancy computation (T, qᵛ) used in ∂z_b and AMD viscosity; the condensate enters
#    through the microphysics carried by `AtmosphereModelBuoyancy`
# 2. Diffusivity computation for each tracer in closure_fields.κₑ
# The energy_density and moisture_density are first (matching closure_names order),
# followed by user tracers, then diagnostic fields for buoyancy.
function OceanTurbulenceClosures.buoyancy_tracers(model::AtmosphereModel)
    # Diagnostic fields for buoyancy gradient calculation
    buoyancy_tracers = (; T = model.temperature, qᵛ = specific_humidity(model))
    # Prognostic tracer fields for diffusivity computation
    moist_name = moisture_prognostic_name(model.microphysics)
    prognostic_tracers = merge(prognostic_fields(model.formulation), NamedTuple{(moist_name,)}((model.moisture_density,)))
    # Merge with user tracers
    all_prognostic = merge(prognostic_tracers, model.tracers)
    # Final merge - buoyancy tracers at end for named access in ∂z_b
    return merge(all_prognostic, buoyancy_tracers)
end

"""
$(TYPEDSIGNATURES)

The static stability ``N² = g ∂_z \\ln θᵨ`` at (Center, Center, Face), the vertical gradient of the
buoyancy ``b = -g ρ′ / ρᵣ`` that the dynamics uses, written through the density potential
temperature [`density_potential_temperature`](@ref) so that condensate loading is included.
The logarithm is differentiated, ``∂_z \\ln θᵨ = ∂_z θᵨ / θᵨ``, to keep the derivative and its
denominator at consistent grid locations.
"""
@inline function OceanBuoyancyFormulations.∂z_b(i, j, k, grid, b::AtmosphereModelBuoyancy, tracers)
    g = b.thermodynamic_constants.gravitational_acceleration
    ∂z_log_θᵨ = ∂zᶜᶜᶠ(i, j, k, grid, log_density_potential_temperature, b, tracers.T, tracers.qᵛ)
    return g * ∂z_log_θᵨ
end

@inline function log_density_potential_temperature(i, j, k, grid, b, T, qᵛ)
    θᵨ = density_potential_temperature(i, j, k, grid, b, T, qᵛ)
    return log(θᵨ)
end

"""
$(TYPEDSIGNATURES)

The density potential temperature at cell `(i, j, k)`,

```math
θᵨ = \\frac{Rᵐ}{Rᵈ} \\, T \\left( \\frac{pˢᵗ}{p} \\right)^{Rᵈ / cᵖᵈ},
\\qquad Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ, \\qquad qᵈ = 1 - qᵛ - qˡ - qⁱ,
```

the potential temperature of the dry air that has the density of the moist, condensate-laden
air at the same pressure. Since ``ρ = p / (Rᵐ T)``, at fixed pressure ``θᵨ ∝ 1 / ρ`` and the
buoyancy of the dynamics is ``b = -g ρ′ / ρᵣ ≈ g \\, θᵨ′ / θᵨ``; condensate enters through the
dry-air mass fraction ``qᵈ``, so that liquid and ice water load the air by their mass. The
moisture mass fractions come from the microphysics through `grid_moisture_fractions`, and the
pressure is the dynamics pressure.
"""
@inline function density_potential_temperature(i, j, k, grid, b::AtmosphereModelBuoyancy, T, qᵛ)
    constants = b.thermodynamic_constants
    dynamics = b.dynamics
    p_field = dynamics_pressure(dynamics)
    ρ_field = total_density(dynamics)

    @inbounds begin
        p = p_field[i, j, k]
        ρ = ρ_field[i, j, k]
        qᵛᵢ = qᵛ[i, j, k]
        Tᵢ = T[i, j, k]
    end

    pˢᵗ = standard_pressure(dynamics)
    q = grid_moisture_fractions(i, j, k, grid, b.microphysics, ρ, qᵛᵢ, b.microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity

    return Rᵐ / Rᵈ * Tᵢ * (pˢᵗ / p)^(Rᵈ / cᵖᵈ)
end
