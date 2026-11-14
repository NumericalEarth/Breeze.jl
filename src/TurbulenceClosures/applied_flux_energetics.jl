#####
##### The energy conversions of the fluxes `TKEBasedTurbulenceClosure` applies, for the TKE
##### equation of a closure with a `CloudBuoyancyFlux`
#####
##### The closure diffuses every scalar `φ` (θˡⁱ, vapor, every condensate species, ...) with `Kᶜ`
##### and momentum with `Kᵘ`, density weighted, through the flux `ρᶠ K ∂z φ` at the interior
##### faces. The turbulent kinetic energy it produces is the energy those fluxes take from the
##### mean state, face by face:
#####
#####   shear:     Pᶠ = Kᵘ S²                                  (S² of the stored velocities)
#####   buoyancy:  Bᶠ = Σₛ βₛᶠ Fₛ,   Fₛ = -Kᶜ (φₛ,ₖ - φₛ,ₖ₋₁) / Δzᶠ,   βₛ = ∂b/∂φₛ
#####
##### where `b` is the buoyancy of the dynamics (for anelastic dynamics b = g (1 - Rᵐᵣ Tᵣ / (Rᵐ T)),
##### condensate loading through Rᵐ) and `βₛ` its exact derivative with respect to each mixed
##### specific variable at fixed pressure, at the centers, averaged to the face (`βₛᶠ`). The
##### mixed variables that change `b` are θˡⁱ and the vapor, liquid and ice mass fractions of the
##### microphysics (`grid_moisture_fractions`, linear in its prognostic densities, so the flux of
##### all liquid is the sum of the liquid species' fluxes, and so on).
#####
##### Summation by parts makes these exact discrete conversions:
#####
#####   Σₖ uₖ Δzₖ ∂z(ρ Kᵘ ∂z u)ₖ = -Σᶠ ρᶠ Δzᶠ Pᶠ + boundary terms,
#####   d/dt Σₖ ρₖ Δzₖ h‡ₖ (closure part) = -Σᶠ ρᶠ Δzᶠ Bᶠ + boundary terms,
#####
##### with the dynamic enthalpy h‡ = -∫₀ᶻ b dz′ discretized by the trapezoid rule on the
##### centers (∂h‡/∂φₛ at center k is -Zₛ,ₖ, Zₛ,ₖ - Zₛ,ₖ₋₁ = βₛᶠ Δzᶠ). The face conversions are
##### moved to the centers, where the TKE lives, with the half-cell weights of the two faces of
##### each cell,
#####
#####   ρᶜ Δzᶜ Xᶜ = ρᶠₖ (zᶜₖ - zᶠₖ) Xᶠₖ + ρᶠₖ₊₁ (zᶠₖ₊₁ - zᶜₖ) Xᶠₖ₊₁,
#####
##### so that Σ ρᶜ Δzᶜ Xᶜ = Σ ρᶠ Δzᶠ Xᶠ exactly. The closure's diffusivities vanish at the
##### boundary faces; the surface fluxes of the boundary conditions are boundary terms of the
##### energy budget, not conversions.
#####
##### Time level: the conversions are formed from the stage state and the diffusivities the stage
##### solve uses. The solve is backward Euler in each stage, so the fluxes it applies are those of
##### the solved state; the difference is O(Δt) and is not closed here.
#####

using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ

using Breeze.Thermodynamics: mixture_heat_capacity, mixture_gas_constant, exner_exponent,
                             VaporEquivalentExponent, DryExponent

# ∂κ/∂qᵗ of the Exner exponent; κ̂ = R̂ / ĉᵖ with R̂, ĉᵖ linear in qᵗ
@inline exner_exponent_moisture_derivative(::DryExponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ) = zero(R̂)
@inline exner_exponent_moisture_derivative(::VaporEquivalentExponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ) =
    R̂ / ĉᵖ * ((Rᵛ - Rᵈ) / R̂ - (cᵖᵛ - cᵖᵈ) / ĉᵖ)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The derivatives `(βᶿ, βᵛ, βˡ, βⁱ)` of the buoyancy `b` of the dynamics with respect to the
liquid-ice potential temperature and the vapor, liquid and ice mass fractions, at fixed pressure,
at cell center `(i, j, k)`. With ``X = Rᵐ T``, ``T = (ĉᵖ Π θˡⁱ + ℒˡ₀ qˡ + ℒⁱ₀ qⁱ) / cᵖᵐ`` and
``b = g (1 - Xᵣ / X)``, ``∂b/∂φ = (g - b) ∂ ln X / ∂φ``; `g - b` is taken from the buoyancy force
of the dynamics, so the derivatives hold for any dynamics whose buoyancy is a function of
``Rᵐ T`` at fixed pressure.
"""
@inline function buoyancy_derivativesᶜᶜᶜ(i, j, k, grid, s)
    constants = s.constants
    @inbounds begin
        ρ = s.ρ[i, j, k]
        θ = s.ρθ[i, j, k] / ρ
        qᵛᵉ = s.qᵛᵉ[i, j, k]
        p = s.p[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)

    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)

    qᵗ = q.vapor + q.liquid + q.ice
    ĉᵖ = cᵖᵈ + qᵗ * (cᵖᵛ - cᵖᵈ)
    R̂ = Rᵈ + qᵗ * (Rᵛ - Rᵈ)
    κ = exner_exponent(s.exner_exponent, q, constants)
    κ′ = exner_exponent_moisture_derivative(s.exner_exponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    ln_p = log(p / s.standard_pressure)
    Π = exp(κ * ln_p)

    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    H = ĉᵖ * Π * θ + ℒˡ₀ * q.liquid + ℒⁱ₀ * q.ice # = cᵖᵐ T
    ∂H∂qᵗ = θ * Π * ((cᵖᵛ - cᵖᵈ) + ĉᵖ * ln_p * κ′)

    # g - b = g ρ / ρᵣ from the buoyancy force density -g (ρ - ρᵣ) of the dynamics
    g = constants.gravitational_acceleration
    b = AtmosphereModels.buoyancy_forceᶜᶜᶜ(i, j, k, grid, s.dynamics, s.T, s.qᵛᵉ,
                                          s.microphysics, s.microphysical_fields, constants) / ρ
    G = g - b

    βᶿ = G * ĉᵖ * Π / H
    βᵛ = G * ((Rᵛ - Rᵈ) / Rᵐ + ∂H∂qᵗ / H - (cᵖᵛ - cᵖᵈ) / cᵖᵐ)
    βˡ = G * (-Rᵈ / Rᵐ + (∂H∂qᵗ + ℒˡ₀) / H - (cˡ - cᵖᵈ) / cᵖᵐ)
    βⁱ = G * (-Rᵈ / Rᵐ + (∂H∂qᵗ + ℒⁱ₀) / H - (cⁱ - cᵖᵈ) / cᵖᵐ)

    return βᶿ, βᵛ, βˡ, βⁱ, θ, q
end

# Bᶠ at an interior face k, with Kᶜ of the face (zero at the boundary faces)
@inline function applied_buoyancy_conversionᶜᶜᶠ(i, j, k, grid, s, Kᶜ)
    βᶿ⁺, βᵛ⁺, βˡ⁺, βⁱ⁺, θ⁺, q⁺ = buoyancy_derivativesᶜᶜᶜ(i, j, k, grid, s)
    βᶿ⁻, βᵛ⁻, βˡ⁻, βⁱ⁻, θ⁻, q⁻ = buoyancy_derivativesᶜᶜᶜ(i, j, k-1, grid, s)
    Δb = (βᶿ⁺ + βᶿ⁻) / 2 * (θ⁺ - θ⁻) +
         (βᵛ⁺ + βᵛ⁻) / 2 * (q⁺.vapor - q⁻.vapor) +
         (βˡ⁺ + βˡ⁻) / 2 * (q⁺.liquid - q⁻.liquid) +
         (βⁱ⁺ + βⁱ⁻) / 2 * (q⁺.ice - q⁻.ice)
    return - @inbounds(Kᶜ[i, j, k]) * Δb / Δzᶜᶜᶠ(i, j, k, grid)
end

@inline applied_shear_conversionᶜᶜᶠ(i, j, k, grid, Kᵘ, u, v) = @inbounds Kᵘ[i, j, k] * shearᶜᶜᶠ(i, j, k, grid, u, v)

# The conversions at the centers, from those at the faces by half-cell weights. The boundary faces
# carry none (the closure's diffusivities vanish there), so they are not evaluated.
@kernel function _compute_applied_flux_conversions!(B, P, grid, s, Kᶜ, Kᵘ, u, v)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())
    FT = eltype(grid)
    @inbounds begin
        Bˡᵒʷ = zero(FT)
        Pˡᵒʷ = zero(FT)
        for k in 1:Nz
            interior_top = k < Nz
            kᵘ = min(k + 1, Nz)
            Bᵘᵖ = ifelse(interior_top, applied_buoyancy_conversionᶜᶜᶠ(i, j, kᵘ, grid, s, Kᶜ), zero(FT))
            Pᵘᵖ = ifelse(interior_top, applied_shear_conversionᶜᶜᶠ(i, j, kᵘ, grid, Kᵘ, u, v), zero(FT))
            ρˡᵒʷ = ℑzᵃᵃᶠ(i, j, k, grid, s.ρ)
            ρᵘᵖ = ℑzᵃᵃᶠ(i, j, k + 1, grid, s.ρ)
            zᶜ = znode(i, j, k, grid, ccc...)
            wˡᵒʷ = ρˡᵒʷ * (zᶜ - znode(i, j, k, grid, ccf...))
            wᵘᵖ = ρᵘᵖ * (znode(i, j, k + 1, grid, ccf...) - zᶜ)
            ρΔz = s.ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            B[i, j, k] = (wˡᵒʷ * Bˡᵒʷ + wᵘᵖ * Bᵘᵖ) / ρΔz
            P[i, j, k] = (wˡᵒʷ * Pˡᵒʷ + wᵘᵖ * Pᵘᵖ) / ρΔz
            Bˡᵒʷ = Bᵘᵖ
            Pˡᵒʷ = Pᵘᵖ
        end
    end
end

# What the derivatives read
function applied_flux_state(model)
    return (ρ = total_density(model.dynamics),
            ρθ = distribution_potential_temperature_density(model),
            qᵛᵉ = AtmosphereModels.specific_prognostic_moisture(model),
            T = model.temperature,
            p = dynamics_pressure(model.dynamics),
            standard_pressure = standard_pressure(model.dynamics),
            exner_exponent = AtmosphereModels.liquid_ice_exner_exponent(model.formulation),
            dynamics = model.dynamics,
            microphysics = model.microphysics,
            microphysical_fields = model.microphysical_fields,
            constants = model.thermodynamic_constants)
end

compute_applied_flux_conversions!(fields, closure_fields, model) = nothing

function compute_applied_flux_conversions!(fields::SubgridCloudFields{<:CloudBuoyancyFlux}, closure_fields, model)
    grid = model.grid
    cl = fields.cloud
    u, v = model.velocities.u, model.velocities.v
    launch!(grid.architecture, grid, :xy, _compute_applied_flux_conversions!, cl.B, cl.P, grid,
            applied_flux_state(model), closure_fields.Kᶜ, closure_fields.Kᵘ, u, v)
    return nothing
end

# The shear production of the TKE equation: CATKE's without the cloud buoyancy flux, the
# conversion of the applied momentum fluxes with it
@inline tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields, closure_fields, u, v) =
    shear_production(i, j, k, grid, closure_fields.Kᵘ, u, u, v, v)

@inline tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:CloudBuoyancyFlux}, closure_fields, u, v) =
    @inbounds fields.cloud.P[i, j, k]

#####
##### The floor of e in the buoyancy sink rate
#####

# The implicit buoyancy sink is ωᴮ e with ωᴮ = -B⁻ / max(e, floor). Above the floor it removes
# exactly -B⁻, the energy the applied fluxes give the mean state. The plain closure floors at
# `minimum_tke`, so below it the sink removes less than the mean state gains. With the cloud
# buoyancy flux the floor is a tiny positive number instead, kept only so that ωᴮ is finite at
# e = 0 (its benefit: the implicit sink cannot divide by zero, and e stays non-negative in the
# implicit solve); the conversion is exact wherever e exceeds it.
@inline buoyancy_sink_floor(fields, closure) = closure.minimum_tke
@inline buoyancy_sink_floor(::SubgridCloudFields{<:CloudBuoyancyFlux}, closure) = sqrt(floatmin(typeof(closure.minimum_tke)))

#####
##### Dissipative heating
#####

# The energy the sinks of the TKE equation remove by dissipation (and by the damping of negative
# e) heats the air at fixed pressure and composition, d(cᵖᵐ T) = ĉᵖ Π dθˡⁱ. The sink rate is the
# stored implicit coefficient without its buoyancy part, -Lᵉ - ωᴮ with ωᴮ = -B⁻ / max(e, eᵐⁱⁿ),
# both formed from this stage's state, so the heating is the energy that `Lᵉ e` removes.
@kernel function _add_dissipative_heating!(Gρθ, grid, closure, closure_fields, ρe, s)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        e = ρe[i, j, k] / ρ
        B = closure_fields.subgrid_cloud.cloud.B[i, j, k]
        Lᵉ = closure_fields.Lᵉ[i, j, k]
        qᵛᵉ = s.qᵛᵉ[i, j, k]
        p = s.p[i, j, k]
    end
    ωᴮ = -min(0, B) / max(e, buoyancy_sink_floor(closure_fields.subgrid_cloud, closure_ij))
    ε = (-Lᵉ - ωᴮ) * e
    constants = s.constants
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)
    κ = exner_exponent(s.exner_exponent, q, constants)
    Π = (p / s.standard_pressure)^κ
    qᵗ = q.vapor + q.liquid + q.ice
    ĉᵖ = constants.dry_air.heat_capacity + qᵗ * (constants.vapor.heat_capacity - constants.dry_air.heat_capacity)
    active = !inactive_cell(i, j, k, grid)
    @inbounds Gρθ[i, j, k] += active * ρ * ε / (ĉᵖ * Π)
end

add_dissipative_heating!(Gⁿ, fields, closure_fields, closure, model) = nothing

function add_dissipative_heating!(Gⁿ, fields::SubgridCloudFields{<:CloudBuoyancyFlux}, closure_fields, closure, model)
    fields.buoyancy_flux.dissipative_heating || return nothing
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _add_dissipative_heating!, Gⁿ[name], grid, closure, closure_fields,
            model.tracers[TKE_NAME], applied_flux_state(model))
    return nothing
end
