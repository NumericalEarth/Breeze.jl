#####
##### The energy conversions of the fluxes `TKEBasedTurbulenceClosure` applies, for the TKE
##### equation of a closure with a `AppliedFluxEnergetics`
#####
##### The closure diffuses every scalar `φ` (θˡⁱ, vapor, every condensate species, ...) with `Kᶜ`
##### and momentum with `Kᵘ`, density weighted, through the flux `ρᶠ K ∂z φ` at the interior
##### faces. The turbulent kinetic energy it produces is the energy those fluxes take from the
##### mean state, face by face:
#####
#####   shear:     Pᶠ = Kᵘ S²                                  (S² of the stored velocities)
#####   buoyancy:  Bᶠ = Σ_χ b_χᶠ F_χ,   Fₛ = -Kᶜ (φₛ,ₖ - φₛ,ₖ₋₁) / Δzᶠ,   b_χ = ∂b/∂χ
#####
##### where `b` is the buoyancy of the dynamics (for anelastic dynamics b = g (1 - Rᵐᵣ Tᵣ / (Rᵐ T)),
##### condensate loading through Rᵐ) and `b_χ` its exact derivative with respect to each mixed
##### specific variable at fixed pressure, at the centers, averaged to the face (`b_χᶠ`). The
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
##### centers (∂h‡/∂φₛ at center k is -Zₛ,ₖ, Zₛ,ₖ - Zₛ,ₖ₋₁ = b_χᶠ Δzᶠ). The face conversions are
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
using Oceananigans.Models: boundary_condition_args

using Oceananigans.BoundaryConditions: BoundaryCondition, Flux, Bottom, total_boundary_flux

# The density-weighted flux through the bottom of column (i, j) that the boundary condition `bc` of the
# prognostic field `c` prescribes: positive upward, into the domain. Only `Flux` boundary conditions carry one
# (the default no-flux condition gives zero).
@inline bottom_boundary_flux(bc::BoundaryCondition{<:Flux}, i, j, grid, c, args...) =
    total_boundary_flux(bc, Bottom(), i, j, 1, grid, c, args...)

@inline bottom_boundary_flux(bc, i, j, grid, c, args...) = zero(grid)

using Breeze.Thermodynamics: mixture_heat_capacity, mixture_gas_constant, exner_exponent,
                             VaporEquivalentExponent, DryExponent

# ∂κ/∂qᵗ of the Exner exponent; κ̂ = R̂ / ĉᵖ with R̂, ĉᵖ linear in qᵗ
@inline exner_exponent_moisture_derivative(::DryExponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ) = zero(R̂)
@inline exner_exponent_moisture_derivative(::VaporEquivalentExponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ) =
    R̂ / ĉᵖ * ((Rᵛ - Rᵈ) / R̂ - (cᵖᵛ - cᵖᵈ) / ĉᵖ)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The derivatives `(b_θ, b_qᵛ, b_qˡ, b_qⁱ)` of the buoyancy `b` of the dynamics with respect to the
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

    b_θ = G * ĉᵖ * Π / H
    b_qᵛ = G * ((Rᵛ - Rᵈ) / Rᵐ + ∂H∂qᵗ / H - (cᵖᵛ - cᵖᵈ) / cᵖᵐ)
    b_qˡ = G * (-Rᵈ / Rᵐ + (∂H∂qᵗ + ℒˡ₀) / H - (cˡ - cᵖᵈ) / cᵖᵐ)
    b_qⁱ = G * (-Rᵈ / Rᵐ + (∂H∂qᵗ + ℒⁱ₀) / H - (cⁱ - cᵖᵈ) / cᵖᵐ)

    return b_θ, b_qᵛ, b_qˡ, b_qⁱ, θ, q
end

# The derivatives of the buoyancy and the enthalpy coefficient d(cᵖᵐ T)/dθˡⁱ = ĉᵖ Π at the centers,
# stored once per update so that the conversions after the implicit solves use the same ones
@kernel function _compute_buoyancy_derivatives!(cl, grid, s)
    i, j, k = @index(Global, NTuple)
    b_θ, b_qᵛ, b_qˡ, b_qⁱ, θ, q = buoyancy_derivativesᶜᶜᶜ(i, j, k, grid, s)
    constants = s.constants
    κ = exner_exponent(s.exner_exponent, q, constants)
    Π = (@inbounds(s.p[i, j, k]) / s.standard_pressure)^κ
    qᵗ = q.vapor + q.liquid + q.ice
    ĉᵖ = constants.dry_air.heat_capacity + qᵗ * (constants.vapor.heat_capacity - constants.dry_air.heat_capacity)
    @inbounds begin
        cl.b_θ[i, j, k] = b_θ
        cl.b_qᵛ[i, j, k] = b_qᵛ
        cl.b_qˡ[i, j, k] = b_qˡ
        cl.b_qⁱ[i, j, k] = b_qⁱ
        cl.ĉᵖΠ[i, j, k] = ĉᵖ * Π
    end
end

compute_buoyancy_derivatives!(fields, model) = nothing

function compute_buoyancy_derivatives!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, model)
    grid = model.grid
    cl = fields.cloud
    launch!(grid.architecture, grid, :xyz, _compute_buoyancy_derivatives!, cl, grid, applied_flux_state(model))
    fill_halo_regions!((cl.b_θ, cl.b_qᵛ, cl.b_qˡ, cl.b_qⁱ))
    return nothing
end

# Bᶠ at the interior face k from the stored derivatives and the current state
@inline applied_buoyancy_conversionᶜᶜᶠ(i, j, k, grid, s, cl, Kᶜ) =
    - @inbounds(Kᶜ[i, j, k]) * applied_stratificationᶜᶜᶠ(i, j, k, grid, s, cl)

# The stratification of the mixed variables, Σ b̄_χ δz χ: the buoyancy gradient a displacement
# without phase change feels (the stratification behind the applied buoyancy flux)
@inline function applied_stratificationᶜᶜᶠ(i, j, k, grid, s, cl)
    @inbounds begin
        ρ⁺ = s.ρ[i, j, k]
        ρ⁻ = s.ρ[i, j, k-1]
        θ⁺ = s.ρθ[i, j, k] / ρ⁺
        θ⁻ = s.ρθ[i, j, k-1] / ρ⁻
        q⁺ = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ⁺, s.ρqᵛᵉ[i, j, k] / ρ⁺, s.microphysical_fields)
        q⁻ = grid_moisture_fractions(i, j, k-1, grid, s.microphysics, ρ⁻, s.ρqᵛᵉ[i, j, k-1] / ρ⁻, s.microphysical_fields)
        Δb = (cl.b_θ[i, j, k] + cl.b_θ[i, j, k-1]) / 2 * (θ⁺ - θ⁻) +
             (cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, k-1]) / 2 * (q⁺.vapor - q⁻.vapor) +
             (cl.b_qˡ[i, j, k] + cl.b_qˡ[i, j, k-1]) / 2 * (q⁺.liquid - q⁻.liquid) +
             (cl.b_qⁱ[i, j, k] + cl.b_qⁱ[i, j, k-1]) / 2 * (q⁺.ice - q⁻.ice)
        return Δb / Δzᶜᶜᶠ(i, j, k, grid)
    end
end

@inline applied_shear_conversionᶜᶜᶠ(i, j, k, grid, Kᵘ, u, v) = @inbounds Kᵘ[i, j, k] * shearᶜᶜᶠ(i, j, k, grid, u, v)

# The conversions of the interior fluxes at center k, from those at its two faces by half-cell
# weights (the boundary faces carry none: the closure's diffusivities vanish there)
@inline function interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    kˡᵒʷ = max(k, 2)
    kᵘᵖ = min(k + 1, Nz)
    Bˡᵒʷ = ifelse(k > 1, applied_buoyancy_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, s, cl, Kᶜ), zero(FT))
    Bᵘᵖ = ifelse(k < Nz, applied_buoyancy_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, s, cl, Kᶜ), zero(FT))
    Lˡᵒʷ = ifelse(k > 1, latent_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, cl, Kᶜ), zero(FT))
    Lᵘᵖ = ifelse(k < Nz, latent_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, cl, Kᶜ), zero(FT))
    Pˡᵒʷ = ifelse(k > 1, applied_shear_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, Kᵘ, u, v), zero(FT))
    Pᵘᵖ = ifelse(k < Nz, applied_shear_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, Kᵘ, u, v), zero(FT))
    zᶜ = znode(i, j, k, grid, Center(), Center(), Center())
    wˡᵒʷ = ℑzᵃᵃᶠ(i, j, k, grid, s.ρ) * (zᶜ - znode(i, j, k, grid, Center(), Center(), Face()))
    wᵘᵖ = ℑzᵃᵃᶠ(i, j, k + 1, grid, s.ρ) * (znode(i, j, k + 1, grid, Center(), Center(), Face()) - zᶜ)
    ρΔz = @inbounds s.ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
    return (wˡᵒʷ * Bˡᵒʷ + wᵘᵖ * Bᵘᵖ) / ρΔz, (wˡᵒʷ * Pˡᵒʷ + wᵘᵖ * Pᵘᵖ) / ρΔz, (wˡᵒʷ * Lˡᵒʷ + wᵘᵖ * Lᵘᵖ) / ρΔz
end

# The latent part of the buoyancy production at a face, Kᶜ ΔN²ᴸ (see `_latent_corrected_stability!`)
@inline latent_conversionᶜᶜᶠ(i, j, k, grid, cl, Kᶜ) = @inbounds Kᶜ[i, j, k] * cl.ΔN²ᴸ[i, j, k]

# The conversions of the surface fluxes within the bottom half-cell: the work of the surface
# stress on the lowest level, -u₁ 𝒯ˣ - v₁ 𝒯ʸ (𝒯 the upward flux of ρu, ρv), and the dynamic
# enthalpy the surface fluxes of θˡⁱ and vapor remove, (z₁ - z_surface) Σ b_χ,₁ J_χ, per unit mass
# of the lowest cell. Condensate has no surface flux here.
@inline function surface_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, u, v, bottom_bcs, bc_args)
    FT = grid_float_type(grid)
    Jᶿ = bottom_boundary_flux(bottom_bcs.ρθ, i, j, grid, s.ρθ, bc_args...)
    Jᵛ = bottom_boundary_flux(bottom_bcs.ρqᵛᵉ, i, j, grid, s.ρqᵛᵉ, bc_args...)
    𝒯ˣ = bottom_boundary_flux(bottom_bcs.ρu, i, j, grid, s.ρu, bc_args...)
    𝒯ʸ = bottom_boundary_flux(bottom_bcs.ρv, i, j, grid, s.ρv, bc_args...)
    h = znode(i, j, 1, grid, Center(), Center(), Center()) - znode(i, j, 1, grid, Center(), Center(), Face())
    @inbounds begin
        ρΔz = s.ρ[i, j, 1] * Δzᶜᶜᶜ(i, j, 1, grid)
        Bˢ = h * (cl.b_θ[i, j, 1] * Jᶿ + cl.b_qᵛ[i, j, 1] * Jᵛ) / ρΔz
        Pˢ = -(u[i, j, 1] * 𝒯ˣ + v[i, j, 1] * 𝒯ʸ) / ρΔz
    end
    bottom = k == 1
    return ifelse(bottom, Bˢ, zero(FT)), ifelse(bottom, Pˢ, zero(FT))
end

@kernel function _compute_applied_flux_conversions!(cl, grid, closure, s, Kᶜ, Kᵘ, u, v, e, bottom_bcs, bc_args)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    B, P, Bᴸ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Bˢ, Pˢ = surface_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, u, v, bottom_bcs, bc_args)
    # The conversion of the heating budget flux (zero without one); its latent part is drawn from the internal energy with Bᴸ
    Gᑫ = store_heating_flux_conversions!(cl, i, j, k, grid, s.ρ)
    B = B + Bˢ + Bᴸ + Gᑫ
    @inbounds begin
        cl.B[i, j, k] = B
        cl.Bᴸ[i, j, k] = Bᴸ
        cl.P[i, j, k] = P + Pˢ
        cl.Bˢ[i, j, k] = Bˢ
        cl.Pˢ[i, j, k] = Pˢ
        cl.ωᴮ[i, j, k] = -min(0, B) / max(e[i, j, k], buoyancy_sink_floor_value(grid_float_type(grid)))
    end
end

# What the conversions read
function applied_flux_state(model)
    ρ = total_density(model.dynamics)
    return (ρ = ρ,
            ρθ = distribution_potential_temperature_density(model),
            ρqᵛᵉ = model.moisture_density,
            qᵛᵉ = AtmosphereModels.specific_prognostic_moisture(model),
            ρu = model.momentum.ρu,
            ρv = model.momentum.ρv,
            T = model.temperature,
            p = dynamics_pressure(model.dynamics),
            standard_pressure = standard_pressure(model.dynamics),
            exner_exponent = AtmosphereModels.liquid_ice_exner_exponent(model.formulation),
            dynamics = model.dynamics,
            formulation = model.formulation,
            microphysics = model.microphysics,
            microphysical_fields = model.microphysical_fields,
            constants = model.thermodynamic_constants)
end

function applied_flux_bottom_bcs(model)
    ρθ = distribution_potential_temperature_density(model)
    return (ρθ = ρθ.boundary_conditions.bottom,
            ρqᵛᵉ = model.moisture_density.boundary_conditions.bottom,
            ρu = model.momentum.ρu.boundary_conditions.bottom,
            ρv = model.momentum.ρv.boundary_conditions.bottom)
end

compute_applied_flux_conversions!(fields, closure_fields, closure, model, tracers) = nothing

# Within `update_state!`, where the tracers hold their specific values
function compute_applied_flux_conversions!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, closure, model, tracers)
    grid = model.grid
    arch = grid.architecture
    cl = fields.cloud
    s = applied_flux_state(model)
    u, v = model.velocities.u, model.velocities.v
    launch!(arch, grid, :xyz, _compute_applied_flux_conversions!, cl, grid, closure, s, closure_fields.Kᶜ,
            closure_fields.Kᵘ, u, v, tracers[TKE_NAME], applied_flux_bottom_bcs(model), boundary_condition_args(model))
    return nothing
end

# The shear production of the TKE equation: CATKE's without the cloud buoyancy flux, the
# conversion of the applied momentum fluxes with it
@inline tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields, closure_fields, u, v) =
    shear_production(i, j, k, grid, closure_fields.Kᵘ, u, u, v, v)

@inline tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, u, v) =
    @inbounds fields.cloud.P[i, j, k]

#####
##### The floor of e in the buoyancy sink rate
#####

# The implicit buoyancy sink is ωᴮ e with ωᴮ = -B⁻ / max(e, floor). Above the floor it removes
# exactly -B⁻, the energy the applied fluxes give the mean state. The plain closure floors at
# `minimum_tke`, so below it the sink removes less than the mean state gains. With the cloud
# buoyancy flux the floor is a tiny positive number instead, kept only so that ωᴮ is finite at
# e = 0 (its benefit: the implicit sink cannot divide by zero, and e stays non-negative in the
# implicit solve); the conversion is exact wherever e exceeds it. The floor is taken in the float
# type `FT` of the grid, where ωᴮ is stored: a floor in the type of closure parameters wider than the
# grid's (Float64 parameters on a Float32 grid) would underflow there and make ωᴮ infinite.
@inline buoyancy_sink_floor(fields, closure, FT) = closure.minimum_tke
@inline buoyancy_sink_floor(::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, FT) = buoyancy_sink_floor_value(FT)
@inline buoyancy_sink_floor_value(FT) = sqrt(floatmin(FT))

#####
##### Dissipative heating and the conversions of the solved state
#####
##### Under the vertically implicit time discretization each Runge–Kutta stage solves
##### (1 - Δt ∂z K ∂z) φ = φ⁻ for momentum and every scalar (Δt the stage's α Δt), and the TKE
##### equation solves its sinks ωᴰ e + ωᴮ e (dissipation, damping of negative e, buoyancy sink)
##### implicitly too. The energy the solves take from the mean state is that of the fluxes of the
##### SOLVED state:
#####
#####   momentum:  Σ ρ Δz (|u|² - |u⁻|²)/2 = -Δt Σ ρᶠ Δzᶠ Kᵘ |δz u|² / Δzᶠ² - Σ ρ Δz |u - u⁻|² / 2
#####              (the last term is the dissipation of backward Euler itself),
#####   scalars:   Σ ρ Δz Δh‡ = -Δt Σ ρᶠ Δzᶠ Bᶠ(φ),  h‡ linearized with the stage's b_χ,
#####
##### while the stage tendency gave the TKE the conversions of the stage state. After the solves the
##### closure adds the difference to ρe, so that over the stage the TKE gains exactly what the mean
##### state loses; and before the thermodynamic solve it heats the air by what the implicit sinks
##### removed from the solved e, ρ ĉᵖ Π Δθˡⁱ = Δt ρ ωᴰ e (ωᴰ = -Lᵉ - ωᴮ).
#####

const ImplicitCloudBuoyancyFields = SubgridCloudFields{<:AppliedFluxEnergetics}

implicit_stage(closure, model) =
    time_discretization(closure) isa VerticallyImplicitTimeDiscretization && !isnothing(model.timestepper.implicit_solver)

implicit_stage(closure::TKEClosureArray, model) = implicit_stage(first(closure), model)

# The kinetic energy the convex combination of a Runge–Kutta stage removes from the mean flow. The stage
# forms u = (1 - α) u⁰ + α V, V = u⁽ᵐ⁻¹⁾ + Δt G, and per unit volume
#
#     (1 - α) ρ |u⁰|² / 2 + α ρ |V|² / 2 - ρ |(1 - α) u⁰ + α V|² / 2 = α (1 - α) ρ |V - u⁰|² / 2 ≥ 0,
#
# which reaches no budget unless it is given to the TKE: with it the TKE receives over a whole step exactly
# the kinetic energy the mean flow loses. Of the horizontal momentum, its squares interpolated to the centers.
@inline function squared_combination_increment(i, j, k, grid, ρu, Gρu, ρu⁰, Δt)
    @inbounds δ = ρu[i, j, k] + Δt * Gρu[i, j, k] - ρu⁰[i, j, k]
    return δ * δ
end

@kernel function _stage_combination_kinetic_energy!(K, grid, ρu, ρv, Gρu, Gρv, ρu⁰, ρv⁰, ρ, Δt, α)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        δ² = ℑxᶜᵃᵃ(i, j, k, grid, squared_combination_increment, ρu, Gρu, ρu⁰, Δt) +
             ℑyᵃᶜᵃ(i, j, k, grid, squared_combination_increment, ρv, Gρv, ρv⁰, Δt)
        K[i, j, k] = α * (1 - α) * δ² / (2 * ρ[i, j, k]) * !inactive_cell(i, j, k, grid)
    end
end

function AtmosphereModels.closure_before_stage_combination!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitCloudBuoyancyFields},
                                                             closure::FlavorOfTKEClosure, model, Δt, α)
    implicit_stage(closure, model) || return nothing
    grid = model.grid
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    ρu, ρv = model.momentum.ρu, model.momentum.ρv
    fill_halo_regions!((ρu, ρv, Gⁿ.ρu, Gⁿ.ρv, U⁰.ρu, U⁰.ρv))
    launch!(grid.architecture, grid, :xyz, _stage_combination_kinetic_energy!, closure_fields.subgrid_cloud.cloud.stage_combination,
            grid, ρu, ρv, Gⁿ.ρu, Gⁿ.ρv, U⁰.ρu, U⁰.ρv, total_density(model.dynamics), Δt, α)
    return nothing
end

#####
##### The stages of a multirate infinitesimal step
#####
##### Stage i recombines the states and the slow tendencies of the stages before it (see
##### `AtmosphereModels.StageRecombination`): the state its solves start from is
#####
#####     Vᵢ = Σⱼ wᵢⱼ Yⱼ + dᵢ Δt Σⱼ bᵢⱼ Sⱼ,     Σⱼ wᵢⱼ = 1,   Σⱼ bᵢⱼ = 1,   Sⱼ = S(Yⱼ₋₁),
#####
##### and the solves over dᵢ Δt give Yᵢ. Two things differ from a Shu–Osher stage.
#####
##### The kinetic energy the recombination of the states removes from the mean flow, per unit volume,
#####
#####     Σⱼ wᵢⱼ ρ |uⱼ|² / 2 - ρ |Σⱼ wᵢⱼ uⱼ|² / 2 = Σ_{p<q} wᵢₚ wᵢq ρ |uₚ - u_q|² / 2,
#####
##### is given to the TKE (`stage_combination`). The identity holds for any weights that sum to
##### one. It is the Shu–Osher α (1 - α) ρ |V - u⁰|² / 2 for the weights (1 - α, α). The weights of
##### a multirate infinitesimal step need not be positive (stage 2 of `:SSPMIS43` extrapolates,
##### w = (-0.05, 1.05)), and then the recombination adds kinetic energy to the mean flow, which
##### the TKE pays (or, where it has none, the internal energy: the shortfall).
#####
##### The stage-state conversions are in the slow tendencies, so the stage reapplies those of the
##### stages before it: over the stage the TKE tendency gives Δt Σⱼ βᵢⱼ ρ (Pⱼ + B⁺ⱼ) and the
##### tendency of θˡⁱ draws Δt Σⱼ βᵢⱼ ρ Bᴸⱼ. The correction after the solves therefore uses
#####
#####     Xᵢ = Σⱼ bᵢⱼ xⱼ,     xⱼ = (Pˢ + Bˢ + Bᴸ + Gᑫ - P - B⁺)ⱼ    (Gᑫ of a `HeatingBudgetFlux`),
#####
##### the recombined difference between what the stage-state fluxes converted and what the TKE
##### tendency gave, in place of the stage's own xᵢ: ρe gains dᵢ Δt ρ (Pᵖ + Dᵘ + Bᵖ + ωᴮ e + Xᵢ)
##### and the kinetic energy of the recombination. Over the stage the TKE then receives exactly
##### what the solves converted, what the recombined tendencies of the surface fluxes and the
##### latent conversion carry, and what the recombination removed from the mean flow.
#####

function AtmosphereModels.build_closure_stage_storage(::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitCloudBuoyancyFields},
                                                      closure, grid, stages)
    exchange = ntuple(_ -> CenterField(grid), stages)
    momentum = ntuple(_ -> (ρu = XFaceField(grid), ρv = YFaceField(grid)), stages - 1)
    return (; exchange, recombined_exchange = CenterField(grid), momentum)
end

@kernel function _store_stage_exchange!(x, cl)
    i, j, k = @index(Global, NTuple)
    # the conversion of the heating budget flux is a stage-state conversion too: its flux is in the slow tendency of ρθ
    @inbounds x[i, j, k] = cl.Pˢ[i, j, k] + cl.Bˢ[i, j, k] + cl.Bᴸ[i, j, k] + heating_flux_conversion(cl, i, j, k) -
                           cl.P[i, j, k] - max(cl.B[i, j, k], 0)
end

@inline weighted_sum(i, j, k, x::Tuple{Any}, b::Tuple{Any}) = @inbounds b[1] * x[1][i, j, k]
@inline weighted_sum(i, j, k, x::Tuple, b::Tuple) =
    @inbounds b[1] * x[1][i, j, k] + weighted_sum(i, j, k, Base.tail(x), Base.tail(b))

@kernel function _recombine_stage_exchange!(X, x, b)
    i, j, k = @index(Global, NTuple)
    @inbounds X[i, j, k] = weighted_sum(i, j, k, x, b)
end

# Σ_{q} w_q |u - u_q|² over the states `U` and Σ_{p<q} w_p w_q |u_p - u_q|², at the location of the momentum
@inline weighted_squared_differences(i, j, k, u, ::Tuple{}, ::Tuple{}) = zero(eltype(u))
@inline function weighted_squared_differences(i, j, k, u, U::Tuple, w::Tuple)
    @inbounds δ = u[i, j, k] - U[1][i, j, k]
    return w[1] * δ * δ + weighted_squared_differences(i, j, k, u, Base.tail(U), Base.tail(w))
end

@inline pairwise_squared_differences(i, j, k, grid, U::Tuple{Any}, w::Tuple{Any}) = zero(grid)
@inline pairwise_squared_differences(i, j, k, grid, U::Tuple, w::Tuple) =
    w[1] * weighted_squared_differences(i, j, k, U[1], Base.tail(U), Base.tail(w)) +
    pairwise_squared_differences(i, j, k, grid, Base.tail(U), Base.tail(w))

@kernel function _stage_recombination_kinetic_energy!(K, grid, ρu, ρv, w, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        δ² = ℑxᶜᵃᵃ(i, j, k, grid, pairwise_squared_differences, ρu, w) +
             ℑyᵃᶜᵃ(i, j, k, grid, pairwise_squared_differences, ρv, w)
        K[i, j, k] = δ² / (2 * ρ[i, j, k]) * !inactive_cell(i, j, k, grid)
    end
end

function AtmosphereModels.closure_before_stage_combination!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitCloudBuoyancyFields},
                                                             closure::FlavorOfTKEClosure, model, Δt,
                                                             recombination::AtmosphereModels.StageRecombination)
    implicit_stage(closure, model) || return nothing
    grid = model.grid
    arch = grid.architecture
    cl = closure_fields.subgrid_cloud.cloud
    storage = recombination.storage
    i = recombination.stage

    # The conversions of the stage state Yᵢ₋₁ are in hand: keep their exchange, and recombine
    launch!(arch, grid, :xyz, _store_stage_exchange!, storage.exchange[i], cl)
    launch!(arch, grid, :xyz, _recombine_stage_exchange!, storage.recombined_exchange,
            storage.exchange[1:i], recombination.tendency_weights)

    # The momentum is that of Yᵢ₋₁: keep it for the stages after this one
    ρu, ρv = model.momentum.ρu, model.momentum.ρv
    ρuⁿ, ρvⁿ = recombination.step_start_state.ρu, recombination.step_start_state.ρv
    fill_halo_regions!((ρu, ρv, ρuⁿ, ρvⁿ))
    if i > 1
        parent(storage.momentum[i-1].ρu) .= parent(ρu)
        parent(storage.momentum[i-1].ρv) .= parent(ρv)
    end
    ρus = (ρuⁿ, ntuple(j -> storage.momentum[j].ρu, i - 1)...)
    ρvs = (ρvⁿ, ntuple(j -> storage.momentum[j].ρv, i - 1)...)
    launch!(arch, grid, :xyz, _stage_recombination_kinetic_energy!, cl.stage_combination, grid, ρus, ρvs,
            recombination.state_weights, total_density(model.dynamics))
    return nothing
end

# What the correction after the solves adds to the stage tendency's conversions, per unit mass and
# time: with no recombination, what the solves and the stage-state fluxes converted less what the
# stage gave; with one, the solves' part and the recombined exchange of the stage states.
@inline net_stage_conversion(::Nothing, i, j, k, converted, given, solved) = converted - given
@inline net_stage_conversion(X, i, j, k, converted, given, solved) = @inbounds solved + X[i, j, k]

recombined_stage_exchange(::Nothing) = nothing
recombined_stage_exchange(storage) = storage.recombined_exchange

function AtmosphereModels.closure_before_implicit_solve!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitCloudBuoyancyFields},
                                                          closure::FlavorOfTKEClosure, model, name)
    implicit_stage(closure, model) || return nothing
    cl = closure_fields.subgrid_cloud.cloud
    name === :ρu && (parent(cl.ρu⁻) .= parent(model.momentum.ρu))
    name === :ρv && (parent(cl.ρv⁻) .= parent(model.momentum.ρv))
    pre = closure_fields.subgrid_cloud.presolve
    name ∈ keys(pre) && (parent(pre[name]) .= parent(Oceananigans.prognostic_fields(model)[name]))
    return nothing
end

@kernel function _heat_by_implicit_dissipation!(ρθ, cl, grid, Lᵉ, ρe, ρ, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        e = ρe[i, j, k] / ρᵢ
        ε = (-Lᵉ[i, j, k] - cl.ωᴮ[i, j, k]) * e
        cl.ε[i, j, k] = ε
        ρθ[i, j, k] += Δt * ρᵢ * ε / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
    end
end

function AtmosphereModels.closure_before_thermodynamic_solve!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitCloudBuoyancyFields},
                                                               closure::FlavorOfTKEClosure, model, Δt)
    implicit_stage(closure, model) || return nothing
    fields = closure_fields.subgrid_cloud
    grid = model.grid
    ρθ = distribution_potential_temperature_density(model)
    launch!(grid.architecture, grid, :xyz, _heat_by_implicit_dissipation!, ρθ,
            fields.cloud, grid, closure_fields.Lᵉ, model.tracers[TKE_NAME], total_density(model.dynamics), Δt)
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    name ∈ keys(fields.presolve) && (parent(fields.presolve[name]) .= parent(ρθ))
    return nothing
end

@kernel function _correct_tke_to_solved_conversions!(ρe, ρθ, cl, grid, s, Kᶜ, Kᵘ, u, v, Δt, X)
    i, j, k = @index(Global, NTuple)
    Bᵖ, Pᵖ, _ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    @inbounds begin
        ρᵢ = s.ρ[i, j, k]
        e = ρe[i, j, k] / ρᵢ
        Δu = (s.ρu[i, j, k] - cl.ρu⁻[i, j, k]) / ρᵢ
        Δv = (s.ρv[i, j, k] - cl.ρv⁻[i, j, k]) / ρᵢ
        Dᵘ = (Δu * Δu + Δv * Δv) / (2 * Δt)
        B = cl.B[i, j, k]
        # what the stage gave: P and B⁺ explicitly, -ωᴮ e implicitly; what the solves converted:
        # Pᵖ + Dᵘ + Pˢ and Bᵖ + Bˢ
        given = cl.P[i, j, k] + max(B, 0) - cl.ωᴮ[i, j, k] * e
        # the latent part was drawn from the internal energy in the stage tendency (stage value)
        # (so was the conversion of the heating budget flux, an explicit flux of the stage state)
        Gᑫ = heating_flux_conversion(cl, i, j, k)
        converted = Pᵖ + Dᵘ + cl.Pˢ[i, j, k] + Bᵖ + cl.Bˢ[i, j, k] + cl.Bᴸ[i, j, k] + Gᑫ
        cl.Pᵖ[i, j, k] = Pᵖ + Dᵘ
        cl.Bᵖ[i, j, k] = Bᵖ
        # the stage's values, kept for the energy budget after `update_state!` recomputes the others
        cl.given[i, j, k] = cl.P[i, j, k] + max(B, 0)
        cl.stage_b_θ[i, j, k] = cl.b_θ[i, j, k]
        cl.stage_b_qᵛ[i, j, k] = cl.b_qᵛ[i, j, k]
        cl.stage_b_qˡ[i, j, k] = cl.b_qˡ[i, j, k]
        cl.stage_b_qⁱ[i, j, k] = cl.b_qⁱ[i, j, k]
        cl.stage_surface[i, j, k] = cl.Pˢ[i, j, k] + cl.Bˢ[i, j, k] + cl.Bᴸ[i, j, k] + Gᑫ
        # Where the TKE cannot pay what the solved fluxes converted (e ≈ 0 while the diffusivities,
        # floored at `minimum_tke`, still mix), the shortfall is taken from the internal energy
        # instead of driving e negative: the mixing of that floor is powered by the heat of the air.
        # Benefit: e stays non-negative and the total energy still closes. The TKE also receives the
        # kinetic energy the stage combination removed from the mean flow (`stage_combination`, per volume).
        active = !inactive_cell(i, j, k, grid)
        net = net_stage_conversion(X, i, j, k, converted, given, Pᵖ + Dᵘ + Bᵖ + cl.ωᴮ[i, j, k] * e)
        ρe⁺ = ρe[i, j, k] + (Δt * ρᵢ * net + cl.stage_combination[i, j, k]) * active
        ρe⁺′ = ifelse(ρe⁺ < 0, min(ρe[i, j, k], zero(ρe⁺)), ρe⁺)
        shortfall = ρe⁺′ - ρe⁺ # ≥ 0: energy the TKE did not pay
        ρe[i, j, k] = ρe⁺′
        cl.shortfall[i, j, k] = shortfall
        cl.shortfall_ρθ[i, j, k] = shortfall / cl.ĉᵖΠ[i, j, k]
    end
end

# The cooling by the shortfall, after all the conversions of the solved state are formed (they read
# θ at the neighbouring cells, so it cannot be applied in the same kernel)
@kernel function _cool_by_shortfall!(ρθ, cl)
    i, j, k = @index(Global, NTuple)
    @inbounds ρθ[i, j, k] -= cl.shortfall_ρθ[i, j, k]
end

function AtmosphereModels.closure_after_implicit_solves!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitCloudBuoyancyFields},
                                                          closure::FlavorOfTKEClosure, model, Δt)
    implicit_stage(closure, model) || return nothing
    grid = model.grid
    cl = closure_fields.subgrid_cloud.cloud
    s = applied_flux_state(model)
    ρ = s.ρ
    fill_halo_regions!((model.momentum.ρu, model.momentum.ρv))
    u = SpecificValue(model.momentum.ρu, ρ)
    v = SpecificValue(model.momentum.ρv, ρ)
    launch!(grid.architecture, grid, :xyz, _correct_tke_to_solved_conversions!, model.tracers[TKE_NAME], s.ρθ, cl, grid, s,
            closure_fields.Kᶜ, closure_fields.Kᵘ, u, v, Δt,
            recombined_stage_exchange(AtmosphereModels.closure_stage_storage(model.timestepper)))
    launch!(grid.architecture, grid, :xyz, _cool_by_shortfall!, s.ρθ, cl)
    return nothing
end

# Under the explicit time discretization the sinks are in the stage tendency, and so is the heating
@kernel function _add_dissipative_heating!(Gρθ, grid, closure, closure_fields, ρe, ρ)
    i, j, k = @index(Global, NTuple)
    cl = closure_fields.subgrid_cloud.cloud
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        e = ρe[i, j, k] / ρᵢ
        ε = (-closure_fields.Lᵉ[i, j, k] - cl.ωᴮ[i, j, k]) * e
        Gρθ[i, j, k] += ρᵢ * ε / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
    end
end

add_dissipative_heating!(Gⁿ, fields, closure_fields, closure, model) = nothing

function add_dissipative_heating!(Gⁿ, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, closure, model)
    implicit_stage(closure, model) && return nothing
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _add_dissipative_heating!, Gⁿ[name], grid, closure, closure_fields,
            model.tracers[TKE_NAME], total_density(model.dynamics))
    return nothing
end

#####
##### The latent correction of the static stability and of the buoyancy production
#####
##### A displacement that condenses as it goes feels the saturated stratification N²ˢ; one
##### without phase change feels N²ᵇ = Σ b̄_χ δz χ. With (θˡⁱ, qᵗ, qᶜ) for (θˡⁱ, vapor, cloud
##### condensate), a saturated displacement keeps s = 0 with its condensate following the
##### saturation excess s̄ at fixed pressure, so
#####
#####   N²ˢ - N²ᵇ = (b̄ˡ - b̄ᵛ) (∂z s̄ - ∂z qᶜ),   ∂z s̄ = c̄_q ∂z qʷ - c̄_θ ∂z θˡⁱ
#####
##### (c_q, c_θ: the linearized saturation excess of the level with the model's thermodynamics).
##### The turbulence carries the condensate flux
#####
#####   w′qᶜ′ = η aʷ w′s′ + (1 - η) (-Kᶜ ∂z qᶜ),    w′s′ = -Kᶜ ∂z s̄,    η(x) = 1 - (1 - e⁻ˣ) / x,
#####
##### aʷ the share of the flux of the saturation excess carried by the supersaturated part of the
##### distribution, w′qˡ′ / w′s′ of the double Gaussian (its fluxes share one Kᶜ, which cancels),
##### and η the fraction of the excess that condenses over an eddy time tₑ = e / ε when it relaxes at
##### the rate ωᶜ (η = 1 for instantaneous condensation): the share η of the condensate follows the
##### saturation excess of the air that carries it, and only the share 1 - η, which does not change
##### phase within an eddy time, is mixed as a passive scalar. The closure applies -Kᶜ ∂z qᶜ to the
##### prognostic condensate, so the static stability and the buoyancy production use
#####
#####   N² = N²ᵇ - ΔN²ᴸ,    ΔN²ᴸ = η (b̄ˡ - b̄ᵛ) (∂z qᶜ - aʷ ∂z s̄)
#####      = N²ᶜ + (b̄ˡ - b̄ᵛ) [η aʷ ∂z s̄ + (1 - η) ∂z qᶜ],    N²ᶜ = N²ᵇ - (b̄ˡ - b̄ᵛ) ∂z qᶜ,
#####
##### N²ᶜ the stratification of the conserved variables (θˡⁱ, qʷ). With instantaneous condensation this
##### is (1 - aʷ) N²ᶜ + aʷ N²ˢ: the unsaturated share of the distribution holds no condensate and feels
##### the stratification of the conserved variables, not that of the mean condensate. (Weighting N²ᵇ
##### and N²ˢ instead, N²ᵇ - η aʷ (N²ᵇ - N²ˢ), leaves (1 - aʷ)(b̄ˡ - b̄ᵛ) ∂z qᶜ > 0 wherever the mean
##### condensate rises with height in partly cloudy air: a stable stratification at the base of a
##### well-mixed cloud layer and a buoyancy flux of the wrong sign there; harness notes/total-energy-design.md §11.)
##### The buoyancy production gains Bᴸ = Kᶜ ΔN²ᴸ over the conversion of the applied fluxes; that energy
##### is the work of the latent heat of the subgrid condensation and is taken from the internal energy.
##### aᴺ = η aʷ is stored: the share of the flux of the saturation excess that condenses.
#####

# The rate ωᶜ (s⁻¹) at which the condensate of the cell absorbs vapor excess, from the microphysics
# (`AtmosphereModels.phase_relaxation_rate`; 0 without condensing microphysics). It includes the
# psychrometric correction ξ of the latent heating (P3: ωˡ = 4π Dᵛ Nᶜˡ r̄ / ξˡ), so x = tₑ ωᶜ.
@inline function cell_phase_relaxation_rate(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        qᵛᵉ = s.qᵛᵉ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)
    𝒰 = AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid, s.formulation, s.dynamics, q)
    return AtmosphereModels.phase_relaxation_rate(i, j, k, grid, s.microphysics, s.microphysical_fields, 𝒰, s.constants)
end

"""
    condensing_fraction(x)

The share ``η(x) = 1 - (1 - e^{-x}) / x`` of the saturated response that a parcel displaced for an
eddy time ``tₑ`` feels when its condensate relaxes the vapor excess at the rate ``ωᶜ``, with
``x = tₑ ωᶜ``. Below ``x = 10⁻³`` it is the series ``x/2 - x²/6 + x³/24`` (that of ``1 - (1 - x/2 + x²/6 - x³/24)``) (``1 - (1 - e^{-x})/x`` loses
its digits to cancellation there), so that the value and its derivative are accurate and finite
down to ``x = 0``, where ``η = 0``.
"""
@inline function condensing_fraction(x)
    FT = typeof(x)
    small = x < convert(FT, 1e-3)
    xˢ = ifelse(small, one(FT), x)
    return ifelse(small, x * (1 / convert(FT, 2) - x * (1 / convert(FT, 6) - x / 24)), 1 + expm1(-xˢ) / xˢ)
end

# η at the face k: 1 for a microphysics that removes the vapor excess instantly (saturation
# adjustment), without forming tₑ ωᶜ = ∞; otherwise η(tₑ ωᶜ) with ωᶜ interpolated to the face
@inline saturated_response_shareᶜᶜᶠ(::Val{true}, tₑ, i, j, k, grid, s) = one(tₑ)
@inline saturated_response_shareᶜᶜᶠ(::Val{false}, tₑ, i, j, k, grid, s) =
    condensing_fraction(tₑ * (cell_phase_relaxation_rate(i, j, k, grid, s) + cell_phase_relaxation_rate(i, j, k-1, grid, s)) / 2)

# aʷ = w′qˡ′ / w′s′ at the center k: the condensate flux of the distribution there over the flux of the
# saturation excess s = c_q qᵗ - c_θ θˡⁱ formed from the face moments interpolated to the center, which
# are those the distribution at the center was built from. A ratio of collocated quantities (the
# distribution's own (qˡ₁ - qˡ₂) / (s₁ - s₂)), so the diffusivity cancels; kept in [0, 1].
@inline function center_condensate_flux_fraction(i, j, k, m, cl, c_q, c_θ)
    FT = typeof(c_q)
    @inbounds begin
        w′qᵗ′ = (m.w′qᵗ′[i, j, k] + m.w′qᵗ′[i, j, k+1]) / 2
        w′θˡⁱ′ = (m.w′θˡⁱ′[i, j, k] + m.w′θˡⁱ′[i, j, k+1]) / 2
        w′qˡ′ = cl.w′qˡ′[i, j, k]
    end
    w′s′ = c_q * w′qᵗ′ - c_θ * w′θˡⁱ′
    resolved = abs(w′s′) > sqrt(floatmin(FT))
    return clamp(ifelse(resolved, w′qˡ′ / ifelse(resolved, w′s′, one(FT)), zero(FT)), zero(FT), one(FT))
end

@kernel function _latent_corrected_stability!(N², cl, m, grid, closure, closure_fields, velocities, s, qʷ, e, thermodynamics, sat, c, ω)
    i, j, k = @index(Global, NTuple)
    FT = grid_float_type(grid)
    closure_ij = getclosure(i, j, closure)
    kₛ = max(k, 2)
    @inbounds begin
        ρ⁺ = s.ρ[i, j, kₛ]
        ρ⁻ = s.ρ[i, j, kₛ-1]
        θ⁺ = s.ρθ[i, j, kₛ] / ρ⁺
        θ⁻ = s.ρθ[i, j, kₛ-1] / ρ⁻
        qʷ⁺ = qʷ[i, j, kₛ]
        qʷ⁻ = qʷ[i, j, kₛ-1]
        qᶜ⁺ = qʷ⁺ - s.ρqᵛᵉ[i, j, kₛ] / ρ⁺
        qᶜ⁻ = qʷ⁻ - s.ρqᵛᵉ[i, j, kₛ-1] / ρ⁻
        Δz = Δzᶜᶜᶠ(i, j, kₛ, grid)
        c_q⁺, c_θ⁺, _ = saturation_excess_coefficients(sat, thermodynamics, c, θ⁺, qʷ⁺, s.p[i, j, kₛ])
        c_q⁻, c_θ⁻, _ = saturation_excess_coefficients(sat, thermodynamics, c, θ⁻, qʷ⁻, s.p[i, j, kₛ-1])
        c_q = (c_q⁺ + c_q⁻) / 2
        c_θ = (c_θ⁺ + c_θ⁻) / 2
        ∂z_s = (c_q * (qʷ⁺ - qʷ⁻) - c_θ * (θ⁺ - θ⁻)) / Δz
        ∂z_qᶜ = (qᶜ⁺ - qᶜ⁻) / Δz
        bˡᵛ = (cl.b_qˡ[i, j, kₛ] + cl.b_qˡ[i, j, kₛ-1] - cl.b_qᵛ[i, j, kₛ] - cl.b_qᵛ[i, j, kₛ-1]) / 2
        N²ᵇ = applied_stratificationᶜᶜᶠ(i, j, kₛ, grid, s, cl)

        # aʷ = w′qˡ′ / w′s′ formed at the centers above and below, where numerator and denominator come
        # from the same moments, and averaged to the face
        aʷ = (center_condensate_flux_fraction(i, j, kₛ, m, cl, c_q⁺, c_θ⁺) +
              center_condensate_flux_fraction(i, j, kₛ-1, m, cl, c_q⁻, c_θ⁻)) / 2

        # η over the eddy time tₑ = e / ε = ℓ / (𝕊ᴰ √e) at the face
        ℓ = closure_fields.ℓᵐ[i, j, kₛ]
        u = ℑzᵃᵃᶠ(i, j, kₛ, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)
        𝕊ᴰ = dissipation_stability_functionᶜᶜᶠ(i, j, kₛ, grid, closure_ij, velocities, closure_fields.N², e, closure_fields.ℓᵐ)
        tₑ = ℓ / (𝕊ᴰ * u)
        instantaneous = Val(AtmosphereModels.instantaneous_phase_relaxation(s.microphysics))
        η = saturated_response_shareᶜᶜᶠ(instantaneous, tₑ, i, j, kₛ, grid, s)
        aᴺ = η * aʷ

        ΔN²ᴸ = η * bˡᵛ * (∂z_qᶜ - aʷ * ∂z_s)
        N²ᵉ = N²ᵇ - ΔN²ᴸ
        interior = k > 1
        cl.aʷ[i, j, k] = ifelse(interior, aʷ, zero(FT))
        cl.aᴺ[i, j, k] = ifelse(interior, aᴺ, zero(FT))
        cl.c_θ[i, j, k] = ifelse(interior, c_θ, zero(FT))
        cl.ΔN²ᴸ[i, j, k] = ifelse(interior, ΔN²ᴸ, zero(FT))
        N²[i, j, k] = ifelse(interior, relax_stability(N²[i, j, k], N²ᵉ, ω), N²[i, j, k])
    end
end

# The pass of the stability coupling with the cloud buoyancy flux: the latent-corrected N², then the
# mixing length and the diffusivities from it
function cloud_buoyancy_stability_pass!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, closure, model, tracers,
                                        parameters, ω = nothing)
    grid = model.grid
    arch = grid.architecture
    d = distribution_state(model, nothing)
    dist = subgrid_distribution(closure)
    launch!(arch, grid, parameters, _latent_corrected_stability!, closure_fields.N², fields.cloud, fields.moments, grid, closure,
            closure_fields, model.velocities, applied_flux_state(model), d.qᵗ, tracers[TKE_NAME], d.thermodynamics,
            dist.saturation, dist.constants, ω)
    compute_mixing_length_and_diffusivities!(closure_fields, closure, model, tracers, parameters)
    return nothing
end

# The latent part of the buoyancy production is drawn from the internal energy: ρ ĉᵖ Π ∂t θˡⁱ -= ρ Bᴸ
@kernel function _draw_latent_conversion!(Gρθ, grid, cl, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρθ[i, j, k] -= ρ[i, j, k] * cl.Bᴸ[i, j, k] / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
end

draw_latent_conversion!(Gⁿ, fields, model) = nothing

function draw_latent_conversion!(Gⁿ, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, model)
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _draw_latent_conversion!, Gⁿ[name], grid, fields.cloud, total_density(model.dynamics))
    return nothing
end

#####
##### Without a distribution: the condensate flux implied by the saturated fraction of the microphysics
#####
##### With `static_stability = SaturationWeightedStaticStability(fraction)` and no subgrid
##### distribution, the saturated fraction a of the fraction model (for example
##### `MicrophysicsSaturatedFraction()`, the subgrid model of the microphysics) implies the condensate
##### flux w′qᶜ′ = a w′s′, w′s′ = -Kᶜ ∂z s̄, where the closure applies -Kᶜ ∂z qᶜ. Its buoyancy
##### conversion beyond that of the applied fluxes is the latent part
#####
#####   Bᴸ = (b̄ˡ - b̄ᵛ) (a w′s′ + Kᶜ ∂z qᶜ) = Kᶜ (b̄ˡ - b̄ᵛ) (∂z qᶜ - a ∂z s̄),
#####
##### the same as with the distribution's aʷ and instantaneous condensation: it enters the buoyancy production, its energy is drawn
##### from the internal energy, and the energy budget of the stage covers it. The static stability
##### the fraction gives (for Ri and ℓ) and the buoyancy production then use one saturated fraction.
#####

@kernel function _microphysics_latent_correction!(cl, grid, closure, s, qʷ, thermodynamics, sat, c, tracers, buoyancy, closure_fields)
    i, j, k = @index(Global, NTuple)
    FT = grid_float_type(grid)
    closure_ij = getclosure(i, j, closure)
    kₛ = max(k, 2)
    @inbounds begin
        ρ⁺ = s.ρ[i, j, kₛ]
        ρ⁻ = s.ρ[i, j, kₛ-1]
        θ⁺ = s.ρθ[i, j, kₛ] / ρ⁺
        θ⁻ = s.ρθ[i, j, kₛ-1] / ρ⁻
        qʷ⁺ = qʷ[i, j, kₛ]
        qʷ⁻ = qʷ[i, j, kₛ-1]
        qᶜ⁺ = qʷ⁺ - s.ρqᵛᵉ[i, j, kₛ] / ρ⁺
        qᶜ⁻ = qʷ⁻ - s.ρqᵛᵉ[i, j, kₛ-1] / ρ⁻
        Δz = Δzᶜᶜᶠ(i, j, kₛ, grid)
        c_q⁺, c_θ⁺, _ = saturation_excess_coefficients(sat, thermodynamics, c, θ⁺, qʷ⁺, s.p[i, j, kₛ])
        c_q⁻, c_θ⁻, _ = saturation_excess_coefficients(sat, thermodynamics, c, θ⁻, qʷ⁻, s.p[i, j, kₛ-1])
        c_θ = (c_θ⁺ + c_θ⁻) / 2
        ∂z_s = ((c_q⁺ + c_q⁻) / 2 * (qʷ⁺ - qʷ⁻) - c_θ * (θ⁺ - θ⁻)) / Δz
        ∂z_qᶜ = (qᶜ⁺ - qᶜ⁻) / Δz
        bˡᵛ = (cl.b_qˡ[i, j, kₛ] + cl.b_qˡ[i, j, kₛ-1] - cl.b_qᵛ[i, j, kₛ] - cl.b_qᵛ[i, j, kₛ-1]) / 2
        a = saturated_stability_weightᶜᶜᶠ(i, j, kₛ, grid, closure_ij.static_stability.fraction, buoyancy, tracers, closure_fields)
        ΔN²ᴸ = bˡᵛ * (∂z_qᶜ - a * ∂z_s)
        interior = k > 1
        cl.aʷ[i, j, k] = ifelse(interior, a, zero(FT))
        cl.aᴺ[i, j, k] = ifelse(interior, a, zero(FT))
        cl.c_θ[i, j, k] = ifelse(interior, c_θ, zero(FT))
        cl.ΔN²ᴸ[i, j, k] = ifelse(interior, ΔN²ᴸ, zero(FT))
    end
end

function compute_subgrid_cloud!(fields::SubgridCloudFields{<:AppliedFluxEnergetics, Nothing}, closure_fields, closure, model, tracers, parameters)
    compute_buoyancy_derivatives!(fields, model)
    compute_heating_budget_flux!(fields, heating_budget_flux(closure), closure_fields, closure, model, tracers)
    grid = model.grid
    d = distribution_state(model, nothing)
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)
    launch!(grid.architecture, grid, parameters, _microphysics_latent_correction!, fields.cloud, grid, closure,
            applied_flux_state(model), d.qᵗ, d.thermodynamics, ModelThermodynamicsSaturation(), nothing,
            tracers, buoyancy, closure_fields)
    return nothing
end
