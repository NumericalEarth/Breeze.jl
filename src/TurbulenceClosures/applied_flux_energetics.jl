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
using Oceananigans.Models: boundary_condition_args

using .SimplifiedHigherOrderClosures: bottom_boundary_flux

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

# The derivatives of the buoyancy and the enthalpy coefficient d(cᵖᵐ T)/dθˡⁱ = ĉᵖ Π at the centers,
# stored once per update so that the conversions after the implicit solves use the same ones
@kernel function _compute_buoyancy_derivatives!(cl, grid, s)
    i, j, k = @index(Global, NTuple)
    βᶿ, βᵛ, βˡ, βⁱ, θ, q = buoyancy_derivativesᶜᶜᶜ(i, j, k, grid, s)
    constants = s.constants
    κ = exner_exponent(s.exner_exponent, q, constants)
    Π = (@inbounds(s.p[i, j, k]) / s.standard_pressure)^κ
    qᵗ = q.vapor + q.liquid + q.ice
    ĉᵖ = constants.dry_air.heat_capacity + qᵗ * (constants.vapor.heat_capacity - constants.dry_air.heat_capacity)
    @inbounds begin
        cl.βᶿ[i, j, k] = βᶿ
        cl.βᵛ[i, j, k] = βᵛ
        cl.βˡ[i, j, k] = βˡ
        cl.βⁱ[i, j, k] = βⁱ
        cl.ĉᵖΠ[i, j, k] = ĉᵖ * Π
    end
end

# Bᶠ at the interior face k from the stored derivatives and the current state
@inline function applied_buoyancy_conversionᶜᶜᶠ(i, j, k, grid, s, cl, Kᶜ)
    @inbounds begin
        ρ⁺ = s.ρ[i, j, k]
        ρ⁻ = s.ρ[i, j, k-1]
        θ⁺ = s.ρθ[i, j, k] / ρ⁺
        θ⁻ = s.ρθ[i, j, k-1] / ρ⁻
        q⁺ = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ⁺, s.ρqᵛᵉ[i, j, k] / ρ⁺, s.microphysical_fields)
        q⁻ = grid_moisture_fractions(i, j, k-1, grid, s.microphysics, ρ⁻, s.ρqᵛᵉ[i, j, k-1] / ρ⁻, s.microphysical_fields)
        Δb = (cl.βᶿ[i, j, k] + cl.βᶿ[i, j, k-1]) / 2 * (θ⁺ - θ⁻) +
             (cl.βᵛ[i, j, k] + cl.βᵛ[i, j, k-1]) / 2 * (q⁺.vapor - q⁻.vapor) +
             (cl.βˡ[i, j, k] + cl.βˡ[i, j, k-1]) / 2 * (q⁺.liquid - q⁻.liquid) +
             (cl.βⁱ[i, j, k] + cl.βⁱ[i, j, k-1]) / 2 * (q⁺.ice - q⁻.ice)
        return - Kᶜ[i, j, k] * Δb / Δzᶜᶜᶠ(i, j, k, grid)
    end
end

@inline applied_shear_conversionᶜᶜᶠ(i, j, k, grid, Kᵘ, u, v) = @inbounds Kᵘ[i, j, k] * shearᶜᶜᶠ(i, j, k, grid, u, v)

# The conversions of the interior fluxes at center k, from those at its two faces by half-cell
# weights (the boundary faces carry none: the closure's diffusivities vanish there)
@inline function interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Nz = size(grid, 3)
    FT = eltype(grid)
    kˡᵒʷ = max(k, 2)
    kᵘᵖ = min(k + 1, Nz)
    Bˡᵒʷ = ifelse(k > 1, applied_buoyancy_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, s, cl, Kᶜ), zero(FT))
    Bᵘᵖ = ifelse(k < Nz, applied_buoyancy_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, s, cl, Kᶜ), zero(FT))
    Pˡᵒʷ = ifelse(k > 1, applied_shear_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, Kᵘ, u, v), zero(FT))
    Pᵘᵖ = ifelse(k < Nz, applied_shear_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, Kᵘ, u, v), zero(FT))
    zᶜ = znode(i, j, k, grid, Center(), Center(), Center())
    wˡᵒʷ = ℑzᵃᵃᶠ(i, j, k, grid, s.ρ) * (zᶜ - znode(i, j, k, grid, Center(), Center(), Face()))
    wᵘᵖ = ℑzᵃᵃᶠ(i, j, k + 1, grid, s.ρ) * (znode(i, j, k + 1, grid, Center(), Center(), Face()) - zᶜ)
    ρΔz = @inbounds s.ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
    return (wˡᵒʷ * Bˡᵒʷ + wᵘᵖ * Bᵘᵖ) / ρΔz, (wˡᵒʷ * Pˡᵒʷ + wᵘᵖ * Pᵘᵖ) / ρΔz
end

# The conversions of the surface fluxes within the bottom half-cell: the work of the surface
# stress on the lowest level, -u₁ 𝒯ˣ - v₁ 𝒯ʸ (𝒯 the upward flux of ρu, ρv), and the dynamic
# enthalpy the surface fluxes of θˡⁱ and vapor remove, (z₁ - z_surface) Σ βₛ,₁ Jₛ, per unit mass
# of the lowest cell. Condensate has no surface flux here.
@inline function surface_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, u, v, bottom_bcs, bc_args)
    FT = eltype(grid)
    Jᶿ = bottom_boundary_flux(bottom_bcs.ρθ, i, j, grid, s.ρθ, bc_args...)
    Jᵛ = bottom_boundary_flux(bottom_bcs.ρqᵛᵉ, i, j, grid, s.ρqᵛᵉ, bc_args...)
    𝒯ˣ = bottom_boundary_flux(bottom_bcs.ρu, i, j, grid, s.ρu, bc_args...)
    𝒯ʸ = bottom_boundary_flux(bottom_bcs.ρv, i, j, grid, s.ρv, bc_args...)
    h = znode(i, j, 1, grid, Center(), Center(), Center()) - znode(i, j, 1, grid, Center(), Center(), Face())
    @inbounds begin
        ρΔz = s.ρ[i, j, 1] * Δzᶜᶜᶜ(i, j, 1, grid)
        Bˢ = h * (cl.βᶿ[i, j, 1] * Jᶿ + cl.βᵛ[i, j, 1] * Jᵛ) / ρΔz
        Pˢ = -(u[i, j, 1] * 𝒯ˣ + v[i, j, 1] * 𝒯ʸ) / ρΔz
    end
    bottom = k == 1
    return ifelse(bottom, Bˢ, zero(FT)), ifelse(bottom, Pˢ, zero(FT))
end

@kernel function _compute_applied_flux_conversions!(cl, grid, closure, s, Kᶜ, Kᵘ, u, v, e, bottom_bcs, bc_args)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    B, P = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Bˢ, Pˢ = surface_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, u, v, bottom_bcs, bc_args)
    B = B + Bˢ
    @inbounds begin
        cl.B[i, j, k] = B
        cl.P[i, j, k] = P + Pˢ
        cl.Bˢ[i, j, k] = Bˢ
        cl.Pˢ[i, j, k] = Pˢ
        cl.ωᴮ[i, j, k] = -min(0, B) / max(e[i, j, k], buoyancy_sink_floor_value(closure_ij))
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
function compute_applied_flux_conversions!(fields::SubgridCloudFields{<:CloudBuoyancyFlux}, closure_fields, closure, model, tracers)
    grid = model.grid
    arch = grid.architecture
    cl = fields.cloud
    s = applied_flux_state(model)
    launch!(arch, grid, :xyz, _compute_buoyancy_derivatives!, cl, grid, s)
    fill_halo_regions!((cl.βᶿ, cl.βᵛ, cl.βˡ, cl.βⁱ))
    u, v = model.velocities.u, model.velocities.v
    launch!(arch, grid, :xyz, _compute_applied_flux_conversions!, cl, grid, closure, s, closure_fields.Kᶜ,
            closure_fields.Kᵘ, u, v, tracers[TKE_NAME], applied_flux_bottom_bcs(model), boundary_condition_args(model))
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
@inline buoyancy_sink_floor(::SubgridCloudFields{<:CloudBuoyancyFlux}, closure) = buoyancy_sink_floor_value(closure)
@inline buoyancy_sink_floor_value(closure) = sqrt(floatmin(typeof(closure.minimum_tke)))

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
#####   scalars:   Σ ρ Δz Δh‡ = -Δt Σ ρᶠ Δzᶠ Bᶠ(φ),  h‡ linearized with the stage's βₛ,
#####
##### while the stage tendency gave the TKE the conversions of the stage state. After the solves the
##### closure adds the difference to ρe, so that over the stage the TKE gains exactly what the mean
##### state loses; and before the thermodynamic solve it heats the air by what the implicit sinks
##### removed from the solved e, ρ ĉᵖ Π Δθˡⁱ = Δt ρ ωᴰ e (ωᴰ = -Lᵉ - ωᴮ).
#####

const ImplicitCloudBuoyancyFields = SubgridCloudFields{<:CloudBuoyancyFlux}

implicit_stage(closure, model) =
    time_discretization(closure) isa VerticallyImplicitTimeDiscretization && !isnothing(model.timestepper.implicit_solver)

implicit_stage(closure::TKEClosureArray, model) = implicit_stage(first(closure), model)

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
    fields.buoyancy_flux.dissipative_heating || return nothing
    grid = model.grid
    ρθ = distribution_potential_temperature_density(model)
    launch!(grid.architecture, grid, :xyz, _heat_by_implicit_dissipation!, ρθ,
            fields.cloud, grid, closure_fields.Lᵉ, model.tracers[TKE_NAME], total_density(model.dynamics), Δt)
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    name ∈ keys(fields.presolve) && (parent(fields.presolve[name]) .= parent(ρθ))
    return nothing
end

@kernel function _correct_tke_to_solved_conversions!(ρe, ρθ, cl, grid, s, Kᶜ, Kᵘ, u, v, Δt)
    i, j, k = @index(Global, NTuple)
    Bᵖ, Pᵖ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
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
        converted = Pᵖ + Dᵘ + cl.Pˢ[i, j, k] + Bᵖ + cl.Bˢ[i, j, k]
        cl.Pᵖ[i, j, k] = Pᵖ + Dᵘ
        cl.Bᵖ[i, j, k] = Bᵖ
        # the stage's values, kept for the energy budget after `update_state!` recomputes the others
        cl.given[i, j, k] = cl.P[i, j, k] + max(B, 0)
        cl.stage_βᶿ[i, j, k] = cl.βᶿ[i, j, k]
        cl.stage_βᵛ[i, j, k] = cl.βᵛ[i, j, k]
        cl.stage_βˡ[i, j, k] = cl.βˡ[i, j, k]
        cl.stage_βⁱ[i, j, k] = cl.βⁱ[i, j, k]
        cl.stage_surface[i, j, k] = cl.Pˢ[i, j, k] + cl.Bˢ[i, j, k]
        # Where the TKE cannot pay what the solved fluxes converted (e ≈ 0 while the diffusivities,
        # floored at `minimum_tke`, still mix), the shortfall is taken from the internal energy
        # instead of driving e negative: the mixing of that floor is powered by the heat of the air.
        # Benefit: e stays non-negative and the total energy still closes.
        active = !inactive_cell(i, j, k, grid)
        ρe⁺ = ρe[i, j, k] + Δt * ρᵢ * (converted - given) * active
        ρe⁺′ = ifelse(ρe⁺ < 0, min(ρe[i, j, k], zero(ρe⁺)), ρe⁺)
        shortfall = ρe⁺′ - ρe⁺ # ≥ 0: energy the TKE did not pay
        ρe[i, j, k] = ρe⁺′
        ρθ[i, j, k] -= shortfall / cl.ĉᵖΠ[i, j, k]
        cl.shortfall[i, j, k] = shortfall
        cl.shortfall_ρθ[i, j, k] = shortfall / cl.ĉᵖΠ[i, j, k]
    end
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
            closure_fields.Kᶜ, closure_fields.Kᵘ, u, v, Δt)
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

function add_dissipative_heating!(Gⁿ, fields::SubgridCloudFields{<:CloudBuoyancyFlux}, closure_fields, closure, model)
    fields.buoyancy_flux.dissipative_heating || return nothing
    implicit_stage(closure, model) && return nothing
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _add_dissipative_heating!, Gⁿ[name], grid, closure, closure_fields,
            model.tracers[TKE_NAME], total_density(model.dynamics))
    return nothing
end
