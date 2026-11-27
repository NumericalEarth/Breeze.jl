using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ
using Oceananigans.Models: boundary_condition_args
using SpecialFunctions: erfc

using Oceananigans.BoundaryConditions: BoundaryCondition, Flux, Bottom, total_boundary_flux

@inline has_field(::NamedTuple{names}, ::Val{name}) where {names, name} = Val(name ∈ names)

@inline stored_or_zero(cl, name::Val, i, j, k, FT) = stored_or_zero(has_field(cl, name), cl, name, i, j, k, FT)
@inline stored_or_zero(::Val{true}, cl, ::Val{name}, i, j, k, FT) where name = @inbounds getproperty(cl, name)[i, j, k]
@inline stored_or_zero(::Val{false}, cl, name, i, j, k, FT) = zero(FT)

@inline bottom_boundary_flux(bc::BoundaryCondition{<:Flux}, i, j, grid, c, args...) =
    total_boundary_flux(bc, Bottom(), i, j, 1, grid, c, args...)

@inline bottom_boundary_flux(bc, i, j, grid, c, args...) = zero(grid)

using Breeze.Thermodynamics: mixture_heat_capacity, mixture_gas_constant, exner_exponent,
                             VaporEquivalentExponent, DryExponent

@inline exner_exponent_moisture_derivative(::DryExponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ) = zero(R̂)
@inline exner_exponent_moisture_derivative(::VaporEquivalentExponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ) =
    R̂ / ĉᵖ * ((Rᵛ - Rᵈ) / R̂ - (cᵖᵛ - cᵖᵈ) / ĉᵖ)

@inline function buoyancy_derivativesᶜᶜᶜ(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        θ = s.ρθ[i, j, k] / ρ
        qᵛᵉ = s.qᵛᵉ[i, j, k]
        p = s.p[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)

    g = s.constants.gravitational_acceleration
    b = AtmosphereModels.buoyancy_forceᶜᶜᶜ(i, j, k, grid, s.dynamics, s.T, s.qᵛᵉ,
                                          s.microphysics, s.microphysical_fields, s.constants) / ρ

    b_θ, b_qᵛ, b_qˡ, b_qⁱ = buoyancy_derivatives(θ, p, q, g - b, s.constants, s.exner_exponent, s.standard_pressure)
    return b_θ, b_qᵛ, b_qˡ, b_qⁱ, θ, q
end

@inline function buoyancy_derivatives(θ, p, q, G, constants, exner_exponent_formulation, pˢᵗ)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)

    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    R̂ = vapor_equivalent_gas_constant(q, constants)
    κ = exner_exponent(exner_exponent_formulation, q, constants)
    κ′ = exner_exponent_moisture_derivative(exner_exponent_formulation, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    ln_p = log(p / pˢᵗ)
    Π = (p / pˢᵗ)^κ

    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    H = ĉᵖ * Π * θ + ℒˡ₀ * q.liquid + ℒⁱ₀ * q.ice # = cᵖᵐ T
    ∂H∂qᵗ = θ * Π * ((cᵖᵛ - cᵖᵈ) + ĉᵖ * ln_p * κ′)

    b_θ = G * ĉᵖ * Π / H
    b_qᵛ = G * ((Rᵛ - Rᵈ) / Rᵐ + ∂H∂qᵗ / H - (cᵖᵛ - cᵖᵈ) / cᵖᵐ)
    b_qˡ = G * (-Rᵈ / Rᵐ + (∂H∂qᵗ + ℒˡ₀) / H - (cˡ - cᵖᵈ) / cᵖᵐ)
    b_qⁱ = G * (-Rᵈ / Rᵐ + (∂H∂qᵗ + ℒⁱ₀) / H - (cⁱ - cᵖᵈ) / cᵖᵐ)

    return b_θ, b_qᵛ, b_qˡ, b_qⁱ
end

@kernel function _compute_buoyancy_derivatives!(cl, grid, s)
    i, j, k = @index(Global, NTuple)
    b_θ, b_qᵛ, b_qˡ, b_qⁱ, θ, q = buoyancy_derivativesᶜᶜᶜ(i, j, k, grid, s)
    constants = s.constants
    κ = exner_exponent(s.exner_exponent, q, constants)
    Π = (@inbounds(s.p[i, j, k]) / s.standard_pressure)^κ
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    λ, _ = distributed_condensate_liquid_fraction(s.phase_equilibrium, @inbounds(s.T[i, j, k]))
    @inbounds begin
        cl.b_θ[i, j, k] = b_θ
        cl.b_qᵛ[i, j, k] = b_qᵛ
        cl.b_qˡ[i, j, k] = b_qˡ
        cl.b_qⁱ[i, j, k] = b_qⁱ
        cl.b_qᶜ[i, j, k] = λ * b_qˡ + (1 - λ) * b_qⁱ
        cl.ĉᵖΠ[i, j, k] = ĉᵖ * Π
    end
end

compute_buoyancy_derivatives!(fields, model) = nothing

function compute_buoyancy_derivatives!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, model)
    grid = model.grid
    cl = fields.cloud
    launch!(grid.architecture, grid, :xyz, _compute_buoyancy_derivatives!, cl, grid, applied_flux_state(model))
    fill_halo_regions!((cl.b_θ, cl.b_qᵛ, cl.b_qˡ, cl.b_qⁱ, cl.b_qᶜ))
    return nothing
end

@inline applied_buoyancy_conversionᶜᶜᶠ(i, j, k, grid, s, cl, Kᶜ) =
    - @inbounds(Kᶜ[i, j, k]) * applied_stratificationᶜᶜᶠ(i, j, k, grid, s, cl)

@inline function current_moisture_fractionsᶜᶜᶜ(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        qᵛᵉ = s.ρqᵛᵉ[i, j, k] / ρ
        qᵛᵉₛ = s.qᵛᵉ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉₛ, s.microphysical_fields)
    return MoistureMassFractions(q.vapor + (qᵛᵉ - qᵛᵉₛ), q.liquid, q.ice)
end

@inline function applied_stratificationᶜᶜᶠ(i, j, k, grid, s, cl)
    @inbounds begin
        ρ⁺ = s.ρ[i, j, k]
        ρ⁻ = s.ρ[i, j, k-1]
        θ⁺ = s.ρθ[i, j, k] / ρ⁺
        θ⁻ = s.ρθ[i, j, k-1] / ρ⁻
        q⁺ = current_moisture_fractionsᶜᶜᶜ(i, j, k, grid, s)
        q⁻ = current_moisture_fractionsᶜᶜᶜ(i, j, k-1, grid, s)
        Δb = (cl.b_θ[i, j, k] + cl.b_θ[i, j, k-1]) / 2 * (θ⁺ - θ⁻) +
             (cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, k-1]) / 2 * (q⁺.vapor - q⁻.vapor) +
             (cl.b_qˡ[i, j, k] + cl.b_qˡ[i, j, k-1]) / 2 * (q⁺.liquid - q⁻.liquid) +
             (cl.b_qⁱ[i, j, k] + cl.b_qⁱ[i, j, k-1]) / 2 * (q⁺.ice - q⁻.ice)
        return Δb / Δzᶜᶜᶠ(i, j, k, grid)
    end
end

@inline function half_cell_weights(i, j, k, grid, ρ)
    zᶜ = znode(i, j, k, grid, Center(), Center(), Center())
    wˡᵒʷ = ℑzᵃᵃᶠ(i, j, k, grid, ρ) * (zᶜ - znode(i, j, k, grid, Center(), Center(), Face()))
    wᵘᵖ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ) * (znode(i, j, k + 1, grid, Center(), Center(), Face()) - zᶜ)
    ρΔz = @inbounds ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
    return wˡᵒʷ, wᵘᵖ, ρΔz
end

@inline function half_cell_conversionᶜᶜᶜ(i, j, k, grid, ρ, lower, upper)
    wˡᵒʷ, wᵘᵖ, ρΔz = half_cell_weights(i, j, k, grid, ρ)
    wˡᵒʷ = ifelse(k > 1, wˡᵒʷ, zero(wˡᵒʷ))
    wᵘᵖ = ifelse(k < size(grid, 3), wᵘᵖ, zero(wᵘᵖ))
    return (wˡᵒʷ * lower + wᵘᵖ * upper) / ρΔz
end

@inline function interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    kˡᵒʷ = max(k, 2)
    kᵘᵖ = min(k + 1, Nz)
    Bˡᵒʷ = ifelse(k > 1, applied_buoyancy_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, s, cl, Kᶜ), zero(FT))
    Bᵘᵖ = ifelse(k < Nz, applied_buoyancy_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, s, cl, Kᶜ), zero(FT))
    Lˡᵒʷ = ifelse(k > 1, latent_conversionᶜᶜᶠ(i, j, kˡᵒʷ, grid, cl, Kᶜ), zero(FT))
    Lᵘᵖ = ifelse(k < Nz, latent_conversionᶜᶜᶠ(i, j, kᵘᵖ, grid, cl, Kᶜ), zero(FT))
    Pˡᵒʷ = ifelse(k > 1, shear_productionᶜᶜᶠ(i, j, kˡᵒʷ, grid, Kᵘ, u, v), zero(FT))
    Pᵘᵖ = ifelse(k < Nz, shear_productionᶜᶜᶠ(i, j, kᵘᵖ, grid, Kᵘ, u, v), zero(FT))
    wˡᵒʷ, wᵘᵖ, ρΔz = half_cell_weights(i, j, k, grid, s.ρ)
    return (wˡᵒʷ * Bˡᵒʷ + wᵘᵖ * Bᵘᵖ) / ρΔz, (wˡᵒʷ * Pˡᵒʷ + wᵘᵖ * Pᵘᵖ) / ρΔz, (wˡᵒʷ * Lˡᵒʷ + wᵘᵖ * Lᵘᵖ) / ρΔz
end

@inline latent_conversionᶜᶜᶠ(i, j, k, grid, cl, Kᶜ) = @inbounds Kᶜ[i, j, k] * cl.ΔN²ᴸ[i, j, k]

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

@kernel function _compute_applied_flux_conversions!(cl, grid, s, Kᶜ, Kᵘ, u, v, e, bottom_bcs, bc_args)
    i, j, k = @index(Global, NTuple)
    B, P, Bᴸ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Bˢ, Pˢ = surface_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, u, v, bottom_bcs, bc_args)
    Gᑫ = store_heating_flux_conversions!(cl, i, j, k, grid, s.ρ)
    B = B + Bˢ + Bᴸ + Gᑫ
    @inbounds begin
        cl.B[i, j, k] = B
        cl.Bᴸ[i, j, k] = Bᴸ
        cl.P[i, j, k] = P + Pˢ
        cl.Bˢ[i, j, k] = Bˢ
        cl.Pˢ[i, j, k] = Pˢ
        cl.ωᴮ[i, j, k] = -min(0, B) / applied_flux_buoyancy_sink_tke(i, j, k, grid, e)
    end
end

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
            constants = model.thermodynamic_constants,
            phase_equilibrium = condensate_phase_equilibrium(model.microphysics))
end

function applied_flux_bottom_bcs(model)
    ρθ = distribution_potential_temperature_density(model)
    return (ρθ = ρθ.boundary_conditions.bottom,
            ρqᵛᵉ = model.moisture_density.boundary_conditions.bottom,
            ρu = model.momentum.ρu.boundary_conditions.bottom,
            ρv = model.momentum.ρv.boundary_conditions.bottom)
end

compute_applied_flux_conversions!(fields, closure_fields, closure, model, tracers) = nothing

function compute_applied_flux_conversions!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, closure, model, tracers)
    grid = model.grid
    arch = grid.architecture
    cl = fields.cloud
    s = applied_flux_state(model)
    u, v = model.velocities.u, model.velocities.v
    launch!(arch, grid, :xyz, _compute_applied_flux_conversions!, cl, grid, s, closure_fields.Kᶜ,
            closure_fields.Kᵘ, u, v, tracers[TKE_NAME], applied_flux_bottom_bcs(model), boundary_condition_args(model))
    return nothing
end

@inline tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields, closure_fields, u, v) =
    shear_production(i, j, k, grid, closure_fields.Kᵘ, u, u, v, v)

@inline tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, u, v) =
    @inbounds fields.cloud.P[i, j, k]

@inline explicit_tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields, closure_fields, u, v, injected) =
    tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields, closure_fields, u, v)

@inline explicit_tke_shear_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, u, v, injected) =
    explicit_shear_conversion(fields.cloud, i, j, k, injected)

@inline explicit_shear_conversion(cl, i, j, k, injected) = @inbounds ifelse(injected, cl.Pˢ[i, j, k], cl.P[i, j, k])

@inline explicit_tke_buoyancy_production(fields, i, j, k, B, injected) = max(zero(B), B)
@inline explicit_tke_buoyancy_production(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, i, j, k, B, injected) =
    explicit_buoyancy_conversion(fields.cloud, i, j, k, B, injected)

solved_conversions_before_tke_solve(fields, closure, model) = false
solved_conversions_before_tke_solve(::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, model) =
    implicit_stage(closure, model) && AtmosphereModels.closure_prognostics_solved_last(model)

AtmosphereModels.solved_after_thermodynamic_variable(::FlavorOfTKEClosure, name) = name === TKE_NAME

@inline function squared_increment(i, j, k, grid, ρu, ρu⁻)
    @inbounds δ = ρu[i, j, k] - ρu⁻[i, j, k]
    return δ * δ
end

@inline function implicit_momentum_dissipationᶜᶜᶜ(i, j, k, grid, s, cl, Δt)
    δ² = ℑxᶜᵃᵃ(i, j, k, grid, squared_increment, s.ρu, cl.ρu⁻) + ℑyᵃᶜᵃ(i, j, k, grid, squared_increment, s.ρv, cl.ρv⁻)
    ρᵢ = @inbounds s.ρ[i, j, k]
    return δ² / (ρᵢ * ρᵢ) / (2 * Δt)
end

@inline function latent_buoyancy_derivativesᶜᶜᶠ(i, j, k, cl)
    @inbounds begin
        bᶜᵛ = (cl.b_qᶜ[i, j, k] + cl.b_qᶜ[i, j, k-1] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, k-1]) / 2
        bⁱᵛ = (cl.b_qⁱ[i, j, k] + cl.b_qⁱ[i, j, k-1] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, k-1]) / 2
        bˡᵛ = (cl.b_qˡ[i, j, k] + cl.b_qˡ[i, j, k-1] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, k-1]) / 2
    end
    return bᶜᵛ, bⁱᵛ, bˡᵛ
end

@inline function solved_latent_conversionᶜᶜᶠ(i, j, k, grid, s, cl, Kᶜ, qʷ, θʷ)
    @inbounds begin
        θ⁺ = θʷ[i, j, k]
        θ⁻ = θʷ[i, j, k-1]
        qʷ⁺ = qʷ[i, j, k]
        qʷ⁻ = qʷ[i, j, k-1]
        qᶜ⁺ = distributed_condensateᶜᶜᶜ(i, j, k, grid, s)
        qᶜ⁻ = distributed_condensateᶜᶜᶜ(i, j, k-1, grid, s)
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        ∂z_s = (cl.c_q[i, j, k] * (qʷ⁺ - qʷ⁻) - cl.c_θ[i, j, k] * (θ⁺ - θ⁻)) / Δz
        ∂z_sⁱ = (cl.c_qⁱ[i, j, k] * (qʷ⁺ - qʷ⁻) - cl.c_θⁱ[i, j, k] * (θ⁺ - θ⁻)) / Δz
        ∂z_qᶜ = (qᶜ⁺ - qᶜ⁻) / Δz
        bᶜᵛ, bⁱᵛ, bˡᵛ = latent_buoyancy_derivativesᶜᶜᶠ(i, j, k, cl)
        condensate = bᶜᵛ * (cl.η[i, j, k] * ∂z_qᶜ - cl.aᴺ[i, j, k] * ∂z_s)
        outside = - bⁱᵛ * cl.ηⁱ[i, j, k] * ∂z_sⁱ - bˡᵛ * cl.ηʷⁱ[i, j, k] * ∂z_s
        return Kᶜ[i, j, k] * (condensate + outside)
    end
end

@inline function solved_latent_conversionᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, qʷ, θʷ)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    Lˡᵒʷ = ifelse(k > 1, solved_latent_conversionᶜᶜᶠ(i, j, max(k, 2), grid, s, cl, Kᶜ, qʷ, θʷ), zero(FT))
    Lᵘᵖ = ifelse(k < Nz, solved_latent_conversionᶜᶜᶠ(i, j, min(k + 1, Nz), grid, s, cl, Kᶜ, qʷ, θʷ), zero(FT))
    wˡᵒʷ, wᵘᵖ, ρΔz = half_cell_weights(i, j, k, grid, s.ρ)
    return (wˡᵒʷ * Lˡᵒʷ + wᵘᵖ * Lᵘᵖ) / ρΔz
end

@kernel function _give_solved_conversions!(ρe, cl, grid, closure, s, Kᶜ, Kᵘ, u, v)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    θʷ, qʷ, _ = distributed_water(s, grid)
    Bᵖ, Pᵖ, _ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    Dᵘ = implicit_momentum_dissipationᶜᶜᶜ(i, j, k, grid, s, cl, @inbounds(cl.stage_time_step[1, 1, 1]))
    Bᴸⁱ = solved_latent_conversionᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, qʷ, θʷ)
    Pⁱ = Pᵖ + Dᵘ
    Bⁱ = max(Bᵖ + Bᴸⁱ, zero(Bᵖ))
    ℂᴾ = shear_production_fraction(closure_ij.stability_functions)
    @inbounds begin
        cl.Pⁱ[i, j, k] = Pⁱ
        cl.Bⁱ[i, j, k] = Bⁱ
        cl.Bᴸⁱ[i, j, k] = Bᴸⁱ
        ρe[i, j, k] += cl.stage_time_step[1, 1, 1] * s.ρ[i, j, k] * (ℂᴾ * Pⁱ + Bⁱ) * !inactive_cell(i, j, k, grid)
    end
end

function give_solved_conversions!(closure_fields, closure, model)
    grid = model.grid
    cl = closure_fields.subgrid_cloud.cloud
    s = applied_flux_state(model)
    fill_halo_regions!((model.momentum.ρu, model.momentum.ρv))
    u = XFaceVelocity(model.momentum.ρu, s.ρ, grid)
    v = YFaceVelocity(model.momentum.ρv, s.ρ, grid)
    launch!(grid.architecture, grid, :xyz, _give_solved_conversions!, model.tracers[TKE_NAME], cl, grid, closure, s,
            closure_fields.Kᶜ, closure_fields.Kᵘ, u, v)
    return nothing
end

@inline explicit_buoyancy_conversion(cl, i, j, k, B, injected) =
    @inbounds ifelse(injected, max(cl.Bˢ[i, j, k] + heating_flux_conversion(cl, i, j, k), zero(B)), max(B, zero(B)))

@inline stage_state_conversion(cl, i, j, k, injected) =
    @inbounds cl.Pˢ[i, j, k] + cl.Bˢ[i, j, k] + heating_flux_conversion(cl, i, j, k) + ifelse(injected, zero(cl.Bᴸ[i, j, k]), cl.Bᴸ[i, j, k])

#####
##### The floor of e in the buoyancy sink rate
#####

@inline buoyancy_sink_floor(fields, closure, FT) = closure.minimum_tke
@inline buoyancy_sink_floor(::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, FT) = buoyancy_sink_floor_value(FT)
@inline buoyancy_sink_floor_value(FT) = sqrt(floatmin(FT))

@inline buoyancy_sink_tke(i, j, k, grid, fields, closure, e) = @inbounds max(e[i, j, k], buoyancy_sink_floor(fields, closure, grid_float_type(grid)))

@inline buoyancy_sink_tke(i, j, k, grid, ::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, e) =
    applied_flux_buoyancy_sink_tke(i, j, k, grid, e)

@inline positive_tkeᶜᶜᶜ(i, j, k, grid, e) = @inbounds max(e[i, j, k], zero(eltype(e)))

@inline function applied_flux_buoyancy_sink_tke(i, j, k, grid, e)
    FT = grid_float_type(grid)
    Nz = size(grid, 3)
    zᶜ = znode(i, j, k, grid, Center(), Center(), Center())
    wˡᵒʷ = ifelse(k > 1, zᶜ - znode(i, j, k, grid, Center(), Center(), Face()), zero(FT))
    wᵘᵖ = ifelse(k < Nz, znode(i, j, k + 1, grid, Center(), Center(), Face()) - zᶜ, zero(FT))
    eˡᵒʷ = ifelse(k > 1, ℑzᵃᵃᶠ(i, j, max(k, 2), grid, positive_tkeᶜᶜᶜ, e), zero(FT))
    eᵘᵖ = ifelse(k < Nz, ℑzᵃᵃᶠ(i, j, min(k + 1, Nz), grid, positive_tkeᶜᶜᶜ, e), zero(FT))
    ē = (wˡᵒʷ * eˡᵒʷ + wᵘᵖ * eᵘᵖ) / (wˡᵒʷ + wᵘᵖ)
    return max(ē, buoyancy_sink_floor_value(FT))
end

const ImplicitAppliedFluxFields = SubgridCloudFields{<:AppliedFluxEnergetics}

implicit_stage(closure, model) =
    time_discretization(closure) isa VerticallyImplicitTimeDiscretization && !isnothing(model.timestepper.implicit_solver)

implicit_stage(closure::TKEClosureArray, model) = implicit_stage(first(closure), model)

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

function AtmosphereModels.closure_before_stage_combination!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                             closure::FlavorOfTKEClosure, model, Δt, α)
    implicit_stage(closure, model) || return nothing
    fill!(closure_fields.subgrid_cloud.cloud.stage_time_step, α * Δt)
    grid = model.grid
    U⁰ = model.timestepper.U⁰
    Gⁿ = model.timestepper.Gⁿ
    ρu, ρv = model.momentum.ρu, model.momentum.ρv
    fill_halo_regions!((ρu, ρv, Gⁿ.ρu, Gⁿ.ρv, U⁰.ρu, U⁰.ρv))
    launch!(grid.architecture, grid, :xyz, _stage_combination_kinetic_energy!, closure_fields.subgrid_cloud.cloud.stage_combination,
            grid, ρu, ρv, Gⁿ.ρu, Gⁿ.ρv, U⁰.ρu, U⁰.ρv, total_density(model.dynamics), Δt, α)
    return nothing
end

function AtmosphereModels.build_closure_stage_storage(::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                      closure, grid, stages)
    exchange = ntuple(_ -> CenterField(grid), stages)
    momentum = ntuple(_ -> (ρu = XFaceField(grid), ρv = YFaceField(grid)), stages - 1)
    return (; exchange, recombined_exchange = CenterField(grid), momentum)
end

@kernel function _store_stage_exchange!(x, cl, injected)
    i, j, k = @index(Global, NTuple)
    @inbounds x[i, j, k] = stage_state_conversion(cl, i, j, k, injected) -
                           explicit_shear_conversion(cl, i, j, k, injected) -
                           explicit_buoyancy_conversion(cl, i, j, k, cl.B[i, j, k], injected)
end

@inline weighted_sum(i, j, k, x::Tuple{Any}, b::Tuple{Any}) = @inbounds b[1] * x[1][i, j, k]
@inline weighted_sum(i, j, k, x::Tuple, b::Tuple) =
    @inbounds b[1] * x[1][i, j, k] + weighted_sum(i, j, k, Base.tail(x), Base.tail(b))

@kernel function _recombine_stage_exchange!(X, x, b)
    i, j, k = @index(Global, NTuple)
    @inbounds X[i, j, k] = weighted_sum(i, j, k, x, b)
end

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

function AtmosphereModels.closure_before_stage_combination!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                             closure::FlavorOfTKEClosure, model, Δt,
                                                             recombination::AtmosphereModels.StageRecombination)
    implicit_stage(closure, model) || return nothing
    fill!(closure_fields.subgrid_cloud.cloud.stage_time_step, Δt)
    grid = model.grid
    arch = grid.architecture
    cl = closure_fields.subgrid_cloud.cloud
    storage = recombination.storage
    i = recombination.stage

    launch!(arch, grid, :xyz, _store_stage_exchange!, storage.exchange[i], cl, solved_conversions_before_tke_solve(closure_fields.subgrid_cloud, closure, model))
    launch!(arch, grid, :xyz, _recombine_stage_exchange!, storage.recombined_exchange,
            storage.exchange[1:i], recombination.tendency_weights)

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

@inline net_stage_conversion(::Nothing, i, j, k, converted, given, solved) = converted - given
@inline net_stage_conversion(X, i, j, k, converted, given, solved) = @inbounds solved + X[i, j, k]

recombined_stage_exchange(::Nothing) = nothing
recombined_stage_exchange(storage) = storage.recombined_exchange

function AtmosphereModels.closure_before_implicit_solve!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                          closure::FlavorOfTKEClosure, model, name)
    implicit_stage(closure, model) || return nothing
    cl = closure_fields.subgrid_cloud.cloud
    name === :ρu && (interior(cl.ρu⁻) .= interior(model.momentum.ρu); fill_halo_regions!(cl.ρu⁻))
    name === :ρv && (interior(cl.ρv⁻) .= interior(model.momentum.ρv); fill_halo_regions!(cl.ρv⁻))
    pre = closure_fields.subgrid_cloud.presolve
    name ∈ keys(pre) && (parent(pre[name]) .= parent(Oceananigans.prognostic_fields(model)[name]))
    name === TKE_NAME && solved_conversions_before_tke_solve(closure_fields.subgrid_cloud, closure, model) &&
        give_solved_conversions!(closure_fields, closure, model)
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

function AtmosphereModels.closure_before_thermodynamic_solve!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                               closure::FlavorOfTKEClosure, model, Δt)
    implicit_stage(closure, model) || return nothing
    fields = closure_fields.subgrid_cloud
    grid = model.grid
    ρθ = distribution_potential_temperature_density(model)
    solved_conversions_before_tke_solve(fields, closure, model) ||
        launch!(grid.architecture, grid, :xyz, _heat_by_implicit_dissipation!, ρθ,
                fields.cloud, grid, closure_fields.Lᵉ, model.tracers[TKE_NAME], total_density(model.dynamics), Δt)
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    name ∈ keys(fields.presolve) && (parent(fields.presolve[name]) .= parent(ρθ))
    return nothing
end

@kernel function _correct_tke_to_solved_conversions!(ρe, cl, grid, closure, s, Kᶜ, Kᵘ, u, v, Lᵉ, Δt, X, injected)
    i, j, k = @index(Global, NTuple)
    Bᵖ, Pᵖ, _ = interior_conversionsᶜᶜᶜ(i, j, k, grid, s, cl, Kᶜ, Kᵘ, u, v)
    @inbounds begin
        ρᵢ = s.ρ[i, j, k]
        e = ρe[i, j, k] / ρᵢ
        Dᵘ = implicit_momentum_dissipationᶜᶜᶜ(i, j, k, grid, s, cl, Δt)
        B = cl.B[i, j, k]
        Pⁱ = ifelse(injected, cl.Pⁱ[i, j, k], zero(e))
        Bⁱ = ifelse(injected, cl.Bⁱ[i, j, k], zero(e))
        Bᴸⁱ = ifelse(injected, cl.Bᴸⁱ[i, j, k], zero(e))
        Pᵉ = explicit_shear_conversion(cl, i, j, k, injected)
        Bᵉ = explicit_buoyancy_conversion(cl, i, j, k, B, injected)
        given = Pᵉ + Pⁱ + Bᵉ + Bⁱ - cl.ωᴮ[i, j, k] * e
        stage_state = stage_state_conversion(cl, i, j, k, injected)
        converted = Pᵖ + Dᵘ + Bᵖ + Bᴸⁱ + stage_state
        ε = (-Lᵉ[i, j, k] - cl.ωᴮ[i, j, k]) * e
        cl.Pᵖ[i, j, k] = Pᵖ + Dᵘ
        cl.Bᵖ[i, j, k] = Bᵖ
        ℂᴾ = shear_production_fraction(getclosure(i, j, closure).stability_functions)
        cl.given[i, j, k] = Pᵉ + Bᵉ + (1 - ℂᴾ) * Pⁱ
        cl.stage_b_θ[i, j, k] = cl.b_θ[i, j, k]
        cl.stage_b_qᵛ[i, j, k] = cl.b_qᵛ[i, j, k]
        cl.stage_b_qˡ[i, j, k] = cl.b_qˡ[i, j, k]
        cl.stage_b_qⁱ[i, j, k] = cl.b_qⁱ[i, j, k]
        cl.stage_surface[i, j, k] = stage_state
        cl.Bᴸⁱ[i, j, k] = Bᴸⁱ
        active = !inactive_cell(i, j, k, grid)
        net = net_stage_conversion(X, i, j, k, converted, given, Pᵖ + Dᵘ - Pⁱ + Bᵖ + Bᴸⁱ - Bⁱ + cl.ωᴮ[i, j, k] * e)
        ρe⁺ = ρe[i, j, k] + (Δt * ρᵢ * net + cl.stage_combination[i, j, k]) * active
        ρe⁺′ = ifelse(ρe⁺ < 0, min(ρe[i, j, k], zero(ρe⁺)), ρe⁺)
        shortfall = ρe⁺′ - ρe⁺ # ≥ 0: energy the TKE did not pay
        ρe[i, j, k] = ρe⁺′
        cl.shortfall[i, j, k] = shortfall
        heating = ifelse(injected, Δt * ρᵢ * (ε + (1 - ℂᴾ) * Pⁱ - Bᴸⁱ), zero(e))
        cl.ε[i, j, k] = ifelse(injected, ε, cl.ε[i, j, k])
        cl.after_solves_ρθ[i, j, k] = (heating - shortfall) / cl.ĉᵖΠ[i, j, k] * active
    end
end

@kernel function _apply_after_solves_heating!(ρθ, cl)
    i, j, k = @index(Global, NTuple)
    @inbounds ρθ[i, j, k] += cl.after_solves_ρθ[i, j, k]
end

function AtmosphereModels.closure_after_implicit_solves!(closure_fields::TKEClosureFields{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImplicitAppliedFluxFields},
                                                          closure::FlavorOfTKEClosure, model, Δt)
    implicit_stage(closure, model) || return nothing
    grid = model.grid
    cl = closure_fields.subgrid_cloud.cloud
    s = applied_flux_state(model)
    ρ = s.ρ
    fill_halo_regions!((model.momentum.ρu, model.momentum.ρv))
    u = XFaceVelocity(model.momentum.ρu, ρ, grid)
    v = YFaceVelocity(model.momentum.ρv, ρ, grid)
    launch!(grid.architecture, grid, :xyz, _correct_tke_to_solved_conversions!, model.tracers[TKE_NAME], cl, grid, closure, s,
            closure_fields.Kᶜ, closure_fields.Kᵘ, u, v, closure_fields.Lᵉ, Δt,
            recombined_stage_exchange(AtmosphereModels.closure_stage_storage(model.timestepper)),
            solved_conversions_before_tke_solve(closure_fields.subgrid_cloud, closure, model))
    launch!(grid.architecture, grid, :xyz, _apply_after_solves_heating!, s.ρθ, cl)
    return nothing
end

@kernel function _add_dissipative_heating!(Gρθ, grid, closure_fields, ρe, ρ)
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
    launch!(grid.architecture, grid, :xyz, _add_dissipative_heating!, Gⁿ[name], grid, closure_fields,
            model.tracers[TKE_NAME], total_density(model.dynamics))
    return nothing
end

add_shear_production_heating!(Gⁿ, stability_functions, fields, closure_fields, closure, model) = nothing

@kernel function _add_shear_production_heating!(Gρθ, grid, closure, cl, ρ, injected)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    ℂᴾ = shear_production_fraction(closure_ij.stability_functions)
    P = explicit_shear_conversion(cl, i, j, k, injected)
    @inbounds Gρθ[i, j, k] += ρ[i, j, k] * (1 - ℂᴾ) * P / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
end

function add_shear_production_heating!(Gⁿ, ::ColumnStabilityDamping, fields::SubgridCloudFields{<:AppliedFluxEnergetics},
                                       closure_fields, closure, model)
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _add_shear_production_heating!, Gⁿ[name], grid, closure, fields.cloud,
            total_density(model.dynamics), solved_conversions_before_tke_solve(fields, closure, model))
    return nothing
end

@inline function distributed_condensateᶜᶜᶜ(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        qᵛᵉ = s.ρqᵛᵉ[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)
    return distributed_condensate(cloud_liquid_density(s.microphysical_fields), i, j, k, ρ, q)
end

@inline distributed_condensate(::Nothing, i, j, k, ρ, q) = q.liquid + q.ice
@inline distributed_condensate(ρqᶜˡ, i, j, k, ρ, q) = @inbounds ρqᶜˡ[i, j, k] / ρ

@inline function cell_saturation_linearization(thermodynamics, i, j, k, grid, s, θʷ)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        T = s.T[i, j, k]
        p = s.p[i, j, k]
        qᵛᵉ = s.ρqᵛᵉ[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)
    return saturation_linearization(thermodynamics, T, p, q, θʷ, s.exner_exponent, s.standard_pressure)
end

@inline function saturation_linearization(thermodynamics, T, p, q, θʷ, exner_exponent_formulation, pˢᵗ)
    constants = thermodynamics.constants
    equilibrium = thermodynamics.phase_equilibrium
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    R̂ = vapor_equivalent_gas_constant(q, constants)
    κ = exner_exponent(exner_exponent_formulation, q, constants)
    κ′ = exner_exponent_moisture_derivative(exner_exponent_formulation, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    ln_p = log(p / pˢᵗ)
    Π = (p / pˢᵗ)^κ

    qᵛ⁺ = saturation_specific_humidity(T, p / (Rᵐ * T), constants, saturation_surface(equilibrium, T))
    ℒ, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, T)
    ∂qᵛ⁺∂T = qᵛ⁺ * log_slope
    B = θʷ * Π * ((cᵖᵛ - cᵖᵈ) + ĉᵖ * ln_p * κ′) - (cᵖᵛ - cᵖᵈ) * T
    D = 1 - qᵛ⁺ * Rᵛ / Rᵐ + ∂qᵛ⁺∂T * ℒ / cᵖᵐ
    v_q = 1 - qᵛ⁺ * (Rᵛ - Rᵈ) / Rᵐ - ∂qᵛ⁺∂T * B / cᵖᵐ
    v_θ = ∂qᵛ⁺∂T * ĉᵖ * Π / cᵖᵐ
    return (; qᵛ⁺, v_q, v_θ, D)
end

@inline function saturated_condensate_coefficients(thermodynamics, i, j, k, grid, s, θʷ)
    lin = cell_saturation_linearization(thermodynamics, i, j, k, grid, s, θʷ)
    return lin.v_q / lin.D, lin.v_θ / lin.D
end

@inline function vapor_excess_coefficients(thermodynamics, i, j, k, grid, s, θʷ)
    lin = cell_saturation_linearization(thermodynamics, i, j, k, grid, s, θʷ)
    return lin.v_q, lin.v_θ
end

@inline function mean_saturation_excess(thermodynamics, i, j, k, grid, s, θʷ, qʷ)
    lin = cell_saturation_linearization(thermodynamics, i, j, k, grid, s, θʷ)
    qᶜ = distributed_condensateᶜᶜᶜ(i, j, k, grid, s)
    return qᶜ + (qʷ - qᶜ - lin.qᵛ⁺) / lin.D
end

@inline function cell_phase_relaxation_rates(i, j, k, grid, s)
    @inbounds begin
        ρ = s.ρ[i, j, k]
        qᵛᵉ = s.qᵛᵉ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, s.microphysics, ρ, qᵛᵉ, s.microphysical_fields)
    𝒰 = AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid, s.formulation, s.dynamics, q)
    return AtmosphereModels.phase_relaxation_rates(i, j, k, grid, s.microphysics, s.microphysical_fields, 𝒰, s.constants)
end

@inline function condensing_fraction(x)
    FT = typeof(x)
    small = x < convert(FT, 1e-3)
    xˢ = ifelse(small, one(FT), x)
    return ifelse(small, x * (1 / convert(FT, 2) - x * (1 / convert(FT, 6) - x / 24)), 1 + expm1(-xˢ) / xˢ)
end

@inline saturated_response_sharesᶜᶜᶠ(::Val{true}, tₑ, aʷ, i, j, k, grid, s) = (one(tₑ), zero(tₑ), zero(tₑ))

@inline function saturated_response_sharesᶜᶜᶠ(::Val{false}, tₑ, aʷ, i, j, k, grid, s)
    above = cell_phase_relaxation_rates(i, j, k, grid, s)
    below = cell_phase_relaxation_rates(i, j, k-1, grid, s)
    ωˡ = (above.ωˡ + below.ωˡ) / 2
    ωⁱ = (above.ωᵈⁱ + below.ωᵈⁱ) / 2
    ωʷⁱ = (above.ωʷⁱ + below.ωʷⁱ) / 2
    ω⁺ = ωˡ + ωⁱ + ωʷⁱ
    ω⁻ = ωⁱ + ωʷⁱ
    FT = typeof(ω⁺)
    η⁺ = condensing_fraction(tₑ * ω⁺)
    η⁻ = condensing_fraction(tₑ * ω⁻)
    share(ω, total, fallback) = ifelse(total > 0, ω / ifelse(total > 0, total, one(FT)), fallback)
    η = share(ωˡ, ω⁺, one(FT)) * η⁺
    ηⁱ = aʷ * share(ωⁱ, ω⁺, zero(FT)) * η⁺ + (1 - aʷ) * share(ωⁱ, ω⁻, zero(FT)) * η⁻
    ηʷⁱ = aʷ * share(ωʷⁱ, ω⁺, zero(FT)) * η⁺ + (1 - aʷ) * share(ωʷⁱ, ω⁻, zero(FT)) * η⁻
    return η, ηⁱ, ηʷⁱ
end

@inline function center_condensate_flux_fraction(i, j, k, cl)
    @inbounds begin
        μ₁ = cl.μ₁[i, j, k]
        μ₂ = cl.μ₂[i, j, k]
        σ₁ = cl.σ₁[i, j, k]
        σ₂ = cl.σ₂[i, j, k]
    end
    return (saturated_share_of_segment(μ₁, μ₂, σ₁) + saturated_share_of_segment(μ₁, μ₂, σ₂)) / 2
end

@inline function saturated_share_of_segment(μ₁, μ₂, σ)
    FT = typeof(μ₁)
    Δμ = μ₁ - μ₂
    resolved = abs(Δμ) > sqrt(eps(FT)) * (abs(μ₁) + abs(μ₂) + σ)
    secant = (component_saturated_excess(μ₁, σ) - component_saturated_excess(μ₂, σ)) / ifelse(resolved, Δμ, one(FT))
    return ifelse(resolved, secant, component_saturated_fraction((μ₁ + μ₂) / 2, σ))
end

@inline function component_saturated_excess(μ, σ)
    FT = typeof(μ)
    z = standard_score(μ, zero(FT), σ)
    Φ = erfc(-z / sqrt(FT(2))) / 2
    φ = exp(-z * z / 2) / sqrt(2 * FT(π))
    return μ * Φ + σ * φ
end

@inline component_saturated_fraction(μ, σ) = erfc(-standard_score(μ, zero(μ), σ) / sqrt(2 * one(μ))) / 2

@inline face_condensate_flux_fraction(i, j, k, cl) =
    (center_condensate_flux_fraction(i, j, k, cl) + center_condensate_flux_fraction(i, j, k-1, cl)) / 2

@inline function face_eddy_time(i, j, k, grid, closure_ij, closure_fields, velocities, e)
    ℓ = @inbounds closure_fields.ℓᵐ[i, j, k]
    u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)
    𝕊ᴰ = dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, closure_fields.N², e, closure_fields.ℓᵐ)
    return ℓ / (𝕊ᴰ * u)
end

@inline function latent_correctionᶜᶜᶠ(i, j, k, grid, closure_ij, closure_fields, velocities, s, qʷ, θʷ, e, thermodynamics, cl,
                                      aʷ)
    @inbounds begin
        θ⁺ = θʷ[i, j, k]
        θ⁻ = θʷ[i, j, k-1]
        qʷ⁺ = qʷ[i, j, k]
        qʷ⁻ = qʷ[i, j, k-1]
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        c_q⁺, c_θ⁺ = saturated_condensate_coefficients(thermodynamics, i, j, k, grid, s, θ⁺)
        c_q⁻, c_θ⁻ = saturated_condensate_coefficients(thermodynamics, i, j, k-1, grid, s, θ⁻)
        ice = ice_saturation_thermodynamics(thermodynamics)
        c_qⁱ⁺, c_θⁱ⁺ = saturated_condensate_coefficients(ice, i, j, k, grid, s, θ⁺)
        c_qⁱ⁻, c_θⁱ⁻ = saturated_condensate_coefficients(ice, i, j, k-1, grid, s, θ⁻)
        c_q = (c_q⁺ + c_q⁻) / 2
        c_θ = (c_θ⁺ + c_θ⁻) / 2
        c_qⁱ = (c_qⁱ⁺ + c_qⁱ⁻) / 2
        c_θⁱ = (c_θⁱ⁺ + c_θⁱ⁻) / 2
        ∂z_s = (c_q * (qʷ⁺ - qʷ⁻) - c_θ * (θ⁺ - θ⁻)) / Δz
        ∂z_sⁱ = (c_qⁱ * (qʷ⁺ - qʷ⁻) - c_θⁱ * (θ⁺ - θ⁻)) / Δz
        Δqᶜ = distributed_condensateᶜᶜᶜ(i, j, k, grid, s) - distributed_condensateᶜᶜᶜ(i, j, k-1, grid, s)
        ∂z_qᶜ = Δqᶜ / Δz
        bᵛ = cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, k-1]
        bᶜᵛ = (cl.b_qᶜ[i, j, k] + cl.b_qᶜ[i, j, k-1] - bᵛ) / 2
        bⁱᵛ = (cl.b_qⁱ[i, j, k] + cl.b_qⁱ[i, j, k-1] - bᵛ) / 2
        bˡᵛ = (cl.b_qˡ[i, j, k] + cl.b_qˡ[i, j, k-1] - bᵛ) / 2
    end
    tₑ = face_eddy_time(i, j, k, grid, closure_ij, closure_fields, velocities, e)
    instantaneous = Val(AtmosphereModels.instantaneous_phase_relaxation(s.microphysics))
    η, ηⁱ, ηʷⁱ = saturated_response_sharesᶜᶜᶠ(instantaneous, tₑ, aʷ, i, j, k, grid, s)
    ΔN²ᴸ = η * bᶜᵛ * (∂z_qᶜ - aʷ * ∂z_s) - ηⁱ * bⁱᵛ * ∂z_sⁱ - ηʷⁱ * bˡᵛ * ∂z_s
    return (; ΔN²ᴸ, η, ηⁱ, ηʷⁱ, aᴺ = η * aʷ, c_q, c_θ, c_qⁱ, c_θⁱ, bᶜᵛ, Δqᶜ, Δz,
              outside = -ηⁱ * bⁱᵛ * ∂z_sⁱ - ηʷⁱ * bˡᵛ * ∂z_s)
end

@inline function store_latent_correction!(cl, i, j, k, interior, aʷ, L)
    FT = typeof(L.ΔN²ᴸ)
    @inbounds begin
        cl.aʷ[i, j, k] = ifelse(interior, aʷ, zero(FT))
        cl.aᴺ[i, j, k] = ifelse(interior, L.aᴺ, zero(FT))
        cl.c_θ[i, j, k] = ifelse(interior, L.c_θ, zero(FT))
        cl.c_q[i, j, k] = ifelse(interior, L.c_q, zero(FT))
        cl.c_θⁱ[i, j, k] = ifelse(interior, L.c_θⁱ, zero(FT))
        cl.c_qⁱ[i, j, k] = ifelse(interior, L.c_qⁱ, zero(FT))
        cl.η[i, j, k] = ifelse(interior, L.η, zero(FT))
        cl.ηⁱ[i, j, k] = ifelse(interior, L.ηⁱ, zero(FT))
        cl.ηʷⁱ[i, j, k] = ifelse(interior, L.ηʷⁱ, zero(FT))
        cl.ΔN²ᴸ[i, j, k] = ifelse(interior, L.ΔN²ᴸ, zero(FT))
    end
    return nothing
end

@kernel function _latent_corrected_stability!(N², grid, closure, closure_fields, velocities, s, e)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    cl = closure_fields.subgrid_cloud.cloud
    θʷ, qʷ, thermodynamics = distributed_water(s, grid)
    kₛ = max(k, 2)
    aʷ = face_condensate_flux_fraction(i, j, kₛ, cl)
    L = latent_correctionᶜᶜᶠ(i, j, kₛ, grid, closure_ij, closure_fields, velocities, s, qʷ, θʷ, e, thermodynamics, cl, aʷ)
    interior = k > 1
    store_latent_correction!(cl, i, j, k, interior, aʷ, L)
    @inbounds begin
        N²ᵇ = applied_stratificationᶜᶜᶠ(i, j, kₛ, grid, s, cl)
        N²ᵉ = N²ᵇ - L.ΔN²ᴸ
        N²⁰ = N²[i, j, k]
        N²⁺ = condensate_flux_stratification(closure_ij.static_stability.fraction.stratification, N²ᵉ, N²ᵇ, L.bᶜᵛ,
                                             cl.qˡ[i, j, kₛ] - cl.qˡ[i, j, kₛ-1], L.Δqᶜ, L.Δz, N²⁰)
        N²[i, j, k] = ifelse(interior, N²⁺, N²⁰)
    end
end

function applied_flux_stability_pass!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, closure, model, tracers,
                                      parameters)
    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, parameters, _latent_corrected_stability!, closure_fields.N², grid, closure, closure_fields,
            model.velocities, applied_flux_state(model), tracers[TKE_NAME])
    compute_mixing_length_and_diffusivities!(closure_fields, closure, model, tracers, parameters)
    return nothing
end

@kernel function _draw_latent_conversion!(Gρθ, grid, cl, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρθ[i, j, k] -= ρ[i, j, k] * cl.Bᴸ[i, j, k] / cl.ĉᵖΠ[i, j, k] * !inactive_cell(i, j, k, grid)
end

draw_latent_conversion!(Gⁿ, fields, closure, model) = nothing

function draw_latent_conversion!(Gⁿ, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure, model)
    solved_conversions_before_tke_solve(fields, closure, model) && return nothing
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _draw_latent_conversion!, Gⁿ[name], grid, fields.cloud, total_density(model.dynamics))
    return nothing
end
