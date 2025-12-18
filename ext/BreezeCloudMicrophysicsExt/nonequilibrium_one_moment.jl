#####
##### Non-equilibrium one-moment microphysics (WPNE1M)
#####
# Cloud liquid is prognostic and evolves via condensation/evaporation tendencies
# following Morrison and Milbrandt (2015) relaxation formulation.

# Non-equilibrium cloud formation with 1M precipitation (warm-phase only for now)
const WarmPhaseNonEquilibrium1M = BulkMicrophysics{<:NonEquilibriumCloudFormation{<:CloudLiquid, Nothing}, <:CM1MCategories, <:Any}
const WPNE1M = WarmPhaseNonEquilibrium1M

prognostic_field_names(::WPNE1M) = (:ρqᶜˡ, :ρqʳ)

function materialize_microphysical_fields(bμp::WPNE1M, grid, bcs)
    center_names = (:qᵛ, :qˡ, :qᶜˡ, :qʳ, :ρqᶜˡ, :ρqʳ)
    center_fields = center_field_tuple(grid, center_names...)
    # Rain terminal velocity (negative = downward)
    # bottom = nothing ensures the kernel-set value is preserved during fill_halo_regions!
    wʳ_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    wʳ = ZFaceField(grid; boundary_conditions=wʳ_bcs)
    return (; zip(center_names, center_fields)..., wʳ)
end

@inline function update_microphysical_fields!(μ, bμp::WPNE1M, i, j, k, grid, ρ, 𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor
    qˡ = q.liquid  # total liquid from thermodynamic state
    categories = bμp.categories

    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ  # cloud liquid from prognostic field
        qʳ = μ.ρqʳ[i, j, k] / ρ    # rain from prognostic field
        μ.qᵛ[i, j, k] = qᵛ
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qʳ[i, j, k] = qʳ
        μ.qˡ[i, j, k] = qᶜˡ + qʳ  # total liquid (cloud + rain)

        # Terminal velocity for rain (negative = downward)
        wᵗ = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, qʳ)
        μ.wʳ[i, j, k] = -wᵗ

        # For ImpenetrableBottom, set wʳ = 0 at bottom face to prevent rain from exiting
        μ.wʳ[i, j, 1] = bottom_terminal_velocity(bμp.precipitation_boundary_condition, μ.wʳ[i, j, 1])
    end

    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::WPNE1M, ρ, qᵗ, μ)
    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        qʳ = μ.ρqʳ[i, j, k] / ρ
    end

    # Vapor is diagnosed from total moisture minus condensates
    qᵛ = qᵗ - qᶜˡ - qʳ
    qˡ = qᶜˡ + qʳ
    qⁱ = zero(qˡ)

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

@inline maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::WPNE1M, args...) = 𝒰₀

@inline function thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)
    ℒˡ = liquid_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ * (ℒˡ / (Rᵛ * T^2) - 1 / T)
    return 1 + (ℒˡ / cᵖᵐ) * dqᵛ⁺_dT
end

@inline function condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)
    Γˡ = thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)
    Sᶜᵒⁿᵈ = (qᵛ - qᵛ⁺) / (Γˡ * τᶜˡ)
    
    # Limit evaporation (Sᶜᵒⁿᵈ < 0) to available cloud liquid
    # This prevents qᶜˡ from going negative
    Sᶜᵒⁿᵈ_limited = ifelse(Sᶜᵒⁿᵈ < 0, max(Sᶜᵒⁿᵈ, -qᶜˡ / τᶜˡ), Sᶜᵒⁿᵈ)
    
    return Sᶜᵒⁿᵈ_limited
end

#####
##### Microphysical tendencies for non-equilibrium 1M
#####

# Rain tendency for non-equilibrium 1M: autoconversion + accretion + evaporation
@inline function microphysical_tendency(i, j, k, grid, bμp::WPNE1M, ::Val{:ρqʳ}, ρ, μ, 𝒰, constants)
    categories = bμp.categories
    ρⁱʲᵏ = @inbounds ρ[i, j, k]

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]  # cloud liquid
    @inbounds qʳ = μ.qʳ[i, j, k]    # rain

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρⁱʲᵏ)

    # Rain evaporation using translated CloudMicrophysics physics
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions

    Sᵉᵛᵃᵖ = rain_evaporation(categories.rain,
                             categories.hydrometeor_velocities.rain,
                             categories.air_properties,
                             q, qʳ, ρⁱʲᵏ, T, constants)

    # Limit evaporation to available rain (relaxation-style limiter)
    # Use condensation timescale as reference for limiting
    τᶜˡ = bμp.cloud_formation.liquid.τ_relax
    Sᵉᵛᵃᵖ_limited = max(Sᵉᵛᵃᵖ, -qʳ / τᶜˡ)

    # Total tendency for ρqʳ (positive = rain increase)
    return ρⁱʲᵏ * (Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ_limited)
end


# Cloud liquid tendency for non-equilibrium 1M: condensation/evaporation - (autoconversion + accretion)
@inline function microphysical_tendency(i, j, k, grid, bμp::WPNE1M, ::Val{:ρqᶜˡ}, ρ, μ, 𝒰, constants)
    categories = bμp.categories
    cloud_formation = bμp.cloud_formation
    τᶜˡ = cloud_formation.liquid.τ_relax

    ρⁱʲᵏ = @inbounds ρ[i, j, k]

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]

    # Get thermodynamic state
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    # Saturation specific humidity over liquid
    qᵛ⁺ = saturation_specific_humidity(T, ρⁱʲᵏ, constants, PlanarLiquidSurface())

    # Condensation/evaporation rate (positive = condensation = cloud liquid increase)
    # Limited to prevent qᶜˡ from going negative
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρⁱʲᵏ, q, τᶜˡ, constants)

    # Autoconversion: cloud liquid → rain (sink for cloud liquid)
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain (sink for cloud liquid)
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρⁱʲᵏ)

    # Total tendency for ρqᶜˡ: condensation - autoconversion - accretion
    return ρⁱʲᵏ * (Sᶜᵒⁿᵈ - Sᵃᶜⁿᵛ - Sᵃᶜᶜ)
end

#####
##### Precipitation rate diagnostic for non-equilibrium 1M
#####

# Non-equilibrium 1M uses the same precipitation rate calculation (autoconversion + accretion)
function precipitation_rate(model, microphysics::WPNE1M, ::Val{:liquid})
    grid = model.grid
    qᶜˡ = model.microphysical_fields.qᶜˡ
    ρqʳ = model.microphysical_fields.ρqʳ
    ρ = model.formulation.reference_state.density
    kernel = OneMomentPrecipitationRateKernel(microphysics.categories, qᶜˡ, ρqʳ, ρ)
    op = KernelFunctionOperation{Center, Center, Center}(kernel, grid)
    return Field(op)
end

