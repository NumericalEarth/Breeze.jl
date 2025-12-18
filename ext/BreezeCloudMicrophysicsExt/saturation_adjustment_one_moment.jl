#####
##### Saturation adjustment one-moment microphysics (WP1M and MP1M)
#####
# Cloud liquid and ice are diagnosed via saturation adjustment.
# Rain (and snow for mixed-phase) are prognostic.

#####
##### Warm-phase saturation adjustment 1M (WP1M)
#####

prognostic_field_names(::WP1M) = tuple(:ρqʳ)

function materialize_microphysical_fields(bμp::WP1M, grid, bcs)
    center_names = (:qᵛ, :qˡ, :qᶜˡ, :qʳ, :ρqʳ)
    center_fields = center_field_tuple(grid, center_names...)
    # Rain terminal velocity (negative = downward)
    # bottom = nothing ensures the kernel-set value is preserved during fill_halo_regions!
    wʳ_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    wʳ = ZFaceField(grid; boundary_conditions=wʳ_bcs)
    return (; zip(center_names, center_fields)..., wʳ)
end

@inline function update_microphysical_fields!(μ, bμp::WP1M, i, j, k, grid, ρ, 𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qᶜˡ = 𝒰.moisture_mass_fractions.liquid  # cloud liquid from saturation adjustment
    categories = bμp.categories

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = qᵛ
        μ.qʳ[i, j, k] = qʳ             # rain mass fraction (diagnostic)
        μ.qᶜˡ[i, j, k] = qᶜˡ           # cloud liquid (non-precipitating)
        μ.qˡ[i, j, k] = qʳ + qᶜˡ       # total liquid (cloud + rain)

        # Terminal velocity for rain (negative = downward)
        wᵗ = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, qʳ)
        μ.wʳ[i, j, k] = -wᵗ

        # For ImpenetrableBottom, set wʳ = 0 at bottom face to prevent rain from exiting
        μ.wʳ[i, j, 1] = bottom_terminal_velocity(bμp.precipitation_boundary_condition, μ.wʳ[i, j, 1])
    end

    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::WP1M, ρ, qᵗ, μ)
    @inbounds begin
        qʳ = μ.qʳ[i, j, k]
        qᶜˡ = μ.qᶜˡ[i, j, k]
        qᵛ = μ.qᵛ[i, j, k]
    end

    qˡ = qᶜˡ + qʳ
    qⁱ = zero(qˡ)

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

"""
$(TYPEDSIGNATURES)

Compute thermodynamic state for one-moment bulk microphysics with saturation adjustment.

Saturation adjustment is performed on cloud moisture only, excluding precipitating
species (rain and snow). The precipitating moisture is then added back to the
final liquid/ice fractions.

This is required because:
1. Saturation adjustment represents fast vapor↔cloud condensate equilibration
2. Rain/snow represent slower precipitation processes that don't equilibrate instantly
3. Excluding rain/snow from adjustment prevents spurious evaporation of precipitation
"""
@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::WP1M, ρᵣ, μ, qᵗ, constants)
    # Get rain mass fraction from diagnostic microphysical field
    @inbounds qʳ = μ.ρqʳ[i, j, k] / ρᵣ
    
    # Compute cloud moisture (excluding rain)
    qᵗᶜ = qᵗ - qʳ
    
    # Build moisture state for cloud-only adjustment
    qᶜ = MoistureMassFractions(qᵗᶜ)
    𝒰ᶜ = with_moisture(𝒰₀, qᶜ)
    
    # Perform saturation adjustment on cloud moisture only
    𝒰′ = adjust_thermodynamic_state(𝒰ᶜ, bμp.cloud_formation, constants)
    
    # Add rain back to the liquid fraction
    q′ = 𝒰′.moisture_mass_fractions
    qᵛ = q′.vapor
    qˡ = q′.liquid + qʳ  # cloud liquid + rain
    q = MoistureMassFractions(qᵛ, qˡ)
    
    return with_moisture(𝒰′, q)
end

# Rain mass tendency (ρqʳ): autoconversion + accretion
# Note: ρqᵗ tendency is the negative of ρqʳ tendency (conservation of moisture)
@inline function microphysical_tendency(i, j, k, grid, bμp::WP1M, ::Val{:ρqʳ}, ρ, μ, 𝒰, constants)
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

    # Total tendency for ρqʳ (positive = rain increase)
    return ρⁱʲᵏ * (Sᵃᶜⁿᵛ + Sᵃᶜᶜ)
end

# Moisture tendency (ρqᵗ): loss to precipitation (currently zero since rain is tracked separately)
# TODO: add rain evaporation
@inline function microphysical_tendency(i, j, k, grid, bμp::WP1M, ::Val{:ρqᵗ}, ρ, μ, 𝒰, constants)
    return zero(grid)
end

#####
##### Mixed-phase saturation adjustment 1M (MP1M)
#####

prognostic_field_names(::MP1M) = (:ρqʳ, :ρqˢ)

function materialize_microphysical_fields(bμp::MP1M, grid, bcs)
    center_names = (:qᵛ, :qˡ, :qᶜˡ, :qᶜⁱ, :qʳ, :qˢ, :ρqʳ, :ρqˢ)
    center_fields = center_field_tuple(grid, center_names...)
    # Rain terminal velocity (negative = downward)
    # bottom = nothing ensures the kernel-set value is preserved during fill_halo_regions!
    wʳ_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    wʳ = ZFaceField(grid; boundary_conditions=wʳ_bcs)
    return (; zip(center_names, center_fields)..., wʳ)
end

@inline function update_microphysical_fields!(μ, bμp::MP1M, i, j, k, grid, ρ, 𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qᶜˡ = 𝒰.moisture_mass_fractions.liquid
    qᶜⁱ = 𝒰.moisture_mass_fractions.ice
    categories = bμp.categories

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        qˢ = μ.ρqˢ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = qᵛ
        μ.qʳ[i, j, k] = qʳ             # rain mass fraction (diagnostic)
        μ.qˢ[i, j, k] = qˢ             # snow mass fraction (diagnostic)
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qˡ[i, j, k] = qʳ + qᶜˡ
        μ.qᶜⁱ[i, j, k] = qᶜⁱ

        # Terminal velocity for rain (negative = downward)
        𝒲ʳ = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, qʳ)
        μ.wʳ[i, j, k] = -𝒲ʳ

        # For ImpenetrableBottom, set wʳ = 0 at bottom face to prevent rain from exiting
        μ.wʳ[i, j, 1] = bottom_terminal_velocity(bμp.precipitation_boundary_condition, μ.wʳ[i, j, 1])
    end

    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::MP1M, ρ, qᵗ, μ)
    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        qˢ = μ.ρqˢ[i, j, k] / ρ
        qᶜˡ = μ.qᶜˡ[i, j, k]
        qᶜⁱ = μ.qᶜⁱ[i, j, k]
        qᵛ = μ.qᵛ[i, j, k]
    end

    qˡ = qᶜˡ + qʳ
    qⁱ = qᶜⁱ + qˢ

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::MP1M, ρᵣ, μ, qᵗ, constants)
    # Get rain and snow mass fractions from diagnostic microphysical fields
    @inbounds qʳ = μ.ρqʳ[i, j, k] / ρᵣ   
    @inbounds qˢ = μ.ρqˢ[i, j, k] / ρᵣ
    
    # Compute cloud moisture (excluding rain and snow)
    qᵗᶜ = qᵗ - qʳ - qˢ
    
    # Build moisture state for cloud-only adjustment
    qᶜ = MoistureMassFractions(qᵗᶜ)
    𝒰ᶜ = with_moisture(𝒰₀, qᶜ)
    
    # Perform saturation adjustment on cloud moisture only
    𝒰′ = adjust_thermodynamic_state(𝒰ᶜ, bμp.cloud_formation, constants)
    
    # Add rain to liquid and snow to ice
    q′ = 𝒰′.moisture_mass_fractions
    qᵛ = q′.vapor
    qˡ = q′.liquid + qʳ  # cloud liquid + rain
    qⁱ = q′.ice + qˢ     # cloud ice + snow
    q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
    
    return with_moisture(𝒰′, q)
end

#####
##### Precipitation rate diagnostic for saturation adjustment 1M
#####

"""
    precipitation_rate(model, microphysics::WP1M, ::Val{:liquid})

Return a `Field` representing the liquid precipitation rate (rain production rate) in kg/kg/s.

For one-moment microphysics, this is the rate at which cloud liquid water
is converted to rain via autoconversion and accretion.
"""
function precipitation_rate(model, microphysics::WP1M, ::Val{:liquid})
    grid = model.grid
    qᶜˡ = model.microphysical_fields.qᶜˡ
    ρqʳ = model.microphysical_fields.ρqʳ
    ρ = model.formulation.reference_state.density
    kernel = OneMomentPrecipitationRateKernel(microphysics.categories, qᶜˡ, ρqʳ, ρ)
    op = KernelFunctionOperation{Center, Center, Center}(kernel, grid)
    return Field(op)
end

