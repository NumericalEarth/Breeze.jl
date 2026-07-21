using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField
using Oceananigans.Utils: launch!

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

using Breeze.Thermodynamics: MoistureMassFractions

using Breeze: Microphysics

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index

const P3 = PredictedParticlePropertiesMicrophysics

#####
##### Stage-local tendency preparation
#####

"""
$(TYPEDSIGNATURES)

Prepare P3 diagnostics for the current RK stage.

Launches a separate GPU kernel to compute terminal velocities, process rates,
and tendency cache fields. This heavy computation is split out of the thermodynamic
variables kernel to avoid overwhelming the GPU compiler with force-inlined P3 physics
(~1000 lines of code). The lighter `update_microphysical_auxiliaries!` only writes
basic diagnostic quantities in the thermodynamic kernel.
"""
function AM.prepare_microphysical_tendencies!(p3::P3, model)
    grid = model.grid
    arch = grid.architecture
    μ = model.microphysical_fields
    ρ_field = AM.total_density(model.dynamics)
    constants = model.thermodynamic_constants
    velocities = model.velocities

    launch!(arch, grid, :xyz,
            _p3_compute_and_cache_kernel!,
            μ, model.formulation, model.dynamics, grid, constants, p3, ρ_field,
            velocities, model.temperature)

    # The scalar advection operators interpolate terminal velocities through halo
    # cells. The preparation kernel overwrites interiors after update_state!'s generic
    # halo fill, so refresh these diagnostic halos before sedimentation is evaluated.
    sedimentation_velocities = (μ.wᶜˡ, μ.wᶜˡₙ, μ.wʳ, μ.wʳₙ,
                                μ.wⁱ, μ.wⁱₙ, μ.wⁱ_z, μ.wⁱ_z̃)
    fill_halo_regions!(sedimentation_velocities)

    return nothing
end

# P3 evolves through RK-stage tendencies; it has no full-Δt operator-split update.
AM.microphysics_model_update!(::P3, model) = nothing

@kernel function _p3_compute_and_cache_kernel!(μ, formulation, dynamics, grid, constants, p3,
                                               ρ_field, velocities, temperature_field)
    i, j, k = @index(Global, NTuple)

    @inbounds ρ = ρ_field[i, j, k]

    # Reconstruct thermodynamic state (same as in the thermodynamic kernel)
    qᵛᵉ = μ.qᵛ[i, j, k]
    # moisture_fractions does not read ℳ.w; pass ZeroField placeholders to skip the
    # ℑzᵃᵃᶜ interpolation here. The real velocities are forwarded to p3_compute_and_cache!.
    q = AM.moisture_fractions(p3, AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ,
            nothing, (u = ZeroField(), v = ZeroField(), w = ZeroField())), qᵛᵉ)
    𝒰₀ = AM.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
    𝒰 = AM.maybe_adjust_thermodynamic_state(𝒰₀, p3, qᵛᵉ, constants)

    # Materialize a true column-bottom temperature diagnostic, including
    # immersed-bottom lookup and vertical-partition communication. Local k=1 is
    # the Fortran kbot equivalent only on regular, vertically unpartitioned grids.
    @inbounds surface_temperature = temperature_field[i, j, 1]
    p3_compute_and_cache!(μ, i, j, k, grid, p3, ρ, 𝒰, constants, velocities,
                          surface_temperature)
end


#####
##### Fused tendency override (fast path for AtmosphereModel)
#####
#
# `prepare_microphysical_tendencies!` already wrote every cell's microphysics contribution
# into the `cache_ρ*` fields. The fused override simply `+=`s those cached values
# into `Gⁿ` in a single kernel launch after the dynamics tendency kernels run.
# The state-based `microphysical_tendency` methods above remain the gridless
# fallback used by ParcelModels.

@kernel function _add_p3_base_tendencies_kernel!(Gρqᵛ, Gρqᶜˡ, Gρnᶜˡ, Gρqʳ, Gρnʳ,
                                                 Gρqⁱ, Gρnⁱ, Gρqᶠ, Gρbᶠ,
                                                 Gρqʷⁱ, Gρsˢᵃᵗ, μ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Gρqᵛ[i, j, k]   += μ.cache_ρqᵛ[i, j, k]
        Gρqᶜˡ[i, j, k]  += μ.cache_ρqᶜˡ[i, j, k]
        Gρnᶜˡ[i, j, k]  += μ.cache_ρnᶜˡ[i, j, k]
        Gρqʳ[i, j, k]   += μ.cache_ρqʳ[i, j, k]
        Gρnʳ[i, j, k]   += μ.cache_ρnʳ[i, j, k]
        Gρqⁱ[i, j, k]   += μ.cache_ρqⁱ[i, j, k]
        Gρnⁱ[i, j, k]   += μ.cache_ρnⁱ[i, j, k]
        Gρqᶠ[i, j, k]   += μ.cache_ρqᶠ[i, j, k]
        Gρbᶠ[i, j, k]   += μ.cache_ρbᶠ[i, j, k]
        Gρqʷⁱ[i, j, k]  += μ.cache_ρqʷⁱ[i, j, k]
        Gρsˢᵃᵗ[i, j, k] += μ.cache_ρsˢᵃᵗ[i, j, k]
    end
end

@kernel function _add_p3_z̃ⁱ_tendency_kernel!(Gρz̃ⁱ, cache_ρz̃ⁱ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρz̃ⁱ[i, j, k] += cache_ρz̃ⁱ[i, j, k]
end

@kernel function _add_p3_aerosol_tendency_kernel!(Gρnᵃ, cache_ρnᵃ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρnᵃ[i, j, k] += cache_ρnᵃ[i, j, k]
end

function AM.compute_microphysical_tendencies!(p3::P3, model)
    grid = model.grid
    arch = grid.architecture
    G = model.timestepper.Gⁿ
    μ = model.microphysical_fields

    launch!(arch, grid, :xyz, _add_p3_base_tendencies_kernel!,
            G.ρqᵛ, G.ρqᶜˡ, G.ρnᶜˡ, G.ρqʳ, G.ρnʳ,
            G.ρqⁱ, G.ρnⁱ, G.ρqᶠ, G.ρbᶠ, G.ρqʷⁱ, G.ρsˢᵃᵗ, μ)

    if is_three_moment_ice(p3)
        launch!(arch, grid, :xyz, _add_p3_z̃ⁱ_tendency_kernel!, G.ρz̃ⁱ, μ.cache_ρz̃ⁱ)
    end

    if !isnothing(p3.aerosol)
        launch!(arch, grid, :xyz, _add_p3_aerosol_tendency_kernel!, G.ρnᵃ, μ.cache_ρnᵃ)
    end

    return nothing
end

#####
##### Number concentration diagnostic
#####
#
# P3 carries prognostic number-density fields for cloud liquid, rain, and ice,
# so `number_concentration` just hands the requested field back. This keeps the
# diagnostic interface uniform with `OneMomentCloudMicrophysics` and
# `TwoMomentCloudMicrophysics`.

Microphysics.number_concentration(model, ::P3, ::Val{:rain}) =
    get(model.microphysical_fields, :ρnʳ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val{:cloud_liquid}) =
    get(model.microphysical_fields, :ρnᶜˡ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val{:ice}) =
    get(model.microphysical_fields, :ρnⁱ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val) = nothing
