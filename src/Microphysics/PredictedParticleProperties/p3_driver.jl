using Oceananigans: CenterField
using Oceananigans.Fields: ZeroField
using DocStringExtensions: TYPEDSIGNATURES

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

using Breeze.Thermodynamics: MoistureMassFractions

using Breeze: Microphysics

const P3 = PredictedParticlePropertiesMicrophysics

#####
##### Model update
#####

"""
$(TYPEDSIGNATURES)

Apply P3 model update during state update phase.

Launches a separate GPU kernel to compute terminal velocities, process rates,
and tendency cache fields. This heavy computation is split out of the thermodynamic
variables kernel to avoid overwhelming the GPU compiler with force-inlined P3 physics
(~1000 lines of code). The lighter `update_microphysical_auxiliaries!` only writes
basic diagnostic quantities in the thermodynamic kernel.
"""
function AM.microphysics_model_update!(p3::P3, model)
    grid = model.grid
    arch = grid.architecture
    μ = model.microphysical_fields
    ρ_field = AM.dynamics_density(model.dynamics)
    constants = model.thermodynamic_constants

    launch!(arch, grid, :xyz,
            _p3_compute_and_cache_kernel!,
            μ, model.formulation, model.dynamics, grid, constants, p3, ρ_field)

    return nothing
end

using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

@kernel function _p3_compute_and_cache_kernel!(μ, formulation, dynamics, grid, constants, p3, ρ_field)
    i, j, k = @index(Global, NTuple)

    @inbounds ρ = ρ_field[i, j, k]

    # Reconstruct thermodynamic state (same as in the thermodynamic kernel)
    ρqᵛᵉ = μ.qᵛ[i, j, k] * ρ  # qᵛ was already written by update_microphysical_auxiliaries!
    qᵛᵉ = μ.qᵛ[i, j, k]
    q = AM.moisture_fractions(p3, AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ,
            nothing, (; u=zero(ρ), v=zero(ρ), w=zero(ρ))), qᵛᵉ)
    𝒰₀ = AM.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
    𝒰 = AM.maybe_adjust_thermodynamic_state(𝒰₀, p3, qᵛᵉ, constants)

    p3_compute_and_cache!(μ, i, j, k, grid, p3, ρ, 𝒰, constants)
end


#####
##### Fused tendency override (fast path for AtmosphereModel)
#####
#
# `microphysics_model_update!` already wrote every cell's microphysics contribution
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
