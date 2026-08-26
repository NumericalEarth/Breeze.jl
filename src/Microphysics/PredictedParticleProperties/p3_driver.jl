using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Center, ZeroField
using Oceananigans.Grids: inactive_cell
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: Utils, launch!

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

using Breeze: Microphysics

using KernelAbstractions: @kernel, @index

const P3 = PredictedParticlePropertiesMicrophysics

# P3 evolves through RK-stage tendencies; it has no full-Δt operator-split update.
AM.microphysics_model_update!(::P3, model) = nothing

#####
##### Surface air temperature
#####
##### Hallett-Mossop rime splintering is switched off over a warm surface
##### (`maximum_splintering_surface_temperature`); that shutoff is the only part of the
##### scheme that needs a column-bottom value rather than a local one, so the bottom-most
##### active temperature of each column is gathered once per stage into `μ.surface_temperature`.
#####

@kernel function _p3_surface_temperature_kernel!(surface_temperature, temperature_field)
    i, j = @index(Global, NTuple)
    @inbounds surface_temperature[i, j, 1] = temperature_field[i, j, 1]
end

# With an immersed bottom, k = 1 can sit inside the topography, so the lowest active cell
# has to be found by scanning up the column. Branch-free, so it still compiles for the GPU.
@kernel function _p3_immersed_surface_temperature_kernel!(surface_temperature, temperature_field, grid)
    i, j = @index(Global, NTuple)

    FT = eltype(grid)
    bottom_temperature = zero(FT)
    found_active_cell = false

    for k in 1:grid.Nz
        active_cell = !inactive_cell(i, j, k, grid)
        use_this_cell = active_cell & !found_active_cell
        @inbounds local_temperature = temperature_field[i, j, k]
        bottom_temperature = ifelse(use_this_cell, local_temperature, bottom_temperature)
        found_active_cell = found_active_cell | active_cell
    end

    @inbounds surface_temperature[i, j, 1] = bottom_temperature
end

"""
$(TYPEDSIGNATURES)

Fill `surface_temperature` with the air temperature of the bottom-most active cell in
each column. Without an immersed boundary every column is active down to `k = 1`, so
this is a plain copy; `ImmersedBoundaryGrid` dispatches to a column scan.
"""
function compute_p3_surface_temperature!(surface_temperature, temperature_field, grid)
    launch!(grid.architecture, grid, :xy, _p3_surface_temperature_kernel!,
            surface_temperature, temperature_field)
    return nothing
end

# TODO: Add a vertically distributed column reduction in Oceananigans. Its distributed
# top/bottom halo fills are currently no-ops, so this column scan is correct for serial
# and horizontally partitioned grids, but cannot broadcast across a z partition.
function compute_p3_surface_temperature!(surface_temperature, temperature_field,
                                         grid::ImmersedBoundaryGrid)
    launch!(grid.architecture, grid, :xy, _p3_immersed_surface_temperature_kernel!,
            surface_temperature, temperature_field, grid)
    return nothing
end

#####
##### Process-rate cache
#####
##### "The P3 tendencies" are the tendencies of P3's own microphysical prognostics
##### (ρqᵛ, ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρqʷⁱ, and the optional ρnᶜˡ/ρnᵃ/ρsᵛ⁺ˡ).
##### They are computed *jointly*: the coupled donor-budget limiters see every species at
##### once, so one kernel evaluates all of them per cell and adds each straight into `Gⁿ`.
#####

@kernel function _p3_add_tendencies_kernel!(G, μ, formulation, dynamics, grid, constants, p3, ρ_field, velocities)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = ρ_field[i, j, k]
        qᵛᵉ = μ.qᵛ[i, j, k]
        ℳ = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, nothing, velocities)
        q = AM.moisture_fractions(p3, ℳ, qᵛᵉ)
        𝒰₀ = AM.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
        𝒰 = AM.maybe_adjust_thermodynamic_state(𝒰₀, p3, qᵛᵉ, constants)

        # Approximate the resolved diffusional-growth driver with adiabatic
        # cooling from the local vertical velocity. External vapor forcing is
        # intentionally neglected.
        temperature_tendency = p3_adiabatic_temperature_tendency(ℳ, 𝒰, constants)
        vapor_tendency = zero(temperature_tendency)

        surface_temperature = μ.surface_temperature[i, j, 1]
    end

    properties = p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
    result = p3_tendency_compute(p3, ρ, ℳ, 𝒰, constants, properties,
                                 surface_temperature, temperature_tendency,
                                 vapor_tendency)
    add_p3_tendencies!(G, i, j, k, p3, result)
end

#####
##### Fused tendency override (fast path for AtmosphereModel)
#####
#
# P3 evaluates its process rates from the current state with the adiabatic-only
# diffusional-growth driver described above, and adds them to `Gⁿ` in the same kernel.
# P3 sedimentation is assembled separately by the scalar transport operators using the
# fall speeds `update_microphysical_auxiliaries!` established during `update_state!`.
# The state-based `microphysical_tendency` methods above remain the gridless fallback
# used by ParcelModels.

# Just the `Gⁿ` slots P3 writes, not the model's whole tendency tuple. Dispatch is on a
# *type*, so the reduced tuple is a compile-time constant.
@inline p3_tendency_fields(G, p3::P3) =
    merge((; G.ρqᵛ, G.ρqᶜˡ, G.ρqʳ, G.ρnʳ, G.ρqⁱ, G.ρnⁱ, G.ρqᶠ, G.ρbᶠ, G.ρqʷⁱ),
          p3_aerosol_tendency_fields(G, p3.aerosol),
          p3_supersaturation_tendency_fields(G, p3.process_rates))

@inline p3_aerosol_tendency_fields(G, ::Nothing) = (;)
@inline p3_aerosol_tendency_fields(G, _) = (; G.ρnᶜˡ, G.ρnᵃ)

@inline p3_supersaturation_tendency_fields(G, ::ProcessRateParameters{FT, false}) where FT = (;)
@inline p3_supersaturation_tendency_fields(G, ::ProcessRateParameters{FT, true}) where FT = (; G.ρsᵛ⁺ˡ)

function AM.compute_microphysical_tendencies!(p3::P3, model)
    grid = model.grid
    arch = grid.architecture
    μ = model.microphysical_fields

    ρ_field = AM.total_density(model.dynamics)

    compute_p3_surface_temperature!(μ.surface_temperature, model.temperature, grid)

    launch!(arch, grid, :xyz, _p3_add_tendencies_kernel!,
            p3_tendency_fields(model.timestepper.Gⁿ, p3),
            μ, model.formulation, model.dynamics, grid,
            model.thermodynamic_constants, p3, ρ_field,
            model.velocities)

    return nothing
end

#####
##### Number concentration diagnostic
#####
#
# P3 carries prognostic number-density fields for rain and ice, and for cloud liquid when
# aerosol activation is enabled. The default prescribed-Nᶜˡ path has no `ρnᶜˡ` prognostic,
# but cloud droplets are still part of the model: their total number concentration is the
# constant `p3.cloud.number_concentration`. Represent that constant as a lazy operation so
# `number_concentration_field` works uniformly.

struct PrescribedCloudNumberKernelFunction{FT}
    number_concentration :: FT
end

Utils.prettysummary(::PrescribedCloudNumberKernelFunction) =
    "PrescribedCloudNumberKernelFunction"

@inline (kernel::PrescribedCloudNumberKernelFunction)(i, j, k, grid) =
    kernel.number_concentration

Microphysics.number_concentration(model, ::P3, ::Val{:rain}) =
    get(model.microphysical_fields, :ρnʳ, nothing)

Microphysics.number_concentration(model, p3::P3, ::Val{:cloud_liquid}) =
    cloud_number_concentration(model, p3, p3.aerosol)

function cloud_number_concentration(model, p3, ::Nothing)
    kernel = PrescribedCloudNumberKernelFunction(p3.cloud.number_concentration)
    return KernelFunctionOperation{Center, Center, Center}(kernel, model.grid)
end

cloud_number_concentration(model, p3, aerosol) =
    get(model.microphysical_fields, :ρnᶜˡ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val{:ice}) =
    get(model.microphysical_fields, :ρnⁱ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val) = nothing
