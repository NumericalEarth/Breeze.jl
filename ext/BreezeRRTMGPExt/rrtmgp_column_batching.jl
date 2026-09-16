#####
##### Column batching for the RRTMGP solvers
#####
#
# Radiative transfer is column-local, so columns can be solved in batches sharing one workspace.
# `RRTMGPGridParams(; ncol)` sizes that workspace (~39 nlay floats per column, ~12 kB at nlay = 40
# in Float64); the state and boundary conditions stay full-size (~13 nlay), so only the workspace
# shrinks.
#
# A batch is a contiguous run of whole latitude rows: `rrtmgp_column_index` is `i + (j - 1) * Nx`,
# so a j-slab is a unit-stride `view`. On the GPU such a view of a `CuArray` is itself a
# `CuArray`, so the solver kernels see what they would see unbatched.

using RRTMGP: update_lw_fluxes!, update_sw_fluxes!
using RRTMGP.AtmosphericStates: AerosolState, AtmosphericState, CloudState
using RRTMGP.BCs: LwBCs, SwBCs
using RRTMGP.Fluxes: update_presentation!
using RRTMGP.RTE: TwoStreamLWRTE, TwoStreamSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!
using RRTMGP.VolumeMixingRatios: VmrGM

"""
$(TYPEDSIGNATURES)

Number of whole latitude rows per radiation batch.

`column_batch_size` is a requested number of *columns*, rounded up to whole rows and then to a
divisor of `Ny`. `nothing` means "one batch", the unbatched default.

Equal-width batches are required: the solver's `TransposedStateCache` is sized from
`RRTMGPGridParams(; ncol)` and refreshed with `permutedims!`, which throws `DimensionMismatch`
on a short final batch.
"""
function resolve_column_batch_rows(column_batch_size, Nx, Ny)
    isnothing(column_batch_size) && return Ny

    column_batch_size > 0 ||
        throw(ArgumentError("column_batch_size must be positive, got $column_batch_size"))

    rows = cld(column_batch_size, Nx)
    rows ≥ Ny && return Ny

    # Round up to a divisor of Ny; the guard above keeps this finite.
    while Ny % rows != 0
        rows += 1
    end

    # Few divisors push the batch past the request, at the limit back to one full-size batch.
    if rows * Nx > 2 * column_batch_size
        @warn "column_batch_size = $column_batch_size was rounded up to $(rows * Nx) columns " *
              "($rows of $Ny latitude rows) so that every batch is the same width. " *
              "An Ny with more divisors lands closer to the request."
    end

    return rows
end

#####
##### Column views of RRTMGP's state types
#####
#
# Each rebuilds the struct around `view`s of the column dimension; non-column members pass
# through unchanged.
#
# TODO: these transcribe RRTMGP's positional constructors, coupling us to its field order.
# RRTMGP has no column-subsetting API; the fix is upstream — a way to rebind `bcs` on an existing
# workspace, or a column-range argument on `update_lw_fluxes!`/`update_sw_fluxes!`.

@inline column_view(::Nothing, columns) = nothing

# Column index is the trailing dimension, except for the incident-flux fields below.
@inline column_view(a::AbstractVector, columns) = view(a, columns)
@inline column_view(a::AbstractMatrix, columns) = view(a, :, columns)
@inline column_view(a::AbstractArray{<:Any, 3}, columns) = view(a, :, :, columns)

# `inc_flux` and `inc_flux_diffuse` are `(ncol, ngpt)` — column *first*; RRTMGP indexes them
# `inc_flux[gcol, igpt]`.
@inline incident_flux_view(::Nothing, columns) = nothing
@inline incident_flux_view(a::AbstractMatrix, columns) = view(a, columns, :)

@inline column_view(vmr::VmrGM, columns) =
    VmrGM(column_view(vmr.vmr_h2o, columns),
          column_view(vmr.vmr_o3, columns),
          vmr.vmr)                              # per gas, not per column

@inline column_view(cloud_state::CloudState, columns) =
    CloudState(column_view(cloud_state.cld_r_eff_liq, columns),
               column_view(cloud_state.cld_r_eff_ice, columns),
               column_view(cloud_state.cld_path_liq, columns),
               column_view(cloud_state.cld_path_ice, columns),
               column_view(cloud_state.cld_frac, columns),
               column_view(cloud_state.cld_cover_sw, columns),
               column_view(cloud_state.cld_cover_lw, columns),
               column_view(cloud_state.mask_lw, columns),
               column_view(cloud_state.mask_sw, columns),
               cloud_state.mask_type,
               cloud_state.ice_rgh)

@inline column_view(aerosol_state::AerosolState, columns) =
    AerosolState(column_view(aerosol_state.aod_sw_ext, columns),
                 column_view(aerosol_state.aod_sw_sca, columns),
                 column_view(aerosol_state.aero_mask, columns),
                 column_view(aerosol_state.aero_size, columns),
                 column_view(aerosol_state.aero_mass, columns))

@inline column_view(as::AtmosphericState, columns) =
    AtmosphericState(column_view(as.lon, columns),
                     column_view(as.lat, columns),
                     column_view(as.layerdata, columns),
                     column_view(as.p_lev, columns),
                     column_view(as.t_lev, columns),
                     column_view(as.t_sfc, columns),
                     column_view(as.vmr, columns),
                     column_view(as.cloud_state, columns),
                     column_view(as.aerosol_state, columns))

@inline column_view(bcs::LwBCs, columns) =
    LwBCs(column_view(bcs.sfc_emis, columns),
          incident_flux_view(bcs.inc_flux, columns))

@inline column_view(bcs::SwBCs, columns) =
    SwBCs(column_view(bcs.cos_zenith, columns),
          column_view(bcs.toa_flux, columns),
          column_view(bcs.sfc_alb_direct, columns),
          incident_flux_view(bcs.inc_flux_diffuse, columns),
          column_view(bcs.sfc_alb_diffuse, columns))

# Only `bcs` is sliced: the optics, sources, flux buffers and cache are the shared batch-width
# workspace, and `columns` is a global range. Allocates nothing.
@inline column_view(lws::TwoStreamLWRTE, columns) =
    TwoStreamLWRTE(lws.context, lws.op, lws.src, column_view(lws.bcs, columns),
                   lws.fluxb, lws.flux, lws.band_flux, lws.state_cache)

@inline column_view(sws::TwoStreamSWRTE, columns) =
    TwoStreamSWRTE(sws.context, sws.op, sws.src, column_view(sws.bcs, columns),
                   sws.fluxb, sws.flux, sws.band_flux, sws.state_cache)

"""
$(TYPEDSIGNATURES)

Latitude rows per batch. The workspace is allocated from `RRTMGPGridParams(; ncol)`, so `ncol`
is the batch width.
"""
column_batch_rows(solver, Nx) = solver.grid_params.ncol ÷ Nx

"""
$(TYPEDSIGNATURES)

Solve longwave and shortwave over each batch of columns in turn and copy the fluxes into the
Oceananigans fields.
"""
function solve_radiation_batches!(rtm, solver, grid)
    Nx, Ny, _ = size(grid)
    batch_rows = column_batch_rows(solver, Nx)

    # Unbatched default: RRTMGP's own entry points, which dispatch on `radiation_method`. Keeps
    # this path identical to pre-batching, with no second CPU specialization for `SubArray` state.
    if batch_rows == Ny
        update_lw_fluxes!(solver)
        update_sw_fluxes!(solver)
        copy_rrtmgp_fluxes_to_fields!(rtm, solver, grid, Ny, 0)
        return nothing
    end

    # TODO: open-codes `update_lw_fluxes!(solver, ::AllSkyRadiation)` to inject a column view, so
    # the all-sky lookup triple is hardcoded and clear-sky cannot reuse this loop.
    lookups = solver.lookups
    scaling = solver.deep_atmosphere_inverse_scaling

    for j_offset in 0:batch_rows:(Ny - 1)
        columns = (j_offset * Nx + 1):((j_offset + batch_rows) * Nx)

        as = column_view(solver.as, columns)
        batch_scaling = column_view(scaling, columns)

        solve_lw!(column_view(solver.lws, columns), as,
                  lookups.lookup_lw, lookups.lookup_lw_cld, lookups.lookup_lw_aero,
                  batch_scaling)
        update_presentation!(solver.presented_flux_lw, solver.lws.flux)

        solve_sw!(column_view(solver.sws, columns), as,
                  lookups.lookup_sw, lookups.lookup_sw_cld, lookups.lookup_sw_aero,
                  batch_scaling)
        update_presentation!(solver.presented_flux_sw, solver.sws.flux)

        copy_rrtmgp_fluxes_to_fields!(rtm, solver, grid, batch_rows, j_offset)
    end

    return nothing
end
