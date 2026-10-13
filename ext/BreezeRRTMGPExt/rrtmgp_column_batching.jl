#####
##### Column batching for the RRTMGP solvers
#####
#
# Radiative transfer is column-local, so columns can be solved in batches sharing one workspace.
# `RRTMGPGridParams(; ncol)` sizes that workspace (~39 nlay floats per column, ~12 kB at nlay = 40
# in Float64); the state and boundary conditions stay full-size (~13 nlay), so only the workspace
# shrinks.
#
# A batch is a contiguous run of whole j-rows: `rrtmgp_column_index` is `i + (j - 1) * Nx`, so a
# j-slab is a unit-stride `view`. On the GPU such a view of a `CuArray` is itself a `CuArray`, so
# the solver kernels see what they would see unbatched.

using RRTMGP: AllSkyRadiation, ClearSkyRadiation
using RRTMGP.AtmosphericStates: AerosolState, AtmosphericState, CloudState
using RRTMGP.BCs: LwBCs, SwBCs
using RRTMGP.Fluxes: update_presentation!
using RRTMGP.RTE: TwoStreamLWRTE, TwoStreamSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!
using RRTMGP.VolumeMixingRatios: VmrGM

# Shared by the clear-sky and all-sky `RadiativeTransferModel` docstrings.
const column_batches_docstring = """
- `column_batches`: Number of batches to split the radiation solve into, trading speed for memory
  (default: `nothing`, one unbatched solve). The solver workspace, RRTMGP's dominant allocation,
  is then sized for one batch, so it shrinks by about that factor. Batches are whole j-rows of
  `cld(Ny, column_batches)` rows each, so the domain may need fewer batches than requested; when
  the rows do not divide `Ny`, the final batch overlaps the one before it. On GPUs the solver runs
  one thread per column, so batches of more than ~10⁵ columns cost nothing extra, while smaller
  batches take about as long as a full solve each."""

"""
$(TYPEDSIGNATURES)

RRTMGP grid parameters for `grid` with `ncol` set to the width of one batch of columns. These size
the solver workspace, which every batch shares; the state arrays stay sized for the whole domain.
"""
function rrtmgp_grid_params(FT, context, grid, column_batches)
    Nx, Ny, Nz = size(grid)
    batch_rows = resolve_column_batch_rows(column_batches, Ny)
    return RRTMGPGridParams(FT; context, domain_nlay=Nz, ncol=Nx * batch_rows)
end

#####
##### Batch views of RRTMGP's state types
#####
#
# Each rebuilds the struct around `view`s of the column dimension; non-column members pass
# through unchanged.
#
# TODO: these transcribe RRTMGP's positional constructors, coupling us to its field order.
# RRTMGP has no column-subsetting API; the fix is upstream — a way to rebind `bcs` on an existing
# workspace, or a column-range argument on `update_lw_fluxes!`/`update_sw_fluxes!`.

batch_view(::Nothing, columns) = nothing

# Column index is the trailing dimension, except for the incident-flux fields below.
batch_view(a::AbstractVector, columns) = view(a, columns)
batch_view(a::AbstractMatrix, columns) = view(a, :, columns)
batch_view(a::AbstractArray{<:Any, 3}, columns) = view(a, :, :, columns)

# `inc_flux` and `inc_flux_diffuse` are `(ncol, ngpt)` — column *first*; RRTMGP indexes them
# `inc_flux[gcol, igpt]`.
incident_flux_batch_view(::Nothing, columns) = nothing
incident_flux_batch_view(a::AbstractMatrix, columns) = view(a, columns, :)

batch_view(vmr::VmrGM, columns) =
    VmrGM(batch_view(vmr.vmr_h2o, columns),
          batch_view(vmr.vmr_o3, columns),
          vmr.vmr)                              # per gas, not per column

batch_view(cloud_state::CloudState, columns) =
    CloudState(batch_view(cloud_state.cld_r_eff_liq, columns),
               batch_view(cloud_state.cld_r_eff_ice, columns),
               batch_view(cloud_state.cld_path_liq, columns),
               batch_view(cloud_state.cld_path_ice, columns),
               batch_view(cloud_state.cld_frac, columns),
               batch_view(cloud_state.cld_cover_sw, columns),
               batch_view(cloud_state.cld_cover_lw, columns),
               batch_view(cloud_state.mask_lw, columns),
               batch_view(cloud_state.mask_sw, columns),
               cloud_state.mask_type,
               cloud_state.ice_rgh)

batch_view(aerosol_state::AerosolState, columns) =
    AerosolState(batch_view(aerosol_state.aod_sw_ext, columns),
                 batch_view(aerosol_state.aod_sw_sca, columns),
                 batch_view(aerosol_state.aero_mask, columns),
                 batch_view(aerosol_state.aero_size, columns),
                 batch_view(aerosol_state.aero_mass, columns))

batch_view(as::AtmosphericState, columns) =
    AtmosphericState(batch_view(as.lon, columns),
                     batch_view(as.lat, columns),
                     batch_view(as.layerdata, columns),
                     batch_view(as.p_lev, columns),
                     batch_view(as.t_lev, columns),
                     batch_view(as.t_sfc, columns),
                     batch_view(as.vmr, columns),
                     batch_view(as.cloud_state, columns),
                     batch_view(as.aerosol_state, columns))

batch_view(bcs::LwBCs, columns) =
    LwBCs(batch_view(bcs.sfc_emis, columns),
          incident_flux_batch_view(bcs.inc_flux, columns))

batch_view(bcs::SwBCs, columns) =
    SwBCs(batch_view(bcs.cos_zenith, columns),
          batch_view(bcs.toa_flux, columns),
          batch_view(bcs.sfc_alb_direct, columns),
          incident_flux_batch_view(bcs.inc_flux_diffuse, columns),
          batch_view(bcs.sfc_alb_diffuse, columns))

# Only `bcs` is sliced: the optics, sources, flux buffers and cache are the shared batch-width
# workspace, and `columns` is a global range. Allocates nothing.
batch_view(lws::TwoStreamLWRTE, columns) =
    TwoStreamLWRTE(lws.context, lws.op, lws.src, batch_view(lws.bcs, columns),
                   lws.fluxb, lws.flux, lws.band_flux, lws.state_cache)

batch_view(sws::TwoStreamSWRTE, columns) =
    TwoStreamSWRTE(sws.context, sws.op, sws.src, batch_view(sws.bcs, columns),
                   sws.fluxb, sws.flux, sws.band_flux, sws.state_cache)

# The (gas, cloud, aerosol) lookup tables each radiation method hands `solve_lw!`/`solve_sw!`,
# transcribed from RRTMGP's `update_lw_fluxes!(solver, method)` methods. Clear-sky has no cloud
# optics.
longwave_lookups(lookups, ::ClearSkyRadiation) = (lookups.lookup_lw, nothing, lookups.lookup_lw_aero)
longwave_lookups(lookups, ::AllSkyRadiation) = (lookups.lookup_lw, lookups.lookup_lw_cld, lookups.lookup_lw_aero)
shortwave_lookups(lookups, ::ClearSkyRadiation) = (lookups.lookup_sw, nothing, lookups.lookup_sw_aero)
shortwave_lookups(lookups, ::AllSkyRadiation) = (lookups.lookup_sw, lookups.lookup_sw_cld, lookups.lookup_sw_aero)

"""
$(TYPEDSIGNATURES)

Solve longwave and shortwave over each batch of columns in turn and copy the fluxes into the
Oceananigans fields.

Every batch must span the full workspace width: the solver's `TransposedStateCache` is refreshed
with `permutedims!`, which throws `DimensionMismatch` on a narrower batch. So when the batch rows
do not divide `Ny`, the final batch is shifted back to end at row `Ny`, re-solving rows the
previous batch already covered.
"""
function solve_radiation_batches!(rtm, solver, grid)
    Nx, Ny, _ = size(grid)

    # `column_batches = nothing` makes `with_batching` a compile-time `false`, so the unbatched
    # solve never compiles the batched branch's `SubArray` specializations of the RRTMGP kernels.
    if with_batching(rtm.column_batches)
        batch_rows = resolve_column_batch_rows(rtm.column_batches, Ny)

        for j in 0:batch_rows:(Ny - 1)
            j_offset = min(j, Ny - batch_rows)
            columns = (j_offset * Nx + 1):((j_offset + batch_rows) * Nx)

            solve_radiation_batch!(rtm, solver, grid,
                                   batch_view(solver.as, columns),
                                   batch_view(solver.lws, columns),
                                   batch_view(solver.sws, columns),
                                   batch_view(solver.deep_atmosphere_inverse_scaling, columns),
                                   batch_rows, j_offset)
        end
    else
        solve_radiation_batch!(rtm, solver, grid, solver.as, solver.lws, solver.sws,
                               solver.deep_atmosphere_inverse_scaling, Ny, 0)
    end

    return nothing
end

with_batching(::Nothing) = false
with_batching(::Integer) = true

# Solve one batch: `as`, `lws`, `sws` and `scaling` are the solver's own (unbatched) or batch
# views of them, and `batch_rows` and `j_offset` locate the batch's j-rows in the grid.
#
# TODO: open-codes `update_lw_fluxes!`/`update_sw_fluxes!` to inject a batch view, so
# `longwave_lookups`/`shortwave_lookups` must track RRTMGP's per-method dispatch.
function solve_radiation_batch!(rtm, solver, grid, as, lws, sws, scaling, batch_rows, j_offset)
    method = solver.radiation_method

    solve_lw!(lws, as, longwave_lookups(solver.lookups, method)..., scaling)
    update_presentation!(solver.presented_flux_lw, solver.lws.flux)

    solve_sw!(sws, as, shortwave_lookups(solver.lookups, method)..., scaling)
    update_presentation!(solver.presented_flux_sw, solver.sws.flux)

    copy_rrtmgp_fluxes_to_fields!(rtm, solver, grid, batch_rows, j_offset)

    return nothing
end
