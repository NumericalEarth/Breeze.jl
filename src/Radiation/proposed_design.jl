# Minimal RRTMGP↔Oceananigans interface sketch.
# Focus: allocate once, preset gas composition, then per-call compute optics + fluxes.

module ProposedDesign

using Oceananigans
using RRTMGP

"""
    struct RRTMGPSkeleton

Holds everything we pre-allocate once: contexts, solver, and flux buffers.
"""
struct RRTMGPSkeleton
    grid_params::RRTMGP.RRTMGPGridParams
    solver::RRTMGP.RRTMGPSolver
    atmos::RRTMGP.AtmosphericStates.AtmosphericState
end

"""
    initialize_skeleton(arch, grid; vmr_background)

1. Allocate RRTMGP state + Oceananigans flux fields.
2. Write constant/background gas concentrations into the VMR slots.
"""
function initialize_skeleton(arch, grid; vmr_background = Dict("n2" => 0.7808))
    # -- allocate memory ----------------------------------------------------
    context = _context_from_arch(arch)
    grid_params = RRTMGP.RRTMGPGridParams(eltype(grid); context, nlay = grid.Nz, ncol = grid.Nx * grid.Ny)
    atmos = _allocate_atmospheric_state(grid_params)
    solver = _allocate_solver(grid_params, atmos)

    # -- set gas concentrations --------------------------------------------
    _apply_background_vmr!(solver, atmos, vmr_background)

    # -- Oceananigans flux fields ------------------------------------------
    flux_lw_up = ZFaceField(grid)
    flux_lw_down = ZFaceField(grid)
    flux_sw_up = ZFaceField(grid)
    flux_sw_down = ZFaceField(grid)

    return RRTMGPSkeleton(grid_params, solver, atmos, flux_lw_up, flux_lw_down, flux_sw_up, flux_sw_down)
end

"""
    update_and_solve!(skel, column)

Per call:
1. Fill thermodynamic profiles (p, T, vmr) into `skel.atmos`.
2. Call RRTMGP to compute optical properties.
3. Call the RRTMGP solvers to get fluxes.
4. Copy results into Oceananigans fields.
"""
function update_and_solve!(skel::RRTMGPSkeleton, column)
    # -- compute optical properties of the gaseous atmosphere --------------
    _populate_atmos!(skel.atmos, column)
    _compute_optics!(skel.solver, skel.atmos)

    # -- compute radiative fluxes ------------------------------------------
    RRTMGP.update_lw_fluxes!(skel.solver)
    RRTMGP.update_sw_fluxes!(skel.solver)

    _copy_fluxes!(skel, skel.solver)
    return nothing
end

# -- helpers (left unimplemented; fill in with real code when wiring up) ----

_context_from_arch(arch) = error("TODO: return ClimaComms context for $arch")
_allocate_atmospheric_state(grid_params) = error("TODO: allocate AtmosphericState arrays")
_allocate_solver(grid_params, atmos) = error("TODO: call RRTMGPSolver with desired method")
_apply_background_vmr!(solver, atmos, vmr_background) = error("TODO: map species names to indices")
_populate_atmos!(atmos, column) = error("TODO: zero-copy views into column arrays")
_compute_optics!(solver, atmos) = error("TODO: loop over g-points and call compute_optical_props!")
_copy_fluxes!(skel, solver) = error("TODO: write solver fluxes into ZFaceField interiors")

end # module ProposedDesign
