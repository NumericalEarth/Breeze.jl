#####
##### Compute hydrostatic pressure
#####

using ..Thermodynamics: dry_air_gas_constant, surface_pressure_from_cell_center
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!

@kernel function _compute_hydrostatic_pressure!(ph, grid, p, ρ, temperature, constants)
    i, j = @index(Global, NTuple)

    Nz = grid.Nz
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)

    @inbounds begin
        # The integration starts at the bottom face of this column, so it must start from the
        # pressure *there*, not from the z = 0 datum: on a terrain-following grid the bottom face
        # is the terrain surface and the two differ by O(ρgh) per column. Diagnose it from the live
        # state by extrapolating the first cell center down half a cell, which is independent of
        # where the reference-pressure datum sits.
        p_interface_bottom = surface_pressure_from_cell_center(p[i, j, 1], ρ[i, j, 1],
                                                              Δzᶜᶜᶜ(i, j, 1, grid), g)

        # Compute cell-mean pressure and interface pressures in a single pass
        for k in 1:Nz
            T_k = temperature[i, j, k]
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            H = Rᵈ * T_k / g

            # Compute cell-mean pressure analytically for an isothermal grid
            ph[i, j, k] = p_interface_bottom * (H / Δz) * (1 - exp(-Δz / H))

            # Compute pressure at top interface of this cell (becomes bottom for next cell)
            p_interface_bottom = exp(-Δz / H) * p_interface_bottom
        end
    end
end

function compute_hydrostatic_pressure!(ph, model)
    grid = model.grid
    arch = grid.architecture

    launch!(arch, grid, :xy, _compute_hydrostatic_pressure!,
            ph, grid, dynamics_pressure(model.dynamics), total_density(model.dynamics),
            model.temperature, model.thermodynamic_constants)

    fill_halo_regions!(ph)

    return ph
end
