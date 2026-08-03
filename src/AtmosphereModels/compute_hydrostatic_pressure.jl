#####
##### Compute hydrostatic pressure
#####

using ..Thermodynamics: dry_air_gas_constant, column_surface_pressure,
                        surface_pressure_from_cell_center
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center

@kernel function _compute_diagnostic_surface_pressure!(pˢ, grid, p, ρ, g)
    i, j = @index(Global, NTuple)
    Δz = Δzᶜᶜᶜ(i, j, 1, grid)
    @inbounds pˢ[i, j, 1] = surface_pressure_from_cell_center(p[i, j, 1], ρ[i, j, 1], Δz, g)
end

"""
$(TYPEDSIGNATURES)

Diagnose the live pressure at the bottom face of every model column by hydrostatically
extrapolating the first-cell pressure downward by half a cell. The result is a 2D field
and is independent of the reference-pressure datum convention.
"""
function diagnostic_surface_pressure(model)
    grid = model.grid
    arch = grid.architecture
    pˢ = Field{Center, Center, Nothing}(grid)
    p = total_pressure(model.dynamics)
    ρ = total_density(model.dynamics)
    g = model.thermodynamic_constants.gravitational_acceleration

    launch!(arch, grid, :xy, _compute_diagnostic_surface_pressure!, pˢ, grid, p, ρ, g)
    fill_halo_regions!(pˢ)
    return pˢ
end

@kernel function _compute_hydrostatic_pressure!(ph, grid, pˢ, temperature, constants)
    i, j = @index(Global, NTuple)

    # The integration starts at the bottom face of this column, so it must start from the pressure
    # *there*, not from the z = 0 datum. On a terrain-following grid the bottom face is the terrain
    # surface and the two differ by O(ρgh) per column; on a height-coordinate grid `pˢ` is a scalar.
    p₀ = column_surface_pressure(pˢ, i, j)
    Nz = grid.Nz
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)

    @inbounds begin
        # Start with pressure at bottom interface
        p_interface_bottom = p₀

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

    pˢ = diagnostic_surface_pressure(model)

    launch!(arch, grid, :xy, _compute_hydrostatic_pressure!,
            ph, grid, pˢ, model.temperature, model.thermodynamic_constants)

    fill_halo_regions!(ph)

    return ph
end
