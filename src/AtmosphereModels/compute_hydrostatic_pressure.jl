#####
##### Compute hydrostatic pressure
#####

using ..Thermodynamics: dry_air_gas_constant
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!

@kernel function _compute_hydrostatic_pressure!(ph, grid, formulation, temperature, constants)
    i, j = @index(Global, NTuple)

    p₀ = formulation.reference_state.surface_pressure
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

    launch!(arch, grid, :xy, _compute_hydrostatic_pressure!,
            ph, grid, model.formulation, model.temperature, model.thermodynamic_constants)

    fill_halo_regions!(ph)

    return ph
end
