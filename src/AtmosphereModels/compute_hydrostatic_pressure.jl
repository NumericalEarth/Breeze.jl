#####
##### update pressure
#####

using Oceananigans: Center, Face
using Oceananigans.Grids: topology
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ℑzᵃᵃᶠ
using Oceananigans.Utils: KernelParameters
using Oceananigans.BoundaryConditions: fill_halo_regions!

const c = Center()
const f = Face()

@kernel function _compute_hydrostatic_pressure!(ph, grid, args...)
    i, j = @index(Global, NTuple)

    Nz = grid.Nz
    bᴺ = ℑzᵃᵃᶠ(i, j, Nz+1, grid, ρ_bᶜᶜᶜ, args...)
    @inbounds ph[i, j, Nz] = - bᴺ * Δzᶜᶜᶠ(i, j, Nz+1, grid)

    # Integrate downwards
    @inbounds for k in grid.Nz-1:-1:1
        b⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ_bᶜᶜᶜ, args...)
        Δp′ = b⁺ * Δzᶜᶜᶠ(i, j, k+1, grid)
        ph[i, j, k] = ph[i, j, k+1] - Δp′
    end
end

function compute_hydrostatic_pressure!(ph, model)
    grid = model.grid
    arch = grid.architecture

    launch!(arch, grid, :xy, _compute_hydrostatic_pressure!, ph, grid,
            model.formulation,
            model.formulation.reference_state.density,
            model.temperature,
            model.specific_moisture,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    fill_halo_regions!(ph)

    return nothing
end
