#####
##### update pressure
#####

using Oceananigans: Center, Face
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ℑzᵃᵃᶠ
using Oceananigans.BoundaryConditions: fill_halo_regions!

const c = Center()
const f = Face()

@kernel function _compute_hydrostatic_pressure!(ph, grid, formulation, args...)
    i, j = @index(Global, NTuple)

    p₀ = formulation.reference_state.base_pressure
    Nz = grid.Nz
    b¹ = ℑzᵃᵃᶠ(i, j, 1, grid, ρ_bᶜᶜᶜ, formulation, args...)

    # ph⁺ - phᵏ = Δz * b
    # ph⁺ = phᵏ + Δz * b

    @inbounds ph[i, j, 1] = p₀ - b¹ * Δzᶜᶜᶠ(i, j, 1, grid) *0.5
    # Integrate update downwards
    for k in 2:Nz
        bᵏ = ℑzᵃᵃᶠ(i, j, k, grid, ρ_bᶜᶜᶜ, formulation, args...)
        @inbounds ph[i, j, k] = ph[i, j, k-1] - bᵏ * Δzᶜᶜᶜ(i, j, k, grid)
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

    return ph
end
