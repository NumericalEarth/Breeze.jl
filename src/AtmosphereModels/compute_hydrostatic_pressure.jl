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

    ρᵣ = formulation.reference_state.density
    pᵣ = formulation.reference_state.pressure
    Nz = grid.Nz
    b¹ = ℑzᵃᵃᶠ(i, j, 1, grid, ρ_bᶜᶜᶜ, formulation, args...)
    ρᵣ¹ = ℑzᵃᵃᶠ(1, 1, 1, grid, ρᵣ)

    # ph⁺ - phᵏ = Δz * b * pᵣ
    # ph⁺ = phᵏ + Δz * b * pᵣ

    # Pressume no pressure perturbation at the surface
    @inbounds ph[i, j, 1] = ρᵣ¹ * b¹ * Δzᶜᶜᶠ(i, j, 1, grid) *0.5
    # Integrate update downwards
    for k in 2:Nz
        bᵏ = ℑzᵃᵃᶠ(i, j, k, grid, ρ_bᶜᶜᶜ, formulation, args...)
        ρᵣᵏ = ℑzᵃᵃᶠ(1, 1, k, grid, ρᵣ)
        @inbounds ph[i, j, k] = ph[i, j, k-1] + ρᵣᵏ * bᵏ * Δzᶜᶜᶜ(i, j, k, grid)
    end

    # Add reference pressure
    for k in 1:Nz
        @inbounds ph[i, j, k] = ph[i, j, k] + pᵣ[1, 1, k]
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
