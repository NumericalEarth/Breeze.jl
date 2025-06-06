#####
##### update pressure
#####

using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ℑzᵃᵃᶠ
using Oceananigans.ImmersedBoundaries: PartialCellBottom, ImmersedBoundaryGrid
using Oceananigans.Grids: topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Utils: KernelParameters

const c = Center()
const f = Face()

@inline function buoyancy(i, j, k, grid, formulation, temperature, specific_humidity, thermo)
    α = specific_volume(i, j, k, grid, formulation, temperature, specific_humidity, thermo)
    αʳ = reference_specific_volume(i, j, k, grid, formulation, thermo)
    g = thermo.gravitational_acceleration
    return g * (α - αʳ) / αʳ
end

@kernel function _update_hydrostatic_pressure!(pₕ′, grid, formulation, T, q, thermo)
    i, j = @index(Global, NTuple)

    @inbounds pₕ′[i, j, 0] = 0
    @inbounds pₕ′[i, j, 1] = 0

    @inbounds for k in 2:grid.Nz
        bₖ = ℑzᵃᵃᶠ(i, j, k, grid, buoyancy, formulation, T, q, thermo)
        Δp′ = bₖ * Δzᶜᶜᶠ(i, j, k, grid)
        pₕ′[i, j, k] = pₕ′[i, j, k-1] + Δp′
    end
end

function update_hydrostatic_pressure!(model)
    grid = model.grid
    arch = grid.architecture
    pₕ′ = model.hydrostatic_pressure_anomaly
    formulation = model.formulation
    T = model.temperature
    q = model.specific_humidity
    thermo = model.thermodynamics
    Nx, Ny, Nz = size(grid)
    kernel_parameters = KernelParameters(0:Nx+1, 0:Ny+1)
    launch!(arch, grid, kernel_parameters, _update_hydrostatic_pressure!, pₕ′, grid, formulation, T, q, thermo)
    fill_halo_regions!(pₕ′)
    return nothing
end
