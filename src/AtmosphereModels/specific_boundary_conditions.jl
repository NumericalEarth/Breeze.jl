# Density values and transported specific fields have different units. Use the
# same face carrier density as the transport operator, and retain the owner's
# Dirichlet halo convention. Flux/default/sedimentation BCs keep their policies.
using Oceananigans.BoundaryConditions: BoundaryCondition, Value, getbc,
    _fill_west_halo!, _fill_east_halo!, _fill_south_halo!,
    _fill_north_halo!, _fill_bottom_halo!, _fill_top_halo!
using Oceananigans.Grids: Center
using Oceananigans.Models: boundary_condition_args
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

# Formulations opt in using their own transported field and coupling density.
fill_thermodynamic_boundary_halos!(formulation, model) = nothing

function fill_specific_boundary_halos!(model)
    args = boundary_condition_args(model)
    fill_density_specific_boundary_halos!(specific_prognostic_moisture(model),
        model.moisture_density, total_density(model.dynamics), args...)
    fill_microphysical_boundary_halos!(model.microphysics, model)
    fill_thermodynamic_boundary_halos!(model.formulation, model)
    return nothing
end

function fill_density_specific_boundary_halos!(specific, raw, density, args...)
    grid = specific.grid
    bcs = raw.boundary_conditions
    if bcs.west isa BoundaryCondition{<:Value} || bcs.east isa BoundaryCondition{<:Value}
        launch!(grid.architecture, grid, :yz, _fill_density_specific_x_halos!,
                specific, density, grid, bcs.west, bcs.east, args)
    end
    if bcs.south isa BoundaryCondition{<:Value} || bcs.north isa BoundaryCondition{<:Value}
        launch!(grid.architecture, grid, :xz, _fill_density_specific_y_halos!,
                specific, density, grid, bcs.south, bcs.north, args)
    end
    if bcs.bottom isa BoundaryCondition{<:Value} || bcs.top isa BoundaryCondition{<:Value}
        launch!(grid.architecture, grid, :xy, _fill_density_specific_z_halos!,
                specific, density, grid, bcs.bottom, bcs.top, args)
    end
    return nothing
end

@inline fill_density_specific_west_halo!(j, k, grid, q, density, bc, args...) = nothing
@inline function fill_density_specific_west_halo!(j, k, grid, q, density, bc::BoundaryCondition{<:Value}, args...)
    value = getbc(bc, j, k, grid, args...) / ℑxᶠᵃᵃ(1, j, k, grid, density)
    _fill_west_halo!(j, k, grid, q, BoundaryCondition(Value(), value),
                       (Center(), Center(), Center()), args...)
    return nothing
end

@inline fill_density_specific_east_halo!(j, k, grid, q, density, bc, args...) = nothing
@inline function fill_density_specific_east_halo!(j, k, grid, q, density, bc::BoundaryCondition{<:Value}, args...)
    value = getbc(bc, j, k, grid, args...) / ℑxᶠᵃᵃ(grid.Nx + 1, j, k, grid, density)
    _fill_east_halo!(j, k, grid, q, BoundaryCondition(Value(), value),
                       (Center(), Center(), Center()), args...)
    return nothing
end

@inline fill_density_specific_south_halo!(i, k, grid, q, density, bc, args...) = nothing
@inline function fill_density_specific_south_halo!(i, k, grid, q, density, bc::BoundaryCondition{<:Value}, args...)
    value = getbc(bc, i, k, grid, args...) / ℑyᵃᶠᵃ(i, 1, k, grid, density)
    _fill_south_halo!(i, k, grid, q, BoundaryCondition(Value(), value),
                       (Center(), Center(), Center()), args...)
    return nothing
end

@inline fill_density_specific_north_halo!(i, k, grid, q, density, bc, args...) = nothing
@inline function fill_density_specific_north_halo!(i, k, grid, q, density, bc::BoundaryCondition{<:Value}, args...)
    value = getbc(bc, i, k, grid, args...) / ℑyᵃᶠᵃ(i, grid.Ny + 1, k, grid, density)
    _fill_north_halo!(i, k, grid, q, BoundaryCondition(Value(), value),
                       (Center(), Center(), Center()), args...)
    return nothing
end

@inline fill_density_specific_bottom_halo!(i, j, grid, q, density, bc, args...) = nothing
@inline function fill_density_specific_bottom_halo!(i, j, grid, q, density, bc::BoundaryCondition{<:Value}, args...)
    value = getbc(bc, i, j, grid, args...) / ℑzᵃᵃᶠ(i, j, 1, grid, density)
    _fill_bottom_halo!(i, j, grid, q, BoundaryCondition(Value(), value),
                       (Center(), Center(), Center()), args...)
    return nothing
end

@inline fill_density_specific_top_halo!(i, j, grid, q, density, bc, args...) = nothing
@inline function fill_density_specific_top_halo!(i, j, grid, q, density, bc::BoundaryCondition{<:Value}, args...)
    value = getbc(bc, i, j, grid, args...) / ℑzᵃᵃᶠ(i, j, grid.Nz + 1, grid, density)
    _fill_top_halo!(i, j, grid, q, BoundaryCondition(Value(), value),
                       (Center(), Center(), Center()), args...)
    return nothing
end

@kernel function _fill_density_specific_x_halos!(q, density, grid, left, right, args)
    j, k = @index(Global, NTuple)
    fill_density_specific_west_halo!(j, k, grid, q, density, left, args...)
    fill_density_specific_east_halo!(j, k, grid, q, density, right, args...)
end

@kernel function _fill_density_specific_y_halos!(q, density, grid, left, right, args)
    i, k = @index(Global, NTuple)
    fill_density_specific_south_halo!(i, k, grid, q, density, left, args...)
    fill_density_specific_north_halo!(i, k, grid, q, density, right, args...)
end

@kernel function _fill_density_specific_z_halos!(q, density, grid, left, right, args)
    i, j = @index(Global, NTuple)
    fill_density_specific_bottom_halo!(i, j, grid, q, density, left, args...)
    fill_density_specific_top_halo!(i, j, grid, q, density, right, args...)
end
