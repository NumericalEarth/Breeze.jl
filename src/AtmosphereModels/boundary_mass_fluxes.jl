using Oceananigans.AbstractOperations: Integral, Ax, Ay, Az, grid_metric_operation
using Oceananigans.BoundaryConditions: BoundaryCondition, Open
using Oceananigans.Fields: Field, interior, compute!
using Oceananigans.Grids: Face, Center
using GPUArraysCore: @allowscalar

const OBC  = BoundaryCondition{<:Open}  # OpenBoundaryCondition
const IOBC = BoundaryCondition{<:Open{<:Nothing}}  # "Imposed-velocity" OpenBoundaryCondition (with no scheme)
const FIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Number}  # "Fixed-imposed-velocity" OpenBoundaryCondition
const ZIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Nothing}  # "Zero-imposed-velocity" OpenBoundaryCondition (no-inflow)

#####
##### Boundary area calculations
#####

function get_west_area(grid)
    dA = grid_metric_operation((Face, Center, Center), Ax, grid)
    ∫dA = sum(dA, dims=(2, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function get_east_area(grid)
    dA = grid_metric_operation((Face, Center, Center), Ax, grid)
    ∫dA = sum(dA, dims=(2, 3))
    return @allowscalar ∫dA[grid.Nx+1, 1, 1]
end

function get_south_area(grid)
    dA = grid_metric_operation((Center, Face, Center), Ay, grid)
    ∫dA = sum(dA, dims=(1, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function get_north_area(grid)
    dA = grid_metric_operation((Center, Face, Center), Ay, grid)
    ∫dA = sum(dA, dims=(1, 3))
    return @allowscalar ∫dA[1, grid.Ny+1, 1]
end

function get_bottom_area(grid)
    dA = grid_metric_operation((Center, Center, Face), Az, grid)
    ∫dA = sum(dA, dims=(1, 2))
    return @allowscalar ∫dA[1, 1, 1]
end

function get_top_area(grid)
    dA = grid_metric_operation((Center, Center, Face), Az, grid)
    ∫dA = sum(dA, dims=(1, 2))
    return @allowscalar ∫dA[1, 1, grid.Nz+1]
end

#####
##### Boundary mass flux integrals (using momentum fields ρu, ρv, ρw)
#####

# Left boundary integrals for momentum components
@inline west_mass_flux(ρu)   = Field(Integral(view(ρu, 1, :, :), dims=(2, 3)))
@inline south_mass_flux(ρv)  = Field(Integral(view(ρv, :, 1, :), dims=(1, 3)))
@inline bottom_mass_flux(ρw) = Field(Integral(view(ρw, :, :, 1), dims=(1, 2)))

# Right boundary integrals for momentum components
@inline east_mass_flux(ρu)   = Field(Integral(view(ρu, ρu.grid.Nx + 1, :, :), dims=(2, 3)))
@inline north_mass_flux(ρv)  = Field(Integral(view(ρv, :, ρv.grid.Ny + 1, :), dims=(1, 3)))
@inline top_mass_flux(ρw)    = Field(Integral(view(ρw, :, :, ρw.grid.Nz + 1), dims=(1, 2)))

#####
##### Initialize boundary mass flux for each boundary
#####

initialize_boundary_mass_flux(ρu, bc::OBC, ::Val{:west})   = (; west_mass_flux = west_mass_flux(ρu), west_area = get_west_area(ρu.grid))
initialize_boundary_mass_flux(ρu, bc::OBC, ::Val{:east})   = (; east_mass_flux = east_mass_flux(ρu), east_area = get_east_area(ρu.grid))
initialize_boundary_mass_flux(ρv, bc::OBC, ::Val{:south})  = (; south_mass_flux = south_mass_flux(ρv), south_area = get_south_area(ρv.grid))
initialize_boundary_mass_flux(ρv, bc::OBC, ::Val{:north})  = (; north_mass_flux = north_mass_flux(ρv), north_area = get_north_area(ρv.grid))
initialize_boundary_mass_flux(ρw, bc::OBC, ::Val{:bottom}) = (; bottom_mass_flux = bottom_mass_flux(ρw), bottom_area = get_bottom_area(ρw.grid))
initialize_boundary_mass_flux(ρw, bc::OBC, ::Val{:top})    = (; top_mass_flux = top_mass_flux(ρw), top_area = get_top_area(ρw.grid))

initialize_boundary_mass_flux(ρu, bc::ZIOBC, ::Val{:west})   = NamedTuple()
initialize_boundary_mass_flux(ρu, bc::ZIOBC, ::Val{:east})   = NamedTuple()
initialize_boundary_mass_flux(ρv, bc::ZIOBC, ::Val{:south})  = NamedTuple()
initialize_boundary_mass_flux(ρv, bc::ZIOBC, ::Val{:north})  = NamedTuple()
initialize_boundary_mass_flux(ρw, bc::ZIOBC, ::Val{:bottom}) = NamedTuple()
initialize_boundary_mass_flux(ρw, bc::ZIOBC, ::Val{:top})    = NamedTuple()

initialize_boundary_mass_flux(ρu, bc::FIOBC, ::Val{:west})   = (; west_mass_flux = bc.condition * get_west_area(ρu.grid), west_area = get_west_area(ρu.grid))
initialize_boundary_mass_flux(ρu, bc::FIOBC, ::Val{:east})   = (; east_mass_flux = bc.condition * get_east_area(ρu.grid), east_area = get_east_area(ρu.grid))
initialize_boundary_mass_flux(ρv, bc::FIOBC, ::Val{:south})  = (; south_mass_flux = bc.condition * get_south_area(ρv.grid), south_area = get_south_area(ρv.grid))
initialize_boundary_mass_flux(ρv, bc::FIOBC, ::Val{:north})  = (; north_mass_flux = bc.condition * get_north_area(ρv.grid), north_area = get_north_area(ρv.grid))
initialize_boundary_mass_flux(ρw, bc::FIOBC, ::Val{:bottom}) = (; bottom_mass_flux = bc.condition * get_bottom_area(ρw.grid), bottom_area = get_bottom_area(ρw.grid))
initialize_boundary_mass_flux(ρw, bc::FIOBC, ::Val{:top})    = (; top_mass_flux = bc.condition * get_top_area(ρw.grid), top_area = get_top_area(ρw.grid))

initialize_boundary_mass_flux(momentum, ::Nothing, side) = NamedTuple()
initialize_boundary_mass_flux(momentum, bc, side) = NamedTuple()

#####
##### Mass flux correction classification
#####

needs_mass_flux_correction(::IOBC) = false
needs_mass_flux_correction(::OBC) = true
needs_mass_flux_correction(::Nothing) = false
needs_mass_flux_correction(bc) = false

"""
$(TYPEDSIGNATURES)

Initialize boundary mass fluxes for boundaries with OpenBoundaryConditions,
returning a NamedTuple of boundary fluxes. For AtmosphereModel, this integrates
momentum fields (ρu, ρv, ρw) directly rather than velocity times density.
"""
function initialize_boundary_mass_fluxes(momentum::NamedTuple)

    ρu, ρv, ρw = momentum
    ρu_bcs = ρu.boundary_conditions
    ρv_bcs = ρv.boundary_conditions
    ρw_bcs = ρw.boundary_conditions

    boundary_fluxes = NamedTuple()
    right_scheme_boundaries = Symbol[]
    left_scheme_boundaries = Symbol[]
    total_area_scheme_boundaries = zero(eltype(ρu))

    # Check west boundary (ρu momentum)
    west_flux_and_area = initialize_boundary_mass_flux(ρu, ρu_bcs.west, Val(:west))
    boundary_fluxes = merge(boundary_fluxes, west_flux_and_area)
    if needs_mass_flux_correction(ρu_bcs.west)
        push!(left_scheme_boundaries, :west)
        total_area_scheme_boundaries += boundary_fluxes.west_area
    end

    # Check east boundary (ρu momentum)
    east_flux_and_area = initialize_boundary_mass_flux(ρu, ρu_bcs.east, Val(:east))
    boundary_fluxes = merge(boundary_fluxes, east_flux_and_area)
    if needs_mass_flux_correction(ρu_bcs.east)
        push!(right_scheme_boundaries, :east)
        total_area_scheme_boundaries += boundary_fluxes.east_area
    end

    # Check south boundary (ρv momentum)
    south_flux_and_area = initialize_boundary_mass_flux(ρv, ρv_bcs.south, Val(:south))
    boundary_fluxes = merge(boundary_fluxes, south_flux_and_area)
    if needs_mass_flux_correction(ρv_bcs.south)
        push!(left_scheme_boundaries, :south)
        total_area_scheme_boundaries += boundary_fluxes.south_area
    end

    # Check north boundary (ρv momentum)
    north_flux_and_area = initialize_boundary_mass_flux(ρv, ρv_bcs.north, Val(:north))
    boundary_fluxes = merge(boundary_fluxes, north_flux_and_area)
    if needs_mass_flux_correction(ρv_bcs.north)
        push!(right_scheme_boundaries, :north)
        total_area_scheme_boundaries += boundary_fluxes.north_area
    end

    # Check bottom boundary (ρw momentum)
    bottom_flux_and_area = initialize_boundary_mass_flux(ρw, ρw_bcs.bottom, Val(:bottom))
    boundary_fluxes = merge(boundary_fluxes, bottom_flux_and_area)
    if needs_mass_flux_correction(ρw_bcs.bottom)
        push!(left_scheme_boundaries, :bottom)
        total_area_scheme_boundaries += boundary_fluxes.bottom_area
    end

    # Check top boundary (ρw momentum)
    top_flux_and_area = initialize_boundary_mass_flux(ρw, ρw_bcs.top, Val(:top))
    boundary_fluxes = merge(boundary_fluxes, top_flux_and_area)
    if needs_mass_flux_correction(ρw_bcs.top)
        push!(right_scheme_boundaries, :top)
        total_area_scheme_boundaries += boundary_fluxes.top_area
    end

    boundary_fluxes = merge(boundary_fluxes, (; left_scheme_boundaries = Tuple(left_scheme_boundaries),
                                                right_scheme_boundaries = Tuple(right_scheme_boundaries),
                                                total_area_scheme_boundaries))

    if length(boundary_fluxes.left_scheme_boundaries) == 0 && length(boundary_fluxes.right_scheme_boundaries) == 0
        return nothing
    else
        return boundary_fluxes
    end
end

#####
##### Update and compute boundary mass fluxes
#####

update_open_boundary_mass_fluxes!(model) = map(compute!, model.boundary_mass_fluxes)

open_boundary_mass_flux(model, bc::OBC, ::Val{:west}, ρu) = @allowscalar model.boundary_mass_fluxes.west_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:east}, ρu) = @allowscalar model.boundary_mass_fluxes.east_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:south}, ρv) = @allowscalar model.boundary_mass_fluxes.south_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:north}, ρv) = @allowscalar model.boundary_mass_fluxes.north_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:bottom}, ρw) = @allowscalar model.boundary_mass_fluxes.bottom_mass_flux[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:top}, ρw) = @allowscalar model.boundary_mass_fluxes.top_mass_flux[]

open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:west}, ρu) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:east}, ρu) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:south}, ρv) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:north}, ρv) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:bottom}, ρw) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:top}, ρw) = zero(model.grid)

open_boundary_mass_flux(model, bc, side, momentum) = zero(model.grid)

"""
$(TYPEDSIGNATURES)

Compute the total mass inflow through all open boundaries. Positive values
indicate net mass entering the domain.
"""
function open_boundary_mass_inflow(model)
    update_open_boundary_mass_fluxes!(model)

    ρu, ρv, ρw = model.momentum
    total_flux = zero(model.grid)

    # Add flux through left boundaries
    total_flux += open_boundary_mass_flux(model, ρu.boundary_conditions.west, Val(:west), ρu)
    total_flux += open_boundary_mass_flux(model, ρv.boundary_conditions.south, Val(:south), ρv)
    total_flux += open_boundary_mass_flux(model, ρw.boundary_conditions.bottom, Val(:bottom), ρw)

    # Subtract flux through right boundaries
    total_flux -= open_boundary_mass_flux(model, ρu.boundary_conditions.east, Val(:east), ρu)
    total_flux -= open_boundary_mass_flux(model, ρv.boundary_conditions.north, Val(:north), ρv)
    total_flux -= open_boundary_mass_flux(model, ρw.boundary_conditions.top, Val(:top), ρw)

    return total_flux
end

#####
##### Correct boundary mass fluxes to enforce conservation
#####

correct_left_boundary_mass_flux!(ρu, bc::OBC, ::Val{:west},    A⁻¹_∮ρudA) = interior(ρu, 1, :, :) .-= A⁻¹_∮ρudA
correct_left_boundary_mass_flux!(ρv, bc::OBC, ::Val{:south},   A⁻¹_∮ρudA) = interior(ρv, :, 1, :) .-= A⁻¹_∮ρudA
correct_left_boundary_mass_flux!(ρw, bc::OBC, ::Val{:bottom},  A⁻¹_∮ρudA) = interior(ρw, :, :, 1) .-= A⁻¹_∮ρudA
correct_left_boundary_mass_flux!(ρu, bc::IOBC, ::Val{:west},   A⁻¹_∮ρudA) = nothing
correct_left_boundary_mass_flux!(ρv, bc::IOBC, ::Val{:south},  A⁻¹_∮ρudA) = nothing
correct_left_boundary_mass_flux!(ρw, bc::IOBC, ::Val{:bottom}, A⁻¹_∮ρudA) = nothing
correct_left_boundary_mass_flux!(ρu, bc, side, A⁻¹_∮ρudA) = nothing

correct_right_boundary_mass_flux!(ρu, bc::OBC, ::Val{:east},   A⁻¹_∮ρudA) = interior(ρu, ρu.grid.Nx + 1, :, :) .+= A⁻¹_∮ρudA
correct_right_boundary_mass_flux!(ρv, bc::OBC, ::Val{:north},  A⁻¹_∮ρudA) = interior(ρv, :, ρv.grid.Ny + 1, :) .+= A⁻¹_∮ρudA
correct_right_boundary_mass_flux!(ρw, bc::OBC, ::Val{:top},    A⁻¹_∮ρudA) = interior(ρw, :, :, ρw.grid.Nz + 1) .+= A⁻¹_∮ρudA
correct_right_boundary_mass_flux!(ρu, bc::IOBC, ::Val{:east},  A⁻¹_∮ρudA) = nothing
correct_right_boundary_mass_flux!(ρv, bc::IOBC, ::Val{:north}, A⁻¹_∮ρudA) = nothing
correct_right_boundary_mass_flux!(ρw, bc::IOBC, ::Val{:top},   A⁻¹_∮ρudA) = nothing
correct_right_boundary_mass_flux!(ρu, bc, side, A⁻¹_∮ρudA) = nothing

enforce_open_boundary_mass_conservation!(model, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Correct boundary mass fluxes for perturbation advection boundary conditions to ensure
zero net mass flux through all open boundaries. This corrects the momentum fields
directly (ρu, ρv, ρw).
"""
function enforce_open_boundary_mass_conservation!(model, boundary_mass_fluxes)
    ρu, ρv, ρw = model.momentum

    ∮ρudA = open_boundary_mass_inflow(model)
    A = boundary_mass_fluxes.total_area_scheme_boundaries

    A⁻¹_∮ρudA = ∮ρudA / A

    correct_left_boundary_mass_flux!(ρu, ρu.boundary_conditions.west, Val(:west), A⁻¹_∮ρudA)
    correct_left_boundary_mass_flux!(ρv, ρv.boundary_conditions.south, Val(:south), A⁻¹_∮ρudA)
    correct_left_boundary_mass_flux!(ρw, ρw.boundary_conditions.bottom, Val(:bottom), A⁻¹_∮ρudA)

    correct_right_boundary_mass_flux!(ρu, ρu.boundary_conditions.east, Val(:east), A⁻¹_∮ρudA)
    correct_right_boundary_mass_flux!(ρv, ρv.boundary_conditions.north, Val(:north), A⁻¹_∮ρudA)
    correct_right_boundary_mass_flux!(ρw, ρw.boundary_conditions.top, Val(:top), A⁻¹_∮ρudA)
end
