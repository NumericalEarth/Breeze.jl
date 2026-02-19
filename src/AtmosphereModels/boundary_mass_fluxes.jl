#####
##### Open boundary mass flux computation and conservation enforcement
#####
##### For the anelastic pressure equation to be solvable, the net mass flux
##### through open boundaries must vanish: ∮ ρ₀u · dA = 0.
##### This file provides GPU-compatible Integral-based mass flux computation
##### following the pattern in Oceananigans' NonhydrostaticModel.
#####

using Oceananigans.AbstractOperations: Integral, grid_metric_operation, Ax, Ay
using Oceananigans.BoundaryConditions: Open, BoundaryCondition
using Oceananigans.Fields: Field, compute!, interior, view
using Oceananigans.Grids: topology, Bounded, Face, Center
using GPUArraysCore: @allowscalar

const OpenBC = BoundaryCondition{<:Open}

# Left boundary integrals for density-weighted momentum ρu, ρv
@inline west_mass_flux(ρu)  = Field(Integral(view(ρu, 1, :, :), dims=(2, 3)))
@inline south_mass_flux(ρv) = Field(Integral(view(ρv, :, 1, :), dims=(1, 3)))

# Right boundary integrals
@inline east_mass_flux(ρu)  = Field(Integral(view(ρu, ρu.grid.Nx + 1, :, :), dims=(2, 3)))
@inline north_mass_flux(ρv) = Field(Integral(view(ρv, :, ρv.grid.Ny + 1, :), dims=(1, 3)))

# Boundary face areas following Oceananigans' pattern
function get_west_area(grid)
    dA = grid_metric_operation((Face, Center, Center), Ax, grid)
    ∫dA = sum(dA, dims=(2, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function get_east_area(grid)
    dA = grid_metric_operation((Face, Center, Center), Ax, grid)
    ∫dA = sum(dA, dims=(2, 3))
    return @allowscalar ∫dA[grid.Nx + 1, 1, 1]
end

function get_south_area(grid)
    dA = grid_metric_operation((Center, Face, Center), Ay, grid)
    ∫dA = sum(dA, dims=(1, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function get_north_area(grid)
    dA = grid_metric_operation((Center, Face, Center), Ay, grid)
    ∫dA = sum(dA, dims=(1, 3))
    return @allowscalar ∫dA[1, grid.Ny + 1, 1]
end

"""
$(TYPEDSIGNATURES)

Container for Integral-based mass flux fields at open boundaries.
Each field is either an `Integral` `Field` or `nothing` if the boundary is not open.
"""
struct BoundaryMassFluxes{W, E, S, N, FT}
    west  :: W
    east  :: E
    south :: S
    north :: N
    total_open_area :: FT
end

Adapt.adapt_structure(to, bmf::BoundaryMassFluxes) =
    BoundaryMassFluxes(adapt(to, bmf.west),
                       adapt(to, bmf.east),
                       adapt(to, bmf.south),
                       adapt(to, bmf.north),
                       bmf.total_open_area)

"""
$(TYPEDSIGNATURES)

Singleton indicating no open boundaries require mass flux tracking.
"""
struct NoBoundaryMassFluxes end

Adapt.adapt_structure(to, ::NoBoundaryMassFluxes) = NoBoundaryMassFluxes()

"""
$(TYPEDSIGNATURES)

Initialize boundary mass fluxes container based on the momentum fields.
Returns `NoBoundaryMassFluxes()` if no open boundaries are detected.
"""
function initialize_boundary_mass_fluxes(momentum)
    ρu = momentum.ρu
    ρv = momentum.ρv
    grid = ρu.grid

    TX, TY, _ = topology(grid)

    ρu_bcs = ρu.boundary_conditions
    ρv_bcs = ρv.boundary_conditions

    has_west  = TX == Bounded && ρu_bcs.west  isa OpenBC
    has_east  = TX == Bounded && ρu_bcs.east  isa OpenBC
    has_south = TY == Bounded && ρv_bcs.south isa OpenBC
    has_north = TY == Bounded && ρv_bcs.north isa OpenBC

    if !has_west && !has_east && !has_south && !has_north
        return NoBoundaryMassFluxes()
    end

    west  = has_west  ? west_mass_flux(ρu)  : nothing
    east  = has_east  ? east_mass_flux(ρu)  : nothing
    south = has_south ? south_mass_flux(ρv) : nothing
    north = has_north ? north_mass_flux(ρv) : nothing

    # Compute total open boundary area for uniform correction
    FT = eltype(grid)
    total_area = zero(FT)
    has_west  && (total_area += get_west_area(grid))
    has_east  && (total_area += get_east_area(grid))
    has_south && (total_area += get_south_area(grid))
    has_north && (total_area += get_north_area(grid))

    return BoundaryMassFluxes(west, east, south, north, total_area)
end

#####
##### Mass flux computation and conservation enforcement
#####

"""
$(TYPEDSIGNATURES)

No-op for models without open boundaries.
"""
enforce_open_boundary_mass_conservation!(model, ::NoBoundaryMassFluxes) = nothing

# Compute scalar mass flux through a boundary (GPU-compatible via Integral)
@inline function get_boundary_mass_flux(flux_field)
    compute!(flux_field)
    return @allowscalar flux_field[]
end

@inline get_boundary_mass_flux(::Nothing) = 0

# Correct left boundary momentum by subtracting uniform correction
correct_left_mass_flux!(ρu, ::OpenBC, ::Val{:west}, correction)  = interior(ρu, 1, :, :) .-= correction
correct_left_mass_flux!(ρv, ::OpenBC, ::Val{:south}, correction) = interior(ρv, :, 1, :) .-= correction
correct_left_mass_flux!(field, bc, side, correction) = nothing

# Correct right boundary momentum by adding uniform correction
correct_right_mass_flux!(ρu, ::OpenBC, ::Val{:east}, correction)  = interior(ρu, ρu.grid.Nx + 1, :, :) .+= correction
correct_right_mass_flux!(ρv, ::OpenBC, ::Val{:north}, correction) = interior(ρv, :, ρv.grid.Ny + 1, :) .+= correction
correct_right_mass_flux!(field, bc, side, correction) = nothing

"""
$(TYPEDSIGNATURES)

Enforce mass conservation for models with open boundaries by uniformly
correcting boundary momentum to ensure the anelastic solvability condition
``∮ ρ₀ u · dA = 0``.

The correction distributes the net mass imbalance uniformly across all
open boundary faces:

```math
(ρu)_{\\text{corrected}} = (ρu) ± \\frac{∮ ρu · dA}{A_{\\text{total}}}
```
"""
function enforce_open_boundary_mass_conservation!(model, bmf::BoundaryMassFluxes)
    ρu = model.momentum.ρu
    ρv = model.momentum.ρv

    # Compute mass flux through each open boundary (GPU-compatible)
    Φ_west  = get_boundary_mass_flux(bmf.west)
    Φ_east  = get_boundary_mass_flux(bmf.east)
    Φ_south = get_boundary_mass_flux(bmf.south)
    Φ_north = get_boundary_mass_flux(bmf.north)

    # Net mass inflow (positive ρu at west = inflow, positive ρu at east = outflow)
    net_inflow = (Φ_west - Φ_east) + (Φ_south - Φ_north)

    # Uniform correction: distribute imbalance across all open boundaries
    correction = net_inflow / bmf.total_open_area

    ρu_bcs = ρu.boundary_conditions
    ρv_bcs = ρv.boundary_conditions

    correct_left_mass_flux!(ρu, ρu_bcs.west,   Val(:west),  correction)
    correct_left_mass_flux!(ρv, ρv_bcs.south,   Val(:south), correction)
    correct_right_mass_flux!(ρu, ρu_bcs.east,   Val(:east),  correction)
    correct_right_mass_flux!(ρv, ρv_bcs.north,   Val(:north), correction)

    return nothing
end
