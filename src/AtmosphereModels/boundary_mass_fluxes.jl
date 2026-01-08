#####
##### Open boundary mass flux container and mass conservation enforcement
#####

using GPUArraysCore: @allowscalar
using Oceananigans.BoundaryConditions: Open, BoundaryCondition
using Oceananigans.Grids: topology, Bounded, Flat
using Oceananigans.Operators: Δyᶠᶜᶜ, Δzᶠᶜᶜ, Δxᶜᶠᶜ, Δzᶜᶠᶜ
using Oceananigans.Architectures: architecture

# Type alias for checking open boundary conditions
const OpenBC = BoundaryCondition{<:Open}

"""
    BoundaryMassFluxes

Container for storing mass fluxes through open boundaries. Used to enforce
mass conservation in models with open boundary conditions.
"""
struct BoundaryMassFluxes{W, E, S, N}
    west :: W
    east :: E
    south :: S
    north :: N
end

Adapt.adapt_structure(to, bmf::BoundaryMassFluxes) =
    BoundaryMassFluxes(adapt(to, bmf.west),
                       adapt(to, bmf.east),
                       adapt(to, bmf.south),
                       adapt(to, bmf.north))

"""
    NoBoundaryMassFluxes

Singleton type indicating no open boundaries require mass flux tracking.
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

    # Check if we have open boundaries (OpenBC = BoundaryCondition{<:Open})
    has_x_open = TX == Bounded && (
        ρu.boundary_conditions.west isa OpenBC ||
        ρu.boundary_conditions.east isa OpenBC
    )
    
    has_y_open = TY == Bounded && (
        ρv.boundary_conditions.south isa OpenBC ||
        ρv.boundary_conditions.north isa OpenBC
    )

    if !has_x_open && !has_y_open
        return NoBoundaryMassFluxes()
    end

    # Initialize flux storage as Refs
    west  = has_x_open && ρu.boundary_conditions.west  isa OpenBC ? Ref(zero(eltype(grid))) : nothing
    east  = has_x_open && ρu.boundary_conditions.east  isa OpenBC ? Ref(zero(eltype(grid))) : nothing
    south = has_y_open && ρv.boundary_conditions.south isa OpenBC ? Ref(zero(eltype(grid))) : nothing
    north = has_y_open && ρv.boundary_conditions.north isa OpenBC ? Ref(zero(eltype(grid))) : nothing

    return BoundaryMassFluxes(west, east, south, north)
end

#####
##### Mass conservation enforcement
#####

"""
$(TYPEDSIGNATURES)

Enforce mass conservation for models with open boundaries.
This is a no-op for models without open boundaries.
"""
enforce_open_boundary_mass_conservation!(model, ::NoBoundaryMassFluxes) = nothing

"""
$(TYPEDSIGNATURES)

Enforce mass conservation for models with open boundaries by adjusting
the outflow boundary values to balance the net mass flux.

The algorithm:
1. Compute total mass flux through each open boundary
2. Compute net imbalance
3. Distribute correction to outflow boundaries proportional to their flux magnitude
"""
function enforce_open_boundary_mass_conservation!(model, bmf::BoundaryMassFluxes)
    grid = model.grid
    ρu = model.momentum.ρu
    ρv = model.momentum.ρv

    # Compute boundary mass fluxes
    compute_boundary_mass_fluxes!(bmf, grid, ρu, ρv)

    # Get flux values
    Φ_west  = bmf.west  === nothing ? zero(eltype(grid)) : bmf.west[]
    Φ_east  = bmf.east  === nothing ? zero(eltype(grid)) : bmf.east[]
    Φ_south = bmf.south === nothing ? zero(eltype(grid)) : bmf.south[]
    Φ_north = bmf.north === nothing ? zero(eltype(grid)) : bmf.north[]

    # Net imbalance (positive = more mass entering than leaving)
    net_imbalance = (Φ_west - Φ_east) + (Φ_south - Φ_north)

    # Compute total outflow
    # Outflow is: positive flux at east/north, negative flux at west/south
    outflow_east  = max(zero(eltype(grid)), Φ_east)
    outflow_west  = max(zero(eltype(grid)), -Φ_west)
    outflow_north = max(zero(eltype(grid)), Φ_north)
    outflow_south = max(zero(eltype(grid)), -Φ_south)
    
    total_outflow = outflow_east + outflow_west + outflow_north + outflow_south

    # Avoid division by zero
    if total_outflow ≤ eps(eltype(grid))
        return nothing
    end

    # Correction factor
    α = net_imbalance / total_outflow

    # Apply corrections to outflow boundaries
    if bmf.east !== nothing && Φ_east > 0
        apply_mass_flux_correction!(ρu, :east, α, grid)
    end
    
    if bmf.west !== nothing && Φ_west < 0
        apply_mass_flux_correction!(ρu, :west, α, grid)
    end
    
    if bmf.north !== nothing && Φ_north > 0
        apply_mass_flux_correction!(ρv, :north, α, grid)
    end
    
    if bmf.south !== nothing && Φ_south < 0
        apply_mass_flux_correction!(ρv, :south, α, grid)
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute mass fluxes through each open boundary.
"""
function compute_boundary_mass_fluxes!(bmf::BoundaryMassFluxes, grid, ρu, ρv)
    arch = architecture(grid)
    
    # Compute west boundary flux (flux into domain is positive for ρu at i=1)
    if bmf.west !== nothing
        bmf.west[] = compute_west_mass_flux(grid, ρu)
    end
    
    # Compute east boundary flux (flux out of domain is positive for ρu at i=Nx+1)
    if bmf.east !== nothing
        bmf.east[] = compute_east_mass_flux(grid, ρu)
    end
    
    # Compute south boundary flux (flux into domain is positive for ρv at j=1)
    if bmf.south !== nothing
        bmf.south[] = compute_south_mass_flux(grid, ρv)
    end
    
    # Compute north boundary flux (flux out of domain is positive for ρv at j=Ny+1)
    if bmf.north !== nothing
        bmf.north[] = compute_north_mass_flux(grid, ρv)
    end

    return nothing
end

# Individual boundary flux computations
# For now, use @allowscalar for simplicity; can be optimized with kernels later

function compute_west_mass_flux(grid, ρu)
    FT = eltype(grid)
    Ny, Nz = grid.Ny, grid.Nz
    flux = zero(FT)
    @allowscalar for k in 1:Nz, j in 1:Ny
        flux += ρu[1, j, k] * Δyᶠᶜᶜ(1, j, k, grid) * Δzᶠᶜᶜ(1, j, k, grid)
    end
    return flux
end

function compute_east_mass_flux(grid, ρu)
    FT = eltype(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    flux = zero(FT)
    @allowscalar for k in 1:Nz, j in 1:Ny
        flux += ρu[Nx+1, j, k] * Δyᶠᶜᶜ(Nx+1, j, k, grid) * Δzᶠᶜᶜ(Nx+1, j, k, grid)
    end
    return flux
end

function compute_south_mass_flux(grid, ρv)
    FT = eltype(grid)
    Nx, Nz = grid.Nx, grid.Nz
    flux = zero(FT)
    @allowscalar for k in 1:Nz, i in 1:Nx
        flux += ρv[i, 1, k] * Δxᶜᶠᶜ(i, 1, k, grid) * Δzᶜᶠᶜ(i, 1, k, grid)
    end
    return flux
end

function compute_north_mass_flux(grid, ρv)
    FT = eltype(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    flux = zero(FT)
    @allowscalar for k in 1:Nz, i in 1:Nx
        flux += ρv[i, Ny+1, k] * Δxᶜᶠᶜ(i, Ny+1, k, grid) * Δzᶜᶠᶜ(i, Ny+1, k, grid)
    end
    return flux
end

"""
$(TYPEDSIGNATURES)

Apply mass flux correction to a boundary.
"""
function apply_mass_flux_correction!(field, side::Symbol, α, grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    
    if side == :west
        @allowscalar for k in 1:Nz, j in 1:Ny
            field[1, j, k] *= (1 + α)
        end
    elseif side == :east
        @allowscalar for k in 1:Nz, j in 1:Ny
            field[Nx+1, j, k] *= (1 + α)
        end
    elseif side == :south
        @allowscalar for k in 1:Nz, i in 1:Nx
            field[i, 1, k] *= (1 + α)
        end
    elseif side == :north
        @allowscalar for k in 1:Nz, i in 1:Nx
            field[i, Ny+1, k] *= (1 + α)
        end
    end
    
    return nothing
end

