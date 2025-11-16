"""
Grid conversion utilities for converting between Breeze's 3D grid format
and RRTMGP's column-based format.
"""

using Oceananigans: RectilinearGrid, CPU, GPU, interior
using Oceananigans.Grids: znode
using ClimaComms

"""
    create_climacomms_context(arch)

Create a ClimaComms context from an Oceananigans architecture.
"""
function create_climacomms_context(arch)
    if arch isa CPU
        return ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    elseif arch isa GPU
        return ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())
    else
        error("Unsupported architecture: $(typeof(arch))")
    end
end

"""
    grid_to_columns(grid::RectilinearGrid)

Convert Breeze's 3D grid to RRTMGP column format.
Returns `(nlay, ncol)` where:
- `nlay` is the number of vertical layers
- `ncol` is the number of horizontal columns (nx * ny)
"""
function grid_to_columns(grid::RectilinearGrid)
    nx, ny, nz = size(grid)
    nlay = nz
    ncol = nx * ny
    return nlay, ncol
end

"""
    extract_column_data(field, grid, icol)

Extract column data from a 3D field for a given column index.
`icol` is a linear index into the horizontal grid (1 to nx*ny).
"""
function extract_column_data(field, grid, icol)
    nx, ny, nz = size(grid)
    i = mod1(icol, nx)
    j = cld(icol, nx)
    
    column_data = similar(field, nz)
    for k in 1:nz
        column_data[k] = field[i, j, k]
    end
    
    return column_data
end

"""
    extract_surface_temperature(temperature, grid)

Extract surface temperature (bottom boundary) from temperature field.
Returns a 2D array `(nx, ny)` which can be reshaped to `(ncol,)` for RRTMGP.
"""
function extract_surface_temperature(temperature, grid)
    nx, ny, nz = size(grid)
    FT = eltype(grid)
    T_data = interior(temperature)
    
    sfc_T = Array{FT}(undef, nx, ny)
    
    for j in 1:ny, i in 1:nx
        sfc_T[i, j] = T_data[i, j, 1]
    end
    
    return sfc_T
end

"""
    compute_pressure_levels(p_ref, grid)

Compute pressure levels from reference pressure field.
For anelastic formulation, pressure levels are computed at cell faces (levels).
Returns `p_lev` with shape `(nlay+1, ncol)`.
"""
function compute_pressure_levels(p_ref, grid)
    nx, ny, nz = size(grid)
    nlay = nz
    ncol = nx * ny
    nlev = nlay + 1
    
    FT = eltype(grid)
    p_ref_data = interior(p_ref)
    
    # Allocate output array
    p_lev = Array{FT}(undef, nlev, ncol)
    
    for icol in 1:ncol
        i = mod1(icol, nx)
        j = cld(icol, nx)
        
        # Bottom level (surface) - use bottom cell pressure
        p_lev[1, icol] = p_ref_data[i, j, 1]
        
        # Interior levels - average adjacent cell pressures
        for k in 1:nlay-1
            p_lev[k+1, icol] = (p_ref_data[i, j, k] + p_ref_data[i, j, k+1]) / 2
        end
        
        # Top level - use top cell pressure
        p_lev[nlev, icol] = p_ref_data[i, j, nlay]
    end
    
    return p_lev
end

"""
    compute_layer_pressures(p_lev)

Compute layer pressures from level pressures.
Layer pressure is the average of adjacent level pressures.
Returns `p_lay` with shape `(nlay, ncol)`.
"""
function compute_layer_pressures(p_lev)
    nlev, ncol = size(p_lev)
    nlay = nlev - 1
    
    p_lay = similar(p_lev, nlay, ncol)
    
    for icol in 1:ncol
        for k in 1:nlay
            p_lay[k, icol] = (p_lev[k, icol] + p_lev[k+1, icol]) / 2
        end
    end
    
    return p_lay
end

"""
    compute_level_temperatures(temperature, grid)

Compute temperatures at pressure levels from cell-centered temperatures.
Returns `t_lev` with shape `(nlay+1, ncol)`.
"""
function compute_level_temperatures(temperature, grid)
    nx, ny, nz = size(grid)
    nlay = nz
    ncol = nx * ny
    nlev = nlay + 1
    
    FT = eltype(grid)
    T_data = interior(temperature)
    
    # Allocate output array
    t_lev = Array{FT}(undef, nlev, ncol)
    
    for icol in 1:ncol
        i = mod1(icol, nx)
        j = cld(icol, nx)
        
        # Bottom level (surface) - use bottom cell temperature
        t_lev[1, icol] = T_data[i, j, 1]
        
        # Interior levels - average adjacent cell temperatures
        for k in 1:nlay-1
            t_lev[k+1, icol] = (T_data[i, j, k] + T_data[i, j, k+1]) / 2
        end
        
        # Top level - use top cell temperature
        t_lev[nlev, icol] = T_data[i, j, nlay]
    end
    
    return t_lev
end

"""
    compute_layer_temperatures(temperature, grid)

Compute layer temperatures from cell-centered temperatures.
For now, layer temperature equals cell-centered temperature.
Returns `t_lay` with shape `(nlay, ncol)`.
"""
function compute_layer_temperatures(temperature, grid)
    nx, ny, nz = size(grid)
    nlay = nz
    ncol = nx * ny
    
    t_lay = similar(temperature, nlay, ncol)
    
    for icol in 1:ncol
        i = mod1(icol, nx)
        j = cld(icol, nx)
        
        for k in 1:nlay
            t_lay[k, icol] = temperature[i, j, k]
        end
    end
    
    return t_lay
end

"""
    compute_level_altitudes(grid)

Compute altitudes at pressure levels from grid coordinates.
Returns `z_lev` with shape `(nlay+1, ncol)`.
"""
function compute_level_altitudes(grid)
    nx, ny, nz = size(grid)
    nlay = nz
    ncol = nx * ny
    nlev = nlay + 1
    
    FT = eltype(grid)
    z_lev = zeros(FT, nlev, ncol)
    
    for icol in 1:ncol
        i = mod1(icol, nx)
        j = cld(icol, nx)
        
        # Get z coordinates at cell faces (levels)
        for k in 1:nlev
            # Use znode to get face locations - for now approximate
            # In practice, we'd want the actual face z coordinates
            if k == 1
                z_lev[k, icol] = znode(i, j, 1, grid, Center(), Center(), Face())
            elseif k == nlev
                z_lev[k, icol] = znode(i, j, nlay+1, grid, Center(), Center(), Face())
            else
                # Average of adjacent cell centers
                z_bot = znode(i, j, k-1, grid, Center(), Center(), Center())
                z_top = znode(i, j, k, grid, Center(), Center(), Center())
                z_lev[k, icol] = (z_bot + z_top) / 2
            end
        end
    end
    
    return z_lev
end

"""
    reshape_to_columns(array_3d, grid)

Reshape a 3D array `(nx, ny, nz)` to column format `(nz, nx*ny)`.
Note: For Oceananigans Fields, use `interior(field)` first to get the array.
"""
function reshape_to_columns(array_3d, grid)
    nx, ny, nz = size(grid)
    ncol = nx * ny
    nlay = nz
    
    FT = eltype(array_3d)
    array_2d = Array{FT}(undef, nlay, ncol)
    
    for j in 1:ny, i in 1:nx
        icol = (j - 1) * nx + i
        for k in 1:nz
            array_2d[k, icol] = array_3d[i, j, k]
        end
    end
    
    return array_2d
end

"""
    reshape_from_columns(array_2d, grid)

Reshape a column-format array `(nz, nx*ny)` back to 3D format `(nx, ny, nz)`.
"""
function reshape_from_columns(array_2d, grid)
    nx, ny, nz = size(grid)
    ncol = nx * ny
    
    @assert size(array_2d) == (nz, ncol) "Array size mismatch"
    
    array_3d = similar(array_2d, nx, ny, nz)
    
    for j in 1:ny, i in 1:nx
        icol = (j - 1) * nx + i
        for k in 1:nz
            array_3d[i, j, k] = array_2d[k, icol]
        end
    end
    
    return array_3d
end

