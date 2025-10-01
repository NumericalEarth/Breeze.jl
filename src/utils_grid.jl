using Oceananigans: RectilinearGrid

"""
    ncols(grid::RectilinearGrid)

Return the number of columns in the grid.
"""
ncols(grid::RectilinearGrid) = grid.Nx * grid.Ny


