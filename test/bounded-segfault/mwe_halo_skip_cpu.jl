# MWE: Face field on Bounded axis — halo fill skips the N+1 point (CPU)
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!

grid = RectilinearGrid(CPU();
    size=(2, 2), extent=(1e3, 1e3), halo=(1, 1),
    topology=(Bounded, Bounded, Flat))

u = Field{Face, Center, Center}(grid)
fill!(parent(u), 0)
set!(u, (x, y) -> 1.0)
fill_halo_regions!(u)

print(u.boundary_conditions)

display(parent(u)[:, :, 1])
