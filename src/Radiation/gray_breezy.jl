using Oceananigans
using Oceananigans: field
using Oceananigans.Architectures: array_type
using Oceananigans: RectilinearGrid

arch = CPU()
grid = RectilinearGrid(
    CPU(), 
    size=(1, 1, 64), 
    x=(0, 2π), 
    y=(0, 2π), 
    z=(0, 1), 
    topology=(Periodic, Periodic, Bounded)
)



function t_p_profiles(grid::RectilinearGrid; lat, p0, pe, otp)

    context = context_from_arch(grid.architecture)
    nlay = grid.Nz
    param_set = RRTMGPParameters(eltype(grid))
    DA = array_type(grid.architecture)
    gray_as = setup_gray_as_pr_grid(context, nlay, lat, p0, pe, otp, param_set, DA)

    pressure = field((Center(), Center(), Face()), gray_as.p_lay, grid)
    temperature = field((Center(), Center(), Face()), gray_as.t_lay, grid)

    return pressure, temperature
end
