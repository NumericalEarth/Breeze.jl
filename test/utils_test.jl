using Breeze.Radiation: context_from_arch, ncols

using RRTMGP.Parameters: RRTMGPParameters
using RRTMGP.GrayUtils: GrayOpticalThicknessOGorman2008
using RRTMGP.GrayUtils: setup_gray_as_pr_grid

using Oceananigans: RectilinearGrid
using Oceananigans: field, ynodes, Center, CenterField, FieldBoundaryConditions
using Oceananigans.Architectures: array_type
using Oceananigans: fill_halo_regions!
using Oceananigans: set!
using Oceananigans.BoundaryConditions: ValueBoundaryCondition

"""
    latitude(grid; lat_center=0, planet_radius=6_371_000, unit=:degrees)

Return a device array of latitudes on the grid's horizontal layout with shape
`(Nx, Ny)`. Values are centered around `lat_center` and computed from the
grid's y-nodes and `planet_radius`. `unit` can be `:degrees` or `:radians`.
"""
function latitude_from_grid(grid::RectilinearGrid; lat_center=0, planet_radius=6_371_000, unit=:degrees)
    FT = eltype(grid)
    DA = array_type(grid.architecture)

    y = ynodes(grid, Center())
    lat = lat_center .+ y / planet_radius * (unit == :degrees ? FT(180/π) : 1)
    lat = DA{FT}(repeat(lat', grid.Nx, 1))

    return lat
end

"""
    gray_test_t_p_profiles(grid; p0, pe)

Generate test-case pressure and temperature `Field`s for a gray atmosphere over
the provided `grid`. Profiles are derived from RRTMGP's gray utilities and
returned as Oceananigans fields shaped `(Nx, Ny, Nz)`.

Returns `(pressure, temperature)`.
"""
function gray_test_t_p_profiles(grid::RectilinearGrid; p0, pe)
    # get data types and context from grid
    FT = eltype(grid)
    context = context_from_arch(grid.architecture)
    DA = array_type(grid.architecture)

    # derive latitude from grid
    Nx, Ny, Nz = size(grid)
    Ncols = ncols(grid)
    if Ncols == 1
        lat = DA{FT}([0])                   
    else
        lat = DA{FT}(range(FT(-90), FT(90), length = Ncols))
    end
    
    # get test case temperature and pressure profiles
    param_set = RRTMGPParameters(FT)
    otp = GrayOpticalThicknessOGorman2008(FT)
    gray_as = setup_gray_as_pr_grid(context, Nz, lat, FT(p0), FT(pe), otp, param_set, DA)
    p_lay_as_array = reshape(gray_as.p_lay', Nx, Ny, Nz)
    t_lay_as_array = reshape(gray_as.t_lay', Nx, Ny, Nz)

    # create Fields from test case temperature and pressure profiles    
    p_bcs = FieldBoundaryConditions(
        grid, (Center(), Center(), Center());
        bottom = ValueBoundaryCondition(gray_as.p_lev[1]),
        top    = ValueBoundaryCondition(gray_as.p_lev[end]),
    )    
    T_bcs = FieldBoundaryConditions(
        grid, (Center(), Center(), Center());
        bottom = ValueBoundaryCondition(gray_as.t_lev[1]),
        top    = ValueBoundaryCondition(gray_as.t_lev[end]),
    )
    pressure = CenterField(grid; boundary_conditions = p_bcs)
    temperature = CenterField(grid; boundary_conditions = T_bcs)    
    set!(pressure, p_lay_as_array)
    fill_halo_regions!(pressure) # ensure halos reflect BCs
    set!(temperature, t_lay_as_array)
    fill_halo_regions!(temperature) # ensure halos reflect BCs

    return pressure, temperature, gray_as
end
