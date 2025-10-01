using Oceananigans
using Oceananigans: field
using Oceananigans.Architectures: array_type
using Oceananigans: RectilinearGrid

using Breeze.Radiation: GrayRadiationModel

include("utils_test.jl")

# Parameters
FT = Float64
architecture = CPU()
DA = array_type(architecture)
Nx, Ny, Nz = 1, 1, 60
Lx, Ly, Lz = 1, 20_000_000, 1_000 # domain size in meters

# Set up grid
grid = RectilinearGrid(
    architecture,
    FT,
    size=(Nx, Ny, Nz,), 
    x=(-Lx/2, Lx/2),
    y=(-Ly/2, Ly/2), 
    z=(0, Lz), 
    topology=(Periodic, Periodic, Bounded)
)

# Radiation parameters for Gray Radiation Model
Nbnd = 1 # This is gray band
zenith_angle = DA{FT}(undef, Nx, Ny)
zenith_angle .= FT(52.95) # zenith angle in degrees
sw_inc_flux = DA{FT}(undef, Nx, Ny)
sw_inc_flux .= FT(1407.679) # incoming shortwave flux
albedo_direct = DA{FT}(undef, Nbnd, Nx, Ny)
albedo_direct .= FT(0.1) # surface albedo (direct)
albedo_diffuse = DA{FT}(undef, Nbnd, Nx, Ny)
albedo_diffuse .= FT(0.1) # surface albedo (diffuse)
sfc_emissivity = DA{FT}(undef, Nbnd, Nx, Ny)
sfc_emissivity .= FT(1) # surface emissivity
latitude = latitude_from_grid(grid; lat_center=0)

# construct the radiation model
radiation_model = GrayRadiationModel(
    grid; 
    zenith_angle=zenith_angle, 
    sfc_emissivity=sfc_emissivity, 
    sfc_albedo_direct=albedo_direct, 
    sfc_albedo_diffuse=albedo_diffuse, 
    toa_sw_flux_inc=sw_inc_flux, 
    latitude=latitude,
    isothermal_boundary_layer=false
)

# Set up test case temperature and pressure profiles for the atmosphere
if grid.Nx > 1
    throw("Grid must not be wider than 1 point in the x direction")
else
    # assumes only 1 point in the x direction
    p_surface = 100_000 # surface pressure (Pa)
    p_top = 9_000 # top of atmosphere pressure / emission level (Pa)
    pressure, temperature = gray_test_t_p_profiles(grid; p0=p_surface, pe=p_top)
end

# compute the radiative fluxes
fluxes_net_lw, fluxes_net_sw = radiation_model(temperature, pressure)
