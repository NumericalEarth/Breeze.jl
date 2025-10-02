using RRTMGP.Optics: GrayOpticalThicknessSchneider2004
using RRTMGP.Parameters: RRTMGPParameters
using RRTMGP.AtmosphericStates: GrayAtmosphericState
using RRTMGP.RTE: TwoStreamLWRTE
using RRTMGP.RTESolver: solve_lw!

using Oceananigans
using Oceananigans: field
using Oceananigans.Architectures: array_type
using Oceananigans: RectilinearGrid

include("rrtmgp_interface.jl")

FT = Float64
architecture = CPU()
DA = array_type(architecture)
Nx, Ny, Nz = 1, 1, 64
Lx, Ly, Lz = 1, 20_000_000, 1_000 # domain size in meters
Nbnd, Ngpt = 1, 1 # gray model
lw_inc_flux = nothing # no incoming longwave flux
sfc_emissivity = 1
planet_radius = 6_371_000
p_surface = 100_000 # surface pressure (Pa)
p_top = 9_000 # top of atmosphere pressure / emission level (Pa)
lat_center = 0
optical_properties = GrayOpticalThicknessSchneider2004(FT)

# Grid setup, we also need to say where on the planet our box is located
grid = RectilinearGrid(
    architecture,
    FT,
    size=(Nx, Ny, Nz,), 
    x=(-Lx/2, Lx/2),
    y=(-Ly/2, Ly/2), 
    z=(0, Lz), 
    topology=(Periodic, Periodic, Bounded)
)

# Set up temperature and pressure profiles for the atmosphere
if grid.Nx > 1
    throw("Grid must not be wider than 1 point in the x direction")
else
    # assumes only 1 point in the x direction
    pressure, temperature = gray_test_t_p_profiles(grid; p0=p_surface, pe=p_top)
end

# Set up atmospheric state
atmospheric_state = GrayAtmosphericState(
    grid; 
    temperature, 
    pressure, 
    otp=optical_properties, 
    lat_center=lat_center
)

# Set up radiation model
sfc_emission = DA{FT}(undef, Nbnd, Nx, Ny)
fill!(sfc_emission, FT(sfc_emissivity))
SLVLW = TwoStreamLWRTE
slv_lw = SLVLW(grid; sfc_emission, lw_inc_flux)

# solve the radiation model
solve_lw!(slv_lw, atmospheric_state)

