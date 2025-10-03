using RRTMGP.Optics: GrayOpticalThicknessSchneider2004
using RRTMGP.AtmosphericStates: GrayAtmosphericState
using RRTMGP.RTE: TwoStreamLWRTE, TwoStreamSWRTE
using RRTMGP.RTESolver: solve_lw!, solve_sw!

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
deg2rad = FT(π / 180)

lw_inc_flux = nothing # no incoming longwave flux
sw_inc_flux = FT(1407.679) # no incoming shortwave flux
sw_inc_flux_diffuse = nothing
zenith_angle = FT(52.95) # zenith angle in degrees
sfc_emissivity = FT(1) # surface emissivity
albedo_direct = FT(0.1) # surface albedo (direct)
albedo_diffuse = FT(0.1) # surface albedo (diffuse)

planet_radius = 6_371_000
p_surface = 100_000 # surface pressure (Pa)
p_top = 9_000 # top of atmosphere pressure / emission level (Pa)
lat_center = 0
optical_properties = GrayOpticalThicknessSchneider2004(FT)
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
sfc_alb_direct = DA{FT}(undef, Nbnd, Nx, Ny)
sfc_alb_diffuse = DA{FT}(undef, Nbnd, Nx, Ny)
cos_zenith = DA{FT}(undef, Nx, Ny)
toa_flux = DA{FT}(undef, Nx, Ny)
lw_toa_inc_flux = nothing
inc_flux_diffuse = nothing
fill!(sfc_emission, FT(sfc_emissivity))
fill!(sfc_alb_direct, FT(albedo_direct))
fill!(sfc_alb_diffuse, FT(albedo_diffuse))
fill!(cos_zenith, FT(cos(deg2rad * zenith_angle)))
fill!(toa_flux, FT(sw_inc_flux))
SLVLW = TwoStreamLWRTE
SLVSW = TwoStreamSWRTE
lw_params = (; sfc_emission, lw_inc_flux)
sw_params = (; cos_zenith, toa_flux, sfc_alb_direct, inc_flux_diffuse, sfc_alb_diffuse)
slv_lw = SLVLW(grid; lw_params...)
slv_sw = SLVSW(grid; sw_params...)

# Solve the LW radiation model
function update_radative_fluxes!(slv_lw, slv_sw, atmospheric_state)
    solve_lw!(slv_lw, atmospheric_state)
    solve_sw!(slv_sw, atmospheric_state)
end

update_radative_fluxes!(slv_lw, slv_sw, atmospheric_state)
