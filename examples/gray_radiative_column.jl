# # Gray radiative column
#
# This example constructs a single-column (`1 × 1 × 64`) anelastic atmosphere,
# initializes a simple stratified state, and then computes longwave and shortwave
# radiative fluxes with Breeze's RRTMGP-powered radiative transfer extension.
# The radiative heating profile is finally converted to heating rates in kelvin
# per day for quick inspection.
#
# !!! tip
#     This example requires `RRTMGP.jl`. Make sure it is added to your project
#     so that Breeze's `RadiativeTransferModel` extension is available.
#
# ## Package imports
#
# We load Breeze and Oceananigans for the model setup, and RRTMGP to activate
# the radiative transfer extension. A few utility names from Oceananigans are
# also needed for accessing grid locations.

using Breeze
using RRTMGP

# ## Single-column grid and thermodynamics
#
# We create a vertical column that spans 30 km with 64 uniformly spaced layers,
# and build the default thermodynamic state used by the anelastic formulation.

grid = RectilinearGrid(size=64, z=(0, 30_000), topology=(Flat, Flat, Bounded))

optical_thickness = GrayOpticalThicknessOGorman2008()

#=
rtm = RadiativeTransferModel(
    grid;
    surface_emissivity = 0.98,
    surface_albedo_direct = 0.15,
    surface_albedo_diffuse = 0.15,
    cos_zenith = cosd(60),
    toa_solar_flux = 1_361.0,
    toa_longwave_flux = 0.0,
)
=#
