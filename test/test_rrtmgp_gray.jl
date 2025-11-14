import ClimaComms

using RRTMGP
using RRTMGP: RRTMGPGridParams
using RRTMGP.AngularDiscretizations
using RRTMGP.Fluxes
using RRTMGP.RTE
using RRTMGP.RTESolver
using RRTMGP.Optics
using RRTMGP.GrayUtils
using RRTMGP.AtmosphericStates
using RRTMGP.Sources
using RRTMGP.BCs

import RRTMGP.Parameters.RRTMGPParameters
import ClimaParams as CP

FT = Float64
context = ClimaComms.context()

device = ClimaComms.device(context)
param_set = RRTMGPParameters(FT)
ncol = 1
#     ncol = if device isa ClimaComms.CUDADevice
#         4096
#     else
#         9
#     end

DA = ClimaComms.array_type(device)

nlay = 60                               # number of layers
p0 = FT(100000)                         # surface pressure (Pa)
pe = FT(9000)                           # TOA pressure (Pa)
nbnd, ngpt = 1, 1                       # # of nbands/g-points (=1 for gray radiation)
nlev = nlay + 1                         # # of layers
tb = FT(320)                            # surface temperature
# tstep = 6.0                             # timestep in hours
# Δt = FT(60 * 60 * tstep)                # timestep in seconds
# ndays = 365 * 40                        # # of simulation days
# nsteps = ndays * (24 / tstep)           # number of timesteps
nsteps = 1
temp_toler = FT(0.1)                    # tolerance for temperature (Kelvin)
flux_grad_toler = FT(1e-5)              # tolerance for flux gradient
n_gauss_angles = 1                      # for non-scattering calculation
sfc_emis = DA{FT}(undef, nbnd, ncol)    # surface emissivity
sfc_emis .= FT(1.0)
inc_flux = nothing                      # incoming flux

grid_params = RRTMGPGridParams(FT; context, nlay, ncol)

if ncol == 1
    lat = DA{FT}([0])                   # latitude
else
    lat = DA{FT}(range(FT(-90), FT(90), length = ncol)) # latitude
end

otp = GrayOpticalThicknessOGorman2008(FT) # optical thickness parameters
gray_as = setup_gray_as_pr_grid(context, nlay, lat, p0, pe, otp, param_set, DA)

SLVLW = TwoStreamLWRTE
slv_lw = SLVLW(grid_params; params = param_set, sfc_emis, inc_flux)

(; flux_up, flux_dn, flux_net) = slv_lw.flux
(; t_lay, p_lay, t_lev, p_lev) = gray_as
# sbc = FT(RRTMGP.Parameters.Stefan(param_set))
# cp_d_ = FT(RRTMGP.Parameters.cp_d(param_set))
# grav_ = FT(RRTMGP.Parameters.grav(param_set))
#     #----------------------------------------------------
# hr_lay = DA{FT}(undef, nlay, ncol)
# T_ex_lev = DA{FT}(undef, nlev, ncol)
# flux_grad = DA{FT}(undef, nlay, ncol)
# flux_grad_err = FT(0)

#for i in 1:nsteps
    # calling the long wave gray radiation solver
    solve_lw!(slv_lw, gray_as)
    # computing heating rate
#    compute_gray_heating_rate!(device, hr_lay, p_lev, ncol, nlay, flux_net, cp_d_, grav_)
    # updating t_lay and t_lev based on heating rate
#     update_profile_lw!(
#         device,
#         sbc,
#         t_lay,
#         t_lev,
#         flux_dn,
#         flux_net,
#         hr_lay,
#         flux_grad,
#         T_ex_lev,
#         Δt,
#         nlay,
#         nlev,
#         ncol,
#     )
#     flux_grad_err = maximum(flux_grad)
#     if flux_grad_err < flux_grad_toler
#         break
#     end
# end

#     t_error = maximum(abs.(T_ex_lev .- gray_as.t_lev))
#     color2 = :cyan

#     printstyled("\nGray atmosphere longwave test with ncol = $ncol, nlev = $nlev, solver = $SLVLW\n", color = color2)
#     printstyled("device = $device\n", color = color2)
#     printstyled("Integration time = $(FT(nsteps)/FT(24.0/tstep) / FT(365.0)) years\n\n", color = color2)

#     println("t_error = $(t_error); flux_grad_err = $(flux_grad_err)\n")
# end

# function gray_atmos_sw_test(
#     context,
#     ::Type{SLVSW},
#     ::Type{FT},
#     ncol::Int;
#     exfiltrate = false,
# ) where {FT <: AbstractFloat, SLVSW}
#     param_set = RRTMGPParameters(FT)
#     device = ClimaComms.device(context)
#     DA = ClimaComms.array_type(device)
#     nlay = 60                         # number of layers
#     p0 = FT(100000)                   # surface pressure (Pa)
#     pe = FT(9000)                     # TOA pressure (Pa)
#     nbnd, ngpt = 1, 1                 # # of nbands/g-points (=1 for gray radiation)
#     nlev = nlay + 1                   # # of layers
#     tb = FT(320)                      # surface temperature
#     tstep = 6.0                       # timestep in hours
#     Δt = FT(60 * 60 * tstep)          # timestep in seconds
#     ndays = 365 * 40                  # # of simulation days
#     nsteps = ndays * (24 / tstep)     # number of timesteps
#     n_gauss_angles = 1             # for non-scattering calculation
#     sfc_emis = Array{FT}(undef, nbnd, ncol) # surface emissivity
#     sfc_emis .= FT(1.0)
#     deg2rad = FT(π) / FT(180)
#     grid_params = RRTMGPGridParams(FT; context, nlay, ncol)

#     cos_zenith = DA{FT, 1}(undef, ncol)   # cosine of solar zenith angle
#     toa_flux = DA{FT, 1}(undef, ncol) # top of atmosphere flux
#     sfc_alb_direct = DA{FT, 2}(undef, nbnd, ncol) # surface albedo (direct)
#     sfc_alb_diffuse = DA{FT, 2}(undef, nbnd, ncol) # surface albedo (diffuse)
#     inc_flux_diffuse = nothing
#     otp = GrayOpticalThicknessOGorman2008(FT) # optical thickness parameters

#     top_at_1 = false                          # Top-of-atmos at pt# 1 (true/false)

#     if ncol == 1
#         lat = DA{FT}([0])                     # latitude
#     else
#         lat = DA{FT}(range(FT(-90), FT(90), length = ncol)) # latitude
#     end

#     cos_zenith .= cos(deg2rad * 52.95) # corresponding to ~52.95 deg zenith angle    
#     toa_flux .= FT(1407.679)
#     sfc_alb_direct .= FT(0.1)
#     sfc_alb_diffuse .= FT(0.1)
#     inc_flux_diffuse = nothing

#     as = setup_gray_as_pr_grid(context, nlay, lat, p0, pe, otp, param_set, DA) # init gray atmos state

#     swbcs = (; cos_zenith, toa_flux, sfc_alb_direct, inc_flux_diffuse, sfc_alb_diffuse)
#     slv_sw = SLVSW(grid_params; swbcs...)

#     solve_sw!(slv_sw, as)

#     τ = Array(slv_sw.op.τ)
#     cos_zenith = Array(cos_zenith)
#     flux_dn_dir = Array(slv_sw.flux.flux_dn_dir)
#     toa_flux = Array(toa_flux)

#     # testing with exact solution
#     ot_tot = sum(τ[:, 1]) / cos_zenith[1]
#     exact = (toa_flux[1] * cos_zenith[1]) * exp(-ot_tot)

#     rel_toler = FT(0.001)
#     rel_error = abs(flux_dn_dir[1] - exact) / exact

#     color2 = :cyan

#     printstyled("\nGray atmosphere shortwave test with ncol = $ncol, nlev = $nlev, solver = $SLVSW\n", color = color2)
#     printstyled("device = $device\n\n", color = color2)

#     println("relative error = $rel_error")
# end
