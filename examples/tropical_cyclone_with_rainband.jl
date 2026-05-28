# # [Tropical cyclone with a stationary stratiform rainband](@id tc_rainband_example)
#
# This example reproduces the idealized tropical-cyclone rainband experiments
# of [YuDidlake2019](@citet) (hereafter YD19), who asked a question: if
# you take a mature hurricane and paint a steady, stationary heating pattern
# in one of its stratiform rainbands, what happens to the storm?
#
# Their answer — obtained from a full-physics Weather Research and Forecasting (WRF) model simulation — is a quadrupole
# pattern of secondary-circulation anomalies: rising/sinking pairs that flank
# the imposed heat source, and an accompanying dipole in the tangential wind
# of a few m/s. We get the same pattern here with the Breeze anelastic core,
# a Jordan [Jordan1958](@cite) tropical sounding, a Stern–Nolan balanced
# vortex [SternNolan2009](@citet), and the Moon–Nolan [MoonNolan2010](@cite)
# stratiform heating profile that YD19 borrow.
#
# ## What this example does
#
# We build a balanced tropical-cyclone vortex from the Jordan sounding, run a
# 24 h spinup so the vortex relaxes to numerical equilibrium, and then
# visualize the basic-state vortex alongside the analytic MN10 stratiform
# heating field that YD19 use to drive the rainband response. The two
# follow-on stages (control and heated) and their response diagnostics are
# kept as commented-out stubs at the bottom of the file for users who want
# to reproduce the full YD19 quadrupole.
#
# ## What this simulation teaches
#
# - How to build a balanced-vortex initial condition via Picard iteration
#   (the Nolan 2001 / WRF `em_tropical_cyclone` procedure).
# - How to wire a spatially structured, time-varying source term into the
#   energy equation with `Forcing`.
# - How to inspect a model spinup with azimuthally-averaged diagnostics.
#
# ## Figures produced
#
# | File                             | Content                                           |
# |----------------------------------|---------------------------------------------------|
# | `F02ab_vortex.png`               | basic-state vortex ([YuDidlake2019](@citet); Fig. 2a,b) |
# | `F02cd_heating.png`              | analytic heating ([YuDidlake2019](@citet); Fig. 2c,d)  |

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random
using CairoMakie
using CUDA

Oceananigans.defaults.FloatType = Float32
Random.seed!(42)

if CUDA.functional()
    CUDA.seed!(42)
end

# ## Jordan (1958) hurricane-season mean sounding
#
# The environment is a hydrostatic dry column taken from Table 5 by
# [Jordan1958](@citet) mean West Indies sounding for the
# July–October hurricane season. It's the same climatological profile YD19
# use (and about a million other idealized tropical cyclone studies since).
# Columns are pressure (mb), geopotential height (m), temperature (°C), and
# potential temperature (K).

jordan_p_mb = [
    1015.1, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0,
    550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 175.0, 150.0,
    125.0, 100.0, 80.0, 60.0, 50.0, 40.0, 30.0,
]

jordan_z_m = [
    0.0, 132.0, 583.0, 1054.0, 1547.0, 2063.0, 2609.0, 3182.0,
    3792.0, 4442.0, 5138.0, 5888.0, 6703.0, 7595.0, 8581.0, 9682.0,
    10935.0, 12396.0, 13236.0, 14177.0, 15260.0, 16568.0, 17883.0, 19620.0,
    20743.0, 22139.0, 23971.0,
]

jordan_T_C = [
    26.3, 26.0, 23.0, 19.8, 17.3, 14.6, 11.8, 8.6, 5.1, 1.4,
    -2.5, -6.9, -11.9, -17.7, -24.8, -33.2, -43.3, -55.2, -61.5, -67.6,
    -72.2, -73.5, -69.8, -63.9, -60.6, -57.3, -54.0,
]

jordan_θ_K = [
    298.0, 299.0, 300.0, 302.0, 304.0, 307.0, 309.0, 312.0, 315.0, 318.0,
    321.0, 324.0, 328.0, 332.0, 335.0, 338.0, 342.0, 345.0, 348.0, 354.0,
    364.0, 386.0, 418.0, 468.0, 500.0, 542.0, 597.0,
]

## A 1-D vertical RectilinearGrid carries the Jordan sounding. The 27
## sounding levels map onto the 27 z-faces of a 26-cell Bounded grid, so
## `ZFaceField`s store the data exactly and `Oceananigans.Fields.interpolate`
## handles the linear lookup between levels. The fields are CPU-resident
## because `θ_env`/`T_env`/`p_env` are called host-side during the
## balanced-vortex Picard iteration; `set!` evaluates them on the host
## before copying the result to GPU, so this works for either backend.
const sounding_grid = RectilinearGrid(
    size = length(jordan_z_m) - 1, z = jordan_z_m,
    topology = (Flat, Flat, Bounded)
)

const jordan_θ = ZFaceField(sounding_grid)
const jordan_T = ZFaceField(sounding_grid)
const jordan_p = ZFaceField(sounding_grid)

interior(jordan_θ, 1, 1, :) .= jordan_θ_K
interior(jordan_T, 1, 1, :) .= jordan_T_C .+ 273.15
interior(jordan_p, 1, 1, :) .= jordan_p_mb .* 100

θ_env(z) = Oceananigans.Fields.interpolate(z, jordan_θ)
T_env(z) = Oceananigans.Fields.interpolate(z, jordan_T)
p_env(z) = Oceananigans.Fields.interpolate(z, jordan_p)

# ## YD19 physical parameters
#
# The vortex parameters utilize classic modified-Rankine vortex structure, following the parameters used in [YuDidlake2019](@citet) §3a2:
# a surface maximum wind of 43 m/s with a radius of maximum wind set at 31 km, decaying
# outward as ``r^{-1/2}`` in the modified-Rankine sense. Above ``z ≈ 16`` km
# (the outflow level) the tangential wind is zero. The outer ``\cos^2()`` taper
# between 250 and 300 km is *not* in the original paper. Since our domain is periodic,
# we need to impose this to limit unrealistic stress at the domain boundaries.

f = 5.0e-5                       # f-plane Coriolis parameter, 1/s ([YuDidlake2019](@citet); §3a1)
v_max_surface = 43             # initial surface v_max, m/s ([YuDidlake2019](@citet); §3a2)
a_decay = 0.5                  # modified-Rankine decay exponent ([YuDidlake2019](@citet); Eq. 2)
rmw_surface = 31kilometers     # surface radius of maximum wind, m ([MoonNolan2010](@citet); Appendix A)
z_vortex_top = 16kilometers    # outflow reference level, m ([MoonNolan2010](@citet); v = 0 at RMW at z ≈ 15.9 km)
r_taper_start = 250kilometers  # radial taper start for periodic-domain compatibility, m
r_taper_end = 300kilometers    # radial taper end, m

# ## Output layout
#
# Run outputs live under `examples/output_tc_rainband/figures/` — the two
# F02 PNGs (paper-reproduction outputs).

output_dir = joinpath(@__DIR__, "output_tc_rainband")
figures_dir = joinpath(output_dir, "figures")
mkpath(figures_dir)
nothing #hide

# ## Grid and architecture
#
# [YuDidlake2019](@citet) §3a1 use a 3 km inner-nest resolution with a
# 25 km deep domain. We match that on a ~ 642 km × 642 km periodic-in-x,y box:
# 214² cells horizontally and 75 levels vertically (``Δz ≈ 333`` m). The run
# prefers GPU and falls back to CPU if CUDA isn't functional.
#
# Dynamics: `CompressibleDynamics(SplitExplicitTimeDiscretization())` with
# the [`AcousticRungeKutta3`](@ref) outer time stepper. Acoustic substepping
# replaces the anelastic elliptic pressure solve with linearized acoustic
# substeps, which lets the run go at `Float32` — the anelastic Poisson
# solve loses its precision margin at F32 (the Picard IC's gradient-wind
# residual sits at ~10⁻³ Pa/m on a 10⁵ Pa background, right at F32 ε)
# and so anelastic F32 NaN'd at iter ~99 across all grid resolutions and
# WENO orders tested.

Δx = 3000meters
Lx = 642kilometers
Nx = Ny = Int(Lx / Δx)
Nz = 75
Lz = 25kilometers                 # YD19 §3a1
Δz = Lz / Nz
sponge_rate = 1.0f0 / Float32(333seconds) # ≈ WRF damp_opt=2 `dampcoef`
stage_stop_time = 24hours

arch = CUDA.functional() ? GPU() : CPU()

grid = RectilinearGrid(
    arch;
    size = (Nx, Ny, Nz), halo = (5, 5, 5),
    x = (-Lx / 2, Lx / 2), y = (-Lx / 2, Lx / 2), z = (0, Lz),
    topology = (Periodic, Periodic, Bounded)
)

# ## Reference state and thermodynamic constants
#
# The [`ReferenceState`](@ref Breeze.Thermodynamics.ReferenceState) is
# the Jordan ``θ(z)`` profile, at the observed
# surface pressure, giving us ``pᵣ(z)``, `ρᵣ(z)`, `Tᵣ(z)` — the hydrostatic
# far-field of the anelastic problem. We pull the three columns down to the
# host once, here, for use by the CPU-side balance iteration below.

constants = ThermodynamicConstants()
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
g = constants.gravitational_acceleration
κ = Rᵈ / constants.dry_air.heat_capacity
cᵖᵈ = constants.dry_air.heat_capacity

reference_state = ReferenceState(
    grid, constants;
    surface_pressure = p_env(0.0),
    potential_temperature = θ_env
)

pᵣ = Array(interior(reference_state.pressure, 1, 1, :))
ρᵣ = Array(interior(reference_state.density, 1, 1, :))
Tᵣ = Array(interior(reference_state.temperature, 1, 1, :))
z_centers = collect(range(Δz / 2, step = Δz, length = Nz))

@info @sprintf(
    "Grid: %d × %d × %d   Δx = %.1f km   Lz = %.1f km   Δz = %.1f m",
    Nx, Ny, Nz, Δx / 1.0e3, Lz / 1.0e3, Δz
)
@info @sprintf(
    "Sponge: WRF damp_opt=2 analog, sin²() ramp from z = %.1f to %.1f km, rate %.3f s⁻¹ (ρu, ρv, ρw → 0; ρθ → ρᵣ·θᵣ)",
    20.0, 25.0, sponge_rate
)

# ## Vortex kinematics — RMW(z) and modified-Rankine v(r, z)
#
# The vortex has two shapes to specify: how the radius of maximum wind
# slopes outward with height, and how the tangential wind decays with
# radius. For the first, we follow [SternNolan2009](@citet) Eq. 4.4
# (ultimately due to Eqs. 51 & 56 by [Emanuel1986](@citet)), which gives
#
# ```math
# \text{RMW}(z) = \text{RMW}_\text{sfc} \sqrt{\frac{T(0) - T_\text{out}}{T(z) - T_\text{out}}}
# ```
#
# where ``T_\text{out} = T_\text{env}(z_\text{vortex top})``. This comes
# out of absolute-angular-momentum conservation along M-surfaces that
# coincide with θ*-surfaces in a saturated-neutral tropical atmosphere —
# a *kinematic* slope that uses only the outflow temperature, not the
# full Emanuel PI closure. Above ``z_\text{vortex top} ≈ 16`` km the
# formula is unphysical (``T - T_\text{out}`` changes sign), so we
# hard-zero v there, matching [MoonNolan2010](@cite).
#
# For the tangential wind we use the modified-Rankine profile ([YuDidlake2019](@cite) Eq. 2):
#
# ```math
# v(r, z) = \begin{cases}
#   v_\text{max}(z) \, \dfrac{r}{\text{RMW}(z)} & r \le \text{RMW}(z) \\[8pt]
#   v_\text{max}(z) \left[ \dfrac{\text{RMW}(z)}{r} \right]^{a} & r > \text{RMW}(z)
# \end{cases}
# ```
#
# with ``a = 0.5`` (the modified-Rankine decay exponent). The peak wind
# ``v_\text{max}(z)`` itself scales as ``\text{RMW}_\text{sfc}/\text{RMW}(z)``
# to conserve angular momentum up the vortex column.
#
# The outer cos²() taper between ``r_\text{taper start}`` and ``r_\text{taper end}``
# is our only departure from YD19: their non-periodic outer domain makes
# the taper unnecessary, but we need it so the vortex doesn't wrap around
# on the periodic box and collide with itself.

ΔT_floor_default = 0.01   # micro-floor; avoids ÷ 0 exactly at z = z_vortex_top

function rmw_analytic(z; ΔT_floor = ΔT_floor_default)
    T_out = T_env(z_vortex_top)
    ΔT_0 = T_env(0.0) - T_out
    ΔT_z = max(T_env(z) - T_out, ΔT_floor)
    return rmw_surface * sqrt(ΔT_0 / ΔT_z)
end

function tangential_wind(r, z; ΔT_floor = ΔT_floor_default)
    r ≥ r_taper_end  && return 0.0
    z ≥ z_vortex_top && return 0.0
    rmw_z = rmw_analytic(z; ΔT_floor)
    v_adj = rmw_surface / rmw_z
    vt = r ≤ rmw_z ?
        v_max_surface * v_adj * r / rmw_z :
        v_max_surface * v_adj * (rmw_z / r)^a_decay
    if r > r_taper_start
        ξ = (r - r_taper_start) / (r_taper_end - r_taper_start)
        vt *= cos(π / 2 * ξ)^2
    end
    return vt
end

# ## Balanced-vortex initial condition (Nolan 2001 / WRF `em_tropical_cyclone`)
#
# Having specified ``v(r, z)``, we now need consistent ``p(r, z)``,
# ``ρ(r, z)``, ``T(r, z)`` that put the vortex in *simultaneous* gradient-wind
# and hydrostatic balance on top of the Jordan reference column. That is,
# the fixed point of
#
# ```math
# \begin{aligned}
# \rho &= \frac{p}{R^d \, T}                                       & \text{(ideal gas)} \\
# \frac{\partial p}{\partial r} &= \rho \left( f v + \frac{v^2}{r} \right),
#   \quad \text{outer BC } p(r \to \infty, z) = p_r(z)              & \text{(gradient wind)} \\
# \frac{\partial p}{\partial z} &= -\rho g;
#   \quad \text{recover } T \text{ from } \rho_\text{hyd} = -\frac{1}{g}\frac{\partial p}{\partial z}
#                                                                   & \text{(hydrostatic)}
# \end{aligned}
# ```
#
# This is the initializer used by the WRF `em_tropical_cyclone` ideal test
# case ([Nolan2001](@cite)) and by [YuDidlake2019](@citet) §3a2. We solve
# it by Picard iteration: fix ``T``, compute ``\rho`` from ideal gas,
# sweep ``p`` inward from the outer BC via gradient wind, then recover a
# new ``T`` from the diagnosed hydrostatic density. Under-relaxation
# ``\alpha = 0.5`` on the ``T`` update stabilizes the fixed point, and
# ``T`` is pinned to ``T_r`` at the top level where ``v \approx 0`` (there
# ``\rho_\text{hyd}`` is ill-conditioned at ``p \approx 30`` hPa).
#
# In practice the iteration converges in ~15 sweeps to gradient-wind
# residual ``\sim 10^{-3}`` Pa/m and hydrostatic residual ``\sim 10^{-5}``
# Pa/m — a 10⁴× collapse over the one-shot linearized baseline where
# ``\rho`` is fixed to the environment and not allowed to adjust.

function dpdz_centered(p, k, i, z_centers, Nz)
    if k == 1
        return (p[i, 2] - p[i, 1]) / (z_centers[2] - z_centers[1])
    elseif k == Nz
        return (p[i, Nz] - p[i, Nz - 1]) / (z_centers[Nz] - z_centers[Nz - 1])
    else
        return (p[i, k + 1] - p[i, k - 1]) / (z_centers[k + 1] - z_centers[k - 1])
    end
end

function solve_balanced_vortex_iterative(
        r_grid, z_centers, v2d, p_col, T_col, ρ_col;
        Rᵈ = 287.04, g = 9.81,
        α = 0.5, max_iter = 200, tol = 1.0e-3,
        r_safe_min = 100.0, verbose = true
    )

    Nr = length(r_grid)
    Nz = length(z_centers)
    Δr = r_grid[2] - r_grid[1]
    p = [p_col[k] for i in 1:Nr, k in 1:Nz]
    T = [T_col[k] for i in 1:Nr, k in 1:Nz]
    history = Float64[]
    for iter in 1:max_iter
        T_prev = copy(T)
        ρ = p ./ (Rᵈ .* T)
        for k in 1:Nz
            p[end, k] = p_col[k]
            for i in (Nr - 1):-1:1
                ρ_face = 0.5 * (ρ[i, k] + ρ[i + 1, k])
                v_face = 0.5 * (v2d[i, k] + v2d[i + 1, k])
                r_face = 0.5 * (r_grid[i] + r_grid[i + 1])
                dp_dr = ρ_face * (f * v_face + v_face^2 / max(r_face, r_safe_min))
                p[i, k] = p[i + 1, k] - dp_dr * Δr
            end
        end
        T_new = similar(T)
        for i in 1:Nr
            for k in 1:(Nz - 1)
                dp_dz = dpdz_centered(p, k, i, z_centers, Nz)
                ρ_hyd = max(-dp_dz / g, 1.0e-3)
                T_new[i, k] = p[i, k] / (Rᵈ * ρ_hyd)
            end
            T_new[i, Nz] = T_col[Nz]
        end
        T .= α .* T_new .+ (1 - α) .* T_prev
        maxΔT = maximum(abs.(T .- T_prev))
        push!(history, maxΔT)
        verbose && @info @sprintf("iter %3d   max|ΔT| = %.3e K", iter, maxΔT)
        maxΔT < tol && break
    end
    return (; p, T, ρ = p ./ (Rᵈ .* T), history)
end

## Diagnostic: gradient-wind (∂p/∂r − ρ(fv + v²/r), Pa/m) and hydrostatic
## (∂p/∂z + ρg, Pa/m) residuals on interior points. Iterative solver should
## drive both to round-off; used as a pre-flight sanity check.
function balance_residuals(
        r_grid, z_centers, v2d, p, T;
        Rᵈ = 287.04, g = 9.81, r_safe_min = 100.0
    )
    Nr = length(r_grid)
    Nz = length(z_centers)
    Δr = r_grid[2] - r_grid[1]
    ρ = p ./ (Rᵈ .* T)
    res_gw = zeros(Nr, Nz)
    res_hy = zeros(Nr, Nz)
    for k in 1:Nz, i in 2:(Nr - 1)
        dp_dr = (p[i + 1, k] - p[i - 1, k]) / (2 * Δr)
        res_gw[i, k] = dp_dr -
            ρ[i, k] * (f * v2d[i, k] + v2d[i, k]^2 / max(r_grid[i], r_safe_min))
    end
    for i in 1:Nr, k in 2:(Nz - 1)
        dp_dz = (p[i, k + 1] - p[i, k - 1]) / (z_centers[k + 1] - z_centers[k - 1])
        res_hy[i, k] = dp_dz + ρ[i, k] * g
    end
    return (; gradient_wind = res_gw, hydrostatic = res_hy)
end

rmw_profile = [rmw_analytic(z) for z in z_centers]

Nr_pre = 301
r_pre = collect(range(0.0, r_taper_end, length = Nr_pre))
pˢᵗ = 1.0e5

@info "Computing balanced vortex IC (Picard iteration, Nolan 2001 / WRF em_tropical_cyclone)..."
v2d = [tangential_wind(r_pre[i], z_centers[k]) for i in 1:Nr_pre, k in 1:Nz]
bal = solve_balanced_vortex_iterative(
    r_pre, z_centers, v2d,
    pᵣ, Tᵣ, ρᵣ;
    Rᵈ, g, verbose = false
)

## Anelastic convention (matches Breeze's internal `liquid_ice_potential_temperature`
## diagnostic and the F02b plot): θ is computed with the hydrostatic reference
## pressure, not the balanced p. Keeps the IC warm-core diagnostic consistent
## with the simulation's thermodynamic state.
θ_pre = [bal.T[i, k] * (pˢᵗ / pᵣ[k])^κ for i in 1:Nr_pre, k in 1:Nz]
p′_pre = bal.p .- reshape(pᵣ, 1, Nz)
vortex = (; v = v2d, p = bal.p, θ = θ_pre, T = bal.T, ρ = bal.ρ, p′ = p′_pre)

let res = balance_residuals(r_pre, z_centers, v2d, bal.p, bal.T; Rᵈ, g)
    @info @sprintf(
        "IC residuals: gradient-wind max|res| = %.3e Pa/m,  hydrostatic max|res| = %.3e Pa/m  (iters = %d)",
        maximum(abs, res.gradient_wind), maximum(abs, res.hydrostatic), length(bal.history)
    )
end

@inline function lookup_rz(table, r::Real, z::Real)
    r_c = clamp(r, first(r_pre), last(r_pre))
    z_c = clamp(z, first(z_centers), last(z_centers))
    ir = searchsortedfirst(r_pre, r_c); ir = clamp(ir, 2, length(r_pre))
    iz = searchsortedfirst(z_centers, z_c); iz = clamp(iz, 2, length(z_centers))
    r0, r1 = r_pre[ir - 1], r_pre[ir]
    z0, z1 = z_centers[iz - 1], z_centers[iz]
    fr = (r_c - r0) / (r1 - r0)
    fz = (z_c - z0) / (z1 - z0)
    v00 = table[ir - 1, iz - 1]; v10 = table[ir, iz - 1]
    v01 = table[ir - 1, iz];     v11 = table[ir, iz]
    return (1 - fr) * (1 - fz) * v00 + fr * (1 - fz) * v10 +
        (1 - fr) * fz * v01 + fr * fz * v11
end

uᵢ(x, y, z) = (r = sqrt(x^2 + y^2); r < 1.0 ? 0.0 : -(y / r) * lookup_rz(vortex.v, r, z))
vᵢ(x, y, z) = (r = sqrt(x^2 + y^2); r < 1.0 ? 0.0 : (x / r) * lookup_rz(vortex.v, r, z))
Tᵢ(x, y, z) = lookup_rz(vortex.T, sqrt(x^2 + y^2), z)
ρᵢ(x, y, z) = lookup_rz(vortex.ρ, sqrt(x^2 + y^2), z)

# ## Rainband heating — stationary, spiral, outward-tilted
#
# This is the whole point of the paper: we impose a steady stratiform
# heating profile in the lower-right quadrant of the storm and watch the
# dynamic response. The profile, straight out of [YuDidlake2019](@citet)
# Eq. 3 (which in turn follows [MoonNolan2010](@cite)), is
#
# ```math
# Q(r, \lambda, z, t) = Q_\text{max} \,
#   \exp\!\left\{-\frac{[r - r_\text{bs}(\lambda, z)]^2}{2 \sigma_r^2}\right\} \,
#   V_z(z) \, A_\lambda(\lambda) \, R(t)
# ```
#
# where the radial centerline is
# ``r_\text{bs}(\lambda, z) = [60 - 10\lambda/(\pi/4)]\,\text{km} + z``
# (a spiral with 60 km at the downwind east edge ``\lambda = 0`` and 80 km
# at the upwind south edge ``\lambda = -\pi/2``; the ``+z`` gives outward
# tilt with height), the vertical shape
# ``V_z(z) = \sin[\pi (z - z_\text{bs})/\sigma_{zs}]`` is active for
# ``|z - z_\text{bs}| < \sigma_{zs}`` and zero elsewhere, the azimuthal
# window ``A_\lambda(\lambda) = \exp\{-[\lambda_c/(\pi/4)]^8\}`` is a
# super-Gaussian centered at ``\lambda = -\pi/4``, and the time ramp
# ``R(t)`` is a 1-hour linear ramp from zero to full strength to avoid an
# instantaneous shock.
#
# The energy-equation source is keyed to the model's thermodynamic prognostic.
# This driver uses the default `:LiquidIcePotentialTemperature` formulation
# (prognostic ``\rho\theta``), so a heating rate ``Q`` (K/s) gives
#
# ```math
# \left. \frac{\partial(\rho\theta)}{\partial t} \right|_\text{heat}
#     = \rho \, \frac{Q}{\Pi},
# \qquad \Pi(z) = \left(\frac{p_r(z)}{p^{\text{st}}}\right)^{R^d/c_p^d}
# ```
#
# i.e. heating raises ``T`` at rate ``Q``, which raises ``\theta`` at rate
# ``Q/\Pi``. Within the WRF/MN10 idealized framework, ``\rho \approx \rho_r(z)``
# inside the rainband, so we use the host-side reference profile.

Q_max = 4.24f0 / Float32(hour)     # YD19 Eq. 3 Q_max = 4.24 K/h (stored in K/s)
z_bs = Float32(4kilometers)
σ_r = Float32(6kilometers)
σ_zs = Float32(2kilometers)
t_full = Float32(1hour)            # 1 h ramp to avoid instantaneous onset

ρᵣ_device = arch isa GPU ? CuArray(ρᵣ) : ρᵣ

## Reference Exner-function profile Πᵣ(z) = (pᵣ(z)/pˢᵗ)^κ. Used to convert
## the analytic ``T``-tendency ``Q`` (K/s) into the ρθ-tendency ``ρ Q / Π``.
Πᵣ = Float32[(pᵣ[k] / pˢᵗ)^κ for k in 1:Nz]
Πᵣ_device = arch isa GPU ? CuArray(Πᵣ) : Πᵣ


## Precompute once at module load.
const π_F32 = Float32(π)
const π_4_F32 = Float32(π / 4)
const π_2_F32 = Float32(π / 2)
const twoπ_F32 = Float32(2π)

@inline function rainband_heating(i, j, k, grid, clock, fields, p)
    x = Oceananigans.Grids.xnode(i, j, k, grid, Center(), Center(), Center())
    y = Oceananigans.Grids.ynode(i, j, k, grid, Center(), Center(), Center())
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    t = clock.time

    ramp = clamp((t - p.t_on) / (p.t_full - p.t_on), 0, 1)
    r = sqrt(x^2 + y^2)
    λ = atan(y, x)
    r_bs = (60 - 10 * (λ / π_4_F32)) * 1000 + z
    G_r = exp(-(r - r_bs)^2 / 2p.σ_r^2)

    z_rel = (z - p.z_bs) / p.σ_zs
    V_z = ifelse(abs(z_rel) < 1, sin(π_F32 * z_rel), 0.0f0)

    λ_c = mod(λ + π_4_F32 + π_F32, twoπ_F32) - π_F32
    A_λ = exp(-(λ_c / π_4_F32)^8)

    Q = p.Q_max * G_r * V_z * A_λ * ramp
    ## ρθ tendency: ρᵣ · Q / Πᵣ. WRF uses full ρ; under the WRF/MN10 idealized
    ## framework ρ ≈ ρᵣ(z) inside the rainband, so the difference is second
    ## order in p'/p.
    ρᵣ_k = @inbounds p.ρᵣ[k]
    Πᵣ_k = @inbounds p.Πᵣ[k]
    return ρᵣ_k * Q / Πᵣ_k
end

heating_params = (;
    Q_max, σ_r, σ_zs, z_bs, t_on = 0.0f0, t_full,
    ρᵣ = ρᵣ_device, Πᵣ = Πᵣ_device,
)

heating_forcing = Forcing(rainband_heating; discrete_form = true, parameters = heating_params)

## Analytic heating rate at full strength (K/h) — for figure contours.
function heating_rate_K_per_hour(r, λ, z)
    r_bs = (60.0 - 10.0 * (λ / (π / 4))) * 1000.0 + z
    G_r = exp(-(r - r_bs)^2 / 2σ_r^2)
    z_rel = (z - z_bs) / σ_zs
    V_z = abs(z_rel) < 1 ? sin(π * z_rel) : 0.0
    λ_c = mod(λ - (-π / 4) + π, 2π) - π
    A_λ = exp(-(λ_c / (π / 4))^8)
    return 4.24 * G_r * V_z * A_λ
end

# ## Upper-level sponge (WRF `damp_opt=2` analog)
#
# The top of the domain needs a Rayleigh-damping layer to absorb outgoing
# gravity waves that would otherwise reflect off the rigid lid and
# destabilize the interior. We match WRF's `damp_opt = 2` shape: a sin²() ramp
# from zero at ``z = 20`` km to a max rate of ``3 \times 10^{-3}`` s⁻¹
# at ``z = 25`` km. Momentum components relax to zero; ρθ relaxes to its
# reference profile
#
# ```math
# (\rho\theta)_r(z) = \rho_r(z) \, \theta_r(z)
# ```
#
# Both the momentum and ρθ components are needed: without the ρθ term,
# upper-level ``\theta'`` anomalies persist and the vortex fails to spin down.

sponge_z_bottom = Float32(20kilometers)
sponge_z_top = Float32(25kilometers)

## Reference ρθ profile (kg K / m³). The sponge relaxes ρθ to this profile
## in the upper-level damping layer. Using the default (LiquidIcePotentialTemperature)
## formulation because :StaticEnergy + CompressibleDynamics is currently broken
## on GPU (gpu__compute_temperature_and_pressure! method-error).
ρθᵣ = Float32[ρᵣ[k] * θ_env(z_centers[k]) for k in 1:Nz]
ρθᵣ_device = arch isa GPU ? CuArray(ρθᵣ) : ρθᵣ

sponge_vel_params = (z_bot = sponge_z_bottom, z_top = sponge_z_top, rate = sponge_rate)
sponge_ρθ_params = (
    z_bot = sponge_z_bottom, z_top = sponge_z_top, rate = sponge_rate,
    ρθ_bg = ρθᵣ_device,
)

## WRF `damp_opt=2` analog: zero below z_bot, sin²() ramp to max at z_top.
@inline function sponge_mask(z, z_bot, z_top)
    ξ = (z - z_bot) / (z_top - z_bot)
    return ifelse(ξ ≤ 0, zero(ξ), sin(π_2_F32 * ξ)^2)
end

@inline function sponge_ρu_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Face(), Center(), Center())
    mask = sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρu[i, j, k]
end

@inline function sponge_ρv_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Face(), Center())
    mask = sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρv[i, j, k]
end

@inline function sponge_ρw_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Face())
    mask = sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρw[i, j, k]
end

@inline function sponge_ρθ_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    mask = sponge_mask(z, p.z_bot, p.z_top)
    ρθ_tgt = @inbounds p.ρθ_bg[k]
    return -p.rate * mask * (@inbounds fields.ρθ[i, j, k] - ρθ_tgt)
end

sponge_ρu = Forcing(sponge_ρu_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρv = Forcing(sponge_ρv_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρw = Forcing(sponge_ρw_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρθ = Forcing(sponge_ρθ_fn; discrete_form = true, parameters = sponge_ρθ_params)

# ## Surface fluxes — omitted
#
# YD19 §3a1 uses Emanuel-1986 bulk aerodynamic drag (Cᴰ) and sensible heat
# (Cᵀ) over a 300 K SST. We omit them here so the response field is the
# direct dynamical response to the prescribed rainband heating.

# ## Model builder

function build_model(; with_heating::Bool)
    coriolis = FPlane(; f)
    dynamics = CompressibleDynamics(
        SplitExplicitTimeDiscretization();
        surface_pressure = p_env(0.0),
        reference_potential_temperature = θ_env
    )
    advection = WENO(order = 5)
    forcing = with_heating ?
        (ρu = sponge_ρu, ρv = sponge_ρv, ρw = sponge_ρw, ρθ = (heating_forcing, sponge_ρθ)) :
        (ρu = sponge_ρu, ρv = sponge_ρv, ρw = sponge_ρw, ρθ = sponge_ρθ)

    return AtmosphereModel(
        grid; dynamics, coriolis, advection, forcing
    )
end

# ## Stage runner
# We build the model, run it, and save the full state for later analysis.
struct InMemoryFTS
    times::Vector{Float64}
    data::Vector{Array{Float32, 3}}
end
InMemoryFTS() = InMemoryFTS(Float64[], Array{Float32, 3}[])
Base.getindex(s::InMemoryFTS, n::Int) = s.data[n]
Base.length(s::InMemoryFTS) = length(s.data)

function build_and_run_stage!(
        stage_label::String;
        with_heating::Bool, init, stop_time
    )
    model = build_model(; with_heating)
    set!(model; init...)

    simulation = Simulation(model; Δt = 2.0, stop_time)
    conjure_time_step_wizard!(simulation, cfl = 0.5)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

    u, v, w = model.velocities

    function progress(sim)
        msg = @sprintf(
            "[%s] iter: %d, t: %s, Δt: %s, max|u,v|: %.2f, %.2f m/s, max|w|: %.2e m/s",
            stage_label, iteration(sim), prettytime(sim), prettytime(sim.Δt),
            maximum(abs, u), maximum(abs, v), maximum(abs, w)
        )
        @info msg
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(6hour))

    captures = (
        u = InMemoryFTS(), v = InMemoryFTS(), w = InMemoryFTS(),
        T = InMemoryFTS(), ρ = InMemoryFTS(),
    )
    function capture_state!(sim)
        t = sim.model.clock.time
        push!(captures.u.times, t); push!(captures.u.data, Array(interior(sim.model.velocities.u)))
        push!(captures.v.times, t); push!(captures.v.data, Array(interior(sim.model.velocities.v)))
        push!(captures.w.times, t); push!(captures.w.data, Array(interior(sim.model.velocities.w)))
        push!(captures.T.times, t); push!(captures.T.data, Array(interior(sim.model.temperature)))
        push!(captures.ρ.times, t); push!(captures.ρ.data, Array(interior(sim.model.dynamics.density)))
        return nothing
    end
    add_callback!(simulation, capture_state!, TimeInterval(1hour))

    @info "Running $stage_label stage for $(prettytime(stop_time))"
    run!(simulation)

    post = (
        u = Array(interior(model.velocities.u)),
        v = Array(interior(model.velocities.v)),
        T = Array(interior(model.temperature)),
        ρ = Array(interior(model.dynamics.density)),
    )

    model = nothing
    GC.gc(); CUDA.reclaim()
    return (post = post, captures = captures)
end


@info "=== Spinup: $(prettytime(stage_stop_time)) ==="
spinup_result = build_and_run_stage!(
    "spinup";
    with_heating = false,
    init = (u = uᵢ, v = vᵢ, T = Tᵢ, ρ = ρᵢ),
    stop_time = stage_stop_time,
)
post_spinup = spinup_result.post
nothing # hide

# ## Stage 4 — Analysis and figure production
# Now that the have the full simulation, we replicate the figures from YD19 to verify our results.

let
    @info "=== Stage 4: Analysis ==="

    u_sc = Array{Float32}(undef, Nx, Ny, Nz)
    v_sc = similar(u_sc)
    w_sc = similar(u_sc)
    T_sc = similar(u_sc)

    ## In-place centered interpolation (Arakawa C-grid → Centers).
    function center_u!(out::AbstractArray{Float32, 3}, src)
        Nxs, Nys, Nzs = size(out)
        return @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
            ip = i == Nxs ? 1 : i + 1
            out[i, j, k] = Float32((src[i, j, k] + src[ip, j, k]) / 2)
        end
    end
    function center_v!(out::AbstractArray{Float32, 3}, src)
        Nxs, Nys, Nzs = size(out)
        return @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
            jp = j == Nys ? 1 : j + 1
            out[i, j, k] = Float32((src[i, j, k] + src[i, jp, k]) / 2)
        end
    end
    function center_w!(out::AbstractArray{Float32, 3}, src)
        ## w lives on z-Face (size Nz+1 in the vertical). Average adjacent
        ## faces to get cell-centered values with size (Nx, Ny, Nz).
        Nxs, Nys, Nzs = size(out)
        return @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
            out[i, j, k] = Float32((src[i, j, k] + src[i, j, k + 1]) / 2)
        end
    end
    function copy_f32!(out::AbstractArray{Float32, 3}, src)
        return @inbounds for I in eachindex(out)
            out[I] = Float32(src[I])
        end
    end

    ## Load one captured snapshot
    function load_snapshot!(u, v, w, T, u_ts, v_ts, w_ts, T_ts, n::Int)
        center_u!(u, u_ts[n])
        center_v!(v, v_ts[n])
        center_w!(w, w_ts[n])
        copy_f32!(T, T_ts[n])
        return nothing
    end

    r_bin_edges = collect(range(0.0, 150kilometers, step = Δx))
    Nr_bin = length(r_bin_edges) - 1
    r_bin_centers = 0.5 .* (r_bin_edges[1:(end - 1)] .+ r_bin_edges[2:end])
    xs_center = Float32.(xnodes(grid, Center()))
    ys_center = Float32.(ynodes(grid, Center()))

    ## Instead of plotting r x z cross sections, we are going to calculate a sector azimuthal average.
    ## This will smooth out small-scale perturbations.
    vθ_ws = zeros(Float32, Nr_bin, Nz)
    vr_ws = similar(vθ_ws)
    w_ws = similar(vθ_ws)
    T_ws = similar(vθ_ws)
    ct_ws = zeros(Int32, Nr_bin, Nz)
    r_last = Float32(last(r_bin_edges))

    function azimuthal_mean!(vθ, vr, ww, TT, ct, u, v, w, T)
        fill!(vθ, 0.0f0); fill!(vr, 0.0f0); fill!(ww, 0.0f0); fill!(TT, 0.0f0); fill!(ct, 0)
        Nxs, Nys, Nzs = size(u)
        @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
            x = xs_center[i]; y = ys_center[j]
            r = sqrt(x^2 + y^2)
            r ≥ r_last && continue
            ib = searchsortedlast(r_bin_edges, Float64(r))
            ib = clamp(ib, 1, Nr_bin)
            rs = max(r, 1.0f0)
            xh = x / rs
            yh = y / rs
            uij = u[i, j, k]
            vij = v[i, j, k]
            vθ[ib, k] += -yh * uij + xh * vij
            vr[ib, k] += xh * uij + yh * vij
            ww[ib, k] += w[i, j, k]
            TT[ib, k] += T[i, j, k]
            ct[ib, k] += 1
        end
        return @inbounds for k in 1:Nzs, ib in 1:Nr_bin
            if ct[ib, k] > 0
                inv = 1.0f0 / ct[ib, k]
                vθ[ib, k] *= inv
                vr[ib, k] *= inv
                ww[ib, k] *= inv
                TT[ib, k] *= inv
            end
        end
    end

    ## ---------------------------------------------------------
    ## F02ab — basic-state vortex (YD19 Fig 2a,b)
    ## ---------------------------------------------------------
    let
        @info "Producing F02ab (basic-state vortex)..."
        ts_spin = spinup_result.captures
        n_final = length(ts_spin.u.times)
        t_final = ts_spin.u.times[n_final]
        @info @sprintf(
            "Spinup snapshot %d of %d  (t = %.2f h)",
            n_final, length(ts_spin.u.times), t_final / hour
        )

        load_snapshot!(u_sc, v_sc, w_sc, T_sc, ts_spin.u, ts_spin.v, ts_spin.w, ts_spin.T, n_final)
        azimuthal_mean!(vθ_ws, vr_ws, w_ws, T_ws, ct_ws, u_sc, v_sc, w_sc, T_sc)

        θ_bar = similar(T_ws)
        for k in 1:Nz, ib in 1:Nr_bin
            θ_bar[ib, k] = T_ws[ib, k] * Float32((pˢᵗ / pᵣ[k])^κ)
        end
        θ_env_col = Float32[θ_env(z_centers[k]) for k in 1:Nz]
        θ_anom = θ_bar .- reshape(θ_env_col, 1, :)

        fig = Figure(size = (1300, 520))

        ax_v = Axis(
            fig[1, 1]; xlabel = "Radius (km)", ylabel = "Height (km)",
            title = "(a) Basic-state tangential wind v̄",
            limits = (0, 150, 0, 22)
        )
        v_cr_hi = 50
        hm_v = heatmap!(
            ax_v, r_bin_centers ./ kilometer, z_centers ./ kilometer, vθ_ws;
            colormap = :inferno, colorrange = (0, v_cr_hi)
        )
        contour!(
            ax_v, r_bin_centers ./ kilometer, z_centers ./ kilometer, vθ_ws;
            levels = 5:5:v_cr_hi, color = :white, linewidth = 0.8
        )
        Colorbar(fig[1, 2], hm_v; label = "v̄ (m s⁻¹)")

        ax_θ = Axis(
            fig[1, 3]; xlabel = "Radius (km)", ylabel = "Height (km)",
            title = "(b) Potential-temperature anomaly θ̄'",
            limits = (0, 150, 0, 22)
        )
        θ_span = max(15.0, ceil(maximum(abs, θ_anom)))
        hm_θ = heatmap!(
            ax_θ, r_bin_centers ./ kilometer, z_centers ./ kilometer, θ_anom;
            colormap = :balance, colorrange = (-θ_span, θ_span)
        )
        contour!(
            ax_θ, r_bin_centers ./ kilometer, z_centers ./ kilometer, θ_anom;
            levels = -θ_span:1.69:θ_span, color = :black, linewidth = 0.5
        )
        Colorbar(fig[1, 4], hm_θ; label = "θ̄' (K)")

        Label(
            fig[0, :],
            @sprintf(
                "F02ab — Basic-state vortex at t = %.1f h spin-up (YD19 Fig 2a,b, %.0f km box)",
                t_final / hour, Lx / kilometers
            );
            fontsize = 17
        )

        v_peak_sfc = maximum(@view vθ_ws[:, 1])
        r_peak_sfc = r_bin_centers[argmax(@view vθ_ws[:, 1])] / kilometers
        v_peak_all = maximum(vθ_ws)
        idx_all = argmax(vθ_ws)
        r_peak_all = r_bin_centers[idx_all[1]] / kilometers
        z_peak_all = z_centers[idx_all[2]] / kilometers
        θ_peak = maximum(θ_anom)
        idx_θ = argmax(θ_anom)
        r_θ_peak = r_bin_centers[idx_θ[1]] / kilometers
        z_θ_peak = z_centers[idx_θ[2]] / kilometers
        @info @sprintf(
            "F02a: surface v̄_peak = %.2f m/s at r = %.1f km (YD19 target ≈ 40 m/s)",
            v_peak_sfc, r_peak_sfc
        )
        @info @sprintf(
            "F02a: global v̄_peak  = %.2f m/s at (r,z) = (%.1f, %.1f) km",
            v_peak_all, r_peak_all, z_peak_all
        )
        @info @sprintf(
            "F02b: peak θ'        = %.2f K at (r,z) = (%.1f, %.1f) km (YD19 ~12 K at 10-12 km)",
            θ_peak, r_θ_peak, z_θ_peak
        )

        path = joinpath(figures_dir, "F02ab_vortex.png")
        save(path, fig)
        @info "Saved F02ab" path
        GC.gc()
    end

    ## ---------------------------------------------------------
    ## F02cd — analytic heating field (YD19 Fig 2c,d)
    ## ---------------------------------------------------------
    @info "Producing F02cd (heating field)..."
    r_cs = collect(range(0.0, 150kilometers, length = 151))
    z_cs = collect(range(0.0, 12kilometers, length = 61))
    λ_mid = -π / 4
    Q_cs = [heating_rate_K_per_hour(r, λ_mid, z) for r in r_cs, z in z_cs]

    x_pv = collect(range(-Lx / 2, Lx / 2, length = 300))
    y_pv = copy(x_pv)
    z_level = 4.6kilometers
    Q_pv = zeros(length(x_pv), length(y_pv))
    for j in eachindex(y_pv), i in eachindex(x_pv)
        r = sqrt(x_pv[i]^2 + y_pv[j]^2)
        λ = atan(y_pv[j], x_pv[i])
        Q_pv[i, j] = heating_rate_K_per_hour(r, λ, z_level)
    end

    Q_lim = 4.5
    fig = Figure(size = (1300, 520))

    ax_c = Axis(
        fig[1, 1]; xlabel = "Radius (km)", ylabel = "Height (km)",
        title = "(c) Heating cross section at λ = -π/4 (middle of rainband)",
        limits = (0, 150, 0, 12)
    )
    hm_c = heatmap!(
        ax_c, r_cs ./ kilometer, z_cs ./ kilometer, Q_cs;
        colormap = :balance, colorrange = (-Q_lim, Q_lim)
    )
    contour!(
        ax_c, r_cs ./ kilometer, z_cs ./ kilometer, Q_cs;
        levels = -4:1:4, color = :black, linewidth = 0.6
    )
    Colorbar(fig[1, 2], hm_c; label = "Q (K h⁻¹)")

    ax_d = Axis(
        fig[1, 3]; xlabel = "x (km)", ylabel = "y (km)",
        title = @sprintf("(d) Heating plan view at z = %.1f km", z_level / 1000),
        aspect = DataAspect(), limits = (-120, 120, -120, 120)
    )
    hm_d = heatmap!(
        ax_d, x_pv ./ kilometer, y_pv ./ kilometer, Q_pv;
        colormap = :balance, colorrange = (-Q_lim, Q_lim)
    )
    contour!(
        ax_d, x_pv ./ kilometer, y_pv ./ kilometer, Q_pv;
        levels = -4:0.5:4, color = :black, linewidth = 0.4
    )
    Colorbar(fig[1, 4], hm_d; label = "Q (K h⁻¹)")

    Label(
        fig[0, :],
        @sprintf(
            "F02cd — MN10 stratiform heating field (YD19 Fig 2c,d, %.0f km box)",
            Lx / 1.0e3
        );
        fontsize = 17
    )

    peak_Q_cs = maximum(abs, Q_cs)
    peak_Q_pv = maximum(abs, Q_pv)
    @info @sprintf("F02c peak |Q| = %.2f K/h (YD19 Q_max = 4.24 K/h)", peak_Q_cs)
    @info @sprintf("F02d peak |Q| at z=%.1fkm = %.2f K/h", z_level / 1.0e3, peak_Q_pv)

    path = joinpath(figures_dir, "F02cd_heating.png")
    save(path, fig)
    @info "Saved F02cd" path

    @info "=== Analysis complete ==="
end

# ## Reproducing the full YD19 response (optional)
#
# The two stubs below extend the spinup into a control / heated pair, which is
# what YD19 actually subtract to get the quadrupole response. They are kept
# commented out so this example stays short.
#
# ```julia
# @info "=== Control: $(prettytime(stage_stop_time)) ==="
# control_result = build_and_run_stage!(
#     "control";
#     with_heating = false,
#     init = post_spinup,
#     stop_time = stage_stop_time,
# )
#
# @info "=== Heated: $(prettytime(stage_stop_time)) (MN10 stratiform) ==="
# heated_result = build_and_run_stage!(
#     "heated";
#     with_heating = true,
#     init = post_spinup,
#     stop_time = stage_stop_time,
# )
# ```
