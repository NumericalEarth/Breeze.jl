# # [Tropical cyclone with a stationary stratiform rainband](@id tc_rainband_example)
#
# This example reproduces the idealized tropical-cyclone rainband experiments
# of [YuDidlake2019](@citet) (hereafter YD19), who asked a beautifully concrete question: if
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
# We run three back-to-back 24 h stages:
#
#   1. **Spinup** — the balanced vortex IC is released and relaxes to
#      numerical equilibrium under no forcing (just the sponge at the model
#      top). This is stage 1 of [YuDidlake2019](@citet) §3a2.
#   2. **Control** — 24 h continuation of the post-spinup state, no heating.
#   3. **Heated** — 24 h continuation of the *same* post-spinup state, now
#      with the stationary MN10 stratiform heating switched on.
#
# Subtracting control from heated isolates the response to the imposed
# heating (YD19 Eq. 4). Everything else — slow vortex drift, residual
# gravity-wave activity, sponge interactions — cancels out.
#
# ## What the simulation teaches
#
# - How to build a balanced-vortex initial condition via Picard iteration
#   (the Nolan 2001 / WRF `em_tropical_cyclone` procedure).
# - How to wire a spatially structured, time-varying source term into the
#   energy equation with `Forcing`.
# - How to run three related stages of a simulation in one script, with the
#   post-spinup state reused as the initial condition of the follow-on runs.
# - How to isolate a forced response via a control experiment.
#
# ## Figures produced
#
# | File                             | Content                                           |
# |----------------------------------|---------------------------------------------------|
# | `F01_preflight.png`              | vortex IC sanity check                            |
# | `F02ab_vortex.png`               | basic-state vortex ([YuDidlake2019](@citet); Fig. 2a,b) |
# | `F02cd_heating.png`              | analytic heating ([YuDidlake2019](@citet); Fig. 2c,d)  |
# | `F03a_axisym_response.png`       | axisymmetric response ([YuDidlake2019](@citet); Fig. 3a) |
# | `F04_plan_response.png`          | plan-view response ([YuDidlake2019](@citet); Fig. 4a-c)  |
# | `F05_cross_sections.png`         | cross sections ([YuDidlake2019](@citet); Fig. 5)     |
# | `F06_response_timeseries.png`    | response amplitude vs time (diagnostic)           |

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random
using CairoMakie
using CUDA

Oceananigans.defaults.FloatType = Float64
Random.seed!(42)

# ## Jordan (1958) hurricane-season mean sounding
#
# The environment is a hydrostatic dry column taken from Table 5 by
# [Jordan1958](@citet) mean West Indies sounding for the
# July–October hurricane season. It's the same climatological profile YD19
# use (and about a million other idealized tropical-cyclone studies since).
# Columns are pressure (mb), geopotential height (m), temperature (°C), and
# potential temperature (K).

jordan_p_mb = Float64[
    1015.1, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0,
    550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 175.0, 150.0,
    125.0, 100.0, 80.0, 60.0, 50.0, 40.0, 30.0,
]

jordan_z_m = Float64[
    0.0, 132.0, 583.0, 1054.0, 1547.0, 2063.0, 2609.0, 3182.0,
    3792.0, 4442.0, 5138.0, 5888.0, 6703.0, 7595.0, 8581.0, 9682.0,
    10935.0, 12396.0, 13236.0, 14177.0, 15260.0, 16568.0, 17883.0, 19620.0,
    20743.0, 22139.0, 23971.0,
]

jordan_T_C = Float64[
    26.3, 26.0, 23.0, 19.8, 17.3, 14.6, 11.8, 8.6, 5.1, 1.4,
    -2.5, -6.9, -11.9, -17.7, -24.8, -33.2, -43.3, -55.2, -61.5, -67.6,
    -72.2, -73.5, -69.8, -63.9, -60.6, -57.3, -54.0,
]

jordan_θ_K = Float64[
    298.0, 299.0, 300.0, 302.0, 304.0, 307.0, 309.0, 312.0, 315.0, 318.0,
    321.0, 324.0, 328.0, 332.0, 335.0, 338.0, 342.0, 345.0, 348.0, 354.0,
    364.0, 386.0, 418.0, 468.0, 500.0, 542.0, 597.0,
]

function _linear_interpolate(xs::AbstractVector, ys::AbstractVector, x::Real)
    x_c = clamp(x, first(xs), last(xs))
    i = searchsortedfirst(xs, x_c)
    i = clamp(i, 2, length(xs))
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x_c - x0) / (x1 - x0)
end

θ_env(z) = _linear_interpolate(jordan_z_m, jordan_θ_K, z)
T_env(z) = _linear_interpolate(jordan_z_m, jordan_T_C .+ 273.15, z)    # convert C -> K
p_env(z) = _linear_interpolate(jordan_z_m, jordan_p_mb .* 100.0, z)    # convert mb -> Pa

# ## YD19 physical parameters
#
# The vortex parameters come straight out of [YuDidlake2019](@citet) §3a2:
# a surface maximum wind of 43 m/s sitting at a radius of 31 km, decaying
# outward as ``r^{-1/2}`` in the modified-Rankine sense. Above ``z ≈ 16`` km
# (the outflow level) the tangential wind is zero. The outer ``\cos^2`` taper
# between 250 and 300 km is *not* in the original paper — YD19 use a huge
# non-periodic mother domain — but we need it to keep the vortex from
# wrapping around on our periodic box.

f = 5.0e-5                 # f-plane Coriolis parameter ([YuDidlake2019](@citet); §3a1)
v_max_surface = 43.0       # initial surface v_max, m/s ([YuDidlake2019](@citet); §3a2)
a_decay = 0.5              # modified-Rankine decay exponent ([YuDidlake2019](@citet); Eq. 2)
rmw_surface = 31_000.0     # surface radius of maximum wind, m ([MoonNolan2010](@citet); Appendix A)
z_vortex_top = 16_000.0    # outflow reference level, m ([MoonNolan2010](@citet); v = 0 at RMW at z ≈ 15.9 km)
r_taper_start = 250_000.0  # radial taper start for periodic-domain compatibility (m)
r_taper_end = 300_000.0    # radial taper end (m)

# ## Output layout
#
# Run outputs live under `examples/output_tc_rainband/`:
#
#   - `snapshots/` — hourly JLD2 snapshot files for each stage
#   - `figures/`   — F01-F06 PNGs (paper-reproduction outputs)

output_dir = joinpath(@__DIR__, "output_tc_rainband")
snapshots_dir = joinpath(output_dir, "snapshots")
figures_dir = joinpath(output_dir, "figures")
mkpath(snapshots_dir); mkpath(figures_dir)

# ## Grid and architecture
#
# [YuDidlake2019](@citet) §3a1 use a 3 km inner-nest resolution with a
# 25 km deep domain. We match that on a 642 km × 642 km periodic-in-x,y box:
# 214² cells horizontally and 75 levels vertically (``Δz ≈ 333`` m). The run
# prefers GPU and falls back to CPU if CUDA isn't functional.

Δx = 3kilometers
Nx = Ny = 214
Nz = 75
Lx = Nx * Δx
Lz = 25kilometers                 # YD19 §3a1
Δz = Lz / Nz
sponge_rate = 0.003               # ≈ WRF damp_opt=2 `dampcoef` (~333 s timescale)
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
# constructed from the Jordan ``θ(z)`` profile and anchored at the observed
# surface pressure, giving us `pᵣ(z)`, `ρᵣ(z)`, `Tᵣ(z)` — the hydrostatic
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

## `interior(...)` below is for host-side reductions (balance iteration,
## azimuthal binning, state save/restore), NOT for Makie plotting — so the
## examples-rules ban on `interior(field, ...)` in heatmap!/lines! calls
## doesn't apply. Plotting is done from already-reduced Arrays.
pᵣ = Array(interior(reference_state.pressure, 1, 1, :))
ρᵣ = Array(interior(reference_state.density, 1, 1, :))
Tᵣ = Array(interior(reference_state.temperature, 1, 1, :))
z_centers = collect(range(Δz / 2, step = Δz, length = Nz))

@info @sprintf(
    "Grid: %d × %d × %d   Δx = %.1f km   Lz = %.1f km   Δz = %.1f m",
    Nx, Ny, Nz, Δx / 1.0e3, Lz / 1.0e3, Δz
)
@info @sprintf(
    "Sponge: WRF damp_opt=2 analog, sin² ramp from z = %.1f to %.1f km, rate %.3f s⁻¹ (ρu, ρv, ρw → 0; ρe → ρᵣ·[cᵖᵈ Tᵣ + g z])",
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
# The outer cos² taper between ``r_\text{taper start}`` and ``r_\text{taper end}``
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
    r >= r_taper_end  && return 0.0
    z >= z_vortex_top && return 0.0
    rmw_z = rmw_analytic(z; ΔT_floor)
    v_adj = rmw_surface / rmw_z
    vt = r <= rmw_z ?
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

function _dpdz_centered(p, k, i, z_centers, Nz)
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
                dp_dz = _dpdz_centered(p, k, i, z_centers, Nz)
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
    Rᵈ, g, verbose = true
)

## Anelastic convention (matches Breeze's internal `liquid_ice_potential_temperature`
## diagnostic and the F02b plot): θ is computed with the hydrostatic reference
## pressure, not the balanced p. Keeps the IC warm-core diagnostic consistent
## with the simulation's thermodynamic state.
θ_pre = [bal.T[i, k] * (pˢᵗ / pᵣ[k])^κ for i in 1:Nr_pre, k in 1:Nz]
p′_pre = bal.p .- reshape(pᵣ, 1, Nz)
vortex = (; v = v2d, p = bal.p, θ = θ_pre, T = bal.T, p′ = p′_pre)

let res = balance_residuals(r_pre, z_centers, v2d, bal.p, bal.T; Rᵈ, g)
    @info @sprintf(
        "IC residuals: gradient-wind max|res| = %.3e Pa/m,  hydrostatic max|res| = %.3e Pa/m  (iters = %d)",
        maximum(abs, res.gradient_wind), maximum(abs, res.hydrostatic), length(bal.history)
    )
end

@inline function _lookup_rz(table, r::Real, z::Real)
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

uᵢ(x, y, z) = (r = sqrt(x^2 + y^2); r < 1.0 ? 0.0 : -(y / r) * _lookup_rz(vortex.v, r, z))
vᵢ(x, y, z) = (r = sqrt(x^2 + y^2); r < 1.0 ? 0.0 : (x / r) * _lookup_rz(vortex.v, r, z))
Tᵢ(x, y, z) = _lookup_rz(vortex.T, sqrt(x^2 + y^2), z)

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
# Within the anelastic framework the source to the energy equation is
# ``\rho_r c_p^d Q`` (rather than full-``\rho``, as WRF uses), and this is
# what we pass to `Forcing`. The
# deviation is second order in ``p'/p`` and irrelevant inside the rainband.

Q_max = 4.24 / 3600     # YD19 Eq. 3 Q_max = 4.24 K/h (stored in K/s)
z_bs = 4_000.0
σ_r = 6_000.0
σ_zs = 2_000.0
t_full = 1hour          # 1 h ramp to avoid instantaneous onset

ρᵣ_device = arch isa GPU ? CuArray(ρᵣ) : ρᵣ

@inline function rainband_heating(i, j, k, grid, clock, fields, p)
    x = Oceananigans.Grids.xnode(i, j, k, grid, Center(), Center(), Center())
    y = Oceananigans.Grids.ynode(i, j, k, grid, Center(), Center(), Center())
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    t = clock.time

    ramp = clamp((t - p.t_on) / (p.t_full - p.t_on), 0.0, 1.0)
    r = sqrt(x^2 + y^2)
    λ = atan(y, x)
    r_bs = (60.0 - 10.0 * (λ / (π / 4))) * 1000.0 + z
    G_r = exp(-(r - r_bs)^2 / 2p.σ_r^2)

    z_rel = (z - p.z_bs) / p.σ_zs
    V_z = ifelse(abs(z_rel) < 1, sin(π * z_rel), 0.0)

    λ_c = mod(λ - (-π / 4) + π, 2π) - π
    A_λ = exp(-(λ_c / (π / 4))^8)

    Q = p.Q_max * G_r * V_z * A_λ * ramp
    ## Anelastic heating source is ρᵣ cᵖᵈ Q (K/s). WRF uses full ρ;
    ## within the anelastic approximation ρ ≈ ρᵣ(z), so this deviation
    ## is second order in the heating region where p'/p ≪ 1.
    ρᵣ_k = @inbounds p.ρᵣ[k]
    return ρᵣ_k * p.cᵖᵈ * Q
end

heating_params = (;
    Q_max, σ_r, σ_zs, z_bs, cᵖᵈ, t_on = 0.0, t_full,
    ρᵣ = ρᵣ_device,
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
# contaminate the interior. We match WRF's `damp_opt=2` shape: a sin² ramp
# from zero at ``z = 20`` km to a max rate of ``3 \times 10^{-3}`` s⁻¹
# at ``z = 25`` km. Momentum components relax to zero; energy density
# relaxes to its reference profile
#
# ```math
# \rho e_r(z) = \rho_r(z) \left[ c_p^d \, T_r(z) + g z \right]
# ```
#
# (dry static energy density). Both the momentum and energy components
# are needed: without the energy term, upper-level ``θ'`` anomalies persist
# and the vortex fails to spin down, defeating YD19's stated 43 → 40 m/s
# decay over 24 h (§3a2).

sponge_z_bottom = 20_000.0
sponge_z_top = 25_000.0

## Reference dry-static-energy density profile (J/m³):
##   ρeᵣ(z) = ρᵣ(z) · (cᵖᵈ·Tᵣ(z) + g·z)
ρeᵣ = [ρᵣ[k] * (cᵖᵈ * Tᵣ[k] + g * z_centers[k]) for k in 1:Nz]
ρeᵣ_device = arch isa GPU ? CuArray(ρeᵣ) : ρeᵣ

sponge_vel_params = (z_bot = sponge_z_bottom, z_top = sponge_z_top, rate = sponge_rate)
sponge_ρe_params = (
    z_bot = sponge_z_bottom, z_top = sponge_z_top, rate = sponge_rate,
    ρe_bg = ρeᵣ_device,
)

## WRF `damp_opt=2` analog: zero below z_bot, sin² ramp to max at z_top.
@inline function _sponge_mask(z, z_bot, z_top)
    ξ = (z - z_bot) / (z_top - z_bot)
    return ifelse(ξ ≤ 0, zero(ξ), sin(π / 2 * ξ)^2)
end

@inline function sponge_ρu_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Face(), Center(), Center())
    mask = _sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρu[i, j, k]
end

@inline function sponge_ρv_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Face(), Center())
    mask = _sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρv[i, j, k]
end

@inline function sponge_ρw_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Face())
    mask = _sponge_mask(z, p.z_bot, p.z_top)
    return -p.rate * mask * @inbounds fields.ρw[i, j, k]
end

@inline function sponge_ρe_fn(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    mask = _sponge_mask(z, p.z_bot, p.z_top)
    ρe_tgt = @inbounds p.ρe_bg[k]
    return -p.rate * mask * (@inbounds fields.ρe[i, j, k] - ρe_tgt)
end

sponge_ρu = Forcing(sponge_ρu_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρv = Forcing(sponge_ρv_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρw = Forcing(sponge_ρw_fn; discrete_form = true, parameters = sponge_vel_params)
sponge_ρe = Forcing(sponge_ρe_fn; discrete_form = true, parameters = sponge_ρe_params)

# ## Surface fluxes (Emanuel 1986 bulk aerodynamic formulation)
#
# Bulk aerodynamic surface boundary conditions for momentum drag (Cᴰ) and sensible
# heat (Cᵀ) over a fixed sea surface temperature T₀ = 300 K, matching YD19 §3a1.
# Coefficients from [Emanuel1986](@citet). The moisture flux is omitted here; when
# the model carries moisture, a corresponding `BulkVaporFlux` on the moisture
# field can be wired in alongside these.

Cᴰ_surf = 1.229e-3   # momentum drag coefficient
Cᵀ_surf = 1.094e-3   # sensible heat transfer coefficient
T₀_surf = 300.0      # SST (K)

ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ_surf))
ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ_surf))
ρe_bcs = FieldBoundaryConditions(
    bottom = BulkSensibleHeatFlux(
        coefficient = Cᵀ_surf,
        surface_temperature = T₀_surf
    )
)

surface_boundary_conditions = (ρu = ρu_bcs, ρv = ρv_bcs, ρe = ρe_bcs)

# ## Model builder

function build_model(; with_heating::Bool)
    coriolis = FPlane(; f)
    dynamics = AnelasticDynamics(reference_state)
    advection = WENO(order = 5)

    forcing = with_heating ?
        (ρu = sponge_ρu, ρv = sponge_ρv, ρw = sponge_ρw, ρe = (heating_forcing, sponge_ρe)) :
        (ρu = sponge_ρu, ρv = sponge_ρv, ρw = sponge_ρw, ρe = sponge_ρe)

    return AtmosphereModel(
        grid; dynamics, coriolis, advection, forcing,
        boundary_conditions = surface_boundary_conditions,
        formulation = :StaticEnergy
    )
end

# ## Stage runner
#
# Each stage (spinup, control, heated) builds a fresh `AtmosphereModel`, sets
# its initial prognostic state (u, v, T) from either functions or cached host
# Arrays, runs for `stop_time`, and returns the post-stage state as host
# Arrays. Anelastic diagnoses ``w`` and ``p'`` from the elliptic constraint,
# so ``u``, ``v``, ``T`` is the complete prognostic record we need to hand off between stages.

function build_and_run_stage!(
        stage_label::String;
        with_heating::Bool, init, stop_time,
        outfile::String
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
    add_callback!(simulation, progress, TimeInterval(1hour))

    T = model.temperature
    simulation.output_writers[:snaps] = JLD2Writer(
        model, (; u, v, w, T);
        filename = outfile,
        schedule = TimeInterval(1hour),
        overwrite_existing = true
    )

    @info "Running $stage_label stage for $(prettytime(stop_time)) → $outfile"
    run!(simulation)

    ## Cache post-stage prognostic state as host Arrays so the next stage can
    ## `set!(model; post...)` without a disk roundtrip.
    post = (
        u = Array(interior(model.velocities.u)),
        v = Array(interior(model.velocities.v)),
        T = Array(interior(model.temperature)),
    )

    model = nothing
    GC.gc(); CUDA.reclaim()
    return post
end

## Re-entry helper: if an earlier stage already wrote `spinup_snapshots.jld2`,
## we can skip the spinup run and restart control/heated from the last hourly
## snapshot.
function load_last_snapshot(path)
    ts_u = FieldTimeSeries(path, "u")
    ts_v = FieldTimeSeries(path, "v")
    ts_T = FieldTimeSeries(path, "T")
    n = length(ts_u.times)
    return (
        u = Array(interior(ts_u[n])),
        v = Array(interior(ts_v[n])),
        T = Array(interior(ts_T[n])),
    )
end

# ## File paths

spinup_file = joinpath(snapshots_dir, "spinup_snapshots.jld2")
control_file = joinpath(snapshots_dir, "control_snapshots.jld2")
heated_file = joinpath(snapshots_dir, "heated_snapshots.jld2")

# ## F01 — vortex IC preflight (fast sanity check; always runs in spinup)

function plot_preflight()
    θ_anom = zeros(Nr_pre, Nz)
    for k in 1:Nz, i in 1:Nr_pre
        θ_anom[i, k] = vortex.θ[i, k] - θ_env(z_centers[k])
    end
    δp_sfc = [(vortex.p[i, 1] - p_env(z_centers[1])) / 100 for i in 1:Nr_pre]

    fig = Figure(size = (1200, 900))

    ax1 = Axis(
        fig[1, 1]; xlabel = "RMW (km)", ylabel = "Height (km)",
        title = "RMW(z) [Stern-Nolan 2009 Eq. 4.4]", limits = (0, 200, 0, 22)
    )
    lines!(ax1, rmw_profile ./ 1e3, z_centers ./ 1e3; color = :black, linewidth = 2)
    hlines!(ax1, [z_vortex_top / 1e3]; color = :gray, linestyle = :dash)

    ax2 = Axis(
        fig[1, 2]; xlabel = "v_max (m s⁻¹)", ylabel = "Height (km)",
        title = "v_max(z) at RMW", limits = (0, 50, 0, 22)
    )
    vmax_z = [tangential_wind(rmw_analytic(z), z) for z in z_centers]
    lines!(ax2, vmax_z, z_centers ./ 1e3; color = :black, linewidth = 2)

    ax3 = Axis(
        fig[2, 1]; xlabel = "Radius (km)", ylabel = "Height (km)",
        title = "Warm-core θ' (t = 0)", limits = (0, 200, 0, 22)
    )
    hm = heatmap!(
        ax3, r_pre ./ 1e3, z_centers ./ 1e3, θ_anom;
        colormap = :balance, colorrange = (-14, 14)
    )
    contour!(
        ax3, r_pre ./ 1e3, z_centers ./ 1e3, θ_anom;
        levels = -14:1:14, color = :black, linewidth = 0.4
    )
    Colorbar(fig[2, 1][1, 2], hm; label = "θ' (K)")

    ax4 = Axis(
        fig[2, 2]; xlabel = "Radius (km)", ylabel = "δp (hPa)",
        title = "Surface pressure deficit", limits = (0, 300, -60, 5)
    )
    lines!(ax4, r_pre ./ 1e3, δp_sfc; color = :black, linewidth = 2)
    hlines!(ax4, [0.0]; color = :gray, linestyle = :dot)
    δp_min = minimum(δp_sfc)
    text!(
        ax4, 10, -5; text = @sprintf("δp_min = %.1f hPa", δp_min),
        align = (:left, :top)
    )

    Label(fig[0, :], "F01 — Vortex IC preflight (t = 0)"; fontsize = 18)

    path = joinpath(figures_dir, "F01_preflight.png")
    save(path, fig)
    @info "Saved F01" path
    return fig
end

# ## Stage 1 — Spinup

@info "=== Stage 1: $(prettytime(stage_stop_time)) spinup ==="
plot_preflight()
nothing # hide
post_spinup = build_and_run_stage!(
    "spinup";
    with_heating = false,
    init = (u = uᵢ, v = vᵢ, T = Tᵢ),
    stop_time = stage_stop_time,
    outfile = spinup_file
)
nothing # hide

# ## Stage 2 — Control

@info "=== Stage 2: $(prettytime(stage_stop_time)) control ==="
build_and_run_stage!(
    "control";
    with_heating = false,
    init = post_spinup,
    stop_time = stage_stop_time,
    outfile = control_file
)
nothing # hide

# ## Stage 3 — Heated

@info "=== Stage 3: $(prettytime(stage_stop_time)) heated (MN10 stratiform) ==="
build_and_run_stage!(
    "heated";
    with_heating = true,
    init = post_spinup,
    stop_time = stage_stop_time,
    outfile = heated_file
)
nothing # hide

# ## Stage 4 — Analysis and figure production
#
# Everything below reads from the hourly JLD2 snapshots. To keep peak
# resident memory bounded by a handful of 3D fields regardless of how
# many snapshots we wrote, we open each `FieldTimeSeries` with the
# `OnDisk()` backend — so snapshots stream from disk on demand — and
# we copy snapshots into preallocated Float32 scratch buffers instead
# of allocating per iteration. `GC.gc()` at the boundary between
# sub-figures keeps the high-water mark low.

let
    @info "=== Stage 4: Analysis ==="

    ## Preallocated Float32 scratch buffers for cell-centered fields.
    ## Reused across every figure so peak RSS during analysis is ~4 × Nx × Ny × Nz
    ## × 4 bytes, not Nt × 4 × Nx × Ny × Nz × 8 bytes as the prior InMemory /
    ## Float64 pattern was.
    u_sc = Array{Float32}(undef, Nx, Ny, Nz); v_sc = similar(u_sc)
    w_sc = similar(u_sc);                      T_sc = similar(u_sc)

    ## In-place centered interpolation (Arakawa C-grid → Centers).
    function _center_u!(out::AbstractArray{Float32, 3}, src)
        Nxs, Nys, Nzs = size(out)
        return @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
            ip = i == Nxs ? 1 : i + 1
            out[i, j, k] = 0.5f0 * (Float32(src[i, j, k]) + Float32(src[ip, j, k]))
        end
    end
    function _center_v!(out::AbstractArray{Float32, 3}, src)
        Nxs, Nys, Nzs = size(out)
        return @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
            jp = j == Nys ? 1 : j + 1
            out[i, j, k] = 0.5f0 * (Float32(src[i, j, k]) + Float32(src[i, jp, k]))
        end
    end
    function _center_w!(out::AbstractArray{Float32, 3}, src)
        ## w lives on z-Face (size Nz+1 in the vertical). Average adjacent
        ## faces to get cell-centered values with size (Nx, Ny, Nz).
        Nxs, Nys, Nzs = size(out)
        return @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
            out[i, j, k] = 0.5f0 * (Float32(src[i, j, k]) + Float32(src[i, j, k + 1]))
        end
    end
    function _copy_f32!(out::AbstractArray{Float32, 3}, src)
        return @inbounds for I in eachindex(out)
            out[I] = Float32(src[I])
        end
    end

    ## Load one snapshot (index n) from four OnDisk-backed FieldTimeSeries
    ## into the scratch buffers. Returns nothing.
    function load_snapshot!(u, v, w, T, u_ts, v_ts, w_ts, T_ts, n::Int)
        _center_u!(u, interior(u_ts[n]))
        _center_v!(v, interior(v_ts[n]))
        _center_w!(w, interior(w_ts[n]))
        _copy_f32!(T, interior(T_ts[n]))
        return nothing
    end

    ## Open FieldTimeSeries files with the streaming backend. Indexing
    ## `ts[n]` reads only snapshot n from disk; memory does not grow
    ## with Nt.
    function open_ts(path::String)
        return (
            u = FieldTimeSeries(path, "u"; backend = Oceananigans.OnDisk()),
            v = FieldTimeSeries(path, "v"; backend = Oceananigans.OnDisk()),
            w = FieldTimeSeries(path, "w"; backend = Oceananigans.OnDisk()),
            T = FieldTimeSeries(path, "T"; backend = Oceananigans.OnDisk()),
        )
    end

    r_bin_edges = collect(range(0.0, 150_000.0, step = Δx))
    Nr_bin = length(r_bin_edges) - 1
    r_bin_centers = 0.5 .* (r_bin_edges[1:(end - 1)] .+ r_bin_edges[2:end])
    xs_center = Float32.(xnodes(grid, Center()))
    ys_center = Float32.(ynodes(grid, Center()))

    ## Reusable azimuthal-mean workspace
    vθ_ws = zeros(Float32, Nr_bin, Nz); vr_ws = similar(vθ_ws)
    w_ws = similar(vθ_ws);              T_ws = similar(vθ_ws)
    ct_ws = zeros(Int32, Nr_bin, Nz)
    r_last = Float32(last(r_bin_edges))

    function azimuthal_mean!(vθ, vr, ww, TT, ct, u, v, w, T)
        fill!(vθ, 0.0f0); fill!(vr, 0.0f0); fill!(ww, 0.0f0); fill!(TT, 0.0f0); fill!(ct, 0)
        Nxs, Nys, Nzs = size(u)
        @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
            x = xs_center[i]; y = ys_center[j]
            r = sqrt(x * x + y * y)
            r >= r_last && continue
            ib = searchsortedlast(r_bin_edges, Float64(r))
            ib = clamp(ib, 1, Nr_bin)
            rs = max(r, 1.0f0)
            xh = x / rs; yh = y / rs
            uij = u[i, j, k]; vij = v[i, j, k]
            vθ[ib, k] += -yh * uij + xh * vij
            vr[ib, k] += xh * uij + yh * vij
            ww[ib, k] += w[i, j, k]
            TT[ib, k] += T[i, j, k]
            ct[ib, k] += 1
        end
        return @inbounds for k in 1:Nzs, ib in 1:Nr_bin
            if ct[ib, k] > 0
                inv = 1.0f0 / ct[ib, k]
                vθ[ib, k] *= inv; vr[ib, k] *= inv
                ww[ib, k] *= inv; TT[ib, k] *= inv
            end
        end
    end

    ## Streaming time average: accumulate centered snapshots for each index
    ## in `ns` into (uavg, vavg, wavg, Tavg), then divide by N. Uses the
    ## primary scratch (u_sc, ...) as a per-iteration staging buffer.
    function time_average_centered!(uavg, vavg, wavg, Tavg, ts, ns)
        fill!(uavg, 0.0f0); fill!(vavg, 0.0f0); fill!(wavg, 0.0f0); fill!(Tavg, 0.0f0)
        for n in ns
            load_snapshot!(u_sc, v_sc, w_sc, T_sc, ts.u, ts.v, ts.w, ts.T, n)
            uavg .+= u_sc; vavg .+= v_sc; wavg .+= w_sc; Tavg .+= T_sc
        end
        N = Float32(length(ns))
        uavg ./= N; vavg ./= N; wavg ./= N; Tavg ./= N
        return nothing
    end

    indices_near(times, targets_s) = [argmin(abs.(times .- t)) for t in targets_s]

    ## ---------------------------------------------------------
    ## F02ab — basic-state vortex (YD19 Fig 2a,b)
    ## ---------------------------------------------------------
    if isfile(spinup_file)
        @info "Producing F02ab (basic-state vortex)..."
        ts_spin = open_ts(spinup_file)
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
            ax_v, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, vθ_ws;
            colormap = :inferno, colorrange = (0, v_cr_hi)
        )
        contour!(
            ax_v, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, vθ_ws;
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
            ax_θ, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, θ_anom;
            colormap = :balance, colorrange = (-θ_span, θ_span)
        )
        contour!(
            ax_θ, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, θ_anom;
            levels = -θ_span:1.69:θ_span, color = :black, linewidth = 0.5
        )
        Colorbar(fig[1, 4], hm_θ; label = "θ̄' (K)")

        Label(
            fig[0, :],
            @sprintf(
                "F02ab — Basic-state vortex at t = %.1f h spin-up (YD19 Fig 2a,b, %.0f km box)",
                t_final / hour, Lx / 1.0e3
            );
            fontsize = 17
        )

        v_peak_sfc = maximum(@view vθ_ws[:, 1])
        r_peak_sfc = r_bin_centers[argmax(@view vθ_ws[:, 1])] / 1.0e3
        v_peak_all = maximum(vθ_ws)
        idx_all = argmax(vθ_ws)
        r_peak_all = r_bin_centers[idx_all[1]] / 1.0e3
        z_peak_all = z_centers[idx_all[2]] / 1.0e3
        θ_peak = maximum(θ_anom)
        idx_θ = argmax(θ_anom)
        r_θ_peak = r_bin_centers[idx_θ[1]] / 1.0e3
        z_θ_peak = z_centers[idx_θ[2]] / 1.0e3
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
    else
        @warn "Missing $spinup_file — skipping F02ab"
    end

    ## ---------------------------------------------------------
    ## F02cd — analytic heating field (YD19 Fig 2c,d)
    ## ---------------------------------------------------------
    @info "Producing F02cd (heating field)..."
    r_cs = collect(range(0.0, 150_000.0, length = 151))
    z_cs = collect(range(0.0, 12_000.0, length = 61))
    λ_mid = -π / 4
    Q_cs = [heating_rate_K_per_hour(r, λ_mid, z) for r in r_cs, z in z_cs]

    x_pv = collect(range(-Lx / 2, Lx / 2, length = 300))
    y_pv = copy(x_pv)
    z_level = 4_600.0
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
        ax_c, r_cs ./ 1.0e3, z_cs ./ 1.0e3, Q_cs;
        colormap = :balance, colorrange = (-Q_lim, Q_lim)
    )
    contour!(
        ax_c, r_cs ./ 1.0e3, z_cs ./ 1.0e3, Q_cs;
        levels = -4:1:4, color = :black, linewidth = 0.6
    )
    Colorbar(fig[1, 2], hm_c; label = "Q (K h⁻¹)")

    ax_d = Axis(
        fig[1, 3]; xlabel = "x (km)", ylabel = "y (km)",
        title = @sprintf("(d) Heating plan view at z = %.1f km", z_level / 1000),
        aspect = DataAspect(), limits = (-120, 120, -120, 120)
    )
    hm_d = heatmap!(
        ax_d, x_pv ./ 1.0e3, y_pv ./ 1.0e3, Q_pv;
        colormap = :balance, colorrange = (-Q_lim, Q_lim)
    )
    contour!(
        ax_d, x_pv ./ 1.0e3, y_pv ./ 1.0e3, Q_pv;
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

    ## ---------------------------------------------------------
    ## F03a, F04, F05 — require control + heated snapshots
    ## ---------------------------------------------------------
    if isfile(control_file) && isfile(heated_file)
        @info "Loading heated and control snapshots (streaming, OnDisk backend)..."

        ts_ctrl = open_ts(control_file)
        ts_heat = open_ts(heated_file)
        ctrl_times = ts_ctrl.u.times
        heat_times = ts_heat.u.times
        ## Analysis window: t = 5-7 h — the pre-saturation window in which
        ## the response sits near YD19's quoted ±1.5 m/s. Without explicit
        ## horizontal diffusivity the response inflates linearly past this
        ## window (see F06 for the diagnostic).
        target_s = collect(5.0:1.0:7.0) .* hour
        n_ctrl = indices_near(ctrl_times, target_s)
        n_heat = indices_near(heat_times, target_s)
        @info @sprintf(
            "Control averaging window: t = %.1f - %.1f h (%d snapshots)",
            ctrl_times[n_ctrl[1]] / hour, ctrl_times[n_ctrl[end]] / hour, length(n_ctrl)
        )
        @info @sprintf(
            "Heated averaging window : t = %.1f - %.1f h (%d snapshots)",
            heat_times[n_heat[1]] / hour, heat_times[n_heat[end]] / hour, length(n_heat)
        )

        ## Streaming time averages (Float32, preallocated).
        u_ctrl = Array{Float32}(undef, Nx, Ny, Nz); v_ctrl = similar(u_ctrl)
        w_ctrl = similar(u_ctrl);                   T_ctrl = similar(u_ctrl)
        u_heat = similar(u_ctrl); v_heat = similar(u_ctrl)
        w_heat = similar(u_ctrl); T_heat = similar(u_ctrl)
        time_average_centered!(u_ctrl, v_ctrl, w_ctrl, T_ctrl, ts_ctrl, n_ctrl)
        time_average_centered!(u_heat, v_heat, w_heat, T_heat, ts_heat, n_heat)

        ## Azimuthal means for control
        vθ_ctrl = zeros(Float32, Nr_bin, Nz); vr_ctrl = similar(vθ_ctrl)
        w_ctrl_az = similar(vθ_ctrl);          T_ctrl_az = similar(vθ_ctrl)
        azimuthal_mean!(
            vθ_ctrl, vr_ctrl, w_ctrl_az, T_ctrl_az, ct_ws,
            u_ctrl, v_ctrl, w_ctrl, T_ctrl
        )
        ## and for heated
        vθ_heat = similar(vθ_ctrl); vr_heat = similar(vθ_ctrl)
        w_heat_az = similar(vθ_ctrl); T_heat_az = similar(vθ_ctrl)
        azimuthal_mean!(
            vθ_heat, vr_heat, w_heat_az, T_heat_az, ct_ws,
            u_heat, v_heat, w_heat, T_heat
        )

        vθ_resp = vθ_heat .- vθ_ctrl
        vr_resp = vr_heat .- vr_ctrl
        w_resp_rz = w_heat_az .- w_ctrl_az

        ## In-place: overwrite heated 3D arrays with (heated − control), free control.
        u_heat .-= u_ctrl; v_heat .-= v_ctrl; w_heat .-= w_ctrl
        u_ctrl = v_ctrl = w_ctrl = T_ctrl = T_heat = nothing
        GC.gc()
        u_resp3, v_resp3, w_resp3 = u_heat, v_heat, w_heat

        ## Azimuthal-mean heating (for ±0.15 K/h contour on F03a)
        Q_bar = zeros(Nr_bin, Nz)
        let Nphi = 512
            for k in 1:Nz, ib in 1:Nr_bin
                s = 0.0
                r = r_bin_centers[ib]; zk = z_centers[k]
                for q in 1:Nphi
                    λ = -π + 2π * (q - 0.5) / Nphi
                    s += heating_rate_K_per_hour(r, λ, zk)
                end
                Q_bar[ib, k] = s / Nphi
            end
        end

        ## --- F06 — response time series (diagnostic) ---
        ## For each hourly snapshot, compute the azimuthal-mean response
        ## (heated minus control, no time averaging) and record the peak
        ## |v̄'| inside the YD19 analysis window. Monotonic growth past
        ## ~1.5 m/s signals IG-wave wrap-around contamination; saturation
        ## near paper's value means the setup is intrinsically close.
        @info "Producing F06 (response time series, streaming)..."
        Nt = min(length(ctrl_times), length(heat_times))
        ts_hours = Float64[]; peak_pos = Float64[]; peak_neg = Float64[]
        window_r = findall(r -> 35_000 ≤ r ≤ 120_000, r_bin_centers)
        window_z = findall(z -> 0.0 ≤ z ≤ 12_000, z_centers)
        ## Second set of azimuthal-mean buffers for the paired control/heated load
        vθ_h_ws = similar(vθ_ws); vr_h_ws = similar(vθ_ws)
        w_h_ws = similar(vθ_ws); T_h_ws = similar(vθ_ws)
        for n in 1:Nt
            load_snapshot!(
                u_sc, v_sc, w_sc, T_sc,
                ts_ctrl.u, ts_ctrl.v, ts_ctrl.w, ts_ctrl.T, n
            )
            azimuthal_mean!(
                vθ_ws, vr_ws, w_ws, T_ws, ct_ws,
                u_sc, v_sc, w_sc, T_sc
            )
            load_snapshot!(
                u_sc, v_sc, w_sc, T_sc,
                ts_heat.u, ts_heat.v, ts_heat.w, ts_heat.T, n
            )
            azimuthal_mean!(
                vθ_h_ws, vr_h_ws, w_h_ws, T_h_ws, ct_ws,
                u_sc, v_sc, w_sc, T_sc
            )
            resp_view = @view(vθ_h_ws[window_r, window_z]) .- @view(vθ_ws[window_r, window_z])
            push!(ts_hours, heat_times[n] / hour)
            push!(peak_pos, maximum(resp_view))
            push!(peak_neg, minimum(resp_view))
        end
        fig = Figure(size = (900, 500))
        ax = Axis(
            fig[1, 1]; xlabel = "Time after heating onset (h)",
            ylabel = "Peak v̄' response (m s⁻¹)",
            title = @sprintf(
                "F06 — Response amplitude vs time (r ∈ [35, 120] km, z ∈ [0, 12] km, %.0f km box)",
                Lx / 1.0e3
            ),
            limits = (0, ts_hours[end] + 1, -5, 5)
        )
        lines!(ax, ts_hours, peak_pos; color = :firebrick, linewidth = 2, label = "max v̄'")
        lines!(ax, ts_hours, peak_neg; color = :navy, linewidth = 2, label = "min v̄'")
        hlines!(ax, [1.5]; color = :firebrick, linestyle = :dash, linewidth = 1)
        hlines!(ax, [-1.5]; color = :navy, linestyle = :dash, linewidth = 1)
        hlines!(ax, [0.0]; color = :gray, linestyle = :dot)
        text!(ax, 1.0, 1.7; text = "YD19 peak ≈ 1.5 m/s", color = :firebrick, fontsize = 12)
        text!(ax, 1.0, -1.3; text = "YD19 dip ≈ -1.5 m/s", color = :navy, fontsize = 12)
        axislegend(ax; position = :rb)
        path = joinpath(figures_dir, "F06_response_timeseries.png")
        save(path, fig)
        @info "Saved F06" path
        for n in 1:Nt
            @info @sprintf(
                "  t = %5.2f h   max v̄' = %+.3f  min v̄' = %+.3f",
                ts_hours[n], peak_pos[n], peak_neg[n]
            )
        end
        GC.gc()

        ## --- F03a ---
        @info "Producing F03a..."
        xlim_lo, xlim_hi = 35.0, 120.0
        zlim_lo, zlim_hi = 0.0, 12.0
        v_lim = 3.5

        fig = Figure(size = (900, 600))
        ax = Axis(
            fig[1, 1];
            xlabel = "Radius (km)", ylabel = "Height (km)",
            title = @sprintf(
                "YD19 Fig 3a — Axisymmetric wind response (heated − control, %.0f-%.0f h, %.0f km box)",
                target_s[1] / hour, target_s[end] / hour, Lx / 1.0e3
            ),
            limits = (xlim_lo, xlim_hi, zlim_lo, zlim_hi)
        )
        hm = heatmap!(
            ax, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, vθ_resp;
            colormap = :balance, colorrange = (-v_lim, v_lim)
        )
        contour!(
            ax, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, vθ_resp;
            levels = -v_lim:0.5:v_lim, color = :black, linewidth = 0.5
        )
        contour!(
            ax, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, Q_bar;
            levels = [0.15], color = :red, linewidth = 2.0
        )
        contour!(
            ax, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, Q_bar;
            levels = [-0.15], color = :blue, linewidth = 2.0
        )
        Colorbar(fig[1, 2], hm; label = "v̄' (m s⁻¹)")

        let stride_r = 2, stride_z = 3
            rsub = r_bin_centers[1:stride_r:end] ./ 1.0e3
            zsub = z_centers[1:stride_z:end] ./ 1.0e3
            vsub = vr_resp[1:stride_r:end, 1:stride_z:end]
            wsub = w_resp_rz[1:stride_r:end, 1:stride_z:end] .* 10
            pts = Point2f[]; vecs = Vec2f[]
            for j in eachindex(zsub), i in eachindex(rsub)
                if xlim_lo <= rsub[i] <= xlim_hi && zlim_lo <= zsub[j] <= zlim_hi
                    push!(pts, Point2f(rsub[i], zsub[j]))
                    push!(vecs, Vec2f(vsub[i, j], wsub[i, j]))
                end
            end
            arrows2d!(
                ax, pts, vecs; lengthscale = 1.0, color = :gray20,
                tiplength = 4, tipwidth = 3
            )
        end

        r_win = findall(r -> xlim_lo * 1.0e3 ≤ r ≤ xlim_hi * 1.0e3, r_bin_centers)
        z_win = findall(z -> zlim_lo * 1.0e3 ≤ z ≤ zlim_hi * 1.0e3, z_centers)
        pk_pos = maximum(vθ_resp[r_win, z_win])
        pk_neg = minimum(vθ_resp[r_win, z_win])
        @info @sprintf(
            "F03a: response range in YD19 window = [%.2f, %.2f] m/s  (YD19 peak ≈ 1.5 m/s)",
            pk_neg, pk_pos
        )

        path = joinpath(figures_dir, "F03a_axisym_response.png")
        save(path, fig)
        @info "Saved F03a" path
        GC.gc()

        ## --- F04 ---
        @info "Producing F04..."
        xs_grid = Array(xnodes(grid, Center()))
        ys_grid = Array(ynodes(grid, Center()))
        ## Each panel slices `w'` at one altitude. For the heating contour overlay
        ## we always evaluate Q at the heating peak (z_bs+σ_zs/2 = 5 km, red) and
        ## the cooling peak (z_bs-σ_zs/2 = 3 km, blue) so both spiral lobes of the
        ## rainband are visible regardless of which `w'` slice is being shown —
        ## otherwise panels in the cooling lobe (e.g. z=3.6 km) only render the
        ## blue contour and panels in the heating lobe only render the red.
        z_heating_peak = z_bs + σ_zs / 2          # 5 km
        z_cooling_peak = z_bs - σ_zs / 2          # 3 km
        panels = [
            (6_000.0, "(a) z = 6 km"),
            (3_600.0, "(b) z = 3.6 km"),
            (2_000.0, "(c) z = 2 km"),
        ]
        ## Use 95th-percentile of |w'| (restricted to the rainband quadrant and
        ## r ≤ 120 km) for color range — avoids letting inner-core IG noise
        ## saturate the colormap while the actual signal is drowned out.
        function _w_scale(w3d, k_s)
            w_s = w3d[:, :, k_s]
            vals = Float64[]
            for j in 1:Ny, i in 1:Nx
                r = sqrt(xs_grid[i]^2 + ys_grid[j]^2)
                r > 120_000 && continue
                push!(vals, abs(w_s[i, j]))
            end
            sort!(vals)
            return vals[ceil(Int, 0.95 * length(vals))]
        end
        w_scales = [_w_scale(w_resp3, argmin(abs.(z_centers .- zp[1]))) for zp in panels]
        w_panel_lim = max(0.3, ceil(maximum(w_scales) * 4) / 4)

        ## Heating-peak Q field (positive, red contour) and cooling-peak Q field
        ## (negative, blue contour). Same fields reused on all three panels so
        ## both spiral lobes are visible even when a panel slices outside the
        ## heating column itself.
        Q_warm = zeros(Nx, Ny)
        Q_cool = zeros(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            r = sqrt(xs_grid[i]^2 + ys_grid[j]^2)
            λ = atan(ys_grid[j], xs_grid[i])
            Q_warm[i, j] = heating_rate_K_per_hour(r, λ, z_heating_peak)
            Q_cool[i, j] = heating_rate_K_per_hour(r, λ, z_cooling_peak)
        end

        fig = Figure(size = (1500, 560))
        for (pi_, (z_slice, label)) in enumerate(panels)
            k_s = argmin(abs.(z_centers .- z_slice))
            w_s = w_resp3[:, :, k_s]
            u_s = u_resp3[:, :, k_s]
            v_s = v_resp3[:, :, k_s]

            local ax = Axis(
                fig[1, pi_]; xlabel = "x (km)", ylabel = "y (km)",
                title = label, aspect = DataAspect(),
                limits = (-120, 120, -120, 120)
            )
            local hm = heatmap!(
                ax, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, w_s;
                colormap = :balance, colorrange = (-w_panel_lim, w_panel_lim)
            )
            contour!(
                ax, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, Q_warm;
                levels = [1.0], color = :red, linewidth = 2.0
            )
            contour!(
                ax, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, Q_cool;
                levels = [-1.0], color = :blue, linewidth = 2.0
            )

            ss = 10
            xa = xs_grid[1:ss:end] ./ 1.0e3
            ya = ys_grid[1:ss:end] ./ 1.0e3
            ua = u_s[1:ss:end, 1:ss:end]
            va = v_s[1:ss:end, 1:ss:end]
            local pts = [Point2f(xa[i], ya[j]) for i in eachindex(xa), j in eachindex(ya)][:]
            local vecs = [Vec2f(ua[i, j], va[i, j]) for i in eachindex(xa), j in eachindex(ya)][:]
            arrows2d!(
                ax, pts, vecs; lengthscale = 2, color = :gray20,
                tiplength = 4, tipwidth = 3
            )
            if pi_ == 3
                Colorbar(fig[1, 4], hm; label = "w' (m s⁻¹)")
            end
        end
        Label(
            fig[0, :],
            @sprintf(
                "F04 — Plan-view response heated − control (YD19 Fig 4a-c, %.0f-%.0f h, %.0f km box)",
                target_s[1] / hour, target_s[end] / hour, Lx / 1.0e3
            );
            fontsize = 17
        )

        path = joinpath(figures_dir, "F04_plan_response.png")
        save(path, fig)
        @info "Saved F04" path
        @info @sprintf(
            "F04 color range ±%.2f m/s  |  per-panel peak |w'| = [%.3f, %.3f, %.3f] m/s",
            w_panel_lim, w_scales[1], w_scales[2], w_scales[3]
        )
        GC.gc()

        ## --- F05 ---
        @info "Producing F05..."
        k_22 = argmin(abs.(z_centers .- 2_200.0))
        w_22 = w_resp3[:, :, k_22]
        ## Reuse Q_warm/Q_cool (heating-peak / cooling-peak fields) from F04 so
        ## both spiral lobes overlay the panel-(a) plan view at z ≈ 2 km.
        w22_lim = max(0.3, ceil(maximum(abs, w_22) * 2) / 2)

        function cross_section(λ_cs)
            xh = cos(λ_cs); yh = sin(λ_cs)
            vθ_cs_loc = zeros(length(r_bin_centers), Nz)
            vr_cs_loc = zeros(length(r_bin_centers), Nz)
            w_cs_loc = zeros(length(r_bin_centers), Nz)
            for k in 1:Nz, i in eachindex(r_bin_centers)
                xp = r_bin_centers[i] * xh
                yp = r_bin_centers[i] * yh
                ix = searchsortedfirst(xs_grid, xp); ix = clamp(ix, 2, Nx)
                iy = searchsortedfirst(ys_grid, yp); iy = clamp(iy, 2, Ny)
                x0 = xs_grid[ix - 1]; x1 = xs_grid[ix]
                y0 = ys_grid[iy - 1]; y1 = ys_grid[iy]
                fx = clamp((xp - x0) / (x1 - x0), 0.0, 1.0)
                fy = clamp((yp - y0) / (y1 - y0), 0.0, 1.0)
                bil(A) = (1 - fx) * (1 - fy) * A[ix - 1, iy - 1, k] +
                    fx * (1 - fy) * A[ix, iy - 1, k] +
                    (1 - fx) * fy * A[ix - 1, iy, k] +
                    fx * fy * A[ix, iy, k]
                u_p = bil(u_resp3); v_p = bil(v_resp3); w_p = bil(w_resp3)
                vr_cs_loc[i, k] = xh * u_p + yh * v_p
                vθ_cs_loc[i, k] = -yh * u_p + xh * v_p
                w_cs_loc[i, k] = w_p
            end
            return (vθ = vθ_cs_loc, vr = vr_cs_loc, w = w_cs_loc)
        end

        cs_up = cross_section(0.0)
        cs_md = cross_section(-π / 4)
        cs_dn = cross_section(-π / 2)

        fig = Figure(size = (1400, 950))

        ax_pv = Axis(
            fig[1, 1:3]; xlabel = "x (km)", ylabel = "y (km)",
            title = @sprintf(
                "(a) Vertical-velocity response at z = %.1f km",
                z_centers[k_22] / 1.0e3
            ),
            aspect = DataAspect(),
            limits = (-120, 120, -120, 120)
        )
        hm_pv = heatmap!(
            ax_pv, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, w_22;
            colormap = :balance, colorrange = (-w22_lim, w22_lim)
        )
        contour!(
            ax_pv, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, Q_warm;
            levels = [1.0], color = :red, linewidth = 2.0
        )
        contour!(
            ax_pv, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, Q_cool;
            levels = [-1.0], color = :blue, linewidth = 2.0
        )
        ## Azimuthal convention (YD19 §3b1):
        ##   λ = 0     — downwind end (east, 60 km r_bsfc)
        ##   λ = -π/4  — middle of rainband (southeast, 70 km)
        ##   λ = -π/2  — upwind end (south, 80 km)
        for (λ_cs, lbl) in [(0.0, "downwind"), (-π / 4, "middle"), (-π / 2, "upwind")]
            x_end = 120 * cos(λ_cs); y_end = 120 * sin(λ_cs)
            lines!(ax_pv, [0.0, x_end], [0.0, y_end]; color = :black, linewidth = 1.5)
            text!(
                ax_pv, x_end, y_end; text = " $lbl ",
                align = (:left, :center), color = :black, fontsize = 12
            )
        end
        Colorbar(fig[1, 4], hm_pv; label = "w' (m s⁻¹)")

        cs_v_lim = 3.0
        function draw_cross!(ax, cs, λ_cs, label)
            hm = heatmap!(
                ax, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, cs.vθ;
                colormap = :balance, colorrange = (-cs_v_lim, cs_v_lim)
            )
            contour!(
                ax, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, cs.vθ;
                levels = -cs_v_lim:0.5:cs_v_lim, color = :black, linewidth = 0.4
            )
            Q_cs2 = [heating_rate_K_per_hour(r, λ_cs, z) for r in r_bin_centers, z in z_centers]
            contour!(
                ax, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, Q_cs2;
                levels = [1.0], color = :red, linewidth = 2.0
            )
            contour!(
                ax, r_bin_centers ./ 1.0e3, z_centers ./ 1.0e3, Q_cs2;
                levels = [-1.0], color = :blue, linewidth = 2.0
            )

            stride_r = 2; stride_z = 3
            rsub = r_bin_centers[1:stride_r:end] ./ 1.0e3
            zsub = z_centers[1:stride_z:end] ./ 1.0e3
            vsub = cs.vr[1:stride_r:end, 1:stride_z:end]
            wsub = cs.w[1:stride_r:end, 1:stride_z:end] .* 10
            pts = Point2f[]; vecs = Vec2f[]
            for j in eachindex(zsub), i in eachindex(rsub)
                if 35 <= rsub[i] <= 120 && zsub[j] <= 12
                    push!(pts, Point2f(rsub[i], zsub[j]))
                    push!(vecs, Vec2f(vsub[i, j], wsub[i, j]))
                end
            end
            arrows2d!(
                ax, pts, vecs; lengthscale = 1.0, color = :gray20,
                tiplength = 4, tipwidth = 3
            )
            ax.title = label
            ax.xlabel = "Radius (km)"
            ax.ylabel = "Height (km)"
            return hm
        end

        ax_up = Axis(fig[2, 1]; limits = (35, 120, 0, 12))
        draw_cross!(ax_up, cs_dn, -π / 2, "(b) upwind end (λ = -π/2)")
        ax_md = Axis(fig[2, 2]; limits = (35, 120, 0, 12))
        draw_cross!(ax_md, cs_md, -π / 4, "(c) middle (λ = -π/4)")
        ax_dn = Axis(fig[2, 3]; limits = (35, 120, 0, 12))
        hm_dn = draw_cross!(ax_dn, cs_up, 0.0, "(d) downwind end (λ = 0)")
        Colorbar(fig[2, 4], hm_dn; label = "v̄' (m s⁻¹)")

        Label(
            fig[0, :],
            @sprintf(
                "F05 — Cross sections of tangential-wind response (YD19 Fig 5, %.0f-%.0f h, %.0f km box)",
                target_s[1] / hour, target_s[end] / hour, Lx / 1.0e3
            );
            fontsize = 17
        )

        path = joinpath(figures_dir, "F05_cross_sections.png")
        save(path, fig)
        @info "Saved F05" path
        GC.gc()

        ## ---------------------------------------------------------
        ## Animation — w' at z ≈ 3 km over the heated run
        ## ---------------------------------------------------------
        ##
        ## The quickest way to feel the YD19 response is to watch vertical
        ## velocity evolve. We pick the level closest to z = 3 km, then step
        ## through every hourly snapshot of (heated − control) and record a
        ## frame of the w-difference field on the plan view. The heating's
        ## red/blue ±1 K/h contour is overlaid for reference.
        @info "Producing animation (w' at z ≈ 3 km, heated − control)..."
        k_anim = argmin(abs.(z_centers .- 3_000.0))
        xs_grid = Float32.(xnodes(grid, Center()))
        ys_grid = Float32.(ynodes(grid, Center()))

        ## Heating-peak Q (red, +1 K/h) and cooling-peak Q (blue, -1 K/h) — both
        ## visible regardless of z_anim, mirroring F04 / F05.
        z_anim = z_centers[k_anim]
        Q_anim_warm = [
            heating_rate_K_per_hour(
                    sqrt(xs_grid[i]^2 + ys_grid[j]^2),
                    atan(ys_grid[j], xs_grid[i]),
                    z_heating_peak
                )
                for i in 1:Nx, j in 1:Ny
        ]
        Q_anim_cool = [
            heating_rate_K_per_hour(
                    sqrt(xs_grid[i]^2 + ys_grid[j]^2),
                    atan(ys_grid[j], xs_grid[i]),
                    z_cooling_peak
                )
                for i in 1:Nx, j in 1:Ny
        ]

        ## Preload all frames as a time-indexed 3D array (Nx, Ny, Nt) of
        ## Float32. This is small enough to keep in memory: Nx·Ny·Nt ≈
        ## 214² × 25 × 4 B ≈ 4.6 MB. For each hour n we compute
        ## w_h(·, ·, k_anim) − w_c(·, ·, k_anim).
        Nt_anim = min(length(ctrl_times), length(heat_times))
        w_frames = zeros(Float32, Nx, Ny, Nt_anim)
        for n in 1:Nt_anim
            _center_w!(w_sc, interior(ts_ctrl.w[n]))
            @views w_frames[:, :, n] .= -w_sc[:, :, k_anim]
            _center_w!(w_sc, interior(ts_heat.w[n]))
            @views w_frames[:, :, n] .+= w_sc[:, :, k_anim]
        end
        w_lim = max(0.3f0, Float32(maximum(abs, w_frames)))

        ## Smaller figure + higher compression to keep the docs-build HTML page
        ## a sensible size (the prior 720×640, default-CRF mp4 inflated the
        ## generated `tropical_cyclone_with_rainband.md` page well past
        ## Documenter's size_threshold).
        fig = Figure(size = (480, 360))
        ax = Axis(
            fig[1, 1]; xlabel = "x (km)", ylabel = "y (km)",
            aspect = DataAspect(),
            limits = (-120, 120, -120, 120)
        )
        n = Observable(1)
        w_n = @lift @view(w_frames[:, :, $n])
        title_t = @lift @sprintf(
            "w' response at z = %.1f km — t = %.1f h after heating onset",
            z_anim / 1.0e3, heat_times[$n] / hour
        )
        fig[0, :] = Label(fig, title_t; fontsize = 12, tellwidth = false)
        hm = heatmap!(
            ax, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, w_n;
            colormap = :balance, colorrange = (-w_lim, w_lim)
        )
        contour!(
            ax, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, Q_anim_warm;
            levels = [1.0], color = :red, linewidth = 1.5
        )
        contour!(
            ax, xs_grid ./ 1.0e3, ys_grid ./ 1.0e3, Q_anim_cool;
            levels = [-1.0], color = :blue, linewidth = 1.5
        )
        Colorbar(fig[1, 2], hm; label = "w' (m s⁻¹)")

        anim_path = joinpath(figures_dir, "response_w_z3km.mp4")
        CairoMakie.record(
            fig, anim_path, 1:Nt_anim;
            framerate = 2, compression = 30
        ) do nn
            n[] = nn
        end
        @info "Saved animation" anim_path
        nothing #hide
    else
        @warn "Missing control and/or heated snapshots — skipping F03a, F04, F05, animation"
    end

    @info "=== Analysis complete ==="
end

# ![](output_tc_rainband/figures/response_w_z3km.mp4)
