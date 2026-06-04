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
# 24 h spinup so the vortex relaxes to numerical equilibrium, then run a 24 h
# *heated* continuation with the MN10 stratiform heating switched on. We
# visualize the basic-state vortex, the analytic heating field, and a plan
# view of the vertical velocity that the heating drives. A commented-out
# control stub at the bottom lets users add the heated − control subtraction
# YD19 use for their full quadrupole response.
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
# | `F01_sounding.png`               | Jordan (1958) environmental sounding              |
# | `F02ab_vortex.png`               | basic-state vortex ([YuDidlake2019](@citet); Fig. 2a,b) |
# | `F02cd_heating.png`              | analytic heating ([YuDidlake2019](@citet); Fig. 2c,d)  |
# | `F02e_w_heated.png`              | plan-view w in heated run (z = 3 km)              |

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.AbstractOperations: XNode, YNode, grid_metric_operation
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
] .* 100

jordan_z_m = [
    0.0, 132.0, 583.0, 1054.0, 1547.0, 2063.0, 2609.0, 3182.0,
    3792.0, 4442.0, 5138.0, 5888.0, 6703.0, 7595.0, 8581.0, 9682.0,
    10935.0, 12396.0, 13236.0, 14177.0, 15260.0, 16568.0, 17883.0, 19620.0,
    20743.0, 22139.0, 23971.0,
]

jordan_T_K = [
    26.3, 26.0, 23.0, 19.8, 17.3, 14.6, 11.8, 8.6, 5.1, 1.4,
    -2.5, -6.9, -11.9, -17.7, -24.8, -33.2, -43.3, -55.2, -61.5, -67.6,
    -72.2, -73.5, -69.8, -63.9, -60.6, -57.3, -54.0,
] .+ 273.15

jordan_θ_K = [
    298.0, 299.0, 300.0, 302.0, 304.0, 307.0, 309.0, 312.0, 315.0, 318.0,
    321.0, 324.0, 328.0, 332.0, 335.0, 338.0, 342.0, 345.0, 348.0, 354.0,
    364.0, 386.0, 418.0, 468.0, 500.0, 542.0, 597.0,
]

## A 1-D vertical RectilinearGrid carries the Jordan sounding. The data are given
## on what amounts to cell interfaces, so the 27 sounding levels map onto the 27
## z-faces of a 26-cell Bounded grid and `ZFaceField`s store them exactly;
## `Oceananigans.Fields.interpolate` handles the linear lookup between levels.
## The fields are CPU-resident because `θ_env` and the `interpolate` calls in the
## balanced-vortex solve are evaluated host-side; `set!` evaluates such functions
## on the host before copying the result to GPU.
sounding_grid = RectilinearGrid(
    size = length(jordan_z_m) - 1, z = jordan_z_m,
    topology = (Flat, Flat, Bounded)
)

jordan_θ = ZFaceField(sounding_grid)
jordan_T = ZFaceField(sounding_grid)
jordan_p = ZFaceField(sounding_grid)

interior(jordan_θ, 1, 1, :) .= jordan_θ_K
interior(jordan_T, 1, 1, :) .= jordan_T_K
interior(jordan_p, 1, 1, :) .= jordan_p_mb

θ_env(z) = Oceananigans.Fields.interpolate(z, jordan_θ)

## The only environmental pressure we need as a scalar is the surface value, which
## anchors both the reference state and the acoustic substepper. The rest of p(z)
## then follows hydrostatically from θ(z) inside `ReferenceState`.
surface_pressure = Oceananigans.Fields.interpolate(0, jordan_p)

# ## Output layout
#
# Run outputs — the figures and the JLD2 output file — live under
# `examples/output_tc_rainband/`.

output_dir = joinpath(@__DIR__, "output_tc_rainband")
figures_dir = joinpath(output_dir, "figures")
mkpath(figures_dir)
nothing #hide

# ## A look at the environment
#
# Before building anything, let's look at the sounding we just loaded. CairoMakie
# plots the `Field`s directly, so the vertical coordinate comes straight from the
# grid (in metres) — no need to pull data out by hand.

fig = Figure(size = (1000, 420))
axθ = Axis(fig[1, 1]; xlabel = "θ (K)", ylabel = "z (m)", title = "Potential temperature")
axT = Axis(fig[1, 2]; xlabel = "T (K)", ylabel = "z (m)", title = "Temperature")
axp = Axis(fig[1, 3]; xlabel = "p (Pa)", ylabel = "z (m)", title = "Pressure")
lines!(axθ, jordan_θ; color = :magenta)
lines!(axT, jordan_T; color = :orangered)
lines!(axp, jordan_p; color = :dodgerblue)
save(joinpath(figures_dir, "F01_sounding.png"), fig) #src
fig

# ## YD19 physical parameters
#
# The vortex parameters utilize classic modified-Rankine vortex structure, following the parameters used in [YuDidlake2019](@citet) §3a2:
# a surface maximum wind of 43 m/s with a radius of maximum wind set at 31 km, decaying
# outward as ``r^{-1/2}`` in the modified-Rankine sense. Above ``z ≈ 16`` km
# (the outflow level) the tangential wind is zero. The outer ``\cos^2()`` taper
# between 250 and 300 km is *not* in the original paper. Since our domain is periodic,
# we need to impose this to limit unrealistic stress at the domain boundaries.

f = 5e-5                       # f-plane Coriolis parameter, 1/s ([YuDidlake2019](@citet); §3a1)
v_max_surface = 43             # initial surface v_max, m/s ([YuDidlake2019](@citet); §3a2)
a_decay = 0.5                  # modified-Rankine decay exponent ([YuDidlake2019](@citet); Eq. 2)
rmw_surface = 31kilometers     # surface radius of maximum wind, m ([MoonNolan2010](@citet); Appendix A)
z_vortex_top = 16kilometers    # outflow reference level, m ([MoonNolan2010](@citet); v = 0 at RMW at z ≈ 15.9 km)
r_taper_start = 250kilometers  # radial taper start for periodic-domain compatibility, m
r_taper_end = 300kilometers    # radial taper end, m

# ## Grid and architecture
#
# [YuDidlake2019](@citet) §3a1 use a 5 km inner-nest resolution with a
# 25 km deep domain. We match that on a ~ 642 km × 642 km periodic-in-x,y box:
# 128² cells horizontally and 75 levels vertically (``Δz ≈ 333`` m). The run
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

Δx = 4kilometers
Lx = 642kilometers
Nx = Ny = floor(Int, Lx / Δx)
Nz = 75
Lz = 25kilometers                 # YD19 §3a1
Δz = Lz / Nz
sponge_rate = Float32(1 / 333) # ≈ WRF damp_opt=2 `dampcoef`
stage_stop_time = 24hours

arch = GPU()

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
# far-field of the anelastic problem.

constants = ThermodynamicConstants()
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
g = constants.gravitational_acceleration
κ = Rᵈ / constants.dry_air.heat_capacity
cᵖᵈ = constants.dry_air.heat_capacity

reference_state = ReferenceState(
    grid, constants;
    surface_pressure,
    potential_temperature = θ_env
)

## Cell-center heights, reused by the analysis below.
z_centers = znodes(grid, Center())

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

ΔT_floor = 0.01   # micro-floor; avoids ÷ 0 exactly at z = z_vortex_top

function rmw_analytic(z)
    T_out = Oceananigans.Fields.interpolate(z_vortex_top, jordan_T)
    ΔT_0 = Oceananigans.Fields.interpolate(0, jordan_T) - T_out
    ΔT_z = max(Oceananigans.Fields.interpolate(z, jordan_T) - T_out, ΔT_floor)
    return rmw_surface * sqrt(ΔT_0 / ΔT_z)
end

function tangential_wind(r, z)
    r ≥ r_taper_end  && return 0
    z ≥ z_vortex_top && return 0
    rmw_z = rmw_analytic(z)
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
# case ([Nolan2001](@cite)) and by [YuDidlake2019](@citet) §3a2. We solve it on a
# 1-D-in-radius grid (`Flat` in y) entirely with Oceananigans `Field`s and
# operators. Each Picard sweep:
#
# - forms ``\rho = p / (R^d T)`` and the gradient-wind integrand ``\rho(fv + v^2/r)``;
# - integrates it inward from the far field with [`CumulativeIntegral`](@ref)
#   (`reverse = true`) — its built-in ``\Delta r`` metric *is* the radial integral,
#   so ``p = p_r - \int_r^R \rho(fv + v^2/r)\,dr``;
# - recovers ``\rho`` from hydrostatic balance with `∂z`, where a `Gradient`
#   boundary condition ``\partial p/\partial z = -\rho g`` makes the derivative
#   correct at the top and bottom with no special-casing, then sets
#   ``T = p / (R^d \rho)``;
# - under-relaxes the ``T`` update by ``\alpha`` to stabilize the fixed point.
#
# The iteration converges to a gradient-wind residual ``\sim 10^{-3}`` Pa/m and a
# hydrostatic residual at round-off. The radial grid reaches past the taper to the
# domain corner so the initial-condition interpolation below never extrapolates,
# and it runs in `Float64` even though the model is `Float32` (the gradient-wind
# residual sits near `Float32` ε).

pˢᵗ = 1.0e5

r_max_vortex = 500kilometers   # ≳ Lx/√2 (the domain corner) so the IC never extrapolates
Nr_vortex = 500                # ⇒ Δr = 1 km, fine enough to resolve the RMW

vortex_grid = RectilinearGrid(
    CPU(), Float64;
    size = (Nr_vortex, Nz), x = (0, r_max_vortex), z = (0, Lz),
    topology = (Bounded, Flat, Bounded)
)

vortex_reference = ReferenceState(
    vortex_grid, constants;
    surface_pressure, potential_temperature = θ_env
)

## Reference columns and the prescribed (tangential) wind vⱽ, as Fields on the
## (r, z) grid. The reference fields are reduced in the horizontal — (Nothing,
## Nothing, Center) — so `set!` broadcasts each column across r out of the box. The
## radius is a coordinate (`XNode`) operation, so it needs no Field of its own.
pᵣⱽ = CenterField(vortex_grid)
Tᵣⱽ = CenterField(vortex_grid)
vⱽ = CenterField(vortex_grid)
set!(pᵣⱽ, vortex_reference.pressure)
set!(Tᵣⱽ, vortex_reference.temperature)
set!(vⱽ, (r, z) -> tangential_wind(r, z))
radius = grid_metric_operation((Center(), Center(), Center()), XNode(), vortex_grid)

## Pressure carries hydrostatic Gradient boundary conditions ∂p/∂z = -ρg so that
## `∂z(p)` is correct at the top and bottom once halos are filled. We allocate the
## working fields once and `compute!` them in place each sweep. The density ρ is a
## standalone field — that lets the boundary conditions, which read it, be built
## *before* p — and the boundary values are just -ρg windowed to the bottom and top
## planes with `indices`.
ρⱽ = CenterField(vortex_grid)
integrand = ρⱽ * (f * vⱽ + vⱽ^2 / radius)

∫p = Field(@at((Center, Center, Center),
    Oceananigans.CumulativeIntegral(integrand, dims = 1, reverse = true)))

∂z_pᵇ = Field(-g * ρⱽ, indices = (:, :, 1))
∂z_pᵗ = Field(-g * ρⱽ, indices = (:, :, Nz))

pressure_bcs = FieldBoundaryConditions(
    vortex_grid, (Center(), Center(), Center());
    bottom = GradientBoundaryCondition(∂z_pᵇ),
    top = GradientBoundaryCondition(∂z_pᵗ)
)

pⱽ = CenterField(vortex_grid; boundary_conditions = pressure_bcs)
Tⱽ = CenterField(vortex_grid)
set!(pⱽ, pᵣⱽ)
set!(Tⱽ, Tᵣⱽ)

ρʰ = Field(@at((Center, Center, Center), -∂z(pⱽ) / g))
α = 0.5
T⁺ = Field(α * (pⱽ / (Rᵈ * ρʰ)) + (1 - α) * Tⱽ)

for iter in 1:60
    ρⱽ .= pⱽ / (Rᵈ * Tⱽ)

    ## gradient wind: pⱽ(r, z) = pᵣ(z) - ∫ᵣᴿ ρ(f vⱽ + vⱽ²/r) dr
    compute!(∫p)
    pⱽ .= pᵣⱽ - ∫p

    ## hydrostatic: ρ = -∂pⱽ/∂z / g, then T = pⱽ / (Rᵈ ρ); under-relax by α
    compute!(∂z_pᵇ)
    compute!(∂z_pᵗ)
    Oceananigans.BoundaryConditions.fill_halo_regions!(pⱽ)

    compute!(ρʰ)

    compute!(T⁺)
    Tⱽ .= T⁺
end

ρⱽ .= pⱽ / (Rᵈ * Tⱽ)

## Map the axisymmetric (r, z) solution onto the 3-D model with pointwise initial
## conditions; `set!` evaluates these on the host and `Oceananigans.Fields.interpolate`
## does the (r, z) lookup, so no hand-rolled interpolation table is needed.
r(x, y) = sqrt(x^2 + y^2)
uᵢ(x, y, z) = -y / r(x, y) * Oceananigans.Fields.interpolate((r(x, y), z), vⱽ)
vᵢ(x, y, z) = +x / r(x, y) * Oceananigans.Fields.interpolate((r(x, y), z), vⱽ)
Tᵢ(x, y, z) = Oceananigans.Fields.interpolate((r(x, y), z), Tⱽ)
ρᵢ(x, y, z) = Oceananigans.Fields.interpolate((r(x, y), z), ρⱽ)

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
# inside the rainband, so we use the reference profile. Because that profile is
# indexed by vertical level, the forcing is written in discrete form — the
# `(i, j, k, …)` signature hands us `k` directly to look up ``\rho_r`` and ``\Pi``.

Qₘₐₓ = 4.24f0 / Float32(hour)     # YD19 Eq. 3 peak rate, 4.24 K/h (stored in K/s)
zᵇ = Float32(4kilometers)
σᵣ = Float32(6kilometers)
σᶻ = Float32(2kilometers)
tᵣ = Float32(1hour)            # ramp duration, to avoid an instantaneous onset

## Rainband geometry (YD19 Eq. 3): the heating centerline sits at radius `rᵇ` on
## the downwind (λ = 0) edge and moves outward by `Δrᵇ` per π/4 of azimuth. They
## are `const` so the GPU forcing kernel that reads them stays type-stable.
const rᵇ = 60kilometers
const Δrᵇ = 10kilometers

## Reference Exner function Πᵣ(z) = (pᵣ/pˢᵗ)^κ converts the heating rate Q (K/s) into
## the ρθ tendency ρ Q / Π. It — like the reference density — is a reduced
## (Nothing, Nothing, Center) Field, indexed by vertical level in the forcing below.
Πᵣ = Field((reference_state.pressure / pˢᵗ)^κ)

@inline function rainband_heating(i, j, k, grid, clock, fields, p)
    x = Oceananigans.Grids.xnode(i, j, k, grid, Center(), Center(), Center())
    y = Oceananigans.Grids.ynode(i, j, k, grid, Center(), Center(), Center())
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())

    ramp = clamp((clock.time - p.tᵒⁿ) / p.tᵣ, 0, 1)
    r = sqrt(x^2 + y^2)
    s = 4 * atan(y, x) / π                 # azimuth in quadrants: s = λ / (π/4)
    r_bs = rᵇ - Δrᵇ * s + z         # spiral centerline, tilting outward with height
    G_r = exp(-(r - r_bs)^2 / 2p.σᵣ^2)     # radial Gaussian

    z_rel = (z - p.zᵇ) / p.σᶻ
    V_z = ifelse(abs(z_rel) < 1, sin(π * z_rel), zero(z_rel))   # vertical sine lobe

    A_λ = exp(-(s + 1)^8)                   # azimuthal super-Gaussian, centered at λ = -π/4 (s = -1)

    Q = p.Qₘₐₓ * G_r * V_z * A_λ * ramp
    ## ρθ tendency ρᵣ Q / Πᵣ, reading the level-indexed reference fields (reduced in
    ## x and y, so any i, j returns the column value).
    ρᵣ_k = @inbounds p.ρᵣ[i, j, k]
    Πᵣ_k = @inbounds p.Πᵣ[i, j, k]
    return ρᵣ_k * Q / Πᵣ_k
end

## The ramp switches the heating on only after the spinup, at tᵒⁿ = stage_stop_time.
heating_params = (;
    Qₘₐₓ, σᵣ, σᶻ, zᵇ, tᵒⁿ = Float32(stage_stop_time), tᵣ,
    ρᵣ = reference_state.density, Πᵣ,
)

heating_forcing = Forcing(rainband_heating; discrete_form = true, parameters = heating_params)

## Analytic heating rate at full strength (K/h) — for figure contours.
function heating_rate_K_per_hour(r, λ, z)
    s = 4λ / π
    r_bs = rᵇ - Δrᵇ * s + z
    G_r = exp(-(r - r_bs)^2 / 2σᵣ^2)
    z_rel = (z - zᵇ) / σᶻ
    V_z = abs(z_rel) < 1 ? sin(π * z_rel) : zero(z_rel)
    A_λ = exp(-(s + 1)^8)
    return Qₘₐₓ * hour * G_r * V_z * A_λ   # K/h (Qₘₐₓ is stored in K/s)
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
ρθᵣ = Field(reference_state.density * reference_state.temperature * (pˢᵗ / reference_state.pressure)^κ)

sponge_vel_params = (z_bot = sponge_z_bottom, z_top = sponge_z_top, rate = sponge_rate)
sponge_ρθ_params = (
    z_bot = sponge_z_bottom, z_top = sponge_z_top, rate = sponge_rate,
    ρθ_bg = ρθᵣ,
)

## WRF `damp_opt=2` analog: zero below z_bot, sin²() ramp to max at z_top.
@inline function sponge_mask(z, z_bot, z_top)
    ξ = (z - z_bot) / (z_top - z_bot)
    return ifelse(ξ ≤ 0, zero(ξ), sin(π * ξ / 2)^2)
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
    ρθ_tgt = @inbounds p.ρθ_bg[i, j, k]
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

# ## Build the model
#
# A single model carries both the rainband heating and the upper-level sponge.
# Because the heating's time ramp only switches on at `tᵒⁿ = stage_stop_time`, the
# first `stage_stop_time` is an unforced spinup and the remainder is the heated
# continuation — there's no need for a second model or to re-initialize.

coriolis = FPlane(; f)
dynamics = CompressibleDynamics(
    SplitExplicitTimeDiscretization();
    surface_pressure, reference_potential_temperature = θ_env
)
forcing = (ρu = sponge_ρu, ρv = sponge_ρv, ρw = sponge_ρw, ρθ = (heating_forcing, sponge_ρθ))
model = AtmosphereModel(grid; dynamics, coriolis, advection = WENO(order = 5), forcing)

set!(model; u = uᵢ, v = vᵢ, T = Tᵢ, ρ = ρᵢ)

# ## Run the spinup and heated continuation
#
# We step through `2 * stage_stop_time` — a 24 h spinup followed by the 24 h heated
# run — and write hourly output. The velocities are interpolated to cell centers
# with `@at` and the potential temperature is computed with
# `liquid_ice_potential_temperature`, both *online*, so the analysis below reads
# ready-to-use `Field`s straight back with `FieldTimeSeries`: no manual
# interpolation, no thermodynamic reconstruction.

simulation = Simulation(model; Δt = 2.0, stop_time = 2stage_stop_time)
conjure_time_step_wizard!(simulation, cfl = 1.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

u, v, w = model.velocities
progress(sim) = @info "iter $(iteration(sim)),  t = $(prettytime(sim)),  Δt = $(prettytime(sim.Δt)),  max|w| = $(maximum(abs, w)) m/s"
add_callback!(simulation, progress, TimeInterval(6hours))

## Tangential wind vθ = (-y u + x v)/r, formed online from the XNode/YNode coordinate
## operations — so the azimuthal average below is a pure radial binning.
xᶜ = grid_metric_operation((Center(), Center(), Center()), XNode(), grid)
yᶜ = grid_metric_operation((Center(), Center(), Center()), YNode(), grid)
rᶜ = sqrt(xᶜ^2 + yᶜ^2)

output_filename = joinpath(output_dir, "tc_rainband.jld2")
outputs = (
    vθ = Field(@at((Center, Center, Center), -yᶜ * u + xᶜ * v) / rᶜ),
    w = Field(@at((Center, Center, Center), w)),
    θ = liquid_ice_potential_temperature(model),
)
simulation.output_writers[:fields] = JLD2Writer(
    model, outputs;
    filename = output_filename, schedule = TimeInterval(1hour), overwrite_existing = true
)

run!(simulation)

# ## Stage 4 — Analysis and figure production
# Now that the have the full simulation, we replicate the figures from YD19 to verify our results.

@info "=== Stage 4: Analysis ==="

## The writer stored the (online-rotated) tangential wind, w, and θ, so we read them
## straight back as `FieldTimeSeries`. The analysis snapshots are the end of the
## spinup (t = stage_stop_time) and the end of the heated run.
vθt = FieldTimeSeries(output_filename, "vθ")
wt = FieldTimeSeries(output_filename, "w")
θt = FieldTimeSeries(output_filename, "θ")
times = vθt.times
n_spinup = searchsortedfirst(times, stage_stop_time)   # end of spinup
n_heated = length(times)                               # end of heated run

r_bin_edges = collect(range(0, 150kilometers, step = Δx))
Nr_bin = length(r_bin_edges) - 1
r_bin_centers = 0.5 .* (r_bin_edges[1:(end - 1)] .+ r_bin_edges[2:end])
xs_center = xnodes(grid, Center())
ys_center = ynodes(grid, Center())
r_last = last(r_bin_edges)

## Sector azimuthal average. The tangential wind was already formed online, so here
## we only bin into radial rings (Oceananigans has no polar reduction).
vθ_ws = zeros(Nr_bin, Nz)
θ_ws = similar(vθ_ws)
ct_ws = zeros(Int, Nr_bin, Nz)

function azimuthal_mean!(v̄θ, θ̄, ct, vθ, θ)
    fill!(v̄θ, 0); fill!(θ̄, 0); fill!(ct, 0)
    Nxs, Nys, Nzs = size(vθ)
    @inbounds for k in 1:Nzs, j in 1:Nys, i in 1:Nxs
        r = sqrt(xs_center[i]^2 + ys_center[j]^2)
        r ≥ r_last && continue
        ib = clamp(searchsortedlast(r_bin_edges, r), 1, Nr_bin)
        v̄θ[ib, k] += vθ[i, j, k]
        θ̄[ib, k] += θ[i, j, k]
        ct[ib, k] += 1
    end
    @inbounds for k in 1:Nzs, ib in 1:Nr_bin
        if ct[ib, k] > 0
            v̄θ[ib, k] /= ct[ib, k]
            θ̄[ib, k] /= ct[ib, k]
        end
    end
    return nothing
end

# ## F02ab — basic-state vortex (YD19 Fig 2a,b)

azimuthal_mean!(vθ_ws, θ_ws, ct_ws, interior(vθt[n_spinup]), interior(θt[n_spinup]))

t_final = times[n_spinup]
θ_env_column = [θ_env(z) for z in z_centers]
θ_anom = θ_ws .- reshape(θ_env_column, 1, :)

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
θ_span = 10
hm_θ = heatmap!(
    ax_θ, r_bin_centers ./ kilometer, z_centers ./ kilometer, θ_anom;
    colormap = :balance, colorrange = (-θ_span, θ_span)
)
contour!(
    ax_θ, r_bin_centers ./ kilometer, z_centers ./ kilometer, θ_anom;
    levels = -θ_span:1.69:θ_span, color = :black, linewidth = 0.5
)
Colorbar(fig[1, 4], hm_θ; label = "θ̄' (K)")

Label(fig[0, :],
    "F02ab — Basic-state vortex at t = $(round(t_final / hour, digits = 1)) h spin-up ($(round(Int, Lx / kilometers)) km box)";
    fontsize = 17)

save(joinpath(figures_dir, "F02ab_vortex.png"), fig) #src
fig

# ## F02cd — analytic heating field (YD19 Fig 2c,d)

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
    title = "(d) Heating plan view at z = $(round(z_level / 1000, digits = 1)) km",
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

Label(fig[0, :],
    "F02cd — MN10 stratiform heating field ($(round(Int, Lx / kilometers)) km box)";
    fontsize = 17)

save(joinpath(figures_dir, "F02cd_heating.png"), fig) #src
fig

# ## F02e — plan-view vertical velocity in the heated run

t_heat_final = times[n_heated]
w_heated = interior(wt[n_heated])              # cell-centered w (host array)

z_w_slice = 3kilometers
k_w = argmin(abs.(z_centers .- z_w_slice))
w_slice = w_heated[:, :, k_w]
w_lim = max(0.5, ceil(maximum(abs, w_slice) * 2) / 2)
z_w_km = z_centers[k_w] / kilometer

## Heating overlay at the slice altitude — the red/blue ±1 K/h contours mark
## where the imposed forcing sits.
Q_slice = [
    heating_rate_K_per_hour(
            sqrt(xs_center[i]^2 + ys_center[j]^2),
            atan(ys_center[j], xs_center[i]),
            z_centers[k_w]
        )
        for i in 1:Nx, j in 1:Ny
]

fig = Figure(size = (800, 700))
ax = Axis(
    fig[1, 1];
    xlabel = "x (km)", ylabel = "y (km)",
    title = "(e) Heated-run vertical velocity at z = $(round(z_w_km, digits = 1)) km, t = $(round(t_heat_final / hour, digits = 1)) h",
    aspect = DataAspect(),
    limits = (-120, 120, -120, 120),
)
hm = heatmap!(
    ax, xs_center ./ kilometer, ys_center ./ kilometer, w_slice;
    colormap = :balance, colorrange = (-w_lim, w_lim)
)
maximum(Q_slice) > 1 && contour!(ax, xs_center ./ kilometer, ys_center ./ kilometer, Q_slice;
    levels = [1.0], color = :red, linewidth = 2)
minimum(Q_slice) < -1 && contour!(ax, xs_center ./ kilometer, ys_center ./ kilometer, Q_slice;
    levels = [-1.0], color = :blue, linewidth = 2)
Colorbar(fig[1, 2], hm; label = "w (m s⁻¹)")

Label(fig[0, :],
    "F02e — Plan-view w in heated run (z = $(round(z_w_km, digits = 1)) km, $(round(Int, Lx / kilometers)) km box)";
    fontsize = 17)

save(joinpath(figures_dir, "F02e_w_heated.png"), fig) #src
fig

# ## Reproducing the full YD19 response (optional)
#
# To get the quadrupole *response* (heated − control) that YD19 plot in their
# Figs 3–4, also run a control stage with `with_heating = false` from the
# post-spinup state and subtract it from the heated captures above:
#
# ```julia
# @info "=== Control: $(prettytime(stage_stop_time)) ==="
# control_result = build_and_run_stage!(
#     "control";
#     with_heating = false,
#     init = post_spinup,
#     stop_time = stage_stop_time,
# )
# ```
