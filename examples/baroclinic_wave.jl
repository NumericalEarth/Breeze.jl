# # Baroclinic wave on the sphere
#
# This example simulates the growth of a baroclinic wave on a near-global
# `LatitudeLongitudeGrid` following the DCMIP2016 specification
# [UllrichEtAl2016](@citet), which extends the classic
# [JablonowskiWilliamson2006](@citet) test case.
# A midlatitude jet in thermal-wind balance with a meridional temperature
# gradient is seeded with a localized zonal-wind perturbation that triggers
# baroclinic instability, producing growing Rossby waves over roughly ten days.
#
# This example exercises `CompressibleDynamics` with `ExplicitTimeStepping`
# and `HydrostaticSphericalCoriolis` on a latitude-longitude grid spanning
# 85° S to 85° N.
#
# ## Physical setup
#
# The background state is an analytic steady-state solution of the dry,
# adiabatic, inviscid primitive equations in height coordinates.
# The temperature field has two parts: a horizontally uniform stratification
# controlled by a lapse rate ``Λ`` and a meridional gradient that creates
# warm equator / cold pole contrast:
#
# ```math
# T(φ, z) = \frac{1}{τ_1(z) - τ_2(z)\, F(φ)}
# ```
#
# where ``τ_1`` and ``τ_2`` encode the vertical structure and
# ``F(φ) = \cos^K φ - \frac{K}{K+2} \cos^{K+2} φ`` is the meridional shape
# with jet-width parameter ``K = 3``.
#
# ### Balanced zonal jet
#
# The zonal wind is derived analytically from gradient-wind balance,
# producing a subtropical jet peaking near 30 m/s at 45° latitude
# in the upper troposphere.
#
# ### Perturbation
#
# A localized zonal-wind perturbation centered at
# ``(λ_c, φ_c) = (20°\text{E}, 40°\text{N})`` seeds the instability.
# The perturbation decays exponentially with great-circle distance from the
# center and is tapered smoothly to zero above 15 km:
#
# ```math
# u'(λ, φ, z) = u_p \, \mathcal{T}(z) \, \exp\!\left(-\left(\frac{d}{r_p}\right)^2\right)
# ```
#
# where ``d`` is the great-circle distance, ``r_p = 0.1\,a``, ``u_p = 1`` m/s,
# and ``\mathcal{T}(z) = 1 - 3(z/z_p)^2 + 2(z/z_p)^3`` for ``z < z_p``.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using CUDA

# ## Domain and grid
#
# We use a near-global latitude-longitude grid at roughly 2° horizontal
# resolution, excluding the poles to avoid the coordinate singularity.
# The domain extends from the surface to 30 km with 30 vertical levels.

Nλ = 180
Nφ = 85
Nz = 30
H  = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-85, 85),
                             z = (0, H))

# ## DCMIP2016 parameters
#
# All parameters follow the DCMIP2016 test case document
# (Ullrich, Melvin, Staniforth, and Jablonowski, 2016).

const 𝑎  = 6371220.0   # m — Earth radius
const Ω  = 7.29212e-5  # s⁻¹ — Earth rotation rate
const 𝑔  = 9.80616     # m/s² — gravitational acceleration
const Rᵈ = 287.0       # J/(kg·K) — dry air gas constant
const cₚ = 1004.5      # J/(kg·K) — specific heat capacity
const κ  = 2 / 7       # Rᵈ/cₚ
const p₀ = 100000.0    # Pa — surface pressure

## Temperature profile parameters
const T₀E   = 310.0    # K — equatorial surface temperature
const T₀P   = 240.0    # K — polar surface temperature
const T₀    = 0.5 * (T₀E + T₀P)  # K — mean surface temperature
const K_jet  = 3.0     # jet width parameter
const B_jet  = 2.0     # jet half-width parameter
const Λ      = 0.005   # K/m — lapse rate

## Derived constants
const constA = 1.0 / Λ
const constB = (T₀ - T₀P) / (T₀ * T₀P)
const constC = 0.5 * (K_jet + 2) * (T₀E - T₀P) / (T₀E * T₀P)
const constH = Rᵈ * T₀ / 𝑔

## Perturbation parameters (exponential type)
const pertup   = 1.0          # m/s — perturbation amplitude
const pertexpr = 0.1          # perturbation radius in Earth radii
const pertlon  = π / 9        # 20° E
const pertlat  = 2π / 9       # 40° N
const pertz    = 15000.0      # m — perturbation height cap

# ## Analytic initial conditions
#
# The temperature and pressure are computed from the DCMIP2016 analytic
# formulas. The vertical structure functions ``τ_1, τ_2`` and their
# integrals encode the stratification and meridional gradient.

## Vertical structure functions (shallow atmosphere, X = 1)
function τ_and_integrals(z)
    scaledZ = z / (B_jet * constH)
    expZ2 = exp(-scaledZ^2)

    τ₁    = constA * Λ / T₀ * exp(Λ * z / T₀) + constB * (1 - 2 * scaledZ^2) * expZ2
    τ₂    = constC * (1 - 2 * scaledZ^2) * expZ2
    ∫τ₁   = constA * (exp(Λ * z / T₀) - 1) + constB * z * expZ2
    ∫τ₂   = constC * z * expZ2

    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

## Meridional shape functions
F_T(φ) = cosd(φ)^K_jet - K_jet / (K_jet + 2) * cosd(φ)^(K_jet + 2)
F_U(φ) = cosd(φ)^(K_jet - 1) - cosd(φ)^(K_jet + 1)

## Temperature: T(φ, z) = 1 / (τ₁ - τ₂ F(φ))
function Tᵢ(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1.0 / (τ₁ - τ₂ * F_T(φ))
end

## Pressure: p(φ, z) = p₀ exp(-g/Rᵈ (∫τ₁ - ∫τ₂ F(φ)))
function pᵢ(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-𝑔 / Rᵈ * (∫τ₁ - ∫τ₂ * F_T(φ)))
end

## Density from the ideal gas law
ρᵢ(λ, φ, z) = pᵢ(λ, φ, z) / (Rᵈ * Tᵢ(λ, φ, z))

## Potential temperature: θ = T (p₀/p)^κ
function θᵢ(λ, φ, z)
    T = Tᵢ(λ, φ, z)
    p = pᵢ(λ, φ, z)
    return T * (p₀ / p)^κ
end

# ### Balanced zonal wind
#
# The zonal wind satisfies gradient-wind balance with the temperature field.
# For the shallow atmosphere (``r = a``):
#
# ```math
# u = -Ω a \cos φ + \sqrt{Ω^2 a^2 \cos^2 φ + a \cos φ \, U(φ, z)}
# ```
#
# where ``U = (g/a) K \int τ_2 \, T \, (\cos^{K-1} φ - \cos^{K+1} φ)``.

function uᵢ(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    T = Tᵢ(λ, φ, z)

    bigU = 𝑔 / 𝑎 * K_jet * ∫τ₂ * F_U(φ) * T
    rcosφ = 𝑎 * cosd(φ)
    Ωrcosφ = Ω * rcosφ

    u_bal = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * bigU)

    ## Add the exponential perturbation
    φ_rad = deg2rad(φ)
    λ_rad = deg2rad(λ)
    great_circle = 1 / pertexpr * acos(sin(pertlat) * sin(φ_rad) +
                                       cos(pertlat) * cos(φ_rad) * cos(λ_rad - pertlon))

    taper = ifelse(z < pertz, 1 - 3 * (z / pertz)^2 + 2 * (z / pertz)^3, 0.0)
    u_pert = ifelse(great_circle < 1.0, pertup * taper * exp(-great_circle^2), 0.0)

    return u_bal + u_pert
end

# ## Model configuration
#
# We use fully explicit compressible dynamics. The time step is limited
# by the acoustic CFL. The reference state uses the equatorial column
# ``θ(z)`` profile evaluated at the equator, so the buoyancy force is
# computed as a perturbation for accuracy.
# `HydrostaticSphericalCoriolis` retains the traditional ``f = 2Ω \sin φ``
# Coriolis terms.

## Reference potential temperature at the equator
θ_ref(z) = θᵢ(0, 0, z)

coriolis = HydrostaticSphericalCoriolis()

dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_ref)

model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())

# ## Set initial conditions

set!(model, θ=θᵢ, u=uᵢ, ρ=ρᵢ)

# ## Time-stepping
#
# With explicit time stepping the time step is limited by the acoustic CFL.
# For ``Δx ≈ 200`` km and sound speed ``c_s ≈ 340`` m/s,
# the acoustic CFL gives ``Δt ≈ 2`` s.
# We run for 15 days to observe baroclinic wave growth; the instability
# becomes visible around day 4 and develops explosive cyclogenesis near day 8.

Δt = 2seconds
stop_time = 15days

simulation = Simulation(model; Δt, stop_time)

# Progress callback:

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("Iter %5d | t = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Output
#
# We save the velocities and the potential temperature for visualization.
# Also save surface pressure (bottom-level pressure) for comparison with
# published DCMIP reference solutions.

θ = PotentialTemperature(model)

## Background θ at the equator for computing perturbation θ′
θ_bg = CenterField(grid)
set!(θ_bg, (λ, φ, z) -> θ_ref(z))
θ′ = θ - θ_bg

outputs = merge(model.velocities, (; θ′))

simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              filename = "baroclinic_wave",
                                              schedule = TimeInterval(1hours),
                                              overwrite_existing = true)

# ## Run

run!(simulation)

# ## Visualization
#
# We plot the potential-temperature perturbation ``θ'`` (departure from the
# equatorial background ``θ^{\rm ref}(z)``) and the zonal wind on the sphere.

θ′_ts = FieldTimeSeries("baroclinic_wave.jld2", "θ′")
u_ts = FieldTimeSeries("baroclinic_wave.jld2", "u")
w_ts = FieldTimeSeries("baroclinic_wave.jld2", "w")
times = θ′_ts.times
Nt = length(times)

# Select the mid-level index for horizontal slices:
k_mid = Nz ÷ 2
z_mid = znode(k_mid, grid, Center())

# ### Final snapshot on the sphere

fig = Figure(size = (1200, 600))
sphere_kw = (elevation = π/6, azimuth = -π/2, aspect = :data)

ax1 = Axis3(fig[1, 1];
            title = "θ′ at z = $(z_mid/1e3) km, t = $(prettytime(times[Nt]))", sphere_kw...)
hm1 = surface!(ax1, view(θ′_ts[Nt], :, :, k_mid); colormap = :balance, shading = NoShading)
Colorbar(fig[1, 2], hm1; label = "θ′ (K)")

ax2 = Axis3(fig[1, 3];
            title = "u at z = $(z_mid/1e3) km, t = $(prettytime(times[Nt]))", sphere_kw...)
hm2 = surface!(ax2, view(u_ts[Nt], :, :, k_mid); colormap = :speed, shading = NoShading)
Colorbar(fig[1, 4], hm2; label = "u (m/s)")

for ax in (ax1, ax2)
    hidedecorations!(ax)
    hidespines!(ax)
end

current_figure()

# ### Animation
#
# Animate the potential-temperature perturbation and the vertical velocity
# on the sphere over the full simulation:

n = Observable(1)
θ′n = @lift view(θ′_ts[$n], :, :, k_mid)
wn = @lift view(w_ts[$n], :, :, k_mid)

fig = Figure(size = (1200, 600))
sphere_kw = (elevation = π/6, azimuth = -π/2, aspect = :data)

title = @lift "z = $(z_mid/1e3) km, t = $(prettytime(times[$n]))"

ax1 = Axis3(fig[1, 1]; title = "θ′", sphere_kw...)
hm1 = surface!(ax1, θ′n; colormap = :balance, colorrange = (-2, 2), shading = NoShading)
Colorbar(fig[1, 2], hm1; label = "θ′ (K)")

ax2 = Axis3(fig[1, 3]; title = "w", sphere_kw...)
hm2 = surface!(ax2, wn; colormap = :balance, colorrange = (-1, 1), shading = NoShading)
Colorbar(fig[1, 4], hm2; label = "w (m/s)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

for ax in (ax1, ax2)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig, "baroclinic_wave.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave.mp4)
