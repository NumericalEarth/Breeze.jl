# # Baroclinic wave on the sphere
#
# This example simulates the growth of a baroclinic wave on the Yin-Yang grid of
# [Kageyama and Sato (2004)](@cite KageyamaSato2004), following the DCMIP2016
# specification [UllrichEtAl2016](@citet), which extends the classic
# [JablonowskiWilliamson2006](@citet) test case.
# A midlatitude jet in thermal-wind balance with a meridional temperature
# gradient is seeded with a localized zonal-wind perturbation that triggers
# baroclinic instability, producing growing Rossby waves over roughly ten days.
#
# A latitude-longitude grid has a pole problem: its meridians converge, so the
# polar cells are tiny and set the time step. The Yin-Yang grid covers the
# whole sphere with two identical pieces of a latitude-longitude grid, each
# spanning 270° of longitude and 90° of latitude around its own equator. The
# "Yin" panel is the low-latitude band of the geographic grid; the "Yang" panel
# is the same band rotated so that its equator runs through the geographic
# poles. Neither panel comes near its own poles, so the grid spacing is nearly
# uniform, and each panel fills its halos by interpolation from the other.
#
# This example exercises `CompressibleDynamics` with
# [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization)
# (acoustic substepping via [`AcousticRungeKutta3`](@ref)) and
# `SphericalCoriolis` (non-hydrostatic) on a 1° Yin-Yang grid. Acoustic
# substepping lets the outer time step be set by the *advective* CFL rather
# than the much-tighter acoustic CFL, and the Yin-Yang grid has no polar cells,
# so a fixed 12-minute time step suffices.
#
# A future moist version (one-moment mixed-phase microphysics + bulk surface
# fluxes) will be added once the moist substepper supports the larger ``Δt``
# needed for a tractable runtime. The dry case here runs for 30 days and
# captures the full BCI life cycle: visible perturbations by day 4,
# explosive cyclogenesis around day 8, and saturation afterward.
#
# ## Physical setup
#
# The DCMIP2016 background state is an analytic steady-state solution of
# the dry, adiabatic, inviscid primitive equations in height coordinates,
# expressed in **virtual** temperature ``T_v(\varphi, z)``:
#
# ```math
# T_v(φ, z) = \frac{1}{τ_1(z) - τ_2(z)\, F(φ)}
# ```
#
# where ``τ_1`` and ``τ_2`` encode the vertical structure and
# ``F(φ) = \cos^K φ - \frac{K}{K+2} \cos^{K+2} φ`` is the meridional shape
# with jet-width parameter ``K = 3``. In the dry case, ``T = T_v``.
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
using Oceananigans.Coriolis: TriadScheme
using Oceananigans.Fields: location
using Oceananigans.MultiRegion: remapped_field, latitude_longitude_grid
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Printf
using CairoMakie
using CUDA

# ## DCMIP2016 parameters
#
# All parameters follow the DCMIP2016 test case document
# [UllrichEtAl2016](@citet). We set the Oceananigans defaults and build a
# custom [`ThermodynamicConstants`](@ref) matching the DCMIP specification
# so that the grid, Coriolis, and model thermodynamics are all consistent
# with the analytic initial conditions.

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5    # Pa — surface pressure
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# ## Domain and grid
#
# Each panel of the Yin-Yang grid spans 270° × 90° with 1° cells, so the size
# is `(270, 90, Nz)`. The cells are widest at a panel's equator (`111 km`) and
# narrowest at its edges, 45° from its equator (`79 km`). The domain extends
# from the surface to 30 km with 64 vertical levels, exponentially stretched
# toward the surface with `ExponentialDiscretization`: the interfaces are
# clustered near the ground (`bias = :left`) so the smallest cells sit at the
# surface (`Δz ≈ 150 m`) and coarsen to `≈ 1070 m` at the model top. The
# e-folding `scale = H/2` sets how quickly the spacing grows with height.

Nz = 64
H  = 30kilometers

z_faces = ExponentialDiscretization(Nz, 0, H; scale = H/2, bias = :left)

grid = YinYangGrid(GPU(); size = (270, 90, Nz), z = z_faces)

# ## Temperature profile parameters

Tᴱ = 310     # K — equatorial surface temperature
Tᴾ = 240     # K — polar surface temperature
Tᴹ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005    # K/m — lapse rate
K  = 3        # jet width parameter
b  = 2        # vertical half-width parameter

# ## Analytic initial conditions
#
# The DCMIP2016 balanced state is given in *virtual* temperature ``T_v``.
# We define the IC in terms of ``T_v`` so a future moist version (where
# ``T = T_v / (1 + \epsilon q^v)``) can reuse the same functions.

## Vertical structure functions (shallow atmosphere, X = 1)
function τ_and_integrals(z)
    Hˢ = Rᵈ * Tᴹ / g
    η  = z / (b * Hˢ)
    e  = exp(-η^2)

    A = (Tᴹ - Tᴾ) / (Tᴹ * Tᴾ)
    C = (K + 2) * (Tᴱ - Tᴾ) / (2 * Tᴱ * Tᴾ)

    τ₁  = A * (1 - 2η^2) * e + exp(Γ * z / Tᴹ) / Tᴹ
    ∫τ₁ = A * z * e + (exp(Γ * z / Tᴹ) - 1) / Γ

    τ₂  = C * (1 - 2η^2) * e
    ∫τ₂ = C * z * e

    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

## Meridional shape functions
F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)

## Virtual temperature: Tᵥ(φ, z) = 1 / (τ₁ - τ₂ F(φ))
function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

## Pressure: p(φ, z) = p₀ exp(-g/Rᵈ (∫τ₁ - ∫τ₂ F(φ)))
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

## Density (uses Tᵥ in the ideal gas law; in the dry case, T = Tᵥ).
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

## Potential temperature: θ = Tᵥ (p₀/p)^κ in the dry case.
potential_temperature(λ, φ, z) = virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

# ### Balanced zonal wind
#
# The zonal wind satisfies gradient-wind balance with the temperature field.
# For the shallow atmosphere (``r = a``):
#
# ```math
# u = -Ω a \cos φ + \sqrt{Ω^2 a^2 \cos^2 φ + a \cos φ \, U(φ, z)}
# ```
#
# where ``U = (g/a) K \int τ_2 \, T_v \, (\cos^{K-1} φ - \cos^{K+1} φ)``.

function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    Tᵥ = virtual_temperature(λ, φ, z)

    ## Gradient-wind balance
    U = g / a * K * ∫τ₂ * dF(φ) * Tᵥ
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    u_balanced = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)

    ## Localized perturbation (DCMIP2016 §3.3)
    uₚ = 1         # m/s — amplitude
    rₚ = 0.1       # perturbation radius (Earth radii)
    λₚ = π / 9     # 20°E center longitude
    φₚ = 2π / 9    # 40°N center latitude
    zₚ = 15000     # m — height cap

    φʳ = deg2rad(φ)
    λʳ = deg2rad(λ)
    great_circle = acos(sin(φₚ) * sin(φʳ) + cos(φₚ) * cos(φʳ) * cos(λʳ - λₚ)) / rₚ

    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, zero(z))
    u_perturbation = ifelse(great_circle < 1, uₚ * taper * exp(-great_circle^2), zero(z))

    return u_balanced + u_perturbation
end

# ## Model configuration
#
# We use fully compressible dynamics with **acoustic substepping** via
# [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization)
# and the [`AcousticRungeKutta3`](@ref)
# (Wicker–Skamarock RK3) outer loop. Acoustic substepping handles the
# fast acoustic-mode pressure gradient and buoyancy via a vertically-implicit
# inner loop, so the outer time step is limited only by the *advective*
# CFL — about 100× larger than the acoustic-CFL-limited Δt the fully
# explicit solver requires for the same grid.
#
# We use a hydrostatically-balanced isothermal reference state at
# `T₀ᵣ = 250 K` (matching the MPAS convention) so that the substepper's
# slow tendencies see only perturbations from the background.
# `SphericalCoriolis` is the non-hydrostatic spherical form, which retains
# both the traditional ``f = 2Ω \sin φ`` and the non-traditional
# ``2Ω \cos φ`` cross-terms that couple horizontal momentum to ``w``. Breeze
# evolves prognostic ``ρw`` so the non-traditional terms are physically
# required for self-consistent dynamics on the sphere. The axes of the Yang
# panel are rotated from east and north, so the Coriolis parameter varies
# along both of its grid directions; the triad scheme of
# [Arakawa and Lamb (1981)](@cite ArakawaLamb1981) weights it isotropically,
# so that the balanced jet is discretized alike on both panels.
#
# Tracer and momentum advection uses fifth-order `WENO` reconstruction. No
# explicit closure is applied: WENO's implicit dissipation suffices on this
# near-surface-stretched grid. On a Yin-Yang grid, the `AtmosphereModel`
# also removes the mass that the interpolation between the panels adds.

coriolis = SphericalCoriolis(rotation_rate=Ω, scheme=TriadScheme())

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                base_pressure = p₀,
                                reference_potential_temperature = θᵣ)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO(order=5))

# ## Set initial conditions
#
# On a Yin-Yang grid, `set!` takes the geographic (eastward) velocity and
# rotates it into the basis of each panel.

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density)

# ## Time-stepping
#
# Substepping eliminates the acoustic CFL constraint on the outer Δt; only
# the advective CFL remains. The smallest cells, `79 km` wide, sit at the
# edges of the panels, so a fixed `Δt = 12 min` keeps the advective CFL
# below 0.7 even where the mature wave reaches 70 m/s. We run for 30 days
# to capture the full BCI life cycle. The conservation fixer removes the mass
# that the mismatch of the panels' fluxes through their seam adds.

Δt = 12minutes
stop_time = 30days

simulation = Simulation(model; Δt, stop_time)
add_conservation_fixer!(simulation)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Progress callback

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("Iter %5d | t = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# ## Output
#
# We save the full potential temperature ``θ`` (the classic surface synoptic
# diagnostic for the cold/warm sectors during cyclogenesis) and the vertical
# vorticity ``ζ`` at the surface, and the vertical velocity ``w`` at k = 16
# (lower troposphere, ~2.9 km). Vorticity is a scalar, so each panel's own
# ``ζ`` is correct.
#
# Each panel stores its fields in its own coordinates. For plotting, we remap
# the three slices onto a global 1° latitude-longitude grid: the remap
# interpolates each node from the panel that owns it. The remap requires the
# source and destination to have the same vertical size, so we first reduce
# each field to one level with a kernel function.

u, v, w = model.velocities
ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)
θ = PotentialTemperature(model)

@inline level(i, j, k, grid, f, kˡ) = @inbounds f[i, j, kˡ]

geographic_grid = latitude_longitude_grid(grid)

function remapped_level(f, kˡ)
    LX, LY, _ = location(f)
    fᵏ = Field(KernelFunctionOperation{LX, LY, Nothing}(level, grid, f, kˡ))
    return remapped_field(fᵏ, geographic_grid)
end

outputs = (θ = remapped_level(θ, 1),
           ζ = remapped_level(ζ, 1),
           w = remapped_level(w, 16))

simulation.output_writers[:slices] = JLD2Writer(model, outputs;
                                                filename = "baroclinic_wave.jld2",
                                                schedule = TimeInterval(6hours),
                                                overwrite_files = true)

# ## Run

run!(simulation)

# ## Visualization
#
# We plot three near-surface diagnostics on the sphere: the **surface
# potential temperature** ``θ_{\rm sfc}`` (the classic diagnostic for the
# cold/warm sectors), the **surface vertical vorticity** ``ζ`` (which reveals
# the cyclones and anticyclones), and the **lower-tropospheric vertical
# velocity** ``w`` at ~2.9 km (the warm conveyor belt).

θ_ts = FieldTimeSeries("baroclinic_wave.jld2", "θ")
ζ_ts = FieldTimeSeries("baroclinic_wave.jld2", "ζ")
w_ts = FieldTimeSeries("baroclinic_wave.jld2", "w")
times = θ_ts.times
Nt = length(times)

# Each field is drawn as a surface on the unit sphere at the nodes of its own
# location, seen from above the developing wave.

function sphere(f)
    LX, LY, _ = location(f)
    λ = λnodes(f.grid, LX())
    φ = φnodes(f.grid, LY())
    x = [cosd(φ) * cosd(λ) for λ in λ, φ in φ]
    y = [cosd(φ) * sind(λ) for λ in λ, φ in φ]
    z = [sind(φ) for λ in λ, φ in φ]
    return x, y, z
end

sphere_kw = (elevation = π/6, azimuth = π/2, aspect = :data)
ζlim = 1e-4
wlim = 0.06

θ_kw = (colormap = :thermal, colorrange = (260, 310))
ζ_kw = (colormap = :balance, colorrange = (-ζlim, ζlim))
w_kw = (colormap = :balance, colorrange = (-wlim, wlim))

# ### Animation

n = Observable(1)
θn = @lift interior(θ_ts[$n], :, :, 1)
ζn = @lift interior(ζ_ts[$n], :, :, 1)
wn = @lift interior(w_ts[$n], :, :, 1)

fig = Figure(size = (1800, 700))

title = @lift "t = $(prettytime(times[$n]))"
fig[0, 1:6] = Label(fig, title, fontsize=22, tellwidth=false)

ax1 = Axis3(fig[1, 1]; title = "θ at surface", sphere_kw...)
hm1 = surface!(ax1, sphere(θ_ts[1])...; color = θn, shading = NoShading, θ_kw...)
Colorbar(fig[1, 2], hm1; label = "θ (K)", height=Relative(0.5))

ax2 = Axis3(fig[1, 3]; title = "ζ at surface", sphere_kw...)
hm2 = surface!(ax2, sphere(ζ_ts[1])...; color = ζn, shading = NoShading, ζ_kw...)
Colorbar(fig[1, 4], hm2; label = "ζ (1/s)", height=Relative(0.5))

ax3 = Axis3(fig[1, 5]; title = "w at 2.9 km", sphere_kw...)
hm3 = surface!(ax3, sphere(w_ts[1])...; color = wn, shading = NoShading, w_kw...)
Colorbar(fig[1, 6], hm3; label = "w (m/s)", height=Relative(0.5))

for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig, "baroclinic_wave.mp4", 1:Nt; framerate = 12, compression = 23) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave.mp4)
