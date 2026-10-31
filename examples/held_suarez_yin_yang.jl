# # Held–Suarez on the Yin-Yang grid
#
# This example simulates the dry general circulation of [Held and Suarez (1994)](@cite HeldSuarez1994)
# on the Yin-Yang grid of [Kageyama and Sato (2004)](@cite KageyamaSato2004). The Held–Suarez benchmark
# replaces every physical parameterization with two simple forcings: Rayleigh friction in a
# boundary layer below ``σ = p / pˢ = 0.7``, and Newtonian relaxation of temperature toward a
# zonally symmetric radiative equilibrium ``Tᵉ(φ, p)``. After a few months of spin-up the result
# is an Earth-like climate, with midlatitude jets driven by baroclinic eddies.
#
# A latitude–longitude grid has a pole problem: its meridians converge, so the polar cells are
# tiny and set the time step. The Yin-Yang grid covers the sphere with two identical pieces of a
# latitude–longitude grid, each spanning 270° of longitude and 90° of latitude around its own
# equator. The "Yin" panel is the low-latitude band of the geographic grid; the "Yang" panel is
# the same band rotated so that its equator runs through the geographic poles. Neither panel comes
# near its own poles, so the grid spacing is nearly uniform. The panels overlap slightly, and the
# halo of each panel is interpolated from the other, with horizontal vectors rotated between the
# panels' bases.
#
# Because the forcing is zonally symmetric, any departure of the time-mean climate from zonal
# symmetry is either sampling noise or an imprint of the grid. We look for one at the end.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Coriolis: TriadScheme
using Oceananigans.MultiRegion: remapped_field
using Printf
using CairoMakie
using CUDA

Oceananigans.defaults.FloatType = Float32

# ## Grid
#
# Each panel spans 270° × 90° with 2° cells, and 20 levels reach up to 30 km, stretched toward the
# surface.

Nz = 20
H = 30kilometers
z = ExponentialDiscretization(Nz, 0, H; scale=H/2, bias=:left)
grid = YinYangGrid(GPU(); size=(135, 45, Nz), halo=(5, 5, 5), z)

# ## Model
#
# We use fully compressible dynamics with acoustic substepping and fifth-order WENO advection.
# The Held–Suarez forcing depends only on latitude and pressure, and every node of either panel knows
# its geographic latitude, so the forcing works unchanged on the Yin-Yang grid.
#
# The Coriolis force uses the triad scheme of [Arakawa and Lamb (1981)](@cite ArakawaLamb1981).
# The axes of the Yang panel are rotated from east and north, so the Coriolis parameter varies
# along both of its grid directions. The triads weight it isotropically, whereas the default
# 4-point averaging averages it along one direction only, and so treats the two panels differently.
#
# The interpolation between the panels does not conserve mass: each panel steps the cells it owns
# with its own estimate of the flux through the curve that divides the sphere between the panels,
# and the two estimates differ. Below, `add_conservation_fixer!` adds the callback that evaluates
# the mass each stage adds through this mismatch from the model's own fluxes and removes it. We
# check the mass budget at the end.

constants = ThermodynamicConstants()
g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

T₀ = 250
θᵣ(z) = T₀ * exp(g * z / (cᵖᵈ * T₀))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=θᵣ)
coriolis = SphericalCoriolis(scheme=TriadScheme())
forcing = held_suarez_forcings()

model = AtmosphereModel(grid; dynamics, coriolis, forcing,
                        advection = WENO(order=5),
                        thermodynamic_constants = constants)

# ## Initial condition
#
# The atmosphere starts at rest with the reference potential temperature ``θᵣ(z)`` plus a little
# noise to break the symmetry. `HydrostaticallyBalancedDensity` integrates the density of each
# column into discrete hydrostatic balance with it.

θᵢ(λ, φ, z) = θᵣ(z) + 0.1 * (rand() - 1/2)

set!(model, θ=θᵢ, ρ=HydrostaticallyBalancedDensity())

# ## Simulation
#
# We run for 300 days with an advective CFL of 0.7. The Yin-Yang grid has no converging meridians,
# so the time step is not limited by tiny polar cells.

stop_time = 300days
simulation = Simulation(model; Δt=20minutes, stop_time)
add_conservation_fixer!(simulation)
conjure_time_step_wizard!(simulation; cfl=0.7, max_Δt=30minutes)

function progress(sim)
    u, v, w = sim.model.velocities
    T = sim.model.temperature
    @info @sprintf("Iter %6d | t = %s | Δt = %s | max|u| = %.1f m/s | max|w| = %.3f m/s | T ∈ [%.1f, %.1f] K | wall time = %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, w), minimum(T), maximum(T), prettytime(sim.run_wall_time))
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

# ## Output
#
# The velocities are stored in each panel's own basis, so we remap them to geographic (eastward,
# northward) components on a global latitude–longitude grid and take zonal means there, together
# with the temperature and the pressure. The zonal means are averaged over the last 100 days.

u, v, w = model.velocities
T = model.temperature
p = dynamics_pressure(model.dynamics)
ρ = total_density(model.dynamics)

uᵍ, vᵍ = remapped_field((u, v))
Tᵍ = remapped_field(T)
pᵍ = remapped_field(p)

zonal_means = (u = Field(Average(uᵍ, dims=1)),
               T = Field(Average(Tᵍ, dims=1)),
               p = Field(Average(pᵍ, dims=1)))

# We also want maps of the lowest level: the temperature every day, to watch the weather, and the
# time-mean pressure, to look for an imprint of the grid. We extract the level on each panel and
# remap the two-dimensional result.

@inline lowest_level(i, j, k, grid, f) = @inbounds f[i, j, 1]

Tˢ = remapped_field(Field(KernelFunctionOperation{Center, Center, Nothing}(lowest_level, grid, T)))
pˢ = remapped_field(Field(KernelFunctionOperation{Center, Center, Nothing}(lowest_level, grid, p)))

simulation.output_writers[:time_mean] = JLD2Writer(model, (; zonal_means..., pˢ);
                                                   schedule = AveragedTimeInterval(stop_time, window=100days),
                                                   filename = "held_suarez_yin_yang_time_mean.jld2",
                                                   overwrite_files = true)

simulation.output_writers[:surface] = JLD2Writer(model, (; T = Tˢ);
                                                 schedule = TimeInterval(1day),
                                                 filename = "held_suarez_yin_yang_surface.jld2",
                                                 overwrite_files = true)

# Integrals on a Yin-Yang grid weight each cell by the fraction of its area that its panel owns, so
# that every point of the sphere counts once. We save the total mass ``M = ∫ ρ \, dV`` every day.

simulation.output_writers[:mass] = JLD2Writer(model, (; M = Field(Integral(ρ)));
                                              schedule = TimeInterval(1day),
                                              filename = "held_suarez_yin_yang_mass.jld2",
                                              overwrite_files = true)

run!(simulation)

# ## Three hundred days of weather
#
# The temperature at the lowest level starts out uniform. The relaxation toward ``Tᵉ`` builds an
# equator-to-pole contrast within days, baroclinic eddies grow on it within weeks, and from then on
# a procession of cyclones and anticyclones carries heat poleward. The curve that divides the
# sphere between the panels runs between 38° and 45° of latitude in both hemispheres and crosses
# the poles along the meridians 135°E and 135°W; nothing in the movie marks it.

Tˢ_ts = FieldTimeSeries("held_suarez_yin_yang_surface.jld2", "T")
times = Tˢ_ts.times

n = Observable(1)
Tˢn = @lift Tˢ_ts[$n]
title = @lift "Temperature at the lowest level, t = " * prettytime(times[$n])

fig = Figure(size=(1000, 560))
ax = Axis(fig[1, 1]; xlabel="Longitude (°)", ylabel="Latitude (°)", title, aspect=2, xticks=-180:60:180)
hm = heatmap!(ax, Tˢn; colormap=:magma, colorrange=(240, 310))
Colorbar(fig[1, 2], hm; label="T (K)")

CairoMakie.record(fig, "held_suarez_yin_yang.mp4", 1:length(times); framerate=15) do nn
    n[] = nn
end
nothing #hide

# ![](held_suarez_yin_yang.mp4)
#
# ## The time-mean climate
#
# The time-mean zonal wind has westerly jets of about 30 m/s near 45° in both hemispheres, centered
# on the 250 hPa isobar, and easterlies near the surface in the tropics. The temperature falls from
# the equator to the poles in the troposphere and is nearly uniform above it.

ū_ts = FieldTimeSeries("held_suarez_yin_yang_time_mean.jld2", "u")
T̄_ts = FieldTimeSeries("held_suarez_yin_yang_time_mean.jld2", "T")
p̄_ts = FieldTimeSeries("held_suarez_yin_yang_time_mean.jld2", "p")
p̄ˢ_ts = FieldTimeSeries("held_suarez_yin_yang_time_mean.jld2", "pˢ")

ū = view(ū_ts[end], 1, :, :)
T̄ = view(T̄_ts[end], 1, :, :)
p̄ = view(p̄_ts[end], 1, :, :)

φ = φnodes(ū_ts.grid, Center())
z = znodes(ū_ts.grid, Center()) ./ 1e3

isobars = (levels = [250e2, 500e2, 850e2], color = :black, labels = true,
           labelformatter = p -> @sprintf("%d hPa", p / 100))

fig = Figure(size=(1200, 450))

ax_u = Axis(fig[1, 1]; xlabel="Latitude (°)", ylabel="z (km)", title="Zonal-mean zonal wind, days 200–300")
hm_u = heatmap!(ax_u, φ, z, ū; colormap=:balance, colorrange=(-40, 40))
contour!(ax_u, φ, z, p̄; isobars...)
Colorbar(fig[1, 2], hm_u; label="u (m/s)")

ax_T = Axis(fig[1, 3]; xlabel="Latitude (°)", ylabel="z (km)", title="Zonal-mean temperature, days 200–300")
hm_T = heatmap!(ax_T, φ, z, T̄; colormap=:magma, colorrange=(190, 310))
contour!(ax_T, φ, z, p̄; isobars...)
Colorbar(fig[1, 4], hm_T; label="T (K)")

save("held_suarez_yin_yang_zonal_means.png", fig)
nothing #hide

# ![](held_suarez_yin_yang_zonal_means.png)
#
# ## Looking for the grid
#
# The departure of the time-mean pressure at the lowest level from its zonal mean would vanish in
# a perfectly converged, zonally symmetric climate. What remains is sampling noise, a few hundred
# pascals of it after a 100-day average, plus any imprint of the grid. The largest departures sit
# in the storm tracks, where the pressure varies most from day to day, and none of them follows
# the edges of the panels or the curve between them.

p̄ˢ = p̄ˢ_ts[end]
p̄ˢ′ = Field(p̄ˢ - Field(Average(p̄ˢ, dims=1)))

fig = Figure(size=(1000, 560))
ax = Axis(fig[1, 1]; xlabel="Longitude (°)", ylabel="Latitude (°)", aspect=2, xticks=-180:60:180,
          title="Time-mean pressure at the lowest level minus its zonal mean, days 200–300")
hm = heatmap!(ax, p̄ˢ′; colormap=:balance, colorrange=(-400, 400))
Colorbar(fig[1, 2], hm; label="p′ (Pa)")

save("held_suarez_yin_yang_imprint.png", fig)
nothing #hide

# ![](held_suarez_yin_yang_imprint.png)
#
# Finally, the mass budget. The forcing neither adds nor removes mass, and the conservation fixer removes
# what the coupling between the panels adds, so the total mass stays within a couple of `Float32`
# rounding errors of its initial value for the whole run.

M_ts = FieldTimeSeries("held_suarez_yin_yang_mass.jld2", "M")
M = M_ts[1, 1, 1, :]

fig = Figure(size=(600, 300))
ax = Axis(fig[1, 1]; xlabel="Time (days)", ylabel="M / M(0) − 1", title="Total mass")
lines!(ax, M_ts.times ./ day, M ./ M[1] .- 1)

save("held_suarez_yin_yang_mass.png", fig)
nothing #hide

# ![](held_suarez_yin_yang_mass.png)
