# # Baroclinic wave in a Cartesian channel
#
# The sibling of `baroclinic_wave.jl`: the same dry baroclinic-instability
# life cycle, but on an `f`-plane channel instead of the sphere. The analytic
# balanced jet follows the Ullrich & Jablonowski doubly-periodic channel
# construction (the same background used by `power_method_baroclinic_cartesian.jl`
# to hunt for the wavenumber-9 growth rate) — here we skip the power-method
# rescaling and just let a localized perturbation grow, break, and saturate
# over ~2 weeks.
#
# A periodic-in-``x`` channel is a convenient stand-in for a mid-latitude
# strip of the sphere: no poles to worry about, no gradient-wind subtleties
# — just a zonal jet in thermal-wind balance on an ``f``-plane centered at
# 45°N, perturbed at a single point to seed the instability.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using CUDA

# ## URJ15 parameters
#
# Identical to `power_method_baroclinic_cartesian.jl`: the constants,
# lapse rate, tropopause blending, and jet-width parameters all follow the
# same doubly-periodic channel construction.

Oceananigans.defaults.FloatType = Float64
Oceananigans.defaults.gravitational_acceleration = 9.80616

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ

Ω  = 7.29212e-5
p₀ = 1e5
T₀ = 288
Γ  = 0.005
b  = 2
u₀ = 35
ΔT = 4.8e5
ηₜ = 0.2
κ_T = Rᵈ * Γ / g

φ₀ = π / 4
f₀ = 2Ω * sin(φ₀)

Lx = 40_000kilometers
Ly = 6_000kilometers
Lz = 30kilometers

# ## Domain and grid
#
# A 100 km horizontal mesh (comparable to the 1° production resolution of
# `baroclinic_wave.jl`) with 32 vertical levels up to 30 km.

Nx = 400
Ny = 60
Nz = 32

grid = RectilinearGrid(GPU();
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       topology = (Periodic, Bounded, Bounded))

# ## Analytic initial conditions
#
# The balanced state is given in ``\eta``-coordinates (Ullrich & Jablonowski's
# hybrid pressure-like vertical coordinate) and converted to height via
# Newton iteration. The zonal jet ``\bar u(y, \eta)`` is in thermal-wind
# balance with the meridional temperature gradient ``T'(y, \eta)``.

α_exp = g / (Rᵈ * Γ)
η_mean(z) = (1 - Γ * z / T₀)^α_exp

function T_bar(η)
    T = T₀ * η^κ_T
    return ifelse(η < ηₜ, T + ΔT * (ηₜ - η)^5, T)
end

function urj15_u(y, η)
    s = log(η)
    return -u₀ * sin(π * y / Ly)^2 * s * exp(-(s / b)^2)
end

function T_prime(y, η)
    s = log(η)
    G = (1 - 2 * s^2 / b^2) * exp(-(s / b)^2)
    I = y / 2 - Ly / (4π) * sin(2π * y / Ly) - Ly / 4
    return -(f₀ * u₀ / Rᵈ) * I * G
end

T_full(y, η) = T_bar(η) + T_prime(y, η)

function Φ_bar(η)
    Φ = (g * T₀ / Γ) * (1 - η^κ_T)
    if η < ηₜ
        Φ -= Rᵈ * ΔT * (ηₜ^5 * log(η / ηₜ)
                         - 5 * ηₜ^4 * (η - ηₜ)
                         + 5 * ηₜ^3 * (η^2 - ηₜ^2)
                         - 10/3 * ηₜ^2 * (η^3 - ηₜ^3)
                         + 5/4 * ηₜ * (η^4 - ηₜ^4)
                         - 1/5 * (η^5 - ηₜ^5))
    end
    return Φ
end

function Φ_prime(y, η)
    s = log(η)
    return f₀ * u₀ * (y / 2 - Ly / (4π) * sin(2π * y / Ly) - Ly / 4) * s * exp(-(s / b)^2)
end

Φ_total(y, η) = Φ_bar(η) + Φ_prime(y, η)

function η_at(y, z)
    target = g * z
    η = η_mean(z)
    for _ in 1:10
        Φ = Φ_total(y, η)
        T = T_full(y, η)
        dΦdη = -Rᵈ * T / η
        η = η - (Φ - target) / dΦdη
        η = clamp(η, 1e-8, 1.0)
    end
    return η
end

virtual_temperature(y, z) = T_full(y, η_at(y, z))
pressure(y, z) = p₀ * η_at(y, z)
density(y, z) = pressure(y, z) / (Rᵈ * virtual_temperature(y, z))
potential_temperature(y, z) = virtual_temperature(y, z) * (p₀ / pressure(y, z))^κ
balanced_u(y, z) = urj15_u(y, η_at(y, z))

# ### Perturbation
#
# A localized zonal-wind bump centered on the jet core
# (``x_c = L_x/2``, ``y_c = L_y/2``), decaying as a Gaussian with horizontal
# distance and tapered to zero above 15 km — the Cartesian analog of
# `baroclinic_wave.jl`'s great-circle perturbation, using a physical
# radius (rather than an angular one) for the Gaussian decay.

function zonal_velocity(x, y, z)
    u_balanced = balanced_u(y, z)

    uₚ = 1          # m/s — amplitude
    rₚ = 637000     # m — perturbation radius (matches DCMIP2016's 0.1 Earth radii)
    xₚ = Lx / 2
    yₚ = Ly / 2
    zₚ = 15000      # m — height cap

    r = sqrt((x - xₚ)^2 + (y - yₚ)^2) / rₚ
    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, zero(z))
    u_perturbation = ifelse(r < 1, uₚ * taper * exp(-r^2), zero(z))

    return u_balanced + u_perturbation
end

# ## Model configuration
#
# Fully compressible dynamics with acoustic substepping and an `f`-plane
# Coriolis term fixed at 45°N — the channel analog of `SphericalCoriolis`,
# matching the constant-``f`` assumption baked into the analytic balance
# above (there is no ``\beta`` term in ``T'`` or ``\Phi'``, so the model's
# Coriolis must be constant too). Otherwise this mirrors `baroclinic_wave.jl`:
# the same reference state and the same `WENO(order=9)` advection, with no
# explicit closure — as in the sphere case, ninth-order WENO's own
# dissipation is relied on to keep the grid scale quiet over the run.

coriolis = FPlane(f=f₀)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO(order=9))

# ## Set initial conditions

initial_u(x, y, z) = zonal_velocity(x, y, z)
initial_θ(x, y, z) = potential_temperature(y, z)
initial_ρ(x, y, z) = density(y, z)

set!(model, θ=initial_θ, u=initial_u, ρ=initial_ρ)

# ## Time-stepping
#
# As in `baroclinic_wave.jl`, acoustic substepping frees the outer time
# step from the acoustic CFL; a time-step wizard floats ``\Delta t`` at
# advective CFL ≈ 1.4, capped at 10 min. Twelve days is enough to carry the
# perturbation (``\sim 1`` m/s at ``t=0``) through exponential growth at
# ``\sigma \approx 0.46`` day``^{-1}`` into nonlinear breaking and saturation.

Δt = 10minutes
stop_time = 12days

simulation = Simulation(model; Δt, stop_time)
conjure_time_step_wizard!(simulation; cfl=1.4, max_Δt=10minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Progress callback

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("Iter %5d | t = %s | Δt = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# ## Output
#
# The velocities, potential temperature ``θ``, diagnostic pressure ``p``,
# and vertical vorticity ``ζ``, sliced at the surface (k = 1) and
# mid-troposphere (k = 16, ~5 km) — the same fields and levels as
# `baroclinic_wave.jl`.

using Oceananigans.Operators: ζ₃ᶠᶠᶜ
u, v, w = model.velocities
ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, model.grid, u, v)

θ = PotentialTemperature(model)
p = dynamics_pressure(model.dynamics)

outputs = merge(model.velocities, (; ζ, θ, p))

for k in (1, 16)
    filename = "baroclinic_wave_cartesian_k$k"
    ow = JLD2Writer(model, outputs; filename,
                    indices = (:, :, k),
                    schedule = TimeInterval(6hours),
                    overwrite_existing = true)

    simulation.output_writers[Symbol(filename)] = ow
end

# ## Run

run!(simulation)

# ## Visualization
#
# The same four-panel diagnostic as `baroclinic_wave.jl` — surface ``θ``,
# surface ``ζ``, mid-level ``w``, and surface pressure — but with ``x`` and
# ``y`` in km instead of longitude/latitude.

θ_ts = FieldTimeSeries("baroclinic_wave_cartesian_k1.jld2",  "θ")
ζ_ts = FieldTimeSeries("baroclinic_wave_cartesian_k1.jld2",  "ζ")
p_ts = FieldTimeSeries("baroclinic_wave_cartesian_k1.jld2",  "p")
w_ts = FieldTimeSeries("baroclinic_wave_cartesian_k16.jld2", "w")
times = θ_ts.times
Nt = length(times)

k_sfc = 1
k_mid = 16

ζlim = 1e-4
wlim = 0.06

p_ctr  = 1e5      # Pa — reference pressure (1000 hPa) for diverging colormap
p_half = 2000.0   # Pa — ±20 hPa half-range resolves synoptic highs and lows

θ_kw = (colormap = :thermal,  colorrange = (260, 310))
ζ_kw = (colormap = :balance,  colorrange = (-ζlim, ζlim))
w_kw = (colormap = :balance,  colorrange = (-wlim, wlim))
p_kw = (colormap = Reverse(:RdBu), colorrange = (p_ctr - p_half, p_ctr + p_half),
        lowclip = :darkblue, highclip = :darkred)

# ### Animation

n = Observable(1)
θn  = @lift view(θ_ts[$n], :, :, k_sfc)
ζn  = @lift view(ζ_ts[$n], :, :, k_sfc)
wn  = @lift view(w_ts[$n], :, :, k_mid)
pn  = @lift view(p_ts[$n], :, :, k_sfc)

fig = Figure(size = (2400, 700))

title = @lift "t = $(prettytime(times[$n]))"
fig[0, 1:8] = Label(fig, title, fontsize=22, tellwidth=false)

km_ticks = xs -> string.(round.(Int, xs ./ 1000))
ax_kw = (xlabel = "x (km)", ylabel = "y (km)", xtickformat = km_ticks, ytickformat = km_ticks)

ax1 = Axis(fig[1, 1]; title = "θ at surface", ax_kw...)
hm1 = heatmap!(ax1, θn; θ_kw...)
Colorbar(fig[1, 2], hm1; label = "θ (K)", height=Relative(0.5))

ax2 = Axis(fig[1, 3]; title = "ζ at surface", ax_kw...)
hm2 = heatmap!(ax2, ζn; ζ_kw...)
Colorbar(fig[1, 4], hm2; label = "ζ (1/s)", height=Relative(0.5))

ax3 = Axis(fig[1, 5]; title = "w at mid-level", ax_kw...)
hm3 = heatmap!(ax3, wn; w_kw...)
Colorbar(fig[1, 6], hm3; label = "w (m/s)", height=Relative(0.5))

ax4 = Axis(fig[1, 7]; title = "p at surface", ax_kw...)
hm4 = heatmap!(ax4, pn; p_kw...)
Colorbar(fig[1, 8], hm4; label = "p (Pa)", height=Relative(0.5))

CairoMakie.record(fig, "baroclinic_wave_cartesian.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](baroclinic_wave_cartesian.mp4)
