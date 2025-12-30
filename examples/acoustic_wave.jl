# # Acoustic Wave Refraction by Wind Shear
#
# This example demonstrates how wind shear refracts acoustic waves using
# the fully compressible Euler equations. When wind speed increases with height,
# sound waves traveling **with** the wind are bent **downward** (trapped near surface),
# while sound traveling **against** the wind is bent **upward** (away from surface).
#
# The effective sound speed for a wave traveling in direction ``\hat{n}`` is
# ```math
# c_{\rm eff} = c_s + \mathbf{u} \cdot \hat{n}
# ```
#
# We use a stable stratification to suppress Kelvin-Helmholtz instability,
# and a logarithmic wind profile consistent with the atmospheric surface layer.

using Oceananigans
using Oceananigans.Grids: xnodes, znodes
using Oceananigans.Units
using Breeze
using Statistics: mean
using CairoMakie

# ## Model Setup

Nx = 256
Nz = 48
Lx = 4000  # m
Lz = 300   # m

grid = RectilinearGrid(CPU(),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

dynamics = CompressibleDynamics()
model = AtmosphereModel(grid; dynamics)
constants = model.thermodynamic_constants

# ## Initial Conditions

# Thermodynamic parameters
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
κ = Rᵈ / cᵖᵈ
γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
g = constants.gravitational_acceleration
p₀ = 1e5  # Standard pressure

# Stratification: θ increases with height for stability
θ_surface = 300.0  # K
Γ = 0.02           # K/m (stable lapse rate)
θ_profile(z) = θ_surface + Γ * z

# Log-layer wind profile: u(z) = (u*/κ) × ln((z + z₀)/z₀)
u_star = 0.4       # Friction velocity (m/s) - moderate shear
κ_vk = 0.4         # von Kármán constant
z_roughness = 0.1  # Roughness length (m)

U_profile(z) = (u_star / κ_vk) * log((z + z_roughness) / z_roughness)
U_max = U_profile(Lz)

# Sound speed at surface
cₛ = sqrt(γ * Rᵈ * θ_surface)

# Compute hydrostatic density profile numerically
function hydrostatic_density_profile(z_nodes, p_surface, θ_func)
    Nz = length(z_nodes)
    p = zeros(Nz)
    ρ = zeros(Nz)
    
    p[1] = p_surface
    θ₁ = θ_func(z_nodes[1])
    ρ[1] = p[1]^(1 - κ) * p₀^κ / (Rᵈ * θ₁)
    
    for k in 2:Nz
        θₖ = θ_func(z_nodes[k])
        Δz = z_nodes[k] - z_nodes[k-1]
        p[k] = p[k-1] - ρ[k-1] * g * Δz
        ρ[k] = p[k]^(1 - κ) * p₀^κ / (Rᵈ * θₖ)
    end
    
    return ρ, p
end

z_nodes = znodes(grid, Center())
ρ_profile, p_profile = hydrostatic_density_profile(z_nodes, 101325.0, θ_profile)

function ρ_hydrostatic(z)
    idx = searchsortedlast(z_nodes, z)
    idx = clamp(idx, 1, length(z_nodes)-1)
    z₁, z₂ = z_nodes[idx], z_nodes[idx+1]
    ρ₁, ρ₂ = ρ_profile[idx], ρ_profile[idx+1]
    t = (z - z₁) / (z₂ - z₁)
    return ρ₁ + t * (ρ₂ - ρ₁)
end

# Acoustic pulse parameters - centered in left half of domain
δρ = 0.01    # Density perturbation amplitude (kg/m³)
x₀ = Lx / 3  # ~1333m from left edge
z₀ = 100     # 100m above ground
σ = 50       # Pulse width (m)

# Gaussian shape function
gaussian(x, z) = exp(-((x - x₀)^2 + (z - z₀)^2) / (2σ^2))

# Surface density for velocity scaling
ρ_surface = ρ_profile[1]

# Initial conditions
# For rightward propagation: u' = +(cₛ/ρ₀) × ρ'
ρᵢ(x, z) = ρ_hydrostatic(z) + δρ * gaussian(x, z)
θᵢ(x, z) = θ_profile(z)
uᵢ(x, z) = U_profile(z) + (cₛ / ρ_surface) * δρ * gaussian(x, z)

set!(model, ρ=ρᵢ, θ=θᵢ, u=uᵢ)

@info "Initial state:"
@info "  ρ min/max: $(extrema(model.dynamics.density))"
@info "  θ min/max: $(extrema(model.formulation.potential_temperature))"
@info "  u min/max: $(extrema(model.velocities.u))"
@info "  Sound speed: $(round(cₛ, digits=1)) m/s"
@info "  Max wind (log profile): $(round(U_max, digits=1)) m/s"

# Compute backgrounds for visualization (using correct grid locations)
ρ_background = zeros(Nx, Nz)
for (ii, x) in enumerate(xnodes(model.dynamics.density))
    for (kk, z) in enumerate(znodes(model.dynamics.density))
        ρ_background[ii, kk] = ρ_hydrostatic(z)
    end
end

# u is at Face in x, so use Face x-nodes for background
u_background = zeros(Nx, Nz)
for (ii, x) in enumerate(xnodes(model.velocities.u))
    for (kk, z) in enumerate(znodes(model.velocities.u))
        u_background[ii, kk] = U_profile(z)
    end
end

# ## Time Stepping

Δx = Lx / Nx
Δz = Lz / Nz

c_eff = cₛ + U_max
Δt = 0.02 * min(Δx, Δz) / c_eff  # Very conservative CFL

stop_time = 10.0  # seconds - run longer to see boundary behavior

simulation = Simulation(model; Δt, stop_time)

@info "Time step: $(round(Δt * 1000, digits=3)) ms"
@info "Expected wave travel: $(round(cₛ * stop_time, digits=0)) m"

# ## Save output

ρ = model.dynamics.density
u = model.velocities.u

ρ_ts = []
u_ts = []
times = Float64[]

Nt = 200

function save_fields!(sim)
    push!(ρ_ts, deepcopy(interior(ρ, :, 1, :)))
    push!(u_ts, deepcopy(interior(u, :, 1, :)))
    push!(times, sim.model.clock.time)
    return nothing
end

save_fields!(simulation)

# Check boundary continuity at initial condition
@info "Boundary check at t=0:"
@info "  u[1, 1, 20] = $(u[1, 1, 20]),  u[Nx, 1, 20] = $(u[Nx, 1, 20])"
@info "  ρ[1, 1, 20] = $(ρ[1, 1, 20]),  ρ[Nx, 1, 20] = $(ρ[Nx, 1, 20])"

save_interval = stop_time / Nt
simulation.callbacks[:save] = Callback(save_fields!, IterationInterval(max(1, round(Int, save_interval / Δt))))

run!(simulation)

@info "Simulation completed at t = $(prettytime(model.clock.time))"
@info "  ρ min/max: $(extrema(model.dynamics.density))"
@info "  u min/max: $(extrema(model.velocities.u))"

# ## Create Animation

@info "Creating animation..."

# Use correct coordinates for each field
x_ρ = xnodes(ρ)  # Center in x
z_ρ = znodes(ρ)
x_u = xnodes(u)  # Face in x
z_u = znodes(u)

Nframes = length(times)

fig = Figure(size=(1000, 700), fontsize=12)

# Density perturbation plot
ax_ρ = Axis(fig[2, 1],
            ylabel = "z (m)",
            title = "Density perturbation ρ′",
            xticklabelsvisible = false)

# Density profile panel
ax_ρ_profile = Axis(fig[2, 2],
                    xlabel = "⟨ρ′⟩ × 10³",
                    yticklabelsvisible = false,
                    width = 80)

# Velocity perturbation plot
ax_u = Axis(fig[3, 1],
            xlabel = "x (m)",
            ylabel = "z (m)",
            title = "Velocity perturbation u′")

# Velocity profile panel (shows mean u, not perturbation)
ax_u_profile = Axis(fig[3, 2],
                    xlabel = "⟨u⟩ (m/s)",
                    yticklabelsvisible = false,
                    width = 80)

linkyaxes!(ax_ρ, ax_ρ_profile)
linkyaxes!(ax_u, ax_u_profile)

# Color limits
ρ_lim = δρ / 2
u′_lim = 1.5  # m/s for velocity perturbation

# Observables - both showing perturbations
ρ′_data = Observable(ρ_ts[1] .- ρ_background)
u′_data = Observable(u_ts[1] .- u_background)

hm1 = heatmap!(ax_ρ, x_ρ, z_ρ, ρ′_data,
               colormap = :balance, colorrange = (-ρ_lim, ρ_lim))
Colorbar(fig[2, 3], hm1, label = "ρ′ (kg/m³)")

hm2 = heatmap!(ax_u, x_u, z_u, u′_data,
               colormap = :balance, colorrange = (-u′_lim, u′_lim))
Colorbar(fig[3, 3], hm2, label = "u′ (m/s)")

# Profile lines - show background wind profile (log layer)
u_bg = [U_profile(zz) for zz in z_u]

ρ′_mean_line = Observable(Point2f.(zeros(Nz), z_ρ))
u_mean_line = Observable(Point2f.(u_bg, z_u))

lines!(ax_ρ_profile, ρ′_mean_line, color = :black, linewidth = 2)
vlines!(ax_ρ_profile, [0], color = :gray, linestyle = :dash)

lines!(ax_u_profile, u_mean_line, color = :dodgerblue, linewidth = 2, label = "⟨u⟩")
lines!(ax_u_profile, u_bg, z_u, color = :gray, linestyle = :dash, linewidth = 1, label = "log layer")

xlims!(ax_ρ_profile, (-ρ_lim * 1000, ρ_lim * 1000))
xlims!(ax_u_profile, (0, U_max + 2))

title_text = Observable("Acoustic wave with log-layer shear: t = 0.00 s")
Label(fig[1, :], title_text, fontsize=16, tellwidth=false)

colsize!(fig.layout, 1, Relative(0.7))
colsize!(fig.layout, 2, Auto())
colsize!(fig.layout, 3, Auto())

@info "Recording $(Nframes) frames..."

record(fig, "acoustic_wave.mp4", 1:Nframes; framerate=20) do n
    ρ′_data[] = ρ_ts[n] .- ρ_background
    u′_data[] = u_ts[n] .- u_background
    
    ρ′_mean = vec(mean(ρ_ts[n] .- ρ_background, dims=1)) .* 1000
    u_mean = vec(mean(u_ts[n], dims=1))
    
    ρ′_mean_line[] = Point2f.(ρ′_mean, z_ρ)
    u_mean_line[] = Point2f.(u_mean, z_u)
    
    title_text[] = "Acoustic wave with log-layer shear: t = $(round(times[n], digits=2)) s"
end

@info "Animation saved to acoustic_wave.mp4"

save("acoustic_wave.png", fig)
@info "Final frame saved to acoustic_wave.png"
