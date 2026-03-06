# # Baroclinic wave on the sphere
#
# This example simulates the growth of a baroclinic wave on a near-global
# `LatitudeLongitudeGrid`, inspired by the dynamical core benchmark described
# by [JablonowskiWilliamson2006](@citet).
# A midlatitude jet in thermal wind balance with a meridional temperature
# gradient is seeded with a localized perturbation that triggers baroclinic
# instability, producing growing Rossby waves over roughly ten days.
#
# This version demonstrates **Reactant compilation** and **Enzyme
# differentiation**: the forward simulation is compiled to XLA/StableHLO
# via `Reactant.@compile`, and the adjoint sensitivity ``∂J/∂θ₀`` is
# computed via Enzyme reverse-mode AD through the compiled time stepper.
#
# ## Physical setup
#
# The background atmosphere is stably stratified with a constant Brunt-Väisälä
# frequency ``N``, giving a potential-temperature profile
#
# ```math
# θ^{\rm b}(z) = θ_0 \exp \left( \frac{N^2 z}{g} \right)
# ```
#
# with ``θ_0 = 300\,{\rm K}`` and ``N^2 = 10^{-4}\,{\rm s^{-2}}``.
#
# ### Meridional temperature gradient
#
# A pole-to-equator temperature difference ``Δθ_{\rm ep} = 60\,{\rm K}``
# drives the baroclinic instability. The temperature gradient is confined
# to the troposphere (below the tropopause height ``z_T = 15\,{\rm km}``):
#
# ```math
# θ(φ, z) = θ^{\rm b}(z) - Δθ_{\rm ep} \sin φ \max(0, 1 - z/z_T)
# ```
#
# ### Balanced zonal jet
#
# The zonal wind is derived from the meridional temperature gradient
# via thermal wind balance:
#
# ```math
# u(φ, z) = \frac{g\, Δθ_{\rm ep}}{a\, θ_0\, Ω}\, \cos φ
#            \times \begin{cases}
#              \dfrac{z}{2} \left( 2 - \dfrac{z}{z_T} \right) & z \le z_T \\[6pt]
#              \dfrac{z_T}{2} & z > z_T
#            \end{cases}
# ```
#
# ### Perturbation
#
# A localized potential-temperature Gaussian bump centered at
# ``(λ_c, φ_c) = (90°, 45°)`` seeds the instability:
#
# ```math
# θ'(λ, φ, z) = Δθ \exp \left[ -\frac{(λ - λ_c)^2 + (φ - φ_c)^2}{2σ^2} \right]
#               \sin \left( \frac{π z}{H} \right)
# ```
#
# ## Differentiability
#
# The loss function is the mean squared meridional velocity over the
# full domain — a proxy for eddy kinetic energy. Since ``v = 0``
# in the balanced initial state, all meridional motion comes from the
# growing instability. The sensitivity ``∂J/∂θ₀`` reveals which
# initial temperature perturbations most efficiently amplify baroclinic
# growth.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Printf
using CairoMakie
using CUDA

# ## Domain and grid
#
# Coarse resolution for fast Reactant compilation.
# Increase `Nλ`, `Nφ`, `Nz` for physical runs.

Nλ = 90
Nφ = 40
Nz = 20
H  = 30kilometers

grid = LatitudeLongitudeGrid(ReactantState();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-85, 85),
                             z = (0, H))

FT = eltype(grid)

# ## Physical parameters``

constants = ThermodynamicConstants()
g  = constants.gravitational_acceleration
p₀ = 100000  # Pa — surface pressure
θ₀ = 300     # K — surface potential temperature
N² = 1e-4    # s⁻² — Brunt-Väisälä frequency squared

θᵇ(z) = θ₀ * exp(N² * z / g)

# ## Model configuration
#
# Two separate models on the same grid:
# - `model_vis` runs forward-only for the GIF (cheap, many steps)
# - `model_ad`  runs forward+backward for sensitivity (fewer steps to fit in memory)

coriolis = HydrostaticSphericalCoriolis()

make_dynamics() = CompressibleDynamics(ExplicitTimeStepping();
                                       surface_pressure = p₀,
                                       reference_potential_temperature = θᵇ)

model_vis = AtmosphereModel(grid; dynamics = make_dynamics(), coriolis)
model_ad  = AtmosphereModel(grid; dynamics = make_dynamics(), coriolis)

# ## Initial conditions

Ω     = coriolis.rotation_rate
R     = Oceananigans.defaults.planet_radius
Δθ_ep = 60      # K — equator-to-pole θ difference
z_T   = 15_000   # m — tropopause height
τ_bal = R * θ₀ * Ω / (g * Δθ_ep)

# Perturbation parameters:
λ_c = 90   # degrees — perturbation centre longitude
φ_c = 45   # degrees — perturbation centre latitude
σ   = 10   # degrees — Gaussian half-width
Δθ  = 1    # K — perturbation amplitude

# ### Balanced zonal wind from thermal wind relation

function uᵢ(λ, φ, z)
    vertical_scale = ifelse(z ≤ z_T, z / 2 * (2 - z / z_T), z_T / 2)
    return (vertical_scale / τ_bal) * cosd(φ)
end

# ### Potential temperature: background + meridional gradient + perturbation

function θᵢ(λ, φ, z)
    θ_merid = -Δθ_ep * sind(φ) * max(0, 1 - z / z_T)
    r² = (λ - λ_c)^2 + (φ - φ_c)^2
    θ_pert = Δθ * exp(-r² / (2σ^2)) * sin(π * z / H)
    return θᵇ(z) + θ_merid + θ_pert
end

# ### Hydrostatic density
#
# Integrate the Exner function from the surface to height ``z``,
# then recover ``ρ = p_0 Π^{c_v/R^d} / (R^d θ)``.

Rᵈ = dry_air_gas_constant(constants)
cᵖ = constants.dry_air.heat_capacity
κ  = Rᵈ / cᵖ
cᵥ_over_Rᵈ = (cᵖ - Rᵈ) / Rᵈ

function ρᵢ(λ, φ, z)
    nsteps = max(1, round(Int, z / 100))
    dz = z / nsteps
    Π = 1.0
    for n in 1:nsteps
        zn = (n - 1/2) * dz
        θn = θᵢ(λ, φ, zn)
        Π -= κ * g / (Rᵈ * θn) * dz
    end
    θ = θᵢ(λ, φ, z)
    return p₀ * Π^cᵥ_over_Rᵈ / (Rᵈ * θ)
end

# ### Set model state
#
# Density must be set before velocity so that momentum ``ρu`` is
# computed correctly inside `set_velocity!`.  The expensive ``ρᵢ``
# evaluation is done once on `model_vis`; the density array is then
# copied to `model_ad` to avoid recomputing the numerical integration.

@info "Setting initial conditions (vis model)…"
@time begin
    set!(model_vis; ρ = ρᵢ)
    set!(model_vis; u = uᵢ, θ = θᵢ)
end

@info "Copying initial conditions to AD model…"
@time begin
    parent(model_ad.dynamics.density) .= parent(model_vis.dynamics.density)
    set!(model_ad; u = uᵢ, θ = θᵢ)
end

# ## Time step

γ  = cᵖ / (cᵖ - Rᵈ)
cₛ = sqrt(γ * Rᵈ * θ₀)
Δz = H / Nz
Δt = FT(0.4 * Δz / cₛ)

@info "Time step" Δt cₛ

# ## Visualisation forward run
#
# Step `model_vis` forward and capture mid-level snapshots for a sphere GIF.

nsteps_vis = 2500 * 16
sample_interval = max(1, nsteps_vis ÷ 100)

Δλ = 360.0 / Nλ
Δφ = 170.0 / Nφ
λ = range(Δλ / 2, 360 - Δλ / 2, length = Nλ)
φ = range(-85 + Δφ / 2, 85 - Δφ / 2, length = Nφ)
z = range(Δz / 2, H - Δz / 2, length = Nz)

k_mid  = Nz ÷ 2
z_mid  = z[k_mid]

λrad = deg2rad.(collect(λ))
φrad = deg2rad.(collect(φ))

xs = [cos(ϕv) * cos(λv) for λv in λrad, ϕv in φrad]
ys = [cos(ϕv) * sin(λv) for λv in λrad, ϕv in φrad]
zs = [sin(ϕv) for λv in λrad, ϕv in φrad]

θ_bg_mid = Float64(θᵇ(z_mid))

θ_frames = Matrix{Float64}[]
u_frames = Matrix{Float64}[]
vis_times = Float64[]

function capture_frame!(model, θ_bg_mid, k_mid)
    θ_mid = Array(@view interior(model.formulation.potential_temperature)[:, :, k_mid])
    u_mid = Array(@view interior(model.velocities.u)[:, :, k_mid])
    push!(θ_frames, θ_mid .- θ_bg_mid)
    push!(u_frames, u_mid)
    push!(vis_times, Float64(model.clock.time))
    return nothing
end

function advance_model!(model, Δt, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return nothing
end

@info "Compiling visualisation stepping kernel…"
@time compiled_vis = Reactant.@compile raise=true raise_first=true sync=true advance_model!(
    model_vis, Δt, sample_interval)

nframes = nsteps_vis ÷ sample_interval
capture_frame!(model_vis, θ_bg_mid, k_mid)
@info "Running visualisation forward pass ($nsteps_vis steps, $nframes frames)…"
@time for _ in 1:nframes
    compiled_vis(model_vis, Δt, sample_interval)
    capture_frame!(model_vis, θ_bg_mid, k_mid)
end

Nt = length(vis_times)

# ## Visualisation outputs (before AD)
#
# Save visualization products first, then free visualization memory before AD.

sphere_kw = (elevation = π / 6, azimuth = -π / 2, aspect = :data)

# Sphere snapshot (final frame, forward-only fields).
fig_sphere = Figure(size = (1200, 600))

ax1 = Axis3(fig_sphere[1, 1];
            title = "θ′ at z ≈ $(Int(round(z_mid / 1e3))) km",
            sphere_kw...)
hm1 = surface!(ax1, xs, ys, zs; color = θ_frames[Nt], colormap = :balance, shading = NoShading)
Colorbar(fig_sphere[1, 2], hm1; label = "θ′ (K)")

ax2 = Axis3(fig_sphere[1, 3];
            title = "u at z ≈ $(Int(round(z_mid / 1e3))) km",
            sphere_kw...)
hm2 = surface!(ax2, xs, ys, zs; color = u_frames[Nt], colormap = :speed, shading = NoShading)
Colorbar(fig_sphere[1, 4], hm2; label = "u (m/s)")

for ax in (ax1, ax2)
    hidedecorations!(ax)
    hidespines!(ax)
end

Label(fig_sphere[0, :],
      @sprintf("Baroclinic wave — forward-only snapshot, t = %s",
               prettytime(vis_times[Nt])),
      fontsize = 18, tellwidth = false)

save("baroclinic_wave_sphere.png", fig_sphere; px_per_unit = 2)
@info "Saved baroclinic_wave_sphere.png"

# Sphere MP4 (evolving θ′ and u from the visualisation model).
n = Observable(1)
θ′n = @lift θ_frames[$n]
un  = @lift u_frames[$n]
ttl = @lift @sprintf("z ≈ %d km,  t = %s",
                     round(z_mid / 1e3), prettytime(vis_times[$n]))

θlim = max(maximum(x -> maximum(abs, x), θ_frames), eps(Float64))
ulim = max(maximum(x -> maximum(abs, x), u_frames), eps(Float64))

fig_anim = Figure(size = (1200, 600))
axg1 = Axis3(fig_anim[1, 1]; title = "θ′", sphere_kw...)
hmg1 = surface!(axg1, xs, ys, zs; color = θ′n, colormap = :balance,
                colorrange = (-θlim, θlim), shading = NoShading)
Colorbar(fig_anim[1, 2], hmg1; label = "θ′ (K)")

axg2 = Axis3(fig_anim[1, 3]; title = "u", sphere_kw...)
hmg2 = surface!(axg2, xs, ys, zs; color = un, colormap = :balance,
                colorrange = (-ulim, ulim), shading = NoShading)
Colorbar(fig_anim[1, 4], hmg2; label = "u (m/s)")

fig_anim[0, :] = Label(fig_anim, ttl, fontsize = 20, tellwidth = false)

for ax in (axg1, axg2)
    hidedecorations!(ax)
    hidespines!(ax)
end

@info "Recording MP4 ($Nt frames)…"
CairoMakie.record(fig_anim, "baroclinic_wave_sphere.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
@info "Saved baroclinic_wave_sphere.mp4"

# Release visualization memory before AD.
@info "Releasing visualization memory before AD…"
θ_frames = nothing
u_frames = nothing
vis_times = nothing
xs = nothing
ys = nothing
zs = nothing
fig_sphere = nothing
fig_anim = nothing
GC.gc()
CUDA.reclaim()

# ## AD forward + backward
#
# Use the separate `model_ad` for Enzyme differentiation.
# `nsteps_ad` can be smaller than `nsteps_vis` to limit memory usage
# in the backward pass.

nsteps_ad = 2500  * 16

θ_init  = CenterField(grid); set!(θ_init,  θᵢ)
ρ_init  = CenterField(grid)
parent(ρ_init) .= parent(model_vis.dynamics.density)
dθ_init = CenterField(grid); set!(dθ_init, FT(0))

function loss(model, θ_init, ρ_init, Δt, nsteps)
    set!(model; ρ = ρ_init)
    set!(model; θ = θ_init)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    v = model.velocities.v
    return mean(interior(v) .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, ρ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, loss_val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(ρ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_init, loss_val
end

@info "Compiling AD forward pass…"
@time compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss(
    model_ad, θ_init, ρ_init, Δt, nsteps_ad)

@info "Compiling AD backward pass (Enzyme reverse mode)…"
dmodel = Enzyme.make_zero(model_ad)
@time compiled_bwd = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model_ad, dmodel, θ_init, dθ_init, ρ_init, Δt, nsteps_ad)

@info "Running AD forward pass…"
@time compiled_fwd(model_ad, θ_init, ρ_init, Δt, nsteps_ad)

@info "Computing sensitivity (Enzyme reverse mode)…"
@time dθ, J = compiled_bwd(model_ad, dmodel, θ_init, dθ_init, ρ_init, Δt, nsteps_ad)
sensitivity = Array(interior(dθ))

@info "Loss value" J
@info "Max |∂J/∂θ₀|" maximum(abs, sensitivity)

# ## Sensitivity plot
#
# Save sensitivity heatmap after AD.

i_pert = argmin(abs.(collect(λ) .- λ_c))
slimit = max(maximum(abs, sensitivity), eps(FT))

fig_sens = Figure(size = (1200, 450), fontsize = 14)
Label(fig_sens[0, :],
      @sprintf("Adjoint sensitivity (∂J/∂θ₀), J = %.6e", Float64(J)),
      fontsize = 16, tellwidth = false)

axs1 = Axis(fig_sens[1, 1]; xlabel = "λ (°)", ylabel = "φ (°)",
            title = "∂J/∂θ₀ — z ≈ $(Int(round(z_mid))) m", aspect = DataAspect())
hms1 = heatmap!(axs1, collect(λ), collect(φ), sensitivity[:, :, k_mid];
                colormap = :balance, colorrange = (-slimit, slimit))
Colorbar(fig_sens[1, 2], hms1; label = "∂J/∂θ₀")

axs2 = Axis(fig_sens[1, 3]; xlabel = "φ (°)", ylabel = "z (m)",
            title = "∂J/∂θ₀ — λ ≈ $(Int(round(λ[i_pert])))°")
hms2 = heatmap!(axs2, collect(φ), collect(z), sensitivity[i_pert, :, :];
                colormap = :balance, colorrange = (-slimit, slimit))
Colorbar(fig_sens[1, 4], hms2; label = "∂J/∂θ₀")

save("baroclinic_wave_sensitivity.png", fig_sens; px_per_unit = 2)
@info "Saved baroclinic_wave_sensitivity.png"

nothing #hide
