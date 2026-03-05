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

# ## Domain and grid
#
# Coarse resolution for fast Reactant compilation.
# Increase `Nλ`, `Nφ`, `Nz` for physical runs.

Nλ = 60
Nφ = 30
Nz = 10
H  = 30kilometers

grid = LatitudeLongitudeGrid(ReactantState();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-85, 85),
                             z = (0, H))

FT = eltype(grid)

# ## Physical parameters

constants = ThermodynamicConstants()
g  = constants.gravitational_acceleration
p₀ = 100000  # Pa — surface pressure
θ₀ = 300     # K — surface potential temperature
N² = 1e-4    # s⁻² — Brunt-Väisälä frequency squared

θᵇ(z) = θ₀ * exp(N² * z / g)

# ## Model configuration
#
# Split-explicit compressible dynamics with acoustic substepping.
# The reference state uses the stratified ``θ^{\rm b}(z)`` profile
# so the buoyancy force is a perturbation ``ρ b = -g (ρ - ρ_r)``.

coriolis = HydrostaticSphericalCoriolis()

dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵇ)

model = AtmosphereModel(grid; dynamics, coriolis, advection = WENO())

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
# computed correctly inside `set_velocity!`.

@info "Setting initial conditions…"
@time begin
    set!(model; ρ = ρᵢ)
    set!(model; u = uᵢ, θ = θᵢ)
end

# ## Time step

γ  = cᵖ / (cᵖ - Rᵈ)
cₛ = sqrt(γ * Rᵈ * θ₀)
Δz = H / Nz
Δt = FT(0.4 * Δz / cₛ)

@info "Time step" Δt cₛ

# ## Loss function and adjoint
#
# The loss is the mean squared meridional velocity over the full domain.
# Since ``v = 0`` initially, ``⟨v²⟩`` is exactly the meridional eddy
# kinetic energy — a direct measure of baroclinic growth.

nsteps = 4

θ_init  = CenterField(grid); set!(θ_init,  θᵢ)
dθ_init = CenterField(grid); set!(dθ_init, FT(0))

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = FT(1))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    v = model.velocities.v
    return mean(interior(v) .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, loss_val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_init, loss_val
end

# ## Reactant compilation
#
# `Reactant.@compile` traces both the forward model and the Enzyme-generated
# adjoint into XLA/StableHLO, producing fused, optimised kernels.

@info "Compiling forward pass…"
@time compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss(
    model, θ_init, Δt, nsteps)

@info "Compiling backward pass (Enzyme reverse mode)…"
dmodel = Enzyme.make_zero(model)
@time compiled_bwd = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)

# ## Run

@info "Running compiled forward pass…"
@time compiled_fwd(model, θ_init, Δt, nsteps)

v_evolved = Array(interior(model.velocities.v))
θ_evolved = Array(interior(model.formulation.potential_temperature))

@info "Computing sensitivity (Enzyme reverse mode)…"
@time dθ, J = compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)
sensitivity = Array(interior(dθ))

@info "Loss value" J
@info "Max |∂J/∂θ₀|" maximum(abs, sensitivity)

# ## Visualisation
#
# We save two outputs:
# 1) a sensitivity plot (map + latitude-height section),
# 2) sphere-rendered snapshots + GIF from a short forward simulation.

Δλ = 360.0 / Nλ
Δφ = 170.0 / Nφ
λ = range(Δλ / 2, 360 - Δλ / 2, length = Nλ)
φ = range(-85 + Δφ / 2, 85 - Δφ / 2, length = Nφ)
z = range(Δz / 2, H - Δz / 2, length = Nz)

k_mid  = Nz ÷ 2
z_mid  = z[k_mid]
i_pert = argmin(abs.(collect(λ) .- λ_c))

# Sensitivity figure.
slimit = max(maximum(abs, sensitivity), eps(FT))

fig_sens = Figure(size = (1200, 450), fontsize = 14)
Label(fig_sens[0, :],
      @sprintf("Adjoint sensitivity (∂J/∂θ₀), J = %.6e", J),
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

# Short forward run for sphere movie output.
@info "Running short forward simulation for sphere visualisation..."
@time begin
    set!(model; ρ = ρᵢ)
    set!(model; u = uᵢ, θ = θᵢ)
end

θ = PotentialTemperature(model)
θᵇᵍ = CenterField(grid)
set!(θᵇᵍ, (λ, φ, z) -> θᵇ(z))
θ′ = θ - θᵇᵍ

outputs = merge(model.velocities, (; θ′))
nsteps_vis = 120
vis_filename = "baroclinic_wave_sphere"

simulation = Simulation(model; Δt, stop_iteration = nsteps_vis)
simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                              filename = vis_filename,
                                              schedule = IterationInterval(5),
                                              overwrite_existing = true)
run!(simulation)

θ′_ts = FieldTimeSeries("$(vis_filename).jld2", "θ′")
u_ts = FieldTimeSeries("$(vis_filename).jld2", "u")
times = θ′_ts.times
Nt = length(times)

# Final snapshot on the sphere.
fig_sphere = Figure(size = (1200, 600))
sphere_kw = (elevation = π / 6, azimuth = -π / 2, aspect = :data)

ax1 = Axis3(fig_sphere[1, 1];
            title = "θ′ at z = $(Int(round(z_mid / 1e3))) km, t = $(prettytime(times[Nt]))",
            sphere_kw...)
hm1 = surface!(ax1, view(θ′_ts[Nt], :, :, k_mid);
               colormap = :balance, shading = NoShading)
Colorbar(fig_sphere[1, 2], hm1; label = "θ′ (K)")

ax2 = Axis3(fig_sphere[1, 3];
            title = "u at z = $(Int(round(z_mid / 1e3))) km, t = $(prettytime(times[Nt]))",
            sphere_kw...)
hm2 = surface!(ax2, view(u_ts[Nt], :, :, k_mid);
               colormap = :speed, shading = NoShading)
Colorbar(fig_sphere[1, 4], hm2; label = "u (m/s)")

for ax in (ax1, ax2)
    hidedecorations!(ax)
    hidespines!(ax)
end

save("baroclinic_wave_sphere.png", fig_sphere; px_per_unit = 2)
@info "Saved baroclinic_wave_sphere.png"

# GIF over the sphere.
n = Observable(1)
θ′n = @lift view(θ′_ts[$n], :, :, k_mid)
un = @lift view(u_ts[$n], :, :, k_mid)
title = @lift "z = $(Int(round(z_mid / 1e3))) km, t = $(prettytime(times[$n]))"

fig_anim = Figure(size = (1200, 600))
axg1 = Axis3(fig_anim[1, 1]; title = "θ′", sphere_kw...)
hmg1 = surface!(axg1, θ′n; colormap = :balance, colorrange = (-2, 2), shading = NoShading)
Colorbar(fig_anim[1, 2], hmg1; label = "θ′ (K)")

axg2 = Axis3(fig_anim[1, 3]; title = "u", sphere_kw...)
ulim = max(maximum(abs, Array(interior(u_ts[Nt]))), eps(FT))
hmg2 = surface!(axg2, un; colormap = :balance, colorrange = (-ulim, ulim), shading = NoShading)
Colorbar(fig_anim[1, 4], hmg2; label = "u (m/s)")

fig_anim[0, :] = Label(fig_anim, title, fontsize = 22, tellwidth = false)

for ax in (axg1, axg2)
    hidedecorations!(ax)
    hidespines!(ax)
end

CairoMakie.record(fig_anim, "baroclinic_wave_sphere.gif", 1:Nt; framerate = 12) do nn
    n[] = nn
end
@info "Saved baroclinic_wave_sphere.gif"

nothing #hide
