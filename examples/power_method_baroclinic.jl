# # Power method eigenanalysis of baroclinic instability
#
# This example computes the **most unstable eigenmode** and **growth rate** of
# baroclinic instability in the DCMIP2016 midlatitude jet using the
# **power method** following [ParkEtAl2013](@citet).
#
# Park, Skamarock & Klemp (2013) validated MPAS and WRF by repeatedly
# integrating the Jablonowski–Williamson balanced jet forward three days,
# then rescaling perturbations to a reference amplitude. After enough
# iterations the perturbation locks onto the most unstable normal mode
# wavenumber 9, with growth rate σ ≈ 0.46 day⁻¹ at 1° resolution.
#
# The algorithm:
#
# 1. Start from the DCMIP2016 balanced jet (no perturbation).
# 2. Seed a meridional velocity perturbation at wavenumber 9.
# 3. Integrate forward ``Δτ = 3`` days.
# 4. Measure ``v_{\max}`` = max|v| at the **lowest model level**.
# 5. Growth rate: ``σ = \ln(v_{\max} / v_{\rm ref}) / Δτ``.
# 6. Rescale all perturbation fields by ``v_{\rm ref} / v_{\max}``.
# 7. Reset the clock and repeat until ``σ`` converges.
#
# The growth rate is measured at the lowest level to match
# [ParkEtAl2013](@citet) §3a.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!, reset!
using Printf
using CairoMakie
using CUDA

# ## DCMIP2016 parameters
#
# All parameters follow the DCMIP2016 specification [UllrichEtAl2016](@citet),
# matching the [`baroclinic_wave`](@ref) example.

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
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# ## Grid
#
# A 1° latitude-longitude grid spanning 75° S to 75° N with 32 vertical
# levels up to 30 km — matching the production resolution from
# [`baroclinic_wave`](@ref). Uncomment a different line to switch phases.

Nλ = 360; Nφ = 150; Nz = 128     ## Phase 3: production (1°)

H = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-75, 75),
                             z = (0, H))

# ## Analytic initial conditions
#
# The DCMIP2016 balanced state: virtual temperature ``T_v(\varphi, z)``,
# pressure, density, potential temperature, and gradient-wind-balanced
# zonal velocity. These are identical to [`baroclinic_wave`](@ref).

## Temperature profile parameters
Tᴱ = 310     # K — equatorial surface temperature
Tᴾ = 240     # K — polar surface temperature
Tᴹ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005   # K/m — lapse rate
K  = 3       # jet width parameter
b  = 2       # vertical half-width parameter

## Vertical structure functions
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

function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

potential_temperature(λ, φ, z) = virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

# ### Balanced zonal wind
#
# Gradient-wind balance gives the zonal velocity. We omit the localized
# perturbation from the standard DCMIP test — the power method supplies
# its own controlled perturbation instead.

function zonal_velocity_balanced(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    Tᵥ = virtual_temperature(λ, φ, z)

    U = g / a * K * ∫τ₂ * dF(φ) * Tᵥ
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    return -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
end

# ## Model
#
# Compressible dynamics with acoustic substepping, matching
# [`baroclinic_wave`](@ref).

coriolis = SphericalCoriolis(rotation_rate=Ω)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

# ### Horizontal dissipation
#
# Without an explicit closure the advection scheme's implicit dissipation is the
# only small-scale sink, and it is too weak to enforce the physical short-wave
# cutoff: the inviscid operator amplifies marginally-resolved scales into a
# spurious high-wavenumber band that outgrows the physical wavenumber-9 wave.
# A scale-selective horizontal viscosity fixes this — it damps mode ``k`` at
# rate ``ν k²``, hammering the grid-scale band while barely touching the
# synoptic wave, so wavenumber 9 is restored as the leading baroclinic mode.

using Oceananigans.TurbulenceClosures: HorizontalScalarDiffusivity
closure = HorizontalScalarDiffusivity(ν=3e5, κ=3e5)   # m²/s

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO(), closure)

# ## Balanced initial condition and background snapshot
#
# Set the pure balanced jet, then snapshot every prognostic field
# (including halos) so we can extract perturbations later. This follows
# the pattern from [`adiabatic_balance!`](@ref).

set!(model; θ=potential_temperature, u=zonal_velocity_balanced, ρ=density)

background = map(f -> copy(parent(f)), prognostic_fields(model))

# ## Initial perturbation
#
# A meridional velocity perturbation at wavenumber 9, Gaussian in latitude
# around 40° N, tapered to zero above 15 km. The balanced state has
# ``v = 0`` everywhere, so `set!(model; v=...)` adds a pure perturbation.

v_ref = 1.2  ## m/s — Park et al. reference amplitude

function v_perturbation(λ, φ, z)
    zₚ = 15000
    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, zero(z))
    return v_ref * exp(-(φ - 40)^2 / 225) * sind(9λ) * taper
end

set!(model; v=v_perturbation)

# ## Simulation

Δτ = 3days

simulation = Simulation(model; Δt=12minutes, stop_time=Δτ)
conjure_time_step_wizard!(simulation; cfl=1.4, max_Δt=12minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

function power_method_progress(sim)
    v = sim.model.velocities.v
    v_sfc = maximum(abs, view(v, :, :, 1))
    @info @sprintf("  step %5d | t = %s | Δt = %s | sfc max|v| = %.4f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), v_sfc)
    return nothing
end

add_callback!(simulation, power_method_progress, IterationInterval(100))

# ## Power iteration
#
# Each iteration integrates forward ``Δτ``, measures max|v| at the
# **lowest model level** (matching [ParkEtAl2013](@citet) §3a), computes
# the growth rate, and rescales all prognostic perturbation fields back
# to the reference amplitude. The clock is reset to ``t = 0`` so each
# iteration starts from the same initial time.
#
# No wavenumber or hemispheric filtering is applied — this is the raw
# power method on the full symmetric DCMIP2016 jet.

max_iterations = 80
convergence_threshold = 0.001  ## relative change in σ
σ_history = Float64[]

for n in 1:max_iterations
    run!(simulation)

    ## Measure max|v| at the lowest model level (k=1), following Park et al.
    v_sfc_max = maximum(abs, view(model.velocities.v, :, :, 1))

    ## Growth rate
    σ = log(v_sfc_max / v_ref) / Δτ
    push!(σ_history, σ)

    @info @sprintf("Power iteration %3d | σ = %.4f day⁻¹ | sfc max|v| = %.4e m/s",
                   n, σ * 86400, v_sfc_max)

    ## Rescale all prognostic perturbations: field = background + scale × (field - background)
    scale = v_ref / v_sfc_max
    for (f, bg) in zip(prognostic_fields(model), background)
        parent(f) .= bg .+ scale .* (parent(f) .- bg)
    end

    ## Fill halos after rescaling
    for f in prognostic_fields(model)
        fill_halo_regions!(f)
    end

    ## Check convergence (after rescaling so final state has reference amplitude)
    converged = n ≥ 2 && abs(σ_history[end] - σ_history[end-1]) / abs(σ_history[end]) < convergence_threshold

    if converged
        update_state!(model, compute_tendencies=false)
        @info @sprintf("Converged after %d iterations (σ = %.4f day⁻¹)", n, σ * 86400)
        break
    end

    ## Reset for next iteration
    reset!(model.clock)
    update_state!(model, compute_tendencies=false)
    simulation.stop_time = Δτ
end

# ## Visualization
#
# Three plots: growth rate convergence, and the eigenmode structure in
# meridional velocity ``v`` and potential temperature perturbation
# ``\deltaθ`` at the surface.

# ### Convergence

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1];
          xlabel = "Power iteration",
          ylabel = "Growth rate σ (day⁻¹)",
          title = "Power method convergence")

σ_per_day = σ_history .* 86400
lines!(ax, 1:length(σ_per_day), σ_per_day; linewidth=2, color=:dodgerblue)
scatter!(ax, 1:length(σ_per_day), σ_per_day; markersize=6, color=:dodgerblue)
hlines!(ax, [0.46]; linestyle=:dash, color=:gray60, label="Park et al. (2013)")
axislegend(ax; position=:rb)

save("power_method_convergence.png", fig)
nothing #hide

# ![](power_method_convergence.png)

# ### Eigenmode structure
#
# The meridional velocity ``v`` and the **potential temperature
# perturbation** ``\delta\theta = \theta - \bar\theta`` at the lowest
# model level. The perturbation is what reveals the eigenmode — the
# background ``\bar\theta`` is subtracted out.

v = model.velocities.v
θ_field = Field(PotentialTemperature(model))
compute!(θ_field)

## Compute θ perturbation by subtracting the analytic background
θ_perturbation = Field{Center, Center, Center}(model.grid)
set!(θ_perturbation, potential_temperature)
parent(θ_perturbation) .= parent(θ_field) .- parent(θ_perturbation)

v_sfc = view(v, :, :, 1)
δθ_sfc = view(θ_perturbation, :, :, 1)

vlim = maximum(abs, v_sfc)
δθlim = maximum(abs, δθ_sfc)

fig2 = Figure(size=(1200, 500))

ax1 = Axis(fig2[1, 1]; title="v eigenmode (surface)",
           xlabel="Longitude", ylabel="Latitude")
hm1 = heatmap!(ax1, v_sfc; colormap=:balance, colorrange=(-vlim, vlim))
Colorbar(fig2[1, 2], hm1; label="v (m/s)")

ax2 = Axis(fig2[1, 3]; title="δθ eigenmode (surface)",
           xlabel="Longitude", ylabel="Latitude")
hm2 = heatmap!(ax2, δθ_sfc; colormap=:balance, colorrange=(-δθlim, δθlim))
Colorbar(fig2[1, 4], hm2; label="δθ (K)")

save("power_method_eigenmode.png", fig2)
nothing #hide

# ![](power_method_eigenmode.png)
