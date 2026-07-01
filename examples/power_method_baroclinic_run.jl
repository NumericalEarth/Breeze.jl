# Parameterized power method eigenanalysis — grid search variant.
#
# Usage:
#   julia --project=. power_method_baroclinic_run.jl <float_type> <weno_order> <use_closure>
#
# Outputs:
#   power_method_convergence_<tag>.png
#   power_method_eigenmode_<tag>.png

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models: boundary_condition_args
using Oceananigans.TimeSteppers: update_state!, reset!
using Oceananigans.TurbulenceClosures: HorizontalScalarDiffusivity
using Printf
using CairoMakie
using CUDA

## Parse arguments
length(ARGS) == 3 || error("Usage: julia power_method_baroclinic_run.jl <F64|F32> <weno_order> <true|false>")

float_label = ARGS[1]
weno_order  = parse(Int, ARGS[2])
use_closure = parse(Bool, ARGS[3])

FT = float_label == "F64" ? Float64 :
     float_label == "F32" ? Float32 :
     error("float_type must be F64 or F32, got $(float_label)")

closure_label = use_closure ? "closure" : "noclosure"
tag = "$(lowercase(float_label))_weno$(weno_order)_$(closure_label)"

@info "Power method study: FT=$(FT), WENO order=$(weno_order), closure=$(use_closure)  [tag: $tag]"

# ## DCMIP2016 parameters

Oceananigans.defaults.FloatType = FT
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

Nλ = 360; Nφ = 150; Nz = 128

H = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-75, 75),
                             z = (0, H))

# ## Analytic initial conditions

Tᴱ = 310     # K
Tᴾ = 240     # K
Tᴹ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005
K  = 3
b  = 2

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

function zonal_velocity_balanced(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    Tᵥ = virtual_temperature(λ, φ, z)
    U = g / a * K * ∫τ₂ * dF(φ) * Tᵥ
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    return -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
end

# ## Model

coriolis = SphericalCoriolis(rotation_rate=Ω)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

advection = WENO(order=weno_order)

if use_closure
    closure = HorizontalScalarDiffusivity(ν=3e5, κ=3e5)
    model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants = constants,
                            advection, closure)
else
    model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants = constants,
                            advection)
end

# ## Balanced initial condition and background snapshot

set!(model; θ=potential_temperature, u=zonal_velocity_balanced, ρ=density)

background = map(f -> copy(parent(f)), prognostic_fields(model))

# ## Initial perturbation

v_ref = 1.2

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
    @info @sprintf("[%s]   step %5d | t = %s | Δt = %s | sfc max|v| = %.4f m/s",
                   tag, iteration(sim), prettytime(sim), prettytime(sim.Δt), v_sfc)
    return nothing
end

add_callback!(simulation, power_method_progress, IterationInterval(100))

# ## Power iteration

max_iterations = 80
convergence_threshold = 0.001
σ_history = Float64[]

for n in 1:max_iterations
    run!(simulation)

    v_sfc_max = maximum(abs, view(model.velocities.v, :, :, 1))
    σ = log(v_sfc_max / v_ref) / Δτ
    push!(σ_history, σ)

    @info @sprintf("[%s] Power iteration %3d | σ = %.4f day⁻¹ | sfc max|v| = %.4e m/s",
                   tag, n, σ * 86400, v_sfc_max)

    scale = convert(eltype(model.grid), v_ref / v_sfc_max)
    for (f, bg) in zip(prognostic_fields(model), background)
        parent(f) .= bg .+ scale .* (parent(f) .- bg)
    end
    fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)

    converged = n ≥ 2 && abs(σ_history[end] - σ_history[end-1]) / abs(σ_history[end]) < convergence_threshold

    if converged
        update_state!(model, compute_tendencies=false)
        @info @sprintf("[%s] Converged after %d iterations (σ = %.4f day⁻¹)", tag, n, σ * 86400)
        break
    end

    reset!(model.clock)
    update_state!(model, compute_tendencies=false)
    simulation.stop_time = Δτ
end

# ## Visualization

config_str = "$(FT), WENO-$(weno_order), closure=$(use_closure)"

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1];
          xlabel = "Power iteration",
          ylabel = "Growth rate σ (day⁻¹)",
          title = "Power method convergence — $config_str")

σ_per_day = σ_history .* 86400
lines!(ax, 1:length(σ_per_day), σ_per_day; linewidth=2, color=:dodgerblue)
scatter!(ax, 1:length(σ_per_day), σ_per_day; markersize=6, color=:dodgerblue)
hlines!(ax, [0.46]; linestyle=:dash, color=:gray60, label="Park et al. (2013)")
axislegend(ax; position=:rb)

save("power_method_convergence_$(tag).png", fig)
nothing #hide

v = model.velocities.v
θ_field = Field(PotentialTemperature(model))
compute!(θ_field)

θ_perturbation = Field{Center, Center, Center}(model.grid)
set!(θ_perturbation, potential_temperature)
parent(θ_perturbation) .= parent(θ_field) .- parent(θ_perturbation)

v_sfc   = view(v, :, :, 1)
δθ_sfc  = view(θ_perturbation, :, :, 1)

vlim  = maximum(abs, v_sfc)
δθlim = maximum(abs, δθ_sfc)

fig2 = Figure(size=(1200, 500))
Label(fig2[0, 1:4], "Power method eigenmode — $config_str"; fontsize=16, tellwidth=false)

ax1 = Axis(fig2[1, 1]; title="v eigenmode (surface)", xlabel="Longitude", ylabel="Latitude")
hm1 = heatmap!(ax1, v_sfc; colormap=:balance, colorrange=(-vlim, vlim))
Colorbar(fig2[1, 2], hm1; label="v (m/s)")

ax2 = Axis(fig2[1, 3]; title="δθ eigenmode (surface)", xlabel="Longitude", ylabel="Latitude")
hm2 = heatmap!(ax2, δθ_sfc; colormap=:balance, colorrange=(-δθlim, δθlim))
Colorbar(fig2[1, 4], hm2; label="δθ (K)")

save("power_method_eigenmode_$(tag).png", fig2)
@info "Saved power_method_convergence_$(tag).png and power_method_eigenmode_$(tag).png"
