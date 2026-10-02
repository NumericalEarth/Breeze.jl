# # Power method eigenanalysis of MOIST baroclinic instability
#
# Moist counterpart of `validation/power_method_baroclinic.jl`. Applies the
# [ParkEtAl2013](@citet) power-method procedure to the DCMIP2016
# [UllrichEtAl2016](@citet) jet with moisture — the DCMIP2016 §1.1
# pressure-coordinate specific humidity plus Kessler warm-rain microphysics.
# [ParkEtAl2013](@citet) argue the power-method growth rate is robust to
# correctly-implemented moisture dynamics, so a moist run that recovers the dry
# σ ≈ 0.75 day⁻¹ wavenumber-9 mode is a validation of the moist dynamics.
#
# Measured result: converged σ ≈ 0.765 day⁻¹ (10 iterations), within ~2 % of the
# dry rate — i.e. the growth rate is robust to moisture, as predicted.
#
# The moist balanced state launches a brief first-step adjustment transient, so
# each 3-day power cycle restarts at a gentle `Δt = 1 minute` and lets the wizard
# ramp to 12 min (see `examples/moist_baroclinic_wave.jl`).

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!, reset!
using Printf
using CairoMakie
using CUDA

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    saturation_vapor_pressure = TetensFormula(),
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
Rᵛ  = vapor_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
ε   = Rᵛ / Rᵈ - 1
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# ## Grid — same as the dry power method / baroclinic_wave.jl

Nλ = 360; Nφ = 150; Nz = 64
H = 30kilometers
z_faces = ExponentialDiscretization(Nz, 0, H; scale = H/2, bias = :left)

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                             longitude = (0, 360), latitude = (-75, 75), z = z_faces)

# ## Analytic DCMIP2016 balanced state (+ moisture)

Tᴱ = 310; Tᴾ = 240; Tᴹ = (Tᴱ + Tᴾ) / 2; Γ = 0.005; K = 3; b = 2
q₀ = 0.018; φʷ = 2π / 9; pʷ = 34000; qᵗᵒᵖ = 1e-12

function τ_and_integrals(z)
    Hˢ = Rᵈ * Tᴹ / g; η = z / (b * Hˢ); e = exp(-η^2)
    A = (Tᴹ - Tᴾ) / (Tᴹ * Tᴾ); C = (K + 2) * (Tᴱ - Tᴾ) / (2 * Tᴱ * Tᴾ)
    τ₁ = A * (1 - 2η^2) * e + exp(Γ * z / Tᴹ) / Tᴹ
    ∫τ₁ = A * z * e + (exp(Γ * z / Tᴹ) - 1) / Γ
    τ₂ = C * (1 - 2η^2) * e; ∫τ₂ = C * z * e
    return τ₁, τ₂, ∫τ₁, ∫τ₂
end
F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)

function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z); return 1 / (τ₁ - τ₂ * F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z); return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

## DCMIP2016 §1.1 pressure-coordinate specific humidity (subsaturated).
function vapor_mass_fraction(λ, φ, z)
    φʳ = deg2rad(φ); η = pressure(λ, φ, z) / p₀
    qᵛ = q₀ * exp(-(φʳ / φʷ)^4) * exp(-((η - 1) * p₀ / pʷ)^2)
    return ifelse(η > 0.1, qᵛ, qᵗᵒᵖ)
end
function potential_temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z); qᵛ = vapor_mass_fraction(λ, φ, z)
    T = Tᵥ / (1 + ε * qᵛ); return T * (p₀ / pressure(λ, φ, z))^κ
end

## Balanced jet only — no localized trigger (power method grows its own seed).
function zonal_velocity_balanced(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z); Tᵥ = virtual_temperature(λ, φ, z)
    U = g / a * K * ∫τ₂ * dF(φ) * Tᵥ; rcosφ = a * cosd(φ); Ωrcosφ = Ω * rcosφ
    return -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
end

# ## Moist model

coriolis = SphericalCoriolis(rotation_rate=Ω)
T₀ᵣ = 250; θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))
dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀, reference_potential_temperature = θᵣ)
microphysics = DCMIP2016KesslerMicrophysics()
momentum_advection = WENO(order=5)
scalar_advection = (ρθ  = WENO(order=5),
                    ρqᵛ  = WENO(order=5, bounds=(0, 1)),
                    ρqᶜˡ = WENO(order=5, bounds=(0, 1)),
                    ρqʳ  = WENO(order=5, bounds=(0, 1)))
model = AtmosphereModel(grid; dynamics, coriolis, thermodynamic_constants = constants,
                        microphysics, momentum_advection, scalar_advection)

# ## Balanced IC + background snapshot (before the perturbation)

set!(model, θ=potential_temperature, u=zonal_velocity_balanced, ρ=density, qᵛ=vapor_mass_fraction)
background = map(f -> copy(parent(f)), prognostic_fields(model))

# ## Wavenumber-9 meridional-velocity seed

v_ref = 1.2
function v_perturbation(λ, φ, z)
    zₚ = 15000
    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, zero(z))
    return v_ref * exp(-(φ - 40)^2 / 225) * sind(9λ) * taper
end
set!(model; v=v_perturbation)

# ## Simulation
#
# Δτ = 3-day power cycle. Each cycle restarts near the balanced moist state, so
# we reset Δt to a gentle 60 s before each `run!` and let the wizard ramp it
# (max_change = 1.08) up to 12 min — this clears the first-step moist adjustment
# transient at full resolution (a fraction of each 3-day cycle).

Δτ = 3days
Δt₀ = 60.0
simulation = Simulation(model; Δt=Δt₀, stop_time=Δτ)
conjure_time_step_wizard!(simulation; cfl=0.7, max_Δt=12minutes, max_change=1.08)
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

max_iterations = 80
convergence_threshold = 0.001
σ_history = Float64[]

for n in 1:max_iterations
    global simulation
    simulation.Δt = Δt₀            # gentle restart each cycle; wizard ramps to 12 min
    run!(simulation)

    v_sfc_max = maximum(abs, view(model.velocities.v, :, :, 1))
    σ = log(v_sfc_max / v_ref) / Δτ
    push!(σ_history, σ)
    @info @sprintf("Power iteration %3d | σ = %.4f day⁻¹ | sfc max|v| = %.4e m/s",
                   n, σ * 86400, v_sfc_max)

    scale = convert(eltype(model.grid), v_ref / v_sfc_max)
    for (f, bg) in zip(prognostic_fields(model), background)
        parent(f) .= bg .+ scale .* (parent(f) .- bg)
    end

    σ_scale = max(abs(σ_history[end]), eps())
    converged = n ≥ 2 && σ_history[end] > 0 &&
                abs(σ_history[end] - σ_history[end-1]) / σ_scale < convergence_threshold
    if converged
        update_state!(model, compute_tendencies=false)
        @info @sprintf("Converged after %d iterations (σ = %.4f day⁻¹)", n, σ * 86400)
        break
    end
    reset!(model.clock)   # each cycle integrates t = 0 → Δτ from the same start time
    update_state!(model, compute_tendencies=false)
end

# ## Convergence plot

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1]; xlabel="Power iteration", ylabel="Growth rate σ (day⁻¹)",
          title="Moist power method convergence")
σ_per_day = σ_history .* 86400
lines!(ax, 1:length(σ_per_day), σ_per_day; linewidth=2, color=:dodgerblue)
scatter!(ax, 1:length(σ_per_day), σ_per_day; markersize=6, color=:dodgerblue)
hlines!(ax, [0.75]; linestyle=:dash, color=:gray40, label="dry DCMIP2016 jet (≈0.75)")
axislegend(ax; position=:rb)
save("validation/output/power_method_moist_convergence.png", fig)

σ_converged = σ_history[end] * 86400
@info @sprintf("MOIST converged σ = %.4f day⁻¹ after %d iterations", σ_converged, length(σ_history))
