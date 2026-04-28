#####
##### validation/substepping/48_dry_bw_smoke.jl
#####
##### DCMIP-2016 dry baroclinic wave smoke test at production Δt = 225 s
##### with the new SplitExplicit defaults (forward_weight = 0.6,
##### KlempDivergenceDamping coef = 0.1).
#####
##### Pass: survives ≥ 96 outer steps (6 h simulated time) with finite
##### max|w|. Per the agent report's F2 criterion the prior config NaN'd
##### at step 11 (~41 min) on the same grid.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA

CUDA.functional() || error("GPU required")

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

Nλ = 360
Nφ = 160
Nz = 64
H  = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (0, H))

Tᴱ = 310.0
Tᴾ = 240.0
Tₘ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005
K  = 3
b  = 2

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g
    η  = z / (b * Hₛ)
    e  = exp(-η^2)
    A = (Tₘ - Tᴾ) / (Tₘ * Tᴾ)
    C = (K + 2) / 2 * (Tᴱ - Tᴾ) / (Tᴱ * Tᴾ)
    τ₁  = exp(Γ * z / Tₘ) / Tₘ + A * (1 - 2η^2) * e
    τ₂  = C * (1 - 2η^2) * e
    ∫τ₁ = (exp(Γ * z / Tₘ) - 1) / Γ + A * z * e
    ∫τ₂ = C * z * e
    return τ₁, τ₂, ∫τ₁, ∫τ₂
end
F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)
function temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * temperature(λ, φ, z))
function potential_temperature(λ, φ, z)
    T = temperature(λ, φ, z)
    p = pressure(λ, φ, z)
    return T * (p₀ / p)^κ
end
function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    T = temperature(λ, φ, z)
    U = g / a * K * ∫τ₂ * dF(φ) * T
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    u_balanced = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
    uₚ = 1.0
    rₚ = 0.1
    λₚ = π / 9
    φₚ = 2π / 9
    zₚ = 15000.0
    φʳ = deg2rad(φ)
    λʳ = deg2rad(λ)
    great_circle = acos(sin(φₚ) * sin(φʳ) + cos(φₚ) * cos(φʳ) * cos(λʳ - λₚ)) / rₚ
    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, 0.0)
    u_perturbation = ifelse(great_circle < 1, uₚ * taper * exp(-great_circle^2), 0.0)
    return u_balanced + u_perturbation
end

coriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω)
T₀_ref = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

function build_dry_bw(; forward_weight, damping)
    td = SplitExplicitTimeDiscretization(; forward_weight, damping)
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = p₀,
                                    reference_potential_temperature = θ_ref)
    model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants = constants,
                            advection = WENO(),
                            timestepper = :AcousticRungeKutta3)
    set!(model, θ = potential_temperature, u = zonal_velocity, ρ = density)
    return model
end

function run_smoke(label; forward_weight, damping, n_steps = 96)
    @info "==== $label : ω=$forward_weight, $(typeof(damping).name.name) coef=$(getfield(damping, :coefficient)) ===="
    model = build_dry_bw(; forward_weight, damping)
    Δt = 225.0
    sample_every = 4
    t0 = time()
    crashed = false
    samples = NTuple{4, Float64}[]
    for n in 1:n_steps
        try
            time_step!(model, Δt)
        catch e
            @warn "[$label] crashed at step $n: $(sprint(showerror, e))"
            crashed = true
            break
        end
        if n % sample_every == 0 || n == 1
            u, v, w = model.velocities
            wmax = Float64(maximum(abs, interior(w)))
            umax = Float64(maximum(abs, interior(u)))
            push!(samples, (Float64(n), n * Δt / 3600, wmax, umax))
            @info @sprintf("[%s]  step %3d  t=%.2f h  max|w|=%.4e  max|u|=%.2f",
                           label, n, n * Δt / 3600, wmax, umax)
            if !isfinite(wmax) || !isfinite(umax)
                crashed = true
                break
            end
        end
    end
    elapsed = time() - t0
    final = isempty(samples) ? (0.0, 0.0, NaN, NaN) : samples[end]
    @info @sprintf("[%s] final step=%d t=%.2fh max|w|=%.4e max|u|=%.2f crashed=%s wallclock=%.1fs",
                   label, Int(final[1]), final[2], final[3], final[4], crashed, elapsed)
    return (; label, samples, crashed, last_t = final[2])
end

td_default = SplitExplicitTimeDiscretization()
@info "Defaults: ω=$(td_default.forward_weight), β_d=$(td_default.damping.coefficient)"

results = []
push!(results, run_smoke("default";    forward_weight = td_default.forward_weight,
                                       damping        = td_default.damping,
                                       n_steps = 192))   # 12 h smoke

println()
println("=== SUMMARY (Dry BW smoke, 96 steps × 225 s = 6 h) ===")
for r in results
    mark = r.crashed ? "✗" : "✓"
    @printf("  %s %-15s  reached t=%.2f h, crashed=%s\n", mark, r.label, r.last_t, r.crashed)
end
