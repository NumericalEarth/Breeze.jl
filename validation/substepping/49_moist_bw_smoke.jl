#####
##### validation/substepping/49_moist_bw_smoke.jl
#####
##### DCMIP-2016 moist baroclinic wave smoke test at production Δt = 20 s
##### with the new SplitExplicit defaults (forward_weight = 0.65,
##### KlempDivergenceDamping coef = 0.1).
#####
##### Pass: survives ≥ 180 outer steps (1 h simulated time) with finite
##### max|w|. Per the agent report's F1 criterion the prior config NaN'd
##### at step 15 (~5 min) on the same grid.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using CloudMicrophysics  # triggers BreezeCloudMicrophysicsExt

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
ε_v = 0.608

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
function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end
function specific_humidity(λ, φ, z)
    q₀  = 0.018
    qₜ  = 1e-12
    φʷ  = 2π / 9
    pʷ  = 34000.0
    p = pressure(λ, φ, z)
    η = p / p₀
    φʳ = deg2rad(φ)
    q_troposphere = q₀ * exp(-(φʳ / φʷ)^4) * exp(-((η - 1) * p₀ / pʷ)^2)
    return ifelse(η > 0.1, q_troposphere, qₜ)
end
density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))
function potential_temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z)
    p  = pressure(λ, φ, z)
    q  = specific_humidity(λ, φ, z)
    T  = Tᵥ / (1 + ε_v * q)
    return T * (p₀ / p)^κ
end
function temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z)
    q  = specific_humidity(λ, φ, z)
    return Tᵥ / (1 + ε_v * q)
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

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

coriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω)
T₀_ref = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

td = SplitExplicitTimeDiscretization()
@info "Moist BW smoke: ω=$(td.forward_weight), $(typeof(td.damping).name.name) coef=$(td.damping.coefficient)"

dynamics = CompressibleDynamics(td;
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_ref)

τ_relax = 200.0
relaxation = ConstantRateCondensateFormation(1 / τ_relax)
cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

Cᴰ = 1e-3
Uᵍ = 1e-2
T_surface(λ, φ) = virtual_temperature(λ, φ, 0.0)

ρu_bcs  = FieldBoundaryConditions(bottom = Breeze.BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ, surface_temperature = T_surface))
ρv_bcs  = FieldBoundaryConditions(bottom = Breeze.BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ, surface_temperature = T_surface))
ρθ_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᴰ, gustiness = Uᵍ, surface_temperature = T_surface))
ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = Cᴰ, gustiness = Uᵍ, surface_temperature = T_surface))
boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρθ = ρθ_bcs, ρqᵛ = ρqᵛ_bcs)

weno = WENO()
bounds_preserving_weno = WENO(order = 5, bounds = (0, 1))
scalar_advection = (ρθ = weno, ρqᵛ = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno, ρqᶜⁱ = bounds_preserving_weno,
                    ρqʳ = bounds_preserving_weno, ρqˢ = bounds_preserving_weno)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, boundary_conditions,
                        thermodynamic_constants = constants,
                        momentum_advection = weno,
                        scalar_advection,
                        timestepper = :AcousticRungeKutta3)

set!(model, θ = potential_temperature, u = zonal_velocity, ρ = density, qᵛ = specific_humidity)

Δt = 20.0
n_steps = 180     # 1 h smoke
sample_every = 10

@info @sprintf("Running %d outer steps × Δt = %.1f s = %.0f min simulated",
               n_steps, Δt, n_steps * Δt / 60)

t0 = time()
crashed = false
samples = NTuple{4, Float64}[]
for n in 1:n_steps
    try
        time_step!(model, Δt)
    catch e
        @warn "[moist-BW] crashed at step $n: $(sprint(showerror, e))"
        global crashed = true
        break
    end
    if n % sample_every == 0 || n == 1
        u, v, w = model.velocities
        wmax = Float64(maximum(abs, interior(w)))
        umax = Float64(maximum(abs, interior(u)))
        push!(samples, (Float64(n), n * Δt / 60, wmax, umax))
        @info @sprintf("  step %3d  t=%.1f min  max|w|=%.4e  max|u|=%.2f m/s",
                       n, n * Δt / 60, wmax, umax)
        if !isfinite(wmax) || !isfinite(umax)
            global crashed = true
            break
        end
    end
end
elapsed = time() - t0

if !isempty(samples)
    final_step, final_t, final_w, final_u = samples[end]
    @info @sprintf("=========== Final: step %d, t=%.1f min, max|w|=%.4e, max|u|=%.2f m/s ===========",
                   Int(final_step), final_t, final_w, final_u)
end
@info @sprintf("Wall time: %.1f s for %d steps (%.2f s/step)", elapsed, n_steps, elapsed / max(n_steps, 1))

if !crashed && !isempty(samples) && Int(samples[end][1]) >= n_steps
    @info "✓ PASS: moist baroclinic wave survives 1 h at production Δt = 20 s with new defaults."
else
    @warn "✗ FAIL: moist baroclinic wave crashed before 1 h."
end
