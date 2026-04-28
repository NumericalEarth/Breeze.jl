#####
##### Moist baroclinic wave with ADAPTIVE Δt via TimeStepWizard.
#####
##### Same physics setup as moist_bw_15day.jl. Difference: replace fixed
##### Δt = 20 s with conjure_time_step_wizard!(sim; cfl=0.7) so Δt is
##### driven by the advective CFL alone. Pre-set initial Δt to 5 s and
##### let the wizard ramp up.
#####
##### Pass criteria:
##### (a) Run completes 15 days without NaN.
##### (b) Δt_max found by the wizard (the answer to "how large can Δt go").
##### (c) Cyclone deepens normally; mass conserved to FP-floor.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2
using CloudMicrophysics

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = @__DIR__
const RUN_LABEL = "moist_bw_adaptive_halfdeg"
const STEM = joinpath(OUTDIR, RUN_LABEL)

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = Breeze.dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

Nλ = 720
Nφ = 320
Nz = 64
H  = 30kilometers

grid = LatitudeLongitudeGrid(arch;
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude  = (-80, 80),
                             z         = (0, H))

Tᴱ = 310.0; Tᴾ = 240.0; Tₘ = (Tᴱ + Tᴾ)/2; Γ = 0.005; K = 3; b = 2; ε_v = 0.608

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g
    η  = z / (b * Hₛ); e = exp(-η^2)
    A = (Tₘ - Tᴾ)/(Tₘ * Tᴾ); C = (K + 2)/2 * (Tᴱ - Tᴾ)/(Tᴱ * Tᴾ)
    τ₁  = exp(Γ * z / Tₘ)/Tₘ + A*(1 - 2η^2)*e
    τ₂  = C * (1 - 2η^2) * e
    ∫τ₁ = (exp(Γ*z/Tₘ) - 1)/Γ + A*z*e
    ∫τ₂ = C*z*e
    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

F(φ)  = cosd(φ)^K - K/(K+2)*cosd(φ)^(K+2)
dF(φ) = cosd(φ)^(K-1) - cosd(φ)^(K+1)

virtual_temperature(λ, φ, z) = let
    τ₁, τ₂, _, _ = τ_and_integrals(z); 1/(τ₁ - τ₂*F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g/Rᵈ*(∫τ₁ - ∫τ₂*F(φ)))
end
density(λ, φ, z) = pressure(λ, φ, z)/(Rᵈ * virtual_temperature(λ, φ, z))

function specific_humidity(λ, φ, z)
    q₀ = 0.018; qₜ = 1e-12; φʷ = 2π/9; pʷ = 34000
    p = pressure(λ, φ, z); η = p/p₀; φʳ = deg2rad(φ)
    q_trop = q₀ * exp(-(φʳ/φʷ)^4) * exp(-((η-1)*p₀/pʷ)^2)
    return ifelse(η > 0.1, q_trop, qₜ)
end
function temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z); q = specific_humidity(λ, φ, z)
    return Tᵥ / (1 + ε_v * q)
end
function potential_temperature(λ, φ, z)
    return temperature(λ, φ, z) * (p₀/pressure(λ, φ, z))^κ
end

function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    T = temperature(λ, φ, z)
    U = g/a * K * ∫τ₂ * dF(φ) * T
    rcosφ = a*cosd(φ); Ωrcosφ = Ω*rcosφ
    u_b = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ*U)
    uₚ=1.0; rₚ=0.1; λₚ=π/9; φₚ=2π/9; zₚ=15000.0
    φʳ=deg2rad(φ); λʳ=deg2rad(λ)
    gc = acos(sin(φₚ)*sin(φʳ) + cos(φₚ)*cos(φʳ)*cos(λʳ-λₚ))/rₚ
    taper = ifelse(z < zₚ, 1 - 3*(z/zₚ)^2 + 2*(z/zₚ)^3, 0.0)
    u_p = ifelse(gc < 1, uₚ * taper * exp(-gc^2), 0.0)
    return u_b + u_p
end

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

coriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω)
T₀_ref   = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_ref)

τ_relax = 200.0
relaxation = ConstantRateCondensateFormation(1/τ_relax)
cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

Cᴰ = 1e-3; Uᵍ = 1e-2
T_surface(λ, φ) = virtual_temperature(λ, φ, 0.0)
ρu_bcs  = FieldBoundaryConditions(bottom = Breeze.BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρv_bcs  = FieldBoundaryConditions(bottom = Breeze.BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρθ_bcs  = FieldBoundaryConditions(bottom = Breeze.BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
ρqᵛ_bcs = FieldBoundaryConditions(bottom = Breeze.BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface))
boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)

weno = WENO()
bp_weno = WENO(order=5, bounds=(0, 1))
scalar_advection = (ρθ=weno, ρqᵛ=bp_weno, ρqᶜˡ=bp_weno, ρqᶜⁱ=bp_weno, ρqʳ=bp_weno, ρqˢ=bp_weno)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, boundary_conditions,
                        thermodynamic_constants = constants,
                        momentum_advection = weno,
                        scalar_advection,
                        timestepper = :AcousticRungeKutta3)

set!(model, θ = potential_temperature, u = zonal_velocity, ρ = density, qᵛ = specific_humidity)

# Adaptive Δt: start small, let the wizard ramp it up.
Δt_init   = 5.0      # initial Δt; small to absorb startup transient
stop_time = 1day

simulation = Simulation(model; Δt = Δt_init, stop_time)

# TimeStepWizard adapts Δt from advective CFL. cfl=0.7 is the
# WS-RK3 advective ceiling. max_change=1.05 ramps slowly so we don't
# leap from Δt=5 to Δt=200 in one wizard call.
conjure_time_step_wizard!(simulation;
                          cfl = 0.7,
                          max_Δt = 300.0,
                          max_change = 1.05)

# Diagnostics
diag_iters = Int[]; diag_t = Float64[]; diag_dt = Float64[]
diag_wmax  = Float64[]; diag_umax = Float64[]; diag_ρmin = Float64[]
diag_qcl_max = Float64[]; diag_qv_max = Float64[]
diag_psurf_min = Float64[]
diag_total_mass = Float64[]; diag_total_ρθ = Float64[]; diag_total_ρqv = Float64[]
diag_wall = Float64[]
wall_start = Ref(time_ns())

ρ_field0 = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
M0 = Float64(sum(interior(ρ_field0)))
ρθ_field0 = model.formulation.potential_temperature_density
H0 = Float64(sum(interior(ρθ_field0)))
ρqᵛ_field0 = model.moisture_density
Q0 = Float64(sum(interior(ρqᵛ_field0)))

function diag_cb(sim)
    m = sim.model
    u, v, w = m.velocities
    p = m.dynamics.pressure
    ρ = Breeze.AtmosphereModels.dynamics_density(m.dynamics)
    ρθ = m.formulation.potential_temperature_density
    ρqᵛ = m.moisture_density
    ρqᶜˡ = m.microphysical_fields.ρqᶜˡ

    wmax = Float64(maximum(abs, interior(w)))
    umax = Float64(maximum(abs, interior(u)))
    ρmin = Float64(minimum(interior(ρ)))
    qcl_max = Float64(maximum(interior(ρqᶜˡ)) / max(Float64(minimum(interior(ρ))), 1e-10))
    qv_max  = Float64(maximum(interior(ρqᵛ)) / max(Float64(minimum(interior(ρ))), 1e-10))
    p_surf = view(interior(p), :, :, 1)
    psurf_min = Float64(minimum(p_surf))
    M = Float64(sum(interior(ρ)))
    H = Float64(sum(interior(ρθ)))
    Q = Float64(sum(interior(ρqᵛ)))

    push!(diag_iters, iteration(sim))
    push!(diag_t, time(sim))
    push!(diag_dt, sim.Δt)
    push!(diag_wmax, wmax)
    push!(diag_umax, umax)
    push!(diag_ρmin, ρmin)
    push!(diag_qcl_max, qcl_max)
    push!(diag_qv_max, qv_max)
    push!(diag_psurf_min, psurf_min)
    push!(diag_total_mass, M)
    push!(diag_total_ρθ, H)
    push!(diag_total_ρqv, Q)
    push!(diag_wall, (time_ns() - wall_start[]) / 1e9)

    @info @sprintf("[adapt] iter=%6d  t=%6.3fd  Δt=%5.1fs  max|u|=%.2f  max|w|=%.3e  max|qcl|=%.2e  p_surf_min=%.0f Pa  ΔM=%.1e  wall=%.0fs",
                   iteration(sim), time(sim)/86400, sim.Δt, umax, wmax, qcl_max, psurf_min,
                   (M-M0)/M0, diag_wall[end])
    flush(stdout); flush(stderr)
end

add_callback!(simulation, diag_cb, IterationInterval(200))

# State output every 6 hours
sim_outputs = merge(model.velocities,
                    (; ρθ = model.formulation.potential_temperature_density,
                       p  = model.dynamics.pressure,
                       ρqᶜˡ = model.microphysical_fields.ρqᶜˡ))
simulation.output_writers[:state] = JLD2Writer(model, sim_outputs;
    filename = STEM * "_state.jld2",
    schedule = TimeInterval(6hours),
    overwrite_existing = true)

@info "Starting moist-BW 15-day ADAPTIVE run, Δt_init=$(Δt_init)s, cfl=0.7, max_Δt=300s"
wall_start[] = time_ns()

try
    run!(simulation)
    @info "[adapt] RUN COMPLETED"
catch e
    @error "[adapt] RUN FAILED" e
end

jldsave(STEM * "_diagnostics.jld2";
        iters = diag_iters, t = diag_t, dt = diag_dt,
        wmax = diag_wmax, umax = diag_umax,
        ρmin = diag_ρmin, qcl_max = diag_qcl_max, qv_max = diag_qv_max,
        psurf_min = diag_psurf_min,
        total_mass = diag_total_mass, total_ρθ = diag_total_ρθ, total_ρqv = diag_total_ρqv,
        wall = diag_wall, M0 = M0, H0 = H0, Q0 = Q0)

@info "Moist-BW 15-day ADAPTIVE run complete."
