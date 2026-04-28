#####
##### Test A — Lat-lon moist BW with PASSIVE moisture (no condensation).
#####
##### Goal: test the microphysics-as-cause hypothesis.
##### Setup: full lat-lon BW, all DCMIP IC including qᵛ profile, but
##### `ConstantRateCondensateFormation(0)` so no vapor → cloud
##### condensation happens. qᵛ is therefore just an advected passive
##### tracer. Surface fluxes also OFF.
#####
##### Result interpretation:
##### • completes 5 days → microphysics IS the cause of the moist BW NaN.
##### • NaN like the with-microphysics no-flux run (day ~4) → microphysics
#####   is NOT the cause; the substepper-moist coupling is the cause
#####   (qᵛ alone is enough to destabilise via R_m → γ_m terms in the
#####   substepper, even with no condensation).
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
const RUN_LABEL = "moist_bw_passive_moisture"
const STEM = joinpath(OUTDIR, RUN_LABEL)

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g = constants.gravitational_acceleration
Rᵈ = Breeze.dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ = Rᵈ / cᵖᵈ
p₀ = 1e5
a = Oceananigans.defaults.planet_radius
Ω = Oceananigans.defaults.planet_rotation_rate

Tᴱ=310.0; Tᴾ=240.0; Tₘ=(Tᴱ+Tᴾ)/2; Γ=0.005; K=3; b=2; ε_v=0.608

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g; η = z/(b*Hₛ); e = exp(-η^2)
    A = (Tₘ-Tᴾ)/(Tₘ*Tᴾ); C = (K+2)/2 * (Tᴱ-Tᴾ)/(Tᴱ*Tᴾ)
    return (exp(Γ*z/Tₘ)/Tₘ + A*(1-2η^2)*e,
            C*(1-2η^2)*e,
            (exp(Γ*z/Tₘ)-1)/Γ + A*z*e,
            C*z*e)
end
F(φ)  = cosd(φ)^K - K/(K+2)*cosd(φ)^(K+2)
dF(φ) = cosd(φ)^(K-1) - cosd(φ)^(K+1)

function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z); 1/(τ₁ - τ₂*F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z); p₀*exp(-g/Rᵈ*(∫τ₁ - ∫τ₂*F(φ)))
end
density(λ, φ, z) = pressure(λ, φ, z)/(Rᵈ * virtual_temperature(λ, φ, z))
function specific_humidity(λ, φ, z)
    p = pressure(λ, φ, z); η = p/p₀; φʳ = deg2rad(φ)
    qt = 0.018 * exp(-(φʳ/(2π/9))^4) * exp(-((η-1)*p₀/34000)^2)
    return ifelse(η > 0.1, qt, 1e-12)
end
function temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z); Tᵥ/(1 + ε_v*specific_humidity(λ, φ, z))
end
function potential_temperature(λ, φ, z)
    temperature(λ, φ, z)*(p₀/pressure(λ, φ, z))^κ
end
function zonal_velocity(λ, φ, z)
    _,_,_,∫τ₂ = τ_and_integrals(z); T = temperature(λ, φ, z)
    U = g/a * K * ∫τ₂ * dF(φ) * T
    rcosφ = a*cosd(φ); Ωrcosφ = Ω*rcosφ
    u_b = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ*U)
    uₚ=1.0; rₚ=0.1; λₚ=π/9; φₚ=2π/9; zₚ=15000.0
    φʳ=deg2rad(φ); λʳ=deg2rad(λ)
    gc = acos(sin(φₚ)*sin(φʳ) + cos(φₚ)*cos(φʳ)*cos(λʳ-λₚ))/rₚ
    taper = ifelse(z < zₚ, 1 - 3*(z/zₚ)^2 + 2*(z/zₚ)^3, 0.0)
    u_b + ifelse(gc < 1, uₚ * taper * exp(-gc^2), 0.0)
end

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

grid = LatitudeLongitudeGrid(arch; size=(360,160,64), halo=(5,5,5),
                             longitude=(0,360), latitude=(-80,80), z=(0,30kilometers))
coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)
T₀_ref = 250.0; θ_ref(z) = T₀_ref*exp(g*z/(cᵖᵈ*T₀_ref))
dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure=p₀, reference_potential_temperature=θ_ref)

# ============================================================================
# THE KEY DIFFERENCE: ConstantRateCondensateFormation(0) → no condensation.
# Vapor is purely advected. No latent heat, no cloud water generation, no
# rain, no ice — pure passive moisture.
# ============================================================================
relaxation = ConstantRateCondensateFormation{Float32}(0.0f0)
cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

# Surface fluxes also OFF — passive-moisture pure dynamics
boundary_conditions = NamedTuple()

weno = WENO(); bp_weno = WENO(order=5, bounds=(0,1))
model = AtmosphereModel(grid; dynamics, coriolis, microphysics, boundary_conditions,
                        thermodynamic_constants=constants,
                        momentum_advection=weno,
                        scalar_advection=(ρθ=weno, ρqᵛ=bp_weno, ρqᶜˡ=bp_weno,
                                          ρqᶜⁱ=bp_weno, ρqʳ=bp_weno, ρqˢ=bp_weno),
                        timestepper=:AcousticRungeKutta3)

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density, qᵛ=specific_humidity)

Δt = 20seconds; stop_time = 5days
simulation = Simulation(model; Δt, stop_time)

iters=Int[]; ts=Float64[]; wmax=Float64[]; umax=Float64[]
qcl_max=Float64[]; qv_max=Float64[]; psurf=Float64[]; walls=Float64[]
M0_ref = Ref(0.0); Q0_ref = Ref(0.0)
wall0 = Ref(time_ns())

function diag_cb(sim)
    m = sim.model
    push!(iters, iteration(sim)); push!(ts, time(sim))
    wm = Float64(maximum(abs, interior(m.velocities.w)))
    um = Float64(maximum(abs, interior(m.velocities.u)))
    qcm = Float64(maximum(interior(m.microphysical_fields.ρqᶜˡ)))
    qvm = Float64(maximum(interior(m.moisture_density)))
    pm = Float64(minimum(view(interior(m.dynamics.pressure), :, :, 1)))
    M  = Float64(sum(interior(Breeze.AtmosphereModels.dynamics_density(m.dynamics))))
    if iteration(sim) == 0
        M0_ref[] = M; Q0_ref[] = qvm
    end
    push!(wmax, wm); push!(umax, um); push!(qcl_max, qcm); push!(qv_max, qvm)
    push!(psurf, pm); push!(walls, (time_ns()-wall0[])/1e9)
    @info @sprintf("[passive-q] iter=%6d t=%6.3fd Δt=%5.1fs max|u|=%.2f max|w|=%.3e max(ρqcl)=%.2e max(ρqv)=%.3e p_surf=%.0f ΔM/M0=%.2e wall=%.0fs",
                   iteration(sim), time(sim)/86400, sim.Δt, um, wm, qcm, qvm, pm,
                   (M-M0_ref[])/M0_ref[], walls[end])
    flush(stdout)
end
add_callback!(simulation, diag_cb, IterationInterval(500))

@info "Starting passive-moisture moist BW (microphysics OFF), Δt=$(Δt)s, $(stop_time/86400) days"
wall0[] = time_ns()
crashed = false
try
    run!(simulation)
    @info "[passive-q] RUN COMPLETED"
catch e
    crashed = true
    @error "[passive-q] RUN FAILED" e
end

jldsave(STEM*"_diagnostics.jld2";
        iters=iters, t=ts, wmax=wmax, umax=umax, qcl_max=qcl_max, qv_max=qv_max,
        psurf_min=psurf, walls=walls, crashed=crashed)
@info "Done."
