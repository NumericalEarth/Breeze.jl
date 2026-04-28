#####
##### Moist BW WITHOUT surface fluxes — isolation test for Q2.
#####
##### Same as moist_bw_15day.jl but with `boundary_conditions = NamedTuple()`
##### so all surface BCs default to no-flux. The moisture profile is set by
##### the analytic IC and is then advected/condensed by the dynamics +
##### microphysics with no surface forcing.
#####
##### If THIS runs to 15 days, surface fluxes are the dominant failure
##### mode in the moist BW. If this also fails, microphysics + dynamics
##### + substepper interaction is the issue independent of fluxes.
#####
##### Run length kept to 5 days for diagnostic — ample time to see whether
##### the failure-day-3 timescale shifts.
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
const RUN_LABEL = "moist_bw_dt37_ns24"
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

Nλ = 360; Nφ = 160; Nz = 64; H = 30kilometers

grid = LatitudeLongitudeGrid(arch; size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                             longitude=(0,360), latitude=(-80,80), z=(0,H))

Tᴱ = 310.0; Tᴾ = 240.0; Tₘ = (Tᴱ+Tᴾ)/2; Γ = 0.005; K = 3; b = 2; ε_v = 0.608

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
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g/Rᵈ*(∫τ₁ - ∫τ₂*F(φ)))
end
density(λ,φ,z) = pressure(λ,φ,z)/(Rᵈ*virtual_temperature(λ,φ,z))
function specific_humidity(λ,φ,z)
    p = pressure(λ,φ,z); η = p/p₀; φʳ = deg2rad(φ)
    q_trop = 0.018 * exp(-(φʳ/(2π/9))^4) * exp(-((η-1)*p₀/34000)^2)
    return ifelse(η > 0.1, q_trop, 1e-12)
end
function temperature(λ, φ, z)
    Tᵥ = virtual_temperature(λ, φ, z)
    return Tᵥ / (1 + ε_v * specific_humidity(λ, φ, z))
end
function potential_temperature(λ, φ, z)
    return temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ
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
    return u_b + ifelse(gc < 1, uₚ * taper * exp(-gc^2), 0.0)
end

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)
T₀_ref = 250.0; θ_ref(z) = T₀_ref*exp(g*z/(cᵖᵈ*T₀_ref))
dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=24);
                                surface_pressure=p₀, reference_potential_temperature=θ_ref)

τ_relax = 200.0
relaxation = ConstantRateCondensateFormation(1/τ_relax)
cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

# THE KEY DIFFERENCE: NO surface flux BCs
boundary_conditions = NamedTuple()

weno = WENO(); bp_weno = WENO(order=5, bounds=(0,1))

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, boundary_conditions,
                        thermodynamic_constants=constants,
                        momentum_advection=weno,
                        scalar_advection=(ρθ=weno, ρqᵛ=bp_weno, ρqᶜˡ=bp_weno,
                                          ρqᶜⁱ=bp_weno, ρqʳ=bp_weno, ρqˢ=bp_weno),
                        timestepper=:AcousticRungeKutta3)
set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density, qᵛ=specific_humidity)

Δt = 37seconds; stop_time = 1day
simulation = Simulation(model; Δt, stop_time)

diag_iters=Int[]; diag_t=Float64[]; diag_wmax=Float64[]; diag_umax=Float64[]
diag_qcl_max=Float64[]; diag_psurf=Float64[]; diag_M=Float64[]; diag_wall=Float64[]
wall0 = Ref(time_ns())
ρ_field = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
ρθ_field = model.formulation.potential_temperature_density
M0 = Float64(sum(interior(ρ_field)))

function diag_cb(sim)
    m = sim.model
    push!(diag_iters, iteration(sim)); push!(diag_t, time(sim))
    wm = Float64(maximum(abs, interior(m.velocities.w)))
    um = Float64(maximum(abs, interior(m.velocities.u)))
    qm = Float64(maximum(interior(m.microphysical_fields.ρqᶜˡ)))
    pm = Float64(minimum(view(interior(m.dynamics.pressure), :, :, 1)))
    M  = Float64(sum(interior(ρ_field)))
    push!(diag_wmax, wm); push!(diag_umax, um); push!(diag_qcl_max, qm)
    push!(diag_psurf, pm); push!(diag_M, M)
    push!(diag_wall, (time_ns()-wall0[])/1e9)
    @info @sprintf("[no-flux] iter=%6d t=%6.3fd Δt=%5.1fs max|u|=%.2f max|w|=%.3e max(ρqcl)=%.2e p_surf=%.0f wall=%.0fs",
                   iteration(sim), time(sim)/86400, sim.Δt, um, wm, qm, pm, diag_wall[end])
    flush(stdout)
end
add_callback!(simulation, diag_cb, IterationInterval(500))

@info "Starting moist BW NO-SURFACE-FLUX run, Δt=$(Δt)s, $(stop_time/86400) days"
wall0[] = time_ns()
try
    run!(simulation)
    @info "[no-flux] RUN COMPLETED"
catch e
    @error "[no-flux] RUN FAILED" e
end

jldsave(STEM*"_diagnostics.jld2";
        iters=diag_iters, t=diag_t, wmax=diag_wmax, umax=diag_umax,
        qcl_max=diag_qcl_max, psurf_min=diag_psurf, total_mass=diag_M,
        wall=diag_wall, M0=M0)

@info "Done."
