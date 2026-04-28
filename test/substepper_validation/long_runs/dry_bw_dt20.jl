#####
##### DRY BW lat-lon at Δt=20s — diagnostic for "small-Δt × lat-lon" hypothesis.
#####
##### Same BW IC, same lat-lon grid (360×160×64 DCMIP-2016 resolution),
##### same reference state, but with `microphysics = nothing` AND `qᵛ ≡ 0`.
##### Pure dry BW dynamics on the lat-lon grid at Δt=20s for 5 days.
#####
##### Pre-fix DRY BW production runs at Δt=225s succeed for 14 days. The
##### question here: does the substepper handle the lat-lon grid at the
##### much smaller Δt=20s, in absence of microphysics?
#####
##### If THIS runs to 5 days clean: lat-lon × small-Δt is fine, the
##### moist BW NaN at day 4 is specifically a microphysics-dynamics
##### coupling issue at lat-lon × Δt=20s.
##### If THIS NaNs: the substepper has a small-Δt × lat-lon problem
##### that microphysics merely amplifies / reveals.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = @__DIR__
const RUN_LABEL = "dry_bw_dt20"
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

# Same DCMIP IC formulas as moist BW, but with qᵛ ≡ 0 → all moisture-related
# terms collapse to dry. virtual_temperature(=actual T) since no qᵛ.
Tᴱ = 310.0; Tᴾ = 240.0; Tₘ = (Tᴱ+Tᴾ)/2; Γ = 0.005; K = 3; b = 2

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

function temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g/Rᵈ*(∫τ₁ - ∫τ₂*F(φ)))
end
density(λ,φ,z) = pressure(λ,φ,z)/(Rᵈ*temperature(λ,φ,z))
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

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)
T₀_ref = 250.0; θ_ref(z) = T₀_ref*exp(g*z/(cᵖᵈ*T₀_ref))
dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure=p₀, reference_potential_temperature=θ_ref)

# THE DIFFERENCE FROM moist_bw_no_microphysics.jl: also no qᵛ initial.
# (The model still allocates a qᵛ field but it stays at 0.)
microphysics = nothing
boundary_conditions = NamedTuple()

weno = WENO()

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, boundary_conditions,
                        thermodynamic_constants=constants,
                        momentum_advection=weno,
                        scalar_advection=(ρθ=weno,),
                        timestepper=:AcousticRungeKutta3)
set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density)
# Note: NO `qᵛ=...` — keeps qᵛ at 0.

Δt = 20seconds; stop_time = 5days
simulation = Simulation(model; Δt, stop_time)

diag_iters=Int[]; diag_t=Float64[]; diag_wmax=Float64[]; diag_umax=Float64[]
diag_psurf=Float64[]; diag_M=Float64[]; diag_wall=Float64[]
wall0 = Ref(time_ns())
ρ_field = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
M0 = Float64(sum(interior(ρ_field)))

function diag_cb(sim)
    m = sim.model
    push!(diag_iters, iteration(sim)); push!(diag_t, time(sim))
    wm = Float64(maximum(abs, interior(m.velocities.w)))
    um = Float64(maximum(abs, interior(m.velocities.u)))
    pm = Float64(minimum(view(interior(m.dynamics.pressure), :, :, 1)))
    M  = Float64(sum(interior(ρ_field)))
    push!(diag_wmax, wm); push!(diag_umax, um)
    push!(diag_psurf, pm); push!(diag_M, M)
    push!(diag_wall, (time_ns()-wall0[])/1e9)
    @info @sprintf("[dry-Δt20] iter=%6d t=%6.3fd Δt=%5.1fs max|u|=%.2f max|w|=%.3e p_surf=%.0f wall=%.0fs",
                   iteration(sim), time(sim)/86400, sim.Δt, um, wm, pm, diag_wall[end])
    flush(stdout)
end
add_callback!(simulation, diag_cb, IterationInterval(500))

@info "Starting DRY BW @ Δt=20s on lat-lon, $(stop_time/86400) days"
wall0[] = time_ns()
try
    run!(simulation)
    @info "[dry-Δt20] RUN COMPLETED"
catch e
    @error "[dry-Δt20] RUN FAILED" e
end

jldsave(STEM*"_diagnostics.jld2";
        iters=diag_iters, t=diag_t, wmax=diag_wmax, umax=diag_umax,
        psurf_min=diag_psurf, total_mass=diag_M,
        wall=diag_wall, M0=M0)

@info "Done."
