#####
##### Tier-1 A/B: Skamarock-Klemp 1994 inertia-gravity wave
#####
##### 300x10 km, Δx=1km, U₀=20 m/s background, sin(π z/Lz) θ pulse.
##### CFL guidance: Δx/cs = 1000/350 ≈ 2.86 s. Klemp limit ≈ 1.4·Δx/cs ≈ 4 s.
##### We use Δt = 4 s for the substepper, Δt = 25 s for anelastic.
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf

const Nx, Nz = 300, 10
const Lx, Lz = 300kilometers, 10kilometers
const θ₀, U, N², Δθ, a, x₀ = 300.0, 20.0, 0.01^2, 0.01, 5000.0, Lx/3
const constants = ThermodynamicConstants()
const g = constants.gravitational_acceleration
const surface_pressure = 1e5

θᵇᵍ(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θᵇᵍ(z) + Δθ * sin(π * z / Lz) / (1 + (x - x₀)^2 / a^2)

function build_grid()
    RectilinearGrid(CPU(); size = (Nx, Nz), halo = (5, 5),
                    x = (0, Lx), z = (0, Lz),
                    topology = (Periodic, Flat, Bounded))
end

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure,
                                     potential_temperature = θ₀)
    dyn = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; advection = WENO(), dynamics = dyn)
    set!(model; θ = θᵢ, u = U)
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                          damping = KlempDivergenceDamping(coefficient = 0.1))
    dyn = CompressibleDynamics(td;
                               surface_pressure,
                               reference_potential_temperature = θᵇᵍ)
    model = AtmosphereModel(grid; advection = WENO(), dynamics = dyn,
                            timestepper = :AcousticRungeKutta3)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, u = U, ρ = ref.density)
    return model
end

result = run_pair("inertia_gravity_wave";
                  build_anelastic, build_compressible,
                  Δt_anel  = 25.0,    # advective Δt: cfl=0.5 × Lz/Nz / U = 25
                  Δt_comp  = 4.0,     # CFL ≈ 1.4 (Klemp-damping limit)
                  stop_time = 3000.0,
                  callback_iters = 50,
                  notes = "Skamarock-Klemp 1994; substepper uses Klemp(0.1), Ns=12")

write_row(result)
