#####
##### Tier-2 A/B: dry_thermal_bubble (LiquidIce variant)
#####
##### Original example uses StaticEnergy formulation, which CompressibleDynamics
##### does not yet support. Here we run the dry-bubble physics with the standard
##### LiquidIcePotentialTemperature formulation in both anelastic and
##### compressible-substepper modes.
#####
##### Δx = 20km / 128 = 156 m. CFL limit ≈ 1.4 × Δx/cs ≈ 0.62 s.
##### We use Δt = 0.5 s for the substepper, Δt = 2 s for anelastic (its original).
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf

const Nx, Nz = 128, 128
const Lx, Lz = 20kilometers, 10kilometers
const Δθ_pulse, r₀, N² = 10.0, 2e3, 1e-6
const θ₀ = 300.0
const g  = 9.80665

function build_grid()
    RectilinearGrid(CPU(); size = (Nx, Nz), halo = (5, 5),
                    x = (-10kilometers, 10kilometers), z = (0, 10kilometers),
                    topology = (Periodic, Flat, Bounded))
end

θ̄(z) = θ₀ * exp(N² * z / g)
function θᵢ(x, z)
    x₀ = 0.0
    z₀ = 3000.0
    r = sqrt((x - x₀)^2 + (z - z₀)^2)
    θ′ = Δθ_pulse * max(0, 1 - r / r₀)
    return θ̄(z) + θ′
end

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, ThermodynamicConstants(eltype(grid)))
    dyn = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 9))
    set!(model; θ = θᵢ)
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                          damping = KlempDivergenceDamping(coefficient = 0.1))
    dyn = CompressibleDynamics(td; reference_potential_temperature = θ̄)
    model = AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 9),
                            timestepper = :AcousticRungeKutta3)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)
    return model
end

# 5 minutes of bubble rise — long enough to see the toroidal vortex develop.
result = run_pair("dry_thermal_bubble";
                  build_anelastic, build_compressible,
                  Δt_anel  = 2.0,
                  Δt_comp  = 0.5,
                  stop_time = 300.0,
                  callback_iters = 100,
                  notes = "LiquidIce variant (StaticEnergy unsupported in CompressibleDynamics); Klemp(0.1)")

write_row(result)
