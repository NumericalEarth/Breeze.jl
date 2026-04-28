#####
##### Tier-3b A/B: cloudy_thermal_bubble (moist + saturation adjustment)
#####
##### Same dry-bubble dynamics as 04 but with saturation-adjustment microphysics.
##### Tests substepper compatibility with the moist sat-adjust path.
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf

const grid_kw = (size = (128, 128), halo = (5, 5),
                 x = (-10kilometers, 10kilometers), z = (0, 10kilometers),
                 topology = (Periodic, Flat, Bounded))
const thermodynamic_constants = ThermodynamicConstants()
const θ₀ = 300.0
const Δθ = 2.0
const r₀ = 2e3
const z₀ = 0.3 * 10e3

θ̄(z) = θ₀
function θᵢ(x, z)
    r = sqrt(x^2 + (z - z₀)^2)
    θ′ = Δθ * max(0, 1 - r / r₀)
    return θ̄(z) + θ′
end

build_grid() = RectilinearGrid(CPU(); grid_kw...)

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, thermodynamic_constants;
                                     surface_pressure = 1e5,
                                     potential_temperature = θ₀)
    dyn = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics = dyn, thermodynamic_constants,
                            advection = WENO(order = 9),
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()))
    set!(model; θ = θᵢ, qᵗ = 0.025)
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                          damping = KlempDivergenceDamping(coefficient = 0.1))
    dyn = CompressibleDynamics(td;
                               surface_pressure = 1e5,
                               reference_potential_temperature = θ₀)
    model = AtmosphereModel(grid; dynamics = dyn, thermodynamic_constants,
                            advection = WENO(order = 9),
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                            timestepper = :AcousticRungeKutta3)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, qᵗ = 0.025, ρ = ref.density)
    return model
end

result = run_pair("cloudy_thermal_bubble";
                  build_anelastic, build_compressible,
                  Δt_anel = 2.0,
                  Δt_comp = 0.2,
                  stop_time = 300.0,
                  callback_iters = 50,
                  notes = "Moist + sat-adjust; Δt 0.4s (CFL ≈ 0.9)")

write_row(result)
