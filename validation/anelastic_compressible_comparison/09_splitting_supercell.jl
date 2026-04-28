#####
##### Tier-4 A/B: splitting_supercell (Kessler + deep convection)
#####
##### GPU, Δx=1km, 168×168×40, Δt=2s. CFL with Δt=2 is 0.7 — well under
##### substepper limit. Smoke test 60s window for stability + accuracy.
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Oceananigans
using Oceananigans.Units
using AtmosphericProfilesLibrary
using CUDA
using Printf

const arch = CUDA.functional() ? GPU() : CPU()
const constants = ThermodynamicConstants()
const Nx, Ny, Nz = 168, 168, 40

build_grid() = RectilinearGrid(arch;
                               size = (Nx, Ny, Nz), halo = (5, 5, 5),
                               x = (-84kilometers, 84kilometers),
                               y = (-84kilometers, 84kilometers),
                               z = (0, 20kilometers),
                               topology = (Periodic, Periodic, Bounded))

# Skamarock-Klemp DCMIP2016 supercell — abbreviated background profile
const θ_tr, T_tr, p_eq, q_tr = 343.0, 213.0, 100000.0, 1.4e-2
const Γ = 0.005
function θ_background(z)
    if z <= 12000
        return 300 + Γ * z
    else
        return 343
    end
end

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = 100000,
                                     potential_temperature = 300)
    dyn = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid;
                            dynamics = dyn,
                            advection = WENO(order = 5),
                            thermodynamic_constants = constants)
    set!(model; θ = (x, y, z) -> θ_background(z), qᵗ = q_tr)
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                          damping = KlempDivergenceDamping(coefficient = 0.1))
    dyn = CompressibleDynamics(td;
                               surface_pressure = 100000,
                               reference_potential_temperature = θ_background)
    model = AtmosphereModel(grid;
                            dynamics = dyn,
                            advection = WENO(order = 5),
                            thermodynamic_constants = constants,
                            timestepper = :AcousticRungeKutta3)
    ref = model.dynamics.reference_state
    set!(model; θ = (x, y, z) -> θ_background(z), qᵗ = q_tr, ρ = ref.density)
    return model
end

# 60s smoke test (full 2h is ~7 minutes wall, too long for an A/B comparison run).
result = run_pair("splitting_supercell";
                  build_anelastic, build_compressible,
                  Δt_anel = 2.0,
                  Δt_comp = 2.0,
                  stop_time = 60.0,
                  callback_iters = 5,
                  notes = "60s smoke; no Kessler microphysics in this minimal port (background sounding only)")

write_row(result)
