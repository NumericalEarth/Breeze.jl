#####
##### Tier-0 A/B: stationary_parcel_model
#####
##### 1x1x1 grid, no dynamics, just microphysics. Smoke test for whether
##### CompressibleDynamics + AcousticRungeKutta3 even compiles & runs at this
##### degenerate grid size.
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Oceananigans
using Oceananigans.Units
using CloudMicrophysics
using Printf

const grid = RectilinearGrid(CPU(); size = (1, 1, 1), x = (0, 1), y = (0, 1), z = (0, 1),
                             topology = (Periodic, Periodic, Bounded))

const constants = ThermodynamicConstants()

const BREEZE_MP_EXT = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)

import CloudMicrophysics.Parameters as CMP
const CLOUD_LIQUID = CMP.CloudLiquid{Float64}(τ_relax = 2.0, ρw = 1000.0,
                                              r_eff = 1e-5, N_0 = 5e8)
const MP_CATEGORIES = BREEZE_MP_EXT.one_moment_cloud_microphysics_categories(cloud_liquid = CLOUD_LIQUID)
const MICROPHYSICS = BREEZE_MP_EXT.OneMomentCloudMicrophysics(; categories = MP_CATEGORIES,
                                                               precipitation_boundary_condition = ImpenetrableBoundaryCondition())

function build_anelastic()
    reference_state = ReferenceState(grid, constants; surface_pressure = 101325,
                                     potential_temperature = 300)
    dyn = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics = dyn, thermodynamic_constants = constants,
                            microphysics = MICROPHYSICS)
    set!(model; θ = 300, qᵗ = 0.030)
    return model
end

function build_compressible()
    td = SplitExplicitTimeDiscretization(forward_weight = 0.55,
                                         damping = KlempDivergenceDamping(coefficient = 0.1))
    dyn = CompressibleDynamics(td; reference_potential_temperature = 300.0)
    model = AtmosphereModel(grid; dynamics = dyn, thermodynamic_constants = constants,
                            microphysics = MICROPHYSICS, timestepper = :AcousticRungeKutta3)
    set!(model; θ = 300, qᵗ = 0.030, ρ = model.dynamics.reference_state.density)
    return model
end

result = run_pair("stationary_parcel_model";
                  build_anelastic, build_compressible,
                  Δt_anel = 1.0, Δt_comp = 1.0,
                  stop_time = 1000.0,
                  callback_iters = 100,
                  notes = "1×1×1 grid, OneMoment microphysics, supersaturated IC")

write_row(result; append = false)
