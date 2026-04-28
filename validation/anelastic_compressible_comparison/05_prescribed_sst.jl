#####
##### Tier-2 A/B: prescribed_sea_surface_temperature
#####
##### LES with bulk-flux BCs over an SST front. Original Δt=10s; substepper
##### CFL limit ≈ 0.62s on Δx=156m. Run a short window (300s) just to
##### verify stability + bulk-flux BCs propagate correctly through both.
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Oceananigans
using Oceananigans.Units
using Printf

const constants = ThermodynamicConstants()
const p₀, θ₀ = 101325.0, 285.0
const ΔT, Uᵍ = 4.0, 1e-2

T₀(x) = θ₀ + ΔT / 2 * sign(cos(2π * x / 20kilometers))

function build_grid()
    RectilinearGrid(size = (128, 128), halo = (5, 5),
                    x = (-10kilometers, 10kilometers),
                    z = (0, 10kilometers),
                    topology = (Periodic, Flat, Bounded))
end

function make_bcs(grid)
    coef = PolynomialCoefficient(roughness_length = 1.5e-4)
    filtered_velocities = FilteredSurfaceVelocities(grid; filter_timescale = 1hour)
    ρu_surface_flux = BulkDrag(coefficient = coef; gustiness = Uᵍ,
                               surface_temperature = T₀, filtered_velocities)
    ρv_surface_flux = BulkDrag(coefficient = coef; gustiness = Uᵍ,
                               surface_temperature = T₀, filtered_velocities)
    ρθ_surface_flux  = BulkSensibleHeatFlux(coefficient = coef; gustiness = Uᵍ,
                                            surface_temperature = T₀, filtered_velocities)
    ρqᵗ_surface_flux = BulkVaporFlux(coefficient = coef; gustiness = Uᵍ,
                                     surface_temperature = T₀, filtered_velocities)
    return (
        ρu  = FieldBoundaryConditions(bottom = ρu_surface_flux),
        ρv  = FieldBoundaryConditions(bottom = ρv_surface_flux),
        ρθ  = FieldBoundaryConditions(bottom = ρθ_surface_flux),
        ρqᵗ = FieldBoundaryConditions(bottom = ρqᵗ_surface_flux),
    )
end

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = p₀,
                                     potential_temperature = θ₀)
    dyn = AnelasticDynamics(reference_state)
    bcs = make_bcs(grid)
    model = AtmosphereModel(grid;
                            momentum_advection = WENO(order = 9),
                            scalar_advection   = WENO(order = 5),
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                            dynamics = dyn,
                            boundary_conditions = bcs)
    set!(model; θ = θ₀, u = 1.0)
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                          damping = KlempDivergenceDamping(coefficient = 0.1))
    dyn = CompressibleDynamics(td;
                               surface_pressure = p₀,
                               reference_potential_temperature = θ₀)
    bcs = make_bcs(grid)
    model = AtmosphereModel(grid;
                            momentum_advection = WENO(order = 9),
                            scalar_advection   = WENO(order = 5),
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                            dynamics = dyn,
                            boundary_conditions = bcs,
                            timestepper = :AcousticRungeKutta3)
    ref = model.dynamics.reference_state
    set!(model; θ = θ₀, qᵗ = 0.0, u = 1.0, ρ = ref.density)
    return model
end

# Short 5-minute window; full 4h is too slow at substepper Δt=0.5.
result = run_pair("prescribed_sea_surface_temperature";
                  build_anelastic, build_compressible,
                  Δt_anel = 5.0,
                  Δt_comp = 0.5,
                  stop_time = 300.0,
                  callback_iters = 50,
                  notes = "Both sides run after BulkSensibleHeatFlux dispatch fix; substepper Δt=0.5 limited by Δx/cs CFL")

write_row(result)
