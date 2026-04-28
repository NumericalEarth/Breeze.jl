#####
##### Tier-3b A/B: rico (simplified — uses SaturationAdjustment instead of OneMomentCloudMicrophysics)
#####
##### After Blockers 3 & 5 fixed, RICO's BulkSensibleHeatFlux + BulkVaporFlux setup
##### should now run. We use SaturationAdjustment for simplicity.
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Oceananigans
using Oceananigans.Units
using AtmosphericProfilesLibrary
using CUDA

const arch = CUDA.functional() ? GPU() : CPU()
const constants = ThermodynamicConstants()

# Reduced resolution: 32×32×40 (instead of full RICO ~120×120×100), 5 minutes
const Nx, Ny, Nz = 32, 32, 40
const Lx = Ly = 6400  # m (4× coarser than RICO's 12.8 km / 256)
const Ltop = 4000     # m

build_grid() = RectilinearGrid(arch;
                               size = (Nx, Ny, Nz), halo = (5, 5, 5),
                               x = (0, Lx), y = (0, Ly), z = (0, Ltop),
                               topology = (Periodic, Periodic, Bounded))

const Cᴰ = 1.229e-3
const Cᵀ = 1.094e-3
const Cᵛ = 1.133e-3
const T₀ = 299.8
const p₀_rico = 101540.0
const θ₀_rico = 297.9

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = p₀_rico,
                                     potential_temperature = θ₀_rico)
    dynamics = AnelasticDynamics(reference_state)
    coriolis = FPlane(f = 4.5e-5)

    ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
    ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
    ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ,
                                                                   surface_temperature = T₀))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = Cᵛ,
                                                              surface_temperature = T₀))

    ρᵣ = reference_state.density
    ∂t_ρθ_radiation = Field{Nothing, Nothing, Center}(grid)
    set!(∂t_ρθ_radiation, ρᵣ * (-2.5 / day))
    ρθ_radiation = Forcing(∂t_ρθ_radiation)

    forcing = (; ρθ = ρθ_radiation)
    boundary_conditions = (; ρe = ρe_bcs, ρqᵉ = ρqᵉ_bcs, ρu = ρu_bcs, ρv = ρv_bcs)

    model = AtmosphereModel(grid; dynamics, coriolis,
                            momentum_advection = WENO(order = 9),
                            scalar_advection = WENO(order = 5),
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                            forcing, boundary_conditions)

    FT = eltype(grid)
    θ_init = AtmosphericProfilesLibrary.Rico_θ_liq_ice(FT)
    qᵗ_init = AtmosphericProfilesLibrary.Rico_q_tot(FT)
    u_init = AtmosphericProfilesLibrary.Rico_u(FT)
    v_init = AtmosphericProfilesLibrary.Rico_v(FT)
    set!(model;
         θ = (x, y, z) -> θ_init(z),
         qᵗ = (x, y, z) -> qᵗ_init(z),
         u = (x, y, z) -> u_init(z),
         v = (x, y, z) -> v_init(z))
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                          damping = KlempDivergenceDamping(coefficient = 0.1))
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = p₀_rico,
                                    reference_potential_temperature = θ₀_rico)
    coriolis = FPlane(f = 4.5e-5)

    ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
    ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
    ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ,
                                                                   surface_temperature = T₀))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = Cᵛ,
                                                              surface_temperature = T₀))

    ρᵣ = dynamics.reference_state isa Nothing ? nothing : nothing  # placeholder
    boundary_conditions = (; ρe = ρe_bcs, ρqᵉ = ρqᵉ_bcs, ρu = ρu_bcs, ρv = ρv_bcs)

    # Build model with no forcing (radiation will be added after model construction
    # if needed; for the smoke test we just want stability).
    model = AtmosphereModel(grid; dynamics, coriolis,
                            momentum_advection = WENO(order = 9),
                            scalar_advection = WENO(order = 5),
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                            boundary_conditions,
                            timestepper = :AcousticRungeKutta3)

    ref = model.dynamics.reference_state
    FT = eltype(grid)
    θ_init = AtmosphericProfilesLibrary.Rico_θ_liq_ice(FT)
    qᵗ_init = AtmosphericProfilesLibrary.Rico_q_tot(FT)
    u_init = AtmosphericProfilesLibrary.Rico_u(FT)
    v_init = AtmosphericProfilesLibrary.Rico_v(FT)
    set!(model;
         θ = (x, y, z) -> θ_init(z),
         qᵗ = (x, y, z) -> qᵗ_init(z),
         u = (x, y, z) -> u_init(z),
         v = (x, y, z) -> v_init(z),
         ρ = ref.density)
    return model
end

result = run_pair("rico";
                  build_anelastic, build_compressible,
                  Δt_anel = 5.0,
                  Δt_comp = 0.3,
                  stop_time = 300.0,
                  callback_iters = 50,
                  notes = "Reduced 32×32×40 (Δx=200m, Δz=100m); SaturationAdjustment instead of OneMomentCloudMicrophysics; radiation forcing omitted on Comp side for simplicity")

write_row(result)
