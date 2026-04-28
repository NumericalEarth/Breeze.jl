#####
##### Tier-3a A/B: bomex
#####
##### BOMEX shallow Cu LES — uses ρe-energy-forcing infrastructure for radiation
##### tendency. After Blockers 2/3 fixed (ρe-forcing was already supported in
##### LiquidIcePotentialTemperatureFormulation; BulkSensibleHeatFlux now
##### dispatches), the substepper port is straightforward.
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

# Reduced resolution for A/B run: full Bomex is 64×64×75 at Δx=100m, 6h.
# Use 32×32×40 at Δx=200m, run for 10 minutes.
const Nx, Ny, Nz = 32, 32, 40
const stop_t = 600.0  # 10 minutes — enough to see whether substepper holds together

build_grid() = RectilinearGrid(arch;
                               size = (Nx, Ny, Nz), halo = (5, 5, 5),
                               x = (0, 6400), y = (0, 6400), z = (0, 3000),
                               topology = (Periodic, Periodic, Bounded))

# BOMEX surface fluxes
const w′θ′ = 8e-3     # K m/s
const w′qᵗ′ = 5.2e-5  # m/s
const u★ = 0.28       # m/s

@inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2 + 1e-12)
@inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2 + 1e-12)

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = 101500,
                                     potential_temperature = 299.1)
    dynamics = AnelasticDynamics(reference_state)

    p₀ = reference_state.surface_pressure
    θ₀ = reference_state.potential_temperature
    FT = eltype(grid)
    q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)

    ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′θ′))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′qᵗ′))
    ρu_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρu_drag,
        field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★)))
    ρv_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρv_drag,
        field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★)))

    coriolis = FPlane(f=3.76e-5)

    model = AtmosphereModel(grid; dynamics, coriolis,
                            microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()),
                            advection = WENO(order=9),
                            boundary_conditions = (ρθ=ρθ_bcs, ρqᵉ=ρqᵉ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

    θˡⁱ₀ = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
    qᵗ₀ = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
    u₀ = AtmosphericProfilesLibrary.Bomex_u(FT)
    set!(model, θ = (x,y,z) -> θˡⁱ₀(z), qᵗ = (x,y,z) -> qᵗ₀(z), u = (x,y,z) -> u₀(z))
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                          damping = KlempDivergenceDamping(coefficient = 0.1))
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = 101500,
                                    reference_potential_temperature = 299.1)

    FT = eltype(grid)
    p₀ = 101500.0
    θ₀ = 299.1
    q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρ₀ = Breeze.Thermodynamics.density(FT(θ₀), FT(p₀), q₀, constants)

    ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′θ′))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′qᵗ′))
    ρu_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρu_drag,
        field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★)))
    ρv_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρv_drag,
        field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★)))

    coriolis = FPlane(f=3.76e-5)

    model = AtmosphereModel(grid; dynamics, coriolis,
                            microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()),
                            advection = WENO(order=9),
                            boundary_conditions = (ρθ=ρθ_bcs, ρqᵉ=ρqᵉ_bcs, ρu=ρu_bcs, ρv=ρv_bcs),
                            timestepper = :AcousticRungeKutta3)

    ref = model.dynamics.reference_state
    θˡⁱ₀ = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
    qᵗ₀ = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
    u₀ = AtmosphericProfilesLibrary.Bomex_u(FT)
    set!(model, θ = (x,y,z) -> θˡⁱ₀(z), qᵗ = (x,y,z) -> qᵗ₀(z), u = (x,y,z) -> u₀(z),
         ρ = ref.density)
    return model
end

# Δx = 200m, Δz = 75m. Anelastic Δt limit ~ Δx/U ~ 200/10 = 20s; use Δt=10.
# Compressible substepper: cs ~ 350, so Δt_max ≈ 1.4·min(Δx,Δz)/cs ≈ 1.4·75/350 = 0.3s.
result = run_pair("bomex";
                  build_anelastic, build_compressible,
                  Δt_anel = 10.0,
                  Δt_comp = 0.3,
                  stop_time = stop_t,
                  callback_iters = 50,
                  notes = "Reduced resolution (32×32×40 Δx=200m). Substepper Δt limited by Δz/cs.")

write_row(result)
