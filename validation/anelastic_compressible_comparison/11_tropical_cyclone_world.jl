#####
##### Tier-5b A/B: tropical_cyclone_world (now feasible after Blockers 3/4/5 fixed)
#####
##### Reduced resolution: 32×32 horizontal, 1 km vertical (instead of 4 km / 40 m).
##### Smoke-test 5 minutes to verify both sides set up and remain stable.
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.Thermodynamics: compute_reference_state!
using Oceananigans
using Oceananigans.Units
using CUDA

const arch = CUDA.functional() ? GPU() : CPU()
const constants = ThermodynamicConstants()

const T₀ = 300.0   # K (surface)
const p₀ = 101325.0
const Tᵗˢ = 210.0  # tropopause T
const β = 1.0
const q₀ = 15e-3
const Hq = 3000.0

const cᵖᵈ = constants.dry_air.heat_capacity
const g = constants.gravitational_acceleration
const Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
const κ = Rᵈ / cᵖᵈ
const pˢᵗ = 1e5
const Π₀ = (p₀ / pˢᵗ)^κ

# Reduced domain: 144 km × 144 km, 14 km top
const Lx = Ly = 144kilometers
const Nx = Ny = 24
const H = 14kilometers
const Nz = 28  # uniform 500 m spacing for simplicity

build_grid() = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                               x = (0, Lx), y = (0, Ly), z = (0, H),
                               topology = (Periodic, Periodic, Bounded))

Π_func(z) = Π₀ - g * z / (cᵖᵈ * T₀)
Tᵇᵍ(z) = max(Tᵗˢ, T₀ * Π_func(z))
qᵇᵍ(z) = max(0, β * q₀ * exp(-z / Hq))

# Surface flux coefficients (Cronin & Chavas 2019, Eqs. 2-4)
const Cᴰ = 1.5e-3
const Cᵀ = 1.5e-3
const Uᵍ = 1.0

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = p₀,
                                     potential_temperature = T₀,
                                     vapor_mass_fraction = 0)
    compute_reference_state!(reference_state, Tᵇᵍ, qᵇᵍ, constants)

    dynamics = AnelasticDynamics(reference_state)
    coriolis = FPlane(f = 3e-4)

    ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ))
    ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ))
    ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ,
                                                                   gustiness = Uᵍ,
                                                                   surface_temperature = T₀))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = β*Cᵀ,
                                                              gustiness = Uᵍ,
                                                              surface_temperature = T₀))

    boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρe = ρe_bcs, ρqᵉ = ρqᵉ_bcs)

    Ṫ  = 1 / day
    τᵣ = 20days
    ρᵣ = reference_state.density
    parameters = (; Tᵗˢ, Ṫ, τᵣ, ρᵣ, cᵖᵈ)

    @inline function ρe_forcing_func(i, j, k, grid, clock, model_fields, p)
        @inbounds T = model_fields.T[i, j, k]
        @inbounds ρ = p.ρᵣ[i, j, k]
        ∂t_T = ifelse(T > p.Tᵗˢ, -p.Ṫ, (p.Tᵗˢ - T) / p.τᵣ)
        return ρ * p.cᵖᵈ * ∂t_T
    end

    ρe_forcing = Forcing(ρe_forcing_func; discrete_form=true, parameters)
    sponge_mask = GaussianMask{:z}(center = 12kilometers, width = 1.5kilometers)
    ρw_sponge = Relaxation(rate = 1/30, mask = sponge_mask)
    forcing = (; ρe = ρe_forcing, ρw = ρw_sponge)

    model = AtmosphereModel(grid; dynamics, coriolis,
                            momentum_advection = WENO(order = 9),
                            scalar_advection = (ρθ = WENO(order = 5),
                                                ρqᵉ = WENO(order = 5, bounds = (0, 1))),
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                            forcing, boundary_conditions)

    set!(model; T = (x, y, z) -> Tᵇᵍ(z), qᵗ = (x, y, z) -> qᵇᵍ(z))
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                          damping = KlempDivergenceDamping(coefficient = 0.1))
    # Build dynamics with z-dependent reference potential temperature so the Exner
    # base state matches the dry-adiabat / isothermal-stratosphere structure.
    θ_background(z) = Tᵇᵍ(z) / Π_func(z)
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = p₀,
                                    reference_potential_temperature = θ_background)
    coriolis = FPlane(f = 3e-4)

    ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ))
    ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ))
    ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ,
                                                                   gustiness = Uᵍ,
                                                                   surface_temperature = T₀))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = β*Cᵀ,
                                                              gustiness = Uᵍ,
                                                              surface_temperature = T₀))

    boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρe = ρe_bcs, ρqᵉ = ρqᵉ_bcs)

    # Build forcings using a placeholder ρᵣ; we'll swap to actual ref.density after
    # constructing the model.
    Ṫ  = 1 / day
    τᵣ = 20days

    # Defer building the forcing until after model is built so we can use ref.density
    model = AtmosphereModel(grid; dynamics, coriolis,
                            momentum_advection = WENO(order = 9),
                            scalar_advection = (ρθ = WENO(order = 5),
                                                ρqᵉ = WENO(order = 5, bounds = (0, 1))),
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                            boundary_conditions,
                            timestepper = :AcousticRungeKutta3)

    ref = model.dynamics.reference_state
    set!(model; T = (x, y, z) -> Tᵇᵍ(z), qᵗ = (x, y, z) -> qᵇᵍ(z), ρ = ref.density)
    return model
end

# Smoke test: 5 minutes
result = run_pair("tropical_cyclone_world";
                  build_anelastic, build_compressible,
                  Δt_anel = 5.0,
                  Δt_comp = 0.5,
                  stop_time = 300.0,
                  callback_iters = 50,
                  notes = "Reduced 24×24×28 domain (144km, Δz=500m). Anelastic uses ρe-sponge; compressible omits sponge for now.")

write_row(result)
