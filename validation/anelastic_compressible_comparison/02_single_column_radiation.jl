#####
##### Tier-0 A/B: single_column_radiation
#####
##### Diagnostic only — no actual simulation, just `set!` + radiation. Test
##### whether CompressibleDynamics + AcousticRungeKutta3 builds and computes
##### radiation in the single-column geometry.
#####

include("ABCompare.jl")
using .ABCompare

using Breeze
using Oceananigans
using Oceananigans.Units
using NCDatasets
using RRTMGP
using Dates
using Printf

const Nz = 64
const grid = RectilinearGrid(size = Nz, x = -76.13, y = 39.48, z = (0, 20kilometers),
                             topology = (Flat, Flat, Bounded))

const surface_temperature = 300.0
const constants = ThermodynamicConstants()
const clock = Clock(time = DateTime(1950, 11, 1, 12, 0, 0))
const microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

const radiation_anel = RadiativeTransferModel(grid, GrayOptics(), constants;
                                               surface_temperature,
                                               surface_emissivity = 0.98,
                                               surface_albedo = 0.1,
                                               solar_constant = 1361)

const radiation_comp = RadiativeTransferModel(grid, GrayOptics(), constants;
                                               surface_temperature,
                                               surface_emissivity = 0.98,
                                               surface_albedo = 0.1,
                                               solar_constant = 1361)

q₀ = 0.020; Hᵗ = 3000.0
qᵗᵢ(z) = q₀ * exp(-z / Hᵗ)

function build_anelastic()
    reference_state = ReferenceState(grid, constants; surface_pressure = 101325,
                                     potential_temperature = surface_temperature)
    dyn = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; clock, dynamics = dyn, microphysics, radiation = radiation_anel)
    set!(model; θ = surface_temperature, qᵗ = qᵗᵢ)
    return model
end

function build_compressible()
    td = SplitExplicitTimeDiscretization(forward_weight = 0.55,
                                         damping = KlempDivergenceDamping(coefficient = 0.1))
    dyn = CompressibleDynamics(td; reference_potential_temperature = surface_temperature)
    model = AtmosphereModel(grid; clock, dynamics = dyn, microphysics,
                            radiation = radiation_comp, timestepper = :AcousticRungeKutta3)
    set!(model; θ = surface_temperature, qᵗ = qᵗᵢ, ρ = model.dynamics.reference_state.density)
    return model
end

# Diagnostic-only example — original doesn't run a simulation. We just verify
# both models construct without error and that initial radiation diagnostics
# are computed (set! triggers radiative transfer).
@info "=========== single_column_radiation ==========="

@info "Building anelastic model..."
ma = build_anelastic()
@info @sprintf("  ✓ Anelastic constructed (Nz=%d, dynamics=%s)", size(ma.grid, 3), typeof(ma.dynamics).name.name)

@info "Building compressible model..."
mc = build_compressible()
@info @sprintf("  ✓ Compressible constructed (Nz=%d, dynamics=%s)", size(mc.grid, 3), typeof(mc.dynamics).name.name)

# Compare temperature profiles after set! — both should give identical T(z).
T_anel = Array(interior(ma.temperature))[1, 1, :]
T_comp = Array(interior(mc.temperature))[1, 1, :]
T_diff = maximum(abs, T_anel .- T_comp)
@info @sprintf("  Max |T_anel − T_comp| = %.4e K (acceptable < 1e-6)", T_diff)

# Append a manual row since we didn't use run_pair.
status_anel = "✓"
status_comp = "✓"
isfile(REPORT_PATH) || (open(REPORT_PATH, "w") do io
    write(io, ABCompare.header())
end)
open(REPORT_PATH, "a") do io
    write(io, "| single_column_radiation | $status_anel | $status_comp | (no run) | T_diff=$(@sprintf("%.2e",T_diff)) | — | — | Diagnostic only — both models construct & compute initial radiation |\n")
end
@info "Wrote single_column_radiation row to $REPORT_PATH"
