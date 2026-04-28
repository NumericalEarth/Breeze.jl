#####
##### validation/substepping/43_klemp_3d_rest.jl
#####
##### A/B test: rest-atmosphere drift at production (Δt=20 s, ω=0.55) with
#####   - NoDivergenceDamping (control — known to amplify ~1.77×/step)
#####   - 3D isotropic KlempDivergenceDamping (new implementation)
#####
##### Pass criterion (SUBSTEPPER_INSTABILITY_SUMMARY.md):
#####   "growth/step drops below 1.05"
#####

using Breeze
using Breeze: dynamics_density
using Breeze.CompressibleEquations: KlempDivergenceDamping, NoDivergenceDamping
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using CUDA
using Printf

const arch = CUDA.functional() ? GPU() : CPU()

const T₀ = 250.0
const Lz = 30e3
const Lh = 100e3
const Nx = Ny = 16
const Nz = 64
const G  = 9.80665
const cᵖᵈ = 1005.0

θ_ref(z) = T₀ * exp(G * z / (cᵖᵈ * T₀))

function build_model(damping)
    grid = RectilinearGrid(arch;
                           size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (0, Lh), y = (0, Lh), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(eltype(grid))
    # NoDD case overrides damping; nothing → use default (forward_weight=0.6,
    # KlempDivergenceDamping coef=0.1).
    td = damping === nothing ?
         SplitExplicitTimeDiscretization() :
         SplitExplicitTimeDiscretization(; substeps = nothing,
                                           forward_weight = 0.55,
                                           damping = damping)
    dyn = CompressibleDynamics(td;
                               reference_potential_temperature = θ_ref,
                               surface_pressure = 1e5,
                               standard_pressure = 1e5)
    return AtmosphereModel(grid; dynamics = dyn,
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function set_rest_state!(model)
    ref = model.dynamics.reference_state
    Rᵈ  = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    parent(model.dynamics.density) .= parent(ref.density)
    ρθ_field = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    parent(ρθ_field) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))
    fill!(parent(model.velocities.u), 0)
    fill!(parent(model.velocities.v), 0)
    fill!(parent(model.velocities.w), 0)
    update_state!(model)
    return nothing
end

function run_drift(label, damping; Δt = 20.0, n_steps = 30)
    model = build_model(damping)
    set_rest_state!(model)
    drift = Float64[]
    crashed = false
    for n in 1:n_steps
        try
            time_step!(model, Δt)
        catch e
            @warn "[$label] crashed at step $n: $(sprint(showerror, e))"
            crashed = true
            break
        end
        wmax = Float64(maximum(abs, interior(model.velocities.w)))
        push!(drift, wmax)
        @info @sprintf("[%s] step %3d t=%5.1fs max|w|=%.3e", label, n, n * Δt, wmax)
        if !isfinite(wmax)
            crashed = true
            break
        end
    end
    return (; label, drift, crashed)
end

function geom_mean_growth(drift)
    n = length(drift)
    n < 4 && return NaN
    # Use the last half to avoid the initial transient from machine noise.
    k0 = max(2, n ÷ 2)
    ratios = Float64[]
    for i in (k0 + 1):n
        if drift[i - 1] > 0 && isfinite(drift[i]) && isfinite(drift[i - 1])
            push!(ratios, drift[i] / drift[i - 1])
        end
    end
    isempty(ratios) && return NaN
    return exp(sum(log, ratios) / length(ratios))
end

@info "==== A: NoDivergenceDamping (control) ===="
A = run_drift("noDD", NoDivergenceDamping())

@info "==== B: SplitExplicit defaults (KlempDivergenceDamping coef=0.1, ω=0.6) ===="
B = run_drift("default", nothing)

gA = geom_mean_growth(A.drift)
gB = geom_mean_growth(B.drift)

@info @sprintf("A: NoDivergenceDamping     — geom-mean growth/step = %.4f, crashed = %s, final |w| = %.3e",
               gA, A.crashed, isempty(A.drift) ? NaN : A.drift[end])
@info @sprintf("B: KlempDivergenceDamping  — geom-mean growth/step = %.4f, crashed = %s, final |w| = %.3e",
               gB, B.crashed, isempty(B.drift) ? NaN : B.drift[end])

if isfinite(gB) && gB <= 1.05
    @info "PASS: 3D Klemp damping fixes the rest-atmosphere drift (growth/step ≤ 1.05)."
else
    @warn "FAIL: growth/step still > 1.05 with 3D Klemp damping"
end
