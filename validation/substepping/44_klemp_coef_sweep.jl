#####
##### validation/substepping/44_klemp_coef_sweep.jl
#####
##### Sweep over Klemp damping coefficient at production (Δt=20 s, ω=0.55).
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

function build_model(damping; forward_weight = 0.55)
    grid = RectilinearGrid(arch;
                           size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (0, Lh), y = (0, Lh), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization(; substeps = nothing,
                                            forward_weight = forward_weight,
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

function run_drift(label, damping; Δt = 20.0, n_steps = 30, forward_weight = 0.55)
    model = build_model(damping; forward_weight = forward_weight)
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

cases = [
    ("noDD",       NoDivergenceDamping()),
    ("klemp_0.05", KlempDivergenceDamping(coefficient = 0.05)),
    ("klemp_0.10", KlempDivergenceDamping(coefficient = 0.10)),
    ("klemp_0.20", KlempDivergenceDamping(coefficient = 0.20)),
    ("klemp_0.30", KlempDivergenceDamping(coefficient = 0.30)),
    ("klemp_0.40", KlempDivergenceDamping(coefficient = 0.40)),
]

results = []
for (label, damping) in cases
    @info "Running $label..."
    r = run_drift(label, damping; n_steps = 30)
    g = geom_mean_growth(r.drift)
    final_w = isempty(r.drift) ? NaN : r.drift[end]
    push!(results, (; label, growth = g, final_w, crashed = r.crashed))
    @info @sprintf("  %s: growth/step = %.4f, final |w| = %.3e, crashed = %s",
                   label, g, final_w, r.crashed)
end

println()
println("=== SUMMARY (Δt=20s, ω=0.55, Nx=Ny=16, Nz=64) ===")
for r in results
    mark = (isfinite(r.growth) && r.growth <= 1.05) ? "✓" : "✗"
    @printf("  %s %-12s growth/step = %.4f  final|w| = %.3e  crashed = %s\n",
            mark, r.label, r.growth, r.final_w, r.crashed)
end
