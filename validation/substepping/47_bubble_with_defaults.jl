#####
##### validation/substepping/47_bubble_with_defaults.jl
#####
##### Dry thermal bubble at the new SplitExplicit defaults
##### (forward_weight=0.6, KlempDivergenceDamping coef=0.1).
##### Should evolve a bubble like the canonical case — verifying that the
##### damping doesn't kill physical motion.
#####

using Breeze
using Oceananigans
using Statistics
using CUDA
using Printf

const arch = CUDA.functional() ? GPU() : CPU()

const Δt     = 2.0
const STOP_T = 60.0   # 1 simulated minute — quick smoke test

function build_grid()
    # Match canonical 01_dry_thermal_bubble.jl on CPU.
    RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

θ₀_ref = 300.0
N²     = 1e-6
r₀     = 2e3
Δθ     = 10.0
const g = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    function θᵢ(x, z)
        r = sqrt((x - x₀)^2 + (z - z₀)^2)
        θ′ = Δθ * max(0, 1 - r / r₀)
        return θᵇᵍ(z) + θ′
    end
end

function build_run(td_opts; label)
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; td_opts...)
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                            thermodynamic_constants = constants,
                            timestepper = :AcousticRungeKutta3)

    θᵢ = θᵢ_builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)

    @info "[$label] forward_weight=$(td.forward_weight), damping=$(typeof(td.damping).name.name)"

    n_steps = Int(round(STOP_T / Δt))
    final_w = NaN
    final_u = NaN
    crashed = false
    for n in 1:n_steps
        try
            time_step!(model, Δt)
        catch e
            @warn "[$label] crashed at step $n: $(sprint(showerror, e))"
            crashed = true
            break
        end
        if n % 30 == 0 || n == 1
            wmax = Float64(maximum(abs, interior(model.velocities.w)))
            umax = Float64(maximum(abs, interior(model.velocities.u)))
            @info @sprintf("[%s]  step %4d  t=%6.1fs  max|w|=%.3e  max|u|=%.3e",
                           label, n, n * Δt, wmax, umax)
            if !isfinite(wmax)
                crashed = true
                break
            end
        end
    end
    final_w = Float64(maximum(abs, interior(model.velocities.w)))
    final_u = Float64(maximum(abs, interior(model.velocities.u)))
    @info @sprintf("[%s] final t=%6.1fs max|w|=%.3e max|u|=%.3e crashed=%s",
                   label, model.clock.time, final_w, final_u, crashed)
    return (; label, final_w, final_u, crashed)
end

results = []
push!(results, build_run((;); label = "default"))
push!(results, build_run((forward_weight = 0.55, damping = ThermodynamicDivergenceDamping(coefficient = 0.1));
                          label = "old-default"))

println()
@info "=== SUMMARY ==="
for r in results
    @printf("%-13s  final |w|=%.3e  final |u|=%.3e  crashed=%s\n",
            r.label, r.final_w, r.final_u, r.crashed)
end
