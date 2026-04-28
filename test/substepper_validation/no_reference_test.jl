#####
##### Is the reference state needed at all?
#####
##### The reference state's only role in the substepper is to provide
##### FP-cancellation in the slow vertical-momentum tendency:
#####     Gˢρw = Gⁿρw - ∂z(p⁰ - p_ref) - g·(ρ⁰ - ρ_ref)
##### Setting ref.pressure ≡ 0 and ref.density ≡ 0 reduces this to
#####     Gˢρw = Gⁿρw - ∂z p⁰ - g·ρ⁰
##### which is mathematically identical and at machine ε in Float64
##### for a near-hydrostatic state. (For Float32 the difference matters.)
#####
##### If "no reference" is at least as stable as the current trapezoidal
##### reference, the reference state is not pulling its weight; if "no
##### reference" is more stable, the current reference is actively harmful.
#####

using CUDA
using Oceananigans
using Oceananigans.Grids: znode, Center, Face
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Breeze
using BreezyBaroclinicInstability
using JLD2
using Printf

include("sweep_runner.jl")

const arch = CUDA.functional() ? GPU() : CPU()

function zero_reference!(model)
    ref = model.dynamics.reference_state
    FT = eltype(model.grid)
    Nz = model.grid.Nz
    zeros_z = zeros(FT, Nz)
    Oceananigans.set!(ref.pressure, zeros_z)
    Oceananigans.set!(ref.density,  zeros_z)
    if hasproperty(ref, :exner_function)
        Oceananigans.set!(ref.exner_function, zeros_z)
    end
    fill_halo_regions!(ref.pressure)
    fill_halo_regions!(ref.density)
    hasproperty(ref, :exner_function) && fill_halo_regions!(ref.exner_function)
    return nothing
end

function drift_test_zero_ref(; Δt, ω = 0.55, Nx = 16, Ny = 16, Nz = 64,
                                Lz = 30e3, T₀ = 250.0, sim_time = 600.0)
    model, _ = build_rest_model(; Lz, Nx, Ny, Nz, T₀,
                                  td_kwargs = (forward_weight = ω,))
    zero_reference!(model)
    Oceananigans.TimeSteppers.update_state!(model)

    n_steps = max(5, Int(round(sim_time / Δt)))
    sample = max(1, n_steps ÷ 30)
    iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = sample)
    return (Δt = Δt, ω = ω, ref = "zero",
            wmax_envelope = maximum(wmax),
            growth_per_step = growth_per_step(iters, wmax),
            status = status)
end

function drift_test_orig_ref(; Δt, ω = 0.55, Nx = 16, Ny = 16, Nz = 64,
                                Lz = 30e3, T₀ = 250.0, sim_time = 600.0)
    model, _ = build_rest_model(; Lz, Nx, Ny, Nz, T₀,
                                  td_kwargs = (forward_weight = ω,))

    n_steps = max(5, Int(round(sim_time / Δt)))
    sample = max(1, n_steps ÷ 30)
    iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = sample)
    return (Δt = Δt, ω = ω, ref = "orig",
            wmax_envelope = maximum(wmax),
            growth_per_step = growth_per_step(iters, wmax),
            status = status)
end

results = NamedTuple[]

@info "Zero-reference vs original-reference comparison, Δt × ω matrix"
for ω in (0.50, 0.55, 0.70), Δt in (1.0, 5.0, 20.0, 40.0)
    @info "ORIG ω=$ω Δt=$(Δt)s"
    push!(results, drift_test_orig_ref(; Δt, ω))
    @info "ZERO ω=$ω Δt=$(Δt)s"
    push!(results, drift_test_zero_ref(; Δt, ω))
end

println("\n=== Comparison: env after 600s ===")
println(rpad("Δt", 10), rpad("ω", 8), rpad("orig", 18), rpad("zero", 18))
for ω in (0.50, 0.55, 0.70), Δt in (1.0, 5.0, 20.0, 40.0)
    rO = filter(x -> x.Δt == Δt && x.ω == ω && x.ref == "orig", results)[1]
    rZ = filter(x -> x.Δt == Δt && x.ω == ω && x.ref == "zero", results)[1]
    eO = isnan(rO.wmax_envelope) ? "NaN" : @sprintf("%.2e", rO.wmax_envelope)
    eZ = isnan(rZ.wmax_envelope) ? "NaN" : @sprintf("%.2e", rZ.wmax_envelope)
    println(rpad("Δt=$(Δt)s", 10), rpad("ω=$ω", 8), rpad(eO, 18), rpad(eZ, 18))
end

println("\n=== Growth per outer step ===")
println(rpad("Δt", 10), rpad("ω", 8), rpad("orig", 12), rpad("zero", 12))
for ω in (0.50, 0.55, 0.70), Δt in (1.0, 5.0, 20.0, 40.0)
    rO = filter(x -> x.Δt == Δt && x.ω == ω && x.ref == "orig", results)[1]
    rZ = filter(x -> x.Δt == Δt && x.ω == ω && x.ref == "zero", results)[1]
    println(rpad("Δt=$(Δt)s", 10), rpad("ω=$ω", 8),
            rpad(@sprintf("%.4f", rO.growth_per_step), 12),
            rpad(@sprintf("%.4f", rZ.growth_per_step), 12))
end

jldsave(joinpath(@__DIR__, "no_ref_results.jld2"); results)
@info "Saved no_ref_results.jld2"
