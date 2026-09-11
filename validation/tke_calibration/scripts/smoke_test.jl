# The pipeline end to end on a small problem: two members × two parameter sets on the LES grid extended to 25 km
# with the GCM columns above, strong relaxation aloft, and interactive all-sky radiation (and, for comparison, the
# LES's prescribed heating). Three hours; prints the reference pressure at the top, the radiative heating in the
# cloud layer, and the cost.
#     julia -t auto --project scripts/smoke_test.jl [arch=cpu|gpu]
using BreezeCalibration, Statistics, Printf
using Oceananigans.Units
using Oceananigans: interior, CPU, GPU
options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
get(options, "arch", "cpu") == "gpu" && @eval using CUDA
architecture = get(options, "arch", "cpu") == "gpu" ? GPU() : CPU()
members = [load_member(22, "07"), load_member(17, "07")]
all(m -> !isnothing(m.gcm), members) || error("GCM columns missing: run scripts/fetch_gcm_columns.jl and solar_parameters.jl")
problem = ColumnEnsembleProblem(members)
@printf "grid: %d cells, LES top %.0f m, top %.0f m, spacing at top %.0f m; GCM column for site 22 Jul: θ(20 km) = %.1f K, coszen %.3f, insolation %.0f W/m²\n" length(problem.zf) - 1 problem.les_top problem.zf[end] problem.zf[end] - problem.zf[end-1] members[1].gcm.θ[findlast(<(20_000), members[1].gcm.z)] members[1].gcm.coszen members[1].gcm.insolation
params = hcat(collect(Float64, default_parameters()), collect(Float64, default_parameters()) .* 1.1)
for radiation in (:prescribed, :interactive)
    t = @elapsed means, model = run_ensemble(problem, params; stop_time = 3hours, averaging_window = (0.0, 3hours), radiation, architecture)
    pᵣ = model.dynamics.reference_state.pressure
    zc = problem.zc; kles = zc .≤ 4000
    @printf "%-12s %.0f s (incl. compilation); reference pressure at top %.0f Pa (GCM %.0f); θˡ range below 4 km %.1f–%.1f K; qᵗ at 20 km %.3f g/kg\n" radiation t Array(interior(pᵣ))[1, 1, end] members[1].gcm.p[end] extrema(means.θˡ[1, 1, kles])... 1e3 * means.qᵗ[1, 1, findlast(<(20_000), zc)]
    if radiation == :interactive
        H = model.radiation.flux_divergence            # W m⁻³, positive = heating
        ρ = Array(interior(model.dynamics.reference_state.density)); h = Array(interior(H))
        cᵖ = 1004.7
        rate = 86400 .* h[1, 1, :] ./ (ρ[1, 1, :] .* cᵖ)
        @printf "  radiative heating (K/day) site 22 Jul: min %.2f at %.0f m, at 100 m %.2f, at 10 km %.2f; LES record-mean at cloud top %.2f K/day\n" minimum(rate) zc[argmin(rate)] rate[searchsortedlast(zc, 100)] rate[searchsortedlast(zc, 10_000)] minimum(86400 .* vec(mean(members[1].heating, dims = 2)) ./ cᵖ)
    end
end
