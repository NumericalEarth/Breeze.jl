# Skill versus resolution for several calibrations: held-out median RMSE per field from evaluation files.
#     julia --project scripts/compare_calibrations.jl "label=results/evaluation.jld2;results/evaluation_coarse.jld2" "label2=..." [figures=results/figures]
# Each evaluation file holds, per resolution, the scores of the defaults (row 1) and the calibrated ensemble mean (row 2).
using JLD2, Statistics, Printf, CairoMakie
sθ(s) = haskey(s, :θˡ) ? s.θˡ : s.θ
sq(s) = haskey(s, :qᵗ) ? s.qᵗ : s.q
sl(s) = haskey(s, :qˡ) ? s.qˡ : NaN
sw(s) = haskey(s, :wind) ? s.wind : NaN

options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
figures = pop!(options, "figures", joinpath(@__DIR__, "..", "results", "figures")); mkpath(figures)
order = ["20", "50", "100", "hindcast"]
fields = (("θˡ (K)", sθ), ("qᵗ (g kg⁻¹)", sq), ("qˡ (g kg⁻¹)", sl), ("wind (m s⁻¹)", sw))

# Collect: label → Dict(resolution → (defaults, calibrated) score rows), plus the held-out mask
sets = Dict{String, Any}()
istrain = nothing
for (label, paths) in options            # several files for one label (the same parameters evaluated on different grids) joined by ';'
    d = Dict{String, Any}()
    for path in split(paths, ';')
        ev = load(path)
        global istrain = ev["istrain"]
        merge!(d, Dict(r => (ev["results"][r].scores[1, :], ev["results"][r].scores[2, :]) for r in keys(ev["results"])))
    end
    sets[label] = d
end
heldout = .!istrain

fig = Figure(size = (1500, 420))
colors = Makie.wong_colors()
for (col, (name, f)) in enumerate(fields)
    ax = Axis(fig[1, col]; title = name, xticks = (1:4, ["20 m", "50 m", "100 m", "hindcast"]), ylabel = col == 1 ? "held-out median RMSE" : "")
    # defaults, from whichever file has each resolution
    xs = Int[]; ys = Float64[]
    for (i, r) in enumerate(order)
        having = [d for d in values(sets) if haskey(d, r)]
        isempty(having) && continue
        push!(xs, i); push!(ys, median(f.(having[1][r][1][heldout])))
    end
    scatterlines!(ax, xs, ys; color = :black, linestyle = :dash, label = "Nakanishi–Niino defaults")
    for (k, label) in enumerate(sort(collect(keys(sets))))
        d = sets[label]
        xs = [i for (i, r) in enumerate(order) if haskey(d, r)]
        ys = [median(f.(d[r][2][heldout])) for r in order if haskey(d, r)]
        all(isnan, ys) && continue
        scatterlines!(ax, xs, ys; color = colors[mod1(k, 7)], label)
    end
    ylims!(ax, 0, nothing)
    col == 1 && axislegend(ax; position = :lt, labelsize = 10)
end
Label(fig[0, :], "Skill of calibrated parameter sets across model resolutions (67 held-out members)", fontsize = 16)
save(joinpath(figures, "skill_vs_resolution.png"), fig)

println("held-out median RMSE θˡ (K) / qᵗ (g/kg) / qˡ (g/kg) / wind (m/s):")
for r in order
    having = [label for label in sort(collect(keys(sets))) if haskey(sets[label], r)]
    isempty(having) && continue
    println("  resolution $r")
    rows = vcat([("defaults", sets[having[1]][r][1])], [(label, sets[label][r][2]) for label in having])
    for (label, scores) in rows
        s = scores[heldout]
        @printf "    %-42s %.2f / %.2f / %.3f / %.2f\n" label median(sθ.(s)) median(sq.(s)) median(sl.(s)) median(sw.(s))
    end
end
println("figure: $(joinpath(figures, "skill_vs_resolution.png"))")
