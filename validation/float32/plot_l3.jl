# Compare L3 runs in different precisions: time series, a summary table and per-step increments vs storage spacing.
#
#   julia --project=validation/float32 validation/float32/plot_l3.jl OUTPUT_DIR FIGURE_PREFIX M "VARIANTS" [SEEDS]

using CairoMakie
using DelimitedFiles
using Printf
using Statistics

function read_csv(path)
    data, header = readdlm(path, ',', header=true)
    return NamedTuple{Tuple(Symbol.(vec(header)))}(Tuple(Float64.(data[:, i]) for i in 1:size(data, 2)))
end

const labels = Dict("f64" => "Float64", "f32naive" => "Float32 (as merged)", "f32" => "Float32 + fixes",
                    "f32mixed" => "Float32, Float64 θ and qᵛ", "f32fluct" => "Float32, fluctuation storage")
const colors = Dict("f64" => :black, "f32naive" => :firebrick, "f32" => :darkorange, "f32mixed" => :royalblue,
                    "f32fluct" => :seagreen)

output_dir = ARGS[1]
prefix = ARGS[2]
M = parse(Int, ARGS[3])
variants = split(ARGS[4])
seeds = length(ARGS) ≥ 5 ? parse.(Int, split(ARGS[5], ',')) : [1]

path(v, s) = joinpath(output_dir, "breeze_l3_s$(s)_M$(M)_$(v)_timeseries.csv")
runs = Dict((v, s) => read_csv(path(v, s)) for v in variants for s in seeds if isfile(path(v, s)))

quantities = [(:liquid_water_content, "LWC (g kg⁻¹)", 1e3), (:mean_radius, "r̄ (μm)", 1e6),
              (:std_radius, "σᵣ (μm)", 1e6), (:parcel_supersaturation, "parcel 𝒮 (%)", 100),
              (:droplet_mean_supersaturation, "droplet-seen 𝒮 (%)", 100), (:std_supersaturation, "std 𝒮 (%)", 100),
              (:std_temperature, "std T′ (mK)", 1e3), (:std_vapor, "std qᵛ′ (10⁻⁶)", 1e6)]

fig = Figure(size=(1200, 1100))
for (n, (q, ylabel, scale)) in enumerate(quantities)
    ax = Axis(fig[(n - 1) ÷ 2 + 1, (n - 1) % 2 + 1]; xlabel="t (s)", ylabel)
    for v in variants, s in seeds
        haskey(runs, (v, s)) || continue
        r = runs[(v, s)]
        lines!(ax, r.time, scale .* getfield(r, q); color=colors[v], linestyle=s == 1 ? :solid : :dash,
               label=s == first(seeds) ? labels[v] : nothing)
    end
    n == 1 && axislegend(ax; position=:lt, framevisible=false)
end
Label(fig[0, :], "L3 condensation, $(M)³, seeds $(join(seeds, ", ")) (solid: seed 1)"; fontsize=18)
save("$(prefix)_timeseries.png", fig)

# Differences from Float64, per seed
fig = Figure(size=(1200, 700))
for (n, (q, ylabel, scale)) in enumerate(quantities[[1, 3, 4, 7]])
    ax = Axis(fig[(n - 1) ÷ 2 + 1, (n - 1) % 2 + 1]; xlabel="t (s)", ylabel="Δ " * ylabel * " vs Float64")
    for v in variants, s in seeds
        (v == "f64" || !haskey(runs, (v, s)) || !haskey(runs, ("f64", s))) && continue
        r, r₀ = runs[(v, s)], runs[("f64", s)]
        N = min(length(r.time), length(r₀.time))
        lines!(ax, r.time[1:N], scale .* (getfield(r, q)[1:N] .- getfield(r₀, q)[1:N]); color=colors[v],
               linestyle=s == 1 ? :solid : :dash, label=s == first(seeds) ? labels[v] : nothing)
    end
    # Seed-to-seed spread of Float64, for scale
    f64 = [runs[("f64", s)] for s in seeds if haskey(runs, ("f64", s))]
    if length(f64) > 1
        N = minimum(length(r.time) for r in f64)
        x = reduce(hcat, [scale .* getfield(r, q)[1:N] for r in f64])
        σ = vec(std(x; dims=2))
        band!(ax, f64[1].time[1:N], -σ, σ; color=(:gray, 0.3), label="± std over Float64 seeds")
    end
    n == 1 && axislegend(ax; position=:lb, framevisible=false)
end
save("$(prefix)_differences.png", fig)

# Per-step condensation increments relative to the storage spacing
fig = Figure(size=(1200, 450))
for (n, (inc, spacing, title)) in enumerate(((:theta_increment, :theta_spacing, "θ"), (:vapor_increment, :vapor_spacing, "qᵛ")))
    ax = Axis(fig[1, n]; xlabel="t (s)", ylabel="mean per-step increment / spacing of stored value",
              title="condensation increment of $title", yscale=log10)
    for v in variants
        haskey(runs, (v, first(seeds))) || continue
        r = runs[(v, first(seeds))]
        hasproperty(r, inc) || continue
        valid = getfield(r, inc) .> 0
        lines!(ax, r.time[valid], getfield(r, inc)[valid] ./ getfield(r, spacing)[valid]; color=colors[v], label=labels[v])
    end
    hlines!(ax, [0.5]; color=:gray, linestyle=:dot)
    n == 1 && axislegend(ax; position=:rt, framevisible=false)
end
save("$(prefix)_increments.png", fig)

# Summary table: final values and plateau averages (t ≥ 2.1 s), mean ± std over seeds
plateau(r, x) = mean(x[r.time .≥ min(2.1, r.time[end])])
statistics = [(:liquid_water_content, "LWC (g/kg)", 1e3, :final), (:mean_radius, "r̄ (μm)", 1e6, :final),
              (:std_radius, "σᵣ (μm)", 1e6, :final), (:parcel_supersaturation, "parcel 𝒮 (%)", 100, :final),
              (:droplet_mean_supersaturation, "droplet 𝒮 (%)", 100, :plateau),
              (:std_supersaturation, "std 𝒮 (%)", 100, :plateau), (:std_temperature, "std T′ (mK)", 1e3, :plateau)]
open("$(prefix)_summary.md", "w") do io
    println(io, "| variant | n | ", join([s[2] * (s[4] == :final ? " end" : " plateau") for s in statistics], " | "), " |")
    println(io, "|", repeat("---|", length(statistics) + 2))
    for v in variants
        rs = [runs[(v, s)] for s in seeds if haskey(runs, (v, s))]
        isempty(rs) && continue
        cells = map(statistics) do (q, _, scale, kind)
            x = [scale * (kind == :final ? getfield(r, q)[end] : plateau(r, getfield(r, q))) for r in rs]
            length(x) > 1 ? @sprintf("%.5g ± %.2g", mean(x), std(x)) : @sprintf("%.5g", x[1])
        end
        println(io, "| ", labels[v], " | ", length(rs), " | ", join(cells, " | "), " |")
    end
end
print(read("$(prefix)_summary.md", String))
