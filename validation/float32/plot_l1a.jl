# Compare decaying turbulence (L1a) in Float32 and Float64: ε, Re_λ, gradient moments, spectra and cost.
#
#   julia --project=validation/float32 validation/float32/plot_l1a.jl OUTPUT_DIR FIGURE_PREFIX [NCAR_DIR]

using CairoMakie
using DelimitedFiles
using Printf
using Statistics

function read_csv(path)
    data, header = readdlm(path, ',', header=true)
    return NamedTuple{Tuple(Symbol.(vec(header)))}(Tuple(Float64.(data[:, i]) for i in 1:size(data, 2)))
end

output_dir = ARGS[1]
prefix = ARGS[2]
ncar_dir = length(ARGS) ≥ 3 ? ARGS[3] : "/home/greg_aeolus_earth/Breeze-dns/validation/ncar_dns/output/resolution"

resolutions = [M for M in (64, 128) if isfile(joinpath(output_dir, "breeze_R_M$(M)_centered4_Float64_timeseries.csv"))]
precisions = ("Float64", "Float32")
series(M, FT) = read_csv(joinpath(output_dir, "breeze_R_M$(M)_centered4_$(FT)_timeseries.csv"))
spectra(M, FT) = read_csv(joinpath(output_dir, "breeze_R_M$(M)_centered4_$(FT)_spectra.csv"))
runs = Dict((M, FT) => series(M, FT) for M in resolutions for FT in precisions)
ncar = read_csv(joinpath(ncar_dir, "ncar_decay_n128_timeseries.csv"))
ncar_time = ncar.time .- ncar.time[1]

colors = Dict(64 => :royalblue, 128 => :firebrick)
styles = Dict("Float64" => :solid, "Float32" => :dash)
quantities = [(:dissipation, "ε (m² s⁻³)"), (:taylor_reynolds, "Re_λ"), (:skewness, "S (∂u/∂x)"),
              (:flatness, "F (∂u/∂x)"), (:transverse_flatness, "F_T"), (:kinetic_energy, "K (m² s⁻²)")]

fig = Figure(size=(1200, 1000))
for (n, (q, ylabel)) in enumerate(quantities)
    ax = Axis(fig[(n - 1) ÷ 2 + 1, (n - 1) % 2 + 1]; xlabel="t (s)", ylabel)
    hasproperty(ncar, q) && scatter!(ax, ncar_time, getfield(ncar, q); color=:gray, markersize=6, label="NCAR 128³")
    for M in resolutions, FT in precisions
        r = runs[(M, FT)]
        valid = .!isnan.(getfield(r, q))
        lines!(ax, r.time[valid], getfield(r, q)[valid]; color=colors[M], linestyle=styles[FT], label="Breeze $(M)³ $FT")
    end
    n == 1 && axislegend(ax; position=:rt, framevisible=false)
end
save("$(prefix)_timeseries.png", fig)

# Spectra at the final time and the Float32 / Float64 ratio
fig = Figure(size=(1200, 450))
ax1 = Axis(fig[1, 1]; xlabel="k (shell)", ylabel="E(k)", xscale=log10, yscale=log10, title="final time")
ax2 = Axis(fig[1, 2]; xlabel="k (shell)", ylabel="E₃₂(k) / E₆₄(k) - 1", xscale=log10)
for M in resolutions
    s = Dict(FT => spectra(M, FT) for FT in precisions)
    t = maximum(s["Float64"].time)
    for FT in precisions
        sel = s[FT].time .== maximum(s[FT].time)
        lines!(ax1, s[FT].k[sel], s[FT].energy[sel]; color=colors[M], linestyle=styles[FT], label="$(M)³ $FT")
    end
    a = s["Float64"].time .== t
    b = s["Float32"].time .== maximum(s["Float32"].time)
    lines!(ax2, s["Float64"].k[a], s["Float32"].energy[b] ./ s["Float64"].energy[a] .- 1; color=colors[M], label="$(M)³")
end
axislegend(ax1; position=:lb, framevisible=false)
axislegend(ax2; position=:lt, framevisible=false)
save("$(prefix)_spectra.png", fig)

# Summary: maximum relative difference over the run, and values at the end
open("$(prefix)_summary.md", "w") do io
    println(io, "| M | quantity | Float64 end | Float32 end | max rel. difference over run |")
    println(io, "|---|---|---|---|---|")
    for M in resolutions, (q, name) in quantities
        a, b = runs[(M, "Float64")], runs[(M, "Float32")]
        x, y = getfield(a, q), getfield(b, q)
        N = min(length(x), length(y))
        valid = .!isnan.(x[1:N]) .& .!isnan.(y[1:N])
        δ = maximum(abs.(y[1:N][valid] .- x[1:N][valid]) ./ abs.(x[1:N][valid]))
        println(io, @sprintf("| %d | %s | %.5g | %.5g | %.2e |", M, name, x[N], y[N], δ))
    end
    timing = joinpath(output_dir, "timing.csv")
    if isfile(timing)
        println(io, "\n| run | ms/step (no diagnostics) | device memory (MiB) |")
        println(io, "|---|---|---|")
        for line in readlines(timing)
            f = split(line, ',')
            println(io, @sprintf("| %s | %.3f | %.1f |", f[1], parse(Float64, f[8]), parse(Float64, f[11]) / 2^20))
        end
    end
end
print(read("$(prefix)_summary.md", String))
