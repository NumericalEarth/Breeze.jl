# Cost per time step and device memory of the L1a and L3 runs in each precision.
#
#   julia --project=validation/float32 validation/float32/plot_cost.jl L1A_TIMING L3_TIMING FIGURE

using CairoMakie
using DelimitedFiles
using Printf
using Statistics

l1a = readdlm(ARGS[1], ',', String)
l3 = readdlm(ARGS[2], ',', String)

# L1a rows: name, mode, M, advection, iterations, time, wall, ms per step, GPU time, FT, bytes
l1a_rows = [(label = @sprintf("L1a %s³ %s", r[3], r[10]), ms = parse(Float64, r[8]), MiB = parse(Float64, r[11]) / 2^20)
            for r in eachrow(l1a)]

# L3 rows: label (s<seed>_M<M>_<variant>), FT, seconds per step, bytes; averaged over seeds
names = Dict("f64" => "Float64", "f32naive" => "Float32 as merged", "f32" => "Float32 + fixes",
             "f32mixed" => "Float32, Float64 θ qᵛ", "f32fluct" => "Float32, fluct. (emulated)",
             "f32rn" => "Float32 RN (emulated)", "f32sr" => "Float32 SR (emulated)")
groups = Dict{Tuple{Int, String}, Vector{Tuple{Float64, Float64}}}()
for r in eachrow(l3)
    m = match(r"s\d+_M(\d+)_(\w+)", r[1])
    key = (parse(Int, m[1]), m[2])
    push!(get!(groups, key, Tuple{Float64, Float64}[]), (1e3 * parse(Float64, r[3]), parse(Float64, r[4]) / 2^20))
end
order = ["f64", "f32naive", "f32", "f32mixed", "f32fluct", "f32rn", "f32sr"]
l3_rows = [(label = @sprintf("L3 %d³ %s", M, names[v]), ms = mean(first.(groups[(M, v)])), MiB = mean(last.(groups[(M, v)])))
           for M in sort(unique(first.(keys(groups)))) for v in order if haskey(groups, (M, v))]

rows = vcat(l1a_rows, l3_rows)
fig = Figure(size=(1100, 120 + 28 * length(rows)))
labels = [r.label for r in rows]
colors = [occursin("Float64", r.label) && !occursin("θ", r.label) ? :black : :royalblue for r in rows]
ax1 = Axis(fig[1, 1]; xlabel="ms per step (A100, no diagnostics)", yticks=(1:length(rows), labels), yreversed=true)
barplot!(ax1, 1:length(rows), [r.ms for r in rows]; direction=:x, color=colors)
ax2 = Axis(fig[1, 2]; xlabel="device memory in use (MiB)", yticks=(1:length(rows), fill("", length(rows))), yreversed=true)
barplot!(ax2, 1:length(rows), [r.MiB for r in rows]; direction=:x, color=colors)
save(ARGS[3], fig)

for r in rows
    println(@sprintf("| %s | %.2f | %.1f |", r.label, r.ms, r.MiB))
end
