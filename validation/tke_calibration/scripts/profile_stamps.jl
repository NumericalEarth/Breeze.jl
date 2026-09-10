# Postage stamps of every member: LES versus the column with the defaults and with the calibrated parameters.
#     julia --project scripts/profile_stamps.jl [evaluation=results/evaluation.jld2] [resolution=50] [figures=results/figures]
using BreezeCalibration, JLD2, Statistics, Printf, CairoMakie
# Scores from evaluations before the per-variable RMSE carried the names θ and q
sθ(s) = haskey(s, :θˡ) ? s.θˡ : s.θ
sq(s) = haskey(s, :qᵗ) ? s.qᵗ : s.q


options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
evaluation = get(options, "evaluation", joinpath(@__DIR__, "..", "results", "evaluation.jld2"))
resolution = get(options, "resolution", "50")
figures = get(options, "figures", joinpath(@__DIR__, "..", "results", "figures"))
mkpath(figures)

ev = load(evaluation)
res = ev["results"][resolution]
scores, means, zc = res.scores, res.means, res.zc
members, istrain, les = ev["members"], ev["istrain"], ev["les"]
sites = sort(unique(first.(members)))
months = ["01", "04", "07", "10"]
month_names = Dict("01" => "Jan", "04" => "Apr", "07" => "Jul", "10" => "Oct")
kz = zc .≤ 3000; kl = les.zc .≤ 3000
per_row = 2                                   # sites per row → 8 stamps per row
per_figure = 11                               # sites per figure → two figures per variable
groups = [sites[i:min(i + per_figure - 1, end)] for i in 1:per_figure:length(sites)]

function stamps(var, label, scale, group, filename)
    nrows = ceil(Int, length(group) / per_row)
    fig = Figure(size = (8 * 300, nrows * 330 + 40), fontsize = 12)
    Label(fig[0, 1:8], "$label below 3 km, sites $(group[1])–$(group[end]): LES (thick gray), Nakanishi–Niino defaults (blue), EKI ensemble mean (red); red titles are training members, shaded band the LES cloud layer", fontsize = 16)
    for (n, site) in enumerate(group), (mi, month) in enumerate(months)
        j = findfirst(==((site, month)), members)
        isnothing(j) && continue
        row = (n - 1) ÷ per_row + 1
        col = ((n - 1) % per_row) * 4 + mi
        s = scores[:, j]
        Δ = var == :θˡ ? @sprintf("%.2f → %.2f K", sθ(s[1]), sθ(s[2])) : @sprintf("%.2f → %.2f g/kg", sq(s[1]), sq(s[2]))
        ax = Axis(fig[row, col]; title = "site $site $(month_names[month])   $Δ", titlecolor = istrain[j] ? :firebrick : :black,
                  titlesize = 12, xticklabelsize = 10, yticklabelsize = 10, yticks = 0:1000:3000,
                  yticklabelsvisible = mi == 1, ylabelvisible = false, xgridvisible = false, ygridvisible = false)
        y_les = scale .* getproperty(les, var)[kl, j]
        y_def = scale .* getproperty(means, var)[1, j, kz]
        y_cal = scale .* getproperty(means, var)[2, j, kz]
        cloudy = les.qˡ[kl, j] .> 1e-6
        any(cloudy) && hspan!(ax, minimum(les.zc[kl][cloudy]), maximum(les.zc[kl][cloudy]); color = (:gray, 0.12))
        lines!(ax, y_les, les.zc[kl]; color = (:black, 0.3), linewidth = 6)
        lines!(ax, y_def, zc[kz]; color = :steelblue, linewidth = 1.8)
        lines!(ax, y_cal, zc[kz]; color = :firebrick, linewidth = 1.8)
        lo, hi = extrema(vcat(y_les, y_def, y_cal))
        xlims!(ax, lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ylims!(ax, 0, 3000)
    end
    colgap!(fig.layout, 8); rowgap!(fig.layout, 8)
    save(joinpath(figures, filename), fig)
end

for (g, group) in enumerate(groups)
    stamps(:θˡ, "θˡ (K)", 1, group, "stamps_theta_$g.png")
    stamps(:qᵗ, "qᵗ (g kg⁻¹)", 1e3, group, "stamps_q_$g.png")
end
println("wrote stamps_theta_{1,2}.png and stamps_q_{1,2}.png to $figures")
