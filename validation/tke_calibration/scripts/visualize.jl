# Figures from a calibration: parameter drift, misfit trajectory, parameter importance, per-member RMSE and
# LES-versus-column profiles.
#
#     julia --project scripts/visualize.jl [checkpoint=results/eki.jld2] [evaluation=results/evaluation.jld2] [resolution=50] [figures=results/figures]
using BreezeCalibration, JLD2, Statistics, LinearAlgebra, Printf, CairoMakie
# Scores from evaluations before the per-variable RMSE carried the names θ and q
sθ(s) = haskey(s, :θˡ) ? s.θˡ : s.θ
sq(s) = haskey(s, :qᵗ) ? s.qᵗ : s.q


options = Dict(split(a, '=', limit = 2) for a in filter(a -> occursin('=', a), ARGS))
checkpoint = get(options, "checkpoint", joinpath(@__DIR__, "..", "results", "eki.jld2"))
evaluation = get(options, "evaluation", joinpath(@__DIR__, "..", "results", "evaluation.jld2"))
resolution = get(options, "resolution", "50")
figures = get(options, "figures", joinpath(@__DIR__, "..", "results", "figures"))
mkpath(figures)

history = load(checkpoint, "history"); y = load(checkpoint, "y"); σ = sqrt.(load(checkpoint, "Γ"))
space = space_of(size(history[1].ϕ, 1))
names = collect(String.(parameter_names(space)))
N = length(names)
ϕ₁, ϕₑ = history[1].ϕ, history[end].ϕ
defaults = collect(Float64, default_parameters(space))
catke = space isa RiDependentSpace ? collect(Float64, catke_calibration_parameters()) : nothing

##### Parameters: the prior and final ensembles, the defaults and CATKE's values
fig = Figure(size = (1400, 520))
ax = Axis(fig[1, 1]; yscale = log10, xticks = (1:N, names), xticklabelrotation = π / 4, ylabel = "parameter value",
          title = "Parameters: prior ensemble (light), final ensemble (dark), Nakanishi–Niino defaults (○), CATKE (△)")
for k in 1:N
    boxplot!(ax, fill(k - 0.18, size(ϕ₁, 2)), ϕ₁[k, :]; width = 0.3, color = (:steelblue, 0.35), show_outliers = false)
    boxplot!(ax, fill(k + 0.18, size(ϕₑ, 2)), ϕₑ[k, :]; width = 0.3, color = (:firebrick, 0.7), show_outliers = false)
end
positive = defaults .> 0
scatter!(ax, (1:N)[positive], defaults[positive]; marker = :circle, color = :white, strokecolor = :black, strokewidth = 2, markersize = 13)
isnothing(catke) || scatter!(ax, 1:N, catke; marker = :utriangle, color = :goldenrod, strokecolor = :black, strokewidth = 1, markersize = 13)
save(joinpath(figures, "parameters.png"), fig)

##### Misfit trajectory
fig = Figure(size = (800, 450))
ax = Axis(fig[1, 1]; xlabel = "EKI iteration", ylabel = "normalized misfit (RMS of (G − y)/σ)", title = "Misfit on the 16 training members")
its = [h.iteration for h in history]
lines!(ax, its, [mean(h.misfit) for h in history]; color = :firebrick, linewidth = 2, label = "ensemble mean")
lines!(ax, its, [median(h.misfit) for h in history]; color = :firebrick, linestyle = :dash, label = "ensemble median")
lines!(ax, its, [minimum(h.misfit) for h in history]; color = :black, label = "best member")
haskey(history[end], :pseudotime) && text!(ax, its, [mean(h.misfit) for h in history]; text = [@sprintf("T=%.2f", h.pseudotime) for h in history], offset = (6, 6), fontsize = 11)
axislegend(ax; position = :rt)
save(joinpath(figures, "misfit.png"), fig)

##### Importance: collapse of the ensemble, sensitivity of the misfit, correlation with the misfit
logθ₁, logθₑ = log.(ϕ₁), log.(ϕₑ)
collapse = vec(std(logθₑ, dims = 2) ./ std(logθ₁, dims = 2))
# Linear response of the normalized residual to standardized log-parameters over the final ensemble
R = (history[end].G .- y) ./ σ                                          # (N_obs, N_ens)
X = (logθₑ .- mean(logθₑ, dims = 2)) ./ std(logθₑ, dims = 2)            # (N_par, N_ens)
β = (R .- mean(R, dims = 2)) * X' * inv(X * X' + 1e-6I)                 # (N_obs, N_par)
sensitivity = vec(sqrt.(mean(β .^ 2, dims = 1)))                        # RMS misfit change per prior-std of log θ
correlation = [cor(logθₑ[k, :], history[end].misfit) for k in 1:N]
order = sortperm(sensitivity; rev = true)
fig = Figure(size = (1400, 450))
ax1 = Axis(fig[1, 1]; xticks = (1:N, names[order]), xticklabelrotation = π / 4, ylabel = "RMS Δ(normalized misfit) per ensemble std", title = "Sensitivity of the misfit (final ensemble)")
barplot!(ax1, 1:N, sensitivity[order]; color = :firebrick)
ax2 = Axis(fig[1, 2]; xticks = (1:N, names[order]), xticklabelrotation = π / 4, ylabel = "final std / prior std (log parameter)", title = "Collapse of the ensemble")
barplot!(ax2, 1:N, collapse[order]; color = :steelblue)
hlines!(ax2, [1]; color = :black, linestyle = :dot)
ax3 = Axis(fig[1, 3]; xticks = (1:N, names[order]), xticklabelrotation = π / 4, ylabel = "correlation of log parameter with misfit", title = "Direction: + means larger is worse")
barplot!(ax3, 1:N, correlation[order]; color = [c > 0 ? :darkorange : :seagreen for c in correlation[order]])
hlines!(ax3, [0]; color = :black)
save(joinpath(figures, "importance.png"), fig)

println("parameter table (final ensemble):")
@printf "%-6s %9s %9s %9s %9s %8s %11s %9s\n" "name" "default" "CATKE" "EKI mean" "EKI std" "ratio" "sensitivity" "collapse"
for k in order
    @printf "%-6s %9.3f %9s %9.3f %9.3f %8.2f %11.3f %9.2f\n" names[k] defaults[k] (isnothing(catke) ? "" : @sprintf("%.3f", catke[k])) mean(ϕₑ[k, :]) std(ϕₑ[k, :]) (defaults[k] > 0 ? mean(ϕₑ[k, :]) / defaults[k] : NaN) sensitivity[k] collapse[k]
end

##### Evaluation: per-member RMSE and profiles
if isfile(evaluation)
    ev = load(evaluation)
    res = ev["results"][resolution]
    scores, means, zc = res.scores, res.means, res.zc
    labels, members, istrain, les = ev["labels"], ev["members"], ev["istrain"], ev["les"]

    fig = Figure(size = (1000, 480))
    for (col, (var, unit)) in enumerate((("θ", "K"), ("q", "g kg⁻¹")))
        local ax = Axis(fig[1, col]; xlabel = "RMSE, defaults ($unit)", ylabel = "RMSE, EKI ensemble mean ($unit)", title = var == "θ" ? "θˡ RMSE per member" : "qᵗ RMSE per member", aspect = 1)
        d = [(var == "θ" ? sθ(s) : sq(s)) for s in scores[1, :]]; c = [(var == "θ" ? sθ(s) : sq(s)) for s in scores[2, :]]
        lim = 1.05 * max(maximum(d), maximum(c))
        lines!(ax, [0, lim], [0, lim]; color = :black, linestyle = :dash)
        scatter!(ax, d[istrain], c[istrain]; color = :firebrick, markersize = 10, label = "training")
        scatter!(ax, d[.!istrain], c[.!istrain]; color = :steelblue, markersize = 10, label = "held out")
        xlims!(ax, 0, lim); ylims!(ax, 0, lim)
        col == 1 && axislegend(ax; position = :lt)
    end
    save(joinpath(figures, "rmse_scatter.png"), fig)

    # Profiles: the members with the largest and smallest gains, plus fixed representatives
    gain = [sθ(scores[1, j]) - sθ(scores[2, j]) for j in eachindex(members)]
    wanted = [(17, "07"), (22, "07"), (2, "07"), (10, "01"), (5, "01"), (14, "04")]
    picks = unique(vcat([findfirst(==(w), members) for w in wanted if w in members], argmax(gain), argmin(gain)))[1:min(8, end)]
    fig = Figure(size = (280 * length(picks), 760))
    kz = zc .≤ 3000; kl = les.zc .≤ 3000
    for (col, j) in enumerate(picks)
        site, month = members[j]
        ttl = @sprintf("site %d, month %s (%s)\nΔθ RMSE %.2f → %.2f K", site, month, istrain[j] ? "training" : "held out", sθ(scores[1, j]), sθ(scores[2, j]))
        axθ = Axis(fig[1, col]; title = ttl, xlabel = "θˡ (K)", ylabel = col == 1 ? "z (m)" : "")
        axq = Axis(fig[2, col]; xlabel = "qᵗ (g kg⁻¹)", ylabel = col == 1 ? "z (m)" : "")
        lines!(axθ, les.θˡ[kl, j], les.zc[kl]; color = :black, linewidth = 3, label = "LES")
        lines!(axq, 1e3 .* les.qᵗ[kl, j], les.zc[kl]; color = :black, linewidth = 3)
        for (i, (label, color, style)) in enumerate((("defaults", :steelblue, :solid), ("EKI mean", :firebrick, :solid), ("EKI best", :darkorange, :dash)))
            lines!(axθ, means.θˡ[i, j, kz], zc[kz]; color, linestyle = style, linewidth = 2, label)
            lines!(axq, 1e3 .* means.qᵗ[i, j, kz], zc[kz]; color, linestyle = style, linewidth = 2)
        end
        # cloud layer of the LES
        cloudy = les.qˡ[kl, j] .> 1e-6
        any(cloudy) && hspan!(axθ, minimum(les.zc[kl][cloudy]), maximum(les.zc[kl][cloudy]); color = (:gray, 0.15))
        col == 1 && axislegend(axθ; position = :rb, framevisible = false)
    end
    save(joinpath(figures, "profiles.png"), fig)
    println("figures written to $figures")
else
    println("no evaluation at $evaluation; wrote parameters, misfit and importance figures to $figures")
end
