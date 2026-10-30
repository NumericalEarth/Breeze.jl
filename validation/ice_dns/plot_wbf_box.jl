# Figures of the ice DNS prototype (wbf_box.jl).
#
#   julia --project=PLOT_ENV validation/ice_dns/plot_wbf_box.jl OUTPUT_DIR FIGURE_DIR [M]

using CairoMakie
using DelimitedFiles
using Printf
using Statistics

output_dir = ARGS[1]
figure_dir = ARGS[2]
M = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 128
mkpath(figure_dir)

function read_timeseries(label)
    data, header = readdlm(joinpath(output_dir, "ice_dns_$(label)_timeseries.csv"), ',', Float64; header=true)
    return Dict(String(h) => data[:, n] for (n, h) in enumerate(vec(header)))
end

snapshot(label, t, name) = readdlm(joinpath(output_dir, @sprintf("ice_dns_%s_t%.2f_%s.txt", label, t, name)))
save_figure(fig, name) = (save(joinpath(figure_dir, name), fig; px_per_unit=2); @info "saved" joinpath(figure_dir, name))

const dns_color = :royalblue
const parcel_color = :orangered

#####
##### Time series
#####

function timeseries_figure(label, title)
    ts = read_timeseries(label)
    t = ts["time"]
    fig = Figure(size=(1100, 700))
    ax1 = Axis(fig[1, 1]; ylabel="mass fraction (g/kg)", title="$title: liquid and ice")
    lines!(ax1, t, 1e3 .* ts["liquid"]; color=dns_color, label="liquid, DNS")
    lines!(ax1, t, 1e3 .* ts["parcel_liquid"]; color=dns_color, linestyle=:dash, label="liquid, parcel")
    lines!(ax1, t, 1e3 .* ts["ice"]; color=parcel_color, label="ice, DNS")
    lines!(ax1, t, 1e3 .* ts["parcel_ice"]; color=parcel_color, linestyle=:dash, label="ice, parcel")
    axislegend(ax1; position=:rc)

    ax2 = Axis(fig[1, 2]; ylabel="Δ mass fraction (mg/kg)", title="change since t = 0")
    for (key, color, name) in (("liquid", dns_color, "liquid"), ("ice", parcel_color, "ice"))
        lines!(ax2, t, 1e6 .* (ts[key] .- ts[key][1]); color, label="$name, DNS")
        lines!(ax2, t, 1e6 .* (ts["parcel_" * key] .- ts["parcel_" * key][1]); color, linestyle=:dash, label="$name, parcel")
    end
    axislegend(ax2; position=:lb)

    ax3 = Axis(fig[2, 1]; xlabel="t (s)", ylabel="𝒮 (%)", title="supersaturation over liquid")
    band!(ax3, t, 100 .* (ts["S_mean"] .- ts["S_std"]), 100 .* (ts["S_mean"] .+ ts["S_std"]); color=(dns_color, 0.25))
    lines!(ax3, t, 100 .* ts["S_mean"]; color=dns_color, label="DNS mean ± std over cells")
    lines!(ax3, t, 100 .* ts["parcel_S"]; color=:black, linestyle=:dash, label="parcel")
    axislegend(ax3; position=:rb)

    ax4 = Axis(fig[2, 2]; xlabel="t (s)", ylabel="𝒮ⁱ (%)", title="supersaturation over ice")
    band!(ax4, t, 100 .* (ts["Si_mean"] .- ts["Si_std"]), 100 .* (ts["Si_mean"] .+ ts["Si_std"]); color=(parcel_color, 0.25))
    lines!(ax4, t, 100 .* ts["Si_mean"]; color=parcel_color, label="DNS mean ± std over cells")
    lines!(ax4, t, 100 .* ts["parcel_Si"]; color=:black, linestyle=:dash, label="parcel")
    axislegend(ax4; position=:rb)
    return fig, ts
end

#####
##### Size distributions
#####

function size_distribution_figure(label, title, t_final)
    fig = Figure(size=(1100, 450))
    data = snapshot(label, t_final, "ice_radius")
    rⁱ, rⁱᵖ, 𝒮ⁱ, rⁱ₀ = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    alive = rⁱ .> 0
    aliveᵖ = rⁱᵖ .> 0
    ax1 = Axis(fig[1, 1]; xlabel="r (μm)", ylabel="count", title=@sprintf("%s: ice radius at t = %.0f s", title, t_final))
    edges = range(0, 1.05 * 1e6 * maximum(vcat(rⁱ, rⁱᵖ, rⁱ₀)), length=60)
    hist!(ax1, 1e6 .* rⁱ₀; bins=edges, color=(:gray, 0.4), label="initial")
    hist!(ax1, 1e6 .* rⁱᵖ[aliveᵖ]; bins=edges, color=(parcel_color, 0.5), label="parcel")
    hist!(ax1, 1e6 .* rⁱ[alive]; bins=edges, color=(dns_color, 0.5), label="DNS")
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[1, 2]; xlabel="r² - r₀² (μm²), DNS", ylabel="count",
               title=@sprintf("growth spread: σ/mean = %.3f (DNS), %.3f (parcel)",
                              std((rⁱ.^2 .- rⁱ₀.^2)[alive]) / abs(mean((rⁱ.^2 .- rⁱ₀.^2)[alive])),
                              std((rⁱᵖ.^2 .- rⁱ₀.^2)[aliveᵖ]) / abs(mean((rⁱᵖ.^2 .- rⁱ₀.^2)[aliveᵖ]))))
    hist!(ax2, 1e12 .* (rⁱ.^2 .- rⁱ₀.^2)[alive]; bins=50, color=(dns_color, 0.6), label="DNS")
    hist!(ax2, 1e12 .* (rⁱᵖ.^2 .- rⁱ₀.^2)[aliveᵖ]; bins=50, color=(parcel_color, 0.6), label="parcel")
    axislegend(ax2; position=:rt)
    return fig
end

#####
##### Slices
#####

function slice_figure(label, times; L = 0.064)
    fig = Figure(size=(380 * length(times) + 80, 760))
    x = range(0, 100L, length=M + 1)
    for (n, t) in enumerate(times)
        for (row, name, symbol, colormap) in ((1, "S", "𝒮", :balance), (2, "Si", "𝒮ⁱ", :balance))
            field = 100 .* snapshot(label, t, name)
            limit = row == 1 ? max(0.05, maximum(abs, field)) : maximum(abs, field)
            ax = Axis(fig[row, n]; aspect=1, xlabel="x (cm)", ylabel="z (cm)",
                      title=@sprintf("%s (%%), t = %.1f s", symbol, t))
            hm = heatmap!(ax, x, x, field; colormap, colorrange=(-limit, limit))
            d = snapshot(label, t, "droplet_slab")
            if !isempty(d)
                live = d[:, 3] .> 1e-6
                scatter!(ax, 100 .* d[live, 1], 100 .* d[live, 2]; markersize=1.5, color=(:black, 0.4))
            end
            ice = snapshot(label, t, "ice_slab")
            if !isempty(ice)
                alive = ice[:, 3] .> 0
                scatter!(ax, 100 .* ice[alive, 1], 100 .* ice[alive, 2]; markersize=1.5 .+ 0.4e6 .* ice[alive, 3],
                         color=:cyan, strokecolor=:black, strokewidth=0.7)
                scatter!(ax, 100 .* ice[.!alive, 1], 100 .* ice[.!alive, 2]; marker=:xcross, markersize=7, color=:magenta)
            end
            n == length(times) && Colorbar(fig[row, n + 1], hm)
        end
    end
    return fig
end

#####
##### Fragments: survival
#####

function survival_figure(label, t_final)
    ts = read_timeseries(label)
    t = ts["time"]
    fig = Figure(size=(1500, 420))
    ax1 = Axis(fig[1, 1]; xlabel="t (s)", ylabel="fraction of fragments surviving", title="secondary-ice fragment survival")
    lines!(ax1, t, ts["ice_surviving"]; color=dns_color, label="DNS, all")
    lines!(ax1, t, ts["ice_surviving_clear"]; color=dns_color, linestyle=:dot, label="DNS, starting in clear air")
    lines!(ax1, t, ts["parcel_ice_surviving"]; color=parcel_color, linestyle=:dash, label="parcel")
    lines!(ax1, t, ts["droplets_surviving"]; color=:gray, label="droplets (DNS)")
    axislegend(ax1; position=:lb)
    ax2 = Axis(fig[1, 2]; xlabel="t (s)", ylabel="mean radius of surviving ice (μm)")
    lines!(ax2, t, 1e6 .* ts["ice_radius_mean"]; color=dns_color, label="DNS")
    lines!(ax2, t, 1e6 .* ts["parcel_ice_radius_mean"]; color=parcel_color, linestyle=:dash, label="parcel")
    axislegend(ax2; position=:lt)

    # Survival by initial radius
    data = snapshot(label, t_final, "ice_radius")
    rⁱ, rⁱᵖ, rⁱ₀, clear = data[:, 1], data[:, 2], data[:, 4], data[:, 5] .> 0
    edges = exp.(range(log(minimum(rⁱ₀)), log(maximum(rⁱ₀)) + 1e-9, length=13))
    centers = sqrt.(edges[1:end-1] .* edges[2:end])
    fraction(r, selection) = [let in_bin = selection .& (edges[b] .≤ rⁱ₀ .< edges[b+1])
                                  count(in_bin .& (r .> 0)) / max(1, count(in_bin)) end for b in 1:length(centers)]
    everywhere = trues(length(rⁱ₀))
    ax3 = Axis(fig[1, 3]; xlabel="initial radius (μm)", ylabel=@sprintf("fraction surviving at t = %.0f s", t_final),
               xscale=log10, title="survival by initial size")
    scatterlines!(ax3, 1e6 .* centers, fraction(rⁱ, everywhere); color=dns_color, label="DNS, all")
    scatterlines!(ax3, 1e6 .* centers, fraction(rⁱ, clear); color=dns_color, linestyle=:dot, label="DNS, starting in clear air")
    scatterlines!(ax3, 1e6 .* centers, fraction(rⁱ, .!clear); color=:green, linestyle=:dash, label="DNS, starting in cloud")
    scatterlines!(ax3, 1e6 .* centers, fraction(rⁱᵖ, everywhere); color=parcel_color, linestyle=:dash, label="parcel")
    axislegend(ax3; position=:rb)
    return fig
end

# Joint distribution of the ice-cell supersaturation and the growth of each crystal
function growth_scatter_figure(label, title, t_final)
    data = snapshot(label, t_final, "ice_radius")
    rⁱ, 𝒮ⁱ, rⁱ₀ = data[:, 1], data[:, 3], data[:, 4]
    fig = Figure(size=(550, 450))
    ax = Axis(fig[1, 1]; xlabel="r₀ (μm)", ylabel="r (μm)", title=@sprintf("%s: ice radius at t = %.0f s vs initial", title, t_final))
    scatter!(ax, 1e6 .* rⁱ₀, 1e6 .* rⁱ; markersize=4, color=(dns_color, 0.6))
    return fig
end

labels = (wbf = "wbf_M$M", fragments = "fragments_M$M")

if isfile(joinpath(output_dir, "ice_dns_$(labels.wbf)_timeseries.csv"))
    fig, ts = timeseries_figure(labels.wbf, "WBF")
    save_figure(fig, "ice_dns_wbf_timeseries.png")
    t_final = round(ts["time"][end], digits=1)
    save_figure(size_distribution_figure(labels.wbf, "WBF", t_final), "ice_dns_wbf_size_distribution.png")
    save_figure(slice_figure(labels.wbf, [0.0, 2.0, t_final]), "ice_dns_wbf_slice.png")
end

if isfile(joinpath(output_dir, "ice_dns_$(labels.fragments)_timeseries.csv"))
    fig, ts = timeseries_figure(labels.fragments, "Fragments")
    save_figure(fig, "ice_dns_fragments_timeseries.png")
    t_final = round(ts["time"][end], digits=1)
    save_figure(size_distribution_figure(labels.fragments, "Fragments", t_final), "ice_dns_fragments_size_distribution.png")
    save_figure(slice_figure(labels.fragments, [0.0, 0.25, 0.5, t_final]), "ice_dns_fragments_slice.png")
    save_figure(survival_figure(labels.fragments, t_final), "ice_dns_fragments_survival.png")
    save_figure(growth_scatter_figure(labels.fragments, "Fragments", t_final), "ice_dns_fragments_growth_scatter.png")
end
