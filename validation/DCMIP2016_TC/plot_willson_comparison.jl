# Willson et al. (2024) Fig. 5/7/8 comparison figures (CairoMakie), overlaying Breeze on the
# DCMIP2016 ensemble. Reads the published reference profiles from refdata/ (see refdata/DOWNLOAD.md)
# and the Breeze tangential-wind reductions from postproc/breeze_*_010.csv (written by
# extract_willson_comparison_data.jl); writes postproc/willson_fig{5,7,8}.png. Both sides use the
# azimuthal-mean TANGENTIAL wind on the published 0.25°-great-circle radial grid.
#   julia --project=<env-with-CairoMakie> plot_willson_comparison.jl
using CairoMakie
using DelimitedFiles
using Printf

const BASE = @__DIR__
const WD   = joinpath(BASE, "refdata")
const PP   = joinpath(BASE, "postproc")
const A_KM = 6371.22
const DEG  = π / 180
radii(n) = [(k-1) * 0.25 * DEG * A_KM for k in 1:n]      # r_k = k·0.25°-gc (km), k = 0,1,…

CairoMakie.activate!(type = "png")

# ---- published reference parsing (TempestExtremes radial profiles + trajectories) ----
sim_day(h) = (parse(Int, h[4]) - 1) + parse(Int, h[5]) / 24    # head: track,year,month,day,hour,…

"per-timestep (day, rprof) pairs for track_id 0"
function read_rprof(path)
    out = Tuple{Float64, Vector{Float64}}[]
    for ln in readlines(path)
        occursin('"', ln) || continue
        head = filter(!isempty, strip.(split(split(ln, '"')[1], ',')))
        (length(head) < 5 || parse(Int, head[1]) != 0) && continue
        prof = parse.(Float64, split(match(r"\[(.*?)\]", ln).captures[1], ','))
        push!(out, (sim_day(head), prof))
    end
    return out
end

"days-4-10 time-mean radial profile"
function mean_profile(path; matureday = 4.0)
    profs = [p for (d, p) in read_rprof(path) if d >= matureday]
    return reduce(+, profs) ./ length(profs)
end

function willson_profiles(sub, res, suffix)              # name => days-4-10 mean profile
    d = Dict{String, Vector{Float64}}()
    for f in readdir(joinpath(WD, sub, res); join = true)
        endswith(f, suffix) || continue
        d[replace(basename(f), suffix => "")] = mean_profile(f)
    end
    return d
end

function willson_rz(model, res)                          # (r_km, z_km, S[r,z]) composite
    alts = Float64[]; profs = Vector{Float64}[]
    for f in readdir(joinpath(WD, "wind_radial_profiles", res); join = true)
        m = match(Regex("^" * model * "_(\\d+)\\.txt\$"), basename(f))
        m === nothing && continue
        push!(alts, parse(Int, m.captures[1])); push!(profs, mean_profile(f))
    end
    o = sortperm(alts); z = alts[o] ./ 1000; P = profs[o]
    r = radii(length(P[1])); S = zeros(length(r), length(z))
    for (iz, p) in enumerate(P); S[:, iz] .= p; end
    return r, z, S
end

function traj_msp(path)                                  # (days, msp_hPa) sorted, track 0
    out = Tuple{Float64, Float64}[]
    for ln in readlines(path)[2:end]
        c = strip.(split(ln, ','))
        (length(c) < 11 || isempty(c[1]) || parse(Int, c[1]) != 0) && continue
        push!(out, ((parse(Int, c[4]) - 1) + parse(Int, c[5]) / 24, parse(Float64, c[10]) / 100))
    end
    sort!(out); return first.(out), last.(out)
end
function wind_mws_ts(path)                               # (days, mws_ms) per-timestep peak tangential
    out = [(d, maximum(p)) for (d, p) in read_rprof(path)]; sort!(out)
    return first.(out), last.(out)
end
ensemble(f, sub, res, suf) = Dict(replace(basename(p), suf => "") => f(p)
    for p in readdir(joinpath(WD, sub, res); join = true) if endswith(p, suf))

# ---- Breeze reductions (postproc CSVs) ----
function breeze_table(name)
    data, _ = readdlm(joinpath(PP, name), ','; header = true)
    return data
end
function breeze_cols(data, run, cols...)
    idx = data[:, 1] .== run
    return (Float64.(data[idx, c]) for c in cols)
end

const STYLE = [("weno5_0.25deg", :dodgerblue, "Breeze WENO5 (0.25°)"),
               ("weno9_0.25deg", :crimson,    "Breeze WENO9 (0.25°)")]
const GREY = RGBAf(0.45, 0.45, 0.45, 0.75)

# ===================== Fig 5 — intensity time series (cf. Willson Fig. 5) =====================
let
    m50, m25 = ensemble(traj_msp, "trajectories", "50km", "_trajectories.csv"),
               ensemble(traj_msp, "trajectories", "25km", "_trajectories.csv")
    w50, w25 = ensemble(wind_mws_ts, "wind_radial_profiles", "50km", "_1000.txt"),
               ensemble(wind_mws_ts, "wind_radial_profiles", "25km", "_1000.txt")
    wp = breeze_table("breeze_windpressure_010.csv")

    fig = Figure(size = (780, 880))
    axm = Axis(fig[1, 1]; ylabel = "minimum sea-level pressure (hPa)",
               title = "DCMIP2016 TC intensity (time series): Breeze vs. intercomparison")
    axw = Axis(fig[2, 1]; xlabel = "time (days)", ylabel = "max 1 km tangential wind (m/s)")
    drawens(ax, d, ls) = for (name, (x, y)) in d
        startswith(name, "fv3") || lines!(ax, x, y; color = GREY, linestyle = ls, linewidth = 1)
    end
    drawens(axm, m50, :dash); drawens(axm, m25, :solid)
    drawens(axw, w50, :dash); drawens(axw, w25, :solid)
    lines!(axm, m50["fv3_dzlow"]...; color = :black, linestyle = :dash, linewidth = 1.8)
    lines!(axw, w50["fv3_dzlow"]...; color = :black, linestyle = :dash, linewidth = 1.8)
    for (run, c, _) in STYLE
        d, p = breeze_cols(wp, run, 2, 3); _, w = breeze_cols(wp, run, 2, 4)
        lines!(axm, d, p; color = c, linewidth = 2.4); lines!(axw, d, w; color = c, linewidth = 2.4)
    end
    xlims!(axm, 0, 10); ylims!(axm, 910, 1015); xlims!(axw, 0, 10); ylims!(axw, 0, 72)
    elems = [LineElement(color = GREY, linestyle = :dash), LineElement(color = GREY),
             LineElement(color = :black, linestyle = :dash),
             [LineElement(color = c) for (_, c, _) in STYLE]...]
    labs = ["DCMIP2016 ensemble (50 km)", "DCMIP2016 ensemble (25 km)", "FV3 (50 km)",
            [l for (_, _, l) in STYLE]...]
    axislegend(axw, elems, labs; position = :rb, framevisible = false, labelsize = 10)
    save(joinpath(PP, "willson_fig5_intensity.png"), fig)
end

# ===================== Fig 7 — mature radial structure (cf. Willson Fig. 7) =====================
let
    w50, w25 = willson_profiles("wind_radial_profiles", "50km", "_1000.txt"),
               willson_profiles("wind_radial_profiles", "25km", "_1000.txt")
    p50, p25 = willson_profiles("surface_pressure_radial_profiles", "50km", ".txt"),
               willson_profiles("surface_pressure_radial_profiles", "25km", ".txt")
    prof = breeze_table("breeze_profiles_010.csv")

    fig = Figure(size = (780, 880))
    axp = Axis(fig[1, 1]; ylabel = "azim-avg surface pressure (hPa)",
               title = "DCMIP2016 TC radial structure (days 4–10 mean): Breeze vs. intercomparison")
    axw = Axis(fig[2, 1]; xlabel = "distance from center (km)", ylabel = "azim-avg 1 km tangential wind (m/s)")
    drawp(ax, d, ls) = for (name, y) in d
        startswith(name, "fv3") && continue
        yy = copy(y) ./ 100; yy[1] = NaN
        lines!(ax, radii(length(yy)), yy; color = GREY, linestyle = ls, linewidth = 1)
    end
    draww(ax, d, ls) = for (name, y) in d
        startswith(name, "fv3") || lines!(ax, radii(length(y)), y; color = GREY, linestyle = ls, linewidth = 1)
    end
    drawp(axp, p50, :dash); drawp(axp, p25, :solid)
    draww(axw, w50, :dash); draww(axw, w25, :solid)
    yp = copy(p50["fv3_dzlow"]) ./ 100; yp[1] = NaN
    lines!(axp, radii(length(yp)), yp; color = :black, linestyle = :dash, linewidth = 1.8)
    lines!(axw, radii(length(w50["fv3_dzlow"])), w50["fv3_dzlow"]; color = :black, linestyle = :dash, linewidth = 1.8)
    for (run, c, _) in STYLE
        r, p = breeze_cols(prof, run, 2, 4); _, w = breeze_cols(prof, run, 2, 3)
        lines!(axp, r, p; color = c, linewidth = 2.4); lines!(axw, r, w; color = c, linewidth = 2.4)
    end
    xlims!(axp, 0, 1000); ylims!(axp, 910, 1018); xlims!(axw, 0, 1000); ylims!(axw, 0, 60)
    elems = [LineElement(color = GREY, linestyle = :dash), LineElement(color = GREY),
             LineElement(color = :black, linestyle = :dash),
             [LineElement(color = c) for (_, c, _) in STYLE]...]
    labs = ["DCMIP2016 ensemble (50 km)", "DCMIP2016 ensemble (25 km)", "FV3 (50 km)",
            [l for (_, _, l) in STYLE]...]
    axislegend(axp, elems, labs; position = :rt, framevisible = false, labelsize = 10)
    save(joinpath(PP, "willson_fig7_radial.png"), fig)
end

# ===================== Fig 8 — radius–height tangential wind (cf. Willson Fig. 8) =====================
let
    function breeze_rz(run)
        data = breeze_table("breeze_fields_rz_010.csv")
        idx = data[:, 1] .== run
        r = Float64.(data[idx, 2]); z = Float64.(data[idx, 3]); vt = Float64.(data[idx, 4])
        rs = sort(unique(r)); zs = sort(unique(z))
        S = fill(NaN, length(rs), length(zs))
        ri = Dict(v => i for (i, v) in enumerate(rs)); zi = Dict(v => i for (i, v) in enumerate(zs))
        for n in eachindex(r); S[ri[r[n]], zi[z[n]]] = vt[n]; end
        return rs, zs, S
    end
    rows = [(:w, "fv3_dzlow", "50km", "FV3 (50 km)"), (:w, "acme-a", "25km", "ACME-A (25 km)"),
            (:w, "cam-se", "25km", "CAM-SE (25 km)"), (:b, "weno5_0.25deg", "", "Breeze WENO5 (0.25°)"),
            (:b, "weno9_0.25deg", "", "Breeze WENO9 (0.25°)")]
    # diverging colormap, white at vt = 0 over [-10, 60] (cyclonic red / anticyclonic blue)
    cmap = cgrad([:navy, :dodgerblue, :white, :tomato, :darkred], [0.0, 0.10, 10/70, 0.55, 1.0])
    levels = -10:2.5:60

    fig = Figure(size = (720, 1080))
    local cf
    for (i, (kind, key, res, title)) in enumerate(rows)
        last = i == length(rows)
        ax = Axis(fig[i, 1]; ylabel = "height (km)", xlabel = last ? "radius (km)" : "",
                  title = title, titlecolor = (kind == :b ? :crimson : :black), titlesize = 12)
        r, z, S = kind == :w ? willson_rz(key, res) : breeze_rz(key)
        cf = contourf!(ax, r, z, S; levels = levels, colormap = cmap, extendlow = :auto, extendhigh = :auto)
        contour!(ax, r, z, S; levels = [0.0], color = :black, linewidth = 0.6)
        xlims!(ax, 0, 500); ylims!(ax, 0, 16)
        last || hidexdecorations!(ax; grid = false)
    end
    Colorbar(fig[1:length(rows), 2], cf; label = "azim-mean tangential wind vₜ (m/s)  (+ cyclonic / − anticyclonic)")
    Label(fig[0, 1:2], "DCMIP2016 TC radius–height tangential wind (days 4–10):\n" *
          "FV3 (50 km) and ACME-A / CAM-SE / Breeze (25 km · 0.25°)", fontsize = 12)
    save(joinpath(PP, "willson_fig8_rz.png"), fig)
end

println("wrote postproc/willson_fig{5,7,8}.png")
