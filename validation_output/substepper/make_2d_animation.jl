# Generic 2D mountain-wave animation script.
# Reads a JLD2 of Breeze snapshots and renders an x-z time-series MP4.
#
# Env knobs (all required):
#   ANIM_JLD2          path to snapshots .jld2
#   ANIM_OUTPUT        destination .mp4
#   ANIM_NX, ANIM_NZ   interior grid dims
#   ANIM_LX_KM         x-extent (km)
#   ANIM_LZ_KM         z-extent (km)
#   ANIM_WLIM          color limit for w (m/s)
#   ANIM_TITLE         figure suptitle
#   ANIM_HILL          terrain function (Julia expression evaluated on the fly)
#                       e.g. "x -> 250*exp(-(x/5e3)^2)*cos(π*x/4e3)^2"
#   ANIM_FPS           frames per second (default 6)

using JLD2
using CairoMakie
using Printf

const HALO = 5

jld2_path = ENV["ANIM_JLD2"]
output    = ENV["ANIM_OUTPUT"]
nx        = parse(Int,     ENV["ANIM_NX"])
nz        = parse(Int,     ENV["ANIM_NZ"])
Lx_km     = parse(Float64, ENV["ANIM_LX_KM"])
Lz_km     = parse(Float64, ENV["ANIM_LZ_KM"])
wlim      = parse(Float64, ENV["ANIM_WLIM"])
title_str = get(ENV, "ANIM_TITLE", "Mountain-wave evolution")
hill_str  = get(ENV, "ANIM_HILL", "x -> 0.0")
fps       = parse(Int,     get(ENV, "ANIM_FPS", "6"))

# Build the hill function from the user-supplied expression
const hill_fn = eval(Meta.parse(hill_str))

# Grid axes (centers)
dx = Lx_km / nx
xb_km = collect(range(-Lx_km / 2 + dx / 2, Lx_km / 2 - dx / 2; length = nx))
zb_face_km = collect(range(0, Lz_km; length = nz + 1))
zb_cc_km   = 0.5 .* (zb_face_km[1:end-1] .+ zb_face_km[2:end])

@info "Reading $jld2_path"
times, w_arr = jldopen(jld2_path, "r") do f
    ts = f["timeseries"]
    iters = sort(parse.(Int, filter(k -> all(isdigit, k), keys(ts["w"]))))
    nt = length(iters)
    ts_arr = zeros(Float64, nt)
    w_full = zeros(Float64, nx, nz, nt)
    for (n, it) in enumerate(iters)
        sit = string(it)
        ts_arr[n] = sit in keys(ts["t"]) ?
            ts["t/$sit"] : (n == 1 ? 0.0 : ts_arr[n - 1])
        raw = ts["w/$sit"]              # (nx_h, ny_h, nz_face_h)
        int_w = raw[HALO + 1:HALO + nx, :, HALO + 1:HALO + nz + 1]
        int_w_2d = dropdims(int_w; dims = 2)             # (nx, nz_face)
        w_full[:, :, n] = 0.5 .* (int_w_2d[:, 1:end-1] .+ int_w_2d[:, 2:end])
    end
    ts_arr, w_full
end
@info @sprintf("loaded %d frames, t = %.0f..%.0f s", length(times), times[1], times[end])

h_terrain_km = [Float64(hill_fn(x * 1000)) / 1000 for x in xb_km]
h_max = maximum(h_terrain_km)

fig = Figure(; size = (1200, 500), fontsize = 13)
suptitle = Label(fig[0, 1:2], ""; tellwidth = false, fontsize = 15,
                 halign = :center, padding = (0, 0, 6, 0))

ax = Axis(fig[1, 1]; xlabel = "x (km)", ylabel = "z (km)")
xlims!(ax, -Lx_km / 2, Lx_km / 2); ylims!(ax, 0, Lz_km)

w_obs = Observable(w_arr[:, :, 1])
hm = heatmap!(ax, xb_km, zb_cc_km, w_obs;
              colormap = :RdBu, colorrange = (-wlim, wlim))
# Terrain overlay
band!(ax, xb_km, zeros(length(xb_km)), h_terrain_km; color = (:black, 0.7))

Colorbar(fig[1, 2], hm; label = "w (m/s)", width = 14)

mkpath(dirname(output))
record(fig, output, eachindex(times); framerate = fps) do n
    w_obs[] = w_arr[:, :, n]
    suptitle.text = @sprintf("%s — t = %.0f s (%.1f min)",
                              title_str, times[n], times[n] / 60)
end
@info "wrote $output"
