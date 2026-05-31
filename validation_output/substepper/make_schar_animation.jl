# Animate Breeze vs CM1 Schär mountain wave — Julia + Makie.
#
# 3-panel: Breeze | CM1 (time-interpolated to Breeze cadence) | Breeze − CM1.
#
# Reads:
#   - Breeze JLD2 snapshots (any cadence)
#   - CM1 cm1out_NNNNNN.nc reference frames (600 s native cadence)
#
# Writes:
#   - .mp4 (or .gif if ffmpeg missing)
#
# Usage:
#   julia --project=docs validation_output/substepper/make_schar_animation.jl
#
# Env knobs (mirror the Python version):
#   SCHAR_BREEZE_JLD2
#   SCHAR_CM1_DIR
#   SCHAR_OUTPUT
#   SCHAR_NX, SCHAR_NZ          (Breeze interior dims)
#   SCHAR_LX_KM, SCHAR_LZ_KM
#   SCHAR_WLIM                   color limit
#   SCHAR_FPS                    output fps
#   SCHAR_FIELD                  "w" or "u"

using JLD2
using NCDatasets
using CairoMakie
using Printf

const HALO = 5

const breeze_jld2 = get(ENV, "SCHAR_BREEZE_JLD2",
    "validation_output/substepper/terrain_schar_1h_400x200_sleve_weno9_dense/terrain_schar_snapshots.jld2")
const cm1_dir = get(ENV, "SCHAR_CM1_DIR",
    "validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference")
const output_path = get(ENV, "SCHAR_OUTPUT",
    "validation_output/substepper/schar_dense_animation.mp4")
const nx_breeze = parse(Int,     get(ENV, "SCHAR_NX", "400"))
const nz_breeze = parse(Int,     get(ENV, "SCHAR_NZ", "200"))
const Lx_km     = parse(Float64, get(ENV, "SCHAR_LX_KM", "200"))
const Lz_km     = parse(Float64, get(ENV, "SCHAR_LZ_KM", "30"))
const wlim      = parse(Float64, get(ENV, "SCHAR_WLIM", "0.5"))
const fps       = parse(Int,     get(ENV, "SCHAR_FPS", "10"))
const field     = lowercase(get(ENV, "SCHAR_FIELD", "w"))

# ── Read Breeze JLD2 ─────────────────────────────────────────────────────────
@info "Reading Breeze: $breeze_jld2"
times_b, breeze_field = jldopen(breeze_jld2, "r") do file
    keys_t = filter(k -> all(isdigit, k), keys(file["timeseries/t"]))
    iters = sort(parse.(Int, keys_t))

    keys_w = filter(k -> all(isdigit, k), keys(file["timeseries/$field"]))
    iters_w = sort(parse.(Int, keys_w))

    times = Float64[]
    frames = []
    n_x_field = field == "u" ? nx_breeze : nx_breeze  # always nx (u face-x, but interior strip same)
    n_z_field = field == "w" ? nz_breeze + 1 : nz_breeze

    for it in iters_w
        sit = string(it)
        sit in keys_t ?
            push!(times, read(file["timeseries/t/$sit"])) :
            push!(times, isempty(times) ? 0.0 : last(times))
        raw = read(file["timeseries/$field/$sit"])  # storage layout in Julia: (nx_h, ny_h, nz_h) after readback
        raw3 = dropdims(raw; dims = 2)  # drop ny=1 → (nx_h, nz_h)
        # halos: 5 on each side. Take interior in x then z.
        interior = raw3[HALO + 1:HALO + n_x_field, HALO + 1:HALO + n_z_field]
        push!(frames, interior)
    end
    times, cat(frames...; dims = 3)  # (nx, nz, nt)
end
@info @sprintf("  Breeze: %d frames, t = %.0f..%.0f s, field shape = %s",
               length(times_b), times_b[1], times_b[end], string(size(breeze_field)))

# Subtract background U for the u' panel
if field == "u"
    breeze_field .-= 10.0
end

# ── Read CM1 native frames ────────────────────────────────────────────────────
function read_cm1_native(dir, field)
    files = sort([joinpath(dir, f) for f in readdir(dir) if
                  occursin(r"^cm1out_\d+\.nc$", f)])
    times = Float64[]
    arrays = []
    xh = nothing; zh = nothing
    var = field == "w" ? "winterp" : "uinterp"
    for f in files
        NCDataset(f, "r") do ds
            t = Float64(only(ds["time"][:]))
            push!(times, t)
            push!(arrays, Array(ds[var][:, :, 1, 1])')  # (nx, nz)
            if xh === nothing
                xh = Array(ds["xh"][:])
                zh = Array(ds["zh"][:])
            end
        end
    end
    perm = sortperm(times)
    arr3 = cat(arrays[perm]...; dims = 3)  # (nx, nz, nt)
    return times[perm], xh, zh, arr3
end

@info "Reading CM1 native frames: $cm1_dir"
cm1_times, x_cm1_km, z_cm1_km, cm1_native = read_cm1_native(cm1_dir, field)
if field == "u"
    cm1_native .-= 10.0
end
@info @sprintf("  CM1: %d native frames, t = %.0f..%.0f s",
               length(cm1_times), cm1_times[1], cm1_times[end])

# ── Build Breeze grid and interpolate Breeze → CM1 grid ──────────────────────
dx_b = Lx_km / nx_breeze
xb_km = collect(range(-Lx_km / 2 + dx_b / 2, Lx_km / 2 - dx_b / 2; length = nx_breeze))
nz_b_field = size(breeze_field, 2)
zb_km = field == "w" ?
    collect(range(0.0, Lz_km; length = nz_b_field)) :
    collect(range(Lz_km / nz_b_field / 2, Lz_km - Lz_km / nz_b_field / 2;
                  length = nz_b_field))

# Bilinear interp Breeze (nx_b, nz_b, nt) → CM1 grid (nx_c, nz_c, nt)
function linear_interp_1d(x_target, x_source, y_source)
    # y_source same length as x_source
    n = length(x_target)
    out = similar(x_target, length(y_source) > 0 ? typeof(y_source[1]) : Float64)
    for i in 1:n
        xt = x_target[i]
        if xt <= x_source[1]
            out[i] = y_source[1]
        elseif xt >= x_source[end]
            out[i] = y_source[end]
        else
            j = searchsortedlast(x_source, xt)
            α = (xt - x_source[j]) / (x_source[j + 1] - x_source[j])
            out[i] = (1 - α) * y_source[j] + α * y_source[j + 1]
        end
    end
    return out
end

function interp_to_grid(xb, zb, breeze_field, xc, zc)
    nx_c = length(xc); nz_c = length(zc); nt = size(breeze_field, 3)
    out = zeros(Float64, nx_c, nz_c, nt)
    # Interp in z at every (x, t), then in x
    tmp = zeros(Float64, length(xb), nz_c)
    for t_idx in 1:nt
        for i in eachindex(xb)
            tmp[i, :] = linear_interp_1d(zc, zb, breeze_field[i, :, t_idx])
        end
        for k in 1:nz_c
            out[:, k, t_idx] = linear_interp_1d(xc, xb, tmp[:, k])
        end
    end
    return out
end

@info "Interpolating Breeze → CM1 grid..."
breeze_on_cm1 = interp_to_grid(xb_km, zb_km, breeze_field, x_cm1_km, z_cm1_km)
@info @sprintf("  done: %s", string(size(breeze_on_cm1)))

# ── Time-interpolate CM1 to Breeze cadence ────────────────────────────────────
@info "Time-interpolating CM1 to Breeze cadence..."
function time_interp_cm1(cm1_times, cm1_arr, target_times)
    nx, nz, _ = size(cm1_arr)
    nt = length(target_times)
    out = zeros(Float64, nx, nz, nt)
    for t_idx in 1:nt
        t = target_times[t_idx]
        if t <= cm1_times[1]
            out[:, :, t_idx] .= cm1_arr[:, :, 1]
        elseif t >= cm1_times[end]
            out[:, :, t_idx] .= cm1_arr[:, :, end]
        else
            j = searchsortedlast(cm1_times, t)
            α = (t - cm1_times[j]) / (cm1_times[j + 1] - cm1_times[j])
            @views out[:, :, t_idx] .= (1 - α) * cm1_arr[:, :, j] .+ α * cm1_arr[:, :, j + 1]
        end
    end
    return out
end

cm1_interp = time_interp_cm1(cm1_times, cm1_native, times_b)
@info @sprintf("  done: %s", string(size(cm1_interp)))

diff_field = breeze_on_cm1 .- cm1_interp

# ── Animate with Makie ────────────────────────────────────────────────────────
@info "Rendering animation: $output_path"

fig = Figure(; size = (1500, 480), fontsize = 14)
Label(fig[0, 1:3], @sprintf("Schär mountain wave · %s field · Breeze 400×200 SLEVE+WENO9 vs CM1",
                             field == "w" ? "vertical-velocity" : "u′ = u − U");
      tellwidth = false, fontsize = 16)

ax1 = Axis(fig[1, 1]; title = "Breeze 400×200 SLEVE+WENO9",
           xlabel = "x (km)", ylabel = "z (km)")
ax2 = Axis(fig[1, 2]; title = "CM1 reference (time-interp)",
           xlabel = "x (km)")
ax3 = Axis(fig[1, 3]; title = "Breeze − CM1",
           xlabel = "x (km)")

for ax in (ax1, ax2, ax3)
    xlims!(ax, -80, 80)
    ylims!(ax, 0, 25)
    hlines!(ax, 20.0; color = :black, linestyle = :dash, linewidth = 0.5)
end

field_obs1 = Observable(breeze_on_cm1[:, :, 1])
field_obs2 = Observable(cm1_interp[:, :, 1])
field_obs3 = Observable(diff_field[:, :, 1])

cm = heatmap!(ax1, x_cm1_km, z_cm1_km, field_obs1;
              colormap = :RdBu, colorrange = (-wlim, wlim), interpolate = false)
heatmap!(ax2, x_cm1_km, z_cm1_km, field_obs2;
         colormap = :RdBu, colorrange = (-wlim, wlim))
heatmap!(ax3, x_cm1_km, z_cm1_km, field_obs3;
         colormap = :RdBu, colorrange = (-wlim * 0.6, wlim * 0.6))

unit_label = field == "w" ? "w (m/s)" : "u' (m/s)"
Colorbar(fig[1, 4], cm; label = unit_label, width = 14)

title_obs = Observable(@sprintf("t = %.0f s (%.1f min)", times_b[1], times_b[1] / 60))
Label(fig[0, 1:3], title_obs; tellwidth = false, padding = (0, 0, 40, 0), fontsize = 14)

mkpath(dirname(output_path))
record(fig, output_path, eachindex(times_b); framerate = fps) do i
    field_obs1[] = breeze_on_cm1[:, :, i]
    field_obs2[] = cm1_interp[:, :, i]
    field_obs3[] = diff_field[:, :, i]
    title_obs[]  = @sprintf("t = %.0f s (%.1f min)", times_b[i], times_b[i] / 60)
end

@info "wrote $output_path"
