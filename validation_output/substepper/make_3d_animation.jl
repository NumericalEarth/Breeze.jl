# Animate the 3D bell-mountain wave: x-z centerline | y-z downstream | x-y plan
# view at z = 8 km. Three panels evolving together at the script's snapshot
# cadence.
#
# Inputs:
#   validation_output/substepper/terrain_3d_mountain_wave_200x200x100_gpu/
#     terrain_3d_mountain_wave_snapshots.jld2
#
# Output:
#   validation_output/substepper/schar_3d_mountain_wave_animation.mp4
#
# Run with the plot env (has CairoMakie + JLD2):
#   julia --project=validation_output/substepper/plot_env \
#         validation_output/substepper/make_3d_animation.jl

using JLD2
using CairoMakie
using Printf

const HALO = 5

jld2_path   = get(ENV, "SCHAR3D_JLD2",
    "validation_output/substepper/terrain_3d_mountain_wave_200x200x100_gpu/" *
    "terrain_3d_mountain_wave_snapshots.jld2")
output_path = get(ENV, "SCHAR3D_OUTPUT",
    "validation_output/substepper/schar_3d_mountain_wave_animation.mp4")
nx          = parse(Int,     get(ENV, "SCHAR3D_NX", "200"))
ny          = parse(Int,     get(ENV, "SCHAR3D_NY", "200"))
nz          = parse(Int,     get(ENV, "SCHAR3D_NZ", "100"))
Lx_km       = parse(Float64, get(ENV, "SCHAR3D_LX_KM", "80"))
Ly_km       = parse(Float64, get(ENV, "SCHAR3D_LY_KM", "80"))
Lz_km       = parse(Float64, get(ENV, "SCHAR3D_LZ_KM", "25"))
wlim        = parse(Float64, get(ENV, "SCHAR3D_WLIM", "0.3"))
z_plan_km   = parse(Float64, get(ENV, "SCHAR3D_Z_PLAN", "8"))
x_lee_km    = parse(Float64, get(ENV, "SCHAR3D_X_LEE", "20"))
fps         = parse(Int,     get(ENV, "SCHAR3D_FPS", "8"))

# Grid axes (centers)
xb_km = collect(range(-Lx_km / 2 + Lx_km / (2nx), Lx_km / 2 - Lx_km / (2nx); length = nx))
yb_km = collect(range(-Ly_km / 2 + Ly_km / (2ny), Ly_km / 2 - Ly_km / (2ny); length = ny))
zb_face_km = collect(range(0, Lz_km; length = nz + 1))
zb_cc_km   = 0.5 .* (zb_face_km[1:end-1] .+ zb_face_km[2:end])

j_mid   = ny ÷ 2 + 1
i_lee   = clamp(searchsortedfirst(xb_km, x_lee_km), 1, nx)
k_plan  = clamp(searchsortedfirst(zb_cc_km, z_plan_km), 1, nz)

@info "Reading 3D snapshots from $jld2_path"
times, slice_xz, slice_yz, plan_xy = jldopen(jld2_path, "r") do file
    ts = file["timeseries"]
    iters = sort(parse.(Int, filter(k -> all(isdigit, k), keys(ts["w"]))))
    nt = length(iters)
    times = zeros(Float64, nt)
    s_xz  = zeros(Float64, nx, nz, nt)   # x-z at y = j_mid (cell centers)
    s_yz  = zeros(Float64, ny, nz, nt)   # y-z at x = i_lee
    p_xy  = zeros(Float64, nx, ny, nt)   # x-y at z = k_plan

    for (n, it) in enumerate(iters)
        sit = string(it)
        times[n] = sit in keys(ts["t"]) ?
            ts["t/$sit"] : (n == 1 ? 0.0 : times[n - 1])

        # Storage on disk (column-major) is (nx_with_halo, ny_with_halo,
        # nz_face_with_halo). Interior strip:
        raw = ts["w/$sit"]
        int_w = raw[HALO + 1:HALO + nx, HALO + 1:HALO + ny, HALO + 1:HALO + nz + 1]
        # face-z → cell-center z
        int_w_cc = 0.5 .* (int_w[:, :, 1:end-1] .+ int_w[:, :, 2:end])

        s_xz[:, :, n] = int_w_cc[:, j_mid, :]      # (nx, nz)
        s_yz[:, :, n] = int_w_cc[i_lee, :, :]      # (ny, nz)
        p_xy[:, :, n] = int_w_cc[:, :, k_plan]     # (nx, ny)
    end
    times, s_xz, s_yz, p_xy
end
@info @sprintf("loaded %d frames, t = %.0f..%.0f s", length(times), times[1], times[end])

# Build figure
fig = Figure(; size = (1500, 580), fontsize = 13)
suptitle = Label(fig[0, 1:3], "";
                 tellwidth = false, fontsize = 16,
                 halign = :center, padding = (0, 0, 8, 0))

ax_xz = Axis(fig[1, 1]; title = @sprintf("x–z slice at y = 0"),
             xlabel = "x (km)", ylabel = "z (km)")
ax_yz = Axis(fig[1, 2]; title = @sprintf("y–z slice at x = %.0f km", xb_km[i_lee]),
             xlabel = "y (km)", ylabel = "z (km)")
ax_xy = Axis(fig[1, 3]; title = @sprintf("x–y plan at z = %.1f km", zb_cc_km[k_plan]),
             xlabel = "x (km)", ylabel = "y (km)",
             aspect = DataAspect())

xlims!(ax_xz, -Lx_km / 2, Lx_km / 2); ylims!(ax_xz, 0, Lz_km)
xlims!(ax_yz, -Ly_km / 2, Ly_km / 2); ylims!(ax_yz, 0, Lz_km)
xlims!(ax_xy, -Lx_km / 2, Lx_km / 2); ylims!(ax_xy, -Ly_km / 2, Ly_km / 2)

xz_obs = Observable(slice_xz[:, :, 1])
yz_obs = Observable(slice_yz[:, :, 1])
xy_obs = Observable(plan_xy[:, :, 1])

hm = heatmap!(ax_xz, xb_km, zb_cc_km, xz_obs;
              colormap = :RdBu, colorrange = (-wlim, wlim))
heatmap!(ax_yz, yb_km, zb_cc_km, yz_obs;
        colormap = :RdBu, colorrange = (-wlim, wlim))
heatmap!(ax_xy, xb_km, yb_km, xy_obs;
        colormap = :RdBu, colorrange = (-wlim, wlim))

Colorbar(fig[1, 4], hm; label = "w (m/s)", width = 14)

mkpath(dirname(output_path))
record(fig, output_path, eachindex(times); framerate = fps) do n
    xz_obs[] = slice_xz[:, :, n]
    yz_obs[] = slice_yz[:, :, n]
    xy_obs[] = plan_xy[:, :, n]
    suptitle.text = @sprintf("3D bell mountain wave (200×200×100 SLEVE+WENO9, H100) · t = %.0f s (%.0f min)",
                              times[n], times[n] / 60)
end

@info "wrote $output_path"
