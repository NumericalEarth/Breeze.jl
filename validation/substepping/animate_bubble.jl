#####
##### Side-by-side animation of anelastic vs compressible w for the dry thermal
##### bubble. Reads the JLD2 outputs written by
##### `07_dry_thermal_bubble_wizard.jl`.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")

isfile(joinpath(OUTDIR, "anelastic.jld2"))    || error("run 07_dry_thermal_bubble_wizard.jl first")
isfile(joinpath(OUTDIR, "compressible.jld2")) || error("run 07_dry_thermal_bubble_wizard.jl first")

wa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),    "w")
wc_ts = FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "w")

ta = wa_ts.times
tc = wc_ts.times
Nt = min(length(ta), length(tc))
@info "Building animation" Nt tmax = ta[Nt]

# Grid axes in km. The w field lives on Center, Center, Face.
grid = wa_ts.grid
x_km = collect(xnodes(grid, Center())) ./ 1e3
z_km = collect(znodes(grid, Face()))   ./ 1e3

# Shared colour scale based on peak |w| across all frames of both runs.
vmax = let v = 0.0
    for i in 1:Nt
        v = max(v,
                maximum(abs, interior(wa_ts[i])),
                maximum(abs, interior(wc_ts[i])))
    end
    isfinite(v) && v > 0 ? v : 1.0
end
@info "color range" vmax

n = Observable(1)

wa_slice = @lift Array(interior(wa_ts[$n]))[:, 1, :]
wc_slice = @lift Array(interior(wc_ts[$n]))[:, 1, :]

fig = Figure(size = (1500, 620), fontsize = 15)

title_node = @lift @sprintf("Dry thermal bubble (Δθ = 10 K, N² = 10⁻⁶ s⁻²) — t = %6.1f s", ta[$n])
fig[0, 1:3] = Label(fig, title_node, fontsize = 20, tellwidth = false)

ax_a = Axis(fig[1, 1]; title = "Anelastic",        xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
ax_c = Axis(fig[1, 2]; title = "Compressible (substepper)", xlabel = "x (km)", aspect = DataAspect())

hm_a = heatmap!(ax_a, x_km, z_km, wa_slice; colormap = :balance, colorrange = (-vmax, vmax))
heatmap!(           ax_c, x_km, z_km, wc_slice; colormap = :balance, colorrange = (-vmax, vmax))

Colorbar(fig[1, 3], hm_a; label = "w (m/s)")

mp4 = joinpath(OUTDIR, "bubble_side_by_side.mp4")
record(fig, mp4, 1:Nt; framerate = 15) do i
    n[] = i
end
@info "wrote $mp4"

gif = joinpath(OUTDIR, "bubble_side_by_side.gif")
record(fig, gif, 1:Nt; framerate = 15) do i
    n[] = i
end
@info "wrote $gif"
