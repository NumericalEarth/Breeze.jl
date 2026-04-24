#####
##### Multi-field three-panel animation: anelastic | explicit compressible |
##### substepped compressible. Rows: w, θ, ρ″ (density anomaly, compressible
##### only), ρw (vertical momentum). Column headers annotate Δt/Ns/CFL so
##### it is visually clear how each variant is being integrated.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Statistics
using Printf

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")

wa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),    "w")
we_ts = FieldTimeSeries(joinpath(OUTDIR, "explicit.jld2"),     "w")
ws_ts = FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "w")

θa_ts = FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"),    "θ")
θe_ts = FieldTimeSeries(joinpath(OUTDIR, "explicit.jld2"),     "θ")
θs_ts = FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "θ")

# Anelastic does not save ρ (it is fixed = ρ_ref(z)).
ρe_ts = FieldTimeSeries(joinpath(OUTDIR, "explicit.jld2"),     "ρ")
ρs_ts = FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "ρ")

ta, te, ts = wa_ts.times, we_ts.times, ws_ts.times
tmax = min(ta[end], te[end], ts[end])
mask_a = findall(t -> t ≤ tmax + 1e-6, ta)
mask_e = findall(t -> t ≤ tmax + 1e-6, te)
mask_s = findall(t -> t ≤ tmax + 1e-6, ts)
Nt = min(length(mask_a), length(mask_e), length(mask_s))
@info "Animation" Nt tmax

grid = wa_ts.grid
x_km   = collect(xnodes(grid, Center())) ./ 1e3
z_km_c = collect(znodes(grid, Center())) ./ 1e3      # Center (θ, ρ)
z_km_f = collect(znodes(grid, Face()))   ./ 1e3      # Face (w)

# Reference profile for ρ″ (density perturbation) from the explicit run at t=0.
ρ_ref = mean(interior(ρe_ts[1])[:, 1, :], dims = 1)[1, :]      # size = Nz
ρ″_of(ρ_field) = interior(ρ_field)[:, 1, :] .- reshape(ρ_ref, 1, :)

# Reference θ profile as well — for θ anomaly display.
θ_ref = mean(interior(θa_ts[1])[:, 1, :], dims = 1)[1, :]
θ_anom(θ_field) = interior(θ_field)[:, 1, :] .- reshape(θ_ref, 1, :)

# Vertical momentum ρw: interpolate ρ (center) to the z-face, multiply by w.
function ρw_of(ρ_field_or_nothing, w_field)
    ρw = interior(w_field)[:, 1, :]
    if ρ_field_or_nothing === nothing
        # Anelastic: ρ = ρ_ref(z); use the reference profile interpolated to faces.
        ρᶜ = ρ_ref
        ρᶠ = [(ρᶜ[max(k-1,1)] + ρᶜ[min(k, length(ρᶜ))]) / 2 for k in 1:length(ρᶜ)+1]
        ρw .*= reshape(ρᶠ, 1, :)
    else
        ρᶜ = interior(ρ_field_or_nothing)[:, 1, :]
        Nz = size(ρᶜ, 2)
        ρᶠ = similar(ρw)
        for k in 1:size(ρw, 2)
            kc_hi = min(k, Nz)
            kc_lo = max(k - 1, 1)
            ρᶠ[:, k] .= (ρᶜ[:, kc_lo] .+ ρᶜ[:, kc_hi]) ./ 2
        end
        ρw .*= ρᶠ
    end
    return ρw
end

# Color ranges from the maxima over the entire animation.
function span(iter)
    v = 0.0
    for arr in iter
        v = max(v, maximum(abs, arr))
    end
    return v > 0 ? v : 1.0
end

# Color range from the ANELASTIC baseline (physics-only, no substepper noise).
# The substepper's local ringing can spike θ and ρ to unphysical values; letting
# those set the colorbar would wash out the other two panels.
w_max     = span(interior(wa_ts[mask_a[i]]) for i in 1:Nt)
θanom_max = span(θ_anom(θa_ts[mask_a[i]])   for i in 1:Nt)
ρanom_max = span(ρ″_of(ρe_ts[mask_e[i]])    for i in 1:Nt)   # explicit — no noise
ρw_max    = span(ρw_of(nothing, wa_ts[mask_a[i]]) for i in 1:Nt)

@info "Color spans" w_max θanom_max ρanom_max ρw_max

n = Observable(1)

# All nine heatmap slices (3 columns × {w, θ_anom, ρ″, ρw}).
wa_slice = @lift Array(interior(wa_ts[mask_a[$n]]))[:, 1, :]
we_slice = @lift Array(interior(we_ts[mask_e[$n]]))[:, 1, :]
ws_slice = @lift Array(interior(ws_ts[mask_s[$n]]))[:, 1, :]

θa_slice = @lift θ_anom(θa_ts[mask_a[$n]])
θe_slice = @lift θ_anom(θe_ts[mask_e[$n]])
θs_slice = @lift θ_anom(θs_ts[mask_s[$n]])

# Anelastic ρ″ is identically 0 (ρ ≡ ρ_ref). Display zeros — the panel exists so
# the ρ″ row has a consistent footprint.
zeros_ρa = zeros(size(interior(wa_ts[1])[:, 1, :]))
ρa_slice = @lift begin $n; zeros_ρa end
ρe_slice = @lift ρ″_of(ρe_ts[mask_e[$n]])
ρs_slice = @lift ρ″_of(ρs_ts[mask_s[$n]])

ρwa_slice = @lift ρw_of(nothing,             wa_ts[mask_a[$n]])
ρwe_slice = @lift ρw_of(ρe_ts[mask_e[$n]],   we_ts[mask_e[$n]])
ρws_slice = @lift ρw_of(ρs_ts[mask_s[$n]],   ws_ts[mask_s[$n]])

fig = Figure(size = (1800, 1300), fontsize = 14)

title_node = @lift @sprintf("Dry thermal bubble — t = %5.1f s", ta[mask_a[$n]])
fig[0, 1:4] = Label(fig, title_node, fontsize = 22, tellwidth = false)

# Column headers annotate the integrator config.
col_titles = ("Anelastic\nΔt adaptive (cfl=0.3)",
              "Compressible explicit\nΔt = 0.1 s (acoustic CFL ≈ 0.45)",
              "Compressible substepper\nΔt adaptive (cfl=0.3), Ns auto, ω=0.8\nPressureProjectionDamping(0.5)")

for (col, title) in enumerate(col_titles)
    Label(fig[1, col], title; fontsize = 14, tellwidth = false)
end

# Row labels (left-hand axis label doubles as row description).
row_labels = ("w (m/s)", "θ − ⟨θ⟩_x (K)", "ρ − ⟨ρ⟩_x (kg/m³)", "ρw (kg/(m²·s))")

function make_row(fig, row, col, field, x, z, span, label; showx = false, showy = false)
    ax = Axis(fig[row, col];
              xlabel = showx ? "x (km)" : "",
              ylabel = showy ? "z (km)" : "",
              aspect = DataAspect(),
              title  = col == 1 ? label : "")
    hm = heatmap!(ax, x, z, field; colormap = :balance, colorrange = (-span, span))
    return hm
end

# Row 1: w
hm_w1 = make_row(fig, 2, 1, wa_slice, x_km, z_km_f, w_max, row_labels[1]; showy = true)
        make_row(fig, 2, 2, we_slice, x_km, z_km_f, w_max, row_labels[1])
        make_row(fig, 2, 3, ws_slice, x_km, z_km_f, w_max, row_labels[1])
Colorbar(fig[2, 4], hm_w1; label = row_labels[1])

# Row 2: θ anomaly
hm_θ1 = make_row(fig, 3, 1, θa_slice, x_km, z_km_c, θanom_max, row_labels[2]; showy = true)
        make_row(fig, 3, 2, θe_slice, x_km, z_km_c, θanom_max, row_labels[2])
        make_row(fig, 3, 3, θs_slice, x_km, z_km_c, θanom_max, row_labels[2])
Colorbar(fig[3, 4], hm_θ1; label = row_labels[2])

# Row 3: ρ anomaly
hm_ρ1 = make_row(fig, 4, 1, ρa_slice, x_km, z_km_c, ρanom_max, row_labels[3]; showy = true)
        make_row(fig, 4, 2, ρe_slice, x_km, z_km_c, ρanom_max, row_labels[3])
        make_row(fig, 4, 3, ρs_slice, x_km, z_km_c, ρanom_max, row_labels[3])
Colorbar(fig[4, 4], hm_ρ1; label = row_labels[3])

# Row 4: ρw
hm_m1 = make_row(fig, 5, 1, ρwa_slice, x_km, z_km_f, ρw_max, row_labels[4]; showx = true, showy = true)
        make_row(fig, 5, 2, ρwe_slice, x_km, z_km_f, ρw_max, row_labels[4]; showx = true)
        make_row(fig, 5, 3, ρws_slice, x_km, z_km_f, ρw_max, row_labels[4]; showx = true)
Colorbar(fig[5, 4], hm_m1; label = row_labels[4])

mp4 = joinpath(OUTDIR, "bubble_three_way_fields.mp4")
record(fig, mp4, 1:Nt; framerate = 15) do i
    n[] = i
end
@info "wrote $mp4"

# Also dump a still at the biggest-ringing frame (around t=400 s in 07).
frame_t400 = argmin(abs.(ta[mask_a] .- 400.0))
n[] = frame_t400
save(joinpath(OUTDIR, "three_way_fields_t400.png"), fig)
@info "wrote three_way_fields_t400.png"
