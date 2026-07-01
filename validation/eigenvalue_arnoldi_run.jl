# Parameterized Arnoldi eigenvalue study — grid search variant.
#
# Usage:
#   julia --project=. eigenvalue_arnoldi_run.jl <F64|F32> <weno_order> <true|false> [n_operator_steps [krylov_dim]]
#
# n_operator_steps  Number of Δt steps per Arnoldi operator call (default 1).
#                   Use 240 for a 2-day forward operator (Δt=12 min × 240).
# krylov_dim        Krylov subspace dimension (default 80; reduce for expensive multi-step runs).
#
# Outputs:
#   arnoldi_eigenvalues_<tag>.jld2
#   arnoldi_eigenvalues_<tag>.png
#   arnoldi_eigenmodes_<tag>.png

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models: boundary_condition_args
using Oceananigans.TimeSteppers: time_step!, update_state!
using Oceananigans.TurbulenceClosures: HorizontalScalarDiffusivity
using Printf
using CairoMakie
using CUDA
using LinearAlgebra
using Statistics
using Random
using JLD2

## Parse arguments
length(ARGS) in (3, 4, 5) || error("Usage: julia eigenvalue_arnoldi_run.jl <F64|F32> <weno_order> <true|false> [n_operator_steps [krylov_dim]]")

float_label      = ARGS[1]
weno_order       = parse(Int, ARGS[2])
use_closure      = parse(Bool, ARGS[3])
n_operator_steps = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
krylov_dim_arg   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 80

FT = float_label == "F64" ? Float64 :
     float_label == "F32" ? Float32 :
     error("float_type must be F64 or F32, got $(float_label)")

closure_label = use_closure ? "closure" : "noclosure"
## Keep the 1-step tag unchanged so existing JLD2 files are reused by the skip guard.
tag = n_operator_steps == 1 ?
    "$(lowercase(float_label))_weno$(weno_order)_$(closure_label)" :
    "$(lowercase(float_label))_weno$(weno_order)_$(closure_label)_$(n_operator_steps)step"

operator_days = n_operator_steps * 12 * 60 / 86400
@info "Arnoldi study: FT=$(FT), WENO order=$(weno_order), closure=$(use_closure), operator=$(round(operator_days; digits=3)) days  [tag: $tag]"

## Skip simulation if results already exist
jld2_out = "arnoldi_eigenvalues_$(tag).jld2"
arnoldi_done = isfile(jld2_out)
arnoldi_done && @info "Results found ($jld2_out) — skipping to visualization."

if !arnoldi_done

# ## DCMIP2016 parameters

Oceananigans.defaults.FloatType = FT
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# ## Grid — Nz = 32 to match DCMIP standard and keep cost manageable

Nλ = 360; Nφ = 150; Nz = 32

H = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-75, 75),
                             z = (0, H))

# ## Analytic initial conditions

Tᴱ = 310; Tᴾ = 240; Tᴹ = (Tᴱ + Tᴾ) / 2
Γ = 0.005; K = 3; b = 2

function τ_and_integrals(z)
    Hˢ = Rᵈ * Tᴹ / g
    η  = z / (b * Hˢ)
    e  = exp(-η^2)
    A = (Tᴹ - Tᴾ) / (Tᴹ * Tᴾ)
    C = (K + 2) * (Tᴱ - Tᴾ) / (2 * Tᴱ * Tᴾ)
    τ₁  = A * (1 - 2η^2) * e + exp(Γ * z / Tᴹ) / Tᴹ
    ∫τ₁ = A * z * e + (exp(Γ * z / Tᴹ) - 1) / Γ
    τ₂  = C * (1 - 2η^2) * e
    ∫τ₂ = C * z * e
    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)

virtual_temperature(λ, φ, z) = 1 / (τ_and_integrals(z)[1] - τ_and_integrals(z)[2] * F(φ))

function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))
potential_temperature(λ, φ, z) = virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

function zonal_velocity_balanced(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    Tᵥ = virtual_temperature(λ, φ, z)
    U = g / a * K * ∫τ₂ * dF(φ) * Tᵥ
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    return -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
end

# ## Model

coriolis = SphericalCoriolis(rotation_rate=Ω)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

advection = WENO(order=weno_order)

if use_closure
    closure = HorizontalScalarDiffusivity(ν=3e5, κ=3e5)
    model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants = constants,
                            advection, closure)
else
    model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants = constants,
                            advection)
end

# ## Background state

set!(model; θ=potential_temperature, u=zonal_velocity_balanced, ρ=density)

background = map(f -> copy(parent(f)), prognostic_fields(model))

# ## State vector utilities

n_per_field = Nλ * Nφ * Nz
n_fields = length(prognostic_fields(model))
N = n_fields * n_per_field

@info @sprintf("State vector dimension N = %d (%.1f M)", N, N / 1e6)

iλ = grid.Hx .+ (1:Nλ)
iφ = grid.Hy .+ (1:Nφ)
iz = grid.Hz .+ (1:Nz)

function pack_perturbation(model, background)
    x = Vector{Float64}(undef, N)
    offset = 0
    for (f, bg) in zip(prognostic_fields(model), background)
        chunk = Array(parent(f)[iλ, iφ, iz] .- bg[iλ, iφ, iz])
        x[offset+1:offset+n_per_field] .= vec(Float64.(chunk))
        offset += n_per_field
    end
    return x
end

function unpack_perturbation!(model, x, background)
    offset = 0
    for (f, bg) in zip(prognostic_fields(model), background)
        chunk = reshape(x[offset+1:offset+n_per_field], Nλ, Nφ, Nz)
        parent(f)[iλ, iφ, iz] .= bg[iλ, iφ, iz] .+ CuArray(FT.(chunk))
        offset += n_per_field
    end
    fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)
end

# ## Forward operator: n_operator_steps × Δt integration

Δt = 12 * 60

@info @sprintf("Stepping reference background (%d × Δt = %.3f days)...",
               n_operator_steps, operator_days)

for (f, bg) in zip(prognostic_fields(model), background)
    parent(f) .= bg
end
fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)
update_state!(model, compute_tendencies=false)

t_warmup = @elapsed begin
    for _ in 1:n_operator_steps
        time_step!(model, Δt)
    end
end
bg_stepped = map(f -> copy(parent(f)), prognostic_fields(model))
@info @sprintf("Reference trajectory complete (%.2f s)", t_warmup)

pass_count = Ref(0)

function apply_forward_operator(x)
    pass_count[] += 1
    unpack_perturbation!(model, x, background)
    update_state!(model, compute_tendencies=false)
    for _ in 1:n_operator_steps
        time_step!(model, Δt)
    end
    y = Vector{Float64}(undef, N)
    offset = 0
    for (f, bg_s) in zip(prognostic_fields(model), bg_stepped)
        chunk = Array(parent(f)[iλ, iφ, iz] .- bg_s[iλ, iφ, iz])
        y[offset+1:offset+n_per_field] .= vec(Float64.(chunk))
        offset += n_per_field
    end
    if pass_count[] % 10 == 0
        @info @sprintf("  Arnoldi pass %d | ||y||/||x|| = %.6f",
                       pass_count[], norm(y) / norm(x))
    end
    return y
end

# ## Arnoldi iteration

krylov_dim = krylov_dim_arg

@info @sprintf("Arnoldi iteration: k = %d (%d time_step! per operator call = %.3f days each)",
               krylov_dim, n_operator_steps, operator_days)

## Band-limited random perturbation: random phase and amplitude at each zonal
## wavenumber m=1..50, summed into a smooth v-field.  Pure white noise is
## avoided because explicit diffusion applied to 2Δx-scale noise creates large
## second-derivative tendencies that produce spurious real positive eigenvalues
## unrelated to physical instability.
Random.seed!(42)
set!(model; θ=potential_temperature, u=zonal_velocity_balanced, ρ=density)
λ_angles_ic = FT.(2π .* (0:Nλ-1) ./ Nλ)
v_band = zeros(FT, Nλ, Nφ, Nz)
m_max = 50
for m in 1:m_max
    amplitude = randn(Float32) / sqrt(Float32(m_max))
    phase     = rand(Float32) * 2π
    v_band .+= amplitude .* reshape(sin.(m .* λ_angles_ic .+ phase), Nλ, 1, 1)
end
v_band .*= FT(1.2)
## Perturb the prognostic momentum ρv (not the diagnostic model.velocities.v) —
## pack_perturbation only reads prognostic_fields(model), so seeding the
## diagnostic velocity would never reach x₀.
parent(model.momentum.ρv)[iλ, iφ, iz] .+= CuArray(v_band)
fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)
x₀ = pack_perturbation(model, background)

V = zeros(Float64, N, krylov_dim + 1)
H = zeros(Float64, krylov_dim + 1, krylov_dim)

V[:, 1] = x₀ / norm(x₀)

@info "Starting Arnoldi iteration..."

## Use a Ref to track the actual dimension used so we never assign to
## krylov_dim_used itself inside the loop (avoids Julia soft-scope warnings).
krylov_dim_used = Ref(krylov_dim)

for j in 1:krylov_dim
    @info @sprintf("=== Arnoldi step %d / %d ===", j, krylov_dim)
    w = apply_forward_operator(V[:, j])
    for i in 1:j
        H[i, j] = dot(V[:, i], w)
        w .-= H[i, j] .* V[:, i]
    end
    for i in 1:j
        s = dot(V[:, i], w)
        H[i, j] += s
        w .-= s .* V[:, i]
    end
    H[j+1, j] = norm(w)
    if H[j+1, j] < 1e-12
        @info @sprintf("Arnoldi breakdown at step %d (invariant subspace found)", j)
        krylov_dim_used[] = j
        break
    end
    V[:, j+1] = w / H[j+1, j]
    if j ≥ 5 && j % 5 == 0
        Hk_tmp = H[1:j, 1:j]
        λ_ritz = eigvals(Hk_tmp)
        abs_sorted = sort(abs.(λ_ritz); rev=true)
        top5 = abs_sorted[1:min(5, length(abs_sorted))]
        @info @sprintf("  Top 5 |λ|: %s  (>1 = amplifying per step)",
                       join([@sprintf("%.6f", s) for s in top5], ", "))
    end
end

krylov_dim = krylov_dim_used[]

# ## Extract eigenvalues and eigenvectors

Hk = H[1:krylov_dim, 1:krylov_dim]
F_eig = eigen(Hk)
λ_all = F_eig.values
eigvecs_H = F_eig.vectors

σ_all = log.(abs.(λ_all)) / (Δt * n_operator_steps) * 86400
phase_all = angle.(λ_all)

order = sortperm(abs.(λ_all); rev=true)
λ_sorted     = λ_all[order]
σ_sorted     = σ_all[order]
phase_sorted = phase_all[order]
eigvecs_sorted = eigvecs_H[:, order]

ρ_A = maximum(abs, λ_all)
@info @sprintf("Spectral radius ρ(A) = %.6f  (> 1 means unstable per step!)", ρ_A)
@info @sprintf("Over 360 steps (3 days): ρ^360 = %.4e → σ_eff = %.4f day⁻¹",
               ρ_A^360, log(ρ_A^360) / (3 * 86400) * 86400)

@info "=== Top 20 eigenvalues (sorted by |λ|) ==="
for i in 1:min(20, length(λ_sorted))
    λi = λ_sorted[i]
    @info @sprintf("  %2d | |λ| = %.6f | σ = %+.4f day⁻¹ | phase = %+.4f rad | λ = %.4f %+.4fi",
                   i, abs(λi), σ_sorted[i], phase_sorted[i], real(λi), imag(λi))
end

# ## Estimate dominant zonal wavenumber for top-N Ritz eigenmodes
#
# Each Ritz eigenvector xi = V * yi lives in state-vector space.  We extract
# the v-velocity component, average the power over latitude and altitude, and
# identify the dominant zonal wavenumber via a discrete inner product with
# exp(-im*m*λ) for m = 1…25.

@info "Estimating zonal wavenumbers and surface eigenmodes from Ritz vectors..."

pf_keys = keys(prognostic_fields(model))
v_idx = findfirst(==(:ρv), pf_keys)
isnothing(v_idx) && error("Could not locate :ρv in prognostic_fields(model); got keys $(pf_keys)")

n_label         = min(10, krylov_dim)
max_m           = Nλ ÷ 2   # check all physically distinct zonal wavenumbers
λ_angles        = (0:Nλ-1) .* (2π / Nλ)
wavenumbers_top = zeros(Int, n_label)
v_sfc_modes     = zeros(Float64, Nλ, Nφ, n_label)  # surface v-field per mode

for i in 1:n_label
    ## Take real part of eigenvector so the physical mode is real-valued
    yi = real.(eigvecs_sorted[:, i])
    xi = V[:, 1:krylov_dim] * yi      # Ritz vector (N,)

    v_off   = (v_idx - 1) * n_per_field
    v_chunk = reshape(xi[v_off+1:v_off+n_per_field], Nλ, Nφ, Nz)

    ## Save surface (k=1) v-field for eigenmode visualization
    v_sfc_modes[:, :, i] = v_chunk[:, :, 1]

    ## DFT amplitude at each wavenumber, summed over latitude and altitude
    powers = zeros(max_m)
    for m in 1:max_m
        phase_fac  = reshape(exp.(-im * m * λ_angles), Nλ, 1, 1)
        powers[m]  = abs(sum(phase_fac .* v_chunk)) / (Nφ * Nz)
    end
    wavenumbers_top[i] = argmax(powers)
end

@info "Dominant wavenumbers for top $n_label Ritz eigenmodes: $(wavenumbers_top)"

jldsave(jld2_out;
        λ_all, σ_all, σ_sorted, λ_sorted, phase_sorted,
        wavenumbers_top, v_sfc_modes,
        H = Hk, krylov_dim,
        Nλ, Nφ, Nz, Δt, n_operator_steps)

@info "Saved eigenvalues to $jld2_out"

end # !arnoldi_done

# ## Visualization — always load from JLD2

jld2_data          = jldopen(jld2_out)
λ_sorted           = jld2_data["λ_sorted"]
σ_sorted           = jld2_data["σ_sorted"]
krylov_dim_saved   = jld2_data["krylov_dim"]
Δt_saved           = jld2_data["Δt"]
n_op_saved         = get(jld2_data, "n_operator_steps", 1)
Nλ_saved           = jld2_data["Nλ"]
Nφ_saved           = jld2_data["Nφ"]
wavenumbers_top    = get(jld2_data, "wavenumbers_top", nothing)
v_sfc_modes_saved  = get(jld2_data, "v_sfc_modes", nothing)
close(jld2_data)

config_str = "$(FT), WENO-$(weno_order), closure=$(use_closure)"

fig = Figure(size=(1800, 500))
Label(fig[0, 1:6], "Arnoldi spectrum — $config_str"; fontsize=18, tellwidth=false)

ax1 = Axis(fig[1, 1];
           title = "Ritz values in the complex plane",
           xlabel = "Re(λ)", ylabel = "Im(λ)")

scatter!(ax1, real.(λ_sorted), imag.(λ_sorted);
         color = log10.(abs.(λ_sorted)), colormap = :RdBu, markersize = 8)

θ_circle = range(0, 2π; length=200)
lines!(ax1, cos.(θ_circle), sin.(θ_circle);
       color = :gray40, linestyle = :dash, linewidth = 2, label = "|λ| = 1 (neutral)")
axislegend(ax1; position = :lb)

Colorbar(fig[1, 2]; colormap = :RdBu,
         limits = extrema(log10.(abs.(λ_sorted))),
         label = "log₁₀|λ|")

op_label_str = n_op_saved == 1 ? "Δt = $(Δt_saved÷60) min" :
               "$(n_op_saved) × Δt = $(round(n_op_saved * Δt_saved / 86400; digits=2)) days"
ax2 = Axis(fig[1, 3];
           title = "|λ| spectrum ($op_label_str)",
           xlabel = "Eigenvalue index (sorted by |λ|)",
           ylabel = "|λ|")

point_colors = [abs(λi) > 1.0 ? :red : :dodgerblue for λi in λ_sorted]

scatter!(ax2, 1:krylov_dim_saved, abs.(λ_sorted); markersize = 6, color = point_colors)
hlines!(ax2, [1.0]; color = :black, linewidth = 1.5, linestyle = :dash, label = "|λ| = 1")
axislegend(ax2; position = :rt)

ax3 = Axis(fig[1, 4];
           title = "Growth rate (day⁻¹)",
           xlabel = "Eigenvalue index",
           ylabel = "σ (day⁻¹)")

scatter!(ax3, 1:krylov_dim_saved, σ_sorted; markersize = 6, color = point_colors)
hlines!(ax3, [0.46]; linestyle = :dash, color = :gray40, label = "Park et al. (2013)")
hlines!(ax3, [0.0]; color = :black, linewidth = 1.5)

if !isnothing(wavenumbers_top)
    for i in eachindex(wavenumbers_top)
        abs(λ_sorted[i]) > 1.0 || continue
        text!(ax3, i, σ_sorted[i];
              text = "m=$(wavenumbers_top[i])",
              align = (:left, :bottom), fontsize = 10, color = :red)
    end
end

axislegend(ax3; position = :rt)

png_out = "arnoldi_eigenvalues_$(tag).png"
save(png_out, fig)
@info "Saved $png_out"

# ## Eigenmode heatmaps — surface v-field for the top Ritz eigenmodes

if !isnothing(v_sfc_modes_saved)
    n_plot = size(v_sfc_modes_saved, 3)  # number of saved modes (= n_label)
    lon_plot = range(0, 360; length = Nλ_saved)
    lat_plot = range(-75, 75; length = Nφ_saved)

    fig_modes = Figure(size = (max(600, 380 * n_plot), 340))
    Label(fig_modes[0, 1:n_plot],
          "Surface v eigenmodes — $config_str"; fontsize = 16, tellwidth = false)

    ## Symmetric color limits shared across all panels
    vlim = maximum(abs, v_sfc_modes_saved)
    vlim = vlim == 0 ? 1.0 : vlim

    for i in 1:n_plot
        λi       = λ_sorted[i]
        σi       = σ_sorted[i]
        mi       = isnothing(wavenumbers_top) ? "?" : string(wavenumbers_top[i])
        unstable = abs(λi) > 1.0

        ax = Axis(fig_modes[1, i];
                  xlabel = "Longitude (°)",
                  ylabel = i == 1 ? "Latitude (°)" : "",
                  title  = @sprintf("Mode %d | m=%s | σ=%+.2f day⁻¹", i, mi, σi),
                  titlecolor = unstable ? :red : :black)

        hm = heatmap!(ax, lon_plot, lat_plot, v_sfc_modes_saved[:, :, i];
                      colormap = :balance, colorrange = (-vlim, vlim))

        if i == n_plot
            Colorbar(fig_modes[1, n_plot + 1], hm; label = "v′ (m s⁻¹)")
        end
    end

    png_modes = "arnoldi_eigenmodes_$(tag).png"
    save(png_modes, fig_modes)
    @info "Saved $png_modes"
end
