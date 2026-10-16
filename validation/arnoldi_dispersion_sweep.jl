# Arnoldi normal-mode dispersion sweep for the DCMIP2016 baroclinic jet.
#
# The follow-up experiment proposed in
# `validation/baroclinic_wave_shortwave/README.md`: because the balanced
# Jablonowski–Williamson basic state is zonally symmetric, the linearized
# propagator block-diagonalizes over zonal wavenumber m — each m is an
# independent (φ, z) eigenproblem. Sweeping an Arnoldi eigen-solve over
# m = 1…120 therefore traces out the full growth-rate dispersion diagram
# σ(m), which should show *two* baroclinic branches:
#
#   * a deep synoptic branch peaking near m ≈ 10 (σ ≈ 0.76 day⁻¹ F32/WENO5,
#     0.46 F64/WENO9), and
#   * a shallow Charney-type short-wave branch peaking near m ≈ 47 (σ ≈ 1.0 —
#     the m ≈ 40–55 wave of the nonlinear runs). What amplifies this peak
#     above the deep branch is an open question — see the README; the 1D
#     Charney theory gives a declining (not flat) cut-off-free tail.
#
# Branch identity is read off the eigenfunction vertical structure: the deep
# branch fills the troposphere, the short-wave branch is surface-trapped with
# an e-folding (Rossby penetration) depth d(m) = f / (N K), K = m / (a cos φ).
#
# Method. For each m we seed a pure sin(mλ) meridional-velocity perturbation
# (which stays in the m subspace under the zonally-symmetric operator), build
# a Krylov subspace with the finite-difference tangent propagator
# A = (step ∘ perturb) − (step ∘ background) over n_operator_steps × Δt, and
# extract its Ritz values. An explicit zonal-Fourier projection onto ±m is
# applied to every operator output so nonlinear self-interaction and roundoff
# cannot leak the iterate out of the m subspace — this is what makes each m a
# clean, independent linear eigenproblem.
#
# Usage:
#   julia --project=. arnoldi_dispersion_sweep.jl [F64|F32] [weno_order] [true|false] \
#         [m_min m_max m_stride [n_operator_steps [krylov_dim [Nz [Nλ [Nφ]]]]]]
#
#   F64|F32 weno_order    float type and WENO order (default F32 5, matching examples/baroclinic_wave.jl)
#   true|false            apply a horizontal closure (default false — WENO dissipation only)
#   m_min m_max m_stride  wavenumber range (default 1 120 1)
#   n_operator_steps      Δt steps per Arnoldi operator call (default 40; Δt = 12 min → 8 h)
#   krylov_dim            Krylov subspace dimension per wavenumber (default 24)
#   Nz Nλ Nφ              grid resolution (default 64 360 150, matching examples/baroclinic_wave.jl)
#
# Outputs (tag = <ft>_weno<order>_<closure>):
#   arnoldi_dispersion_<tag>.jld2   incremental: σ(m), top Ritz values, vertical profiles
#   arnoldi_dispersion_<tag>.png    dispersion diagram σ(m) + trapping depth d(m)

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
using FFTW
using LinearAlgebra
using Statistics
using JLD2

## Parse arguments
length(ARGS) in (0, 1, 2, 3, 6, 7, 8, 9, 10, 11) ||
    error("Usage: julia arnoldi_dispersion_sweep.jl [F64|F32] [weno_order] [true|false] [m_min m_max m_stride [n_operator_steps [krylov_dim [Nz [Nλ [Nφ]]]]]]")

float_label      = length(ARGS) >= 1 ? ARGS[1] : "F32"
weno_order       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
use_closure      = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : false
m_min            = length(ARGS) >= 6 ? parse(Int, ARGS[4]) : 1
m_max            = length(ARGS) >= 6 ? parse(Int, ARGS[5]) : 120
m_stride         = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1
n_operator_steps = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 40
krylov_dim       = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 24
Nz_arg           = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 64
Nλ_arg           = length(ARGS) >= 10 ? parse(Int, ARGS[10]) : 360
Nφ_arg           = length(ARGS) >= 11 ? parse(Int, ARGS[11]) : 150

FT = float_label == "F64" ? Float64 :
     float_label == "F32" ? Float32 :
     error("float_type must be F64 or F32, got $(float_label)")

closure_label = use_closure ? "closure" : "noclosure"
tag           = "$(lowercase(float_label))_weno$(weno_order)_$(closure_label)" *
                (Nz_arg == 64 ? "" : "_nz$(Nz_arg)") *
                (Nλ_arg == 360 ? "" : "_nl$(Nλ_arg)")

Δt = 12 * 60
operator_days = n_operator_steps * Δt / 86400
m_max = min(m_max, Nλ_arg ÷ 2 - 2)          # zonal Nyquist: rfft has Nλ/2+1 bins, keep index m+1 valid
m_sweep = collect(m_min:m_stride:m_max)

@info "Arnoldi dispersion sweep: FT=$(FT), WENO=$(weno_order), closure=$(use_closure)"
@info @sprintf("  m = %d:%d:%d  (%d wavenumbers) | operator = %d × Δt = %.3f days | krylov = %d",
               m_min, m_stride, m_max, length(m_sweep), n_operator_steps, operator_days, krylov_dim)

jld2_out = "arnoldi_dispersion_$(tag).jld2"

# ## DCMIP2016 parameters (matching examples/baroclinic_wave.jl)

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

# ## Grid (Nλ = 360 resolves m ≤ 180; Nz = 64, exponentially stretched toward the surface)
#
# The vertical grid matches examples/baroclinic_wave.jl: 64 levels clustered near
# the ground (bias = :left, Δz ≈ 150 m at the surface, coarsening to ≈ 1070 m at
# the model top), with e-folding scale = H/2.

Nλ = Nλ_arg; Nφ = Nφ_arg; Nz = Nz_arg
H  = 30kilometers

z_faces = ExponentialDiscretization(Nz, 0, H; scale = H/2, bias = :left)

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-75, 75),
                             z = z_faces)

# ## Analytic balanced initial condition (JW / DCMIP2016)

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

function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))
potential_temperature(λ, φ, z) = virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

## Balanced (zonally symmetric, unperturbed) zonal wind. This is the gradient-wind
## balanced part of examples/baroclinic_wave.jl's `zonal_velocity`; the sweep
## linearizes about this steady, zonally symmetric base state and supplies the
## perturbation itself via the pure-m seed, so the localized DCMIP2016 bump is
## omitted here (a non-zonal base state would break the block-diagonal-in-m structure).
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

# ## State-vector utilities

n_per_field = Nλ * Nφ * Nz
n_fields    = length(prognostic_fields(model))
N           = n_fields * n_per_field

pf_keys = keys(prognostic_fields(model))
ρv_idx  = something(findfirst(==(:ρv), pf_keys), 3)
ρ_idx   = something(findfirst(==(:ρ),  pf_keys), 1)

@info @sprintf("State vector N = %d (%.1f M); %d fields %s; ρv at %d",
               N, N / 1e6, n_fields, string(pf_keys), ρv_idx)

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

# ## Zonal-Fourier projection onto wavenumber m
#
# Keep only the ±m harmonic of every field, in longitude. Enforces the block-
# diagonal (single-m) structure of the linearized operator exactly.

function project_onto_m!(x, m)
    offset = 0
    for _ in 1:n_fields
        chunk = reshape(view(x, offset+1:offset+n_per_field), Nλ, Nφ, Nz)
        f̂ = rfft(chunk, 1)          # (Nλ÷2+1, Nφ, Nz)
        keep = f̂[m+1, :, :]
        fill!(f̂, 0)
        f̂[m+1, :, :] .= keep
        chunk .= irfft(f̂, Nλ, 1)
        offset += n_per_field
    end
    return x
end

# ## Forward operator: n_operator_steps × Δt, differenced against stepped background

@info @sprintf("Stepping reference background (%d × Δt = %.3f days)...", n_operator_steps, operator_days)
for (f, bg) in zip(prognostic_fields(model), background)
    parent(f) .= bg
end
fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)
update_state!(model, compute_tendencies=false)

t_ref = @elapsed begin
    for _ in 1:n_operator_steps
        time_step!(model, Δt)
    end
end
bg_stepped = map(f -> copy(parent(f)), prognostic_fields(model))
@info @sprintf("Reference trajectory complete (%.2f s → %.3f s/step)", t_ref, t_ref / n_operator_steps)

current_m = Ref(m_sweep[1])

function apply_forward_operator(x)
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
    project_onto_m!(y, current_m[])   # stay in the m subspace
    return y
end

# ## Pure-m seed (sin(mλ) in ρv, tiny amplitude, scaled by background density)

λ_angles_ic = FT.(2π .* (0:Nλ-1) ./ Nλ)
ρ_bg = Array(parent(prognostic_fields(model)[ρ_idx])[iλ, iφ, iz])

function seed_wavenumber(m)
    set!(model; θ=potential_temperature, u=zonal_velocity_balanced, ρ=density)
    δv  = reshape(sin.(FT(m) .* λ_angles_ic), Nλ, 1, 1) .* ones(FT, 1, Nφ, Nz)
    δv .*= FT(1e-3)
    δρv = ρ_bg .* δv
    ρv_field = prognostic_fields(model)[ρv_idx]
    parent(ρv_field)[iλ, iφ, iz] .+= CuArray(FT.(δρv))
    fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)
    return pack_perturbation(model, background)
end

# ## Sweep

φ_centers = FT.(LinRange(-75 + 150/Nφ/2, 75 - 150/Nφ/2, Nφ))
z_centers = Array(znodes(grid, Center()))   # exponentially stretched vertical centers

σ_lead        = fill(NaN, length(m_sweep))          # leading growth rate σ(m)
ω_lead        = fill(NaN, length(m_sweep))          # angular phase speed (rad per operator)
σ_top         = fill(NaN, length(m_sweep), 5)       # top-5 Ritz growth rates
trap_depth_km = fill(NaN, length(m_sweep))          # e-folding depth of |v(z)| at jet core
lat_max_deg   = fill(NaN, length(m_sweep))          # latitude of surface v maximum
vprof_z       = zeros(Float64, Nz, length(m_sweep)) # |v(z)| profile at jet core (normalised)
vsfc_lat      = zeros(Float64, Nφ, length(m_sweep)) # |v(φ)| surface profile (normalised)

# Rossby penetration depth prediction d = f/(NK), K = m/(a cosφ), for reference
N_buoy = 1e-2   # representative stratification (s⁻¹)

V = zeros(Float64, N, krylov_dim + 1)
Hess = zeros(Float64, krylov_dim + 1, krylov_dim)

sweep_start = time_ns()

for (im, m) in enumerate(m_sweep)
  try
    current_m[] = m
    x₀ = seed_wavenumber(m)
    project_onto_m!(x₀, m)

    fill!(V, 0); fill!(Hess, 0)
    V[:, 1] = x₀ / norm(x₀)
    kused = krylov_dim

    for j in 1:krylov_dim
        w = apply_forward_operator(V[:, j])
        for i in 1:j
            Hess[i, j] = dot(V[:, i], w)
            w .-= Hess[i, j] .* V[:, i]
        end
        for i in 1:j                         # re-orthogonalise (double Gram–Schmidt)
            s = dot(V[:, i], w)
            Hess[i, j] += s
            w .-= s .* V[:, i]
        end
        Hess[j+1, j] = norm(w)
        if Hess[j+1, j] < 1e-12
            kused = j
            break
        end
        V[:, j+1] = w / Hess[j+1, j]
    end

    Hk = Hess[1:kused, 1:kused]
    Feig = eigen(Hk)
    λ = Feig.values
    order = sortperm(abs.(λ); rev=true)
    λ = λ[order]
    σ = log.(abs.(λ)) ./ (Δt * n_operator_steps) .* 86400
    for r in 1:min(5, length(σ))
        σ_top[im, r] = σ[r]
    end
    σ_lead[im] = σ[1]
    ω_lead[im] = angle(λ[1])

    # Leading eigenfunction → vertical / meridional structure of ρv
    yi = real.(Feig.vectors[:, order[1]])
    xi = V[:, 1:kused] * yi
    v_off   = (ρv_idx - 1) * n_per_field
    v_chunk = reshape(xi[v_off+1:v_off+n_per_field], Nλ, Nφ, Nz)
    v̂ = abs.(rfft(v_chunk, 1)[m+1, :, :])   # (Nφ, Nz) amplitude of the m harmonic

    jmax = argmax(v̂[:, 1])                   # jet-core latitude at the surface
    lat_max_deg[im] = φ_centers[jmax]
    prof = v̂[jmax, :]
    prof ./= (maximum(prof) == 0 ? 1 : maximum(prof))
    vprof_z[:, im] = prof
    sfc = v̂[:, 1]; sfc ./= (maximum(sfc) == 0 ? 1 : maximum(sfc))
    vsfc_lat[:, im] = sfc

    # e-folding trapping depth: lowest z where |v| drops below 1/e of surface
    kfold = findfirst(<(1/ℯ), prof)
    trap_depth_km[im] = isnothing(kfold) ? H/1e3 : z_centers[kfold] / 1e3

    elapsed = (time_ns() - sweep_start) / 1e9
    @info @sprintf("[%3d/%3d] m=%3d | σ=%+.4f day⁻¹ | ω=%+.3f | φ_max=%+5.1f° | d=%.2f km | %d modes | %.0f s elapsed",
                   im, length(m_sweep), m, σ_lead[im], ω_lead[im], lat_max_deg[im],
                   trap_depth_km[im], kused, elapsed)

    # Incremental save (resumable / usable while running)
    jldsave(jld2_out;
            m_sweep, σ_lead, ω_lead, σ_top, trap_depth_km, lat_max_deg,
            vprof_z, vsfc_lat, z_centers, φ_centers,
            Δt, n_operator_steps, krylov_dim, Nλ, Nφ, Nz, H,
            weno_order, use_closure, float_label, N_buoy, a, Ω, done_upto = im)
  catch err
    @warn @sprintf("[%3d/%3d] m=%3d FAILED: %s — recording NaN and continuing",
                   im, length(m_sweep), m, sprint(showerror, err))
  end
end

@info "Sweep complete → $jld2_out"

# ## Dispersion diagram + trapping-depth plot

d = jldopen(jld2_out)
m_sweep       = d["m_sweep"]
σ_lead        = d["σ_lead"]
trap_depth_km = d["trap_depth_km"]
lat_max_deg   = d["lat_max_deg"]
close(d)

valid = findall(!isnan, σ_lead)
ms  = m_sweep[valid]
σs  = σ_lead[valid]
ds  = trap_depth_km[valid]

# Rossby-depth prediction d(m) = f/(N K), K = m/(a cos φ_max), at the diagnosed latitude
f_of(φ) = 2Ω * sind(abs(φ))
d_pred = [ (f_of(lat_max_deg[valid][i]) / (N_buoy * (ms[i] / (a * cosd(lat_max_deg[valid][i]))))) / 1e3
           for i in eachindex(ms) ]

config_str = "$(FT), WENO-$(weno_order), closure=$(use_closure) — DCMIP JW jet"

fig = Figure(size = (1500, 560))
Label(fig[0, 1:2], "Arnoldi normal-mode dispersion — $config_str"; fontsize = 18, tellwidth = false)

ax1 = Axis(fig[1, 1];
           xlabel = "zonal wavenumber m",
           ylabel = "growth rate σ (day⁻¹)",
           title  = "Dispersion diagram σ(m)")
lines!(ax1, ms, σs; color = :dodgerblue, linewidth = 2)
scatter!(ax1, ms, σs; color = :dodgerblue, markersize = 6)
hlines!(ax1, [0.0]; color = :black, linewidth = 1)
if any(9 .∈ Ref(ms))
    vlines!(ax1, [9]; color = :seagreen, linestyle = :dash, label = "deep branch m≈9")
end
vspan!(ax1, 40, 55; color = (:orange, 0.15), label = "observed short-wave peak")
axislegend(ax1; position = :rt)

ax2 = Axis(fig[1, 2];
           xlabel = "zonal wavenumber m",
           ylabel = "trapping depth d (km)",
           title  = "Eigenfunction depth vs Rossby prediction f/(NK)",
           yscale = log10)
scatter!(ax2, ms, max.(ds, 1e-1); color = :crimson, markersize = 7, label = "eigenfunction e-folding")
lines!(ax2,  ms, max.(d_pred, 1e-1); color = :black, linestyle = :dash, label = "f/(NK)")
axislegend(ax2; position = :rt)

png_out = "arnoldi_dispersion_$(tag).png"
save(png_out, fig)
@info "Saved $png_out"
