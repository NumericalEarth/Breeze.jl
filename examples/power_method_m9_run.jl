# Power method eigenanalysis for a target zonal wavenumber.
#
# Usage:
#   julia --project=. power_method_m9_run.jl <F64|F32> <weno_order> <true|false> [target_m [n_steps_per_iter [n_iter]]]
#
# target_m          Zonal wavenumber to isolate (default 9).
# n_steps_per_iter  Model time steps per power iteration (default 1 = 12 min).
#                   Increase (e.g. 120 = 1 day) for faster convergence at the
#                   cost of more compute per iteration.
# n_iter            Maximum number of power iterations (default 500).
#
# Because the background state is zonally symmetric, A preserves each zonal
# wavenumber.  A pure m=target_m initial condition therefore stays in the
# m=target_m invariant subspace and the power method converges to the dominant
# eigenvalue within that subspace without any explicit projection step.
#
# Outputs:
#   power_method_m<m>_<tag>.jld2
#   power_method_m<m>_<tag>.png

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
using JLD2

## Parse arguments
length(ARGS) in (3, 4, 5, 6) ||
    error("Usage: julia power_method_m9_run.jl <F64|F32> <weno_order> <true|false> [target_m [n_steps_per_iter [n_iter]]]")

float_label      = ARGS[1]
weno_order       = parse(Int, ARGS[2])
use_closure      = parse(Bool, ARGS[3])
target_m         = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 9
n_steps_per_iter = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
n_iter           = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 500

FT = float_label == "F64" ? Float64 :
     float_label == "F32" ? Float32 :
     error("float_type must be F64 or F32, got $(float_label)")

closure_label = use_closure ? "closure" : "noclosure"
step_label    = n_steps_per_iter == 1 ? "1step" : "$(n_steps_per_iter)step"
tag           = "m$(target_m)_$(lowercase(float_label))_weno$(weno_order)_$(closure_label)_$(step_label)"

operator_days = n_steps_per_iter * 12 * 60 / 86400
@info "Power method: m=$(target_m), FT=$(FT), WENO=$(weno_order), closure=$(use_closure), operator=$(round(operator_days; digits=3)) days/iter, max_iter=$(n_iter)  [tag: $tag]"

jld2_out  = "power_method_$(tag).jld2"
png_out   = "power_method_$(tag).png"
done      = isfile(jld2_out)
done && @info "Results found ($jld2_out) — skipping to visualization."

if !done

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

# ## Grid (Nz=32 consistent with Arnoldi study)

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

# ## State vector utilities (identical to eigenvalue_arnoldi_run.jl)

n_per_field = Nλ * Nφ * Nz
n_fields    = length(prognostic_fields(model))
N           = n_fields * n_per_field

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

# ## Forward operator (n_steps_per_iter time steps)

Δt = 12 * 60

@info @sprintf("Stepping reference background (%d × Δt = %.3f days)...",
               n_steps_per_iter, operator_days)

for (f, bg) in zip(prognostic_fields(model), background)
    parent(f) .= bg
end
fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)
update_state!(model, compute_tendencies=false)

t_ref = @elapsed begin
    for _ in 1:n_steps_per_iter
        time_step!(model, Δt)
    end
end
bg_stepped = map(f -> copy(parent(f)), prognostic_fields(model))
@info @sprintf("Reference trajectory complete (%.2f s)", t_ref)

function apply_forward_operator(x)
    unpack_perturbation!(model, x, background)
    update_state!(model, compute_tendencies=false)
    for _ in 1:n_steps_per_iter
        time_step!(model, Δt)
    end
    y = Vector{Float64}(undef, N)
    offset = 0
    for (f, bg_s) in zip(prognostic_fields(model), bg_stepped)
        chunk = Array(parent(f)[iλ, iφ, iz] .- bg_s[iλ, iφ, iz])
        y[offset+1:offset+n_per_field] .= vec(Float64.(chunk))
        offset += n_per_field
    end
    return y
end

# ## Initial condition — pure m=target_m sine wave in ρv, uniform in φ and z
#
# The prognostic variable is ρv (momentum density), not v.
# model.velocities.v is a separate diagnostic array.
# We seed ρv = ρ₀ × δv so that the velocity perturbation is δv = 1e-3 m/s.

set!(model; θ=potential_temperature, u=zonal_velocity_balanced, ρ=density)
λ_angles_ic = FT.(2π .* (0:Nλ-1) ./ Nλ)
δv = reshape(sin.(FT(target_m) .* λ_angles_ic), Nλ, 1, 1) .* ones(FT, 1, Nφ, Nz)
δv .*= FT(1e-3)   ## 1e-3 m/s amplitude

## Scale by background density so the prognostic perturbation is δ(ρv) = ρ₀ × δv
ρ_bg = Array(parent(prognostic_fields(model)[:ρ])[iλ, iφ, iz])
δρv  = ρ_bg .* δv

ρv_field = prognostic_fields(model)[:ρv]
parent(ρv_field)[iλ, iφ, iz] .+= CuArray(FT.(δρv))
fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)

x_ref = Ref(pack_perturbation(model, background))

## Reference amplitude: surface max|v| of the seed perturbation.
## model.velocities.v is the velocity (m/s), updated from ρv after set!.
## Background meridional wind v = 0 everywhere, so this is purely the perturbation.
v_ref_amplitude = Float64(maximum(abs, Array(parent(model.velocities.v)[iλ, iφ, iz[1]])))
@info @sprintf("Reference surface max|v| = %.4e m/s", v_ref_amplitude)

# ## Power iteration
#
# Normalization follows Park et al. (2013): track max|v| at the surface.
# σ = log(max|v_sfc| / v_ref) / Δt.  All prognostic perturbations are
# rescaled by the same factor so the surface amplitude returns to v_ref
# each iteration.

@info @sprintf("Starting power method: target m=%d, max %d iterations", target_m, n_iter)

σ_history      = zeros(Float64, n_iter)
σ_mean_history = zeros(Float64, n_iter)
converged_iter = Ref(n_iter)

## For complex eigenvalues (propagating waves), the instantaneous growth
## factor oscillates due to interference of conjugate pair λ, λ* — so
## convergence is judged on a trailing window-mean of σ, not the raw
## per-iteration value, to avoid latching onto a spurious oscillation node.
window = 5

for k in 1:n_iter
    y = apply_forward_operator(x_ref[])
    ## Surface max|v| after one operator application
    ## model.velocities.v is updated by time_step! (v = ρv / ρ)
    v_sfc_max     = Float64(maximum(abs, Array(parent(model.velocities.v)[iλ, iφ, iz[1]])))
    growth_factor  = v_sfc_max / v_ref_amplitude
    σ_history[k]   = log(growth_factor) / (Δt * n_steps_per_iter) * 86400
    σ_mean_history[k] = mean(σ_history[max(1, k - window + 1):k])
    ## Rescale all fields to restore surface max|v| = v_ref_amplitude
    x_ref[]        = y .* (v_ref_amplitude / v_sfc_max)

    if k % 10 == 0
        @info @sprintf("  iter %4d | σ = %+.6f day⁻¹  (max|v_sfc| = %.4e m/s)",
                       k, σ_history[k], v_sfc_max)
    end

    ## Convergence: relative change in the trailing window-mean σ < 0.01%
    if k >= window + 1 && abs(σ_mean_history[k] - σ_mean_history[k-1]) / (abs(σ_mean_history[k]) + 1e-10) < 1e-4
        converged_iter[] = k
        @info @sprintf("Converged at iteration %d | σ (window mean) = %+.6f day⁻¹", k, σ_mean_history[k])
        break
    end
end

n_conv  = converged_iter[]
σ_final = σ_history[n_conv]
σ_mean = mean(σ_history[1:n_conv])
@info @sprintf("Final (last iter) σ = %+.6f day⁻¹", σ_final)
@info @sprintf("Mean σ over %d iters = %+.6f day⁻¹  (|λ| per step = %.8f)",
               n_conv, σ_mean, exp(σ_mean / 86400 * Δt))

## The ρv prognostic field holds the m=target_m mode; extract surface k=1
pf_keys = keys(prognostic_fields(model))
ρv_idx  = something(findfirst(==(:ρv), pf_keys), 3)
v_off   = (ρv_idx - 1) * n_per_field
v_sfc_eigenmode = reshape(x_ref[][v_off+1:v_off+n_per_field], Nλ, Nφ, Nz)[:, :, 1]

jldsave(jld2_out;
        σ_history = σ_history[1:n_conv],
        σ_final, σ_mean, target_m, n_steps_per_iter, Δt, Nλ, Nφ, Nz,
        v_sfc_eigenmode)

@info "Saved $jld2_out"

end # !done

# ## Visualization

jld2_data      = jldopen(jld2_out)
σ_history      = jld2_data["σ_history"]
σ_final        = jld2_data["σ_final"]
target_m_saved = jld2_data["target_m"]
Nλ_saved       = jld2_data["Nλ"]
Nφ_saved       = jld2_data["Nφ"]
v_sfc          = jld2_data["v_sfc_eigenmode"]
close(jld2_data)

config_str = "m=$(target_m_saved), $(FT), WENO-$(weno_order), closure=$(use_closure)"

fig = Figure(size = (1400, 450))
Label(fig[0, 1:3], "Power method — $config_str"; fontsize = 16, tellwidth = false)

## Panel 1: convergence of σ
ax1 = Axis(fig[1, 1];
           xlabel = "Power iteration",
           ylabel = "σ (day⁻¹)",
           title  = "Growth rate convergence")

lines!(ax1, 1:length(σ_history), σ_history; linewidth = 2, color = :dodgerblue)
hlines!(ax1, [0.46]; linestyle = :dash, color = :gray40, label = "Park et al. (2013)")
hlines!(ax1, [σ_final]; linestyle = :dot, color = :red, linewidth = 1.5,
        label = @sprintf("σ = %+.4f day⁻¹", σ_final))
hlines!(ax1, [0.0]; color = :black, linewidth = 1)
axislegend(ax1; position = :rb)

## Panel 2: eigenmode surface v-field
lon_plot = range(0, 360; length = Nλ_saved)
lat_plot = range(-75, 75; length = Nφ_saved)
vlim = maximum(abs, v_sfc)
vlim = vlim == 0 ? 1.0 : vlim

ax2 = Axis(fig[1, 2];
           xlabel = "Longitude (°)",
           ylabel = "Latitude (°)",
           title  = "Converged eigenmode — surface v′")

hm = heatmap!(ax2, lon_plot, lat_plot, v_sfc;
              colormap = :balance, colorrange = (-vlim, vlim))
Colorbar(fig[1, 3], hm; label = "v′ (normalised)")

save(png_out, fig)
@info "Saved $png_out"
