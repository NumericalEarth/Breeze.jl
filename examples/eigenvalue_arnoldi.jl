# # Eigenvalue spectrum of the single-timestep operator via Arnoldi
#
# The power method finds only the **dominant** eigenvalue. To diagnose
# whether a numerical mode is contaminating the growth rate, we need the
# full leading eigenvalue spectrum of the **single time step** operator.
#
# One explicit time step IS the forward operator — the 3-day power method
# integration is this operator composed ~360 times. Any eigenvalue |λ| > 1
# of a single step gets amplified as |λ|^360 over 3 days.
#
# Arnoldi iteration finds the top k eigenvalues with only O(k) steps,
# each nearly instant on GPU at the production resolution.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: time_step!, update_state!
using Printf
using CairoMakie
using CUDA
using LinearAlgebra

# ## DCMIP2016 parameters

Oceananigans.defaults.FloatType = Float32
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

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO())

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

function unpack_perturbation!(model, x, background)
    offset = 0
    for (f, bg) in zip(prognostic_fields(model), background)
        chunk = reshape(x[offset+1:offset+n_per_field], Nλ, Nφ, Nz)
        parent(f)[iλ, iφ, iz] .= bg[iλ, iφ, iz] .+ CuArray(Float32.(chunk))
        offset += n_per_field
    end
    for f in prognostic_fields(model)
        fill_halo_regions!(f)
    end
end

# ## Single time step as the forward operator
#
# We compute the Jacobian of one time step about the background state:
#   column i = (L(bg + ε eᵢ) - L(bg)) / ε
# First step the unperturbed background to get the reference output.

Δt = 12 * 60  ## 12 minutes in seconds

@info "Stepping unperturbed background (JIT warmup + reference)..."

for (f, bg) in zip(prognostic_fields(model), background)
    parent(f) .= bg
end
for f in prognostic_fields(model)
    fill_halo_regions!(f)
end
update_state!(model, compute_tendencies=false)

t_warmup = @elapsed time_step!(model, Δt)
bg_stepped = map(f -> copy(parent(f)), prognostic_fields(model))
@info @sprintf("Reference step complete (%.2f s)", t_warmup)

pass_count = Ref(0)

function apply_forward_operator(x)
    pass_count[] += 1

    ## Reset to background + perturbation
    unpack_perturbation!(model, x, background)
    update_state!(model, compute_tendencies=false)

    ## One time step
    time_step!(model, Δt)

    ## Extract output perturbation relative to stepped background
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
#
# We build an orthonormal Krylov basis V = [v₁ v₂ … vₖ] and upper
# Hessenberg matrix H such that A V_k ≈ V_{k+1} H_{k+1,k}. The
# eigenvalues of the k×k leading submatrix of H (Ritz values)
# approximate the dominant eigenvalues of A.

krylov_dim = 80  ## number of Arnoldi steps

@info @sprintf("Arnoldi iteration: k = %d (each step = one time_step! call)", krylov_dim)

## Initial vector: wavenumber-9 v perturbation (same as power method seed)
v_ref = 1.2
function v_perturbation(λ, φ, z)
    zₚ = 15000
    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, zero(z))
    return v_ref * exp(-(φ - 40)^2 / 225) * sind(9λ) * taper
end

set!(model; θ=potential_temperature, u=zonal_velocity_balanced, ρ=density)
set!(model; v=v_perturbation)
x₀ = pack_perturbation(model, background)

## Arnoldi storage
V = zeros(Float64, N, krylov_dim + 1)  ## Krylov basis
H = zeros(Float64, krylov_dim + 1, krylov_dim)  ## Hessenberg matrix

V[:, 1] = x₀ / norm(x₀)

@info "Starting Arnoldi iteration..."

for j in 1:krylov_dim
    @info @sprintf("=== Arnoldi step %d / %d ===", j, krylov_dim)

    ## Matrix-vector product: w = A * vⱼ
    w = apply_forward_operator(V[:, j])

    ## Gram-Schmidt orthogonalization (modified, with one reorthogonalization pass)
    for i in 1:j
        H[i, j] = dot(V[:, i], w)
        w .-= H[i, j] .* V[:, i]
    end

    ## Reorthogonalization for numerical stability
    for i in 1:j
        s = dot(V[:, i], w)
        H[i, j] += s
        w .-= s .* V[:, i]
    end

    H[j+1, j] = norm(w)

    if H[j+1, j] < 1e-12
        @info @sprintf("Arnoldi breakdown at step %d (invariant subspace found)", j)
        krylov_dim = j
        break
    end

    V[:, j+1] = w / H[j+1, j]

    ## Print intermediate Ritz values every 5 steps
    if j ≥ 5 && j % 5 == 0
        Hk = H[1:j, 1:j]
        λ_ritz = eigvals(Hk)
        abs_sorted = sort(abs.(λ_ritz); rev=true)
        top5 = abs_sorted[1:min(5, length(abs_sorted))]
        @info @sprintf("  Top 5 |λ|: %s  (>1 = amplifying per step)",
                       join([@sprintf("%.6f", s) for s in top5], ", "))
    end
end

# ## Extract eigenvalues

Hk = H[1:krylov_dim, 1:krylov_dim]
λ_all = eigvals(Hk)

## For a single time step: |λ| > 1 means amplifying, |λ| < 1 means damping.
## Growth rate per day: σ = ln|λ| / Δt * 86400
σ_all = log.(abs.(λ_all)) / Δt * 86400  ## day⁻¹
phase_all = angle.(λ_all)

## Sort by |λ| (descending)
order = sortperm(abs.(λ_all); rev=true)
λ_sorted = λ_all[order]
σ_sorted = σ_all[order]
phase_sorted = phase_all[order]

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

# ## Save results

using JLD2
jldsave("arnoldi_eigenvalues.jld2";
        λ_all, σ_all, σ_sorted, λ_sorted, phase_sorted,
        H = Hk, krylov_dim,
        Nλ, Nφ, Nz, Δt)

@info "Saved eigenvalues to arnoldi_eigenvalues.jld2"

# ## Visualization

# ### Eigenvalue spectrum in the complex plane

fig = Figure(size=(1800, 500))

## Complex plane — unit circle is the stability boundary
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

## |λ| spectrum
ax2 = Axis(fig[1, 3];
           title = "|λ| spectrum (single Δt = $(Δt÷60) min)",
           xlabel = "Eigenvalue index (sorted by |λ|)",
           ylabel = "|λ|")

scatter!(ax2, 1:krylov_dim, abs.(λ_sorted); markersize = 6, color = :dodgerblue)
hlines!(ax2, [1.0]; color = :red, linewidth = 2, linestyle = :dash, label = "|λ| = 1")
axislegend(ax2; position = :rt)

## Growth rate (day⁻¹)
ax3 = Axis(fig[1, 4];
           title = "Growth rate (day⁻¹)",
           xlabel = "Eigenvalue index",
           ylabel = "σ (day⁻¹)")

scatter!(ax3, 1:krylov_dim, σ_sorted; markersize = 6, color = :dodgerblue)
hlines!(ax3, [0.46]; linestyle = :dash, color = :red, label = "Park et al. (2013)")
hlines!(ax3, [0.0]; color = :gray60)
axislegend(ax3; position = :rt)

save("arnoldi_eigenvalues.png", fig)
@info "Saved arnoldi_eigenvalues.png"
nothing
