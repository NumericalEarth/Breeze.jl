# # Eigenvalue spectrum of the baroclinic instability via Arnoldi iteration
#
# The power method finds only the **dominant** eigenvalue. To diagnose why
# it overshoots the Park et al. reference (σ ≈ 0.46 day⁻¹), we need the
# full leading eigenvalue spectrum.
#
# The 3-day forward integration defines a linear operator
# ``L : \delta\mathbf{x} \to \delta\mathbf{x}'`` on the perturbation
# state vector. Constructing the explicit matrix is intractable
# (8.64 M DOF → 33 years of wall time), but **Arnoldi iteration** finds
# the top k eigenvalues with only O(k) matrix-vector products.
#
# ## Cost estimate
#
# | Quantity | Value |
# |----------|-------|
# | State DOF | 5 × 360 × 150 × 32 = 8,640,000 |
# | One forward pass (A100) | ~2 min |
# | Krylov dimension k = 80 | ~2.7 hours |
# | Krylov basis memory | 80 × 8.64M × 8 B ≈ 5.5 GB (CPU) |
# | Explicit matrix (comparison) | 8.64M² × 4 B ≈ 300 TB — infeasible |

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!, reset!
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

# ## Forward operator: pack/unpack state ↔ vector
#
# The perturbation state vector is the concatenation of the interior
# values of all prognostic fields (ρ, ρu, ρv, ρw, ρθ), stored as
# Float64 on CPU. The forward operator transfers to GPU Float32,
# integrates 3 days, and extracts the result back to CPU Float64.

n_per_field = Nλ * Nφ * Nz
n_fields = length(prognostic_fields(model))
N = n_fields * n_per_field

@info @sprintf("State vector dimension N = %d (%.1f M)", N, N / 1e6)
@info @sprintf("Explicit matrix would need %d forward passes (%.1f years at 2 min/pass)",
               N, N * 2 / 60 / 24 / 365)

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
        parent(f)[iλ, iφ, iz] .= bg[iλ, iφ, iz] .+ CuArray(Float32.(chunk))
        offset += n_per_field
    end
    for f in prognostic_fields(model)
        fill_halo_regions!(f)
    end
end

# ## Simulation (reused for every forward pass)

Δτ = 3days

simulation = Simulation(model; Δt=12minutes, stop_time=Δτ)
conjure_time_step_wizard!(simulation; cfl=1.4, max_Δt=12minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

pass_count = Ref(0)

function arnoldi_progress(sim)
    if iteration(sim) % 100 == 0
        v = sim.model.velocities.v
        @info @sprintf("  [pass %d] step %d | t = %s | max|v| = %.4e",
                       pass_count[], iteration(sim), prettytime(sim), maximum(abs, v))
    end
    return nothing
end

add_callback!(simulation, arnoldi_progress, IterationInterval(1))

function apply_forward_operator(x)
    pass_count[] += 1
    t_start = time()

    ## Set model state = background + perturbation
    unpack_perturbation!(model, x, background)
    reset!(model.clock)
    simulation.stop_time = Δτ
    update_state!(model, compute_tendencies=false)

    ## Integrate 3 days
    run!(simulation)

    ## Extract perturbation
    y = pack_perturbation(model, background)

    elapsed = time() - t_start
    ynorm = norm(y)
    amplification = ynorm / norm(x)
    @info @sprintf("Forward pass %d complete (%.1f s) | ||y||/||x|| = %.4f | σ_inst = %.4f day⁻¹",
                   pass_count[], elapsed, amplification,
                   log(amplification) / (3 * 86400) * 86400)
    return y
end

# ## Arnoldi iteration
#
# We build an orthonormal Krylov basis V = [v₁ v₂ … vₖ] and upper
# Hessenberg matrix H such that A V_k ≈ V_{k+1} H_{k+1,k}. The
# eigenvalues of the k×k leading submatrix of H (Ritz values)
# approximate the dominant eigenvalues of A.

krylov_dim = 80  ## number of Arnoldi steps

@info @sprintf("Arnoldi iteration: k = %d, estimated wall time ≈ %.1f hours (at 2 min/pass)",
               krylov_dim, krylov_dim * 2 / 60)

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
        σ_ritz = sort(log.(abs.(λ_ritz)) / (3 * 86400) * 86400; rev=true)
        top5 = σ_ritz[1:min(5, length(σ_ritz))]
        @info @sprintf("  Top 5 Ritz values (day⁻¹): %s",
                       join([@sprintf("%.4f", s) for s in top5], ", "))
    end
end

# ## Extract eigenvalues

Hk = H[1:krylov_dim, 1:krylov_dim]
λ_all = eigvals(Hk)

## Growth rates and phase information
amplification = abs.(λ_all)
σ_all = log.(amplification) / (3 * 86400) * 86400  ## day⁻¹
phase_all = angle.(λ_all)  ## radians — nonzero means oscillatory

## Sort by growth rate (descending)
order = sortperm(σ_all; rev=true)
σ_sorted = σ_all[order]
λ_sorted = λ_all[order]
phase_sorted = phase_all[order]

@info "=== Top 20 eigenvalues ==="
for i in 1:min(20, length(σ_sorted))
    λi = λ_sorted[i]
    @info @sprintf("  %2d | σ = %+.4f day⁻¹ | |λ| = %.4f | phase = %+.4f rad | λ = %.4f %+.4fi",
                   i, σ_sorted[i], abs(λi), phase_sorted[i], real(λi), imag(λi))
end

# ## Save results

using JLD2
jldsave("arnoldi_eigenvalues.jld2";
        λ_all, σ_all, σ_sorted, λ_sorted, phase_sorted,
        H = Hk, krylov_dim,
        Nλ, Nφ, Nz, Δτ)

@info "Saved eigenvalues to arnoldi_eigenvalues.jld2"

# ## Visualization

# ### Eigenvalue spectrum in the complex plane

fig = Figure(size=(1400, 500))

ax1 = Axis(fig[1, 1];
           title = "Ritz values in the complex plane",
           xlabel = "Re(λ)", ylabel = "Im(λ)",
           aspect = DataAspect())

scatter!(ax1, real.(λ_sorted), imag.(λ_sorted);
         color = σ_sorted, colormap = :RdBu, markersize = 8)

## Unit circle for reference (|λ| = 1 means neutral)
θ_circle = range(0, 2π; length=200)
lines!(ax1, cos.(θ_circle), sin.(θ_circle);
       color = :gray60, linestyle = :dash, label = "|λ| = 1 (neutral)")
axislegend(ax1; position = :lb)

Colorbar(fig[1, 2]; colormap = :RdBu,
         limits = extrema(σ_sorted),
         label = "σ (day⁻¹)")

# ### Growth rate histogram

ax2 = Axis(fig[1, 3];
           title = "Growth rate spectrum",
           xlabel = "σ (day⁻¹)",
           ylabel = "Count")

hist!(ax2, σ_sorted; bins = 40, color = :dodgerblue)
vlines!(ax2, [0.46]; linestyle = :dash, color = :red, label = "Park et al.")
vlines!(ax2, [0.0]; linestyle = :solid, color = :gray60, label = "neutral")
axislegend(ax2; position = :rt)

save("arnoldi_eigenvalues.png", fig)
@info "Saved arnoldi_eigenvalues.png"

# ### Top eigenvalues bar chart

n_top = min(30, length(σ_sorted))

fig2 = Figure(size=(900, 500))
ax3 = Axis(fig2[1, 1];
           title = "Leading growth rates from Arnoldi (k = $krylov_dim)",
           xlabel = "Eigenvalue index",
           ylabel = "σ (day⁻¹)")

barplot!(ax3, 1:n_top, σ_sorted[1:n_top];
         color = [s > 0 ? :dodgerblue : :gray60 for s in σ_sorted[1:n_top]])
hlines!(ax3, [0.46]; linestyle = :dash, color = :red, linewidth = 2, label = "Park et al. (2013)")
hlines!(ax3, [0.0]; linestyle = :solid, color = :gray40)
axislegend(ax3; position = :rt)

save("arnoldi_top_eigenvalues.png", fig2)
@info "Saved arnoldi_top_eigenvalues.png"
nothing
