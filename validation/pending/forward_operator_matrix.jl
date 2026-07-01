# # Explicit matrix of the Breeze single-timestep operator
#
# Constructs the **full N×N matrix** of a single explicit time step by
# probing with unit vectors e_i. One time step IS the forward operator —
# the 3-day power method integration is just this matrix raised to a
# high power.
#
# On a coarse 20×10×4 grid (N = 4000) each probe is nearly instant after
# JIT warmup, so the full matrix builds in minutes.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: time_step!, update_state!, reset!
using Printf
using CairoMakie
using LinearAlgebra
using JLD2
using Statistics

# ## Coarse grid configuration

Nλ = 20; Nφ = 10; Nz = 4

# ## DCMIP2016 parameters

Oceananigans.defaults.FloatType = Float64
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

# ## Grid — coarse, on CPU for small grids

H = 30kilometers

grid = LatitudeLongitudeGrid(CPU();
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

# ## State packing utilities

n_per_field = Nλ * Nφ * Nz
field_names = [Symbol(k) for k in keys(prognostic_fields(model))]
n_fields = length(prognostic_fields(model))
N = n_fields * n_per_field

iλ = grid.Hx .+ (1:Nλ)
iφ = grid.Hy .+ (1:Nφ)
iz = grid.Hz .+ (1:Nz)

@info @sprintf("Grid: %d × %d × %d | Fields: %d | DOF: N = %d", Nλ, Nφ, Nz, n_fields, N)
@info @sprintf("Matrix size: %d × %d = %.1f M entries (%.1f MB in Float64)",
               N, N, N^2 / 1e6, N^2 * 8 / 1e6)

function pack_perturbation(model, background)
    x = Vector{Float64}(undef, N)
    offset = 0
    for (f, bg) in zip(prognostic_fields(model), background)
        chunk = parent(f)[iλ, iφ, iz] .- bg[iλ, iφ, iz]
        x[offset+1:offset+n_per_field] .= vec(Float64.(chunk))
        offset += n_per_field
    end
    return x
end

function unpack_perturbation!(model, x, background)
    offset = 0
    for (f, bg) in zip(prognostic_fields(model), background)
        chunk = reshape(x[offset+1:offset+n_per_field], Nλ, Nφ, Nz)
        parent(f)[iλ, iφ, iz] .= bg[iλ, iφ, iz] .+ chunk
        offset += n_per_field
    end
    for f in prognostic_fields(model)
        fill_halo_regions!(f)
    end
end

# ## Single time step as the forward operator
#
# One explicit time step maps state → state. We probe it with
# background + ε * e_i, take one step, and read off column i of the
# Jacobian as (result - stepped_background) / ε.
#
# ε must be small enough for linearity but large enough to stay above
# Float32 noise. 1e-3 is O(1e-3) perturbation vs O(1–300) background.

Δt = 12 * 60  ## 12 minutes in seconds (matching the production runs)

ε = 1e-5

## First, step the unperturbed background to get the reference output.
## The operator is L(x) = step(x), and we want the Jacobian dL/dx,
## so column i = (L(bg + ε eᵢ) - L(bg)) / ε.
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
@info @sprintf("Warmup/reference step complete (%.2f s)", t_warmup)

function apply_single_step(x)
    ## Reset to background + perturbation
    reset!(model.clock)
    unpack_perturbation!(model, x, background)
    update_state!(model, compute_tendencies=false)

    ## One time step
    time_step!(model, Δt)

    ## Extract output perturbation relative to stepped background
    y = Vector{Float64}(undef, N)
    offset = 0
    for (f, bg_s) in zip(prognostic_fields(model), bg_stepped)
        chunk = parent(f)[iλ, iφ, iz] .- bg_s[iλ, iφ, iz]
        y[offset+1:offset+n_per_field] .= vec(Float64.(chunk))
        offset += n_per_field
    end
    return y
end

# ## Verification: compare A*x to actual time step
#
# Use a random perturbation and check that the matrix reproduces the
# actual time stepper output. This catches bugs in pack/unpack or
# nonlinearity contamination.

@info "Verifying matrix construction with a random probe..."
using Random
Random.seed!(42)
x_rand = randn(N) * ε
y_actual = apply_single_step(x_rand) / ε  ## still unused — need matrix first

# ## Construct the matrix column by column

A = zeros(Float64, N, N)

## Time a single probe (post-JIT)
e_test = zeros(N); e_test[1] = ε
t_test = @elapsed (A[:, 1] = apply_single_step(e_test) / ε)
@info @sprintf("Per-probe time: %.4f s | Estimated total: %.1f s (%.1f min) for %d columns",
               t_test, N * t_test, N * t_test / 60, N)

t_start = time()
for i in 2:N
    e_i = zeros(N)
    e_i[i] = ε
    A[:, i] = apply_single_step(e_i) / ε

    if i % 500 == 0 || i == N
        elapsed = time() - t_start
        rate = (i - 1) / elapsed
        remaining = (N - i) / rate
        @info @sprintf("Column %d / %d (%.0f%%) | %.4f s/probe | ETA: %.0f s",
                       i, N, 100 * i / N, 1 / rate, remaining)
    end
end

total_time = time() - t_start
@info @sprintf("Matrix construction complete: %.1f seconds (%.1f min)", total_time, total_time / 60)

# ## Save the matrix

jldsave("forward_operator_matrix.jld2";
        A, Nλ, Nφ, Nz, n_fields, N, ε, Δt,
        field_names = string.(field_names))

@info "Saved to forward_operator_matrix.jld2"

# ## Verify: A * x_rand ≈ actual time step output

y_matrix = A * (x_rand / ε)  ## matrix prediction
rel_err = norm(y_matrix - y_actual) / norm(y_actual)
@info @sprintf("Verification: ||A*x - y_actual|| / ||y_actual|| = %.2e", rel_err)
if rel_err > 0.01
    @warn "Matrix may not be accurate — relative error > 1%"
end

# ## Eigenvalue decomposition

λ_all = eigvals(A)

## For a single time step, |λ| > 1 means amplifying, |λ| < 1 means damping.
## Growth rate per day: σ = ln|λ| / Δt * 86400
σ_all = log.(abs.(λ_all)) / Δt * 86400  ## day⁻¹

order = sortperm(abs.(λ_all); rev=true)
λ_sorted = λ_all[order]
σ_sorted = σ_all[order]

## Spectral radius — if > 1, the scheme amplifies something
ρ_A = maximum(abs, λ_all)
@info @sprintf("Spectral radius ρ(A) = %.6f  (> 1 means unstable!)", ρ_A)
@info @sprintf("Equivalent growth rate of dominant mode: %.4f day⁻¹", log(ρ_A) / Δt * 86400)

@info "=== Top 20 eigenvalues (sorted by |λ|) ==="
for i in 1:min(20, length(λ_sorted))
    λi = λ_sorted[i]
    @info @sprintf("  %2d | |λ| = %.6f | σ = %+.4f day⁻¹ | λ = %.4f %+.4fi",
                   i, abs(λi), σ_sorted[i], real(λi), imag(λi))
end

@info "=== Bottom 5 eigenvalues (most damped) ==="
for i in N:-1:max(1, N-4)
    λi = λ_sorted[i]
    @info @sprintf("  %d | |λ| = %.6f | σ = %+.4f day⁻¹", i, abs(λi), σ_sorted[i])
end

# ## Visualization

using SparseArrays

# ### spy(A) — the sparsity structure

A_sparse = sparse(abs.(A) .> 1e-10 * maximum(abs, A))

fig = Figure(size=(900, 800))
ax1 = Axis(fig[1, 1]; title="spy(A) — single time step operator ($(Nλ)×$(Nφ)×$(Nz), N=$N)",
           xlabel="Column (input DOF)", ylabel="Row (output DOF)",
           yreversed=true, aspect=DataAspect())

spy!(ax1, A_sparse; markersize=1)

## Block boundaries: mark where each field starts
for k in 1:n_fields
    boundary = (k - 1) * n_per_field + 0.5
    hlines!(ax1, [boundary]; color=:red, linewidth=0.5, linestyle=:dash)
    vlines!(ax1, [boundary]; color=:red, linewidth=0.5, linestyle=:dash)
end

## Label field blocks
for k in 1:n_fields
    mid = (k - 0.5) * n_per_field
    text!(ax1, -N * 0.02, mid; text=string(field_names[k]),
          fontsize=10, align=(:right, :center), color=:red)
end

save("forward_operator_spy.png", fig)
@info "Saved forward_operator_spy.png"

# ### Matrix heatmap (log scale)

fig2 = Figure(size=(900, 800))
ax2 = Axis(fig2[1, 1]; title="log₁₀|A| — forward operator matrix",
           xlabel="Column (input DOF)", ylabel="Row (output DOF)",
           yreversed=true, aspect=DataAspect())

A_log = log10.(abs.(A) .+ 1e-20)
hm = heatmap!(ax2, A_log; colormap=:inferno)
Colorbar(fig2[1, 2], hm; label="log₁₀|Aᵢⱼ|")

for k in 1:n_fields
    boundary = (k - 1) * n_per_field + 0.5
    hlines!(ax2, [boundary]; color=:white, linewidth=0.5)
    vlines!(ax2, [boundary]; color=:white, linewidth=0.5)
end

save("forward_operator_matrix.png", fig2)
@info "Saved forward_operator_matrix.png"

# ### Eigenvalue spectrum

## Filter out zero eigenvalues for clean plotting
nonzero = abs.(λ_sorted) .> 1e-15
λ_nz = λ_sorted[nonzero]
σ_nz = σ_sorted[nonzero]

fig3 = Figure(size=(1400, 500))

## Complex plane
ax3 = Axis(fig3[1, 1]; title="Eigenvalues in the complex plane",
           xlabel="Re(λ)", ylabel="Im(λ)")

scatter!(ax3, real.(λ_nz), imag.(λ_nz);
         color=abs.(λ_nz), colormap=:RdBu, markersize=6)

θ_circle = range(0, 2π; length=200)
lines!(ax3, cos.(θ_circle), sin.(θ_circle);
       color=:gray40, linestyle=:dash, linewidth=2, label="|λ| = 1 (neutral)")
axislegend(ax3; position=:lb)

Colorbar(fig3[1, 2]; colormap=:RdBu,
         limits=extrema(abs.(λ_nz)),
         label="|λ|")

## |λ| spectrum
ax4 = Axis(fig3[1, 3]; title="|λ| spectrum (single Δt = $(Δt÷60) min)",
           xlabel="Eigenvalue index (sorted by |λ|)",
           ylabel="|λ|")

n_show = min(100, length(λ_nz))
scatter!(ax4, 1:n_show, abs.(λ_nz[1:n_show]); markersize=4, color=:dodgerblue)
hlines!(ax4, [1.0]; color=:red, linewidth=2, linestyle=:dash, label="|λ| = 1")
axislegend(ax4; position=:rt)

save("forward_operator_eigenvalues.png", fig3)
@info "Saved forward_operator_eigenvalues.png"

# ### Field coupling matrix

coupling = zeros(n_fields, n_fields)
for i in 1:n_fields, j in 1:n_fields
    rows = (i-1)*n_per_field+1 : i*n_per_field
    cols = (j-1)*n_per_field+1 : j*n_per_field
    coupling[i, j] = mean(abs.(A[rows, cols]))
end

fig4 = Figure(size=(550, 450))
ax5 = Axis(fig4[1, 1]; title="Field coupling strength (single step)",
           xticks=(1:n_fields, string.(field_names)),
           yticks=(1:n_fields, string.(field_names)),
           xlabel="Input field", ylabel="Output field",
           yreversed=true, aspect=DataAspect(),
           xticklabelrotation=π/4)

hm5 = heatmap!(ax5, log10.(coupling .+ 1e-20); colormap=:viridis)
Colorbar(fig4[1, 2], hm5; label="log₁₀(mean |Aᵢⱼ|)")

for i in 1:n_fields, j in 1:n_fields
    text!(ax5, j, i; text=@sprintf("%.1e", coupling[i, j]),
          align=(:center, :center), fontsize=9, color=:white)
end

save("forward_operator_coupling.png", fig4)
@info "Saved forward_operator_coupling.png"
nothing
