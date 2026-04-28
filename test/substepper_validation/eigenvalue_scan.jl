#####
##### Eigenvalue scan of the substepper's implicit Schur tridiag plus
##### the predictor + post-solve recovery, as a 3Nz × 3Nz full-update
##### operator on (σ, η, μw). Pure Julia, no Breeze source modification.
#####
##### The substepper's documented formulas (acoustic_substepping.jl
##### lines 509-561, 719-806) are replicated here. The Π⁰_face,
##### θ⁰_face fields are read from a built model so the scan is on
##### the actual numerical reference state, not an analytic
##### approximation.
#####

using LinearAlgebra
using CUDA
using Oceananigans
using Oceananigans.Grids: znode, Center, Face
using Breeze
using BreezyBaroclinicInstability
using JLD2
using Printf

include("sweep_runner.jl")  # for build_rest_model

const arch = CUDA.functional() ? GPU() : CPU()

#####
##### Build the 3Nz × 3Nz substep update operator for one column.
#####
##### State vector layout: x = [σ[1..Nz]; η[1..Nz]; μw[1..Nz+1]] of length 3Nz+1.
##### (μw has Nz+1 face values; bottom and top are clamped to 0.)
#####

function build_column_update_operator(Π⁰_face, θ⁰_face, Δz_face, Δz_c, γRᵈ, g, δτ_new, Δτ)
    Nz = length(Δz_c)
    rdz_c = 1.0 ./ Δz_c

    # Total state size: σ has Nz, η has Nz, μw has Nz+1 (face).
    # Bottom face μw[1] and top face μw[Nz+1] are 0 always; we keep them in
    # the state for index alignment but constrain them by zeroing the
    # corresponding rows/columns in the operators below.
    Sσ = 1:Nz
    Sη = (Nz+1):(2*Nz)
    Sμ = (2*Nz+1):(3*Nz+1)
    N = 3*Nz + 1

    # Predictor matrix P: (σ_pred, η_pred, μw_rhs) = P × (σ, η, μw)
    #
    # σ_pred[k] = σ[k] - (δτ/Δz_c[k])(μw[k+1] - μw[k])
    # η_pred[k] = η[k] - (δτ/Δz_c[k])(θ⁰_face[k+1] μw[k+1] - θ⁰_face[k] μw[k])
    # μw_rhs[k] = μw[k] - δτ × γRᵈ Π⁰_face[k]/Δz_face[k] × (∂z_η_old + ∂z_η_pred)
    #                  - g × δτ × (σ_face_old + σ_face_pred)
    # (we'll handle μw_rhs after by combining old + pred parts)
    P = zeros(N, N)

    # σ_pred rows
    for k in 1:Nz
        ksig = Sσ[k]
        P[ksig, ksig] = 1.0
        P[ksig, Sμ[k+1]] += -δτ_new * rdz_c[k]
        P[ksig, Sμ[k]]   += +δτ_new * rdz_c[k]
    end

    # η_pred rows
    for k in 1:Nz
        keta = Sη[k]
        P[keta, keta] = 1.0
        P[keta, Sμ[k+1]] += -δτ_new * rdz_c[k] * θ⁰_face[k+1]
        P[keta, Sμ[k]]   += +δτ_new * rdz_c[k] * θ⁰_face[k]
    end

    # μw_rhs rows: μw_rhs[k_face] = μw[k_face] - sound_force - buoy_force
    # sound_force = γRᵈ × Π⁰_face[k]/Δz_face[k] × δτ × (∂z_η_old + ∂z_η_pred)
    # ∂z_η_old = η[k] - η[k-1]
    # ∂z_η_pred = η_pred[k] - η_pred[k-1]
    # buoy_force = g × δτ × ((σ[k]+σ[k-1])/2 + (σ_pred[k]+σ_pred[k-1])/2)
    # We'll use σ_pred = P * (σ-part), η_pred = P * (η-part), so we substitute.

    # Build the (Nz+1) × (Nz+1) implicit-solve matrix A_μ (acts on μw_new only):
    Nμ = Nz + 1
    A_μ = zeros(Nμ, Nμ)
    A_μ[1, 1] = 1.0
    A_μ[Nμ, Nμ] = 1.0
    for k in 2:Nz
        Δz_face_k = Δz_face[k]
        Π_face = Π⁰_face[k]

        rdz_above = rdz_c[k]
        rdz_below = rdz_c[k-1]

        pgf_diag = δτ_new^2 * γRᵈ * Π_face * θ⁰_face[k] * (rdz_above + rdz_below) / Δz_face_k
        buoy_diag = δτ_new^2 * g * (rdz_above - rdz_below) / 2
        A_μ[k, k] = 1.0 + pgf_diag + buoy_diag

        pgf_lower = -δτ_new^2 * γRᵈ * Π_face * θ⁰_face[k-1] * rdz_below / Δz_face_k
        buoy_lower = δτ_new^2 * g * rdz_below / 2
        A_μ[k, k-1] = pgf_lower + buoy_lower

        pgf_upper = -δτ_new^2 * γRᵈ * Π_face * θ⁰_face[k+1] * rdz_above / Δz_face_k
        buoy_upper = -δτ_new^2 * g * rdz_above / 2
        A_μ[k, k+1] = pgf_upper + buoy_upper
    end

    # Build B_μ: μw_rhs = B_μ × full_state (size Nμ × N)
    B_μ = zeros(Nμ, N)
    for k in 2:Nz
        Δz_face_k = Δz_face[k]
        Π_face = Π⁰_face[k]

        coef_η_old = δτ_new * γRᵈ * Π_face / Δz_face_k

        # ∂z_η_old contribution
        B_μ[k, Sη[k]]   += -coef_η_old
        B_μ[k, Sη[k-1]] += +coef_η_old
        # ∂z_η_pred contribution — substitute via P
        for j in 1:N
            B_μ[k, j] += -coef_η_old * (P[Sη[k], j] - P[Sη[k-1], j])
        end

        coef_σ = g * δτ_new / 2
        # σ_face_old
        B_μ[k, Sσ[k]]   += -coef_σ
        B_μ[k, Sσ[k-1]] += -coef_σ
        # σ_face_pred via P
        for j in 1:N
            B_μ[k, j] += -coef_σ * (P[Sσ[k], j] + P[Sσ[k-1], j])
        end

        # μw[k] carryover
        B_μ[k, Sμ[k]] += 1.0
    end

    # Solve μw_new = A_μ \ B_μ × state. Result is (Nz+1) × N.
    M = A_μ \ B_μ  # μw_new[k_face] = sum_j M[k_face, j] × state[j]

    # Lift M into the full state index space: Ainv_B is N × N with the σ and η rows
    # zero (we'll overwrite them in U later).
    Ainv_B = zeros(N, N)
    for k in 1:Nμ
        Ainv_B[Sμ[k], :] = M[k, :]
    end

    # Build the full update operator U: state_new = U × state
    U = zeros(N, N)

    # μw_new rows
    U[Sμ[1], :] = Ainv_B[Sμ[1], :]
    for k in 2:Nz+1
        U[Sμ[k], :] = Ainv_B[Sμ[k], :]
    end

    # σ_new rows: σ_new = σ_pred - (δτ/Δz_c)(μw_new_above - μw_new_here)
    for k in 1:Nz
        kσ = Sσ[k]
        # σ_pred contribution from P
        U[kσ, :] = P[kσ, :] .- δτ_new * rdz_c[k] .* (Ainv_B[Sμ[k+1], :] - Ainv_B[Sμ[k], :])
    end

    # η_new rows: η_new = η_pred - (δτ/Δz_c)(θ⁰_above × μw_new_above - θ⁰_here × μw_new_here)
    for k in 1:Nz
        kη = Sη[k]
        U[kη, :] = P[kη, :] .- δτ_new * rdz_c[k] .* (θ⁰_face[k+1] .* Ainv_B[Sμ[k+1], :] .-
                                                      θ⁰_face[k]   .* Ainv_B[Sμ[k],   :])
    end

    return U
end

"""Read Π⁰_face and θ⁰_face from a model that's been put at rest +
update_state'd, returning the per-column arrays the substepper would use.
"""
function extract_column_inputs(model)
    grid = model.grid
    Nz = grid.Nz
    ref = model.dynamics.reference_state
    constants = model.thermodynamic_constants
    Rᵈ = Breeze.dry_air_gas_constant(constants)
    cpᵈ = constants.dry_air.heat_capacity
    γ = cpᵈ / (cpᵈ - Rᵈ)
    g_val = Float64(constants.gravitational_acceleration)
    pˢᵗ = ref.standard_pressure
    κ = Rᵈ / cpᵈ

    # Read the 1D z-profiles
    p_arr = Array(interior(ref.pressure))[1, 1, :]
    ρ_arr = Array(interior(ref.density))[1, 1, :]

    # θ⁰[k] = p[k]/(R ρ[k]) × (p_std/p[k])^κ
    T_arr = p_arr ./ (Rᵈ .* ρ_arr)
    θ⁰_center = T_arr .* (pˢᵗ ./ p_arr) .^ κ

    # Π⁰_face[k] for k=1..Nz+1 — interpolate from centers to faces
    # Match the substepper's ℑzᵃᵃᶠ: simple arithmetic mean.
    # Boundary faces (k=1 and Nz+1) take the boundary cell-center value.
    Π_center = (p_arr ./ pˢᵗ) .^ κ
    Π_face = zeros(Nz + 1)
    Π_face[1] = Π_center[1]
    Π_face[Nz+1] = Π_center[Nz]
    for k in 2:Nz
        Π_face[k] = (Π_center[k] + Π_center[k-1]) / 2
    end

    # θ⁰_face[k]: substepper's _theta_at_face uses average of
    # θ⁰_center[k_safe] and θ⁰_center[k_safe-1] for k=2..Nz, returns 0 at boundary faces.
    θ⁰_face = zeros(Nz + 1)
    for k in 2:Nz
        θ⁰_face[k] = (θ⁰_center[k] + θ⁰_center[k-1]) / 2
    end
    # Boundary faces are zero per the substepper's convention.

    # Δz_c (Nz of them, one per cell-center) and Δz_face (Nz+1 of them, one per face)
    z_centers = [znode(1, 1, k, grid, Center(), Center(), Center()) for k in 1:Nz]
    z_faces = [znode(1, 1, k, grid, Center(), Center(), Face()) for k in 1:Nz+1]
    Δz_c = [k == Nz ? z_faces[Nz+1] - z_centers[Nz] :
            (k == 1 ? z_centers[1] - z_faces[1] :
            z_centers[k+1] - z_centers[k])
            for k in 1:Nz]
    # Actually the substepper uses Δzᶜᶜᶜ which on uniform grid = Lz/Nz.
    # Use this consistent definition:
    Lz = grid.Lz
    Δz_uniform = Lz / Nz
    Δz_c .= Δz_uniform
    Δz_face = fill(Δz_uniform, Nz + 1)

    return (Π⁰_face = Π_face, θ⁰_face = θ⁰_face,
            Δz_face = Δz_face, Δz_c = Δz_c,
            γRᵈ = γ * Rᵈ, g = g_val, Nz = Nz)
end

#####
##### Sweep
#####

function eigenvalue_sweep(; Lz = 30e3, Nx = 8, Ny = 8, Nz = 64, T₀ = 250.0,
                            ωs = (0.50, 0.51, 0.55, 0.6, 0.7, 0.8, 0.99),
                            Δts = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
                            label = "spectrum")
    model, _ = build_rest_model(; Lz, Nx, Ny, Nz, T₀,
                                  td_kwargs = (forward_weight = 0.55,))
    inputs = extract_column_inputs(model)

    results = NamedTuple[]
    for ω in ωs, Δt in Δts
        Δτ = Δt / 6
        δτ_new = ω * Δτ
        U = build_column_update_operator(inputs.Π⁰_face, inputs.θ⁰_face,
                                          inputs.Δz_face, inputs.Δz_c,
                                          inputs.γRᵈ, inputs.g, δτ_new, Δτ)
        spec = eigvals(U)
        ρ_spec = maximum(abs, spec)
        complex_count = count(λ -> abs(imag(λ)) > 1e-12 * abs(λ), spec)
        # Spectral norm (largest singular value) → transient amplification per substep
        op2 = opnorm(U)
        # Power iteration for the multi-step asymptotic norm: take ‖U^N_substeps‖
        # to compare to empirical growth/outer step.
        N_substeps = max(6, 6 * cld(max(1, ceil(Int, 2 * Δt * 348 / 1e6 * 32)), 6))
        # For our test, N_substeps is always 6 here.
        UN = U^N_substeps
        opN = opnorm(UN)
        push!(results, (ω = ω, Δt = Δt, Δτ = Δτ, δτ_new = δτ_new,
                        spectral_radius = ρ_spec,
                        op_norm = op2,
                        op_norm_N = opN,
                        N_substeps = N_substeps,
                        n_complex = complex_count,
                        max_real = maximum(real, spec),
                        min_real = minimum(real, spec)))
    end
    return results
end

results = eigenvalue_sweep()

println("\n=== Spectral radius ρ(U) ===")
println(rpad("Δt", 10), join([rpad("ω=$ω", 12) for ω in (0.50, 0.51, 0.55, 0.6, 0.7, 0.8, 0.99)]))
for Δt in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    print(rpad("Δt=$(Δt)s", 10))
    for ω in (0.50, 0.51, 0.55, 0.6, 0.7, 0.8, 0.99)
        r = filter(x -> x.ω == ω && x.Δt == Δt, results)
        if isempty(r); print("    -       "); continue; end
        print(rpad(@sprintf("%.6f", r[1].spectral_radius), 12))
    end
    println()
end

println("\n=== Spectral norm ‖U‖₂ (single-substep transient amplification) ===")
println(rpad("Δt", 10), join([rpad("ω=$ω", 12) for ω in (0.50, 0.51, 0.55, 0.6, 0.7, 0.8, 0.99)]))
for Δt in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    print(rpad("Δt=$(Δt)s", 10))
    for ω in (0.50, 0.51, 0.55, 0.6, 0.7, 0.8, 0.99)
        r = filter(x -> x.ω == ω && x.Δt == Δt, results)
        if isempty(r); print("    -       "); continue; end
        print(rpad(@sprintf("%.4f", r[1].op_norm), 12))
    end
    println()
end

println("\n=== ‖U^N_substeps‖₂ (one outer step) ===")
println(rpad("Δt", 10), join([rpad("ω=$ω", 12) for ω in (0.50, 0.51, 0.55, 0.6, 0.7, 0.8, 0.99)]))
for Δt in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    print(rpad("Δt=$(Δt)s", 10))
    for ω in (0.50, 0.51, 0.55, 0.6, 0.7, 0.8, 0.99)
        r = filter(x -> x.ω == ω && x.Δt == Δt, results)
        if isempty(r); print("    -       "); continue; end
        print(rpad(@sprintf("%.3e", r[1].op_norm_N), 12))
    end
    println()
end

# Save
jldsave(joinpath(@__DIR__, "eigenvalue_results.jld2"); results)
@info "Saved eigenvalue_results.jld2"
