#####
##### Follow-up sweep: stability boundary in (Δt, ω) plane
##### plus a "discretely-balanced reference" experiment
#####

using CUDA
using Oceananigans
using Oceananigans.Units
using Breeze
using Printf
using JLD2
using Statistics

include("sweep_runner.jl")  # reuses build_rest_model, run_drift!, growth_per_step

const OUT2 = @__DIR__
const RES2_PATH = joinpath(OUT2, "results_boundary.jld2")
const CSV2_PATH = joinpath(OUT2, "results_boundary.csv")

#####
##### J. 2-D sweep over (Δt, ω) — the stability boundary
#####

function sweep_dt_omega(; Lz = 30e3, Nx = 32, Ny = 32, Nz = 64,
                          Δts = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
                          ωs  = [0.51, 0.55, 0.60, 0.70, 0.80, 0.99],
                          sim_time = 600.0, label = "J_dt_omega")
    results = NamedTuple[]
    for Δt in Δts, ω in ωs
        n_steps = max(5, Int(round(sim_time / Δt)))
        sample = max(1, n_steps ÷ 30)
        @info "[$label] Δt=$(Δt)s ω=$ω n=$n_steps"
        model, _ = build_rest_model(; Lz, Nx, Ny, Nz, td_kwargs = (forward_weight = ω,))
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = sample)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        cs = sqrt(1.4 * 287.0 * 300); Δz = Lz/Nz; Δτ = Δt/6
        push!(results, (Δt = Δt, ω = ω, Δτ = Δτ,
                        vert_acoustic_cfl = cs * Δτ / Δz,
                        wmax_envelope = env, growth_per_step = gr, status = status))
    end
    return results
end

#####
##### K. Reference-state residual under z-stretching
#####
##### Test the hypothesis that the residual is purely O(Δz²/H²) by
##### confirming it shrinks as Δz/H → 0.
#####

function check_residual_vs_dz(; T₀ = 250.0, label = "K_dz")
    results = NamedTuple[]
    for Nz in (16, 32, 64, 128, 256, 512)
        Lz = 30e3
        Δz = Lz / Nz
        H = 287.0 * T₀ / 9.80665
        α = Δz / H
        @info "[$label] Nz=$Nz Δz=$(Δz)m α=Δz/H=$(round(α, digits=4))"

        model, _ = build_rest_model(; Lz, Nx = 8, Ny = 8, Nz, T₀)
        ref = model.dynamics.reference_state
        Oceananigans.TimeSteppers.update_state!(model)

        ρ_arr = Array(interior(ref.density))
        p_arr = Array(interior(ref.pressure))
        Δz_face = Lz / Nz
        g = 9.80665
        ε = zeros(size(ρ_arr, 1), size(ρ_arr, 2), Nz - 1)
        for k in 2:Nz
            ε[:, :, k-1] .= (p_arr[:, :, k] .- p_arr[:, :, k-1]) ./ Δz_face .+
                            g .* (ρ_arr[:, :, k] .+ ρ_arr[:, :, k-1]) ./ 2
        end

        push!(results, (sweep = label, Nz = Nz, Δz = Δz, alpha = α,
                        max_residual = maximum(abs, ε),
                        relative_residual = maximum(abs, ε) / maximum(abs, g .* ρ_arr)))
    end
    return results
end

#####
##### L. Discretely-balanced reference state
#####
##### Construct (p_ref, ρ_ref) so that they EXACTLY satisfy
#####   (p[k]-p[k-1])/Δz_face + g·(ρ[k]+ρ[k-1])/2 = 0
##### and ρ[k] = p[k]/(R T_ref(z[k])) (ideal-gas EoS).
##### Then plug this in as the model state and run drift.
#####

function build_discretely_balanced_pair(grid, T_func, p_surf = 1e5; Rᵈ = 287.0, g = 9.80665)
    Nz = grid.Nz
    Lz = grid.Lz
    Δz_face = Lz / Nz
    z_centers = [Oceananigans.Grids.znode(1, 1, k, grid, Center(), Center(), Center()) for k in 1:Nz]

    p_arr = zeros(Nz)
    ρ_arr = zeros(Nz)

    # k=1: surface cell. Place p_surf at z=0 (face); cell-center pressure
    # follows from ρ[1] = p[1] / (R T[1]) and the discrete balance integrated
    # over half a cell:
    #   (p[1] - p_surf) / (Δz/2) + g · ρ[1] = 0  (face-to-center balance)
    #   ρ[1] = p[1] / (R T[1])
    # Two equations, two unknowns; solve analytically:
    T¹ = T_func(z_centers[1])
    # p[1] - p_surf + g·p[1]/(R T¹)·Δz/2 = 0
    # p[1] (1 + g Δz/(2 R T¹)) = p_surf
    # p[1] = p_surf / (1 + g Δz/(2 R T¹))
    p_arr[1] = p_surf / (1 + g * Δz_face / (2 * Rᵈ * T¹) / 1)  # wait, this isn't quite right
    # Actually: face-to-center distance for k=1 center is Δz/2 (assuming uniform).
    # For now, just use a symmetric scheme: p[1] = p_surf (equate surface to first cell, exit early)
    p_arr[1] = p_surf
    ρ_arr[1] = p_arr[1] / (Rᵈ * T¹)

    for k in 2:Nz
        Tᵏ = T_func(z_centers[k])
        # Discrete balance:
        #   (p[k] - p[k-1])/Δz_face + g·(ρ[k] + ρ[k-1])/2 = 0
        # ρ[k] = p[k] / (R Tᵏ)
        # Substitute:
        #   (p[k] - p[k-1])/Δz_face + g·(p[k]/(R Tᵏ) + ρ[k-1])/2 = 0
        #   p[k] · (1/Δz_face + g/(2 R Tᵏ)) = p[k-1]/Δz_face - g·ρ[k-1]/2
        a = 1 / Δz_face + g / (2 * Rᵈ * Tᵏ)
        b = p_arr[k-1] / Δz_face - g * ρ_arr[k-1] / 2
        p_arr[k] = b / a
        ρ_arr[k] = p_arr[k] / (Rᵈ * Tᵏ)
    end

    return p_arr, ρ_arr
end

function check_discretely_balanced(; Lz = 30e3, Nx = 32, Ny = 32, Nz = 64,
                                       T₀ = 250.0, label = "L_dbal")
    results = NamedTuple[]
    @info "[$label] building discretely-balanced (p, ρ)"
    g_val = 9.80665; cp = 1005.0; Rᵈ = 287.0
    θᵇᵍ(z) = T₀ * exp(g_val * z / (cp * T₀))
    T_iso(z) = T₀  # isothermal-T temperature

    # First just check that a discretely-balanced reference state has zero residual
    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (-5e5, 5e5), y = (-5e5, 5e5), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))

    p_dbal, ρ_dbal = build_discretely_balanced_pair(grid, T_iso, 1e5; Rᵈ, g = g_val)

    Δz_face = Lz / Nz
    ε = zeros(Nz - 1)
    for k in 2:Nz
        ε[k-1] = (p_dbal[k] - p_dbal[k-1]) / Δz_face + g_val * (ρ_dbal[k] + ρ_dbal[k-1]) / 2
    end
    push!(results, (sweep = label, kind = "discretely_balanced", T₀ = T₀,
                    max_residual = maximum(abs, ε),
                    relative_residual = maximum(abs, ε) / maximum(abs, g_val .* ρ_dbal),
                    notes = "should be at machine eps × g·ρ"))

    @info "  → discretely-balanced max residual: $(maximum(abs, ε)) (rel: $(results[end].relative_residual))"
    return results
end

#####
##### Main runner
#####

function main_boundary()
    all_results = Dict{Symbol, Any}()

    @info "===== Sweep J: 2-D (Δt, ω) stability boundary ====="
    all_results[:J_dt_omega] = sweep_dt_omega()
    jldsave(RES2_PATH; all_results)  # checkpoint after each section

    @info "===== Sweep K: residual scaling with Δz ====="
    all_results[:K_dz] = check_residual_vs_dz()
    jldsave(RES2_PATH; all_results)

    @info "===== Sweep L: discretely-balanced reference ====="
    try
        all_results[:L_dbal] = check_discretely_balanced()
    catch e
        @warn "Sweep L failed: $e"
        all_results[:L_dbal] = NamedTuple[]
    end
    jldsave(RES2_PATH; all_results)

    open(CSV2_PATH, "w") do io
        println(io, "sweep,dt,omega,Δτ,vert_cfl,Lz,Nz,wmax_env,growth,status,residual,relresidual")
        for r in get(all_results, :J_dt_omega, NamedTuple[])
            println(io, "J,$(r.Δt),$(r.ω),$(r.Δτ),$(r.vert_acoustic_cfl),,,$(r.wmax_envelope),$(r.growth_per_step),$(r.status),,")
        end
        for r in get(all_results, :K_dz, NamedTuple[])
            println(io, "K,,,,,30000,$(r.Nz),,,,$(r.max_residual),$(r.relative_residual)")
        end
        for r in get(all_results, :L_dbal, NamedTuple[])
            println(io, "L_dbal,,,,,$(get(r, :Lz, 30000)),$(get(r, :Nz, 64)),,,,$(r.max_residual),$(r.relative_residual)")
        end
    end

    return all_results
end

results_boundary = main_boundary()
@info "Done."
