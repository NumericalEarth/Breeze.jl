#####
##### proof_test.jl
#####
##### Definitive A/B test of the proposed fix (discretely-balanced
##### reference state) without modifying Breeze source. We patch the
##### model's reference-state arrays after construction and re-run the
##### exact sweeps that fail with the original reference state.
#####
##### A "passes" the same way the working ω=0.7 case does in sweep J.
##### B (the proposed fix) passes at the *defaults* (ω=0.55) where the
##### original code fails. That's the proof.
#####

using CUDA
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode, Center, Face
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Breeze
using BreezyBaroclinicInstability
using Printf
using JLD2
using Statistics

include("sweep_runner.jl")  # build_rest_model, run_drift!, growth_per_step

const PROOF_OUT = @__DIR__
const PROOF_PATH = joinpath(PROOF_OUT, "proof_results.jld2")
const PROOF_CSV  = joinpath(PROOF_OUT, "proof_results.csv")

#####
##### Patch: build a discretely-balanced (p_ref, ρ_ref) and inject into the
##### model's reference state, replacing the trapezoidal-integration values.
#####
##### Recurrence (cf. sweep L in REPORT.md):
#####     p[k] = (p[k-1]/Δz_face - g·ρ[k-1]/2) / (1/Δz_face + g/(2 R T[k]))
#####     ρ[k] = p[k] / (R T[k])
##### with p[1] = p_surf, ρ[1] = p[1]/(R T[1]).
#####
##### For an isothermal-T₀ reference, T_func(z) = T₀ everywhere.
#####

function discretely_balanced_arrays(grid, T_func, p_surf;
                                    Rᵈ = 287.0, g = 9.80665)
    Nz = grid.Nz
    z_centers = [znode(1, 1, k, grid, Center(), Center(), Center()) for k in 1:Nz]
    z_faces   = [znode(1, 1, k, grid, Center(), Center(), Face())   for k in 1:Nz+1]

    p_arr = zeros(Float64, Nz)
    ρ_arr = zeros(Float64, Nz)

    Δz_face_1 = z_faces[2] - z_faces[1]
    p_arr[1] = p_surf
    ρ_arr[1] = p_arr[1] / (Rᵈ * T_func(z_centers[1]))

    for k in 2:Nz
        Δz_face = z_centers[k] - z_centers[k-1]   # exact center-to-center distance
        Tk = T_func(z_centers[k])
        a = 1 / Δz_face + g / (2 * Rᵈ * Tk)
        b = p_arr[k-1] / Δz_face - g * ρ_arr[k-1] / 2
        p_arr[k] = b / a
        ρ_arr[k] = p_arr[k] / (Rᵈ * Tk)
    end
    return p_arr, ρ_arr
end

"""
Compute discrete-balance correction (δp, δρ) by linearizing the EoS around
the existing reference state. Keeps ρθ unchanged so the model's actual EoS
(whatever it is) produces p_dbal = p_existing + δp from (ρ_existing + δρ).
"""
function dbal_perturbation(p_existing, ρ_existing, z_centers; g = 9.80665, κ = 287.0/1005.0)
    Nz = length(p_existing)
    # Existing residual ε[k] for k = 2..Nz at z-faces between centers k-1 and k:
    ε = zeros(Nz - 1)
    for k in 2:Nz
        Δz = z_centers[k] - z_centers[k-1]
        ε[k-1] = (p_existing[k] - p_existing[k-1]) / Δz +
                 g * (ρ_existing[k] + ρ_existing[k-1]) / 2
    end

    # Linearization: δρ = (1-κ) × ρ/p × δp (at fixed θ_li).
    # Discrete balance for the perturbed state:
    #   (δp[k] - δp[k-1])/Δz + g·(1-κ)/2 · (ρ[k]/p[k]·δp[k] + ρ[k-1]/p[k-1]·δp[k-1]) = -ε[k]
    # With δp[1] = 0, march upward:
    δp = zeros(Nz)
    for k in 2:Nz
        Δz = z_centers[k] - z_centers[k-1]
        a = 1/Δz + g * (1-κ) * ρ_existing[k] / (2 * p_existing[k])
        b = δp[k-1] * (1/Δz - g * (1-κ) * ρ_existing[k-1] / (2 * p_existing[k-1])) - ε[k-1]
        δp[k] = b / a
    end
    δρ = (1-κ) .* ρ_existing ./ p_existing .* δp

    return δp, δρ
end

function inject_dbal_reference!(model, θᵇᵍ; Rᵈ = 287.0, g = 9.80665, n_iter = 6)
    grid = model.grid
    ref  = model.dynamics.reference_state
    Nz   = grid.Nz
    FT   = eltype(grid)
    cpᵈ  = 1005.0
    κ    = Rᵈ / cpᵈ
    pˢᵗ  = ref.standard_pressure

    p_existing = Array(interior(ref.pressure))[1, 1, :]
    ρ_existing = Array(interior(ref.density))[1, 1, :]
    z_centers = [znode(1, 1, k, grid, Center(), Center(), Center()) for k in 1:Nz]

    # Iterate the linearized correction to nonlinear convergence (residual 0).
    p_curr = copy(p_existing)
    ρ_curr = copy(ρ_existing)
    for iter in 1:n_iter
        δp, δρ = dbal_perturbation(p_curr, ρ_curr, z_centers; g = g, κ = κ)
        p_curr .+= δp
        ρ_curr .+= δρ
        # Compute residual for diagnostics
        max_ε = 0.0
        for k in 2:Nz
            Δz = z_centers[k] - z_centers[k-1]
            ε = (p_curr[k] - p_curr[k-1])/Δz + g * (ρ_curr[k] + ρ_curr[k-1])/2
            max_ε = max(max_ε, abs(ε))
        end
        max_ε < 1e-12 && break
    end

    p_arr = p_curr
    ρ_arr = ρ_curr
    π_arr = (p_arr ./ pˢᵗ) .^ κ

    Oceananigans.set!(ref.pressure, FT.(p_arr))
    Oceananigans.set!(ref.density,  FT.(ρ_arr))
    if hasproperty(ref, :exner_function)
        Oceananigans.set!(ref.exner_function, FT.(π_arr))
    end
    fill_halo_regions!(ref.pressure)
    fill_halo_regions!(ref.density)
    hasproperty(ref, :exner_function) && fill_halo_regions!(ref.exner_function)

    return p_arr, ρ_arr, p_existing, ρ_existing
end

"""
Sync model state to the discretely-balanced reference. The EoS in this
formulation has p depending on ρθ (not on ρ separately), so we must set
ρθ such that p_eos ≈ p_dbal. Use Newton iteration with the local power
law p ∝ ρθ^(1/(1-κ)) to drive p_eos → p_dbal to machine ε.
"""
function sync_state_to_dbal!(model, p_arr, ρ_arr, p_existing, ρ_existing)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    FT = eltype(grid)
    Rᵈ = 287.0; cpᵈ = 1005.0; κ = Rᵈ / cpᵈ

    ρ_field = BreezyBaroclinicInstability.dynamics_density(model.dynamics)
    ρθ_field = model.formulation.potential_temperature_density

    # Build (Nx, Ny, Nz) broadcast helpers
    function bcast(arr1d)
        out = Array{FT}(undef, Nx, Ny, Nz)
        for k in 1:Nz
            out[:, :, k] .= FT(arr1d[k])
        end
        return out
    end

    ρθ_existing = Array(interior(ρθ_field))[1, 1, :]
    # Initial guess: scale ρθ by (p_dbal/p_existing)^(1-κ).
    ρθ_new = ρθ_existing .* (p_arr ./ p_existing) .^ (1 - κ)

    Oceananigans.set!(ρ_field, bcast(ρ_arr))
    Oceananigans.set!(ρθ_field, bcast(ρθ_new))
    Oceananigans.TimeSteppers.update_state!(model, compute_tendencies = false)

    # Newton iteration on ρθ to drive p_eos → p_dbal
    for iter in 1:6
        p_eos = Array(interior(model.dynamics.pressure))[1, 1, :]
        max_rel = maximum(abs, (p_eos .- p_arr) ./ p_arr)
        max_rel < 1e-12 && break
        ρθ_new = ρθ_new .* (p_arr ./ p_eos) .^ (1 - κ)
        Oceananigans.set!(ρθ_field, bcast(ρθ_new))
        Oceananigans.TimeSteppers.update_state!(model, compute_tendencies = false)
    end
    return nothing
end


#####
##### Discrete residual probe (uses the substepper's exact operators)
#####

function discrete_residual(model)
    grid = model.grid
    ref = model.dynamics.reference_state
    Nz = grid.Nz
    z_centers = [znode(1, 1, k, grid, Center(), Center(), Center()) for k in 1:Nz]

    p = Array(interior(ref.pressure))[1, 1, :]
    ρ = Array(interior(ref.density))[1, 1, :]

    g = 9.80665
    ε = zeros(Nz - 1)
    for k in 2:Nz
        Δz_face = z_centers[k] - z_centers[k-1]
        ε[k-1] = (p[k] - p[k-1]) / Δz_face + g * (ρ[k] + ρ[k-1]) / 2
    end
    return maximum(abs, ε), maximum(abs, g .* ρ)
end

#####
##### One-shot drift test, supporting either original or patched reference state
#####

function drift_test(; Δt, ω = 0.55, sim_time = 600.0,
                       Lz = 30e3, Nx = 32, Ny = 32, Nz = 64,
                       T₀ = 250.0, patch_reference = false,
                       topology = (Periodic, Periodic, Bounded))
    model, θᵇᵍ = build_rest_model(; Lz, Nx, Ny, Nz, T₀,
                                  topology,
                                  td_kwargs = (forward_weight = ω,))

    if patch_reference
        p_arr, ρ_arr, p_existing, ρ_existing = inject_dbal_reference!(model, θᵇᵍ)
        sync_state_to_dbal!(model, p_arr, ρ_arr, p_existing, ρ_existing)
    end

    res, scale = discrete_residual(model)

    n_steps = max(5, Int(round(sim_time / Δt)))
    sample = max(1, n_steps ÷ 30)
    iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = sample)

    return (Δt = Δt, ω = ω, patched = patch_reference,
            topology = string(topology),
            residual_abs = res,
            residual_relative = res / scale,
            wmax_envelope = maximum(wmax),
            growth_per_step = growth_per_step(iters, wmax),
            status = status,
            n_steps = n_steps)
end

#####
##### A/B sweeps to prove the fix
#####

function sweep_AB(; ωs = [0.51, 0.55, 0.60, 0.70],
                    Δts = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0],
                    Lz = 30e3, Nx = 16, Ny = 16, Nz = 64,
                    sim_time = 600.0, label = "AB")
    results = NamedTuple[]
    for patched in (false, true)
        for ω in ωs, Δt in Δts
            tag = patched ? "B_dbal" : "A_orig"
            @info "[$label/$tag] ω=$ω Δt=$(Δt)s"
            r = drift_test(; Δt, ω, Lz, Nx, Ny, Nz, sim_time, patch_reference = patched)
            push!(results, merge(r, (sweep = label, variant = tag)))
        end
    end
    return results
end

function sweep_AB_topology(; Δt = 20.0, ω = 0.55, sim_time = 600.0, label = "AB_topo")
    cases = [
        (name = "3D_PPB", grid_kw = (topology = (Periodic, Periodic, Bounded),)),
        (name = "3D_PBB", grid_kw = (topology = (Periodic, Bounded,  Bounded),)),
        (name = "latlon", grid_kw = (Nx = 90, Ny = 60, topology = :latlon)),
    ]
    results = NamedTuple[]
    for c in cases
        for patched in (false, true)
            tag = patched ? "B_dbal" : "A_orig"
            @info "[$label/$tag] $(c.name)"
            kw = (; c.grid_kw...)
            r = drift_test(; Δt, ω, sim_time, patch_reference = patched, kw...)
            push!(results, merge(r, (sweep = label, variant = tag, topology_name = c.name)))
        end
    end
    return results
end

#####
##### Main
#####

function main_proof()
    all = Dict{Symbol, Any}()

    @info "===== A/B over (Δt, ω) ====="
    all[:AB_dt_omega] = sweep_AB()
    jldsave(PROOF_PATH; all)

    @info "===== A/B over topology at Δt=20s ω=0.55 ====="
    all[:AB_topology] = sweep_AB_topology()
    jldsave(PROOF_PATH; all)

    # Flat CSV
    open(PROOF_CSV, "w") do io
        println(io, "sweep,variant,Δt,ω,topology,patched,residual_abs,residual_rel,wmax_env,growth_per_step,status,n_steps")
        for (sk, rs) in all
            for r in rs
                variant = get(r, :variant, "_")
                topo = get(r, :topology_name, r.topology)
                println(io, "$sk,$variant,$(r.Δt),$(r.ω),$topo,$(r.patched),$(r.residual_abs),$(r.residual_relative),$(r.wmax_envelope),$(r.growth_per_step),$(r.status),$(r.n_steps)")
            end
        end
    end

    return all
end

results = main_proof()
@info "Done. proof_results.jld2 + proof_results.csv in $PROOF_OUT"
