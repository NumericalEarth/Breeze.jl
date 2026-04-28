#####
##### Comprehensive substepper diagnostic sweep
#####
##### Runs many parametric sweeps of the rest-atmosphere drift test in a
##### single Julia session to amortize precompilation. Writes:
#####   results.jld2  — every trajectory
#####   results.csv   — flat summary
#####   REPORT.md     — human-readable findings
#####
##### Run from the Breeze project: julia --project=. test/substepper_validation/sweep_runner.jl
#####

using CUDA
using Oceananigans
using Oceananigans.Units
using Breeze
using Printf
using JLD2
using Statistics

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = @__DIR__
const RESULTS_PATH = joinpath(OUTDIR, "results.jld2")
const CSV_PATH = joinpath(OUTDIR, "results.csv")

"""
    build_rest_model(; FT, Lx, Ly, Lz, Nx, Ny, Nz, topology, T₀, N², ref_form,
                       td_kwargs...)

Construct a CompressibleDynamics AtmosphereModel at rest = reference state.

`ref_form` is `:isothermal_T0` (θ_ref = T₀ exp(g z/(cp T₀))) or `:strat_N2`
(θ_ref = T₀ exp(N² z / g)).
"""
function build_rest_model(; FT = Float64, Lx = 1e6, Ly = 1e6, Lz = 30e3,
                            Nx = 32, Ny = 32, Nz = 64,
                            topology = (Periodic, Periodic, Bounded),
                            T₀ = 250.0, N² = 1e-4, ref_form = :isothermal_T0,
                            td_kwargs = NamedTuple())

    Oceananigans.defaults.FloatType = FT
    Oceananigans.defaults.gravitational_acceleration = 9.80665

    g = 9.80665; cp = 1005.0

    is_2D = topology !== :latlon && length(topology) == 3 && topology[2] === Flat
    is_latlon = topology === :latlon

    grid = if is_2D
        RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                        x = (-Lx/2, Lx/2), z = (0, Lz),
                        topology = (Periodic, Flat, Bounded))
    elseif is_latlon
        LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                              longitude = (0, 360),
                              latitude = (-80, 80),
                              z = (0, Lz))
    else
        RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                        x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (0, Lz),
                        topology = topology)
    end

    θᵇᵍ = if ref_form === :isothermal_T0
        z -> T₀ * exp(g * z / (cp * T₀))
    elseif ref_form === :strat_N2
        z -> T₀ * exp(N² * z / g)
    else
        error("unknown ref_form $ref_form")
    end

    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; td_kwargs...)
    dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)

    model = AtmosphereModel(grid; dynamics = dyn,
                            advection = WENO(order = 5),
                            thermodynamic_constants = constants,
                            timestepper = :AcousticRungeKutta3)

    ref = model.dynamics.reference_state

    # Topology-appropriate θ function for set!
    θᵢ = if is_2D
        (x, z) -> θᵇᵍ(z)
    else
        (x_or_λ, y_or_φ, z) -> θᵇᵍ(z)
    end

    set!(model; θ = θᵢ, ρ = ref.density)
    return model, θᵇᵍ
end

"""
    run_drift!(model; Δt, n_steps, sample_every = 1)

Run the model at rest for `n_steps` outer steps, returning vectors of
(iteration, max|w|).
"""
function run_drift!(model; Δt, n_steps, sample_every = 1)
    iters = Int[]
    wmax = Float64[]
    push!(iters, 0)
    push!(wmax, Float64(maximum(abs, interior(model.velocities.w))))
    for n in 1:n_steps
        try
            Oceananigans.TimeSteppers.time_step!(model, Δt)
        catch e
            push!(iters, n)
            push!(wmax, NaN)
            return iters, wmax, :crashed
        end
        if any(isnan, parent(model.velocities.w))
            push!(iters, n)
            push!(wmax, NaN)
            return iters, wmax, :nan
        end
        if n % sample_every == 0 || n == n_steps
            push!(iters, n)
            push!(wmax, Float64(maximum(abs, interior(model.velocities.w))))
        end
    end
    return iters, wmax, :ok
end

"""Estimate per-outer-step growth factor by linear regression of log(wmax)
against iteration over the early growth phase (skipping initial transient
and any saturation / crash)."""
function growth_per_step(iters, wmax)
    valid = filter(i -> wmax[i] > 0 && wmax[i] < 1.0 && !isnan(wmax[i]), eachindex(wmax))
    length(valid) < 6 && return NaN
    # Skip the first 2 samples (initial transient) and last 2 (saturation/crash)
    keep = valid[3:end-2]
    length(keep) < 4 && return NaN
    x = Float64.(iters[keep])
    y = log.(wmax[keep])
    # Linear regression slope
    x̄ = mean(x); ȳ = mean(y)
    num = sum((x .- x̄) .* (y .- ȳ))
    den = sum((x .- x̄).^2)
    slope = num / den
    return exp(slope)
end

#####
##### A. Δt sweep on 3D Cartesian rest atmosphere
#####

function sweep_dt(; FT = Float64, Lz = 30e3, Nx = 32, Ny = 32, Nz = 64,
                    Δts = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
                    sim_time = 600.0, label = "A_dt")
    results = NamedTuple[]
    for Δt in Δts
        n_steps = Int(round(sim_time / Δt))
        sample = max(1, n_steps ÷ 30)
        @info "[$label] Δt=$(Δt)s  n=$n_steps"
        model, _ = build_rest_model(; FT, Lz, Nx, Ny, Nz)
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = sample)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        cs = sqrt(1.4 * 287.0 * 300)
        Δz = Lz / Nz
        Δτ = Δt / 6  # default minimum substep count
        cfl_v = cs * Δτ / Δz
        push!(results, (sweep = label, FT = string(FT), Δt = Δt, Lz = Lz, Nz = Nz,
                        Δτ = Δτ, vert_acoustic_cfl = cfl_v,
                        wmax_envelope = env, growth_per_step = gr, status = status,
                        iters = iters, wmax = wmax))
    end
    return results
end

#####
##### B. Substep count N at fixed Δt=20s
#####

function sweep_substep_count(; Lz = 30e3, Nx = 32, Ny = 32, Nz = 64,
                                Ns = [6, 12, 24, 48, 96, 192],
                                Δt = 20.0, sim_time = 600.0, label = "B_N")
    results = NamedTuple[]
    for N in Ns
        n_steps = Int(round(sim_time / Δt))
        @info "[$label] N=$N"
        model, _ = build_rest_model(; Lz, Nx, Ny, Nz, td_kwargs = (substeps = N,))
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = 1)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        cs = sqrt(1.4 * 287.0 * 300)
        Δτ = Δt / N
        cfl_v = cs * Δτ / (Lz/Nz)
        push!(results, (sweep = label, N_substeps = N, Δt = Δt,
                        Δτ = Δτ, vert_acoustic_cfl = cfl_v,
                        wmax_envelope = env, growth_per_step = gr, status = status,
                        iters = iters, wmax = wmax))
    end
    return results
end

#####
##### C. ω (forward_weight) sweep at fixed Δt=20s — diagnostic only
#####

function sweep_omega(; Lz = 30e3, Nx = 32, Ny = 32, Nz = 64,
                       ωs = [0.5, 0.51, 0.55, 0.6, 0.7, 0.9, 0.99],
                       Δt = 20.0, sim_time = 600.0, label = "C_omega")
    results = NamedTuple[]
    for ω in ωs
        n_steps = Int(round(sim_time / Δt))
        @info "[$label] ω=$ω"
        model, _ = build_rest_model(; Lz, Nx, Ny, Nz, td_kwargs = (forward_weight = ω,))
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = 1)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        push!(results, (sweep = label, omega = ω, Δt = Δt,
                        wmax_envelope = env, growth_per_step = gr, status = status,
                        iters = iters, wmax = wmax))
    end
    return results
end

#####
##### D. Topology sweep at Δt=20s
#####

function sweep_topology(; Lz = 30e3, Δt = 20.0, sim_time = 600.0, label = "D_topo")
    cases = [
        (name = "2D_PFB",       grid_kw = (Nx = 64, Ny = 1,  Nz = 64, topology = (Periodic, Flat, Bounded))),
        (name = "3D_PPB",       grid_kw = (Nx = 32, Ny = 32, Nz = 64, topology = (Periodic, Periodic, Bounded))),
        (name = "3D_PBB",       grid_kw = (Nx = 32, Ny = 32, Nz = 64, topology = (Periodic, Bounded,  Bounded))),
        (name = "latlon",       grid_kw = (Nx = 90, Ny = 60, Nz = 64, topology = :latlon)),
    ]
    results = NamedTuple[]
    for c in cases
        n_steps = Int(round(sim_time / Δt))
        @info "[$label] $(c.name)"
        model, _ = build_rest_model(; Lz, c.grid_kw...)
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = 1)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        push!(results, (sweep = label, topology = c.name, Δt = Δt, Lz = Lz,
                        wmax_envelope = env, growth_per_step = gr, status = status,
                        iters = iters, wmax = wmax))
    end
    return results
end

#####
##### E. Reference state form sweep
#####

function sweep_reference(; Lz = 30e3, Nx = 32, Ny = 32, Nz = 64,
                           Δt = 20.0, sim_time = 600.0, label = "E_ref")
    cases = [
        (name = "isothermal_T220",  ref_form = :isothermal_T0, T₀ = 220.0, N² = 0.0),
        (name = "isothermal_T250",  ref_form = :isothermal_T0, T₀ = 250.0, N² = 0.0),
        (name = "isothermal_T280",  ref_form = :isothermal_T0, T₀ = 280.0, N² = 0.0),
        (name = "strat_N2_1e-4",    ref_form = :strat_N2,      T₀ = 300.0, N² = 1e-4),
        (name = "strat_N2_4e-4",    ref_form = :strat_N2,      T₀ = 300.0, N² = 4e-4),
    ]
    results = NamedTuple[]
    for c in cases
        n_steps = Int(round(sim_time / Δt))
        @info "[$label] $(c.name)"
        model, _ = build_rest_model(; Lz, Nx, Ny, Nz, T₀ = c.T₀, N² = c.N², ref_form = c.ref_form)
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = 1)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        push!(results, (sweep = label, name = c.name, T₀ = c.T₀, N² = c.N²,
                        Δt = Δt, wmax_envelope = env, growth_per_step = gr,
                        status = status, iters = iters, wmax = wmax))
    end
    return results
end

#####
##### F. Lz sweep at Δt=20s
#####

function sweep_lz(; Nx = 32, Ny = 32, Nz = 64, Δt = 20.0, sim_time = 600.0, label = "F_Lz")
    Lzs = [5e3, 10e3, 15e3, 20e3, 30e3, 40e3]
    results = NamedTuple[]
    for Lz in Lzs
        n_steps = Int(round(sim_time / Δt))
        @info "[$label] Lz=$(Lz/1000)km"
        model, _ = build_rest_model(; Lz, Nx, Ny, Nz)
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = 1)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        cs = sqrt(1.4 * 287.0 * 300); Δz = Lz/Nz; Δτ = Δt/6
        push!(results, (sweep = label, Lz_km = Lz/1000, Nz = Nz, Δz = Δz,
                        vert_acoustic_cfl = cs * Δτ / Δz,
                        wmax_envelope = env, growth_per_step = gr,
                        status = status, iters = iters, wmax = wmax))
    end
    return results
end

#####
##### G. Δz sweep (Nz at fixed Lz=10km, Δt=20s)
#####

function sweep_dz(; Lz = 10e3, Nx = 32, Ny = 32, Δt = 20.0, sim_time = 600.0, label = "G_dz")
    Nzs = [16, 32, 64, 128]
    results = NamedTuple[]
    for Nz in Nzs
        n_steps = Int(round(sim_time / Δt))
        @info "[$label] Nz=$Nz Δz=$(Lz/Nz)m"
        model, _ = build_rest_model(; Lz, Nx, Ny, Nz)
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = 1)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        cs = sqrt(1.4 * 287.0 * 300); Δz = Lz/Nz; Δτ = Δt/6
        push!(results, (sweep = label, Nz = Nz, Δz = Δz, Lz_km = Lz/1000,
                        vert_acoustic_cfl = cs * Δτ / Δz,
                        wmax_envelope = env, growth_per_step = gr,
                        status = status, iters = iters, wmax = wmax))
    end
    return results
end

#####
##### H. FloatType sweep (Float32 vs Float64)
#####

function sweep_float(; Lz = 30e3, Nx = 32, Ny = 32, Nz = 64,
                       Δt = 20.0, sim_time = 600.0, label = "H_FT")
    results = NamedTuple[]
    for FT in (Float32, Float64)
        n_steps = Int(round(sim_time / Δt))
        @info "[$label] FT=$FT"
        model, _ = build_rest_model(; FT, Lz, Nx, Ny, Nz)
        iters, wmax, status = run_drift!(model; Δt, n_steps, sample_every = 1)
        env = maximum(wmax)
        gr = growth_per_step(iters, wmax)
        push!(results, (sweep = label, FT = string(FT), Δt = Δt,
                        wmax_envelope = env, growth_per_step = gr,
                        status = status, iters = iters, wmax = wmax))
    end
    return results
end

#####
##### I. Reference-state discrete-balance check
#####

function check_reference_balance(; Lz = 30e3, Nx = 16, Ny = 16, Nz = 64,
                                    T₀_list = [220.0, 250.0, 280.0])
    results = NamedTuple[]
    for T₀ in T₀_list
        @info "[I_ref_bal] T₀=$T₀"
        model, θᵇᵍ = build_rest_model(; Lz, Nx, Ny, Nz, T₀)
        ref = model.dynamics.reference_state
        Oceananigans.TimeSteppers.update_state!(model)

        # Discrete hydrostatic residual at every interior z-face
        ρ_arr = Array(interior(ref.density))
        p_arr = Array(interior(ref.pressure))
        Δz_face = Lz / Nz
        g = 9.80665
        ε = zeros(size(ρ_arr, 1), size(ρ_arr, 2), Nz - 1)
        for k in 2:Nz
            ε[:, :, k-1] .= (p_arr[:, :, k] .- p_arr[:, :, k-1]) ./ Δz_face .+
                            g .* (ρ_arr[:, :, k] .+ ρ_arr[:, :, k-1]) ./ 2
        end
        max_residual = maximum(abs, ε)
        max_g_rho = maximum(abs, g .* ρ_arr)
        relative = max_residual / max_g_rho

        # Also compare model.dynamics.pressure (after update_state!) to ref.pressure
        p_eos = Array(interior(model.dynamics.pressure))
        max_p_diff = maximum(abs, p_eos .- p_arr)
        max_p = maximum(abs, p_arr)

        push!(results, (sweep = "I_refbal", T₀ = T₀, Lz = Lz, Nz = Nz,
                        residual_abs = max_residual,
                        residual_relative = relative,
                        eos_vs_ref_p_abs = max_p_diff,
                        eos_vs_ref_p_relative = max_p_diff / max_p))
    end
    return results
end

#####
##### Main runner
#####

function main()
    all_results = Dict{Symbol, Any}()

    @info "===== Sweep A: Δt sweep ====="
    all_results[:A_dt] = sweep_dt()

    @info "===== Sweep B: substep-count N ====="
    all_results[:B_N] = sweep_substep_count()

    @info "===== Sweep C: ω forward_weight (diagnostic) ====="
    all_results[:C_omega] = sweep_omega()

    @info "===== Sweep D: topology ====="
    all_results[:D_topo] = sweep_topology()

    @info "===== Sweep E: reference state form ====="
    all_results[:E_ref] = sweep_reference()

    @info "===== Sweep F: Lz ====="
    all_results[:F_Lz] = sweep_lz()

    @info "===== Sweep G: Δz (Nz at fixed Lz) ====="
    all_results[:G_dz] = sweep_dz()

    @info "===== Sweep H: Float32 vs Float64 ====="
    all_results[:H_FT] = sweep_float()

    @info "===== Static check I: reference-state balance ====="
    all_results[:I_refbal] = check_reference_balance()

    @info "Saving results to $RESULTS_PATH"
    jldsave(RESULTS_PATH; all_results)

    # Flat CSV
    open(CSV_PATH, "w") do io
        println(io, "sweep,key,value,wmax_envelope,growth_per_step,status,extra")
        for (sk, rs) in all_results
            for r in rs
                k = haskey(r, :Δt) ? "Δt=$(r.Δt)" :
                    haskey(r, :N_substeps) ? "N=$(r.N_substeps)" :
                    haskey(r, :omega) ? "ω=$(r.omega)" :
                    haskey(r, :topology) ? "topo=$(r.topology)" :
                    haskey(r, :name) ? "ref=$(r.name)" :
                    haskey(r, :Lz_km) ? "Lz=$(r.Lz_km)km" :
                    haskey(r, :Nz) ? "Nz=$(r.Nz)" :
                    haskey(r, :FT) ? "FT=$(r.FT)" :
                    haskey(r, :T₀) ? "T₀=$(r.T₀)" : "?"
                env = get(r, :wmax_envelope, missing)
                gr = get(r, :growth_per_step, missing)
                st = get(r, :status, missing)
                println(io, "$sk,$k,_,$env,$gr,$st,_")
            end
        end
    end

    return all_results
end

results = main()
@info "Done. results.jld2 + results.csv in $OUTDIR"
