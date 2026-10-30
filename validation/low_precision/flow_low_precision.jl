# Low-precision storage in 2D decaying turbulence (Oceananigans, GPU, Float64 compute):
#   (a) Lagrangian: particle positions stored as absolute coordinates or as (cell index + low-precision offset),
#       rounded after every time step. The flow is identical to the reference, so per-particle errors are exact.
#   (b) Eulerian: the velocity fields u, v rounded to BFloat16 / Float32 after every time step.
# Metrics: per-particle position error vs the Float64 reference (in cell widths), pair separation statistics
# for pairs initially 0.1 Δ apart, kinetic energy and vorticity correlation.
#
#   julia --project=<env> validation/low_precision/flow_low_precision.jl

using Oceananigans
using Oceananigans.Units
using CUDA
using Random
using Printf
using Statistics

include(joinpath(@__DIR__, "quantize.jl"))
const Float64Dummy = StorageFormat(53, -1022, Inf) # unused placeholder (keeps the variant tuple isbits)

const arch = GPU()
const Nx = 256
const L = 2π
const Δ = L / Nx
const Np = 8192            # particles per variant (4096 pairs)
const Δt = 0.01
const t_end = 30.0
const output_times = (1.0, 3.0, 10.0, 30.0)

# Particle storage variants: (label, mode, format, stochastic, fixed-point bits)
const particle_variants = [
    ("Float64 (reference)", 0, Float64Dummy, false, 0),
    ("Float32 absolute", 1, Float32Format, false, 0),
    ("BFloat16 absolute RN", 1, BFloat16Format, false, 0),
    ("BFloat16 absolute SR", 1, BFloat16Format, true, 0),
    ("Float16 absolute RN", 1, Float16Format, false, 0),
    ("Float16 absolute SR", 1, Float16Format, true, 0),
    ("index + BFloat16 offset RN", 2, BFloat16Format, false, 0),
    ("index + BFloat16 offset SR", 2, BFloat16Format, true, 0),
    ("index + Float16 offset RN", 2, Float16Format, false, 0),
    ("index + Float16 offset SR", 2, Float16Format, true, 0),
    ("index + FP8-E4M3 offset SR", 2, E4M3Format, true, 0),
    ("index + 8-bit fixed offset RN", 3, Float64Dummy, false, 8),
    ("index + 8-bit fixed offset SR", 3, Float64Dummy, true, 8),
    ("index + 16-bit fixed offset RN", 3, Float64Dummy, false, 16),
]
const K = length(particle_variants)

@inline function store_position(x, mode, f, sr, bits, u)
    if mode == 1
        return quantize(x, f, sr, u)
    elseif mode == 2
        ξ = x / Δ
        i = floor(ξ)
        return (i + quantize(ξ - i, f, sr, u)) * Δ
    elseif mode == 3
        ξ = x / Δ
        i = floor(ξ)
        s = exp2(bits)
        φ = (ξ - i) * s
        return (i + ifelse(sr, floor(φ + u), round(φ)) / s) * Δ
    else
        return x
    end
end

function initial_conditions()
    rng = Xoshiro(42)
    # Smooth random streamfunction: sum of random Fourier modes with |k| ≤ 8
    ψ = zeros(Nx, Nx)
    x = (0:Nx-1) .* Δ
    for kx in -8:8, ky in -8:8
        k² = kx^2 + ky^2
        (k² == 0 || k² > 64) && continue
        a = randn(rng) / k²^(3/4); φ = 2π * rand(rng)
        ψ .+= a .* cos.(kx .* x .+ ky .* x' .+ φ)
    end
    # u = -∂ψ/∂y, v = ∂ψ/∂x on the C-grid (u at (i-1/2, j), v at (i, j-1/2)), discretely nondivergent
    sh(A, dx, dy) = circshift(A, (dx, dy))
    ψc = (ψ .+ sh(ψ, 1, 0) .+ sh(ψ, 0, 1) .+ sh(ψ, 1, 1)) ./ 4 # ψ at cell corners (i-1/2, j-1/2)
    u = -(sh(ψc, 0, -1) .- ψc) ./ Δ
    v = (sh(ψc, -1, 0) .- ψc) ./ Δ
    U = maximum(sqrt.(u .^ 2 .+ v .^ 2))
    u ./= U; v ./= U
    # Particle pairs, initial separation 0.1 Δ in a random direction
    xp = zeros(Np); yp = zeros(Np)
    for k in 1:2:Np
        xp[k] = L * rand(rng); yp[k] = L * rand(rng)
        θ = 2π * rand(rng)
        xp[k+1] = mod(xp[k] + 0.1Δ * cos(θ), L); yp[k+1] = mod(yp[k] + 0.1Δ * sin(θ), L)
    end
    return u, v, xp, yp
end

function run_flow(; eulerian_format = nothing, eulerian_sr = false, round_particles = true, seed = 1)
    grid = RectilinearGrid(arch, size = (Nx, Nx), extent = (L, L), topology = (Periodic, Periodic, Flat))
    u₀, v₀, xp, yp = initial_conditions()
    xs = repeat(xp, K); ys = repeat(yp, K)
    particles = LagrangianParticles(x = CuArray(xs), y = CuArray(ys), z = CUDA.zeros(Float64, K * Np))
    model = NonhydrostaticModel(grid; advection = WENO(order = 5), closure = ScalarDiffusivity(ν = 1e-5),
                                particles)
    set!(model, u = u₀, v = v₀)
    simulation = Simulation(model; Δt, stop_time = t_end, verbose = false)
    Random.seed!(CUDA.default_rng(), seed)

    px = model.particles.properties.x
    py = model.particles.properties.y
    function round_storage!(sim)
        if round_particles
            for (k, (_, mode, f, sr, bits)) in enumerate(particle_variants)
                mode == 0 && continue
                r = (k-1)*Np+1:k*Np
                ux = CUDA.rand(Float64, Np); uy = CUDA.rand(Float64, Np)
                vx = view(px, r); vy = view(py, r)
                vx .= store_position.(vx, mode, Ref(f), sr, bits, ux)
                vy .= store_position.(vy, mode, Ref(f), sr, bits, uy)
                vx .= mod.(vx, L); vy .= mod.(vy, L)
            end
        end
        if eulerian_format !== nothing
            for φ in (sim.model.velocities.u, sim.model.velocities.v)
                d = interior(φ)
                d .= quantize.(d, Ref(eulerian_format), eulerian_sr, CUDA.rand(Float64, size(d)))
            end
        end
        return nothing
    end
    add_callback!(simulation, round_storage!, IterationInterval(1))

    snapshots = Dict{Float64, Any}()
    for t in output_times
        simulation.stop_time = t
        run!(simulation)
        u = Array(interior(model.velocities.u)); v = Array(interior(model.velocities.v))
        ζ = (circshift(v, (-1, 0, 0)) .- v) ./ Δ .- (circshift(u, (0, -1, 0)) .- u) ./ Δ
        snapshots[t] = (x = Array(px), y = Array(py), KE = mean(u .^ 2 .+ v .^ 2) / 2, ζ = ζ)
    end
    return snapshots
end

periodic_difference(a, b) = mod(a - b + L / 2, L) - L / 2

function particle_metrics(snap, ref, k)
    r = (k-1)*Np+1:k*Np
    dx = periodic_difference.(snap.x[r], ref.x[1:Np]); dy = periodic_difference.(snap.y[r], ref.y[1:Np])
    rms = sqrt(mean(dx .^ 2 .+ dy .^ 2)) / Δ
    sep(x, y) = [sqrt(periodic_difference(x[j+1], x[j])^2 + periodic_difference(y[j+1], y[j])^2) for j in 1:2:Np]
    s = sep(snap.x[r], snap.y[r]); s₀ = sep(ref.x[1:Np], ref.y[1:Np])
    collapsed = count(==(0), s) / length(s)
    lnR = mean(log.(max.(s, 1e-12) ./ (0.1Δ))); lnR₀ = mean(log.(s₀ ./ (0.1Δ)))
    return (; rms, lnR, lnR₀, collapsed)
end

function main_flow()
    @info "reference + particle-storage variants"
    ref = run_flow()
    println("\n### Lagrangian position storage (flow identical to reference)\n")
    println("| storage | t | RMS position error [Δ] | ⟨ln R/R₀⟩ (ref) | pairs collapsed to 0 [%] |")
    println("|---|---|---|---|---|")
    for (k, v) in enumerate(particle_variants), t in output_times
        m = particle_metrics(ref[t], ref[t], k)
        @printf("| %s | %g | %.3g | %.3f (%.3f) | %.1f |\n", v[1], t, m.rms, m.lnR, m.lnR₀, 100m.collapsed)
    end
    println("\n### Eulerian velocity storage (particles Float64; error of reference-particle trajectories and flow)\n")
    println("| u,v storage | t | KE rel. err [%] | corr(ζ, ζ_ref) | particle RMS err [Δ] | ⟨ln R/R₀⟩ |")
    println("|---|---|---|---|---|---|")
    for (label, f, sr) in (("Float64 rerun (determinism check)", nothing, false), ("Float32 RN", Float32Format, false),
                           ("BFloat16 RN", BFloat16Format, false), ("BFloat16 SR", BFloat16Format, true),
                           ("Float16 RN", Float16Format, false))
        @info "Eulerian $label"
        run = run_flow(; eulerian_format = f, eulerian_sr = sr, round_particles = false)
        for t in output_times
            m = particle_metrics(run[t], ref[t], 1)
            c = cor(vec(run[t].ζ), vec(ref[t].ζ))
            @printf("| %s | %g | %+.3g | %.6f | %.3g | %.3f |\n", label, t, 100 * (run[t].KE - ref[t].KE) / ref[t].KE, c, m.rms, m.lnR)
        end
    end
end

main_flow()
