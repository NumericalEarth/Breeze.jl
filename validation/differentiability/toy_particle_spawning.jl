# Toy differentiable "secondary ice production" on a fixed-capacity particle buffer.
#
#   julia --project=~/droplet-tasks-env toy_particle_spawning.jl [parts]
#
# parts: e = plain Enzyme (CPU arrays), r = Reactant (array ops), k = KernelAbstractions kernels
#        (plain Enzyme on CPU + Reactant raise), g = KA kernels on CUDA with Enzyme, w = Reactant while loop
#
# Each particle slot carries a size r, a multiplicity (weight) w and an activity flag a ∈ {0, 1}.
# Each step: growth r += g Δt; removal of particles with r > rmax ("melting/fallout"); creation
# of one fragment per active parent into a free slot (k-th active parent → k-th free slot, found
# with cumulative sums). Two creation formulations:
#   expected value: the fragment always exists, with multiplicity λ Δt w_parent (continuous in λ)
#   stochastic:     the fragment exists with probability λ Δt (u < λΔt) and multiplicity w_parent
# Loss J = Σ a w r². dJ/dλ and dJ/dg are compared with central finite differences.

using Enzyme
using Printf
using Random
using GPUArraysCore: @allowscalar
using Oceananigans: Oceananigans # KernelAbstractions is not a direct dependency of the environment
const KernelAbstractions = Oceananigans.Architectures.KA
using .KernelAbstractions: @kernel, @index
using Reactant
import CUDA

parts = length(ARGS) ≥ 1 ? ARGS[1] : "erkw"

const N = 256       # capacity
const n₀ = 8        # initially active
const nsteps = 5
const Δt = 0.1
const rmax = 1.45
const rfragment = 0.3
const SORTKEYS = Ref(get(ENV, "SORTKEYS", "false") == "true")

fd(f, x; h=1e-6) = (f(x + h) - f(x - h)) / 2h
report(name, ad, fdv) = @printf("%-62s AD = % .10e  FD = % .10e  rel.err = %.2e\n", name, ad, fdv,
                                abs(ad - fdv) / max(abs(fdv), 1e-300))

function attempt(f, name)
    try
        f()
    catch err
        msg = sprint(showerror, err)
        println("FAILED ", name, ": ", first(msg, 1200))
        length(msg) > 2400 && println("  ... ", last(msg, 1200))
    end
end

function initial_state(T=Float64)
    r = zeros(T, N); w = zeros(T, N); a = zeros(T, N)
    for i in 1:n₀
        r[i] = 0.5 + 0.1i; w[i] = 1; a[i] = 1
    end
    return r, w, a
end

# Array-op step (works for Array and Reactant traced arrays).
# `stochastic` selects the creation formulation; u holds pre-drawn uniforms (common random numbers).
function step_arrays(r, w, a, λ, g, u, stochastic)
    r = r .+ g * Δt .* a
    a = a .* (r .< rmax)                          # removal: deactivate
    active_rank = cumsum(a)                        # k for the k-th active slot
    free = 1 .- a
    free_rank = cumsum(free)                       # q for the q-th free slot
    nactive = sum(a)
    # index of the q-th active slot: stable sort puts active slots first
    # integer permutation, no derivative; unique keys (XLA sort is not guaranteed stable)
    order = SORTKEYS[] ? sortperm(free .* N .+ (1:N)) : sortperm(free)
    q = clamp.(round.(Int, free_rank), 1, N)
    parent = order[q]                              # gather (traced indices)
    spawn = free .* (free_rank .<= nactive)        # mask of slots receiving a fragment
    p = λ * Δt
    if stochastic
        born = spawn .* (u[parent] .< p)
        wnew = w[parent]
    else
        born = spawn
        wnew = p .* w[parent]
    end
    r = ifelse.(born .> 0, rfragment, r)
    w = ifelse.(born .> 0, wnew, w)
    a = a .+ born
    return r, w, a
end

loss_arrays(θ::AbstractVector, args...) = loss_arrays(@allowscalar(θ[1]), @allowscalar(θ[2]), args...)
loss_compacted(θ::AbstractVector, args...) = loss_compacted(@allowscalar(θ[1]), @allowscalar(θ[2]), args...)
loss_traced_loop(θ::AbstractVector, args...) = loss_traced_loop(@allowscalar(θ[1]), @allowscalar(θ[2]), args...)

function loss_arrays(λ, g, r, w, a, u, stochastic)
    for _ in 1:nsteps
        r, w, a = step_arrays(r, w, a, λ, g, u, stochastic)
    end
    return sum(a .* w .* r .^ 2)
end

# Compaction (sort active to the front) does not change J but exercises gather/sort in AD
function loss_compacted(λ, g, r, w, a, u, stochastic)
    for _ in 1:nsteps
        r, w, a = step_arrays(r, w, a, λ, g, u, stochastic)
        order = sortperm(1 .- a)
        r = r[order]; w = w[order]; a = a[order]
    end
    return sum(a .* w .* r .^ 2)
end

r₀, w₀, a₀ = initial_state()
u₀ = rand(Xoshiro(42), N)
λ₀, g₀ = 2.0, 1.0

# Gradient with respect to θ = [λ, g] (Duplicated arrays: Active scalars hit a Reactant KeyError)
function grad_theta(mode, L, θ, dθ, args...)
    fill!(dθ, 0)
    Enzyme.autodiff(mode, L, Active, Duplicated(θ, dθ), map(Const, args)...)
    return dθ
end
θ₀ = [λ₀, g₀]

if 'e' in parts
    println("=== (b) plain Enzyme, CPU Arrays")
    for stochastic in (false, true), (lname, L) in (("", loss_arrays), (" + compaction", loss_compacted))
        label = stochastic ? "stochastic" : "expected-value"
        J = L(λ₀, g₀, r₀, w₀, a₀, u₀, stochastic)
        fdλ = fd(λ -> L(λ, g₀, r₀, w₀, a₀, u₀, stochastic), λ₀)
        fdg = fd(g -> L(λ₀, g, r₀, w₀, a₀, u₀, stochastic), g₀)
        @printf("%s%s: J = %.6f, active = %d\n", label, lname, J, 0)
        attempt("enzyme $label") do
            dλ, dg = grad_theta(set_runtime_activity(Reverse), L, copy(θ₀), zeros(2), r₀, w₀, a₀, u₀, stochastic)
            report("Enzyme $label$lname dJ/dλ", dλ, fdλ)
            report("Enzyme $label$lname dJ/dg", dg, fdg)
        end
    end
    # FD for the stochastic case with large h shows the discontinuous expected-value sensitivity
    for h in (1e-6, 1e-3, 1e-1)
        @printf("stochastic FD dJ/dλ with h = %.0e: % .6e\n", h, fd(λ -> loss_arrays(λ, g₀, r₀, w₀, a₀, u₀, true), λ₀; h))
    end
    # Monte Carlo of the stochastic expected value vs the expected-value formulation
    Jmc(λ, M) = sum(loss_arrays(λ, g₀, r₀, w₀, a₀, rand(Xoshiro(s), N), true) for s in 1:M) / M
    @printf("E[J_stochastic] (2000 samples) = %.4f, J_expected = %.4f (not equal: the mean-field closure is not exact through removal/nonlinearity)\n",
            Jmc(λ₀, 2000), loss_arrays(λ₀, g₀, r₀, w₀, a₀, u₀, false))
    @printf("dE[J_stochastic]/dλ by FD on 2000 CRN samples, h=0.2: %.4f\n", (Jmc(λ₀ + 0.2, 2000) - Jmc(λ₀ - 0.2, 2000)) / 0.4)
end

if 'r' in parts || 'w' in parts || 'k' in parts
    Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
end

if 'r' in parts
    println("=== (a) Reactant, array ops (backend = ", get(ENV, "REACTANT_BACKEND", "cpu"), ")")
    rr, wr, ar, ur = Reactant.to_rarray.((r₀, w₀, a₀, u₀))
    for stochastic in (false, true), (lname, L) in (("", loss_arrays), (" + compaction", loss_compacted))
        label = stochastic ? "stochastic" : "expected-value"
        fdλ = fd(λ -> L(λ, g₀, r₀, w₀, a₀, u₀, stochastic), λ₀)
        fdg = fd(g -> L(λ₀, g, r₀, w₀, a₀, u₀, stochastic), g₀)
        attempt("reactant $label$lname") do
            θr, dθr = Reactant.to_rarray(θ₀), Reactant.to_rarray(zeros(2))
            t = @elapsed compiled = Reactant.@compile grad_theta(Reverse, L, θr, dθr, rr, wr, ar, ur, stochastic)
            dλ, dg = Array(compiled(Reverse, L, θr, dθr, rr, wr, ar, ur, stochastic))
            Jc = @jit L(θr, rr, wr, ar, ur, stochastic)
            @printf("compile %.1f s; J = %.6f (CPU: %.6f)\n", t, Float64(Jc), L(λ₀, g₀, r₀, w₀, a₀, u₀, stochastic))
            report("Reactant $label$lname dJ/dλ", Float64(dλ), fdλ)
            report("Reactant $label$lname dJ/dg", Float64(dg), fdg)
        end
    end
    attempt("hlo size") do
        hlo = repr(@code_hlo loss_arrays(ConcreteRNumber(λ₀), ConcreteRNumber(g₀), rr, wr, ar, ur, false))
        println("StableHLO of loss (unrolled 5 steps): ", count(==('\n'), hlo), " lines; ops: ",
                join(sort(unique(m.match for m in eachmatch(r"stablehlo\.[a-z_]+", hlo))), " "))
    end
end

# Dynamic trip count: run the time loop as a traced loop with a runtime number of steps.
function loss_traced_loop(λ, g, r, w, a, u, n)
    @trace checkpointing=false for _ in 1:n
        r, w, a = step_arrays(r, w, a, λ, g, u, false)
    end
    return sum(a .* w .* r .^ 2)
end

if 'w' in parts
    println("=== (a') Reactant, @trace loop with runtime trip count")
    rr, wr, ar, ur = Reactant.to_rarray.((r₀, w₀, a₀, u₀))
    λr, gr = ConcreteRNumber(λ₀), ConcreteRNumber(g₀)
    nr = ConcreteRNumber(nsteps)
    attempt("traced loop forward") do
        f = Reactant.@compile loss_traced_loop(λr, gr, rr, wr, ar, ur, nr)
        @printf("traced loop J(n=5) = %.6f, J(n=3) = %.6f (CPU n=5: %.6f)\n", Float64(f(λr, gr, rr, wr, ar, ur, nr)),
                Float64(f(λr, gr, rr, wr, ar, ur, ConcreteRNumber(3))), loss_arrays(λ₀, g₀, r₀, w₀, a₀, u₀, false))
        hlo = repr(@code_hlo loss_traced_loop(λr, gr, rr, wr, ar, ur, nr))
        println("contains stablehlo.while: ", occursin("stablehlo.while", hlo), "; HLO lines: ", count(==('\n'), hlo))
    end
    attempt("traced loop reverse") do
        θr, dθr = Reactant.to_rarray(θ₀), Reactant.to_rarray(zeros(2))
        f = Reactant.@compile grad_theta(Reverse, loss_traced_loop, θr, dθr, rr, wr, ar, ur, nr)
        dλ, dg = Array(f(Reverse, loss_traced_loop, θr, dθr, rr, wr, ar, ur, nr))
        report("Reactant traced-loop (runtime n) dJ/dλ", Float64(dλ), fd(λ -> loss_arrays(λ, g₀, r₀, w₀, a₀, u₀, false), λ₀))
        report("Reactant traced-loop (runtime n) dJ/dg", Float64(dg), fd(g -> loss_arrays(λ₀, g, r₀, w₀, a₀, u₀, false), g₀))
    end
end

if 'W' in parts
    rr, wr, ar, ur = Reactant.to_rarray.((r₀, w₀, a₀, u₀))
    λr, gr = ConcreteRNumber(λ₀), ConcreteRNumber(g₀)
    # Data-dependent while: grow until the number of active particles reaches a target
    attempt("data-dependent while forward + reverse") do
        grow_until(θ::AbstractVector, args...) = grow_until(@allowscalar(θ[1]), @allowscalar(θ[2]), args...)
        function grow_until(λ, g, r, w, a, u)
            k = 0 * sum(a)
            @trace while (sum(a) < 100) & (k < 50)
                r, w, a = step_arrays(r, w, a, λ, g, u, false)
                k += 1
            end
            return sum(a .* w .* r .^ 2)
        end
        println("tracing data-dependent while"); flush(stdout)
        hlo = repr(@code_hlo optimize=false grow_until(λr, gr, rr, wr, ar, ur))
        println("unoptimized HLO lines: ", count(==('\n'), hlo)); flush(stdout)
        f = Reactant.@compile grow_until(λr, gr, rr, wr, ar, ur)
        @printf("while (active < 100): J = %.6f (CPU %.6f)\n", Float64(f(λr, gr, rr, wr, ar, ur)), grow_until(λ₀, g₀, r₀, w₀, a₀, u₀))
        θr, dθr = Reactant.to_rarray(θ₀), Reactant.to_rarray(zeros(2))
        fg = Reactant.@compile grad_theta(Reverse, grow_until, θr, dθr, rr, wr, ar, ur)
        dλ, dg = Array(fg(Reverse, grow_until, θr, dθr, rr, wr, ar, ur))
        report("Reactant data-dependent while dJ/dλ", Float64(dλ), fd(λ -> grow_until(λ, g₀, r₀, w₀, a₀, u₀), λ₀))
        report("Reactant data-dependent while dJ/dg", Float64(dg), fd(g -> grow_until(λ₀, g, r₀, w₀, a₀, u₀), g₀))
    end
end

#####
##### KernelAbstractions version: one kernel per step stage, as in Oceananigans particle code
#####

if 'k' in parts || 'g' in parts
    @kernel function _grow_and_remove!(r, a, g)
        i = @index(Global)
        @inbounds begin
            rᵢ = r[i] + g * Δt * a[i]
            r[i] = rᵢ
            a[i] = a[i] * (rᵢ < rmax)
        end
    end
    # Fragment creation into precomputed (parent, spawn) slots
    @kernel function _spawn!(r, w, a, parent, spawn, p)
        i = @index(Global)
        @inbounds begin
            born = spawn[i]
            j = parent[i]
            wnew = p * w[j]
            r[i] = ifelse(born > 0, rfragment, r[i])
            w[i] = ifelse(born > 0, wnew, w[i])
            a[i] = a[i] + born
        end
    end

    function step_kernels!(r, w, a, λ, g)
        backend = KernelAbstractions.get_backend(r)
        _grow_and_remove!(backend)(r, a, g; ndrange=length(r))
        free = 1 .- a
        free_rank = cumsum(free)
        nactive = sum(a)
        order = sortperm(free)
        parent = order[clamp.(round.(Int, free_rank), 1, length(r))]
        spawn = free .* (free_rank .<= nactive)
        _spawn!(backend)(r, w, a, parent, spawn, λ * Δt; ndrange=length(r))
        return nothing
    end

    function loss_kernels(λ, g, r, w, a)
        for _ in 1:nsteps
            step_kernels!(r, w, a, λ, g)
        end
        return sum(a .* w .* r .^ 2)
    end
    loss_kernels(θ::AbstractVector, r, w, a) = loss_kernels(@allowscalar(θ[1]), @allowscalar(θ[2]), r, w, a)
    function grad_kernels(mode, θ, dθ, r, w, a, dr, dw, da)
        fill!(dθ, 0); fill!(dr, 0); fill!(dw, 0); fill!(da, 0)
        Enzyme.autodiff(mode, loss_kernels, Active, Duplicated(θ, dθ), Duplicated(r, dr), Duplicated(w, dw), Duplicated(a, da))
        return dθ
    end
    fresh() = copy.((r₀, w₀, a₀))
    Jk(λ, g) = loss_kernels(λ, g, fresh()...)
end

if 'k' in parts
    println("=== (b) KernelAbstractions kernels, plain Enzyme on CPU")
    fdλ = fd(λ -> Jk(λ, g₀), λ₀); fdg = fd(g -> Jk(λ₀, g), g₀)
    @printf("KA J = %.6f (array version %.6f)\n", Jk(λ₀, g₀), loss_arrays(λ₀, g₀, r₀, w₀, a₀, u₀, false))
    attempt("KA Enzyme CPU") do
        r, w, a = fresh()
        dr, dw, da = zero(r), zero(w), zero(a)
        dλ, dg = grad_kernels(set_runtime_activity(Reverse), copy(θ₀), zeros(2), r, w, a, dr, dw, da)
        report("KA+Enzyme CPU dJ/dλ", dλ, fdλ)
        report("KA+Enzyme CPU dJ/dg", dg, fdg)
    end
    println("=== (a) KernelAbstractions kernels inside Reactant (raise=true, raise_first=true)")
    attempt("KA Reactant forward") do
        r, w, a = Reactant.to_rarray.(fresh())
        f = Reactant.@compile loss_kernels(ConcreteRNumber(λ₀), ConcreteRNumber(g₀), r, w, a)
        r, w, a = Reactant.to_rarray.(fresh())
        @printf("Reactant KA J = %.6f\n", Float64(f(ConcreteRNumber(λ₀), ConcreteRNumber(g₀), r, w, a)))
    end
    attempt("KA Reactant reverse") do
        r, w, a = Reactant.to_rarray.(fresh())
        dr, dw, da = Reactant.to_rarray.(zero.(fresh()))
        θr, dθr = Reactant.to_rarray(θ₀), Reactant.to_rarray(zeros(2))
        f = Reactant.@compile raise=true raise_first=true grad_kernels(Reverse, θr, dθr, r, w, a, dr, dw, da)
        r, w, a = Reactant.to_rarray.(fresh())
        dλ, dg = Array(f(Reverse, θr, dθr, r, w, a, dr, dw, da))
        report("KA+Reactant (raised) dJ/dλ", Float64(dλ), fdλ)
        report("KA+Reactant (raised) dJ/dg", Float64(dg), fdg)
    end
end

if 'g' in parts
    println("=== (b) KernelAbstractions kernels on CUDA with plain Enzyme")
    fdλ = fd(λ -> Jk(λ, g₀), λ₀)
    attempt("KA Enzyme CUDA") do
        r, w, a = CUDA.CuArray.(fresh())
        dr, dw, da = zero(r), zero(w), zero(a)
        dλ, dg = Array(grad_kernels(set_runtime_activity(Reverse), copy(θ₀), zeros(2), r, w, a, dr, dw, da))
        report("KA+Enzyme CUDA dJ/dλ", dλ, fdλ)
    end
end

# Minimal reproducers for the memory blow-up when tracing a data-dependent `@trace while`
if 'V' in parts
    Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
    rr, wr, ar, ur = Reactant.to_rarray.((r₀, w₀, a₀, u₀))
    λr, gr = ConcreteRNumber(λ₀), ConcreteRNumber(g₀)
    function while_trivial(a)
        @trace while sum(a) < 100
            a = a .+ 1
        end
        return sum(a)
    end
    function while_counter(a)
        k = 0
        @trace while k < 5
            a = a .+ 1
            k += 1
        end
        return sum(a)
    end
    function while_step_counter(λ, g, r, w, a, u)
        k = 0
        @trace while k < 5
            r, w, a = step_arrays(r, w, a, λ, g, u, false)
            k += 1
        end
        return sum(a .* w .* r .^ 2)
    end
    function while_step_data(λ, g, r, w, a, u)
        @trace while sum(a) < 100
            r, w, a = step_arrays(r, w, a, λ, g, u, false)
        end
        return sum(a .* w .* r .^ 2)
    end
    for (name, f, args) in (("trivial data-dependent", while_trivial, (ar,)),
                            ("integer counter", while_counter, (ar,)),
                            ("particle step, integer counter", while_step_counter, (λr, gr, rr, wr, ar, ur)),
                            ("particle step, data-dependent", while_step_data, (λr, gr, rr, wr, ar, ur)))
        attempt(name) do
            println("tracing while: ", name); flush(stdout)
            t = @elapsed hlo = repr(@code_hlo optimize=false f(args...))
            @printf("  traced in %.1f s, %d lines, maxrss %.1f GB\n", t, count(==('\n'), hlo), Sys.maxrss() / 1e9); flush(stdout)
            c = Reactant.@compile f(args...)
            println("  result ", Float64(c(args...)))
        end
    end
end
