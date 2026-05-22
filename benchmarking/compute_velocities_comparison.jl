#####
##### Compare three launch strategies for `compute_velocities!` under Reactant + Enzyme.
#####
##### Variants:
#####   1. XYZ   — pre-PR-#722:  `launch!(:xyz, …)` over (Nx, Ny, Nz).
#####                            Boundary faces on Bounded sides are NOT written.
#####   2. FACE  — post-PR-#722: single launch over
#####                            (length(Face(),TX,Nx), length(Face(),TY,Ny), length(Face(),TZ,Nz)).
#####                            Boundary faces ARE written; one homogeneous launch
#####                            shape across u/v/w; extra writes land in halos.
#####   3. SPLIT — proposed:     three per-component launches sized exactly to each
#####                            field's interior (no halo over-writes).
#####
##### For each variant we compile a reverse-mode gradient of
#####   loss(model) = sum(interior(u)^2 + interior(v)^2 + interior(w)^2)
##### through `Reactant.@compile raise=true sync=true`, time the second invocation,
##### and dump the post-Enzyme StableHLO module to /tmp/ for diffing.
#####
##### Run from `benchmarking/`:
#####   julia --color=yes --project compute_velocities_comparison.jl
#####
##### The grid (64x64x32 PBB, WENO5, Float64, --simplified, no microphysics,
##### compressible_explicit) matches the CI AD benchmark config exactly so this
##### is a faithful comparison.
#####
##### NOTE: with this branch's pinned `Enzyme = "~0.13.147"` the bad-fusion path
##### is suppressed and absolute timings should be close across variants. To
##### exercise the 5/21 regression, temporarily loosen the Enzyme compat to
##### allow 0.13.148+ and re-run; the FACE variant should then show the
##### catastrophic spill in the ptxas warnings (and much longer wall time).
#####

using Pkg
Pkg.activate(@__DIR__)

# Reactant must load before CUDA so ReactantCUDAExt activates with CUDA hooks.
using Reactant
using CUDA
using Enzyme
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior
using Oceananigans.Grids: topology
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans: Face

using KernelAbstractions: @kernel, @index

using Breeze
using Breeze: CompressibleDynamics, ExplicitTimeStepping
using Breeze.AtmosphereModels: dynamics_density

using BreezeBenchmarks: convective_boundary_layer

using Printf: @printf, @sprintf
using Statistics: mean, std

#####
##### Variant 1: XYZ — pre-PR-#722 launch shape
#####

@kernel function _compute_velocities_xyz!(u, v, w, ρu, ρv, ρw, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    ρ = dynamics_density(dynamics)
    @inbounds u[i, j, k] = ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
    @inbounds v[i, j, k] = ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
    @inbounds w[i, j, k] = ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρ)
end

function compute_velocities_xyz!(model)
    grid = model.grid
    arch = grid.architecture
    density = dynamics_density(model.dynamics)
    fill_halo_regions!(density)
    fill_halo_regions!(model.momentum)
    launch!(arch, grid, :xyz,
            _compute_velocities_xyz!,
            model.velocities.u, model.velocities.v, model.velocities.w,
            model.momentum.ρu,  model.momentum.ρv,  model.momentum.ρw,
            grid, model.dynamics)
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities)
    return nothing
end

#####
##### Variant 2: FACE — post-PR-#722 single-launch shape
#####

@kernel function _compute_velocities_face!(u, v, w, ρu, ρv, ρw, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    ρ = dynamics_density(dynamics)
    @inbounds u[i, j, k] = ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
    @inbounds v[i, j, k] = ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
    @inbounds w[i, j, k] = ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρ)
end

function compute_velocities_face!(model)
    grid = model.grid
    arch = grid.architecture
    density = dynamics_density(model.dynamics)
    fill_halo_regions!(density)
    fill_halo_regions!(model.momentum)
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    launch!(arch, grid, KernelParameters(1:length(Face(), TX(), Nx),
                                         1:length(Face(), TY(), Ny),
                                         1:length(Face(), TZ(), Nz)),
            _compute_velocities_face!,
            model.velocities.u, model.velocities.v, model.velocities.w,
            model.momentum.ρu,  model.momentum.ρv,  model.momentum.ρw,
            grid, model.dynamics)
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities)
    return nothing
end

#####
##### Variant 3: SPLIT — three per-component launches sized to each field's interior
#####

@kernel function _compute_u_split!(u, ρu, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    ρ = dynamics_density(dynamics)
    @inbounds u[i, j, k] = ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, ρ)
end

@kernel function _compute_v_split!(v, ρv, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    ρ = dynamics_density(dynamics)
    @inbounds v[i, j, k] = ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, ρ)
end

@kernel function _compute_w_split!(w, ρw, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    ρ = dynamics_density(dynamics)
    @inbounds w[i, j, k] = ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, ρ)
end

function compute_velocities_split!(model)
    grid = model.grid
    arch = grid.architecture
    density = dynamics_density(model.dynamics)
    fill_halo_regions!(density)
    fill_halo_regions!(model.momentum)
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    launch!(arch, grid,
            KernelParameters(1:length(Face(), TX(), Nx), 1:Ny, 1:Nz),
            _compute_u_split!,
            model.velocities.u, model.momentum.ρu, grid, model.dynamics)
    launch!(arch, grid,
            KernelParameters(1:Nx, 1:length(Face(), TY(), Ny), 1:Nz),
            _compute_v_split!,
            model.velocities.v, model.momentum.ρv, grid, model.dynamics)
    launch!(arch, grid,
            KernelParameters(1:Nx, 1:Ny, 1:length(Face(), TZ(), Nz)),
            _compute_w_split!,
            model.velocities.w, model.momentum.ρw, grid, model.dynamics)
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities)
    return nothing
end

#####
##### Loss + gradient wrappers (one per variant for type-stable specialization)
#####

# Sum of squared velocities — downstream reduction so the reverse pass has to
# propagate gradients through each divide. Faithful in the sense that the
# divides and the reducer are in the same compiled program, which is what
# matters for XLA fusion decisions. Not identical to the full timestep loss
# in benchmarking/src/timestepping.jl, but isolates the kernel-launch variable.
@inline function _kinetic_energy_loss(model)
    u, v, w = model.velocities.u, model.velocities.v, model.velocities.w
    return sum(interior(u) .^ 2) + sum(interior(v) .^ 2) + sum(interior(w) .^ 2)
end

function loss_xyz(model)
    compute_velocities_xyz!(model)
    return _kinetic_energy_loss(model)
end

function loss_face(model)
    compute_velocities_face!(model)
    return _kinetic_energy_loss(model)
end

function loss_split(model)
    compute_velocities_split!(model)
    return _kinetic_energy_loss(model)
end

# Match `grad_loss!` in benchmarking/src/timestepping.jl: Enzyme reverse-mode
# with strong-zero semantics, Duplicated(model, dmodel) so the gradient is
# accumulated into the shadow.
function _grad!(loss_fn, model, dmodel)
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_fn, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel))
    return loss_value
end

grad_xyz!(model, dmodel)   = _grad!(loss_xyz,   model, dmodel)
grad_face!(model, dmodel)  = _grad!(loss_face,  model, dmodel)
grad_split!(model, dmodel) = _grad!(loss_split, model, dmodel)

#####
##### Driver
#####

const VARIANTS = (
    (name = "XYZ",   fn! = grad_xyz!,   loss = loss_xyz),
    (name = "FACE",  fn! = grad_face!,  loss = loss_face),
    (name = "SPLIT", fn! = grad_split!, loss = loss_split),
)

function build_model(; FT = Float64, Nx = 64, Ny = 64, Nz = 32)
    Reactant.set_default_backend("gpu")
    arch = ReactantState()
    Oceananigans.defaults.FloatType = FT
    dynamics  = CompressibleDynamics(ExplicitTimeStepping())
    advection = WENO(FT; order = 5)
    topology  = (Periodic, Bounded, Bounded)
    model = convective_boundary_layer(arch;
        float_type   = FT,
        Nx, Ny, Nz,
        dynamics,
        advection,
        closure      = nothing,
        microphysics = nothing,
        topology,
        simplified   = true,
    )
    return model
end

# Timing helper. Returns (compile_seconds, per_call_ms, primal_loss_value).
# Pattern mirrors benchmarking/src/utils.jl:80-138: compile, warmup, sync,
# time, sync. `nrepeat` controls how many timed invocations we average over.
function time_variant(name, grad_fn!, model, dmodel; nrepeat = 5)
    @info "[$name] compiling…"
    compile_start = time_ns()
    # Match benchmarking/src/utils.jl:92 — same Reactant flags as the real AD
    # benchmark so we're exercising the same compile pipeline.
    compiled = Reactant.@compile raise=true raise_first=true sync=true grad_fn!(model, dmodel)
    compile_seconds = (time_ns() - compile_start) / 1e9
    @info @sprintf("[%s]   compile time: %.2f s", name, compile_seconds)

    # Warmup. `sync=true` on the compiled call blocks until device work
    # completes, so we don't need a separate device synchronize here.
    @info "[$name] warmup…"
    primal = compiled(model, dmodel)

    # Time `nrepeat` invocations. dmodel accumulates the gradient across calls;
    # that's fine for timing because kernel work-per-call is independent of the
    # shadow's current values.
    @info "[$name] timing $nrepeat invocations…"
    times_ms = Float64[]
    for _ in 1:nrepeat
        t0 = time_ns()
        primal = compiled(model, dmodel)
        push!(times_ms, (time_ns() - t0) / 1e6)
    end

    @info @sprintf("[%s]   per-call: %.2f ± %.2f ms (min %.2f, max %.2f)",
                   name, mean(times_ms), std(times_ms),
                   minimum(times_ms), maximum(times_ms))

    return (; compile_seconds, times_ms, primal)
end

# Dump the post-Enzyme StableHLO module for `grad_fn!` to `path`. Helpful for
# diffing the IR neighborhoods of the divides across variants.
function dump_hlo(name, grad_fn!, model, dmodel; outdir = "/tmp")
    path = joinpath(outdir, "grad_$(lowercase(name)).mlir")
    @info "[$name] dumping HLO to $path"
    hlo = Reactant.@code_hlo raise=true raise_first=true grad_fn!(model, dmodel)
    open(path, "w") do io
        show(io, hlo)
    end
    return path
end

function main(; nrepeat = 5, dump_mlir = true)
    @info "=== compute_velocities! launch-strategy comparison ==="

    # Print resolved versions so we know which Enzyme/Reactant_jll we're on.
    deps = Pkg.dependencies()
    for (name,) in (("Enzyme",), ("Reactant",), ("Reactant_jll",),
                    ("Enzyme_jll",), ("GPUCompiler",), ("CUDA",))
        for (_uuid, info) in deps
            if info.name == name && info.version !== nothing
                @info "  $(info.name) $(info.version)"
                break
            end
        end
    end

    @info "Building model (compressible_explicit, PBB 64x64x32, WENO5, Float64, simplified)…"
    model = build_model()
    dmodel = Enzyme.make_zero(model)
    @info "Model built. Architecture: $(typeof(Oceananigans.Architectures.architecture(model.grid)))"

    # Compile + time each variant. `ReverseWithPrimal` returns the primal loss
    # from the compiled grad call; we log it as a sanity check — primals should
    # match across variants since all three write the same interior values
    # (any differences are confined to halo cells, which `interior(...)` skips).
    #
    # dmodel is NOT reset between variants: Enzyme accumulates gradients into
    # the shadow, but per-call kernel time doesn't depend on existing shadow
    # values. Mirrors benchmarking/src/utils.jl's pattern (the real AD
    # benchmark only zeroes dmodel once, before compilation).
    results = NamedTuple[]
    for v in VARIANTS
        r = time_variant(v.name, v.fn!, model, dmodel; nrepeat)
        @info @sprintf("[%s]   primal loss = %s", v.name, repr(r.primal))
        push!(results, (; v.name, r...))
    end

    # Summary table
    @info "=== Summary ==="
    @printf("%-8s %14s %14s %14s\n", "variant", "compile (s)", "min (ms)", "mean (ms)")
    println("-" ^ 56)
    for r in results
        @printf("%-8s %14.2f %14.2f %14.2f\n",
                r.name, r.compile_seconds,
                minimum(r.times_ms), mean(r.times_ms))
    end

    # MLIR dumps (optional; runs an extra Reactant.@code_hlo per variant)
    if dump_mlir
        @info "=== Dumping StableHLO modules ==="
        for v in VARIANTS
            path = dump_hlo(v.name, v.fn!, model, dmodel)
            @info "  $(v.name): $path"
        end
        @info "Diff with: diff -u /tmp/grad_xyz.mlir /tmp/grad_face.mlir | head -200"
        @info "           diff -u /tmp/grad_face.mlir /tmp/grad_split.mlir | head -200"
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
