include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant compilation tests — shared setup
#####
#
# `run_compilation_tests` builds an `AtmosphereModel` on `ReactantState` for each
# topology, with the given dynamics and advection scheme. Phase structure per topology:
#   (a)   Build model on ReactantState
#   (b)   Compile + raise backward (Enzyme reverse mode)
#   (c)   FD validation of AD gradients

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using GPUArraysCore: @allowscalar
using Statistics: mean
using Test
using CUDA

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Configurations
#####

centered_topologies = [
    ("Periodic, Periodic, Flat",     (Periodic, Periodic, Flat),     2),
    ("Bounded, Bounded, Flat",       (Bounded,  Bounded,  Flat),     2),
    ("Periodic, Periodic, Periodic", (Periodic, Periodic, Periodic), 3),
    ("Periodic, Bounded, Bounded",   (Periodic, Bounded,  Bounded),  3),
    ("Bounded, Bounded, Bounded",    (Bounded,  Bounded,  Bounded),  3),
]

weno_topologies = [
    ("Periodic, Bounded, Bounded",  (Periodic, Bounded,  Bounded), 3),
    ("Periodic, Periodic, Bounded", (Periodic, Periodic, Bounded), 3),
    ("Bounded, Bounded, Bounded",   (Bounded,  Bounded,  Bounded), 3),
]

#####
##### Helpers
#####

function make_grid(topo, nd; arch=ReactantState(), halo=nothing)
    sz  = nd == 2 ? (20, 20)   : (20, 20, 20)
    ext = nd == 2 ? (1e3, 1e3) : (1e3, 1e3, 1e3)
    halo_kw = isnothing(halo) ? (;) : (; halo = ntuple(_ -> halo, nd))
    return RectilinearGrid(arch; size=sz, extent=ext, topology=topo, halo_kw...)
end

make_dynamics(::Type{CompressibleDynamics}, grid) = CompressibleDynamics()
make_dynamics(::Type{AnelasticDynamics}, grid) = AnelasticDynamics(ReferenceState(grid))

get_temperature(model) = Array(interior(model.temperature))

function make_init_fields(grid)
    θ_init  = CenterField(grid); set!(θ_init,  (args...) -> 300.0)
    dθ_init = CenterField(grid); set!(dθ_init, 0)
    return θ_init, dθ_init
end

function initial_density(model)
    FT = eltype(model.grid)
    ref = model.dynamics.reference_state
    return isnothing(ref) ? one(FT) : ref.density
end

# Compressible dynamics also sets its prognostic density; the anelastic density is the
# reference state's.
initial_conditions(::CompressibleDynamics, model, θ) = (; θ, ρ=initial_density(model))
initial_conditions(::AnelasticDynamics, model, θ) = (; θ)

function loss(model, θ_init, Δt, Nsteps)
    set!(model; initial_conditions(model.dynamics, model, θ_init)...)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, Nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(Nsteps))
    return dθ_init, loss_value
end

#####
##### Tests
#####

"""
    run_compilation_tests(name; dynamics=CompressibleDynamics, advection=nothing,
                          topologies=centered_topologies, halo=nothing)

Run the build, raise-backward and FD-validation phases in a testset called `name`, for an
`AtmosphereModel` with `dynamics` (a type `make_dynamics` knows) on each of `topologies`.
`advection = nothing` keeps the model's default scheme, and `halo = nothing` the grid's
default halo.
"""
function run_compilation_tests(name; dynamics=CompressibleDynamics, advection=nothing,
                               topologies=centered_topologies, halo=nothing)
    advection_kw = isnothing(advection) ? (;) : (; advection = advection)
    make_model(grid) = AtmosphereModel(grid; dynamics=make_dynamics(dynamics, grid), advection_kw...)

    @testset "$name" begin
        Δt = 0.02

        @testset "$label" for (label, topo, nd) in topologies
            # Force release of memory
            GC.gc(true); GC.gc(false); GC.gc(true)

            grid = make_grid(topo, nd; halo = halo)

            # ── Build ──
            @testset "Build" begin
                model = make_model(grid)
                @test model isa AtmosphereModel
                @test model.dynamics isa dynamics

                set!(model; initial_conditions(model.dynamics, model, 300.0)...)
                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end

            # Reconstruct for backward + FD phases
            model = make_model(grid)

            θ_init, dθ_init = make_init_fields(grid)
            dmodel = Enzyme.make_zero(model)
            Ns = 1

            compiled_grad = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                model, dmodel, θ_init, dθ_init, Δt, Ns)
            dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
            ad_grad = @allowscalar Array(interior(dθ))

            # ── Raise backward ──
            @testset "Raise backward" begin
                @test loss_val > 0
                @test isfinite(loss_val)
                @test maximum(abs, ad_grad) > 0
                @test !any(isnan, ad_grad)
            end

            # ── FD validation ──
            # Verify AD gradients against one-sided finite differences:
            #   ∂J/∂θ(i,j,k) ≈ (J(θ + ε·eᵢⱼₖ) - J(θ)) / ε
            # Checked at two grid cells and two step sizes to confirm
            # convergence is not an artifact of a particular ε.
            @testset "FD validation" begin
                grid_fd = make_grid(topo, nd; arch=default_arch, halo=halo)

                θ₀_fd = CenterField(grid_fd); set!(θ₀_fd, (args...) -> 300.0)
                J₀ = loss(make_model(grid_fd), θ₀_fd, Δt, Ns)

                test_cells = nd == 2 ? [(1,1,1), (4,4,1)] : [(1,1,1), (4,4,4)]

                for ε in (1e-4, 1e-6), (ic, jc, kc) in test_cells
                    @testset let ε=ε, (ic, jc, kc)=(ic, jc, kc)
                        θ_fd = CenterField(grid_fd); set!(θ_fd, (args...) -> 300.0)
                        @allowscalar interior(θ_fd, ic, jc, kc)[] += ε
                        J₊ = loss(make_model(grid_fd), θ_fd, Δt, Ns)
                        fd = (J₊ - J₀) / ε
                        ad = ad_grad[ic, jc, kc]
                        @test ad ≈ fd rtol=0.001
                    end
                end
            end
        end
    end
end

run_weno_tests(scheme_label, scheme; kw...) =
    run_compilation_tests("reactant_weno_compilation - $scheme_label";
                          advection=scheme, topologies=weno_topologies, halo=5, kw...)
