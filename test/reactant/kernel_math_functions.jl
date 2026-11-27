include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant raises the kernels of the closures to StableHLO, and a math function that the GPU compiler lowers to a
##### library call it cannot raise (`hypot`, as `__nv_hypot`) fails the whole compilation. The math functions of the
##### kernels are compiled here one by one, and the forward and the gradient of a column with
##### `TKEBasedTurbulenceClosure` and with `SimplifiedHigherOrderClosure`.
#####

using Breeze
using Breeze.Utils: hypotenuse
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Reactant
using Reactant: @trace, CompileOptions
using Enzyme
using SpecialFunctions: erf, erfc
using GPUArraysCore: @allowscalar
using Test
using CUDA

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

# As in tke_closure_components.jl: XLA's CPU compiler moves the small time-step loops of a short column into one function
# whose gradient takes hours and tens of GB to optimize; a threshold of zero keeps them as loops
small_while_loop_options = (; xla_backend_extra_options = Dict("xla_cpu_small_while_loop_byte_threshold" => "0"))
compile_options = CompileOptions(; raise = true, raise_first = true, sync = true,
                                 xla_debug_options = default_arch isa GPU ? (;) : small_while_loop_options)

# The math functions of Breeze's kernels: the hypotenuse of Breeze.Utils in place of `hypot`, and those of the
# microphysics, the acoustic substeps and the energetics
kernel_math_functions = (hypotenuse = hypotenuse, cbrt = x -> cbrt(x), expm1 = x -> expm1(x), log1p = x -> log1p(x),
                         erf = x -> erf(x), erfc = x -> erfc(x), log10 = x -> log10(x), exp10 = x -> exp10(x))

@inline math_function_valueᶜᶜᶜ(i, j, k, grid, f, x, y) = @inbounds apply_math_function(f, x[i, j, k], y[i, j, k])
apply_math_function(f::typeof(hypotenuse), x, y) = f(x, y)
apply_math_function(f, x, y) = f(x)

# A field computed by a kernel that applies `f`
math_function_field(grid, f, x, y) = Field(KernelFunctionOperation{Center, Center, Center}(math_function_valueᶜᶜᶜ, grid, f, x, y))

@testset "Kernel math functions compile under Reactant [$name]" for (name, f) in pairs(kernel_math_functions)
    function result(arch)
        grid = RectilinearGrid(arch; size = 8, z = (0, 8), topology = (Flat, Flat, Bounded))
        x, y = CenterField(grid), CenterField(grid)
        set!(x, z -> 1e-3 + z / 8)
        set!(y, z -> 2 - z / 8)
        return math_function_field(grid, f, x, y)
    end
    cpu = result(CPU())
    compute!(cpu)

    reactant = result(ReactantState())
    compiled = Reactant.@compile compile_options=compile_options compute!(reactant)
    compiled(reactant)
    @test Array(interior(reactant)) ≈ Array(interior(cpu)) rtol = 1e-12
end

# The hypotenuse is `hypot` to rounding, and does not overflow or underflow in the squares
@testset "hypotenuse against hypot" begin
    for (a, b) in ((3.0, 4.0), (1e-3, 2.0), (-5.0, 1e-300), (1e300, 1e300), (1e-300, 3e-300), (0.0, 0.0), (0.0, -2.0),
                   (Inf, 1.0), (Inf, Inf), (2f0, 1f-20))
        @test hypotenuse(a, b) ≈ hypot(a, b) rtol = 4eps(typeof(a))
    end
    ∂a = Enzyme.autodiff(Enzyme.Reverse, hypotenuse, Enzyme.Active, Enzyme.Active(0.0), Enzyme.Active(0.0))[1][1]
    @test isfinite(∂a)
end

function column_loss(model, θ_init, Δt, Nsteps)
    set!(model; θ=θ_init, qᵗ=0.012, u=5, ρe=0.1)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return sum(interior(model.tracers.ρe))
end

function grad_column_loss(model, dmodel, θ_init, dθ_init, Δt, Nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), column_loss, Enzyme.Active,
                                    Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(θ_init, dθ_init),
                                    Enzyme.Const(Δt), Enzyme.Const(Nsteps))
    return dθ_init, loss_value
end

@testset "Reactant forward and gradient of a column [$label]" for (label, closure) in
        (("TKEBasedTurbulenceClosure", TKEBasedTurbulenceClosure()), ("SimplifiedHigherOrderClosure", SimplifiedHigherOrderClosure()))
    Δt = 1.0
    Ns = 2
    grid = RectilinearGrid(ReactantState(); size = 8, z = (0, 2000), topology = (Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure = 101325, potential_temperature = 290)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    initial_θ() = (θ = CenterField(grid); set!(θ, z -> 290 + 3e-3 * z); θ)

    model = AtmosphereModel(grid; dynamics, microphysics, closure)
    θ_init = initial_θ()
    compiled = @with_stack_size Reactant.@compile compile_options=compile_options column_loss(model, θ_init, Δt, Ns)
    value = @with_stack_size compiled(model, θ_init, Δt, Ns)
    @test isfinite(value) && value > 0

    model = AtmosphereModel(grid; dynamics, microphysics, closure)
    θ_init = initial_θ()
    dθ_init = CenterField(grid); set!(dθ_init, 0)
    dmodel = Enzyme.make_zero(model)
    compiled_grad = @with_stack_size Reactant.@compile compile_options=compile_options grad_column_loss(
        model, dmodel, θ_init, dθ_init, Δt, Ns)
    dθ, loss_value = @with_stack_size compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
    gradient = @allowscalar Array(interior(dθ))
    @test isfinite(loss_value)
    @test all(isfinite, gradient)
    @test any(!iszero, gradient)
end
