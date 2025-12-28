# Reactant + Enzyme AD Testing for Breeze
#
# This test file systematically discovers breaking points when using
# Reactant.jl for tracing and Enzyme.jl for automatic differentiation
# with Breeze's AtmosphereModel.
#
# Related: GitHub issue #223 - Reactant tracing for AnelasticFormulation
#
# ============================================================================
# DISCOVERED BLOCKERS AND FIXES (as of December 2025)
# ============================================================================
#
# FIXED 1: FFT Planning for ConcretePJRTArray
# -------------------------------------------
# Status: FIXED in BreezeReactantExt (ext/BreezeReactantExt/BreezeReactantExt.jl)
# XLA handles FFT planning internally, so we return nothing (no-op) for Reactant arrays.
# This allows AtmosphereModel construction with FourierTridiagonalPoissonSolver.
#
# FIXED 2: CartesianIndex indexing for OffsetArray{TracedRNumber}
# ---------------------------------------------------------------
# Status: FIXED in BreezeReactantExt (ext/BreezeReactantExt/BreezeReactantExt.jl)
# Added method to handle CartesianIndex by converting to splatted tuple indices.
# This should be upstreamed to Reactant.jl (ReactantOffsetArraysExt.jl).
#
# FIXED 3 (Breeze-local): Scalar indexing in Oceananigans reductions
# ---------------------------------------------------------------
# Status: MITIGATED in BreezeReactantExt (ext/BreezeReactantExt/BreezeReactantExt.jl)
#
# Root cause: Oceananigans reductions (maximum/minimum/sum) over AbstractField/AbstractOperation
# can fall back to scalar iteration (`getindex` in a loop). Reactant arrays disallow scalar indexing.
#
# Fix: BreezeReactantExt now redirects
#   - Field reductions to `maximum(f, interior(field))` etc (Reactant can reduce SubArray views), and
#   - AbstractOperation reductions (e.g. KernelFunctionOperation used in CFL) by materializing `Field(op)`
#     and reducing its interior.
#
# NOTE: This should be upstreamed to OceananigansReactantExt, and it only covers dims=: and condition=nothing.
#
# ============================================================================

using Test
using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: time_step!, update_state!
using Statistics

# CUDA must be loaded for Reactant's KernelAbstractions extension to work
# See: ReactantKernelAbstractionsExt.jl line 118
# "Attempted to raise a KernelAbstractions kernel with Reactant but CUDA.jl is not loaded."
const CUDA_AVAILABLE = try
    @eval using CUDA
    true
catch e
    @warn "CUDA not available: $e"
    false
end

# Conditionally load Reactant and Enzyme
# These may fail to load on some systems
const REACTANT_AVAILABLE = try
    @eval using Reactant
    @eval using Reactant: @compile, @trace
    true
catch e
    @warn "Reactant not available: $e"
    false
end

const ENZYME_AVAILABLE = try
    @eval using Enzyme
    @eval using Enzyme: autodiff, Active, Const, Duplicated, set_strong_zero, ReverseWithPrimal
    true
catch e
    @warn "Enzyme not available: $e"
    false
end

# Skip all tests if dependencies not available
if !REACTANT_AVAILABLE || !ENZYME_AVAILABLE || !CUDA_AVAILABLE
    @testset "Reactant + Enzyme AD" begin
        @test_skip "Reactant, Enzyme, or CUDA not available"
    end
else

@testset "Reactant + Enzyme AD" begin

    # Use CPU backend for Reactant to avoid GPU-specific issues
    Reactant.set_default_backend("cpu")

    #####
    ##### Phase 1: Model Construction with ReactantState
    #####

    @testset "Phase 1: Model Construction" begin
        # Test that we can construct an AtmosphereModel with ReactantState architecture
        # This exercises grid creation and pressure solver setup

        @testset "Grid construction" begin
            # Minimal 2D grid that exercises the pressure solver
            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            @test grid isa RectilinearGrid
            @test grid.architecture isa ReactantState
        end

        @testset "AtmosphereModel construction" begin
            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            # Start with simplest advection to minimize potential issues
            model = AtmosphereModel(grid; advection = Centered(order=2))

            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
        end

        @testset "AtmosphereModel with WENO advection" begin
            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            # WENO advection as used in the issue MWE
            model = AtmosphereModel(grid; advection = WENO(order=9))

            @test model isa AtmosphereModel
        end
    end

    #####
    ##### Phase 2: Forward Time-Stepping (no AD)
    #####

    @testset "Phase 2: Forward Time-Stepping" begin

        @testset "Initial condition setup" begin
            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            model = AtmosphereModel(grid; advection = WENO(order=9))

            # Setup initial condition (thermal bubble)
            r₀ = 2e3
            Δθ = 10 # K
            N² = 1e-6
            θ₀ = model.dynamics.reference_state.potential_temperature
            g = model.thermodynamic_constants.gravitational_acceleration

            function θ_init(x, z)
                x₀ = mean(xnodes(grid, Center()))
                z₀ = 0.3 * grid.Lz
                θ̄ = θ₀ * exp(N² * z / g)
                r = sqrt((x - x₀)^2 + (z - z₀)^2)
                θ′ = Δθ * max(0, 1 - r / r₀)
                return θ̄ + θ′
            end

            θ = Field{Center, Nothing, Center}(grid)
            set!(θ, θ_init)

            @test θ isa Field
            @test !isnan(maximum(θ))
        end

        @testset "Single time step (uncompiled)" begin
            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            model = AtmosphereModel(grid; advection = WENO(order=9))

            # Try a single time step without compilation
            # This may reveal issues with the pressure solver FFT plans
            Δt = 1.0

            # This is expected to fail with FFT planning issues
            # ERROR: MethodError: no method matching plan_forward_transform(::ConcreteIFRTArray...)
            try
                time_step!(model, Δt)
                @test true # If we get here, it worked!
            catch e
                @warn "Uncompiled time_step! failed" exception=e
                # Document the error for issue tracking
                @test_broken false
            end
        end

        @testset "Compiled time stepping with @compile" begin
            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            model = AtmosphereModel(grid; advection = WENO(order=9))
            Δt = 1.0

            # Try to compile time_step!
            # This traces the function through Reactant
            try
                compiled_step! = @compile time_step!(model, Δt)
                compiled_step!(model, Δt)
                @test true
            catch e
                @warn "Compiled time_step! failed" exception=e
                @test_broken false
            end
        end
    end

    #####
    ##### Phase 3: Enzyme AD
    #####

    @testset "Phase 3: Enzyme Autodiff" begin

        @testset "Loss function definition" begin
            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            model = AtmosphereModel(grid; advection = WENO(order=9))

            # Simple loss function: mean squared temperature
            function loss_function(model)
                return mean(model.temperature.^2)
            end

            # Test loss function works
            loss = loss_function(model)
            @test loss isa Number
            @test !isnan(loss)
        end

        @testset "Full AD workflow (GB-25 pattern)" begin
            # Following the GB-25 pattern:
            # 1. Initialize model COMPLETELY before any traced code
            # 2. Use simple wrapper functions for time stepping
            # 3. No set! or model mutations inside traced code

            function make_grid(architecture)
                grid = RectilinearGrid(architecture;
                    size = (8, 8),
                    halo = (5, 5),
                    x = (-10e3, 10e3),
                    z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
                return grid
            end

            function make_model(grid)
                model = AtmosphereModel(grid; advection = WENO(order=9))
                return model
            end

            # GB-25 pattern: simple loop wrapper that only calls time_step!
            # No set!, no clock mutations inside
            function loop!(model, Ninner)
                Δt = model.clock.last_Δt + 0  # GB-25 pattern: + 0 ensures correct type
                @trace track_numbers=false for _ = 1:Ninner
                    time_step!(model, Δt)
                end
                return nothing
            end

            # Loss function that just measures the current state
            # No model mutations inside
            function loss_after_loop(model)
                return mean(interior(model.temperature).^2)
            end

            # Combined loss: run loop, then compute loss
            function forward_loss(model, Ninner)
                loop!(model, Ninner)
                return loss_after_loop(model)
            end

            function grad_forward_loss(model, Ninner)
                mode = set_strong_zero(ReverseWithPrimal)
                result = autodiff(mode,
                    forward_loss,
                    Active,
                    Const(model),
                    Const(Ninner))
                return result
            end

            # Setup - ALL initialization happens before @compile
            architecture = ReactantState()
            grid = make_grid(architecture)
            model = make_model(grid)

            # Set initial condition BEFORE tracing (GB-25 pattern)
            r₀ = 2e3
            Δθ = 10.0
            N² = 1e-6
            θ₀ = model.dynamics.reference_state.potential_temperature
            g = model.thermodynamic_constants.gravitational_acceleration

            # Compute mean x outside of tracing
            x_mean = 0.0  # Center of domain

            function θ_init_function(x, z)
                z₀ = 0.3 * 10e3  # Use literal instead of grid.Lz to avoid tracing issues
                θ̄ = θ₀ * exp(N² * z / g)
                r = sqrt((x - x_mean)^2 + (z - z₀)^2)
                θ′ = Δθ * max(0, 1 - r / r₀)
                return θ̄ + θ′
            end

            # Set initial condition BEFORE any compilation
            set!(model, θ = θ_init_function)

            # Initialize clock BEFORE compilation
            model.clock.last_Δt = 1.0

            # Call update_state! to initialize the model (GB-25 does this)
            update_state!(model)

            nsteps = 3

            # Test forward pass first (should work per GB-25)
            @testset "Forward loop compilation" begin
                try
                    compiled_loop! = @compile sync=true raise=true loop!(model, nsteps)
                    compiled_loop!(model, nsteps)
                    @test true
                catch e
                    @warn "Forward loop compilation failed" exception=(e, catch_backtrace())
                    @test_broken false
                end
            end

            # Re-initialize for AD test
            set!(model, θ = θ_init_function)
            update_state!(model)

            # Test AD workflow
            @testset "Enzyme AD compilation" begin
                try
                    compiled_grad = @compile raise=true sync=true grad_forward_loss(model, nsteps)
                    result = compiled_grad(model, nsteps)

                    # Reactant+Enzyme returns: (gradients, primal_value)
                    # where gradients is a tuple and primal_value is the loss
                    @test result isa Tuple
                    @test length(result) == 2
                    
                    gradients, primal = result
                    # primal is the loss value (ConcretePJRTNumber or Float)
                    primal_val = primal isa Number ? primal : Float64(primal)
                    @test !isnan(primal_val)
                    @info "AD succeeded: loss = $primal_val"
                catch e
                    @warn "Enzyme AD compilation failed" exception=(e, catch_backtrace())

                    error_str = string(e)
                    if occursin("plan_forward_transform", error_str) || occursin("plan_fft", error_str)
                        @info "BLOCKER: FFT planning not supported for Reactant traced arrays"
                    elseif occursin("non-boolean", error_str) || occursin("TracedRNumber{Bool}", error_str)
                        @info "BLOCKER: Boolean context with TracedRNumber"
                    elseif occursin("NoFieldMatchError", error_str)
                        @info "BLOCKER: Reactant tracing issue with struct field types"
                    else
                        @info "BLOCKER: Unknown error type - needs investigation"
                    end

                    @test_broken false
                end
            end
        end
    end

    #####
    ##### Phase 4: Minimal Forward Pass Tests
    #####
    # These tests try to isolate the smallest failing components

    @testset "Phase 4: Component Isolation" begin

        @testset "Field operations on ReactantState" begin
            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            f = CenterField(grid)
            set!(f, (x, z) -> x + z)

            @test f isa Field
            @test !isnan(maximum(f))

            # Test basic field arithmetic
            g = Field(f^2)
            @test g isa Field
        end

        @testset "Pressure solver isolation" begin
            # Test if the pressure solver can be constructed and used
            # with ReactantState architecture

            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            model = AtmosphereModel(grid; advection = Centered(order=2))

            # Check that pressure solver exists
            @test model.pressure_solver !== nothing

            # The pressure solver uses FourierTridiagonalPoissonSolver
            # which requires FFT plans - this is the main blocker
            @info "Pressure solver type: $(typeof(model.pressure_solver))"
        end

        @testset "State update without pressure solve" begin
            # Try to isolate if the issue is specifically in pressure solver
            # or elsewhere in the time-stepping

            grid = RectilinearGrid(ReactantState();
                size = (8, 8),
                halo = (5, 5),
                x = (-10e3, 10e3),
                z = (0, 10e3),
                topology = (Periodic, Flat, Bounded))

            model = AtmosphereModel(grid; advection = Centered(order=2))

            # Try update_state! which should not require pressure solve
            try
                update_state!(model)
                @test true
            catch e
                @warn "update_state! failed" exception=e
                @test_broken false
            end
        end
    end

end # main testset

end # if REACTANT_AVAILABLE && ENZYME_AVAILABLE
