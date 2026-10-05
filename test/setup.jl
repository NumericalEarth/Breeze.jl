import Breeze
import CUDA
import Oceananigans
using GPUArraysCore: @allowscalar
using Oceananigans.Architectures: CPU, GPU
using Test: @inferred, @test

if get(ENV, "BREEZE_ENSURE_CUDA_FUNCTIONAL", "") == "true"
    CUDA.functional() || error("CUDA is not functional but we expect it to be, make sure it's set up correctly")
end

const default_arch = CUDA.functional() ? GPU() : CPU()

# Float type helpers for tests
# Default: Float64 only. Set BREEZE_TEST_FLOAT32=true to also test Float32.
function test_float_types()
    if get(ENV, "BREEZE_TEST_FLOAT32", "false") == "true"
        return (Float32, Float64)
    else
        return (Float64,)
    end
end

# Returns both Float32 and Float64 for tests that need both precision levels
all_float_types() = (Float32, Float64)

# Work around for <https://github.com/JuliaLang/julia/issues/54998>, often seen
# with Reactant code (and often only on the CI machine).
macro with_stack_size(stack_size, expr)
    return quote
        local _size = $(esc(stack_size))
        _size isa Integer || error("Stack size must be an integer")
        local task = Task(() -> $(esc(expr)), _size)
        schedule(task)
        wait(task)
        fetch(task)
    end
end

macro with_stack_size(expr)
    return :(@with_stack_size 16 << 20 $(esc(expr)))
end

# Check that the pointwise functions called inside the `AtmosphereModel` kernels (auxiliary
# thermodynamic variables, microphysical tendencies, momentum and moisture tendencies) are
# inferred at cell `(i, j, k)`. This mirrors the argument assembly in
# `compute_auxiliary_variables!` and `compute_tendencies!`: keep them in sync.
function test_kernel_functions_inferred(model; i=2, j=2, k=2)
    AM = Breeze.AtmosphereModels
    FT = eltype(model.grid)
    grid = model.grid
    dynamics = model.dynamics
    microphysics = model.microphysics
    constants = model.thermodynamic_constants
    μ_fields = model.microphysical_fields
    qᵛᵉ_field = AM.specific_prognostic_moisture(model)
    velocities = AM.transport_velocities(model)
    model_fields = Oceananigans.fields(model)
    moist_name = AM.moisture_prognostic_name(microphysics)

    momentum_args = (AM.dynamics_density(dynamics), model.advection.momentum, model.velocities,
                     model.closure, model.closure_fields, AM.advecting_momentum(model),
                     model.coriolis, model.clock, model_fields)

    w_args = (momentum_args..., model.forcing.ρw, dynamics, model.formulation, model.temperature,
              qᵛᵉ_field, microphysics, μ_fields, constants)

    common_args = (dynamics, model.formulation, constants, qᵛᵉ_field, velocities, microphysics,
                   μ_fields, model.closure, model.closure_fields, model.clock, model_fields)

    ρq_args = (qᵛᵉ_field, Val(2), Val(moist_name), model.forcing[moist_name],
               model.advection[moist_name], common_args...)

    @allowscalar begin
        ρ = AM.total_density(dynamics)[i, j, k]
        qᵛᵉ = qᵛᵉ_field[i, j, k]
        q = @inferred AM.grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, μ_fields)
        𝒰₀ = @inferred AM.diagnose_thermodynamic_state(i, j, k, grid, model.formulation, dynamics, q)
        𝒰₁ = @inferred AM.maybe_adjust_thermodynamic_state(𝒰₀, microphysics, qᵛᵉ, constants)
        @test @inferred(Breeze.Thermodynamics.temperature(𝒰₁, constants)) isa FT

        ℳ = @inferred AM.grid_microphysical_state(i, j, k, grid, microphysics, μ_fields, ρ, 𝒰₁, velocities)
        for name in (moist_name, AM.prognostic_field_names(microphysics)...)
            @test @inferred(AM.microphysical_tendency(microphysics, Val(name), ρ, ℳ, 𝒰₁, constants)) isa FT
        end

        @test @inferred(AM.x_momentum_tendency(i, j, k, grid, momentum_args..., model.forcing.ρu, dynamics)) isa FT
        @test @inferred(AM.y_momentum_tendency(i, j, k, grid, momentum_args..., model.forcing.ρv, dynamics)) isa FT
        @test @inferred(AM.z_momentum_tendency(i, j, k, grid, w_args...)) isa FT
        @test @inferred(AM.scalar_tendency(i, j, k, grid, ρq_args...)) isa FT
    end

    return nothing
end
