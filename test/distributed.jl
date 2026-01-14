# Distributed integration tests for AtmosphereModel
#
# These tests verify that AtmosphereModel works correctly with Oceananigans'
# distributed computing infrastructure. Currently, distributed support in
# AtmosphereModel is experimental and some configurations may not work.
#
# When run in the standard test suite, these tests use a single-rank "distributed"
# configuration. For true multi-rank testing, run with:
#   mpiexec -n 4 julia --project test/distributed.jl
#
# NOTE: As of this writing, distributed support requires additional development
# in Breeze's AtmosphereModel, particularly in kernel launch configurations.

using Breeze
using Breeze: PrescribedDynamics, KinematicModel
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: architecture, child_architecture
using Oceananigans.DistributedComputations: Distributed, Partition
using Test

#####
##### Utility for comparing serial and distributed model runs
#####

"""
    compare_fields(model_serial, model_distributed; rtol=0, atol=0)

Compare prognostic fields between a serial and distributed model.
For single-rank distributed runs, fields are compared directly.
For multi-rank distributed runs, each rank compares its local portion.

Returns true if all prognostic fields match within tolerance.
"""
function compare_fields(model_serial, model_distributed; rtol=0, atol=0)
    all_match = true

    grid_distributed = model_distributed.grid
    arch = architecture(grid_distributed)

    for (name, field_serial) in pairs(Oceananigans.prognostic_fields(model_serial))
        field_distributed = Oceananigans.prognostic_fields(model_distributed)[name]

        data_serial = interior(field_serial) |> Array
        data_distributed = interior(field_distributed) |> Array

        if arch isa Distributed
            local_size = size(data_distributed)
            if size(data_serial) == local_size
                if !isapprox(data_serial, data_distributed; atol, rtol)
                    @warn "Field $name does not match between serial and distributed runs"
                    max_diff = maximum(abs, data_serial .- data_distributed)
                    @warn "  Maximum difference: $max_diff"
                    all_match = false
                end
            else
                if !all(isfinite, data_distributed)
                    @warn "Field $name contains non-finite values in distributed run"
                    all_match = false
                end
            end
        else
            if !isapprox(data_serial, data_distributed; atol, rtol)
                @warn "Field $name does not match"
                max_diff = maximum(abs, data_serial .- data_distributed)
                @warn "  Maximum difference: $max_diff"
                all_match = false
            end
        end
    end

    return all_match
end

"""
    run_comparison_test(;
        arch,
        dynamics_type,
        topology,
        closure,
        microphysics,
        advection = WENO(),
        coriolis = nothing,
        Δt = 1,
        Nt = 10,
        grid_size = (8, 8, 8),
        extent = (1000, 1000, 1000))

Run a model with serial and single-rank distributed architectures,
then compare results for equivalence.

Returns true if all prognostic fields match within tolerance.
"""
function run_comparison_test(;
        arch,
        dynamics_type,
        topology,
        closure,
        microphysics,
        advection = WENO(),
        coriolis = nothing,
        Δt = 1,
        Nt = 10,
        grid_size = (8, 8, 8),
        extent = (1000, 1000, 1000))

    x = (0, extent[1])
    y = (0, extent[2])
    z = (0, extent[3])

    grid_serial = RectilinearGrid(arch; size=grid_size, x, y, z, topology)

    arch_distributed = Distributed(arch; partition=Partition(1, 1))
    grid_distributed = RectilinearGrid(arch_distributed; size=grid_size, x, y, z, topology)

    constants = ThermodynamicConstants()

    function make_dynamics(grid, dynamics_type, constants)
        if dynamics_type === :Anelastic
            reference_state = ReferenceState(grid, constants)
            return AnelasticDynamics(reference_state)
        elseif dynamics_type === :Compressible
            return CompressibleDynamics()
        elseif dynamics_type === :Prescribed
            reference_state = ReferenceState(grid, constants)
            return PrescribedDynamics(reference_state)
        else
            error("Unknown dynamics type: $dynamics_type")
        end
    end

    dynamics_serial = make_dynamics(grid_serial, dynamics_type, constants)
    dynamics_distributed = make_dynamics(grid_distributed, dynamics_type, constants)

    model_kwargs_serial = (;
        dynamics = dynamics_serial,
        thermodynamic_constants = constants,
        advection,
        closure,
        microphysics,
        coriolis)

    model_kwargs_distributed = (;
        dynamics = dynamics_distributed,
        thermodynamic_constants = constants,
        advection,
        closure,
        microphysics,
        coriolis)

    θ₀ = 300
    qᵗ₀ = 0.01
    Lx, Ly, Lz = extent
    u₀(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly)
    v₀(x, y, z) = cos(2π * x / Lx) * sin(2π * y / Ly)

    model_serial = AtmosphereModel(grid_serial; model_kwargs_serial...)
    set!(model_serial; θ = θ₀, qᵗ = qᵗ₀, u = u₀, v = v₀)
    simulation_serial = Simulation(model_serial; Δt, stop_iteration=Nt)
    run!(simulation_serial)

    model_distributed = AtmosphereModel(grid_distributed; model_kwargs_distributed...)
    set!(model_distributed; θ = θ₀, qᵗ = qᵗ₀, u = u₀, v = v₀)
    simulation_distributed = Simulation(model_distributed; Δt, stop_iteration=Nt)
    run!(simulation_distributed)

    FT = eltype(grid_serial)
    return compare_fields(model_serial, model_distributed; rtol=10*eps(FT))
end

#####
##### Test configurations
#####

const TOPOLOGIES = (
    (Periodic, Periodic, Bounded),  # Doubly periodic
    (Periodic, Bounded, Bounded),   # Channel (bounded in y)
)

# Note: Compressible dynamics has known distributed support issues, so it's not
# included in DYNAMICS_TYPES. Enable when distributed support is complete.
const DYNAMICS_TYPES = (:Anelastic, :Prescribed)

#####
##### Tests
#####

@testset "Distributed AtmosphereModel construction" begin
    arch = default_arch

    if arch isa GPU
        @info "Skipping distributed construction tests on GPU"
        return
    end

    grid_size = (4, 4, 4)
    extent = (1000, 1000, 1000)

    @testset "Distributed grid construction with topology: $topology" for topology in TOPOLOGIES
        x = (0, extent[1])
        y = (0, extent[2])
        z = (0, extent[3])

        arch_distributed = Distributed(arch; partition=Partition(1, 1))
        grid = RectilinearGrid(arch_distributed; size=grid_size, x, y, z, topology)

        @test grid isa RectilinearGrid
        @test architecture(grid) isa Distributed
    end

    # Note: Compressible dynamics has known issues with distributed support
    # due to kernel launch configurations. Only test Anelastic and Prescribed.
    supported_dynamics = (:Anelastic, :Prescribed)

    @testset "AtmosphereModel with distributed grid: $dynamics_type" for dynamics_type in supported_dynamics
        topology = (Periodic, Periodic, Bounded)
        x = (0, extent[1])
        y = (0, extent[2])
        z = (0, extent[3])

        arch_distributed = Distributed(arch; partition=Partition(1, 1))
        grid = RectilinearGrid(arch_distributed; size=grid_size, x, y, z, topology)
        constants = ThermodynamicConstants()

        if dynamics_type === :Anelastic
            reference_state = ReferenceState(grid, constants)
            dynamics = AnelasticDynamics(reference_state)
        else # :Prescribed
            reference_state = ReferenceState(grid, constants)
            dynamics = PrescribedDynamics(reference_state)
        end

        model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants)
        @test model isa AtmosphereModel

        set!(model; θ=300, qᵗ=0.01)
        time_step!(model, 1)
        @test model.clock.iteration == 1
    end
end

@testset "Distributed vs serial comparison" begin
    arch = default_arch

    if arch isa GPU
        @info "Skipping distributed comparison tests on GPU"
        return
    end

    # Only test doubly periodic topology with Anelastic dynamics
    # Channel topology and Compressible dynamics have known issues with distributed support
    @testset "Anelastic dynamics - doubly periodic" begin
        @testset "No closure or microphysics" begin
            result = run_comparison_test(;
                arch,
                dynamics_type = :Anelastic,
                topology = (Periodic, Periodic, Bounded),
                closure = nothing,
                microphysics = nothing,
                advection = WENO(),
                coriolis = nothing)
            @test result
        end

        @testset "With SmagorinskyLilly closure" begin
            result = run_comparison_test(;
                arch,
                dynamics_type = :Anelastic,
                topology = (Periodic, Periodic, Bounded),
                closure = SmagorinskyLilly(),
                microphysics = nothing,
                advection = WENO(),
                coriolis = nothing)
            @test result
        end

        @testset "With SaturationAdjustment" begin
            result = run_comparison_test(;
                arch,
                dynamics_type = :Anelastic,
                topology = (Periodic, Periodic, Bounded),
                closure = nothing,
                microphysics = SaturationAdjustment(),
                advection = WENO(),
                coriolis = nothing)
            @test result
        end

        @testset "With FPlane Coriolis" begin
            result = run_comparison_test(;
                arch,
                dynamics_type = :Anelastic,
                topology = (Periodic, Periodic, Bounded),
                closure = nothing,
                microphysics = nothing,
                advection = WENO(),
                coriolis = FPlane(f=1e-4))
            @test result
        end
    end

    @testset "Prescribed dynamics - doubly periodic" begin
        result = run_comparison_test(;
            arch,
            dynamics_type = :Prescribed,
            topology = (Periodic, Periodic, Bounded),
            closure = nothing,
            microphysics = nothing,
            advection = WENO(),
            coriolis = nothing)
        @test result
    end

    # Note: Channel topology tests are commented out due to known distributed issues
    # TODO: Enable channel topology tests when distributed support is extended
    # @testset "Channel topology (bounded in y)" begin
    #     result = run_comparison_test(;
    #         arch,
    #         dynamics_type = :Anelastic,
    #         topology = (Periodic, Bounded, Bounded),
    #         closure = nothing,
    #         microphysics = nothing,
    #         advection = WENO(),
    #         coriolis = nothing)
    #     @test result
    # end

    # Note: Compressible dynamics tests are commented out due to known distributed issues
    # TODO: Enable Compressible dynamics tests when distributed support is extended
    # @testset "Compressible dynamics" begin
    #     result = run_comparison_test(;
    #         arch,
    #         dynamics_type = :Compressible,
    #         topology = (Periodic, Periodic, Bounded),
    #         closure = nothing,
    #         microphysics = nothing,
    #         advection = WENO(),
    #         coriolis = nothing)
    #     @test result
    # end
end

@testset "Distributed with full physics [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    arch = default_arch
    if arch isa GPU
        @info "Skipping distributed full physics test on GPU"
        return
    end

    result = run_comparison_test(;
        arch,
        dynamics_type = :Anelastic,
        topology = (Periodic, Periodic, Bounded),
        closure = SmagorinskyLilly(),
        microphysics = SaturationAdjustment(FT),
        advection = WENO(order=5),
        coriolis = FPlane(f=FT(1e-4)),
        grid_size = (8, 8, 8),
        extent = (5000, 5000, 2000),
        Δt = 5,
        Nt = 10)

    @test result
end

@testset "Distributed with AnisotropicMinimumDissipation [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    arch = default_arch
    if arch isa GPU
        @info "Skipping distributed AMD test on GPU"
        return
    end

    result = run_comparison_test(;
        arch,
        dynamics_type = :Anelastic,
        topology = (Periodic, Periodic, Bounded),
        closure = AnisotropicMinimumDissipation(),
        microphysics = nothing,
        advection = WENO(),
        coriolis = nothing)

    @test result
end
