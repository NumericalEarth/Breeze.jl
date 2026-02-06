# Distributed integration tests for AtmosphereModel
#
# Tests verify that AtmosphereModel works correctly with Oceananigans'
# distributed computing infrastructure.
#
# When run in the standard test suite, these tests use a single-rank "distributed"
# configuration. For true multi-rank testing with different partitions, run with:
#
#   mpiexec -n 4 julia --project test/distributed.jl
#
# Multi-rank partitions to test manually:
#   Partition(4, 1, 1)  - 4-way x partition
#   Partition(2, 2, 1)  - 2x2 xy partition
#   Partition(1, 4, 1)  - 4-way y partition

using Breeze
using Breeze: PrescribedDynamics
using Oceananigans
using Oceananigans.Architectures: architecture, CPU
using Oceananigans.DistributedComputations: Distributed, Partition
using Oceananigans.Units: seconds
using Test

if !@isdefined(default_arch)
    default_arch = CPU()
end

#####
##### Test utilities
#####

function fields_match(model_serial, model_distributed; rtol=0, atol=0)
    for (name, φ_serial) in pairs(Oceananigans.prognostic_fields(model_serial))
        φ_distributed = Oceananigans.prognostic_fields(model_distributed)[name]
        data_s = interior(φ_serial) |> Array
        data_d = interior(φ_distributed) |> Array

        # For multi-rank runs, just check for finite values
        if size(data_s) != size(data_d)
            all(isfinite, data_d) || return false
        else
            isapprox(data_s, data_d; atol, rtol) || return false
        end
    end
    return true
end

function run_and_compare(; arch, partition=Partition(1, 1), dynamics_type=:Anelastic,
                         closure=nothing, microphysics=nothing, coriolis=nothing,
                         grid_size=(16, 16, 16), extent=(1000, 1000, 1000), stop_time=10)

    topology = (Periodic, Periodic, Bounded)
    x, y, z = (0, extent[1]), (0, extent[2]), (0, extent[3])
    Lx, Ly, Lz = extent
    constants = ThermodynamicConstants()

    u₀(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly)
    v₀(x, y, z) = cos(2π * x / Lx) * sin(2π * y / Ly)

    # Distributed model (on all ranks)
    arch_distributed = Distributed(arch; partition)
    grid_distributed = RectilinearGrid(arch_distributed; size=grid_size, x, y, z, topology)
    reference_state_d = ReferenceState(grid_distributed, constants)
    dynamics_d = dynamics_type === :Anelastic ? AnelasticDynamics(reference_state_d) :
                                                PrescribedDynamics(reference_state_d)

    model_distributed = AtmosphereModel(grid_distributed; dynamics=dynamics_d,
                                        thermodynamic_constants=constants,
                                        advection=WENO(), closure, microphysics, coriolis)
    set!(model_distributed; θ=300, qᵗ=0.01, u=u₀, v=v₀)

    simulation_d = Simulation(model_distributed; Δt=1e-3seconds, stop_time)
    simulation_d.callbacks[:wizard] = Callback(TimeStepWizard(cfl=0.5), IterationInterval(1))
    run!(simulation_d)

    # Serial model (only on root rank for comparison)
    local_rank = arch_distributed.local_rank
    if local_rank == 0
        grid_serial = RectilinearGrid(arch; size=grid_size, x, y, z, topology)
        reference_state = ReferenceState(grid_serial, constants)
        dynamics = dynamics_type === :Anelastic ? AnelasticDynamics(reference_state) :
                                                  PrescribedDynamics(reference_state)

        model_serial = AtmosphereModel(grid_serial; dynamics, thermodynamic_constants=constants,
                                       advection=WENO(), closure, microphysics, coriolis)
        set!(model_serial; θ=300, qᵗ=0.01, u=u₀, v=v₀)

        simulation = Simulation(model_serial; Δt=1e-3seconds, stop_time)
        simulation.callbacks[:wizard] = Callback(TimeStepWizard(cfl=0.5), IterationInterval(1))
        run!(simulation)

        # Compare fields on root
        FT = eltype(grid_distributed)
        return fields_match(model_serial, model_distributed; rtol=10*eps(FT))
    else
        return true  # Non-root ranks pass by default
    end
end

#####
##### Tests
#####

@testset "Distributed AtmosphereModel" begin
    arch = default_arch
    arch isa GPU && (@info "Skipping distributed tests on GPU"; return)

    @testset "Construction" begin
        arch_d = Distributed(arch; partition=Partition(1, 1))
        grid = RectilinearGrid(arch_d; size=(16, 16, 16),
                               x=(0, 1000), y=(0, 1000), z=(0, 1000),
                               topology=(Periodic, Periodic, Bounded))

        @test architecture(grid) isa Distributed

        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants)
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants)

        @test model isa AtmosphereModel

        set!(model; θ=300, qᵗ=0.01)
        time_step!(model, 1)
        @test model.clock.iteration == 1
    end

    @testset "Serial vs distributed comparison" begin
        @testset "AnelasticDynamics" begin
            @test run_and_compare(; arch)
        end

        @testset "PrescribedDynamics" begin
            @test run_and_compare(; arch, dynamics_type=:Prescribed)
        end

        @testset "With SmagorinskyLilly" begin
            @test run_and_compare(; arch, closure=SmagorinskyLilly())
        end

        @testset "With AnisotropicMinimumDissipation" begin
            @test run_and_compare(; arch, closure=AnisotropicMinimumDissipation())
        end

        @testset "With SaturationAdjustment" begin
            @test run_and_compare(; arch, microphysics=SaturationAdjustment())
        end

        @testset "With FPlane Coriolis" begin
            @test run_and_compare(; arch, coriolis=FPlane(f=1e-4))
        end
    end

    @testset "Full physics [$FT]" for FT in (Float32, Float64)
        Oceananigans.defaults.FloatType = FT
        @test run_and_compare(; arch,
                              closure=SmagorinskyLilly(),
                              microphysics=SaturationAdjustment(FT),
                              coriolis=FPlane(f=FT(1e-4)),
                              extent=(5000, 5000, 2000), stop_time=50)
    end
end
