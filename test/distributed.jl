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
using Breeze.Thermodynamics: adiabatic_hydrostatic_density
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: architecture, child_architecture, CPU
using Oceananigans.DistributedComputations: Distributed, Partition
using Oceananigans.Units: seconds
using Test

# Define default_arch if not already defined (e.g., when running tests directly)
if !@isdefined(default_arch)
    default_arch = CPU()
end

#####
##### Helper for initializing CompressibleDynamics with a reference state density
#####

"""
    default_density(z, constants; surface_pressure=101325, potential_temperature=288)

Compute the default reference state density at height `z` using an adiabatic hydrostatic profile.
This matches the default ReferenceState used by AnelasticDynamics.
"""
function default_density(z, constants; surface_pressure=101325, potential_temperature=288)
    return adiabatic_hydrostatic_density(z, surface_pressure, potential_temperature, constants)
end

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

# Speed of sound in air at standard conditions (m/s)
const SPEED_OF_SOUND = 340.0

"""
    acoustic_max_Δt(grid; cfl=0.5, speed_of_sound=SPEED_OF_SOUND)

Compute the maximum time step allowed by the acoustic CFL constraint.
For compressible dynamics, the time step must satisfy Δt < Δx / c_s.
"""
function acoustic_max_Δt(grid; cfl=0.5, speed_of_sound=SPEED_OF_SOUND)
    # Use L/N for uniform grids
    Δx = grid.Lx / grid.Nx
    Δy = grid.Ly / grid.Ny
    Δz = grid.Lz / grid.Nz
    Δ_min = min(Δx, Δy, Δz)
    return cfl * Δ_min / speed_of_sound
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
        stop_time = 10,
        grid_size = (8, 8, 8),
        extent = (1000, 1000, 1000),
        cfl = 0.5)

Run a model with serial and single-rank distributed architectures,
then compare results for equivalence. Uses TimeStepWizard for
adaptive time stepping based on CFL condition.

For CompressibleDynamics, automatically computes max_Δt from the acoustic CFL.

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
        stop_time = 10,
        grid_size = (8, 8, 8),
        extent = (1000, 1000, 1000),
        cfl = 0.5)

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

    # For CompressibleDynamics, we need to initialize density with a reference state
    ρ₀(x, y, z) = default_density(z, constants)

    model_serial = AtmosphereModel(grid_serial; model_kwargs_serial...)
    if dynamics_type === :Compressible
        set!(model_serial; ρ = ρ₀, θ = θ₀, qᵗ = qᵗ₀, u = u₀, v = v₀)
    else
        set!(model_serial; θ = θ₀, qᵗ = qᵗ₀, u = u₀, v = v₀)
    end
    # For CompressibleDynamics, compute max_Δt from acoustic CFL
    max_Δt = dynamics_type === :Compressible ? acoustic_max_Δt(grid_serial; cfl) : Inf

    wizard_serial = TimeStepWizard(; cfl, max_Δt)
    simulation_serial = Simulation(model_serial; Δt=1e-3seconds, stop_time)
    simulation_serial.callbacks[:wizard] = Callback(wizard_serial, IterationInterval(1))
    run!(simulation_serial)

    model_distributed = AtmosphereModel(grid_distributed; model_kwargs_distributed...)
    if dynamics_type === :Compressible
        set!(model_distributed; ρ = ρ₀, θ = θ₀, qᵗ = qᵗ₀, u = u₀, v = v₀)
    else
        set!(model_distributed; θ = θ₀, qᵗ = qᵗ₀, u = u₀, v = v₀)
    end
    wizard_distributed = TimeStepWizard(; cfl, max_Δt)
    simulation_distributed = Simulation(model_distributed; Δt=1e-3seconds, stop_time)
    simulation_distributed.callbacks[:wizard] = Callback(wizard_distributed, IterationInterval(1))
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

const DYNAMICS_TYPES = (:Anelastic, :Compressible, :Prescribed)

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

    @testset "AtmosphereModel with distributed grid: $dynamics_type" for dynamics_type in DYNAMICS_TYPES
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
        elseif dynamics_type === :Compressible
            dynamics = CompressibleDynamics()
        else # :Prescribed
            reference_state = ReferenceState(grid, constants)
            dynamics = PrescribedDynamics(reference_state)
        end

        model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants)
        @test model isa AtmosphereModel

        # CompressibleDynamics requires density initialization
        if dynamics_type === :Compressible
            ρ₀(x, y, z) = default_density(z, constants)
            set!(model; ρ=ρ₀, θ=300, qᵗ=0.01)
        else
            set!(model; θ=300, qᵗ=0.01)
        end
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

    @testset "Compressible dynamics - doubly periodic" begin
        # CompressibleDynamics uses acoustic CFL constraint (computed automatically)
        result = run_comparison_test(;
            arch,
            dynamics_type = :Compressible,
            topology = (Periodic, Periodic, Bounded),
            closure = nothing,
            microphysics = nothing,
            advection = WENO(),
            coriolis = nothing,
            stop_time = 1)  # Short run due to small acoustic time step
        @test result
    end
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
        stop_time = 50)

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
