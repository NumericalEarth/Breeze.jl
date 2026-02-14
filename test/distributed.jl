# Distributed integration tests for AtmosphereModel
#
# Section A: Single-rank "distributed" tests (run inline)
# Section B: Multi-rank MPI tests (launched via mpiexec -n 4)
#
# When run in the standard test suite, these tests use a single-rank "distributed"
# configuration. The multi-rank tests self-launch via mpiexec inside the test.

using Breeze
using Breeze: PrescribedDynamics
using Breeze.Thermodynamics: adiabatic_hydrostatic_density
using MPI: mpiexec
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

function make_dynamics(dynamics_type, grid, constants)
    if dynamics_type === :Compressible
        return CompressibleDynamics()
    else
        reference_state = ReferenceState(grid, constants)
        return dynamics_type === :Anelastic ? AnelasticDynamics(reference_state) :
                                              PrescribedDynamics(reference_state)
    end
end

function set_initial_conditions!(model, dynamics_type; u=0, v=0)
    if dynamics_type === :Compressible
        # CompressibleDynamics needs hydrostatically balanced density
        # to avoid acoustic instability.
        d = model.dynamics
        constants = model.thermodynamic_constants
        ρ₀(x, y, z) = adiabatic_hydrostatic_density(z, d.surface_pressure, 300.0,
                                                      d.standard_pressure, constants)
        set!(model; θ=300, qᵗ=0.01, ρ=ρ₀, u, v)
    else
        set!(model; θ=300, qᵗ=0.01, u, v)
    end
end

function make_simulation(model, dynamics_type; stop_time=10)
    if dynamics_type === :Compressible
        # CompressibleDynamics resolves acoustic waves, requiring a small fixed Δt.
        simulation = Simulation(model; Δt=0.05seconds, stop_time=min(stop_time, 1), verbose=false)
    else
        simulation = Simulation(model; Δt=1e-3seconds, stop_time, verbose=false)
        simulation.callbacks[:wizard] = Callback(TimeStepWizard(cfl=0.5), IterationInterval(1))
    end
    return simulation
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
    dynamics_d = make_dynamics(dynamics_type, grid_distributed, constants)

    model_distributed = AtmosphereModel(grid_distributed; dynamics=dynamics_d,
                                        thermodynamic_constants=constants,
                                        advection=WENO(), closure, microphysics, coriolis)
    set_initial_conditions!(model_distributed, dynamics_type; u=u₀, v=v₀)

    simulation_d = make_simulation(model_distributed, dynamics_type; stop_time)
    run!(simulation_d)

    # Serial model (only on root rank for comparison)
    local_rank = arch_distributed.local_rank
    if local_rank == 0
        grid_serial = RectilinearGrid(arch; size=grid_size, x, y, z, topology)
        dynamics = make_dynamics(dynamics_type, grid_serial, constants)

        model_serial = AtmosphereModel(grid_serial; dynamics, thermodynamic_constants=constants,
                                       advection=WENO(), closure, microphysics, coriolis)
        set_initial_conditions!(model_serial, dynamics_type; u=u₀, v=v₀)

        simulation = make_simulation(model_serial, dynamics_type; stop_time)
        run!(simulation)

        # Compare fields on root
        FT = eltype(grid_distributed)
        return fields_match(model_serial, model_distributed; rtol=10*eps(FT))
    else
        return true  # Non-root ranks pass by default
    end
end

#####
##### Serial reference simulation for multi-rank comparison
#####

function run_serial_simulation(; dynamics_type=:Anelastic, microphysics=nothing,
                                grid_size=(16, 16, 16), extent=(1000, 1000, 1000),
                                stop_time=10, output_filename=nothing, output_fields=nothing)
    topology = (Periodic, Periodic, Bounded)
    x, y, z = (0, extent[1]), (0, extent[2]), (0, extent[3])
    Lx, Ly, Lz = extent
    constants = ThermodynamicConstants()

    u₀(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly)
    v₀(x, y, z) = cos(2π * x / Lx) * sin(2π * y / Ly)

    grid = RectilinearGrid(CPU(); size=grid_size, x, y, z, topology)
    dynamics = make_dynamics(dynamics_type, grid, constants)

    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants,
                            advection=WENO(), microphysics)
    set_initial_conditions!(model, dynamics_type; u=u₀, v=v₀)

    simulation = make_simulation(model, dynamics_type; stop_time)

    if !isnothing(output_filename)
        outputs = isnothing(output_fields) ? Oceananigans.prognostic_fields(model) : output_fields
        simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                                      filename = output_filename,
                                                      schedule = IterationInterval(1),
                                                      overwrite_existing = true,
                                                      with_halos = true)
    end

    run!(simulation)

    return model
end

#####
##### Section A — Single-rank tests (run inline)
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

        @testset "CompressibleDynamics" begin
            @test run_and_compare(; arch, dynamics_type=:Compressible)
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

    #####
    ##### Section B — Multi-rank MPI tests (launched via mpiexec -n 4)
    ##### Uses JLD2Writer for distributed output and FieldTimeSeries to combine/compare.
    #####

    @testset "Multi-rank MPI tests" begin
        nranks = 4
        test_project = Base.active_project()
        output_dir = mktempdir()

        partitions = [
            ("x4",    "Partition(4, 1, 1)"),
            ("x2y2",  "Partition(2, 2, 1)"),
            ("y4",    "Partition(1, 4, 1)"),
        ]

        configurations = [
            ("AnelasticDynamics",    ":Anelastic",    "nothing"),
            ("CompressibleDynamics", ":Compressible",  "nothing"),
            ("SaturationAdjustment", ":Anelastic",    "SaturationAdjustment()"),
        ]

        for (config_name, dynamics_type, microphysics) in configurations
            for (part_name, partition_str) in partitions
                dist_prefix = joinpath(output_dir, "breeze_dist_$(config_name)_$(part_name)")
                serial_file = joinpath(output_dir, "breeze_serial_$(config_name)_$(part_name)")
                script_file = joinpath(output_dir, "breeze_mpi_$(config_name)_$(part_name).jl")

                # Build dynamics-specific setup and time stepping for MPI script
                if dynamics_type == ":Compressible"
                    dynamics_block = """
                    dynamics = CompressibleDynamics()
                    """
                    ic_line = """
                    using Breeze.Thermodynamics: adiabatic_hydrostatic_density
                    ρ₀(x, y, z) = adiabatic_hydrostatic_density(z, dynamics.surface_pressure, 300.0,
                                                                 dynamics.standard_pressure, constants)
                    set!(model; θ=300, qᵗ=0.01, ρ=ρ₀, u=u₀, v=v₀)
                    """
                    simulation_block = """
                    simulation = Simulation(model; Δt=0.05seconds, stop_time=1, verbose=false)
                    """
                else
                    dynamics_block = """
                    reference_state = ReferenceState(grid, constants)
                    dynamics = $dynamics_type === :Anelastic ? AnelasticDynamics(reference_state) :
                                                               PrescribedDynamics(reference_state)
                    """
                    ic_line = """
                    set!(model; θ=300, qᵗ=0.01, u=u₀, v=v₀)
                    """
                    simulation_block = """
                    simulation = Simulation(model; Δt=1e-3seconds, stop_time=10, verbose=false)
                    simulation.callbacks[:wizard] = Callback(TimeStepWizard(cfl=0.5), IterationInterval(1))
                    """
                end

                mpi_script = """
                using MPI
                MPI.Init()

                using Breeze
                using Oceananigans
                using Oceananigans.Architectures: CPU
                using Oceananigans.DistributedComputations: Distributed, Partition
                using Oceananigans.Units: seconds

                arch = Distributed(CPU(); partition = $partition_str)

                grid_size = (16, 16, 16)
                extent = (1000, 1000, 1000)
                topology = (Periodic, Periodic, Bounded)
                x, y, z = (0, extent[1]), (0, extent[2]), (0, extent[3])
                Lx, Ly, Lz = extent
                constants = ThermodynamicConstants()

                u₀(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly)
                v₀(x, y, z) = cos(2π * x / Lx) * sin(2π * y / Ly)

                grid = RectilinearGrid(arch; size=grid_size, x, y, z, topology)
                $dynamics_block
                model = AtmosphereModel(grid; dynamics,
                                        thermodynamic_constants=constants,
                                        advection=WENO(),
                                        microphysics=$microphysics)
                $ic_line
                $simulation_block
                simulation.output_writers[:jld2] = JLD2Writer(model,
                                                              Oceananigans.prognostic_fields(model);
                                                              filename = "$(escape_string(dist_prefix))",
                                                              schedule = IterationInterval(1),
                                                              overwrite_existing = true,
                                                              with_halos = true)
                run!(simulation)

                MPI.Barrier(MPI.COMM_WORLD)
                MPI.Finalize()
                """

                @testset "$config_name — $partition_str" begin
                    # Write and run the MPI script
                    write(script_file, mpi_script)
                    try
                        run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) --project=$test_project -O0 $script_file`)
                    finally
                        rm(script_file; force=true)
                    end

                    # Run the serial reference with JLD2Writer output
                    dynamics_sym = dynamics_type == ":Anelastic"    ? :Anelastic :
                                   dynamics_type == ":Compressible" ? :Compressible : :Prescribed
                    microphysics_obj = microphysics == "nothing" ? nothing : SaturationAdjustment()
                    serial_model = run_serial_simulation(; dynamics_type=dynamics_sym,
                                                          microphysics=microphysics_obj,
                                                          output_filename=serial_file)

                    # Compare using FieldTimeSeries (automatically combines distributed rank files)
                    FT = eltype(serial_model.grid)
                    rtol = 10 * eps(FT)
                    atol = 100 * eps(FT) # absolute tolerance for near-zero fields
                    for name in keys(Oceananigans.prognostic_fields(serial_model))
                        varname = string(name)
                        fts_serial = FieldTimeSeries(serial_file * ".jld2", varname)
                        fts_distributed = FieldTimeSeries(dist_prefix * ".jld2", varname)

                        @test size(fts_distributed.grid) == size(fts_serial.grid)
                        @test length(fts_distributed.times) == length(fts_serial.times)

                        for n in 1:length(fts_serial.times)
                            @test isapprox(interior(fts_serial[n]), interior(fts_distributed[n]); rtol, atol)
                        end
                    end

                    # Cleanup
                    rm(serial_file * ".jld2"; force=true)
                    for r in 0:(nranks - 1)
                        rm(dist_prefix * "_rank$r.jld2"; force=true)
                    end
                end
            end
        end

        rm(output_dir; force=true, recursive=true)
    end
end
