#####
##### MPI worker script, launched on several ranks by `test/distributed/open_boundary_seams.jl`.
#####
##### Runs a cloudy, split-explicit `CompressibleDynamics` case with prescribed normal flow on every
##### lateral side on a `Distributed` grid, and compares the reconstructed global solution with the
##### same simulation on a single (serial) grid. A seam between ranks reaches the lateral boundaries,
##### where the acoustic loop zeroes the wall-normal momentum perturbations and relaxes the outermost
##### cells toward the prescribed flow; both must act on a rank as they do on the serial grid. The
##### moisture is what reads the loop's integrated mass flux: the transport velocities and, with
##### condensate, the moisture content of the loop's pressure.
#####
##### Stand-alone usage:
#####
#####     mpiexec -n 4 julia --project=test test/mpi/open_boundary_seams.jl xy
#####
##### where the (optional) argument is the partition: `x`, `y`, or `xy` (default).
#####
##### Uses one GPU per rank when CUDA is functional and there are at least as many devices as ranks.
#####

using MPI
MPI.Init()

using Breeze
using Oceananigans
using Oceananigans.DistributedComputations: Distributed, Partition, reconstruct_global_field
using CUDA: CUDA
using Test

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

const use_gpu = CUDA.functional() && length(CUDA.devices()) >= nranks
const child_architecture = use_gpu ? GPU() : CPU()

# Rounding differs between CPU and GPU, so the comparison with the serial CPU run is looser on GPU.
const rtol = use_gpu ? 1e-6 : 1e-8

const partition_name = isempty(ARGS) ? "xy" : ARGS[1]

function partition_for(name, nranks)
    name == "x"  && return Partition(nranks, 1)
    name == "y"  && return Partition(1, nranks)
    name == "xy" && return Partition(2, nranks ÷ 2)
    error("Unknown partition $(repr(name))")
end

function open_boundary_simulation(arch; stop_iteration=3)
    Nx, Ny, Nz = 16, 16, 8
    Lx, Ly, Lz = 10000, 10000, 5000

    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (0, Lx), y = (0, Ly), z = (0, Lz),
                           topology = (Bounded, Bounded, Bounded))

    # Inflow through the west and south sides, outflow through the east and north sides
    ρU, ρV = 12, 6
    boundary_conditions = (ρu = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρU),
                                                        east = NormalFlowBoundaryCondition(ρU)),
                           ρv = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(ρV),
                                                        north = NormalFlowBoundaryCondition(ρV)))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6);
                                    reference_potential_temperature = 300)
    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    model = AtmosphereModel(grid; dynamics, microphysics, boundary_conditions, advection=WENO(order=5))
    constants = model.thermodynamic_constants

    # Off-center, so the perturbation differs on every seam
    ρᵢ(x, y, z) = adiabatic_hydrostatic_density(z, 101325, 300, 1e5, constants)
    θᵢ(x, y, z) = 300 + 0.5 * exp(-((x - 4000)^2 + (y - 6000)^2) / 1500^2)
    qᵢ(x, y, z) = 0.012 * exp(-z / 3000) # saturated aloft, so the case is cloudy
    set!(model, ρ=ρᵢ, θ=θᵢ, qᵗ=qᵢ, u=10, v=5)

    simulation = Simulation(model; Δt=1, stop_iteration, verbose=false)
    run!(simulation)
    return model
end

@testset "Open-boundary seams on Distributed($(partition_name)) [$(nranks) ranks, $(use_gpu ? "GPU" : "CPU")]" begin
    arch = Distributed(child_architecture; partition=partition_for(partition_name, nranks))
    model = open_boundary_simulation(arch)
    @test model.clock.iteration == 3

    serial_model = rank == 0 ? open_boundary_simulation(CPU()) : nothing
    rank == 0 && @test maximum(interior(serial_model.microphysical_fields.qˡ)) > 0

    distributed_fields = Oceananigans.prognostic_fields(model)
    for name in keys(distributed_fields)
        global_field = reconstruct_global_field(distributed_fields[name])

        if rank == 0
            distributed_data = Array(interior(global_field))
            serial_data = Array(interior(Oceananigans.prognostic_fields(serial_model)[name]))
            @test !any(isnan, distributed_data)
            @info "$(name): max difference $(maximum(abs, distributed_data .- serial_data))"
            @test distributed_data ≈ serial_data rtol=rtol
        end
    end
end

MPI.Barrier(comm)
rank == 0 && println("DISTRIBUTED_OPEN_BOUNDARY_SEAMS_OK")
