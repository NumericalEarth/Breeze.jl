#####
##### MPI worker script, launched on several ranks by `test/distributed/terrain_seams.jl`.
#####
##### Runs a terrain-following, split-explicit `CompressibleDynamics` case with WENO momentum
##### advection and a turbulence closure on a `Distributed` grid, and compares the reconstructed
##### global solution with the same simulation on a single (serial) grid. Values read across a seam
##### between ranks that are neither computed nor exchanged make the two runs differ there:
##### momentum tendencies on the face past the interior (read by the terrain slow vertical-momentum
##### tendency) and closure fields in the halos (read by the closure fluxes).
#####
##### Stand-alone usage:
#####
#####     mpiexec -n 4 julia --project=test test/mpi/terrain_seams.jl xy
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

function terrain_simulation(arch, closure; stop_iteration=3)
    Nx, Ny, Nz = 16, 16, 8
    Lx, Ly, Lz = 10000, 10000, 5000

    z = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation=LinearDecay())
    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z,
                           topology = (Periodic, Periodic, Bounded))

    # Off-centre and anisotropic, so the slopes differ on every seam
    materialize_terrain!(grid, (x, y) -> 400 * exp(-((x - 1000)^2 + (y + 500)^2 / 2) / 2000^2))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6);
                                    reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, closure, advection=WENO(order=5))
    constants = model.thermodynamic_constants

    ρᵢ(x, y, z) = adiabatic_hydrostatic_density(z, 101325, 300, 1e5, constants)
    uᵢ(x, y, z) = 10 + sin(2π * y / Ly)
    vᵢ(x, y, z) = 5 + cos(2π * x / Lx)
    θᵢ(x, y, z) = 300 + 0.004 * z + 0.5 * exp(-(x^2 + y^2) / 1500^2)
    set!(model, ρ=ρᵢ, θ=θᵢ, u=uᵢ, v=vᵢ)

    if :ρe in keys(model.tracers)
        eᵢ(x, y, z) = 0.5 + 0.4 * sin(2π * x / Lx) * cos(2π * y / Ly)
        set!(model, ρe = (x, y, z) -> ρᵢ(x, y, z) * eᵢ(x, y, z))
    end

    simulation = Simulation(model; Δt=1, stop_iteration, verbose=false)
    run!(simulation)
    return model
end

compared_fields(model) = Oceananigans.prognostic_fields(model)

closures = (nothing, SmagorinskyLilly(), TKEBasedTurbulenceClosure())

for closure in closures
    @testset "Terrain seams on Distributed($(partition_name)) with $(nameof(typeof(closure))) [$(nranks) ranks, $(use_gpu ? "GPU" : "CPU")]" begin
        arch = Distributed(child_architecture; partition=partition_for(partition_name, nranks))
        model = terrain_simulation(arch, closure)
        @test model.clock.iteration == 3

        serial_model = rank == 0 ? terrain_simulation(CPU(), closure) : nothing

        distributed_fields = compared_fields(model)
        for name in keys(distributed_fields)
            global_field = reconstruct_global_field(distributed_fields[name])

            if rank == 0
                distributed_data = Array(interior(global_field))
                serial_data = Array(interior(compared_fields(serial_model)[name]))
                @test !any(isnan, distributed_data)
                @info "$(name): max difference $(maximum(abs, distributed_data .- serial_data))"
                @test distributed_data ≈ serial_data rtol=rtol
            end
        end
    end
end

MPI.Barrier(comm)
rank == 0 && println("DISTRIBUTED_TERRAIN_SEAMS_OK")
