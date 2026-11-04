#####
##### MPI worker script, launched on several ranks by `test/distributed/acoustic_wave.jl`.
#####
##### Runs the acoustic wave setup of `examples/acoustic_wave.jl` (a Gaussian density pulse in a
##### log-layer wind shear, with `CompressibleDynamics`), extended to three dimensions so it can be
##### partitioned in both x and y, on a `Distributed` grid. The reconstructed global solution is
##### compared against the same simulation run on a single (serial) grid.
#####
##### Stand-alone usage:
#####
#####     mpiexec -n 4 julia --project=test test/mpi/acoustic_wave.jl xy
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

# Use one GPU per rank when there are enough of them; otherwise run on the CPU.
const use_gpu = CUDA.functional() && length(CUDA.devices()) >= nranks
if !use_gpu && CUDA.functional()
    @warn "Not enough GPU devices available, falling back to CPU"
end
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

function acoustic_wave_simulation(arch; stop_iteration=10)
    Nx, Ny, Nz = 32, 32, 16
    Lx, Ly, Lz = 500, 500, 200  # (m)

    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz),
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))

    model = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_state = nothing))
    constants = model.thermodynamic_constants

    θ₀ = 300      # Reference potential temperature (K)
    p₀ = 101325   # Surface pressure (Pa)
    pˢᵗ = 1e5     # Standard pressure (Pa)

    Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
    cᵖᵈ = constants.dry_air.heat_capacity
    γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
    cᵃᶜ = sqrt(γ * Rᵈ * θ₀)

    # Log-law wind profile of the atmospheric surface layer
    U₀ = 20 # Surface velocity (m/s)
    ℓ = 1   # Roughness length (m)
    Uᵢ(z) = U₀ * log((z + ℓ) / ℓ)

    # Gaussian density pulse on top of the hydrostatic background
    δρ = 0.01    # Density perturbation amplitude (kg/m³)
    σ = 20       # Pulse width (m)
    gaussian(x, y, z) = exp(-(x^2 + y^2 + z^2) / 2σ^2)

    ρᵢ(x, y, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants) + δρ * gaussian(x, y, z)
    uᵢ(x, y, z) = Uᵢ(z)

    set!(model, ρ=ρᵢ, θ=θ₀, u=uᵢ)

    Δx, Δz = Lx / Nx, Lz / Nz
    Δt = 0.5 * min(Δx, Δz) / (cᵃᶜ + Uᵢ(Lz))

    simulation = Simulation(model; Δt, stop_iteration, verbose=false)
    run!(simulation)
    return model
end

@testset "Acoustic wave on Distributed($(partition_name)) [$(nranks) ranks, $(use_gpu ? "GPU" : "CPU")]" begin
    arch = Distributed(child_architecture; partition=partition_for(partition_name, nranks))
    model = acoustic_wave_simulation(arch; stop_iteration=10)
    @test model.clock.iteration == 10
    @test model.grid.architecture isa Distributed

    if rank == 0
        serial_model = acoustic_wave_simulation(CPU())
    end

    for name in (:ρu, :ρv, :ρw)
        local_field = getproperty(model.momentum, name)
        global_field = reconstruct_global_field(local_field)

        if rank == 0
            @test !any(isnan, parent(global_field))
            serial_field = getproperty(serial_model.momentum, name)
            @test global_field ≈ serial_field rtol=rtol
        end
    end

    global_ρᵈ = reconstruct_global_field(model.dynamics.dry_density)
    if rank == 0
        @test global_ρᵈ ≈ serial_model.dynamics.dry_density rtol=rtol
    end
end

MPI.Barrier(comm)
rank == 0 && println("DISTRIBUTED_ACOUSTIC_WAVE_OK")
