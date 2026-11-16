#####
##### MPI worker script, launched on several ranks by `test/distributed/tracer_transport.jl`.
#####
##### Transports two passive tracers, one with a uniform mass fraction and one with a bubble of it,
##### through a rising thermal with `CompressibleDynamics` on a `Distributed` grid: split-explicit in
##### clear and in cloudy air, and fully explicit. Every prognostic field of the reconstructed global
##### solution is compared against the same simulation run on a single (serial) grid.
#####
##### Each ingredient reads halos that nothing else exchanges between ranks:
#####
#####   - a tracer is the one prognostic whose own halos the tendencies read: `update_state!` turns
#####     it into a mass fraction in place and advects that, whereas momentum, density and the
#####     thermodynamic variable have their halos filled again where they are used and moisture is
#####     advected through a separate mass fraction field;
#####   - with condensate, the end of an acoustic loop differentiates the mass flux the loop summed
#####     (`remove_moisture_content!`), a term that vanishes identically in clear air.
#####
##### Stand-alone usage:
#####
#####     mpiexec -n 4 julia --project=test test/mpi/tracer_transport.jl xy
#####
##### where the (optional) argument is the partition: `x`, `y`, or `xy` (default).
#####
##### Uses one GPU per rank when CUDA is functional and there are at least as many devices as ranks.
#####

using MPI
MPI.Init()

using Breeze
using Breeze.AtmosphereModels: total_density
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
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

# No operation of a step is reduced across ranks, so on the CPU the distributed and serial runs are
# bitwise identical. Rounding differs between CPU and GPU, so GPU runs are compared with the serial
# CPU run to a tolerance far below the error of a stale halo (10⁻⁴ to 10⁻² here).
const tolerance = use_gpu ? 1e-6 : 0

relative_difference(distributed, serial) = maximum(abs, distributed .- serial) / maximum(abs, serial)

const partition_name = isempty(ARGS) ? "xy" : ARGS[1]

function partition_for(name, nranks)
    name == "x"  && return Partition(nranks, 1)
    name == "y"  && return Partition(1, nranks)
    name == "xy" && return Partition(2, nranks ÷ 2)
    error("Unknown partition $(repr(name))")
end

const c₀ = 1e-3 # tracer mass fraction

bubble(x, y, z) = cos(π * min(1, sqrt(x^2 + y^2 + (z - 1500)^2) / 1e3) / 2)^2

# With `cloudy = true`, a saturated bubble with saturation adjustment.
function tracer_transport_model(arch, time_discretization; Δt, cloudy=false, stop_iteration=3)
    grid = RectilinearGrid(arch; size = (16, 16, 16), halo = (5, 5, 5),
                           x = (-4e3, 4e3), y = (-4e3, 4e3), z = (0, 4e3),
                           topology = (Periodic, Periodic, Bounded))

    dynamics = CompressibleDynamics(time_discretization;
                                    base_pressure = 1e5, reference_potential_temperature = 300)

    microphysics = cloudy ? SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()) : nothing

    model = AtmosphereModel(grid; dynamics, microphysics, advection = WENO(order=5), tracers = (:ρa, :ρb))

    # Uniform moisture unless `cloudy`: a serial run with a bubble of vapor does not keep a uniform
    # tracer mass fraction (of the total density, vapor included) uniform to round-off either.
    θᵢ(x, y, z) = 300 + 2 * bubble(x, y, z)
    uᵢ(x, y, z) = 10 * z / 4e3
    wᵢ(x, y, z) = 4 * bubble(x, y, z)
    set!(model; θ=θᵢ, qᵗ=0.004, u=uᵢ, v=5, w=wᵢ, ρ=HydrostaticallyBalancedDensity())

    # `HydrostaticallyBalancedDensity` takes no condensate, so the cloud is added afterwards.
    qᵗᵢ(x, y, z) = 0.012 + 0.012 * bubble(x - 500, y + 300, z)
    cloudy && set!(model; qᵗ=qᵗᵢ)

    # A uniform mass fraction `a` and a bubble of `b`, off the centre of the thermal
    ρ = total_density(model.dynamics)
    bᵢ(x, y, z) = c₀ * (1 + bubble(x + 700, y - 400, z - 300))
    set!(model.tracers.ρa, c₀)
    set!(model.tracers.ρb, bᵢ)
    interior(model.tracers.ρa) .*= interior(ρ)
    interior(model.tracers.ρb) .*= interior(ρ)
    fill_halo_regions!(model.tracers)

    simulation = Simulation(model; Δt, stop_iteration, verbose=false)
    run!(simulation)
    return model
end

# (description, time discretization, time step in seconds, cloudy)
cases = (("split-explicit dynamics, clear",  SplitExplicitTimeDiscretization(), 5,   false),
         ("split-explicit dynamics, cloudy", SplitExplicitTimeDiscretization(), 5,   true),
         ("explicit dynamics, clear",        ExplicitTimeStepping(),            0.2, false))

# One enclosing testset, so that a failing case does not keep the later ones from running.
@testset "Tracer transport on Distributed($(partition_name)) [$(nranks) ranks, $(use_gpu ? "GPU" : "CPU")]" begin
    for (description, time_discretization, Δt, cloudy) in cases
        @testset "$(description)" begin
            arch = Distributed(child_architecture; partition=partition_for(partition_name, nranks))
            model = tracer_transport_model(arch, time_discretization; Δt, cloudy)
            @test model.clock.iteration == 3
            @test model.grid.architecture isa Distributed

            if rank == 0
                serial_model = tracer_transport_model(CPU(), time_discretization; Δt, cloudy)
            end

            # Every prognostic, compared on the host: in GPU runs the reconstructed field is on the
            # GPU and the serial reference on the CPU (Breeze issue #1193).
            for name in keys(prognostic_fields(model))
                global_field = reconstruct_global_field(prognostic_fields(model)[name])

                if rank == 0
                    distributed_data = Array(interior(global_field))
                    serial_data = Array(interior(prognostic_fields(serial_model)[name]))
                    difference = relative_difference(distributed_data, serial_data)
                    @info "Relative difference from the serial run" description name difference
                    @testset "$(name)" begin
                        @test !any(isnan, distributed_data)
                        @test difference <= tolerance
                    end
                end
            end
        end
    end
end

MPI.Barrier(comm)
rank == 0 && println("DISTRIBUTED_TRACER_TRANSPORT_OK")
