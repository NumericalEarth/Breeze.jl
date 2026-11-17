#####
##### MPI worker script, launched on several ranks by `test/distributed/tracer_transport.jl`.
#####
##### Transports two passive tracers, one with a uniform mass fraction and one with a bubble of it,
##### through a rising thermal with split-explicit `CompressibleDynamics` on a `Distributed` grid,
##### with each acoustically substepped time stepper and with the scalar tendencies built before and
##### after the acoustic loop. Every prognostic field of the reconstructed global solution is
##### compared against the same simulation run on a single (serial) grid.
#####
##### Each ingredient reads halos that nothing else exchanges between ranks:
#####
#####   - a tracer is the one prognostic whose own halos the tendencies read: `update_state!` turns
#####     it into a mass fraction in place and advects that, whereas momentum, density and the
#####     thermodynamic variable have their halos filled again where they are used and moisture is
#####     advected through a separate mass fraction field;
#####   - with condensate, the end of an acoustic loop differentiates the mass flux the loop summed
#####     (`remove_moisture_content!`), a term that vanishes identically in clear air;
#####   - an eddy viscosity is computed in the interior and read across cell faces.
#####
##### The last cases close the domain with walls in `y`. A rank of a partitioned `Bounded` direction
##### has a wall on one side at most, so its own topology is not `Bounded`, and the acoustic loop has
##### to hold the wall-normal momentum perturbation at zero on the wall it does have.
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

# With `cloudy = true`, a saturated bubble with saturation adjustment and an eddy viscosity.
function tracer_transport_model(arch; timestepper, timestepper_kwargs, cloudy=false, walls=false, stop_iteration=3)
    grid = RectilinearGrid(arch; size = (16, 16, 16), halo = (5, 5, 5),
                           x = (-4e3, 4e3), y = (-4e3, 4e3), z = (0, 4e3),
                           topology = (Periodic, walls ? Bounded : Periodic, Bounded))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    base_pressure = 1e5, reference_potential_temperature = 300)

    microphysics = cloudy ? SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()) : nothing
    closure = cloudy ? SmagorinskyLilly() : nothing

    model = AtmosphereModel(grid; dynamics, microphysics, closure, advection = WENO(order=5),
                            tracers = (:ρa, :ρb), timestepper, timestepper_kwargs)

    # Uniform moisture unless `cloudy`: a serial run with a bubble of vapor does not keep a uniform
    # tracer mass fraction (of the total density, vapor included) uniform to round-off either.
    θᵢ(x, y, z) = 300 + 2 * bubble(x, y, z)
    uᵢ(x, y, z) = 10 * z / 4e3
    wᵢ(x, y, z) = 4 * bubble(x, y, z)
    vᵢ(x, y, z) = walls ? 5 * cos(π * y / 8e3) : 5 # no flow through the walls at y = ±4 km
    set!(model; θ=θᵢ, qᵗ=0.004, u=uᵢ, v=vᵢ, w=wᵢ, ρ=HydrostaticallyBalancedDensity())

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

    simulation = Simulation(model; Δt=5, stop_iteration, verbose=false)
    run!(simulation)
    return model
end

ark3 = :AcousticRungeKutta3
mis = :MultirateInfinitesimalStep
mis_kwargs = (; coefficients = :SSPMIS43)

# (time stepper, its keyword arguments, cloudy, walls in y)
cases = ((ark3, (; post_loop_scalar_tendencies = false), false, false),
         (ark3, (; post_loop_scalar_tendencies = true), false, false),
         (mis, (; mis_kwargs..., post_loop_scalar_tendencies = false), false, false),
         (mis, (; mis_kwargs..., post_loop_scalar_tendencies = true), false, false),
         (mis, (; mis_kwargs..., post_loop_scalar_tendencies = true), true, false),
         (ark3, (; post_loop_scalar_tendencies = false), false, true),
         (mis, (; mis_kwargs..., post_loop_scalar_tendencies = true), true, true))

for (timestepper, timestepper_kwargs, cloudy, walls) in cases
    post_loop_scalar_tendencies = timestepper_kwargs.post_loop_scalar_tendencies
    timing = post_loop_scalar_tendencies ? "post-loop" : "pre-loop"
    air = cloudy ? "cloudy, with an eddy viscosity" : "clear"
    sides = walls ? "walls in y" : "periodic in y"
    @testset "Tracer transport on Distributed($(partition_name)) with $(timestepper), $(timing) scalar tendencies, $(air), $(sides) [$(nranks) ranks, $(use_gpu ? "GPU" : "CPU")]" begin
        arch = Distributed(child_architecture; partition=partition_for(partition_name, nranks))
        model = tracer_transport_model(arch; timestepper, timestepper_kwargs, cloudy, walls)
        @test model.clock.iteration == 3
        @test model.grid.architecture isa Distributed

        if rank == 0
            serial_model = tracer_transport_model(CPU(); timestepper, timestepper_kwargs, cloudy, walls)
        end

        # Every prognostic, compared on the host: in GPU runs the reconstructed field is on the
        # GPU and the serial reference on the CPU (Breeze issue #1193).
        for name in keys(prognostic_fields(model))
            global_field = reconstruct_global_field(prognostic_fields(model)[name])

            if rank == 0
                distributed_data = Array(interior(global_field))
                serial_data = Array(interior(prognostic_fields(serial_model)[name]))
                @testset "$(name)" begin
                    @test !any(isnan, distributed_data)
                    @test relative_difference(distributed_data, serial_data) <= tolerance
                end
            end
        end

        # A uniform mass fraction stays uniform to round-off when each stage transports it with
        # its own loop's mass flux, which a stale halo across a rank boundary would break.
        global_ρa = reconstruct_global_field(model.tracers.ρa)
        global_ρ = reconstruct_global_field(total_density(model.dynamics))
        if rank == 0 && post_loop_scalar_tendencies && !cloudy
            a = Array(interior(global_ρa)) ./ Array(interior(global_ρ))
            @test maximum(abs, a .- c₀) / c₀ < 1e-12
        end
    end
end

MPI.Barrier(comm)
rank == 0 && println("DISTRIBUTED_TRACER_TRANSPORT_OK")
