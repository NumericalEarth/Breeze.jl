#####
##### MPI worker script, launched on several ranks by `test/distributed/closure_fields.jl`.
#####
##### Steps a sheared, cloudy thermal with split-explicit `CompressibleDynamics` and each turbulence
##### closure that carries fields, on a `Distributed` grid, and compares it with the same simulation
##### run on a single (serial) grid: every prognostic field, and every eddy viscosity and diffusivity
##### over the interior of each rank plus one cell into its horizontal halos.
#####
##### The viscous and diffusive fluxes read those halo cells, corners included. They are not
##### exchanged between ranks: `compute_auxiliary_variables!` computes the closure fields there, from
##### the halos of the velocities, thermodynamic variables and tracers. The halo is the smallest the
##### closures accept, so that a stencil reaching further than they declare is a bounds error.
#####
##### Stand-alone usage:
#####
#####     mpiexec -n 4 julia --project=test test/mpi/closure_fields.jl xy
#####
##### where the (optional) argument is the partition: `x`, `y`, or `xy` (default).
#####
##### Uses one GPU per rank when CUDA is functional and there are at least as many devices as ranks.
#####

using MPI
MPI.Init()

using Breeze
using Breeze.TurbulenceClosures: TKEClosureFields
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations: Distributed, Partition, reconstruct_global_field
using Oceananigans.Grids: halo_size
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

const Nx, Ny, Nz = 16, 16, 8
const Lx, Ly, Lz = 8e3, 8e3, 4e3

bubble(x, y, z) = cos(π * min(1, sqrt(x^2 + y^2 + (z - 1500)^2) / 1e3) / 2)^2

function closure_fields_model(arch, closure; stop_iteration=2)
    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (2, 2, 2),
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    base_pressure = 1e5, reference_potential_temperature = 300)

    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    model = AtmosphereModel(grid; dynamics, microphysics, closure, advection = Centered(order=2), tracers = :ρa)

    # Shear in every direction, so that the eddy viscosities vary along the rank boundaries
    θᵢ(x, y, z) = 300 + 2 * bubble(x, y, z) + 0.004 * z
    uᵢ(x, y, z) = 10 * z / Lz + sin(2π * y / Ly)
    vᵢ(x, y, z) = 5 * cos(2π * x / Lx) * cos(2π * y / Ly)
    wᵢ(x, y, z) = 4 * bubble(x, y, z)
    set!(model; θ=θᵢ, qᵗ=0.004, u=uᵢ, v=vᵢ, w=wᵢ, ρ=HydrostaticallyBalancedDensity())

    # `HydrostaticallyBalancedDensity` takes no condensate, so the cloud is added afterwards.
    qᵗᵢ(x, y, z) = 0.012 + 0.012 * bubble(x - 500, y + 300, z)
    set!(model; qᵗ=qᵗᵢ)

    aᵢ(x, y, z) = 1e-3 * (1 + bubble(x + 700, y - 400, z - 300))
    set!(model.tracers.ρa, aᵢ)

    if haskey(model.tracers, :ρe)
        eᵢ(x, y, z) = 0.1 + 0.5 * bubble(x - 300, y + 900, z - 200)
        set!(model.tracers.ρe, eᵢ)
    end

    fill_halo_regions!(model.tracers)

    simulation = Simulation(model; Δt=2, stop_iteration, verbose=false)
    run!(simulation)
    return model
end

# The eddy viscosities and diffusivities of a closure, by name
eddy_diffusivities(fields::TKEClosureFields) = (; Kᵘ = fields.Kᵘ, Kᶜ = fields.Kᶜ, Kᵉ = fields.Kᵉ)
eddy_diffusivities(fields::NamedTuple{(:νₑ,)}) = fields
eddy_diffusivities(fields::NamedTuple{(:νₑ, :κₑ)}) =
    merge((; νₑ = fields.νₑ), NamedTuple(Symbol(:κₑ_, name) => fields.κₑ[name] for name in keys(fields.κₑ)))

# The interior of a field plus `H` cells into each horizontal halo, on the host
function horizontally_extended_data(field, H)
    Hx, Hy, Hz = halo_size(field.grid)
    nx, ny, nz = size(field)
    return Array(parent(field))[Hx+1-H:Hx+nx+H, Hy+1-H:Hy+ny+H, Hz+1:Hz+nz]
end

closures = (SmagorinskyLilly(), AnisotropicMinimumDissipation(), TKEBasedTurbulenceClosure())

for closure in closures
    @testset "$(nameof(typeof(closure))) fields on Distributed($(partition_name)) [$(nranks) ranks, $(use_gpu ? "GPU" : "CPU")]" begin
        arch = Distributed(child_architecture; partition=partition_for(partition_name, nranks))
        model = closure_fields_model(arch, closure)
        @test model.clock.iteration == 2
        @test model.grid.architecture isa Distributed

        if rank == 0
            serial_model = closure_fields_model(CPU(), closure)
        end

        # Compared on the host: in GPU runs the reconstructed field is on the GPU and the serial
        # reference on the CPU (Breeze issue #1193).
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

        # Each rank's closure fields, one cell into its halos, gathered with the offset of the rank
        # in the global grid. A halo cell of a rank is an interior cell of a neighbour, or its
        # periodic image.
        grid = model.grid
        nx, ny, _ = size(grid)
        i₀ = round(Int, (Array(xnodes(grid, Center()))[1] + Lx/2) / (Lx / Nx) - 1/2)
        j₀ = round(Int, (Array(ynodes(grid, Center()))[1] + Ly/2) / (Ly / Ny) - 1/2)
        diffusivities = eddy_diffusivities(model.closure_fields)
        local_data = map(field -> horizontally_extended_data(field, 1), diffusivities)
        gathered = MPI.gather((i₀, j₀, nx, ny, local_data), comm)

        if rank == 0
            serial_diffusivities = eddy_diffusivities(serial_model.closure_fields)
            for name in keys(diffusivities)
                serial_data = horizontally_extended_data(serial_diffusivities[name], 0)
                difference = zero(eltype(serial_data))
                for (i₀, j₀, nx, ny, local_data) in gathered
                    ii = mod1.(i₀ .+ (0:nx+1), Nx)
                    jj = mod1.(j₀ .+ (0:ny+1), Ny)
                    difference = max(difference, maximum(abs, local_data[name] .- serial_data[ii, jj, :]))
                end

                @testset "$(name)" begin
                    @test maximum(serial_data) > minimum(serial_data)
                    @test difference / maximum(abs, serial_data) <= tolerance
                end
            end
        end
    end
end

MPI.Barrier(comm)
rank == 0 && println("DISTRIBUTED_CLOSURE_FIELDS_OK")
