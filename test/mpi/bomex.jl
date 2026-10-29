#####
##### MPI worker script, launched on several ranks by `test/distributed/bomex.jl`.
#####
##### Runs a small version of the BOMEX shallow cumulus case of `examples/bomex.jl`
##### ([Siebesma et al. (2003)](@cite Siebesma2003)) with `AnelasticDynamics` on a `Distributed` grid.
##### Besides the distributed pressure solve (`DistributedFourierTridiagonalPoissonSolver`), this exercises the horizontal averages of
##### `SubsidenceForcing` (reduced across ranks), geostrophic forcing, surface flux and drag
##### boundary conditions, and warm-phase saturation adjustment. The reconstructed global solution
##### is compared against the same simulation run on a single (serial) grid.
#####
##### The BOMEX profiles are written out here as piecewise-linear functions rather than taken from
##### AtmosphericProfilesLibrary, and the initial noise is a smooth deterministic function of position
##### rather than `rand()`, so that the serial and distributed runs start from identical states.
#####
##### Stand-alone usage:
#####
#####     mpiexec -n 4 julia --project=test test/mpi/bomex.jl xy
#####
##### where the (optional) argument is the partition: `x`, `y`, or `xy` (default).
#####
##### Uses one GPU per rank when CUDA is functional and there are at least as many devices as ranks.
#####

using MPI
MPI.Init()

using Breeze
using Oceananigans
using Oceananigans.DistributedComputations: Distributed, DistributedFourierTridiagonalPoissonSolver, Partition,
                                            reconstruct_global_field
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

#####
##### BOMEX profiles (Siebesma et al., 2003, Appendix B)
#####

# Linear interpolation between the points (zs[n], vs[n]), held constant beyond the last point.
function piecewise_linear(z, zs, vs)
    z ≤ zs[1] && return vs[1]
    for n in 2:length(zs)
        if z ≤ zs[n]
            return vs[n-1] + (vs[n] - vs[n-1]) * (z - zs[n-1]) / (zs[n] - zs[n-1])
        end
    end
    return vs[end]
end

bomex_θˡⁱ(z) = piecewise_linear(z, (0, 520, 1480, 2000, 3000), (298.7, 298.7, 302.4, 308.2, 311.85))
bomex_qᵗ(z)  = piecewise_linear(z, (0, 520, 1480, 2000, 3000), (17.0, 16.3, 10.7, 4.2, 3.0)) * 1e-3
bomex_u(z)   = piecewise_linear(z, (0, 700, 3000), (-8.75, -8.75, -4.61))
bomex_uᵍ(z)  = -10 + 1.8e-3 * z
bomex_vᵍ(z)  = 0
bomex_wˢ(z)  = piecewise_linear(z, (0, 1500, 2100), (0, -6.5e-3, 0))
bomex_dqᵗdt(z) = piecewise_linear(z, (0, 300, 500), (-1.2e-8, -1.2e-8, 0))
bomex_dTdt(z)  = piecewise_linear(z, (0, 1500, 2500), (-2 / 86400, -2 / 86400, 0))

function bomex_simulation(arch; stop_iteration=20)
    Nx, Ny, Nz = 32, 32, 20
    Lx, Ly, Lz = 3200, 3200, 3000  # (m)

    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (0, Lx), y = (0, Ly), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, base_pressure = 101500, potential_temperature = 299.1)
    dynamics = AnelasticDynamics(reference_state)

    # Surface fluxes of potential temperature and moisture, and quadratic drag
    FT = eltype(grid)
    p₀ = reference_state.base_pressure
    θ₀ = reference_state.potential_temperature
    q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)

    w′θ′ = 8e-3     # K m/s
    w′qᵗ′ = 5.2e-5  # m/s
    ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′θ′))
    ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′qᵗ′))

    u★ = 0.28  # m/s
    ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2)
    ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2)
    ρu_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★)))
    ρv_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★)))

    # Large-scale subsidence, geostrophic wind, moisture drying, and radiative cooling
    wˢ = Field{Nothing, Nothing, Face}(grid)
    set!(wˢ, bomex_wˢ)
    subsidence = SubsidenceForcing(wˢ)

    coriolis = FPlane(f=3.76e-5)
    geostrophic = geostrophic_forcings(bomex_uᵍ, bomex_vᵍ)

    drying = Field{Nothing, Nothing, Center}(grid)
    set!(drying, bomex_dqᵗdt)

    radiative_cooling = Field{Nothing, Nothing, Center}(grid)
    cᵖᵈ = constants.dry_air.heat_capacity
    set!(radiative_cooling, z -> cᵖᵈ * bomex_dTdt(z))

    forcing = (; u = (subsidence, geostrophic.u),
                 v = (subsidence, geostrophic.v),
                 θ = subsidence,
                 qᵉ = (subsidence, Forcing(drying)),
                 E = Forcing(radiative_cooling))

    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

    model = AtmosphereModel(grid; dynamics, coriolis, microphysics, forcing, advection = WENO(order=5),
                            boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

    # A smooth, deterministic perturbation below zδ, in place of the random noise of the example,
    # so that the serial and distributed initial conditions are identical.
    δθ = 0.1      # K
    δqᵗ = 2.5e-5  # kg/kg
    zδ = 1600     # m
    ϵ(x, y, z) = sin(2π * 3x / Lx + 1) * sin(2π * 2y / Ly + 2) * sin(π * z / zδ) / 2 * (z < zδ)

    θᵢ(x, y, z) = bomex_θˡⁱ(z) + δθ * ϵ(x, y, z)
    qᵢ(x, y, z) = bomex_qᵗ(z) + δqᵗ * ϵ(y, x, z)
    uᵢ(x, y, z) = bomex_u(z)

    set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ)

    simulation = Simulation(model; Δt=5, stop_iteration, verbose=false)
    run!(simulation)
    return model
end

@testset "BOMEX on Distributed($(partition_name)) [$(nranks) ranks, $(use_gpu ? "GPU" : "CPU")]" begin
    arch = Distributed(child_architecture; partition=partition_for(partition_name, nranks))
    model = bomex_simulation(arch; stop_iteration=20)
    @test model.clock.iteration == 20
    @test model.grid.architecture isa Distributed
    @test model.pressure_solver isa DistributedFourierTridiagonalPoissonSolver

    # Recomputing the tridiagonal coefficients reproduces those built by the solver's constructor.
    tridiagonal_solver = model.pressure_solver.batched_tridiagonal_solver
    main_diagonal = copy(tridiagonal_solver.b)
    lower_diagonal = copy(tridiagonal_solver.a)
    Breeze.AtmosphereModels.update_pressure_solver!(model.pressure_solver, model.dynamics)
    @test tridiagonal_solver.b ≈ main_diagonal
    @test tridiagonal_solver.a ≈ lower_diagonal

    if rank == 0
        serial_model = bomex_simulation(CPU())
    end

    for (name, local_field) in pairs(Oceananigans.prognostic_fields(model))
        global_field = reconstruct_global_field(local_field)

        if rank == 0
            @testset "$name" begin
                @test !any(isnan, parent(global_field))
                serial_field = Oceananigans.prognostic_fields(serial_model)[name]
                @test global_field ≈ serial_field rtol=rtol
            end
        end
    end
end

MPI.Barrier(comm)
rank == 0 && println("DISTRIBUTED_BOMEX_OK")
