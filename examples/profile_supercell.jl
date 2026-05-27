## Profile-enabled split-explicit supercell driver.
##
## Wraps the timed loop in `CUDA.@profile` and instruments with NVTX ranges
## so the profile range covers only the benchmark window (compilation/setup
## are excluded). Run under nsys with
##   --capture-range=cudaProfilerApi --capture-range-end=stop
## See examples/profile_supercell.sh for the SLURM wrapper.
##
## Dynamics: `CompressibleDynamics(SplitExplicitTimeDiscretization)` — fast
## acoustic modes resolved with explicit substepping; no Poisson solve.

using MPI
MPI.Init()

using Breeze
using Breeze: CompressibleDynamics, DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.CompressibleEquations: SplitExplicitTimeDiscretization
using Oceananigans: Oceananigans
using Oceananigans.Units

using CUDA
using NVTX
using Printf

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    return i === nothing ? default : ARGS[i+1]
end

FT          = Dict("Float32" => Float32, "Float64" => Float64)[argval("--float-type", "Float32")]
Nx_per_gpu  = parse(Int, argval("--nx-per-gpu", "168"))
Ny          = parse(Int, argval("--ny", "168"))
Nz          = parse(Int, argval("--nz", "40"))
Nwarmup     = parse(Int, argval("--warmup-steps", "5"))
Nprofile    = parse(Int, argval("--profile-steps", "10"))
Δt          = parse(Float64, argval("--dt", "0.1"))
substeps    = parse(Int, argval("--substeps", "12"))

Oceananigans.defaults.FloatType = FT

Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank  = MPI.Comm_rank(MPI.COMM_WORLD)
arch  = Distributed(GPU(); partition=Partition(Ngpus, 1))

## Weak scaling: each GPU gets Nx_per_gpu × Ny × Nz points.
Lx_per_gpu = 168kilometers
Ly, Lz = 168kilometers, 20kilometers

Nx = Nx_per_gpu * Ngpus
Lx = Lx_per_gpu * Ngpus

if rank == 0
    @info "Split-explicit supercell profile" Ngpus FT Nx Ny Nz Nwarmup Nprofile Δt substeps
end

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

time_discretization = SplitExplicitTimeDiscretization(FT; substeps)
dynamics = CompressibleDynamics(time_discretization;
                                surface_pressure = 100000,
                                reference_potential_temperature = 300)

## Supercell background: piecewise potential temperature, shear, warm bubble.
θ₀ = 300       # K surface potential temperature
θᵖ = 343       # K tropopause potential temperature
zᵖ = 12000     # m tropopause height
Tᵖ = 213       # K tropopause temperature
zˢ = 5kilometers
uˢ = 30
uᶜ = 15

g   = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z <= zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

function u_background(z)
    uˡ = uˢ * (z / zˢ) - uᶜ
    uᵗ = (-4/5 + 3 * (z / zˢ) - 5/4 * (z / zˢ)^2) * uˢ - uᶜ
    uᵘ = uˢ - uᶜ
    return (z < (zˢ - 1000)) * uˡ +
           (abs(z - zˢ) <= 1000) * uᵗ +
           (z > (zˢ + 1000)) * uᵘ
end

Δθ  = 3
rᵇʰ = 10kilometers
rᵇᵛ = 1500
zᵇ  = 1500
xᵇ  = Lx / 2
yᵇ  = Ly / 2

function θᵢ(x, y, z)
    θ̄ = θ_background(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos((π / 2) * R)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, y, z) = u_background(z)

microphysics = DCMIP2016KesslerMicrophysics()
advection    = WENO(order=5)
model = AtmosphereModel(grid; dynamics, microphysics, advection,
                        thermodynamic_constants=constants)

reference_density = model.dynamics.reference_state.density
set!(model; θ=θᵢ, u=uᵢ, ρ=reference_density)

function many_time_steps!(model, Nt, Δt)
    for n = 1:Nt
        NVTX.@range "time_step $n" begin
            time_step!(model, Δt)
        end
    end
    return nothing
end

MPI.Barrier(MPI.COMM_WORLD)

elapsed_warmup  = 0.0
elapsed_profile = 0.0

NVTX.@range "warmup" begin
    elapsed_warmup = @elapsed many_time_steps!(model, Nwarmup, Δt)
end
MPI.Barrier(MPI.COMM_WORLD)

CUDA.@profile begin
    NVTX.@range "profile_window" begin
        elapsed_profile = @elapsed many_time_steps!(model, Nprofile, Δt)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @info @sprintf("Warmup  (%d steps): %.3f s", Nwarmup,  elapsed_warmup)
    @info @sprintf("Profile (%d steps): %.3f s", Nprofile, elapsed_profile)
end
