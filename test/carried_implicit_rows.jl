# The density-weighted implicit solves sweep with rows carried from level to level
# (src/AtmosphereModels/carried_implicit_rows.jl). Every such solve of a run is repeated with Oceananigans' generic
# `get_coefficient` sweep on copies, and both must leave the same bits in the solution and the solver's scratch.
# The models run on the CPU.

using Breeze
using CloudMicrophysics
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: Solvers, BatchedTridiagonalSolver
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Test
using Breeze.AtmosphereModels: CarriedRowArguments

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

function tke_one_moment_model(FT)
    Oceananigans.defaults.FloatType = FT
    z = TerrainFollowingVerticalDiscretization(collect(range(0, 12e3, length = 12)))
    grid = LatitudeLongitudeGrid(CPU(), FT; size = (6, 5, 11), halo = (5, 5, 5), z,
                                 longitude = (-100, -99.4), latitude = (35, 35.5),
                                 topology = (Bounded, Bounded, Bounded))
    materialize_terrain!(grid, (λ, φ) -> 600 * exp(-((λ + 99.7)^2 + (φ - 35.25)^2) / 0.15^2))

    cloud_formation = SaturationAdjustment(FT; equilibrium = MixedPhaseEquilibrium(FT))
    bounded = WENO(order = 5, bounds = (0, 1))
    Jᵛ = Field{Center, Center, Nothing}(grid)
    set!(Jᵛ, (λ, φ) -> 1e-4 * (1 + sin(4λ)))

    model = AtmosphereModel(grid;
                            dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                                            reference_potential_temperature = 300),
                            microphysics = OneMomentCloudMicrophysics(FT; cloud_formation),
                            closure = TKEBasedTurbulenceClosure(),
                            boundary_conditions = (; ρqᵉ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(Jᵛ))),
                            momentum_advection = WENO(order = 5),
                            scalar_advection = (ρθ = WENO(order = 5), ρqᵉ = bounded, ρqʳ = bounded, ρqˢⁿ = bounded,
                                                ρe = Centered(order = 2)),
                            timestepper = :AcousticRungeKutta3)

    set!(model; ρ = model.dynamics.reference_state.density,
         θ = (λ, φ, z) -> 300 + 4e-3 * z + 0.5 * sin(3λ) * exp(-z / 3000),
         qᵗ = (λ, φ, z) -> 0.018 * exp(-z / 2500) * (1 + 0.3 * cos(5λ + 2φ)),
         u = (λ, φ, z) -> 8 + 1e-3 * z, v = (λ, φ, z) -> -3 + cos(3λ),
         ρe = (λ, φ, z) -> 0.5 * exp(-z / 1000),
         qʳ = (λ, φ, z) -> 1e-4 * exp(-z / 2000),
         qˢⁿ = (λ, φ, z) -> 5e-5 * exp(-(z - 6000)^2 / 2000^2))

    return model
end

# A closure with constant coefficients, adaptive-implicit vertical advection and an implicit-explicit surface flux:
# the carried rows then also hold the implicit-advection rows and the boundary-flux diagonal.
function scalar_diffusivity_model(FT)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(CPU(), FT; size = (6, 6, 9), halo = (5, 5, 5), x = (0, 4e3), y = (0, 4e3),
                           z = [0, 200, 450, 750, 1100, 1500, 2000, 2600, 3300, 4000],
                           topology = (Periodic, Periodic, Bounded))
    aiva = WENO(FT; time_discretization = AdaptiveVerticallyImplicitDiscretization(FT; cfl = FT(1//100)))
    model = AtmosphereModel(grid;
                            dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                                            reference_potential_temperature = 300),
                            closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = 10, κ = 10),
                            tracers = (:ρa, :ρb),
                            boundary_conditions = (; ρb = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1e-3))),
                            momentum_advection = WENO(order = 5),
                            scalar_advection = (; ρθ = WENO(order = 5), ρqᵛ = aiva, ρa = aiva, ρb = WENO(order = 5)),
                            timestepper = :AcousticRungeKutta3)
    set!(model; ρ = model.dynamics.reference_state.density, θ = 300,
         w = (x, y, z) -> sinpi(z / 4e3), qᵗ = (x, y, z) -> 0.01 * exp(-z / 2000),
         ρa = (x, y, z) -> 1 + z / 4e3, ρb = (x, y, z) -> 1 + sinpi(x / 2e3) * exp(-z / 1e3))
    return model
end

unsigned_of(::Type{Float32}) = UInt32
unsigned_of(::Type{Float64}) = UInt64
bits(x::AbstractFloat) = reinterpret(unsigned_of(typeof(x)), x)   # the sign of zero counts
bits(a::AbstractArray) = reinterpret(unsigned_of(eltype(a)), a)

const checked_solves = Ref(0)
const differing = Ref(0)
const located = Set{Any}()

const GenericSweep = Tuple{Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any}

function Solvers.solve!(ϕ, solver::BatchedTridiagonalSolver, rhs, args::Vararg{Any, 16})
    if args isa CarriedRowArguments && solver.tridiagonal_direction isa ZDirection
        Nx, Ny, Nz = size(solver.grid)
        ϕ₁, ϕ₂ = deepcopy(ϕ), deepcopy(ϕ)
        t₁, t₂ = copy(solver.t), copy(solver.t)
        f₁ = rhs === ϕ ? ϕ₁ : rhs
        f₂ = rhs === ϕ ? ϕ₂ : rhs
        a, b, c, p, d = solver.a, solver.b, solver.c, solver.parameters, solver.tridiagonal_direction
        for j in 1:Ny, i in 1:Nx
            Solvers.solve_batched_tridiagonal_system_z!(i, j, Nz, ϕ₁, a, b, c, f₁, t₁, solver.grid, p, args, d)
            invoke(Solvers.solve_batched_tridiagonal_system_z!, GenericSweep,
                   i, j, Nz, ϕ₂, a, b, c, f₂, t₂, solver.grid, p, args, d)
        end
        checked_solves[] += 1
        push!(located, args[6])
        same = bits(parent(ϕ₁)) == bits(parent(ϕ₂)) && bits(t₁) == bits(t₂)
        same || (differing[] += 1)
    end
    return invoke(Solvers.solve!, Tuple{Any, BatchedTridiagonalSolver, Any, Vararg{Any}}, ϕ, solver, rhs, args...)
end

@testset "Carried implicit-diffusion rows [$FT, $builder]" for FT in (Float32, Float64),
                                                               builder in (tke_one_moment_model, scalar_diffusivity_model)
    checked_solves[] = 0; differing[] = 0; empty!(located)
    model = builder(FT)
    for _ in 1:3
        time_step!(model, 2)
    end
    @info "$builder $FT" checked_solves[] differing[] located
    @test checked_solves[] > 0
    @test differing[] == 0
    @test any(ℓ -> ℓ isa Face, located)
    @test all(isfinite, interior(model.momentum.ρu))
end
