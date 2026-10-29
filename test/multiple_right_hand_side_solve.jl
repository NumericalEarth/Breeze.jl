# Scalars whose implicit rows are identical are solved in one launch (`multiple_right_hand_side_implicit_step!`).
# Each solution, and the solver's scratch, must have the bits of the separate `implicit_step!` solves. CPU.

using Breeze
using CloudMicrophysics
using Oceananigans
using Oceananigans.TimeSteppers: time_step!, implicit_step!
using Test
using Breeze.AtmosphereModels: multiple_right_hand_side_implicit_step!, total_density, closure_scalar_index,
                               field_advection_scheme, implicit_step_scheme, implicit_advection_velocities
using Breeze.TimeSteppers: shared_row_scalar_names, acoustic_prognostic_names, tendency_transport_velocities

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

unsigned_of(::Type{Float32}) = UInt32
unsigned_of(::Type{Float64}) = UInt64
bits(a::AbstractArray) = reinterpret(unsigned_of(eltype(a)), a)

function solve_both_ways(model, Δt)
    solver = model.timestepper.implicit_solver
    names = shared_row_scalar_names(model, solver, acoustic_prognostic_names(model))
    ρ = total_density(model.dynamics)
    velocities = tendency_transport_velocities(model)
    prognostic = Oceananigans.prognostic_fields(model)
    saved = Dict(n => copy(parent(prognostic[n])) for n in names)
    t₀ = copy(solver.t)

    for n in names   # separate solves, as before
        implicit_step!(prognostic[n], solver, model.closure, model.closure_fields, closure_scalar_index(model, n),
                       model.clock, fields(model), Δt, implicit_step_scheme(field_advection_scheme(model.advection, n)),
                       implicit_advection_velocities(model.dynamics, velocities, n, model.microphysics, model.microphysical_fields), ρ)
    end
    separate = Dict(n => copy(parent(prognostic[n])) for n in names)
    t_separate = copy(solver.t)

    for n in names; parent(prognostic[n]) .= saved[n]; end
    solver.t .= t₀
    n₁ = first(names)
    multiple_right_hand_side_implicit_step!(map(n -> prognostic[n], names), solver, model.closure, model.closure_fields,
                                            closure_scalar_index(model, n₁), model.clock, fields(model), Δt,
                                            implicit_step_scheme(field_advection_scheme(model.advection, n₁)),
                                            implicit_advection_velocities(model.dynamics, velocities, n₁, model.microphysics,
                                                                          model.microphysical_fields), ρ)
    same = all(bits(parent(prognostic[n])) == bits(separate[n]) for n in names) && bits(solver.t) == bits(t_separate)
    return names, same
end

@testset "Multiple right-hand sides [$FT]" for FT in (Float32, Float64)
    model = tke_one_moment_model(FT)
    for step in 1:3
        time_step!(model, 2)
        names, same = solve_both_ways(model, FT(2/3))
        @info "step $step" names
        @test :ρe ∉ names
        @test length(names) ≥ 3
        @test same
    end
end
