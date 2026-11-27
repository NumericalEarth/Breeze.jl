include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze: ReferenceState, AnelasticDynamics
using Breeze.AtmosphereModels: enforce_lateral_mass_balance!
using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: Integral
using Oceananigans.BoundaryConditions: BalancedTransport, FieldBoundaryConditions, NormalFlowBoundaryCondition
using Oceananigans.Fields: Field, compute!, interior
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Models: boundary_total_area
using Test

bounded_grid(FT = Float64; size = (16, 16, 12)) =
    RectilinearGrid(default_arch, FT; size, x = (0, 160), y = (0, 160), z = (0, 120),
                    topology = (Bounded, Bounded, Bounded))

# A prescribed boundary that the correction may shift. An imposed `NormalFlowBoundaryCondition` is
# held exactly as given — `BalancedTransport()` is what opts it into the correction — and the default
# impenetrable walls stay closed, so a face this file writes by hand needs one of these.
balanced(value = 0) = NormalFlowBoundaryCondition(value; scheme = BalancedTransport())
open_x(west = 0, east = 0) = FieldBoundaryConditions(west = balanced(west), east = balanced(east))
open_y(south = 0, north = 0) = FieldBoundaryConditions(south = balanced(south), north = balanced(north))

anelastic_model(grid; boundary_conditions = (; ρu = open_x(), ρv = open_y())) = AtmosphereModel(grid;
    dynamics = AnelasticDynamics(ReferenceState(grid, ThermodynamicConstants(eltype(grid));
                                                base_pressure = 1e5, potential_temperature = 300)),
    boundary_conditions)

# Net OUTWARD mass flux through the four lateral faces, and their total area. Measured here with
# Oceananigans' general `Integral`, independently of the boundary-plane reductions the correction
# itself carries.
function imbalance_and_area(momentum, grid)
    ρu, ρv = momentum.ρu, momentum.ρv
    Nx, Ny, _ = size(grid)
    ∮(plane, dims) = Array(interior(compute!(Field(Integral(plane; dims)))))[]

    flux = (∮(view(ρu, Nx + 1, :, :), (2, 3)) - ∮(view(ρu, 1, :, :), (2, 3))) +
           (∮(view(ρv, :, Ny + 1, :), (1, 3)) - ∮(view(ρv, :, 1, :), (1, 3)))
    area = sum(side -> boundary_total_area(side, grid), (:west, :east, :south, :north))

    return flux, area
end

# The invariant: after a completed step, the momentum the model LEAVES must satisfy the constraint its
# own pressure solve assumed. That is not free — every halo fill rewrites the boundary faces with the
# prescribed data, reverting an earlier correction, and Breeze fills momentum halos twice per stage.
#
# The imbalance has to come from the boundary CONDITIONS, not from writing the boundary plane before
# the step: a written plane does not survive the first halo fill.
@testset "prescribed lateral momentum is balanced after a step, if it opts in [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    try
        grid = bounded_grid(FT)
        Nx, Ny, _ = size(grid)

        # Deliberately unbalanced: 2 in through the west face, 1 out through the east. The south and
        # north faces are impenetrable walls and stay closed, so the two faces that opt in each shed
        # offset = A/2A = 0.5: west 2 → 1.5, east 1 → 1.5.
        imposed = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(convert(FT, 2)),
                                          east = NormalFlowBoundaryCondition(convert(FT, 1)))
        opted_in = open_x(convert(FT, 2), convert(FT, 1))

        for (ρu_bcs, opts_in) in ((imposed, false), (opted_in, true))
            model = anelastic_model(grid; boundary_conditions = (; ρu = ρu_bcs))
            set!(model, θ = 300)
            time_step!(model, 0.5)

            @test all(isfinite, Array(interior(model.velocities.u)))

            # The wall values, not just the integral: a reverted correction leaves the raw 2 and 1.
            u = Array(interior(model.momentum.ρu))
            v = Array(interior(model.momentum.ρv))
            @test all(u[1, :, :]      .≈ convert(FT, opts_in ? 1.5 : 2))
            @test all(u[Nx + 1, :, :] .≈ convert(FT, opts_in ? 1.5 : 1))
            @test all(iszero, v[:, 1, :])
            @test all(iszero, v[:, Ny + 1, :])

            imbalance, area = imbalance_and_area(model.momentum, grid)
            @test (abs(imbalance) / area < (FT === Float32 ? 1e-4 : 1e-6)) == opts_in
        end
    finally
        # Restore in a `finally`: a failure above would otherwise leak `Float32` into every later
        # testset in this file, and into whatever else this ParallelTestRunner worker picks up.
        Oceananigans.defaults.FloatType = Float64
    end
end

@testset "the correction is a no-op when no boundary opts in" begin
    # A closed domain, and any topology with a `Periodic` horizontal. A `Periodic` direction carries no
    # net flux, and an impenetrable wall refills to zero, so neither has anything to correct.
    for topo in ((Bounded, Bounded, Bounded), (Periodic, Periodic, Bounded),
                 (Periodic, Bounded, Bounded), (Bounded, Periodic, Bounded))
        grid = RectilinearGrid(default_arch; size = (8, 8, 8), x = (0, 80), y = (0, 80), z = (0, 80),
                               topology = topo)
        model = anelastic_model(grid; boundary_conditions = NamedTuple())
        set!(model.momentum.ρu, 3)
        set!(model.momentum.ρv, -1)
        before_u = copy(Array(interior(model.momentum.ρu)))
        before_v = copy(Array(interior(model.momentum.ρv)))

        enforce_lateral_mass_balance!(model.dynamics, model.momentum)

        @test Array(interior(model.momentum.ρu)) == before_u
        @test Array(interior(model.momentum.ρv)) == before_v
    end
end

@testset "the imbalance is split evenly over the faces that opt in" begin
    grid = bounded_grid()
    model = anelastic_model(grid)
    ρu, ρv = model.momentum.ρu, model.momentum.ρv
    Nx, Ny, _ = size(grid)

    # An already-balanced field must come back unchanged, not merely still balanced.
    set!(ρu, 1)
    set!(ρv, 0)
    before = copy(Array(interior(ρu)))
    enforce_lateral_mass_balance!(model.dynamics, model.momentum)
    @test Array(interior(ρu)) ≈ before

    # A net inflow through the west face alone, with all four faces opted in.
    fill!(parent(ρu), 0)
    fill!(parent(ρv), 0)
    view(interior(ρu), 1, :, :) .= 2
    imbalance, area = imbalance_and_area(model.momentum, grid)
    @test imbalance ≈ -2 * 160 * 120   # one face of 160 m × 120 m carrying momentum 2, inward
    @test area ≈ 4 * 160 * 120         # four faces of 160 m × 120 m
    offset = imbalance / area

    enforce_lateral_mass_balance!(model.dynamics, model.momentum)
    corrected, _ = imbalance_and_area(model.momentum, grid)
    @test corrected ≈ 0 atol = 1e-9

    u = Array(interior(ρu))
    v = Array(interior(ρv))
    @test all(u[1, :, :]      .≈ 2 + offset)
    @test all(u[Nx + 1, :, :] .≈ -offset)
    @test all(v[:, 1, :]      .≈ offset)
    @test all(v[:, Ny + 1, :] .≈ -offset)

    # The interior is untouched: this corrects boundary data, not the flow.
    @test all(iszero, u[2:Nx, :, :])

    # Idempotent — reapplying at the next halo fill must be a fixed point, not a second subtraction.
    again = copy(u)
    enforce_lateral_mass_balance!(model.dynamics, model.momentum)
    @test Array(interior(ρu)) ≈ again
end

@testset "a stretched vertical grid is area-weighted correctly" begin
    # Non-uniform Δz is the case where an unweighted sum would silently mis-balance.
    z = PiecewiseStretchedDiscretization(z = [0, 60, 240], Δz = [5.0, 5.0, 20.0])
    grid = RectilinearGrid(default_arch; size = (12, 12, length(z) - 1),
                           x = (0, 120), y = (0, 120), z,
                           topology = (Bounded, Bounded, Bounded))
    model = anelastic_model(grid)
    ρu, ρv = model.momentum.ρu, model.momentum.ρv

    fill!(parent(ρu), 0)
    fill!(parent(ρv), 0)
    view(interior(ρu), 1, :, :) .= 1

    imbalance, area = imbalance_and_area(model.momentum, grid)
    # One face of 120 m × 240 m carrying unit momentum.
    @test imbalance ≈ -120 * 240
    @test area ≈ 4 * 120 * 240

    enforce_lateral_mass_balance!(model.dynamics, model.momentum)
    corrected, _ = imbalance_and_area(model.momentum, grid)
    @test corrected ≈ 0 atol = 1e-8
end

# The offset is uniform in MOMENTUM, not in velocity, and that is what makes the flux integral exact
# rather than approximate — it is why `AtmosphereModels` hands the correction `model.momentum` rather
# than `model.velocities`. The flux integrates ρᵣu, so a constant subtracted from ρᵣu removes exactly
# `offset × area` whatever ρᵣ(z) does. A constant subtracted from the VELOCITY would remove
# `∫ ρᵣ(z) δ dA`, which equals the imbalance only if ρᵣ is constant over the column.
@testset "the offset is uniform in momentum, so the velocity increment scales as 1/ρᵣ" begin
    z = PiecewiseStretchedDiscretization(z = [0, 600, 3000], Δz = [50.0, 50.0, 200.0])
    grid = RectilinearGrid(default_arch; size = (8, 8, length(z) - 1),
                           x = (0, 800), y = (0, 800), z,
                           topology = (Bounded, Bounded, Bounded))
    model = anelastic_model(grid)
    ρu, ρv = model.momentum.ρu, model.momentum.ρv

    # The test is vacuous unless the reference density actually varies over the column.
    ρᵣ = Array(interior(model.dynamics.reference_state.density))
    @test maximum(ρᵣ) / minimum(ρᵣ) > 1.2

    fill!(parent(ρu), 0)
    fill!(parent(ρv), 0)
    view(interior(ρu), 1, :, :) .= 2

    imbalance, area = imbalance_and_area(model.momentum, grid)
    offset = imbalance / area
    enforce_lateral_mass_balance!(model.dynamics, model.momentum)

    # The momentum increment is the same at every height, which is the claim.
    u = Array(interior(ρu))
    @test all(u[1, :, :] .≈ 2 + offset)

    # So the velocity increment is NOT the same at every height: it spans the ρᵣ ratio.
    velocity_increment = offset ./ vec(ρᵣ)
    @test maximum(abs, velocity_increment) / minimum(abs, velocity_increment) ≈
          maximum(ρᵣ) / minimum(ρᵣ)

    # And the flux is removed to round-off despite that variation.
    corrected, _ = imbalance_and_area(model.momentum, grid)
    @test abs(corrected) / area < 1e-12
end

# A nested child over terrain is an `ImmersedBoundaryGrid` over a laterally-bounded regular grid.
# `pressure_solver_grid` drops the immersed boundary, so the solve runs on the underlying grid and the
# correction acts on the same boundary planes it would without terrain.
@testset "the correction runs on an immersed grid over a laterally-bounded one" begin
    underlying = bounded_grid()
    ridge(x, y) = 60 * exp(-(x / 40)^2)
    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(ridge))
    model = anelastic_model(grid)
    ρu, ρv = model.momentum.ρu, model.momentum.ρv

    fill!(parent(ρu), 0)
    fill!(parent(ρv), 0)
    view(interior(ρu), 1, :, :) .= 2

    imbalance, area = imbalance_and_area(model.momentum, grid)
    @test abs(imbalance) / area > 0.1   # vacuous unless it starts out unbalanced

    enforce_lateral_mass_balance!(model.dynamics, model.momentum)
    corrected, _ = imbalance_and_area(model.momentum, grid)
    @test abs(corrected) / area < 1e-12
end
