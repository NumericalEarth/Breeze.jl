include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze: ReferenceState, AnelasticDynamics
using Breeze.AnelasticEquations: lateral_mass_imbalance
using Breeze.AtmosphereModels: enforce_lateral_mass_balance!
using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, NormalFlowBoundaryCondition
using Oceananigans.Fields: interior
using Oceananigans.Grids: AbstractGrid, XYRegularRG
using Test

bounded_grid(FT = Float64; size = (16, 16, 12)) =
    RectilinearGrid(default_arch, FT; size, x = (0, 160), y = (0, 160), z = (0, 120),
                    topology = (Bounded, Bounded, Bounded))

# Open lateral faces: an imposed zero normal flow, which the correction may shift. The default
# impenetrable walls stay closed, so tests that write the boundary planes by hand need these.
open_x() = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0), east = NormalFlowBoundaryCondition(0))
open_y() = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(0), north = NormalFlowBoundaryCondition(0))

anelastic_model(grid; boundary_conditions = (; ρu = open_x(), ρv = open_y())) = AtmosphereModel(grid;
    dynamics = AnelasticDynamics(ReferenceState(grid, ThermodynamicConstants(eltype(grid));
                                                base_pressure = 1e5, potential_temperature = 300)),
    boundary_conditions)

# The net imbalance and the total area of the four lateral faces
function imbalance_and_area(ρu, ρv, grid)
    imbalance, zonal_area, meridional_area = lateral_mass_imbalance(ρu, ρv, grid)
    return imbalance, 2zonal_area + 2meridional_area
end

@testset "lateral_mass_imbalance" begin
    grid = bounded_grid()
    model = anelastic_model(grid)
    ρu, ρv = model.momentum.ρu, model.momentum.ρv

    # A uniform through-flow is already balanced: what enters the west face leaves the east.
    set!(ρu, 1)
    set!(ρv, 0)
    imbalance, area = imbalance_and_area(ρu, ρv, grid)
    @test imbalance ≈ 0 atol=1e-10
    # Four faces of 160 m × 120 m.
    @test area ≈ 4 * 160 * 120

    # Inflow with no matching outflow is a net convergence, reported negative (outward normal).
    fill!(parent(ρu), 0)
    view(interior(ρu), 1, :, :) .= 2
    imbalance, area = imbalance_and_area(ρu, ρv, grid)
    @test imbalance ≈ -2 * 160 * 120
    @test imbalance / area ≈ -0.5

    # Outflow through the north face alone is a net divergence, reported positive.
    fill!(parent(ρu), 0)
    fill!(parent(ρv), 0)
    view(interior(ρv), :, size(grid, 2) + 1, :) .= 3
    imbalance, _ = imbalance_and_area(ρu, ρv, grid)
    @test imbalance ≈ 3 * 160 * 120
end

@testset "enforce_lateral_mass_balance!" begin
    grid = bounded_grid()
    model = anelastic_model(grid)
    ρu, ρv = model.momentum.ρu, model.momentum.ρv
    Nx, Ny, _ = size(grid)

    # An already-balanced field must come back unchanged, not merely still balanced.
    set!(ρu, 1)
    set!(ρv, 0)
    before = copy(Array(interior(ρu)))
    enforce_lateral_mass_balance!(model.dynamics, model.momentum, grid)
    @test Array(interior(ρu)) ≈ before

    # A net inflow is removed exactly, and the correction is split evenly over all four faces.
    fill!(parent(ρu), 0)
    fill!(parent(ρv), 0)
    view(interior(ρu), 1, :, :) .= 2
    imbalance, area = imbalance_and_area(ρu, ρv, grid)
    offset = imbalance / area
    enforce_lateral_mass_balance!(model.dynamics, model.momentum, grid)

    corrected, _ = imbalance_and_area(ρu, ρv, grid)
    @test corrected ≈ 0 atol=1e-9

    u = Array(interior(ρu))
    v = Array(interior(ρv))
    @test all(u[1, :, :] .≈ 2 + offset)
    @test all(u[Nx + 1, :, :] .≈ -offset)
    @test all(v[:, 1, :] .≈ offset)
    @test all(v[:, Ny + 1, :] .≈ -offset)

    # The interior is untouched: this corrects boundary data, not the flow.
    @test all(iszero, u[2:Nx, :, :])

    # Idempotent — a second call on a balanced field changes nothing.
    again = copy(u)
    enforce_lateral_mass_balance!(model.dynamics, model.momentum, grid)
    @test Array(interior(ρu)) ≈ again
end

@testset "enforce_lateral_mass_balance! is a no-op without bounded laterals" begin
    for topology in ((Periodic, Periodic, Bounded), (Periodic, Bounded, Bounded),
                     (Bounded, Periodic, Bounded))
        grid = RectilinearGrid(default_arch; size = (8, 8, 8), x = (0, 80), y = (0, 80), z = (0, 80),
                               topology = topology)
        model = anelastic_model(grid; boundary_conditions = NamedTuple())
        set!(model.momentum.ρu, 3)
        set!(model.momentum.ρv, -1)
        before_u = copy(Array(interior(model.momentum.ρu)))
        before_v = copy(Array(interior(model.momentum.ρv)))

        enforce_lateral_mass_balance!(model.dynamics, model.momentum, grid)

        @test Array(interior(model.momentum.ρu)) == before_u
        @test Array(interior(model.momentum.ρv)) == before_v
    end
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

    imbalance, area = imbalance_and_area(ρu, ρv, grid)
    # One face of 120 m × 240 m carrying unit momentum.
    @test imbalance ≈ -120 * 240
    @test area ≈ 4 * 120 * 240

    enforce_lateral_mass_balance!(model.dynamics, model.momentum, grid)
    corrected, _ = imbalance_and_area(ρu, ρv, grid)
    @test corrected ≈ 0 atol=1e-8
end

# The device reduction factors `Δy` out of the sum over `j`, which holds only for uniform horizontal
# spacing; a stretched horizontal takes the host path that weights every face individually.
#
# That path cannot be reached through a model: `FourierTridiagonalPoissonSolver` requires regularly
# spaced transform directions, so an anelastic model on a laterally-bounded grid ALWAYS has uniform
# horizontals — including over terrain, since an `ImmersedBoundaryGrid` forwards to its underlying grid.
# The fallback exists because `lateral_mass_imbalance` is also usable standalone, so it is tested that
# way, on hand-built momentum fields with no model and no Poisson solver.
@testset "a stretched HORIZONTAL grid takes the host path and still weights correctly" begin
    # Faces at 0, 10, 30, 60, 100: widths 10, 20, 30, 40 — deliberately non-uniform.
    grid = RectilinearGrid(default_arch; size = (4, 4, 4), x = [0.0, 10.0, 30.0, 60.0, 100.0],
                           y = (0, 80), z = (0, 40), topology = (Bounded, Bounded, Bounded))
    ρu, ρv = XFaceField(grid), YFaceField(grid)

    # Unit outflow through the north face: flux = Σᵢ Δxᵢ × Lz = 100 × 40.
    view(interior(ρv), :, size(grid, 2) + 1, :) .= 1
    imbalance, area = imbalance_and_area(ρu, ρv, grid)
    @test imbalance ≈ 100 * 40
    @test area ≈ 2 * (100 * 40) + 2 * (80 * 40)

    # Each wall loaded ALONE. Loading only the north face, as this testset used to, leaves both wall
    # pairs' signs in this method untested: flipping `east - west` or `north - south` to a sum here is
    # invisible when three of the four planes are zero.
    zonal_area      = 80 * 40     # a west or east face: Ly × Lz
    meridional_area = 100 * 40    # a south or north face: Σᵢ Δxᵢ × Lz

    for (momentum, dim, face_area) in ((ρu, 1, zonal_area), (ρv, 2, meridional_area))
        for (index, direction) in ((1, -1), (size(grid, dim) + 1, +1))
            fill!(parent(ρu), 0)
            fill!(parent(ρv), 0)
            selectdim(interior(momentum), dim, index) .= 1
            wall, _ = imbalance_and_area(ρu, ρv, grid)
            @test wall ≈ direction * face_area
        end
    end

    fill!(parent(ρu), 0)
    fill!(parent(ρv), 0)
    view(interior(ρv), :, size(grid, 2) + 1, :) .= 1

    # The case a uniform shortcut would get wrong: outflow through the WIDEST column only. Uniform
    # weighting would call every column 25 m wide and report 25 × 40.
    fill!(parent(ρv), 0)
    view(interior(ρv), 4, size(grid, 2) + 1, :) .= 1     # Δx = 40 there, of 100 total
    weighted, _ = imbalance_and_area(ρu, ρv, grid)
    @test weighted ≈ 40 * 40
    @test !isapprox(weighted, 25 * 40)
end

# The defect this guards: `xspacings` returns a lazy `KernelFunctionOperation`, never a number and never
# a vector, so selecting the device reduction with a `Δx isa Number` test was always false. The fast path
# was dead code, every call fell to the general one, and indexing a lazy operation per boundary face made
# the whole function infer as `Tuple{Any, Any}` — one dynamic dispatch and one boxed copy of the grid per
# face. The general method's strided plane copies dominate the call on a GPU, so a dead fast path is
# expensive rather than merely untidy. Distinct methods is the
# check that neither path is dead; a concrete return type is the check that nothing in the call infers as
# `Any`, which `interior` does and the `::Matrix{FT}` assertions in the source exist to repair.
@testset "each spacing regime reaches its own method, type stably" begin
    uniform = bounded_grid()
    stretched = RectilinearGrid(default_arch; size = (4, 4, 4), x = [0.0, 10.0, 30.0, 60.0, 100.0],
                                y = (0, 80), z = (0, 40), topology = (Bounded, Bounded, Bounded))

    signature(grid) = typeof.((XFaceField(grid), YFaceField(grid), grid))

    for grid in (uniform, stretched)
        FT = eltype(grid)
        @test only(Base.return_types(lateral_mass_imbalance, signature(grid))) == Tuple{FT, FT, FT}
    end

    # An anelastic model always has uniform horizontals, so it must select the device reduction.
    @test which(lateral_mass_imbalance, signature(uniform)) !==
          which(lateral_mass_imbalance, signature(stretched))
end

@testset "the pressure solve sees a balanced boundary [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    try
        grid = bounded_grid(FT)

        # The imbalance has to come from PRESCRIBED boundary conditions, not from writing the boundary
        # plane before the step. A written plane does not survive: the first halo fill rewrites it from
        # the default impenetrable condition, which is zero and so already balanced, and every assertion
        # below then holds no matter what the correction does.
        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(convert(FT, 2)),
                                         east = NormalFlowBoundaryCondition(convert(FT, 1)))
        model = AtmosphereModel(grid;
            dynamics = AnelasticDynamics(ReferenceState(grid, ThermodynamicConstants(FT);
                                                        base_pressure = 1e5, potential_temperature = 300)),
            boundary_conditions = (; ρu = ρu_bcs))

        set!(model, θ = 300)
        time_step!(model, 0.5)

        @test all(isfinite, Array(interior(model.velocities.u)))

        # The wall values, not just the integral: a reverted correction leaves the raw 2 and 1 here.
        Nx, _, _ = size(grid)
        u = Array(interior(model.momentum.ρu))
        @test all(u[1, :, :]      .≈ convert(FT, 1.5))
        @test all(u[Nx + 1, :, :] .≈ convert(FT, 1.5))

        imbalance, area = imbalance_and_area(model.momentum.ρu, model.momentum.ρv, grid)
        @test abs(imbalance) / area < (FT === Float32 ? 1e-4 : 1e-6)
    finally
        # Restore in a `finally`: a failure above would otherwise leak `Float32` into every later testset
        # in this file, and into whatever else this ParallelTestRunner worker picks up.
        Oceananigans.defaults.FloatType = Float64
    end
end

# The test above cannot catch a correction that is reverted, because its walls carry no PRESCRIBED
# data: the default impenetrable condition refills them with zero, which is already balanced. A nested
# child is the opposite — every halo fill rewrites the boundary faces with the parent's momentum, so a
# correction applied once per stage inside `compute_pressure_correction!` is undone by the fill in
# `compute_velocities!`. The state then carries a flux discontinuity of `imbalance/area` on every
# boundary face, and twice that on each corner cell, where two normal components are prescribed at once.
# Measured on a 16³ domain over twenty steps, correcting at the first site alone gives a corner/edge
# divergence ratio of 2.000 and max|w| = 1.83 m s⁻¹ — worse than applying no correction at all, 0.689 —
# against 0.0476 with both sites corrected.
#
# The invariant: after a completed step, the momentum the model LEAVES must satisfy the lateral
# constraint its own pressure solve assumed.
@testset "prescribed lateral momentum stays balanced after a step" begin
    grid = bounded_grid()

    # Deliberately unbalanced: 2 in through the west face, 1 out through the east.
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(2.0),
                                     east = NormalFlowBoundaryCondition(1.0))
    model = AtmosphereModel(grid;
        dynamics = AnelasticDynamics(ReferenceState(grid, ThermodynamicConstants(eltype(grid));
                                                    base_pressure = 1e5, potential_temperature = 300)),
        boundary_conditions = (; ρu = ρu_bcs))

    set!(model, θ = 300)
    time_step!(model, 0.5)

    # Net outward flux of the raw data is (−2 + 1)A. The south and north faces are impenetrable walls
    # and stay closed, so the two open faces of area A each shed offset = −A/2A = −0.5: west 2 → 1.5,
    # east 1 → 1.5. Testing the values, not just the integral, is what distinguishes "the correction
    # survived the halo fill" from "the walls happened to balance".
    Nx, Ny, _ = size(grid)
    u = Array(interior(model.momentum.ρu))
    v = Array(interior(model.momentum.ρv))
    @test all(u[1, :, :]      .≈ 1.5)
    @test all(u[Nx + 1, :, :] .≈ 1.5)
    @test all(iszero, v[:, 1, :])
    @test all(iszero, v[:, Ny + 1, :])

    balanced, area = imbalance_and_area(model.momentum.ρu, model.momentum.ρv, grid)
    @test abs(balanced) / area < 1e-6
    @test all(isfinite, Array(interior(model.velocities.w)))
end

# The offset is uniform in MOMENTUM, not in velocity, and that is what makes the flux integral exact
# rather than approximate. The flux integrates ρᵣu, so a constant subtracted from ρᵣu removes exactly
# `offset × area` whatever ρᵣ(z) does. A constant subtracted from the VELOCITY would remove
# `∫ ρᵣ(z) δ dA`, which equals the imbalance only if ρᵣ is constant over the column. Nothing else in
# this file touches ρᵣ, so the design claim the implementation rests on would otherwise go untested.
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

    imbalance, area = imbalance_and_area(ρu, ρv, grid)
    offset = imbalance / area
    enforce_lateral_mass_balance!(model.dynamics, model.momentum, grid)

    # The momentum increment is the same at every height, which is the claim.
    u = Array(interior(ρu))
    @test all(u[1, :, :] .≈ 2 + offset)

    # So the velocity increment is NOT the same at every height: it spans the ρᵣ ratio.
    velocity_increment = offset ./ vec(ρᵣ)
    @test maximum(abs, velocity_increment) / minimum(abs, velocity_increment) ≈
          maximum(ρᵣ) / minimum(ρᵣ)

    # And the flux is removed to round-off despite that variation.
    corrected, _ = imbalance_and_area(ρu, ρv, grid)
    @test abs(corrected) / area < 1e-12
end

# `pressure_solver_grid` drops the immersed boundary, so a model over terrain solves its Poisson problem
# on the underlying regular grid and must take the same device reduction. Without a method of its own an
# `ImmersedBoundaryGrid` matches `LaterallyBoundedGrid` but not `XYRegularRG`, so it would fall silently
# to the host loop — and pay a host copy of `bottom_height` and `bottom_cell_index` on all six calls a
# step. A nested child over terrain is exactly this grid.
@testset "an immersed grid over a regular one reaches the device reduction" begin
    underlying = bounded_grid()
    bottom(x, y) = 20 * exp(-((x - 80)^2 + (y - 80)^2) / 1600)
    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))

    # The combination that makes the method necessary.
    @test grid isa AbstractGrid{<:Any, Bounded, Bounded}
    @test !(grid isa XYRegularRG)

    signature(g) = typeof.((XFaceField(g), YFaceField(g), g))
    FT = eltype(grid)
    @test only(Base.return_types(lateral_mass_imbalance, signature(grid))) == Tuple{FT, FT, FT}

    # The assertion that fails without a method for the immersed grid: it must NOT land on the general
    # host loop that a stretched horizontal takes. Checking only that the two answers agree would pass
    # either way, because the two methods are equivalent on a uniform grid.
    stretched = RectilinearGrid(default_arch; size = (4, 4, 4), x = [0.0, 10.0, 30.0, 60.0, 100.0],
                                y = (0, 80), z = (0, 40), topology = (Bounded, Bounded, Bounded))
    @test which(lateral_mass_imbalance, signature(grid)) !==
          which(lateral_mass_imbalance, signature(stretched))

    # The immersed boundary does not enter the solvability condition: every cell is active in the solve,
    # so the same wall data must give the same imbalance with or without the terrain.
    open_u = FieldBoundaryConditions(grid, (Face(), Center(), Center());
                                     west = NormalFlowBoundaryCondition(0), east = NormalFlowBoundaryCondition(0))
    open_v = FieldBoundaryConditions(grid, (Center(), Face(), Center());
                                     south = NormalFlowBoundaryCondition(0), north = NormalFlowBoundaryCondition(0))
    immersed_u, immersed_v = XFaceField(grid; boundary_conditions = open_u), YFaceField(grid; boundary_conditions = open_v)
    plain_u, plain_v = XFaceField(underlying), YFaceField(underlying)
    for (u, v) in ((immersed_u, immersed_v), (plain_u, plain_v))
        view(interior(u), 1, :, :) .= 2
        view(interior(v), :, size(grid, 2) + 1, :) .= 1
    end
    @test lateral_mass_imbalance(immersed_u, immersed_v, grid) ==
          lateral_mass_imbalance(plain_u, plain_v, underlying)

    # And the correction itself works through the immersed grid.
    enforce_lateral_mass_balance!(AnelasticDynamics(ReferenceState(underlying,
                                      ThermodynamicConstants(FT); base_pressure = 1e5,
                                      potential_temperature = 300)),
                                  (; ρu = immersed_u, ρv = immersed_v), grid)
    corrected, area = imbalance_and_area(immersed_u, immersed_v, grid)
    @test abs(corrected) / area < 1e-12
end
