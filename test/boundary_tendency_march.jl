#####
##### Tests for the BoundaryTendencyMarch specified-zone boundary drive (#825)
#####

using Breeze
using Breeze.CompressibleEquations: BoundaryTendencyMarch, boundary_tendency_fields,
                                    OpenSides, specified_zone_faces, specified_zone_cell,
                                    march_scheme
using Oceananigans
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: interior
using GPUArraysCore: @allowscalar
using Test

bounded_grid(FT; Nz=4, Lz=400) =
    RectilinearGrid(CPU(), FT; size = (8, 8, Nz), halo = (3, 3, 3),
                    x = (0, 8_000), y = (0, 8_000), z = (0, Lz),
                    topology = (Bounded, Bounded, Bounded))

@testset "Specified-zone predicates" begin
    grid = bounded_grid(Float64)
    N = 8

    # All four sides marched: the face and cell sets match the truth tables.
    s = OpenSides(true, true, true, true)
    xspec_truth(i, j) = i <= 2 || i == N || j == 1 || j == N
    yspec_truth(i, j) = j <= 2 || j == N || i == 1 || i == N
    cell_truth(i, j)  = i == 1 || i == N || j == 1 || j == N
    for j in 1:N, i in 1:N
        xspec, yspec = specified_zone_faces(i, j, grid, s)
        @test xspec == xspec_truth(i, j)
        @test yspec == yspec_truth(i, j)
        @test specified_zone_cell(i, j, grid, s) == cell_truth(i, j)
    end

    # Single marched side: only that side's band is marked.
    s_west = OpenSides(true, false, false, false)
    for j in 1:N, i in 1:N
        xspec, yspec = specified_zone_faces(i, j, grid, s_west)
        @test xspec == (i <= 2)
        @test yspec == (i <= 1)
        @test specified_zone_cell(i, j, grid, s_west) == (i == 1)
    end

    # No marched sides: nothing marked.
    s_off = OpenSides(false, false, false, false)
    for j in 1:N, i in 1:N
        xspec, yspec = specified_zone_faces(i, j, grid, s_off)
        @test !xspec && !yspec
        @test !specified_zone_cell(i, j, grid, s_off)
    end
end

@testset "Scheme detection and gated allocation" begin
    grid = bounded_grid(Float64)

    scheme = BoundaryTendencyMarch()
    @test march_scheme(NormalFlowBoundaryCondition(0; scheme)) === scheme
    @test march_scheme(NormalFlowBoundaryCondition(0)) === nothing

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    ρv_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(0; scheme),
                                     north = NormalFlowBoundaryCondition(0; scheme))
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs, ρv = ρv_bcs))
    fields = boundary_tendency_fields(model)
    @test fields.ρu isa Field
    @test fields.ρᵈ isa Field

    # Without the scheme, no tendency storage is allocated.
    plain = AtmosphereModel(grid; dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization()))
    @test plain.timestepper.substepper.boundary_momentum_tendency_u === nothing
end

@testset "March composition recovers U⁰ + Δt·∂ₜ over a full RK3 step" begin
    # A constant-in-time tendency ∂ₜ(ρu) = a on the specified zone must advance
    # the recovered specified-face momentum by exactly Δt·a per outer step,
    # independent of the stage substep counts — the regression for the
    # increment-vs-overwrite composition (an overwrite compounds to
    # (β₁+β₂+β₃) = 11/6·Δt·a). Rest state ⇒ the only spec-face forcing is the march.
    grid = bounded_grid(Float64)
    a = 1e-4   # ∂ₜ(ρu) [kg m⁻² s⁻²]
    scheme = BoundaryTendencyMarch(ρu_tendency=(x, y, z, t) -> a)
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)

    Δt = 10.0
    i, j, k = 2, 4, 2          # a west specified face away from corners
    ρu₀ = @allowscalar model.momentum.ρu[i, j, k]
    time_step!(model, Δt)
    ρu₁ = @allowscalar model.momentum.ρu[i, j, k]
    @test isapprox(ρu₁ - ρu₀, Δt * a; rtol=1e-10)

    # Second step: the march re-anchors to the marched state, so the advance
    # stays Δt·a per step (no compounding).
    time_step!(model, Δt)
    ρu₂ = @allowscalar model.momentum.ρu[i, j, k]
    @test isapprox(ρu₂ - ρu₁, Δt * a; rtol=1e-10)
end

@testset "Zero-tendency march holds a rest state" begin
    grid = bounded_grid(Float64)
    scheme = BoundaryTendencyMarch()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    ρv_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(0; scheme),
                                     north = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs, ρv = ρv_bcs))
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)

    for _ in 1:5
        time_step!(model, 10.0)
    end
    @test maximum(abs, interior(model.velocities.u)) < 1e-10
    @test maximum(abs, interior(model.velocities.w)) < 1e-10
end
