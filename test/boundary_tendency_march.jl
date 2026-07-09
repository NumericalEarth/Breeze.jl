#####
##### Tests for the BoundaryTendencyMarch specified-zone boundary drive (#825)
#####

using Breeze
using Breeze.CompressibleEquations: BoundaryTendencyMarch, boundary_tendency_fields,
                                    OpenSides, specified_zone_faces, specified_zone_cell,
                                    march_scheme, reimpose_specified_zone!
using Breeze.Microphysics: InstantaneousPrecipitation, WarmPhaseEquilibrium
using Oceananigans
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: interior
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
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

@testset "reimpose_specified_zone! restores the marched zone" begin
    grid = bounded_grid(Float64)
    a = 1e-4
    scheme = BoundaryTendencyMarch(ρu_tendency=(x, y, z, t) -> a)
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)

    Δt = 10.0
    time_step!(model, Δt)

    substepper = model.timestepper.substepper
    U⁰ = model.timestepper.U⁰
    ρu = model.momentum.ρu
    ρθ = model.formulation.potential_temperature_density

    # Scribble garbage into the zone (a marched face and a specified cell) and
    # a sentinel into the interior, then restore.
    sentinel = 7.89
    @allowscalar begin
        ρu[2, 4, 2] = 123.0
        ρθ[1, 4, 2] = 456.0
        ρu[5, 5, 2] = sentinel
    end
    reimpose_specified_zone!(substepper, model, Δt)
    @allowscalar begin
        @test isapprox(ρu[2, 4, 2], U⁰.ρu[2, 4, 2] + Δt * a; rtol=1e-12)
        @test ρθ[1, 4, 2] == U⁰.ρθ[1, 4, 2]   # zero-tendency hold restored exactly
        @test ρu[5, 5, 2] == sentinel          # interior untouched
    end
end

@testset "March composition with a vertically-implicit closure" begin
    # The implicit vertical solve runs after the substep loop over all columns,
    # including the specified zone; the re-imposition must discard its
    # increments there. A z-dependent tendency with curvature discriminates:
    # without the restore, the implicit diffusion (ν Δt/Δz² ≫ rtol) smooths the
    # marched profile and the composition fails.
    grid = bounded_grid(Float64)
    a = 1e-4
    Lz = 400.0
    scheme = BoundaryTendencyMarch(ρu_tendency=(x, y, z, t) -> a * (z / Lz)^2)
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = 10)
    model = AtmosphereModel(grid; dynamics, closure, boundary_conditions = (ρu = ρu_bcs,))
    @test model.timestepper.implicit_solver !== nothing
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)

    Δt = 10.0
    i, j = 2, 4                # a west specified face away from corners
    ρu = model.momentum.ρu
    ∂ₜρu = boundary_tendency_fields(model).ρu

    ρu₀ = @allowscalar [ρu[i, j, k] for k in 1:4]
    time_step!(model, Δt)
    ρu₁ = @allowscalar [ρu[i, j, k] for k in 1:4]
    time_step!(model, Δt)
    ρu₂ = @allowscalar [ρu[i, j, k] for k in 1:4]

    for k in 1:4
        expected = Δt * @allowscalar ∂ₜρu[i, j, k]
        @test isapprox(ρu₁[k] - ρu₀[k], expected; rtol=1e-10)
        @test isapprox(ρu₂[k] - ρu₁[k], expected; rtol=1e-10)
    end
end

@testset "Moisture march composition" begin
    # ρqᵛ never enters the acoustic loop — it is stepped per stage from the
    # slow tendencies — so its specified-zone march is carried entirely by the
    # re-imposition.
    grid = bounded_grid(Float64)
    b = 1e-7   # ∂ₜ(ρqᵛ) [kg m⁻³ s⁻¹]
    scheme = BoundaryTendencyMarch(ρqᵛ_tendency=(x, y, z, t) -> b)
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)

    Δt = 10.0
    ρqᵛ = model.moisture_density
    q₀ = @allowscalar ρqᵛ[1, 4, 2]      # a west specified cell
    time_step!(model, Δt)
    q₁ = @allowscalar ρqᵛ[1, 4, 2]
    time_step!(model, Δt)
    q₂ = @allowscalar ρqᵛ[1, 4, 2]

    @test isapprox(q₁ - q₀, Δt * b; rtol=1e-10)
    @test isapprox(q₂ - q₁, Δt * b; rtol=1e-10)
    @test abs(@allowscalar ρqᵛ[5, 5, 2]) < 1e-12   # interior moisture stays zero
end

@testset "Specified zone holds through operator-split microphysics" begin
    # `InstantaneousPrecipitation` mutates ρθ and ρqᵛ once per step over all
    # interior cells; the once-per-step re-imposition must restore the zone's
    # frozen hold (all tendency sources `nothing`) while the interior
    # precipitates freely.
    grid = bounded_grid(Float64)
    scheme = BoundaryTendencyMarch()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    microphysics = InstantaneousPrecipitation(equilibrium = WarmPhaseEquilibrium())
    model = AtmosphereModel(grid; dynamics, microphysics, boundary_conditions = (ρu = ρu_bcs,))
    ref = model.dynamics.reference_state
    set!(model; ρ = ref.density, θ = 300, qᵗ = 0.030)   # supersaturated everywhere

    ρθ = model.formulation.potential_temperature_density
    ρqᵛ = model.moisture_density
    zone_ρθ₀, zone_ρq₀ = @allowscalar (ρθ[1, 4, 2], ρqᵛ[1, 4, 2])
    int_ρθ₀, int_ρq₀ = @allowscalar (ρθ[5, 5, 2], ρqᵛ[5, 5, 2])

    time_step!(model, 1.0)

    @allowscalar begin
        @test ρθ[1, 4, 2] == zone_ρθ₀     # zone held exactly through the update
        @test ρqᵛ[1, 4, 2] == zone_ρq₀
        @test ρθ[5, 5, 2] > int_ρθ₀       # interior condensed: latent warming...
        @test ρqᵛ[5, 5, 2] < int_ρq₀      # ...and vapor rained out
    end
end

@testset "Zero-tendency march holds a rest state with implicit closure" begin
    grid = bounded_grid(Float64)
    scheme = BoundaryTendencyMarch()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    ρv_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(0; scheme),
                                     north = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    # Momentum-only diffusion: κ ≠ 0 drifts this discrete rest state by ~1e-4
    # with or without the march (identically, scheme-independent) — a
    # pre-existing θ-diffusion/stratified-reference interaction, not a zone
    # property. ν exercises the implicit solve + per-stage re-imposition.
    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = 1)
    model = AtmosphereModel(grid; dynamics, closure, boundary_conditions = (ρu = ρu_bcs, ρv = ρv_bcs))
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)

    for _ in 1:5
        time_step!(model, 10.0)
    end
    @test maximum(abs, interior(model.velocities.u)) < 1e-10
    @test maximum(abs, interior(model.velocities.w)) < 1e-10
end
