#####
##### Tests for the SubstepBoundaryUpdate specified-zone boundary update
#####

using Breeze
using Breeze.CompressibleEquations: SubstepBoundaryUpdate, boundary_tendencies,
                                    OpenSides, specified_zone_faces, specified_zone_cell,
                                    specified_zone_scheme, reimpose_specified_zone!,
                                    compute_contravariant_velocity!
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

    # All four sides specified: the face and cell sets match the truth tables.
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

    # Single specified side: only that side's band is marked.
    s_west = OpenSides(true, false, false, false)
    for j in 1:N, i in 1:N
        xspec, yspec = specified_zone_faces(i, j, grid, s_west)
        @test xspec == (i <= 2)
        @test yspec == (i <= 1)
        @test specified_zone_cell(i, j, grid, s_west) == (i == 1)
    end

    # No specified sides: nothing marked.
    s_off = OpenSides(false, false, false, false)
    for j in 1:N, i in 1:N
        xspec, yspec = specified_zone_faces(i, j, grid, s_off)
        @test !xspec && !yspec
        @test !specified_zone_cell(i, j, grid, s_off)
    end
end

@testset "Scheme detection and gated allocation" begin
    grid = bounded_grid(Float64)

    scheme = SubstepBoundaryUpdate()
    @test specified_zone_scheme(NormalFlowBoundaryCondition(0; scheme)) === scheme
    @test specified_zone_scheme(NormalFlowBoundaryCondition(0)) === nothing

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    ρv_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(0; scheme),
                                     north = NormalFlowBoundaryCondition(0; scheme))
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs, ρv = ρv_bcs))
    fields = boundary_tendencies(model)
    @test fields.ρu isa Field
    @test fields.ρᵈ isa Field

    # Without the scheme, no tendency storage is allocated.
    plain = AtmosphereModel(grid; dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization()))
    @test plain.timestepper.substepper.boundary_tendencies === nothing
end

@testset "Specified-zone composition recovers U⁰ + Δt·∂ₜ over a full RK3 step" begin
    # A constant-in-time tendency ∂ₜ(ρu) = a on the specified zone must advance
    # the recovered specified-face momentum by exactly Δt·a per outer step,
    # independent of the stage substep counts — the regression for the
    # increment-vs-overwrite composition (an overwrite compounds to
    # (β₁+β₂+β₃) = 11/6·Δt·a). Rest state ⇒ the only spec-face forcing is the update.
    grid = bounded_grid(Float64)
    a = 1e-4   # ∂ₜ(ρu) [kg m⁻² s⁻²]
    scheme = SubstepBoundaryUpdate()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)
    set!(boundary_tendencies(model).ρu, a)

    Δt = 10.0
    i, j, k = 2, 4, 2          # a west specified face away from corners
    ρu₀ = @allowscalar model.momentum.ρu[i, j, k]
    time_step!(model, Δt)
    ρu₁ = @allowscalar model.momentum.ρu[i, j, k]
    @test isapprox(ρu₁ - ρu₀, Δt * a; rtol=1e-10)

    # Second step: the update re-anchors to the specified state, so the advance
    # stays Δt·a per step (no compounding).
    time_step!(model, Δt)
    ρu₂ = @allowscalar model.momentum.ρu[i, j, k]
    @test isapprox(ρu₂ - ρu₁, Δt * a; rtol=1e-10)
end

@testset "Zero-tendency update holds a rest state" begin
    grid = bounded_grid(Float64)
    scheme = SubstepBoundaryUpdate()
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

@testset "reimpose_specified_zone! restores the specified zone" begin
    grid = bounded_grid(Float64)
    a = 1e-4
    scheme = SubstepBoundaryUpdate()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)
    set!(boundary_tendencies(model).ρu, a)

    Δt = 10.0
    time_step!(model, Δt)

    substepper = model.timestepper.substepper
    U⁰ = model.timestepper.U⁰
    ρu = model.momentum.ρu
    ρθ = model.formulation.potential_temperature_density

    # Scribble garbage into the zone (a specified face and a specified cell) and
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

@testset "Specified-zone composition with a vertically-implicit closure" begin
    # The implicit vertical solve runs after the substep loop over all columns,
    # including the specified zone; the re-imposition must discard its
    # increments there. A z-dependent tendency with curvature discriminates:
    # without the restore, the implicit diffusion (ν Δt/Δz² ≫ rtol) smooths the
    # specified profile and the composition fails.
    grid = bounded_grid(Float64)
    a = 1e-4
    Lz = 400.0
    scheme = SubstepBoundaryUpdate()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = 10)
    model = AtmosphereModel(grid; dynamics, closure, boundary_conditions = (ρu = ρu_bcs,))
    @test model.timestepper.implicit_solver !== nothing
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)
    set!(boundary_tendencies(model).ρu, (x, y, z) -> a * (z / Lz)^2)

    Δt = 10.0
    i, j = 2, 4                # a west specified face away from corners
    ρu = model.momentum.ρu
    ∂ₜρu = boundary_tendencies(model).ρu

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

@testset "Moisture specified-zone composition" begin
    # ρqᵛ never enters the acoustic loop — it is stepped per stage from the
    # slow tendencies — so its specified-zone update is carried entirely by the
    # re-imposition.
    grid = bounded_grid(Float64)
    b = 1e-7   # ∂ₜ(ρqᵛ) [kg m⁻³ s⁻¹]
    scheme = SubstepBoundaryUpdate()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
    ref = model.dynamics.reference_state
    set!(model; θ = 300, u = 0, ρ = ref.density)
    set!(boundary_tendencies(model).ρqᵛ, b)

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
    scheme = SubstepBoundaryUpdate()
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

@testset "Zero-tendency update holds a rest state with implicit closure" begin
    grid = bounded_grid(Float64)
    scheme = SubstepBoundaryUpdate()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    ρv_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(0; scheme),
                                     north = NormalFlowBoundaryCondition(0; scheme))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    # Momentum-only diffusion: κ ≠ 0 drifts this discrete rest state by ~1e-4
    # with or without the update (identically, scheme-independent) — a
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

#####
##### SubstepBoundaryUpdate over TerrainCompressibleDynamics (#839)
#####
##### Shared CPU terrain test problem (duplicated in-file per the two-PR plan — no shared
##### helper file, since `find_tests` auto-discovers every `test/*.jl`). The specified-zone-only
##### `SubstepBoundaryUpdate` is fully qualified so no extra import is needed. The hill
##### `h₀ sin(πx/Lx)` has zero height and MAX slope at both x-walls (discrete ccf wall
##### slope ≈ 0.039), so the specified west/east zone sits where the terrain correction is
##### strongest — the sharpest test of the specified-zone gating.

function terrain_testproblem_grid_and_dynamics(arch)
    Nx, Nz = 16, 8
    Lx, Lz = 16e3, 4e3
    h₀ = 400.0
    hill(x) = h₀ * sin(π * x / Lx)

    z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length = Nz + 1));
                                                     formulation = LinearDecay())
    grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                           x = (0, Lx), z = z_faces,
                           topology = (Bounded, Flat, Bounded))
    materialize_terrain!(grid, hill)

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(acoustic_cfl = 0.5);
                                    reference_potential_temperature = 300)
    return grid, dynamics
end

# Discrete, density-consistent ρu inflow momentum ρᵣ·U at the west/east faces, read from
# the model's OWN discrete terrain reference density (NOT a bare constant).
function terrain_inflow_momentum_columns(grid, dynamics, U)
    ρᵣ = AtmosphereModel(grid; dynamics).dynamics.terrain_reference_density
    Nx = size(grid, 1)
    ρᵣ_int = Array(interior(ρᵣ))          # (Nx, 1, Nz) on CPU
    west = U .* ρᵣ_int[1, 1, :]
    east = U .* ρᵣ_int[Nx, 1, :]
    return west, east
end

# (a) rest, (b) flowing, (c) specified.
function build_terrain_testproblem(variant; arch = CPU(), U = 10.0)
    grid, dynamics = terrain_testproblem_grid_and_dynamics(arch)

    if variant === :rest
        model = AtmosphereModel(grid; dynamics)     # default impenetrable walls
        set!(model; θ = 300, u = 0, ρ = model.dynamics.terrain_reference_density)
        return model
    end

    # Density-consistent DISCRETE inflow momentum ρu = ρᵣ·U at the inflow faces.
    ρu_west, ρu_east = terrain_inflow_momentum_columns(grid, dynamics, U)
    Nx = size(grid, 1)
    west_value(j, k, grid, clock, fields) = @inbounds ρu_west[k]
    east_value(j, k, grid, clock, fields) = @inbounds ρu_east[k]

    if variant === :flowing
        ρu_bcs = FieldBoundaryConditions(
            west = NormalFlowBoundaryCondition(west_value; discrete_form = true),
            east = NormalFlowBoundaryCondition(east_value; discrete_form = true))
        model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
        set!(model; θ = 300, u = U, ρ = model.dynamics.terrain_reference_density,
             enforce_mass_conservation = false)
        return model

    elseif variant === :specified
        scheme = Breeze.CompressibleEquations.SubstepBoundaryUpdate()   # fieldless marker
        ρu_bcs = FieldBoundaryConditions(
            west = NormalFlowBoundaryCondition(west_value; discrete_form = true, scheme),
            east = NormalFlowBoundaryCondition(east_value; discrete_form = true, scheme))
        model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
        set!(model; θ = 300, u = U, ρ = model.dynamics.terrain_reference_density,
             enforce_mass_conservation = false)
        return model
    end
    error("unknown variant $variant")
end

# A specified-REST terrain model: the shared terrain grid/dynamics with a
# `SubstepBoundaryUpdate` west/east zone but zero inflow, so the only specified-zone
# forcing is whatever `boundary_tendencies` supplies. Used by the three terrain
# specified-zone testsets below (a flowing base would confound the clean Δρu = Δt·∂ₜ composition).
function terrain_specified_rest_model(; arch = CPU())
    grid, dynamics = terrain_testproblem_grid_and_dynamics(arch)
    scheme = SubstepBoundaryUpdate()
    ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme),
                                     east = NormalFlowBoundaryCondition(0; scheme))
    model = AtmosphereModel(grid; dynamics, boundary_conditions = (ρu = ρu_bcs,))
    set!(model; θ = 300, u = 0, ρ = model.dynamics.terrain_reference_density)
    return model
end

@testset "Terrain specified zone holds a rest state" begin
    # The specified west/east zone with zero boundary tendencies must hold the discrete
    # terrain rest state to machine precision — the contravariant w̃ (the terrain
    # transport velocity) included, since a leak in the slope-projected acoustic/slow
    # terms would show there first. `Δt = 10` from `acoustic_cfl = 0.5`.
    model = terrain_specified_rest_model()
    w̃ = model.dynamics.contravariant_vertical_velocity
    ρᵈ = model.dynamics.dry_density
    mass₀ = sum(interior(ρᵈ))

    for _ in 1:5
        time_step!(model, 10.0)
    end
    compute_contravariant_velocity!(model)

    @test maximum(abs, interior(model.velocities.u)) < 1e-10
    @test maximum(abs, interior(model.velocities.w)) < 1e-10
    @test maximum(abs, interior(w̃)) < 1e-10
    @test abs(sum(interior(ρᵈ)) - mass₀) / mass₀ ≤ 1e-13
end

@testset "Terrain specified-zone composition recovers Δρu = Δt·∂ₜ" begin
    # A constant ∂ₜ(ρu) = a on the specified zone advances the recovered west-face
    # momentum by exactly Δt·a per outer step over terrain, just as on a flat grid —
    # the terrain slope corrections must not perturb the increment. A supplied-zero
    # ρθ tendency holds the zone's thermodynamic density exactly. Flat-y ⇒ j = 1.
    model = terrain_specified_rest_model()
    a = 1e-4
    set!(boundary_tendencies(model).ρu, a)

    Δt = 10.0
    i, j, k = 2, 1, 2          # a west specified face (i ≤ 2), Flat-y column
    ρu = model.momentum.ρu
    ρθ = model.formulation.potential_temperature_density
    zone_ρθ₀ = @allowscalar ρθ[1, 1, 2]

    ρu₀ = @allowscalar ρu[i, j, k]
    time_step!(model, Δt)
    ρu₁ = @allowscalar ρu[i, j, k]
    @test isapprox(ρu₁ - ρu₀, Δt * a; rtol = 1e-10)

    time_step!(model, Δt)
    ρu₂ = @allowscalar ρu[i, j, k]
    @test isapprox(ρu₂ - ρu₁, Δt * a; rtol = 1e-10)

    @test (@allowscalar ρθ[1, 1, 2]) == zone_ρθ₀   # zero-tendency ρθ held exactly
end

@testset "Terrain specified zone does not leak specified-zone scalars into the interior" begin
    # Discriminator for the terrain slope-correction gating (Path A + the slow-w̃
    # pressure/momentum substitution): heat ONLY the specified zone via a ρθ tendency.
    # The zone updates by Δt·∂ₜ, but with the terrain horizontal-PGF stencils gated on
    # the specified side the heated cell's perturbation pressure must NOT project into any
    # interior column's vertical momentum. An ungated leak drives w ~ 1e-3 at column 2
    # (≈ 7 orders above the machine-level bound here).
    model = terrain_specified_rest_model()
    set!(boundary_tendencies(model).ρθ, 1e-2)

    Δt = 10.0
    ρθ = model.formulation.potential_temperature_density
    zone₀ = @allowscalar ρθ[1, 1, 2]
    time_step!(model, Δt)
    zone₁ = @allowscalar ρθ[1, 1, 2]

    @test isapprox(zone₁ - zone₀, Δt * 1e-2; rtol = 1e-10)      # zone updated
    @test maximum(abs, interior(model.velocities.w)) < 1e-10    # interior did not leak
end
