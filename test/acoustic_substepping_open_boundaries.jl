include(joinpath(@__DIR__, "setup.jl"))

#####
##### Open-boundary tests for acoustic substepping in CompressibleDynamics
#####
##### These tests verify that the acoustic substepper handles open
##### (normal-flow) lateral boundaries correctly: the perturbation
##### fields must not inherit prognostic open BCs (issue #716), and the
##### per-substep open-boundary relaxation must pull the outermost cells
##### toward the prescribed wall values (issue #738).
#####
##### Component-level tests live in
##### `test/acoustic_substepping_components.jl`; longer time
##### integrations live in `test/acoustic_substepping_stability.jl`.
#####

using Breeze
using Breeze: AcousticSubstepper
using Breeze.CompressibleEquations: SplitExplicitTimeDiscretization
using Breeze.AtmosphereModels: thermodynamic_density
using Breeze.Thermodynamics: ExnerReferenceState
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Statistics: mean
using Test
using Metal: Metal, MetalBackend

const arches = (Metal.functional() || get(ENV, "BREEZE_FORCE_METAL_FUNCTIONAL", "false") == "true") ? (default_arch, GPU(MetalBackend())) : (default_arch,)

as_test_float_types(arch) = arch isa GPU{MetalBackend} ? (Float32,) : test_float_types()

for arch in arches

    #####
    ##### Regression for issue #716: nonzero OBC on prognostic momentum must
    ##### not bleed onto the perturbation halo. Build a model with `Bounded`
    ##### x-topology and `NormalFlowBoundaryCondition(ρ·U)` on `ρu`, then confirm
    ##### that (1) the substepper's perturbation field uses topology defaults
    ##### on the open sides (not the inherited OBC), and (2) a forward step
    ##### does not produce a `DomainError` from runaway-acoustic amplification
    ##### at the wall.
    #####

    @testset "Nonzero momentum OBC: defaults on perturbation, stable step [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers),
                               topology=(Bounded, Periodic, Bounded))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)

        # Representative low-altitude ρ·U; exact value is irrelevant — the test
        # just needs a nonzero scalar `NormalFlowBoundaryCondition` value.
        ρU = FT(6)

        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρU),
                                         east = NormalFlowBoundaryCondition(ρU))
        boundary_conditions = (; ρu = ρu_bcs)

        model = AtmosphereModel(grid;
                                advection = WENO(),
                                dynamics,
                                boundary_conditions)

        # The perturbation field uses topology defaults — west/east sides on
        # a Bounded XFaceField default to `nothing`, so the prognostic's
        # `NormalFlowBoundaryCondition(ρU)` is not propagated.
        substepper = model.timestepper.substepper
        ρu_pert_bcs = substepper.momentum_perturbation.u.boundary_conditions
        @test ρu_pert_bcs.west === nothing
        @test ρu_pert_bcs.east === nothing

        ref = model.dynamics.reference_state
        set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

        # One forward step must not throw DomainError (the failure mode of #716)
        # nor produce NaNs.
        simulation = Simulation(model; Δt=1, stop_iteration=1, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 1
        @test !any(isnan, parent(model.momentum.ρu))
        @test !any(isnan, parent(model.momentum.ρw))
        @test !any(isnan, parent(model.dynamics.dry_density))
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### Per-substep open-boundary enforcement (issue #738). Three tests:
    ##### (1) `open_boundary_relaxation` kwarg propagates and is validated;
    ##### (2) the relaxation is a no-op when no side carries an active open BC;
    ##### (3) the outermost open-boundary cell of `ρ` tracks the prescribed
    #####     wall value across the acoustic substeps.
    #####

    @testset "open_boundary_relaxation kwarg propagation [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

        td_default = SplitExplicitTimeDiscretization()
        @test td_default.open_boundary_relaxation isa FT
        @test td_default.open_boundary_relaxation ≈ FT(0.5)
        acoustic_default = AcousticSubstepper(grid, td_default)
        @test acoustic_default.open_boundary_relaxation ≈ FT(0.5)

        td_custom = SplitExplicitTimeDiscretization(; open_boundary_relaxation = 0.25)
        @test td_custom.open_boundary_relaxation ≈ FT(0.25)
        acoustic_custom = AcousticSubstepper(grid, td_custom)
        @test acoustic_custom.open_boundary_relaxation ≈ FT(0.25)

        # α must lie in (0, 1]: 0 (would disable the relaxation), >1 (would
        # overshoot the prescribed value), and negative values are rejected.
        @test_throws ArgumentError SplitExplicitTimeDiscretization(; open_boundary_relaxation = 0)
        @test_throws ArgumentError SplitExplicitTimeDiscretization(; open_boundary_relaxation = 1.5)
        @test_throws ArgumentError SplitExplicitTimeDiscretization(; open_boundary_relaxation = -0.1)
        Oceananigans.defaults.FloatType = old_FT
    end

    @testset "Open-boundary relaxation is a no-op without active open BCs [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT

        # Doubly-periodic: no `Open` BC anywhere; `is_active_open_bc` should
        # return false on every side and the relaxation should be a no-op.
        grid_periodic = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                                        x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers),
                                        topology=(Periodic, Periodic, Bounded))
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid_periodic; advection=WENO(), dynamics)
        set!(model; θ=300, u=0, qᵗ=0, ρ=model.dynamics.reference_state.density)
        run!(Simulation(model; Δt=1, stop_iteration=1, verbose=false))
        @test model.clock.iteration == 1
        @test !any(isnan, parent(model.dynamics.dry_density))

        # Bounded but no OBC supplied: the prognostic-momentum BCs default to
        # impenetrable walls, which `is_active_open_bc` returns false for.
        grid_walls = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                                     x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers),
                                     topology=(Bounded, Bounded, Bounded))
        model_walls = AtmosphereModel(grid_walls; advection=WENO(), dynamics)
        set!(model_walls; θ=300, u=0, qᵗ=0, ρ=model_walls.dynamics.reference_state.density)
        run!(Simulation(model_walls; Δt=1, stop_iteration=1, verbose=false))
        @test model_walls.clock.iteration == 1
        @test !any(isnan, parent(model_walls.dynamics.dry_density))
        Oceananigans.defaults.FloatType = old_FT
    end

    @testset "Open-boundary relaxation pulls outermost cell toward prescribed ρ, ρθ [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT

        # Thin column so the hydrostatic ρ_ref variation is small (<1%) compared
        # to the deliberate ρ_wall jump below — keeps the test discriminating.
        # All four lateral sides are bounded + open so both the x- and y-direction
        # relaxation kernels fire and both ρ′ and (ρθ)′ are exercised.
        grid = RectilinearGrid(arch; size=(8, 8, 4), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 200),
                               topology=(Bounded, Bounded, Bounded))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        ref = ExnerReferenceState(grid; potential_temperature=FT(300))
        ρ_ref0 = @allowscalar interior(ref.density)[1, 1, 1]

        # Drive the lateral boundaries off the interior state by 5%: a
        # `ValueBoundaryCondition` sets ρ_wall = 1.05·ρ_ref on the open faces,
        # paired with `NormalFlowBoundaryCondition(ρ_wall·U)` / `NormalFlowBoundaryCondition(ρ_wall·V)`
        # for small inflows `U`, `V` on `ρu`, `ρv`. With the per-substep relaxation,
        # the outermost cell of ρ and (ρθ) is pulled toward the wall value each
        # substep; over the cumulative ~`Nτ` substeps per outer step the pull
        # saturates and the cell tracks the wall value closely.
        U       = FT(2)
        V       = FT(2)
        ρ_wall  = FT(1.05 * ρ_ref0)
        ρθ_wall = FT(ρ_wall * 300)
        ρu_val  = FT(ρ_wall * U)
        ρv_val  = FT(ρ_wall * V)
        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρu_val),
                                         east = NormalFlowBoundaryCondition(ρu_val))
        ρv_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(ρv_val),
                                         north = NormalFlowBoundaryCondition(ρv_val))
        ρ_bcs  = FieldBoundaryConditions(west  = ValueBoundaryCondition(ρ_wall),
                                         east  = ValueBoundaryCondition(ρ_wall),
                                         south = ValueBoundaryCondition(ρ_wall),
                                         north = ValueBoundaryCondition(ρ_wall))
        ρθ_bcs = FieldBoundaryConditions(west  = ValueBoundaryCondition(ρθ_wall),
                                         east  = ValueBoundaryCondition(ρθ_wall),
                                         south = ValueBoundaryCondition(ρθ_wall),
                                         north = ValueBoundaryCondition(ρθ_wall))
        boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρᵈ = ρ_bcs, ρθ = ρθ_bcs)

        model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                boundary_conditions)
        set!(model; θ=300, u=0, qᵗ=0, ρ=ρ_ref0)

        run!(Simulation(model; Δt=1, stop_iteration=3, verbose=false))
        @test model.clock.iteration == 3
        @test !any(isnan, parent(model.dynamics.dry_density))

        # After the relaxation has fired across ~`Nτ` substeps per outer step,
        # the cumulative pull `1 − (1−α)^Nτ` saturates and the outermost cell of
        # both ρ and (ρθ) should be much closer to the wall value than to the
        # deep interior. We sample the interior bulk at (Nx/2, Ny/2), away from
        # boundary influence in both horizontal directions.
        Nx = size(grid, 1)
        Ny = size(grid, 2)
        ρ_int  = interior(model.dynamics.dry_density)
        ρθ_int = interior(thermodynamic_density(model.formulation))

        ρ_west  = @allowscalar mean(ρ_int[1,    :, :])
        ρ_east  = @allowscalar mean(ρ_int[Nx,   :, :])
        ρ_south = @allowscalar mean(ρ_int[:, 1,    :])
        ρ_north = @allowscalar mean(ρ_int[:, Ny,   :])
        ρ_bulk  = @allowscalar mean(ρ_int[Nx÷2, Ny÷2, :])

        ρθ_west  = @allowscalar mean(ρθ_int[1,    :, :])
        ρθ_east  = @allowscalar mean(ρθ_int[Nx,   :, :])
        ρθ_south = @allowscalar mean(ρθ_int[:, 1,    :])
        ρθ_north = @allowscalar mean(ρθ_int[:, Ny,   :])
        ρθ_bulk  = @allowscalar mean(ρθ_int[Nx÷2, Ny÷2, :])

        # Require the outermost cell to be at least halfway from the bulk to the
        # prescribed wall value — a loose threshold that is comfortably met when
        # the relaxation is firing (cumulative pull > 0.9 in practice) and would
        # not be met if the boundary perturbation propagated only by interior
        # acoustic dynamics over 3 small steps.
        ρ_threshold  = ρ_bulk  + FT(0.5) * (ρ_wall  - ρ_bulk)
        ρθ_threshold = ρθ_bulk + FT(0.5) * (ρθ_wall - ρθ_bulk)

        @test ρ_west  ≥ ρ_threshold
        @test ρ_east  ≥ ρ_threshold
        @test ρ_south ≥ ρ_threshold
        @test ρ_north ≥ ρ_threshold

        @test ρθ_west  ≥ ρθ_threshold
        @test ρθ_east  ≥ ρθ_threshold
        @test ρθ_south ≥ ρθ_threshold
        @test ρθ_north ≥ ρθ_threshold
        Oceananigans.defaults.FloatType = old_FT
    end

    @testset "Asymmetric wall values: each side tracks its own prescribed ρ [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT

        # Distinct ρ_wall on each side catches a kernel where east/west or
        # south/north indices are transposed: the symmetric test would still
        # pass under such a swap, but here each outermost cell must be
        # closer to its own prescribed value than to the opposite side's.
        grid = RectilinearGrid(arch; size=(8, 8, 4), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 200),
                               topology=(Bounded, Bounded, Bounded))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        ref = ExnerReferenceState(grid; potential_temperature=FT(300))
        ρ_ref0 = @allowscalar interior(ref.density)[1, 1, 1]

        ρ_wall_west  = FT(1.05 * ρ_ref0)
        ρ_wall_east  = FT(0.97 * ρ_ref0)
        ρ_wall_south = FT(1.03 * ρ_ref0)
        ρ_wall_north = FT(0.96 * ρ_ref0)
        U = FT(2); V = FT(2)

        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(FT(ρ_wall_west * U)),
                                         east = NormalFlowBoundaryCondition(FT(ρ_wall_east * U)))
        ρv_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(FT(ρ_wall_south * V)),
                                         north = NormalFlowBoundaryCondition(FT(ρ_wall_north * V)))
        ρ_bcs  = FieldBoundaryConditions(west  = ValueBoundaryCondition(ρ_wall_west),
                                         east  = ValueBoundaryCondition(ρ_wall_east),
                                         south = ValueBoundaryCondition(ρ_wall_south),
                                         north = ValueBoundaryCondition(ρ_wall_north))
        ρθ_bcs = FieldBoundaryConditions(west  = ValueBoundaryCondition(FT(ρ_wall_west  * 300)),
                                         east  = ValueBoundaryCondition(FT(ρ_wall_east  * 300)),
                                         south = ValueBoundaryCondition(FT(ρ_wall_south * 300)),
                                         north = ValueBoundaryCondition(FT(ρ_wall_north * 300)))
        boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρᵈ = ρ_bcs, ρθ = ρθ_bcs)

        model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                boundary_conditions)
        set!(model; θ=300, u=0, qᵗ=0, ρ=ρ_ref0)
        run!(Simulation(model; Δt=1, stop_iteration=3, verbose=false))

        Nx = size(grid, 1); Ny = size(grid, 2)
        ρ_int = interior(model.dynamics.dry_density)
        ρ_west  = @allowscalar mean(ρ_int[1,    :, :])
        ρ_east  = @allowscalar mean(ρ_int[Nx,   :, :])
        ρ_south = @allowscalar mean(ρ_int[:, 1,    :])
        ρ_north = @allowscalar mean(ρ_int[:, Ny,   :])

        # Each side's outermost cell must be closer to its own prescribed wall
        # value than to the opposite side's. An index transposition would flip
        # one or both pairs.
        @test abs(ρ_west  - ρ_wall_west)  < abs(ρ_west  - ρ_wall_east)
        @test abs(ρ_east  - ρ_wall_east)  < abs(ρ_east  - ρ_wall_west)
        @test abs(ρ_south - ρ_wall_south) < abs(ρ_south - ρ_wall_north)
        @test abs(ρ_north - ρ_wall_north) < abs(ρ_north - ρ_wall_south)

        Oceananigans.defaults.FloatType = old_FT
    end

    @testset "Relaxation factor α controls pull strength [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT

        # Two identical models differing only in α. With cumulative pull
        # 1 − (1−α)^Nτ saturating monotonically in α, the high-α run must
        # track ρ_wall more tightly than the low-α run. Catches bugs where
        # α is ignored or hard-coded downstream.
        function build_α_model(α)
            grid = RectilinearGrid(arch; size=(8, 8, 4), halo=(5, 5, 5),
                                   x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 200),
                                   topology=(Bounded, Periodic, Bounded))
            td = SplitExplicitTimeDiscretization(; open_boundary_relaxation = FT(α))
            dynamics = CompressibleDynamics(td; reference_potential_temperature=300)
            ref = ExnerReferenceState(grid; potential_temperature=FT(300))
            ρ_ref0 = @allowscalar interior(ref.density)[1, 1, 1]
            ρ_wall = FT(1.05 * ρ_ref0)
            ρu_val = FT(ρ_wall * 2)
            ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(ρu_val),
                                             east = NormalFlowBoundaryCondition(ρu_val))
            ρ_bcs  = FieldBoundaryConditions(west = ValueBoundaryCondition(ρ_wall),
                                             east = ValueBoundaryCondition(ρ_wall))
            ρθ_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(FT(ρ_wall * 300)),
                                             east = ValueBoundaryCondition(FT(ρ_wall * 300)))
            boundary_conditions = (; ρu = ρu_bcs, ρᵈ = ρ_bcs, ρθ = ρθ_bcs)
            model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                    boundary_conditions)
            set!(model; θ=300, u=0, qᵗ=0, ρ=ρ_ref0)
            run!(Simulation(model; Δt=1, stop_iteration=3, verbose=false))
            return model, ρ_wall
        end

        model_low,  ρ_wall = build_α_model(0.05)
        model_high, _      = build_α_model(1.0)

        Nx = size(model_low.grid, 1)
        ρ_west_low  = @allowscalar mean(interior(model_low.dynamics.dry_density)[1, :, :])
        ρ_west_high = @allowscalar mean(interior(model_high.dynamics.dry_density)[1, :, :])

        @test abs(ρ_west_high - ρ_wall) < abs(ρ_west_low - ρ_wall)
        Oceananigans.defaults.FloatType = old_FT
    end

    @testset "NormalFlowBoundaryCondition(nothing) skips relaxation [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT

        # `is_active_open_bc` excludes `NormalFlowBoundaryCondition(nothing)` via its
        # `!(bc.condition isa Nothing)` clause. Verify the kernel is not
        # invoked in that case by setting ρ's `ValueBoundaryCondition` to a
        # value the relaxation would visibly track if it fired, and checking
        # the outermost cell stays near the initial state instead.
        grid = RectilinearGrid(arch; size=(8, 8, 4), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 200),
                               topology=(Bounded, Periodic, Bounded))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        ref = ExnerReferenceState(grid; potential_temperature=FT(300))
        ρ_ref0 = @allowscalar interior(ref.density)[1, 1, 1]
        ρ_wall = FT(1.05 * ρ_ref0)

        ρu_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(nothing),
                                         east = NormalFlowBoundaryCondition(nothing))
        ρ_bcs  = FieldBoundaryConditions(west = ValueBoundaryCondition(ρ_wall),
                                         east = ValueBoundaryCondition(ρ_wall))
        ρθ_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(FT(ρ_wall * 300)),
                                         east = ValueBoundaryCondition(FT(ρ_wall * 300)))
        boundary_conditions = (; ρu = ρu_bcs, ρᵈ = ρ_bcs, ρθ = ρθ_bcs)

        model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                boundary_conditions)
        set!(model; θ=300, u=0, qᵗ=0, ρ=ρ_ref0)
        run!(Simulation(model; Δt=1, stop_iteration=3, verbose=false))

        Nx = size(grid, 1)
        ρ_int = interior(model.dynamics.dry_density)
        ρ_west = @allowscalar mean(ρ_int[1,  :, :])
        ρ_east = @allowscalar mean(ρ_int[Nx, :, :])

        # Outermost cell must stay closer to the initial state than to ρ_wall;
        # if the relaxation fired despite the Nothing condition, the cells
        # would be pulled toward ρ_wall.
        @test abs(ρ_west - ρ_ref0) < abs(ρ_west - ρ_wall)
        @test abs(ρ_east - ρ_ref0) < abs(ρ_east - ρ_wall)
        Oceananigans.defaults.FloatType = old_FT
    end

end
