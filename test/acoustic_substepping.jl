#####
##### Tests for acoustic substepping in CompressibleDynamics
#####
##### These tests verify that the AcousticRungeKutta3 (WS-RK3)
##### time steppers produce stable, correct results with the Exner pressure
##### acoustic substepping formulation.
#####

using Breeze
using Breeze: AcousticSubstepper
using Breeze.CompressibleEquations: ExplicitTimeStepping, SplitExplicitTimeDiscretization,
                                    compute_acoustic_substeps,
                                    sponge_term_diag, sponge_rhs,
                                    apply_horizontal_pressure_gradient_substep,
                                    AcousticTridiagLower, AcousticTridiagDiagonal,
                                    AcousticTridiagUpper
using Breeze.CompressibleEquations: _explicit_horizontal_step!
using Breeze.AtmosphereModels: SlowTendencyMode, HorizontalSlowMode,
                               x_pressure_gradient, y_pressure_gradient, z_pressure_gradient,
                               buoyancy_forceᶜᶜᶜ, dynamics_density, thermodynamic_density
using Breeze.Thermodynamics: adiabatic_hydrostatic_density, ExnerReferenceState, surface_density
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: get_coefficient
using Oceananigans.Units
using Oceananigans.Utils: launch!
using Statistics: mean
using Test
using Metal: Metal, MetalBackend

const arches = (Metal.functional() || get(ENV, "BREEZE_FORCE_METAL_FUNCTIONAL", "false") == "true") ? (default_arch, GPU(MetalBackend())) : (default_arch,)

as_test_float_types(arch) = arch isa GPU{MetalBackend} ? (Float32,) : test_float_types()

@testset "MPAS first-small-step pressure-gradient sequencing" begin
    @test apply_horizontal_pressure_gradient_substep(1, 1)
    @test !apply_horizontal_pressure_gradient_substep(1, 2)
    @test apply_horizontal_pressure_gradient_substep(2, 2)
    @test !apply_horizontal_pressure_gradient_substep(1, 6)
    @test apply_horizontal_pressure_gradient_substep(6, 6)
end

@testset "First acoustic substep retains frozen horizontal pressure gradient" begin
    FT = Float64
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(CPU();
                           size = (4, 4, 4),
                           halo = (3, 3, 3),
                           x = (0, 4),
                           y = (0, 4),
                           z = (0, 4),
                           topology = (Periodic, Periodic, Bounded))

    ρu′ = XFaceField(grid)
    ρv′ = YFaceField(grid)
    ρθ′ = CenterField(grid)
    Πᴸ = CenterField(grid)
    model = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping()))
    pᴸ = model.dynamics.pressure
    Gρu = XFaceField(grid)
    Gρv = YFaceField(grid)
    γRᵐᴸ = CenterField(grid)

    set!(Πᴸ, 1)
    set!(γRᵐᴸ, 1)
    set!(pᴸ, (x, y, z) -> 2x + 3y)
    set!(ρθ′, 0)
    fill!(Gρu, 0)
    fill!(Gρv, 0)
    fill!(ρu′, 0)
    fill!(ρv′, 0)

    launch!(CPU(), grid, :xyz, _explicit_horizontal_step!,
            ρu′, ρv′, grid, model.dynamics, FT(0.5), ρθ′, Πᴸ, Gρu, Gρv, γRᵐᴸ, false)

    @test @allowscalar(ρu′[2, 2, 2]) == -1
    @test @allowscalar(ρv′[2, 2, 2]) == -1.5
end

@testset "Acoustic vertical tridiagonal coefficients" begin
    FT = Float64
    Oceananigans.defaults.FloatType = FT
    Nz = 5
    Lz = FT(1000)
    grid = RectilinearGrid(CPU();
                           size = (4, 4, Nz),
                           halo = (3, 3, 3),
                           x = (0, 1),
                           y = (0, 1),
                           z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))

    Πᴸ = CenterField(grid)
    θᴸ = CenterField(grid)
    γRᵐᴸ = CenterField(grid)

    @allowscalar begin
        for k in 1:Nz
            Πᴸ[2, 2, k] = FT(0.90 + 0.02k)
            θᴸ[2, 2, k] = FT(280 + 3k)
            γRᵐᴸ[2, 2, k] = FT(390 + 5k)
        end
    end

    fill_halo_regions!(Πᴸ, θᴸ, γRᵐᴸ)

    δτᵐ⁺ = FT(0.7)
    dᵐ⁺ = FT(0.03)
    g = FT(9.81)
    Δz = Lz / Nz

    C(k) = @allowscalar γRᵐᴸ[2, 2, k] * Πᴸ[2, 2, k]
    θ_face(k) = ifelse(k == 1,
                       @allowscalar(θᴸ[2, 2, 1]),
                       ifelse(k == Nz + 1,
                              @allowscalar(θᴸ[2, 2, Nz]),
                              (@allowscalar(θᴸ[2, 2, k]) + @allowscalar(θᴸ[2, 2, k - 1])) / 2))

    direction = ZDirection()

    code_diag(k) = get_coefficient(2, 2, k, grid, AcousticTridiagDiagonal(), nothing, direction,
                                   Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, nothing)
    code_upper(k) = get_coefficient(2, 2, k, grid, AcousticTridiagUpper(), nothing, direction,
                                    Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, nothing)
    # Oceananigans' Press-indexed tridiagonal solver asks the lower
    # diagonal for row k as `a[k - 1]`.
    code_lower_for_row(k) = get_coefficient(2, 2, k - 1, grid, AcousticTridiagLower(), nothing, direction,
                                            Πᴸ, θᴸ, γRᵐᴸ, g, δτᵐ⁺, dᵐ⁺, nothing)

    expected_lower(k) = - δτᵐ⁺^2 * C(k - 1) * θ_face(k - 1) / Δz^2 +
                         δτᵐ⁺^2 * g / (2Δz) -
                         dᵐ⁺ / Δz^2
    expected_diag(k) = 1 + δτᵐ⁺^2 * θ_face(k) * (C(k) + C(k - 1)) / Δz^2 +
                           2dᵐ⁺ / Δz^2
    expected_upper(k) = - δτᵐ⁺^2 * C(k) * θ_face(k + 1) / Δz^2 -
                         δτᵐ⁺^2 * g / (2Δz) -
                         dᵐ⁺ / Δz^2

    @test code_diag(1) == 1
    @test code_upper(1) == 0

    for k in 2:Nz
        @test code_lower_for_row(k) ≈ expected_lower(k)
        @test code_diag(k) ≈ expected_diag(k)
    end

    for k in 2:Nz-1
        @test code_upper(k) ≈ expected_upper(k)
    end
end

#####
##### Test AcousticSubstepper construction
#####

for arch in arches

    @testset "AcousticSubstepper construction [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

        @testset "Default construction (adaptive substeps)" begin
            damping = ThermalDivergenceDamping()
            sponge = UpperSponge()
            @test damping.coefficient isa FT
            @test sponge.damping_rate isa FT
            @test sponge.depth isa FT

            td = SplitExplicitTimeDiscretization()
            @test td.forward_weight isa FT
            @test td.damping.coefficient isa FT
            acoustic = AcousticSubstepper(grid, td)
            @test acoustic.substeps === nothing  # adaptive by default
            @test acoustic.forward_weight ≈ FT(0.65)  # off-centered CN, ε = 2ω - 1 = 0.3
            # Default damping is DirectDivergenceDamping (NoDivergenceDamping was found to blow up
            # the baroclinic wave — CN off-centering alone is insufficient for horizontal stability).
            @test acoustic.damping isa DirectDivergenceDamping
            @test acoustic.damping.coefficient ≈ FT(0.1)
            @test acoustic.linearization_potential_temperature isa Oceananigans.Fields.Field
        end

        @testset "Custom parameters" begin
            length_scale = Float64(250)
            sponge_rate = Float64(0.3)
            sponge_depth = Float64(1200)
            td = SplitExplicitTimeDiscretization(substeps=10,
                                                 forward_weight=0.55,
                                                 damping=ThermalDivergenceDamping(coefficient=0.2,
                                                                                   length_scale=length_scale),
                                                 sponge=UpperSponge(damping_rate=sponge_rate,
                                                                    depth=sponge_depth))
            @test td.forward_weight isa FT
            @test td.damping.coefficient isa FT
            @test td.damping.length_scale isa FT
            @test td.sponge.damping_rate isa FT
            @test td.sponge.depth isa FT
            acoustic = AcousticSubstepper(grid, td)
            @test acoustic.substeps == 10
            @test acoustic.forward_weight ≈ FT(0.55)
            @test acoustic.damping isa ThermalDivergenceDamping
            @test acoustic.damping.coefficient ≈ FT(0.2)
            @test acoustic.damping.length_scale ≈ FT(length_scale)
            @test acoustic.sponge isa UpperSponge
            @test acoustic.sponge.damping_rate ≈ FT(sponge_rate)
            @test acoustic.sponge.depth ≈ FT(sponge_depth)
        end

        @testset "Invalid damping parameters" begin
            @test_throws ArgumentError SplitExplicitTimeDiscretization(
                damping=(ThermalDivergenceDamping(), NoDivergenceDamping()))
        end
    end

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
        @test !any(isnan, parent(model.dynamics.density))
    end

    #####
    ##### Per-substep open-boundary enforcement (issue #738). Three tests:
    ##### (1) `open_boundary_relaxation` kwarg propagates and is validated;
    ##### (2) the relaxation is a no-op when no side carries an active open BC;
    ##### (3) the outermost open-boundary cell of `ρ` tracks the prescribed
    #####     wall value across the acoustic substeps.
    #####

    @testset "open_boundary_relaxation kwarg propagation [$(arch), $(FT)]" for FT in as_test_float_types(arch)
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
    end

    @testset "Open-boundary relaxation is a no-op without active open BCs [$(arch), $(FT)]" for FT in as_test_float_types(arch)
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
        @test !any(isnan, parent(model.dynamics.density))

        # Bounded but no OBC supplied: the prognostic-momentum BCs default to
        # impenetrable walls, which `is_active_open_bc` returns false for.
        grid_walls = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                                     x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers),
                                     topology=(Bounded, Bounded, Bounded))
        model_walls = AtmosphereModel(grid_walls; advection=WENO(), dynamics)
        set!(model_walls; θ=300, u=0, qᵗ=0, ρ=model_walls.dynamics.reference_state.density)
        run!(Simulation(model_walls; Δt=1, stop_iteration=1, verbose=false))
        @test model_walls.clock.iteration == 1
        @test !any(isnan, parent(model_walls.dynamics.density))
    end

    @testset "Open-boundary relaxation pulls outermost cell toward prescribed ρ, ρθ [$(arch), $(FT)]" for FT in as_test_float_types(arch)
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
        boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρ = ρ_bcs, ρθ = ρθ_bcs)

        model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                boundary_conditions)
        set!(model; θ=300, u=0, qᵗ=0, ρ=ρ_ref0)

        run!(Simulation(model; Δt=1, stop_iteration=3, verbose=false))
        @test model.clock.iteration == 3
        @test !any(isnan, parent(model.dynamics.density))

        # After the relaxation has fired across ~`Nτ` substeps per outer step,
        # the cumulative pull `1 − (1−α)^Nτ` saturates and the outermost cell of
        # both ρ and (ρθ) should be much closer to the wall value than to the
        # deep interior. We sample the interior bulk at (Nx/2, Ny/2), away from
        # boundary influence in both horizontal directions.
        Nx = size(grid, 1)
        Ny = size(grid, 2)
        ρ_int  = interior(model.dynamics.density)
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
    end

    @testset "Asymmetric wall values: each side tracks its own prescribed ρ [$(arch), $(FT)]" for FT in as_test_float_types(arch)
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
        boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs, ρ = ρ_bcs, ρθ = ρθ_bcs)

        model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                boundary_conditions)
        set!(model; θ=300, u=0, qᵗ=0, ρ=ρ_ref0)
        run!(Simulation(model; Δt=1, stop_iteration=3, verbose=false))

        Nx = size(grid, 1); Ny = size(grid, 2)
        ρ_int = interior(model.dynamics.density)
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
    end

    @testset "Relaxation factor α controls pull strength [$(arch), $(FT)]" for FT in as_test_float_types(arch)
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
            boundary_conditions = (; ρu = ρu_bcs, ρ = ρ_bcs, ρθ = ρθ_bcs)
            model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                    boundary_conditions)
            set!(model; θ=300, u=0, qᵗ=0, ρ=ρ_ref0)
            run!(Simulation(model; Δt=1, stop_iteration=3, verbose=false))
            return model, ρ_wall
        end

        model_low,  ρ_wall = build_α_model(0.05)
        model_high, _      = build_α_model(1.0)

        Nx = size(model_low.grid, 1)
        ρ_west_low  = @allowscalar mean(interior(model_low.dynamics.density)[1, :, :])
        ρ_west_high = @allowscalar mean(interior(model_high.dynamics.density)[1, :, :])

        @test abs(ρ_west_high - ρ_wall) < abs(ρ_west_low - ρ_wall)
    end

    @testset "NormalFlowBoundaryCondition(nothing) skips relaxation [$(arch), $(FT)]" for FT in as_test_float_types(arch)
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
        boundary_conditions = (; ρu = ρu_bcs, ρ = ρ_bcs, ρθ = ρθ_bcs)

        model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                boundary_conditions)
        set!(model; θ=300, u=0, qᵗ=0, ρ=ρ_ref0)
        run!(Simulation(model; Δt=1, stop_iteration=3, verbose=false))

        Nx = size(grid, 1)
        ρ_int = interior(model.dynamics.density)
        ρ_west = @allowscalar mean(ρ_int[1,  :, :])
        ρ_east = @allowscalar mean(ρ_int[Nx, :, :])

        # Outermost cell must stay closer to the initial state than to ρ_wall;
        # if the relaxation fired despite the Nothing condition, the cells
        # would be pulled toward ρ_wall.
        @test abs(ρ_west - ρ_ref0) < abs(ρ_west - ρ_wall)
        @test abs(ρ_east - ρ_ref0) < abs(ρ_east - ρ_wall)
    end

    #####
    ##### Test adaptive substep computation
    #####

    @testset "compute_acoustic_substeps [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        constants = ThermodynamicConstants()
        ν = 0.5  # default `acoustic_cfl` (ERF/WRF target)

        @testset "1 km grid, Δt=12" begin
            grid = RectilinearGrid(arch; size=(100, 6, 10), halo=(5, 5, 5),
                                   x=(0, 100kilometers), y=(0, 6kilometers), z=(0, 10kilometers))
            # Δx = 1000 m, ℂᵃᶜ ≈ 347 m/s, acoustic_cfl = 0.5 (ERF/WRF target)
            # N = ceil(12 * 347 / (0.5 * 1000)) = ceil(8.33) = 9
            N = compute_acoustic_substeps(grid, 12, constants, ν)
            @test N isa Int
            @test N ≥ 1
            @test N == ceil(Int, 12 * sqrt(1.4 * 287.0 * 300) / (ν * 1000))
        end

        @testset "Flat y-topology" begin
            grid = RectilinearGrid(arch; size=(100, 10), halo=(5, 5),
                                   x=(0, 100kilometers), z=(0, 10kilometers),
                                   topology=(Periodic, Flat, Bounded))
            # Should use only Δx, not Δy
            N = compute_acoustic_substeps(grid, 12, constants, ν)
            N_expected = ceil(Int, 12 * sqrt(1.4 * 287.0 * 300) / (ν * 1000))
            @test N == N_expected
        end

        @testset "acoustic_cfl scales N as 1/ν" begin
            grid = RectilinearGrid(arch; size=(100, 6, 10), halo=(5, 5, 5),
                                   x=(0, 100kilometers), y=(0, 6kilometers), z=(0, 10kilometers))
            N_default = compute_acoustic_substeps(grid, 12, constants, 0.5)
            N_strict  = compute_acoustic_substeps(grid, 12, constants, 0.25)
            N_loose   = compute_acoustic_substeps(grid, 12, constants, 1.0)
            # Halving ν doubles the substep count; doubling ν halves it
            # (within ceil rounding).
            @test N_strict == ceil(Int, 12 * sqrt(1.4 * 287.0 * 300) / (0.25 * 1000))
            @test N_loose  == ceil(Int, 12 * sqrt(1.4 * 287.0 * 300) / (1.0  * 1000))
            @test N_strict > N_default > N_loose
        end

        @testset "Backward Δt yields same substep count" begin
            grid = RectilinearGrid(arch; size=(100, 6, 10), halo=(5, 5, 5),
                                   x=(0, 100kilometers), y=(0, 6kilometers), z=(0, 10kilometers))
            N_fwd = compute_acoustic_substeps(grid, +12, constants, ν)
            N_bwd = compute_acoustic_substeps(grid, -12, constants, ν)
            @test N_fwd ≥ 1
            @test N_bwd == N_fwd
        end
    end

    @testset "acoustic_cfl plumbed to AcousticSubstepper [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        td_default = SplitExplicitTimeDiscretization()
        td_strict  = SplitExplicitTimeDiscretization(; acoustic_cfl = 0.25)
        @test td_default.acoustic_cfl == FT(0.5)
        @test td_strict.acoustic_cfl  == FT(0.25)

        # Rejects nonpositive values.
        @test_throws ArgumentError SplitExplicitTimeDiscretization(; acoustic_cfl = 0)
        @test_throws ArgumentError SplitExplicitTimeDiscretization(; acoustic_cfl = -0.1)

        # Round-trips through the substepper.
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 1), y=(0, 1), z=(0, 1),
                               topology=(Periodic, Periodic, Bounded))
        sub = AcousticSubstepper(grid, td_strict)
        @test sub.acoustic_cfl == FT(0.25)
    end

    #####
    ##### Test time stepper construction
    #####

    @testset "AcousticRungeKutta3 construction [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization())
        model = AtmosphereModel(grid;
                                dynamics)

        @test model.timestepper isa AcousticRungeKutta3
        @test model.timestepper.substepper isa AcousticSubstepper
        @test model.timestepper.β₁ ≈ FT(1//3)
        @test model.timestepper.β₂ ≈ FT(1//2)
        @test model.timestepper.β₃ ≈ FT(1)
    end

    #####
    ##### Test that default time stepper for split-explicit is AcousticRungeKutta3 (WS-RK3)
    #####

    @testset "Default time stepper for SplitExplicitTimeDiscretization [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization())
        model = AtmosphereModel(grid; dynamics)

        @test model.timestepper isa AcousticRungeKutta3
    end

    #####
    ##### Test that models with acoustic substepping run without NaN
    #####

    @testset "WS-RK3 model runs without NaN [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid;
                                advection=WENO(),
                                dynamics)

        ref = model.dynamics.reference_state
        set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

        simulation = Simulation(model; Δt=6, stop_iteration=5, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 5
        @test !any(isnan, parent(model.momentum.ρu))
        @test !any(isnan, parent(model.momentum.ρw))
        @test !any(isnan, parent(model.dynamics.density))
    end

    #####
    ##### Backward integration: one step forward, one step back
    #####
    ##### A coarse sanity test that `time_step!(model, -Δt)` does not blow
    ##### up and produces a state close to the initial one. Exact
    ##### reversibility is not expected: off-centered Crank–Nicolson, the
    ##### Klemp 2018 horizontal divergence damping, and WENO upwinding in
    ##### the slow tendency all introduce one-sided dissipation. We only
    ##### check that the round-trip stays bounded and finite.
    #####

    @testset "Backward integration: one step forward and back [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid; advection=WENO(), dynamics)

        ref = model.dynamics.reference_state
        # Small smooth θ anomaly so the forward step produces non-trivial
        # dynamics; the reverse step is what the new code path exercises.
        Lz = grid.Lz
        θ₀(x, y, z) = FT(300) + FT(0.1) * sin(π * z / Lz)
        set!(model; θ=θ₀, u=0, qᵗ=0, ρ=ref.density)

        ρ_init  = Array(parent(model.dynamics.density))
        ρu_init = Array(parent(model.momentum.ρu))
        ρw_init = Array(parent(model.momentum.ρw))

        Δt = FT(6)
        time_step!(model, +Δt)
        time_step!(model, -Δt)

        # Clock counts both steps but net time returns to zero.
        @test model.clock.iteration == 2
        @test model.clock.time ≈ 0 atol=sqrt(eps(FT))

        # Doesn't blow up.
        for field in (model.dynamics.density, model.momentum.ρu, model.momentum.ρw)
            @test !any(isnan, parent(field))
            @test !any(isinf, parent(field))
        end

        # Round-trip is dissipative but tight: residuals are orders of
        # magnitude smaller than the disturbance produced by the forward step.
        # Use a relative tolerance for ρ (which has a meaningful baseline)
        # and an absolute tolerance for ρu, ρw (which start from rest).
        ρ_final  = Array(parent(model.dynamics.density))
        ρu_final = Array(parent(model.momentum.ρu))
        ρw_final = Array(parent(model.momentum.ρw))
        @test isapprox(ρ_final,  ρ_init;  rtol=1e-3)
        @test isapprox(ρu_final, ρu_init; atol=1e-3)
        @test isapprox(ρw_final, ρw_init; atol=1e-3)
    end

    #####
    ##### SK94 inertia-gravity wave stability test
    #####
    ##### Run the IGW benchmark for a short time with both time steppers
    ##### at advection-limited Δt=12 to verify the acoustic substepping is stable.
    #####

    function build_igw_model(arch; Ns=8, κᵈ=0.05)
        Nx, Ny, Nz = 100, 6, 10
        Lx, Ly, Lz = 100kilometers, 6kilometers, 10kilometers

        grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                               x=(0, Lx), y=(0, Ly), z=(0, Lz))

        p₀ = 100000
        θ₀ = 300
        U  = 20
        N² = 0.01^2

        constants = ThermodynamicConstants()
        g  = constants.gravitational_acceleration

        θᵇᵍ(z) = θ₀ * exp(N² * z / g)

        Δθ = 0.01
        a  = 5000
        x₀ = Lx / 3
        θᵢ(x, y, z) = θᵇᵍ(z) + Δθ * sin(π * z / Lz) / (1 + (x - x₀)^2 / a^2)

        td = SplitExplicitTimeDiscretization(substeps=Ns,
                                             damping=ThermalDivergenceDamping(coefficient=κᵈ))
        dynamics = CompressibleDynamics(td; surface_pressure=p₀,
                                        reference_potential_temperature=θᵇᵍ)

        model = AtmosphereModel(grid; advection=WENO(), dynamics)

        ref = model.dynamics.reference_state
        set!(model; θ=θᵢ, u=U, qᵗ=0, ρ=ref.density)

        return model
    end

    @testset "IGW stability: WS-RK3 (Δt=12, Ns=8) [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT

        model = build_igw_model(arch; Ns=8, κᵈ=0.10)

        simulation = Simulation(model; Δt=12, stop_iteration=20, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 20
        @test !any(isnan, parent(model.dynamics.density))
        @test !any(isnan, parent(model.momentum.ρw))

        # max|w| should remain bounded
        w_max = @allowscalar maximum(abs, interior(model.velocities.w))
        @test w_max < 1.0

        # Density should remain physical
        ρ_min = @allowscalar minimum(interior(model.dynamics.density))
        @test ρ_min > 0
    end

    #####
    ##### Dry thermal bubble: split-explicit / explicit / anelastic consistency
    #####
    ##### This is a small wiring regression, not a benchmark. The documented
    ##### examples cover longer physical integrations. Here we only check that
    ##### the split-explicit path produces the same short-time buoyant response
    ##### scale as explicit compressible dynamics and the anelastic model.
    #####

    function build_tiny_dry_bubble_model(kind)
        grid = RectilinearGrid(arch;
                               size = (16, 16),
                               halo = (5, 5),
                               x = (-8kilometers, 8kilometers),
                               z = (0, 8kilometers),
                               topology = (Periodic, Flat, Bounded))

        constants = ThermodynamicConstants()
        g = constants.gravitational_acceleration
        Rᵈ = dry_air_gas_constant(constants)
        cᵖᵈ = constants.dry_air.heat_capacity
        κ = Rᵈ / cᵖᵈ
        surface_pressure = 100000
        standard_pressure = 100000
        θ₀ = 300
        N² = 0
        θ_background(z) = θ₀ * exp(N² * z / g)
        reference_exner(z) = (surface_pressure / standard_pressure)^κ - g * z / (cᵖᵈ * θ₀)
        reference_pressure(z) = standard_pressure * reference_exner(z)^(1 / κ)

        if kind === :anelastic
            reference_state = ReferenceState(grid, constants;
                                             surface_pressure,
                                             potential_temperature = θ_background)
            dynamics = AnelasticDynamics(reference_state)
            timestepper = :SSPRungeKutta3
        elseif kind === :explicit
            dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                            surface_pressure,
                                            standard_pressure,
                                            reference_potential_temperature = θ_background)
            timestepper = :SSPRungeKutta3
        elseif kind === :split_explicit
            time_discretization = SplitExplicitTimeDiscretization(; substeps = 6)
            dynamics = CompressibleDynamics(time_discretization;
                                            surface_pressure,
                                            standard_pressure,
                                            reference_potential_temperature = θ_background)
            timestepper = nothing  # auto-selects :AcousticRungeKutta3 for split-explicit dynamics
        else
            error("Unknown tiny bubble model kind: $kind")
        end

        model = isnothing(timestepper) ?
            AtmosphereModel(grid; advection = WENO(), dynamics) :
            AtmosphereModel(grid; advection = WENO(), dynamics, timestepper)

        Δθ = 10
        radius = 2kilometers
        xᵇ = 0
        zᵇ = 3kilometers
        θ_initial(x, z) = θ_background(z) + Δθ * max(0, 1 - sqrt((x - xᵇ)^2 + (z - zᵇ)^2) / radius)
        ρ_initial(x, z) = reference_pressure(z) / (Rᵈ * θ_initial(x, z) * reference_exner(z))

        if kind === :anelastic
            set!(model; θ = θ_initial, qᵗ = 0)
        else
            set!(model; θ = θ_initial, ρ = ρ_initial, qᵗ = 0)
        end

        return model
    end

    function tiny_bubble_diagnostics(model)
        w = Array(interior(model.velocities.w))
        positive_w = max.(0, w)
        max_w = maximum(positive_w)
        total_positive_w = sum(positive_w)

        grid = model.grid
        z_faces = [znode(1, 1, k, grid, Center(), Center(), Face()) for k in axes(w, 3)]
        zᵂ = sum(sum(view(positive_w, :, :, k)) * z_faces[k] for k in axes(w, 3)) / total_positive_w

        return (; max_w, zᵂ)
    end

    @testset "Tiny dry thermal bubble consistency [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT

        # Note: anelastic models aren't currently supported by the Metal backend because
        # they require FFTs.  This may change in the future when Metal.jl will support FFTs.

        if !(arch isa GPU{MetalBackend})
            anelastic_model = build_tiny_dry_bubble_model(:anelastic)
        end
        explicit_model = build_tiny_dry_bubble_model(:explicit)
        split_model = build_tiny_dry_bubble_model(:split_explicit)

        simulations = [
            Simulation(explicit_model; Δt = 0.25, stop_time = 0.5, verbose = false),
            Simulation(split_model; Δt = 0.5, stop_time = 0.5, verbose = false),
        ]
        if !(arch isa GPU{MetalBackend})
            push!(simulations, Simulation(anelastic_model; Δt = 0.5, stop_time = 0.5, verbose = false))
        end

        run!.(simulations)

        if !(arch isa GPU{MetalBackend})
            anelastic = tiny_bubble_diagnostics(anelastic_model)
        end
        explicit = tiny_bubble_diagnostics(explicit_model)
        split = tiny_bubble_diagnostics(split_model)

        models = AtmosphereModel[explicit_model, split_model]
        if !(arch isa GPU{MetalBackend})
            push!(models, anelastic_model)
        end

        for model in models
            @test !any(isnan, parent(model.velocities.w))
            @test !any(isinf, parent(model.velocities.w))
        end

        if !(arch isa GPU{MetalBackend})
            @test anelastic.max_w > 0
        end
        @test explicit.max_w > 0
        @test split.max_w > 0

        @test isapprox(split.max_w, explicit.max_w; rtol = 0.25)
        if !(arch isa GPU{MetalBackend})
            # Anelastic dynamics filters acoustic adjustment, so only require the
            # same short-time buoyant response scale and centroid.
            @test isapprox(split.max_w, anelastic.max_w; rtol = 1.25)

            Δz = anelastic_model.grid.Lz / anelastic_model.grid.Nz
            @test abs(split.zᵂ - explicit.zᵂ) ≤ Δz
            @test abs(split.zᵂ - anelastic.zᵂ) ≤ 2Δz
        end
    end

    #####
    ##### Test balanced state stability (no perturbation → near-zero motion)
    #####

    @testset "Balanced state stays quiet [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT

        Nx, Ny, Nz = 16, 8, 10
        grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                               x=(0, 16kilometers), y=(0, 8kilometers), z=(0, 10kilometers))

        td = SplitExplicitTimeDiscretization(substeps=8)
        dynamics = CompressibleDynamics(td; surface_pressure=100000,
                                        reference_potential_temperature=300)

        model = AtmosphereModel(grid; advection=WENO(), dynamics)

        ref = model.dynamics.reference_state
        set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

        simulation = Simulation(model; Δt=12, stop_iteration=10, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 10

        # With no perturbation and balanced reference state, w should be near zero
        w_max = @allowscalar maximum(abs, interior(model.velocities.w))
        @test w_max < sqrt(eps(FT))  # Should be at machine precision level
    end

    #####
    ##### Test acoustic divergence damping (Klemp 2018)
    #####

    @testset "Acoustic divergence damping [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

        # Exercise the divergence-damping path with the typed AcousticDampingStrategy.
        td = SplitExplicitTimeDiscretization(substeps=8,
                                             damping=ThermalDivergenceDamping(coefficient=FT(0.5)))
        dynamics = CompressibleDynamics(td; reference_potential_temperature=300)
        model = AtmosphereModel(grid; advection=WENO(), dynamics)

        ref = model.dynamics.reference_state
        set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

        simulation = Simulation(model; Δt=6, stop_iteration=3, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 3
        @test !any(isnan, parent(model.dynamics.density))
    end

    @testset "Direct DirectDivergenceDamping [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT

        # Construction + propagation through the split-explicit time discretization.
        @test DirectDivergenceDamping().coefficient isa FT
        td0 = SplitExplicitTimeDiscretization(damping=DirectDivergenceDamping(coefficient=0.2))
        @test td0.damping isa DirectDivergenceDamping
        @test td0.damping.coefficient ≈ FT(0.2)

        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

        # Direct 3-D divergence damping: forms ∇·(ρ𝐮)′ explicitly rather than via the (ρθ)′ proxy.
        td = SplitExplicitTimeDiscretization(substeps=8, damping=DirectDivergenceDamping(coefficient=FT(0.5)))
        dynamics = CompressibleDynamics(td; reference_potential_temperature=300)
        model = AtmosphereModel(grid; advection=WENO(), dynamics)

        ref = model.dynamics.reference_state
        # Seed a horizontally divergent momentum perturbation for the damping to act on.
        set!(model; θ=300, u=(x, y, z) -> FT(0.1) * sinpi(2x / 8kilometers), qᵗ=0, ρ=ref.density)

        simulation = Simulation(model; Δt=6, stop_iteration=3, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 3
        @test !any(isnan, parent(model.momentum.ρu))
        @test !any(isnan, parent(model.dynamics.density))
    end

    #####
    ##### Test acoustic upper sponge
    #####

    @testset "UpperSponge coefficients [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 100), y=(0, 100), z=(0, 8000))

        damping_rate = FT(0.2)
        depth = FT(2000)
        δτᵐ⁺ = FT(3)
        δτˢ⁻ = FT(2)
        old_ρw = ZFaceField(grid)
        set!(old_ρw, FT(4))

        sponge = UpperSponge(damping_rate=damping_rate, depth=depth, ramp=LinearRamp())

        bottom_diag = sponge_term_diag(1, 1, 1, grid, sponge, δτᵐ⁺)
        lid_diag = sponge_term_diag(1, 1, grid.Nz + 1, grid, sponge, δτᵐ⁺)
        lid_rhs = @allowscalar sponge_rhs(1, 1, grid.Nz + 1, grid, sponge, δτˢ⁻, old_ρw)

        @test bottom_diag == 0
        @test lid_diag ≈ δτᵐ⁺ * damping_rate
        @test lid_rhs ≈ δτˢ⁻ * damping_rate * FT(4)
        @test sponge_term_diag(1, 1, grid.Nz + 1, grid, nothing, δτᵐ⁺) == 0
        @test @allowscalar sponge_rhs(1, 1, grid.Nz + 1, grid, nothing, δτˢ⁻, old_ρw) == 0
    end

    #####
    ##### Test explicit time stepping default
    #####

    @testset "Default time stepper for ExplicitTimeStepping [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

        dynamics = CompressibleDynamics(ExplicitTimeStepping())
        model = AtmosphereModel(grid; dynamics)

        @test model.timestepper isa SSPRungeKutta3
    end

    #####
    ##### CompressibleDynamics show methods
    #####

    @testset "CompressibleDynamics show methods" begin
        # Pre-materialization
        dynamics = CompressibleDynamics()
        s = sprint(show, dynamics)
        @test occursin("CompressibleDynamics", s)
        @test occursin("ExplicitTimeStepping", s)
        @test occursin("not materialized", s)

        # With split-explicit
        td = SplitExplicitTimeDiscretization(substeps=8)
        dynamics2 = CompressibleDynamics(td; reference_potential_temperature=300)
        s2 = sprint(show, dynamics2)
        @test occursin("SplitExplicitTimeDiscretization", s2)
    end

    #####
    ##### ExnerReferenceState construction and show
    #####

    @testset "ExnerReferenceState [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 10000),
                               topology=(Periodic, Periodic, Bounded))
        constants = ThermodynamicConstants(FT)

        @testset "Construction and basic properties" begin
            ref = ExnerReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
            @test ref isa ExnerReferenceState
            @test eltype(ref) == FT
            @test ref.surface_pressure == FT(101325)
            @test ref.surface_potential_temperature == FT(300)

            # Pressure should decrease monotonically
            for k in 2:grid.Nz
                pᵏ = @allowscalar ref.pressure[1, 1, k]
                pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
                @test pᵏ < pᵏ⁻¹
            end
        end

        @testset "show/summary" begin
            ref = ExnerReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
            s = sprint(show, ref)
            @test occursin("ExnerReferenceState", s)
            @test occursin("p₀", s)
        end

        @testset "surface_density" begin
            ref = ExnerReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
            ρ₀ = surface_density(ref)
            @test ρ₀ > 0
            @test ρ₀ isa FT
        end

        @testset "Function-valued θ₀" begin
            g = constants.gravitational_acceleration
            θ_func(z) = FT(300) * exp(FT(1e-4) * z / g)
            ref = ExnerReferenceState(grid, constants; surface_pressure=100000, potential_temperature=θ_func)
            @test ref isa ExnerReferenceState

            # Pressure should still decrease monotonically
            for k in 2:grid.Nz
                pᵏ = @allowscalar ref.pressure[1, 1, k]
                pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
                @test pᵏ < pᵏ⁻¹
            end
        end
    end

    #####
    ##### SlowTendencyMode and HorizontalSlowMode
    #####

    @testset "SlowTendencyMode and HorizontalSlowMode [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 100), y=(0, 100), z=(0, 1000))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid; advection=WENO(), dynamics)
        ref = model.dynamics.reference_state
        set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

        @testset "SlowTendencyMode" begin
            slow = SlowTendencyMode(model.dynamics)
            @test x_pressure_gradient(1, 1, 1, grid, slow) == 0
            @test y_pressure_gradient(1, 1, 1, grid, slow) == 0
            @test z_pressure_gradient(1, 1, 1, grid, slow) == 0
            @test buoyancy_forceᶜᶜᶜ(1, 1, 1, grid, slow) == 0
            @test dynamics_density(slow) === model.dynamics.density
        end

        @testset "HorizontalSlowMode" begin
            hslow = HorizontalSlowMode(model.dynamics)
            @test z_pressure_gradient(1, 1, 1, grid, hslow) == 0
            @test buoyancy_forceᶜᶜᶜ(1, 1, 1, grid, hslow) == 0
            @test dynamics_density(hslow) === model.dynamics.density
        end
    end

    #####
    ##### CompressibleDynamics without reference state (ExplicitTimeStepping)
    #####

    @testset "CompressibleDynamics without reference state [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 4000), y=(0, 4000), z=(0, 4000))

        dynamics = CompressibleDynamics()
        model = AtmosphereModel(grid; advection=WENO(), dynamics)

        set!(model; θ=300, u=0, qᵗ=0, ρ=FT(1.2))
        simulation = Simulation(model; Δt=0.1, stop_iteration=3, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 3
        @test !any(isnan, parent(model.dynamics.density))
        @test model.dynamics.reference_state === nothing
    end

end
