include(joinpath(@__DIR__, "setup.jl"))

#####
##### Component-level tests for acoustic substepping in CompressibleDynamics
#####
##### These tests cover the building blocks of the AcousticRungeKutta3
##### (WS-RK3) Exner-pressure acoustic substepping formulation: substep
##### sequencing, the explicit horizontal step and vertical tridiagonal
##### coefficients, substepper and time-stepper construction, divergence
##### damping strategies, the upper sponge, the Exner reference state,
##### slow-tendency modes, and show methods.
#####
##### Open-boundary behavior lives in
##### `test/acoustic_substepping_open_boundaries.jl`; longer time
##### integrations live in `test/acoustic_substepping_stability.jl`.
#####

using Breeze
using Breeze: AcousticSubstepper
using Breeze.CompressibleEquations: ExplicitTimeStepping, SplitExplicitTimeDiscretization,
                                    compute_acoustic_substeps,
                                    sponge_term_diag, sponge_rhs,
                                    apply_horizontal_pressure_gradient_substep,
                                    AcousticTridiagLower, AcousticTridiagDiagonal,
                                    AcousticTridiagUpper,
                                    horizontal_damping_scale, κˣ, κʸ,
                                    FixedHorizontalDampingScale, LocalHorizontalDampingScale,
                                    NoHorizontalDampingScale
using Breeze.CompressibleEquations: _build_vertical_rhs!, _explicit_horizontal_step!,
                                    implicit_damping_factors
using Breeze.AtmosphereModels: SlowTendencyMode, HorizontalSlowMode,
                               x_pressure_gradient, y_pressure_gradient, z_pressure_gradient,
                               buoyancy_forceᶜᶜᶜ, dynamics_density
using Breeze.Thermodynamics: ExnerReferenceState, surface_density
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: get_coefficient
using Oceananigans.Units
using Oceananigans.Utils: KernelParameters, launch!
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
        old_FT = Oceananigans.defaults.FloatType
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
            # Default damping is ThermalDivergenceDamping (the proven config; isolating whether the
            # baroclinic-wave blow-up is the damping form or the recip/per-stage substep changes).
            @test acoustic.damping isa ThermalDivergenceDamping
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

        @testset "Horizontal damping-scale diffusivities" begin
            # The per-direction diffusivities κˣ, κʸ carry the 1/Δτ factor at call
            # time (Δτ is deliberately kept out of the scale structs so a traced
            # substep size never lands in a struct field). Exercise the accessors
            # directly — they are otherwise only hit inside the damping kernel, so a
            # wrong field name or missing 1/Δτ would slip through construction tests.
            Δτ = FT(2)
            Δ  = FT(25)  # min(Δx, Δy) on this 4×4 grid over a 100 m extent

            # Fixed length scale ⇒ γ = α ℓ² / Δτ, uniform in x and y.
            ℓ = FT(250)
            fixed_damping = ThermalDivergenceDamping(coefficient=0.2, length_scale=ℓ)
            fixed_scale = horizontal_damping_scale(fixed_damping, fixed_damping.coefficient)
            @test fixed_scale isa FixedHorizontalDampingScale
            @test fixed_scale.coefficient ≈ FT(0.2) * ℓ^2
            @allowscalar begin
                @test κˣ(2, 2, 4, grid, fixed_scale, Δτ) ≈ FT(0.2) * ℓ^2 / Δτ
                @test κʸ(2, 2, 4, grid, fixed_scale, Δτ) ≈ FT(0.2) * ℓ^2 / Δτ
            end

            # Default (no length scale) ⇒ mesh-local γ = α min(Δx, Δy)² / Δτ.
            local_damping = ThermalDivergenceDamping(coefficient=0.2)
            local_scale = horizontal_damping_scale(local_damping, local_damping.coefficient)
            @test local_scale isa LocalHorizontalDampingScale
            @allowscalar begin
                @test κˣ(2, 2, 4, grid, local_scale, Δτ) ≈ FT(0.2) * Δ^2 / Δτ
                @test κʸ(2, 2, 4, grid, local_scale, Δτ) ≈ FT(0.2) * Δ^2 / Δτ
            end

            # A Flat direction gets a no-op scale that damps nothing.
            @allowscalar begin
                @test κˣ(2, 2, 4, grid, NoHorizontalDampingScale(), Δτ) == 0
                @test κʸ(2, 2, 4, grid, NoHorizontalDampingScale(), Δτ) == 0
            end
        end

        @testset "Invalid damping parameters" begin
            @test_throws ArgumentError SplitExplicitTimeDiscretization(
                damping=(ThermalDivergenceDamping(), NoDivergenceDamping()))
        end
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### Test adaptive substep computation
    #####

    @testset "compute_acoustic_substeps [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        Oceananigans.defaults.FloatType = old_FT
    end

    @testset "acoustic_cfl plumbed to AcousticSubstepper [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### Test time stepper construction
    #####

    @testset "AcousticRungeKutta3 construction [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### Test that default time stepper for split-explicit is AcousticRungeKutta3 (WS-RK3)
    #####

    @testset "Default time stepper for SplitExplicitTimeDiscretization [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization())
        model = AtmosphereModel(grid; dynamics)

        @test model.timestepper isa AcousticRungeKutta3
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### Test acoustic divergence damping (Klemp 2018)
    #####

    @testset "Acoustic divergence damping [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        @test !any(isnan, parent(model.dynamics.dry_density))
        Oceananigans.defaults.FloatType = old_FT
    end

    # `_build_vertical_rhs!` reads (ρw)′ at faces k−1, k, k+1 for the implicit vertical-damping
    # term `∂z²(ρw)′`, so its destination must not be `momentum_perturbation.w`: a thread would
    # otherwise read a neighbour another thread has already overwritten. The driver writes into
    # the dedicated `vertical_solver_source_term` field instead.
    @testset "Vertical RHS build leaves (ρw)′ intact [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers))

        damping = ThermalDivergenceDamping(coefficient=FT(0.05), damp_vertical=true)
        td = SplitExplicitTimeDiscretization(substeps=4; damping)
        dynamics = CompressibleDynamics(td; reference_potential_temperature=300)
        model = AtmosphereModel(grid; dynamics)
        set!(model; θ=300, u=0, ρ=model.dynamics.reference_state.density)

        substepper = model.timestepper.substepper
        @test parent(substepper.vertical_solver_source_term) !== parent(substepper.momentum_perturbation.w)

        # `dˢ⁻ ≠ 0` is what makes the aliased stencil observable in the first place.
        ω = FT(substepper.forward_weight)
        dᵐ⁺, dˢ⁻ = implicit_damping_factors(substepper.damping, ω, grid, FT)
        @test dˢ⁻ != 0

        time_step!(model, 6)

        ρw′ = deepcopy(interior(substepper.momentum_perturbation.w))
        Δτ = FT(1.5)
        launch!(architecture(grid), grid,
                KernelParameters(1:size(grid, 1), 1:size(grid, 2), 1:size(grid, 3) + 1),
                _build_vertical_rhs!,
                substepper.vertical_solver_source_term,
                substepper.density_predictor,
                substepper.density_potential_temperature_predictor,
                substepper.density_perturbation,
                substepper.density_potential_temperature_perturbation,
                substepper.momentum_perturbation.w,
                grid, model.dynamics, Δτ, ω * Δτ, (1 - ω) * Δτ,
                substepper.linearization_exner, substepper.linearization_gamma_R_mixture,
                FT(model.thermodynamic_constants.gravitational_acceleration), dˢ⁻,
                substepper.vertical_momentum_tendency_factor,
                substepper.slow_vertical_momentum_tendency,
                substepper.sponge, true)

        # Building the RHS must not touch its own (ρw)′ input.
        @test interior(substepper.momentum_perturbation.w) == ρw′
        @test !any(isnan, parent(substepper.vertical_solver_source_term))
        Oceananigans.defaults.FloatType = old_FT
    end

    @testset "Direct DirectDivergenceDamping [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        @test !any(isnan, parent(model.dynamics.dry_density))
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### Test acoustic upper sponge
    #####

    @testset "UpperSponge coefficients [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### Test explicit time stepping default
    #####

    @testset "Default time stepper for ExplicitTimeStepping [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

        dynamics = CompressibleDynamics(ExplicitTimeStepping())
        model = AtmosphereModel(grid; dynamics)

        @test model.timestepper isa SSPRungeKutta3
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### CompressibleDynamics show methods
    #####

    @testset "CompressibleDynamics show methods" begin
        old_FT = Oceananigans.defaults.FloatType
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
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### ExnerReferenceState construction and show
    #####

    @testset "ExnerReferenceState [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### SlowTendencyMode and HorizontalSlowMode
    #####

    @testset "SlowTendencyMode and HorizontalSlowMode [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
            @test dynamics_density(slow) === model.dynamics.dry_density
        end

        @testset "HorizontalSlowMode" begin
            hslow = HorizontalSlowMode(model.dynamics)
            @test z_pressure_gradient(1, 1, 1, grid, hslow) == 0
            @test buoyancy_forceᶜᶜᶜ(1, 1, 1, grid, hslow) == 0
            @test dynamics_density(hslow) === model.dynamics.dry_density
        end
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### CompressibleDynamics without reference state (ExplicitTimeStepping)
    #####

    @testset "CompressibleDynamics without reference state [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 4000), y=(0, 4000), z=(0, 4000))

        # Explicitly disable the reference (the default now builds one on every grid) to exercise
        # the full-pressure PGF/buoyancy path.
        dynamics = CompressibleDynamics(; reference_state=nothing)
        model = AtmosphereModel(grid; advection=WENO(), dynamics)

        set!(model; θ=300, u=0, qᵗ=0, ρ=FT(1.2))
        simulation = Simulation(model; Δt=0.1, stop_iteration=3, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 3
        @test !any(isnan, parent(model.dynamics.dry_density))
        @test model.dynamics.reference_state === nothing
        Oceananigans.defaults.FloatType = old_FT
    end

end
