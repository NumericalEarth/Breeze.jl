include(joinpath(@__DIR__, "setup.jl"))

#####
##### Component-level tests for acoustic substepping in CompressibleDynamics
#####
##### These tests cover the building blocks of the AcousticRungeKutta3
##### (WS-RK3) Exner-pressure acoustic substepping formulation: substep
##### sequencing, the explicit horizontal step and vertical tridiagonal
##### coefficients, substepper and time-stepper construction, divergence
##### damping strategies, the upper sponge, the Exner reference state,
##### the slow-dynamics wrapper, and show methods.
#####
##### Open-boundary behavior lives in
##### `test/acoustic_substepping_open_boundaries.jl`; longer time
##### integrations live in `test/acoustic_substepping_stability.jl`.
#####

using Breeze
using Breeze: AcousticSubstepper
using Breeze.CompressibleEquations: ExplicitTimeStepping, SplitExplicitTimeDiscretization,
                                    compute_acoustic_substeps, minimum_horizontal_spacing,
                                    sponge_term_diag, sponge_rhs, sponge_slow_tendency,
                                    materialize_sponge,
                                    assemble_slow_vertical_momentum_tendency!,
                                    AcousticTridiagLower, AcousticTridiagDiagonal,
                                    AcousticTridiagUpper,
                                    horizontal_damping_scale, κˣ, κʸ,
                                    FixedHorizontalDampingScale, LocalHorizontalDampingScale,
                                    NoHorizontalDampingScale
using Breeze.CompressibleEquations: _build_vertical_rhs!, _explicit_horizontal_step!,
                                    implicit_damping_factors, prepare_acoustic_cache!
using Breeze.AtmosphereModels: SlowDynamics,
                               x_pressure_gradient, y_pressure_gradient, z_pressure_gradient,
                               buoyancy_forceᶜᶜᶜ, continuity_density, standard_pressure
using Breeze.Thermodynamics: ExnerReferenceState, surface_density, LiquidIceDensityState, MoistureMassFractions,
                             VaporEquivalentExponent, DryExponent, temperature, air_pressure,
                             mixture_heat_capacity, dry_air_gas_constant, condensate_latent_energy
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

@testset "Horizontal acoustic step applies the frozen and perturbation pressure gradients" begin
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
    model = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping()))
    pᴸ = model.dynamics.pressure
    Gρu = XFaceField(grid)
    Gρv = YFaceField(grid)
    Cᴸ = CenterField(grid)

    set!(Cᴸ, 1)
    set!(pᴸ, (x, y, z) -> 2x + 3y)
    set!(ρθ′, (x, y, z) -> x + y)
    fill!(Gρu, 0)
    fill!(Gρv, 0)
    fill!(ρu′, 0)
    fill!(ρv′, 0)

    launch!(CPU(), grid, :xyz, _explicit_horizontal_step!,
            ρu′, ρv′, grid, model.dynamics, FT(0.5), ρθ′, Cᴸ, Gρu, Gρv)

    # −Δτ (∂x pᴸ + ∂x(Cᴸ (ρθ)′)) with Cᴸ = 1
    @test @allowscalar(ρu′[2, 2, 2]) == -0.5 * (2 + 1)
    @test @allowscalar(ρv′[2, 2, 2]) == -0.5 * (3 + 1)
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

    θᴸ = CenterField(grid)
    Cᴸ = CenterField(grid)

    @allowscalar begin
        for k in 1:Nz
            θᴸ[2, 2, k] = FT(280 + 3k)
            Cᴸ[2, 2, k] = FT(350 + 8k)
        end
    end

    fill_halo_regions!(θᴸ, Cᴸ)

    δτᵐ⁺ = FT(0.7)
    dᵐ⁺ = FT(0.03)
    g = FT(9.81)
    Δz = Lz / Nz

    C(k) = @allowscalar Cᴸ[2, 2, k]
    θ_face(k) = ifelse(k == 1,
                       @allowscalar(θᴸ[2, 2, 1]),
                       ifelse(k == Nz + 1,
                              @allowscalar(θᴸ[2, 2, Nz]),
                              (@allowscalar(θᴸ[2, 2, k]) + @allowscalar(θᴸ[2, 2, k - 1])) / 2))

    direction = ZDirection()

    # `nothing` in the dynamics slot takes the default unit mass fraction and checks the bare
    # coefficient algebra. A uniformly moist compressible state then weights every PGF and
    # buoyancy entry of a row by that row's face fraction qᵈ = ρᵈ/ρ = 1 − qᵗ, and leaves the
    # damping entry alone — the matrix side of the weighting `_build_vertical_rhs!` applies.
    qᵗ = FT(0.02)
    moist_model = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping()))
    set!(moist_model; θ=300, ρ=1.1, qᵗ=qᵗ)
    weightings = ((nothing, one(FT)), (moist_model.dynamics, 1 - qᵗ))

    code_diag(k, dynamics) = get_coefficient(2, 2, k, grid, AcousticTridiagDiagonal(), nothing, direction,
                                             θᴸ, Cᴸ, g, δτᵐ⁺, dᵐ⁺, nothing, dynamics)
    code_upper(k, dynamics) = get_coefficient(2, 2, k, grid, AcousticTridiagUpper(), nothing, direction,
                                              θᴸ, Cᴸ, g, δτᵐ⁺, dᵐ⁺, nothing, dynamics)
    # Oceananigans' Press-indexed tridiagonal solver asks the lower
    # diagonal for row k as `a[k - 1]`.
    code_lower_for_row(k, dynamics) = get_coefficient(2, 2, k - 1, grid, AcousticTridiagLower(), nothing, direction,
                                                      θᴸ, Cᴸ, g, δτᵐ⁺, dᵐ⁺, nothing, dynamics)

    expected_lower(k, qᵈ) = qᵈ * (- δτᵐ⁺^2 * C(k - 1) * θ_face(k - 1) / Δz^2 +
                                   δτᵐ⁺^2 * g / (2Δz)) -
                            dᵐ⁺ / Δz^2
    expected_diag(k, qᵈ) = 1 + qᵈ * δτᵐ⁺^2 * θ_face(k) * (C(k) + C(k - 1)) / Δz^2 +
                               2dᵐ⁺ / Δz^2
    expected_upper(k, qᵈ) = qᵈ * (- δτᵐ⁺^2 * C(k) * θ_face(k + 1) / Δz^2 -
                                   δτᵐ⁺^2 * g / (2Δz)) -
                            dᵐ⁺ / Δz^2

    for (dynamics, qᵈ) in weightings
        @test code_diag(1, dynamics) == 1
        @test code_upper(1, dynamics) == 0

        for k in 2:Nz
            @test code_lower_for_row(k, dynamics) ≈ expected_lower(k, qᵈ)
            @test code_diag(k, dynamics) ≈ expected_diag(k, qᵈ)
        end

        for k in 2:Nz-1
            @test code_upper(k, dynamics) ≈ expected_upper(k, qᵈ)
        end
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
            # Δx = 1000 m, cᵃᶜ ≈ 347 m/s, acoustic_cfl = 0.5 (ERF/WRF target)
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

        @testset "Substepper caches the minimum horizontal spacing" begin
            # Δx = 1000 m, Δy = 500 m
            grid = RectilinearGrid(arch; size=(100, 6, 10), halo=(5, 5, 5),
                                   x=(0, 100kilometers), y=(0, 3kilometers), z=(0, 10kilometers))
            Δx_min = minimum_horizontal_spacing(grid)
            @test Δx_min isa FT
            @test Δx_min == 500

            substepper = AcousticSubstepper(grid, SplitExplicitTimeDiscretization())
            @test substepper.minimum_horizontal_spacing === Δx_min
            @test compute_acoustic_substeps(Δx_min, 12, constants, ν) ==
                  compute_acoustic_substeps(grid, 12, constants, ν)

            flat_grid = RectilinearGrid(arch; size=(100, 10), halo=(5, 5),
                                        x=(0, 100kilometers), z=(0, 10kilometers),
                                        topology=(Periodic, Flat, Bounded))
            @test minimum_horizontal_spacing(flat_grid) == 1000
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
                substepper.linearization_pressure_coefficient,
                FT(model.thermodynamic_constants.gravitational_acceleration), dˢ⁻,
                substepper.vertical_momentum_tendency_factor,
                substepper.slow_vertical_momentum_tendency,
                substepper.sponge)

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

        # The substep kernels read a precomputed γ profile, so the helpers need the materialized
        # form the `AcousticSubstepper` constructor would build.
        sponge = materialize_sponge(grid, UpperSponge(damping_rate=damping_rate, depth=depth, ramp=LinearRamp()))

        bottom_diag, lid_diag, lid_rhs = @allowscalar begin
            sponge_term_diag(1, 1, 1, grid, sponge, δτᵐ⁺),
            sponge_term_diag(1, 1, grid.Nz + 1, grid, sponge, δτᵐ⁺),
            sponge_rhs(1, 1, grid.Nz + 1, grid, sponge, δτˢ⁻, old_ρw)
        end

        @test bottom_diag == 0
        @test lid_diag ≈ δτᵐ⁺ * damping_rate
        @test lid_rhs ≈ δτˢ⁻ * damping_rate * FT(4)
        @test sponge_term_diag(1, 1, grid.Nz + 1, grid, nothing, δτᵐ⁺) == 0
        @test @allowscalar sponge_rhs(1, 1, grid.Nz + 1, grid, nothing, δτˢ⁻, old_ρw) == 0
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### `GaussianRamp` is deliberately NOT clamped at the layer base: its weak tail below the
    ##### nominal layer is what keeps a long mountain-wave integration from retaining a far-field
    ##### residual. The clamped ramps must stay clamped.
    #####

    @testset "Sponge ramp shapes [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        H = FT(20000)
        depth = FT(5000)

        for ramp in (CubicRamp(), Sin2Ramp(), LinearRamp())
            @test ramp(H, H, depth) ≈ 1                     # full strength at the lid
            @test ramp(H - depth, H, depth) == 0            # exactly zero at the base
            @test ramp(H - 2depth, H, depth) == 0            # and clamped below it
        end

        gaussian = GaussianRamp()
        @test gaussian(H, H, depth) ≈ 1
        @test gaussian(H - depth, H, depth) ≈ exp(-one(FT))  # e⁻¹ one depth down, not zero
        @test 0 < gaussian(H - 2depth, H, depth) < FT(0.02)  # weak, but a real tail
        @test gaussian(H - 3depth, H, depth) > 0
        # Monotone increasing toward the lid, like the others.
        @test gaussian(H - depth, H, depth) > gaussian(H - 2depth, H, depth)
    end

    #####
    ##### The sponge must damp the TOTAL vertical momentum ρw = ρwᴸ + (ρw)′, not only the acoustic
    ##### perturbation. `(ρw)′` measures the change over an RK stage, so a sponge that sees only the
    ##### perturbation leaves a quasi-steady gravity wave untouched — the absorbing layer becomes a
    ##### no-op and upgoing waves reflect off the lid. The stage-entry half of the damping therefore
    ##### rides in the slow tendency Gˢρw; check that it is there, and that it vanishes at rest.
    #####

    @testset "UpperSponge damps stage-entry ρw via Gˢρw [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 16), halo=(5, 5, 5),
                               x=(0, 1kilometers), y=(0, 1kilometers), z=(0, 8kilometers),
                               topology=(Periodic, Periodic, Bounded))

        damping_rate = FT(0.2)
        depth = FT(4kilometers)
        ramp = LinearRamp()
        sponge = UpperSponge(; damping_rate, depth, ramp)   # skeleton, for the discretization below
        materialized = materialize_sponge(grid, sponge)     # what the kernels actually see
        ρw_value = FT(3)

        ρwᴸ = ZFaceField(grid)
        set!(ρwᴸ, ρw_value)

        @allowscalar begin
            @test sponge_slow_tendency(1, 1, 1, grid, materialized, ρwᴸ, 1) == 0                # below the layer
            @test sponge_slow_tendency(1, 1, grid.Nz + 1, grid, materialized, ρwᴸ, 1) ≈ -damping_rate * ρw_value
            @test sponge_slow_tendency(1, 1, grid.Nz + 1, grid, nothing, ρwᴸ, 1) == 0
            # `Gˢρw` is scaled by the signed Δτ, so backward integration flips the sign here to
            # stay dissipative rather than amplifying.
            @test sponge_slow_tendency(1, 1, grid.Nz + 1, grid, materialized, ρwᴸ, -1) ≈ damping_rate * ρw_value
        end

        # The precomputed profile must reproduce the ramp exactly at every face it is read at.
        # This is a height-coordinate grid, so the reference coordinate `r` is just `z`.
        @allowscalar begin
            face_z = znodes(grid, Face())
            @test all(materialized.damping_profile[1, 1, k] ≈ damping_rate * ramp(face_z[k], grid.Lz, depth)
                      for k in 1:(size(grid, 3) + 1))
        end

        # Assemble Gˢρw with and without the sponge from an otherwise identical state: the
        # difference must be exactly the Rayleigh damping of the stage-entry momentum.
        function balanced_model(sponge)
            td = SplitExplicitTimeDiscretization(; substeps=4, sponge)
            model = AtmosphereModel(grid; dynamics=CompressibleDynamics(td; reference_potential_temperature=300))
            set!(model; θ=300, u=0, qᵗ=0, ρ=model.dynamics.reference_state.density)
            return model
        end

        function assembled_slow_tendency(model, ρw)
            set!(model.momentum.ρw, ρw)
            substepper = model.timestepper.substepper
            assemble_slow_vertical_momentum_tendency!(substepper, model, 1)
            return Array(interior(substepper.slow_vertical_momentum_tendency, 1, 1, :))
        end

        sponge_model = balanced_model(sponge)
        unsponged_model = balanced_model(nothing)

        Gˢρw_sponge = assembled_slow_tendency(sponge_model, ρw_value)
        Gˢρw_none = assembled_slow_tendency(unsponged_model, ρw_value)

        # Faces 2:Nz are the assembled interior (face 1 is pinned by the `k > 1` gate; the lid face
        # Nz+1 lies outside the `:xyz` launch and is pinned in the tridiag RHS).
        z_faces = znodes(grid, Face())
        damped_faces = 0
        for k in 2:size(grid, 3)
            γ = damping_rate * ramp(z_faces[k], grid.Lz, depth)
            @test Gˢρw_sponge[k] - Gˢρw_none[k] ≈ -γ * ρw_value atol=1e-4 * damping_rate * ρw_value
            γ > 0 && (damped_faces += 1)
        end
        @test damped_faces ≥ 4  # the layer really covers part of the column

        # At rest (ρwᴸ = 0) the sponge contributes nothing, so a balanced state stays balanced.
        @test assembled_slow_tendency(sponge_model, 0) ≈ assembled_slow_tendency(unsponged_model, 0)
    end

    #####
    ##### A `mask` extends the same implicit damping horizontally (e.g. over a nested model's lateral
    ##### relaxation zone): γ = rate × max(ramp(z), mask(x, y, z)).
    #####

    @testset "Masked UpperSponge damping profile [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 6, 8), halo=(5, 5, 5),
                               x=(0, 8kilometers), y=(0, 4kilometers), z=(0, 8kilometers),
                               topology=(Bounded, Periodic, Bounded))
        damping_rate = FT(0.2)
        depth = FT(2kilometers)
        wall_mask(x, y, z) = x < 2000 ? one(x) : zero(x)
        sponge = materialize_sponge(grid, UpperSponge(; damping_rate, depth, ramp=LinearRamp(), mask=wall_mask))
        γ = Array(interior(sponge.damping_profile))
        @test size(γ) == (8, 6, 9)
        @test all(γ[1:2, :, 1] .≈ damping_rate)             # inside the mask, all the way down
        @test all(γ[3:8, :, 1] .== 0)                       # outside the mask and below the lid layer
        @test all(γ[3:8, :, 9] .≈ damping_rate)             # the lid layer is unchanged
        @test sponge.mask === nothing                       # the closure stays on the host

        # On a terrain-following LatitudeLongitudeGrid the mask sees (λ, φ, z) in degrees.
        z_faces = Breeze.TerrainFollowingDiscretization.TerrainFollowingVerticalDiscretization(
            collect(range(0, 8kilometers, length=9)); formulation=Breeze.TerrainFollowingDiscretization.LinearDecay())
        ll = LatitudeLongitudeGrid(arch; size=(16, 16, 8), halo=(5, 5, 5), longitude=(-1, 1), latitude=(-1, 1),
                                   z=z_faces, topology=(Bounded, Bounded, Bounded))
        Breeze.TerrainFollowingDiscretization.materialize_terrain!(ll, (λ, φ) -> 200 * exp(-(λ^2 + φ^2) / 0.1))
        west_wall(λ, φ, z) = λ < -0.75 ? one(λ) : zero(λ)
        ll_sponge = materialize_sponge(ll, UpperSponge(; damping_rate, depth, ramp=LinearRamp(), mask=west_wall))
        γˡˡ = Array(interior(ll_sponge.damping_profile))
        @test all(γˡˡ[1:2, :, 1] .≈ damping_rate)
        @test all(γˡˡ[3:16, :, 1] .== 0)
        @test all(γˡˡ[:, :, 9] .≈ damping_rate)
    end

    #####
    ##### Implicit damping is unconditionally stable: a damping rate far above 2.5/Δt (the explicit
    ##### RK3 limit) must still monotonically drain ρw instead of blowing up.
    #####

    @testset "UpperSponge is stable for rate × Δt ≫ 1 [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 16), halo=(5, 5, 5),
                               x=(0, 80kilometers), y=(0, 80kilometers), z=(0, 8kilometers),
                               topology=(Periodic, Periodic, Bounded))

        function maximum_w_after(sponge; Δt = 20, steps = 5)
            td = SplitExplicitTimeDiscretization(; sponge)
            model = AtmosphereModel(grid; dynamics=CompressibleDynamics(td; reference_potential_temperature=300))
            set!(model; θ=300, qᵗ=0, ρ=model.dynamics.reference_state.density,
                 w=(x, y, z) -> sin(π * z / 8000) * sin(2π * x / 80000))
            for _ in 1:steps
                time_step!(model, Δt)
            end
            return maximum(abs, interior(model.velocities.w))
        end

        everywhere(x, y, z) = one(x)
        # rate × Δt = 20, eight times the explicit RK3 limit of ≈2.5
        w_sponge = maximum_w_after(UpperSponge(; damping_rate=1, depth=1kilometers, mask=everywhere))
        w_free = maximum_w_after(nothing)
        @test isfinite(w_sponge)
        @test w_sponge < FT(0.05)       # from max|w| ≈ 0.92
        @test w_sponge < w_free / 5
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
            ref = ExnerReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
            @test ref isa ExnerReferenceState
            @test eltype(ref) == FT
            @test ref.base_pressure == FT(101325)
            @test ref.surface_potential_temperature == FT(300)

            # Pressure should decrease monotonically
            for k in 2:grid.Nz
                pᵏ = @allowscalar ref.pressure[1, 1, k]
                pᵏ⁻¹ = @allowscalar ref.pressure[1, 1, k-1]
                @test pᵏ < pᵏ⁻¹
            end
        end

        @testset "show/summary" begin
            ref = ExnerReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
            s = sprint(show, ref)
            @test occursin("ExnerReferenceState", s)
            @test occursin("p₀", s)
        end

        @testset "surface_density" begin
            ref = ExnerReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
            ρ₀ = surface_density(ref)
            @test ρ₀ > 0
            @test ρ₀ isa FT
        end

        @testset "Function-valued θ₀" begin
            g = constants.gravitational_acceleration
            θ_func(z) = FT(300) * exp(FT(1e-4) * z / g)
            ref = ExnerReferenceState(grid, constants; base_pressure=100000, potential_temperature=θ_func)
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
    ##### SlowDynamics
    #####

    @testset "SlowDynamics [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(arch; size=(8, 8, 8), halo=(5, 5, 5),
                               x=(0, 100), y=(0, 100), z=(0, 1000))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid; advection=WENO(), dynamics)
        ref = model.dynamics.reference_state
        set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

        @testset "SlowDynamics" begin
            slow = SlowDynamics(model.dynamics)
            @test x_pressure_gradient(1, 1, 1, grid, slow) == 0
            @test y_pressure_gradient(1, 1, 1, grid, slow) == 0
            @test z_pressure_gradient(1, 1, 1, grid, slow) == 0
            @test buoyancy_forceᶜᶜᶜ(1, 1, 1, grid, slow) == 0
            @test continuity_density(slow) === model.dynamics.dry_density
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

#####
##### The linearized pressure coefficient is the slope of the equation of state
#####

# Cᴸ must be the slope ∂p/∂(ρθ), at fixed dry density and composition, of the equation of state the
# core closes with: finite differences of p = ρ Rᵐ T through the core's own inversion, in dry, moist
# and cloudy cells, for every temperature solver. Compressing every partial density and ρθ together
# changes p by Cᴸ (ρθ)′ alone without condensate; in cloud the loop omits ∂p/∂ρᵈ.
@testset "Linearized pressure coefficient is the slope of the equation of state [$FT, $(nameof(typeof(κ̂))), $(summary(temperature_solver))]" for FT in test_float_types(),
        κ̂ in (VaporEquivalentExponent(), DryExponent()),
        temperature_solver in temperature_solver_variants(FT)
    old_FT = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size = (6, 6, 6), halo = (5, 5, 5),
                           x = (0, 2e3), y = (0, 2e3), z = (0, 3e3),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(FT)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    base_pressure = 1e5, standard_pressure = 1e5,
                                    reference_potential_temperature = z -> 300)
    formulation = LiquidIcePotentialTemperatureFormulation(; exner_exponent = κ̂, temperature_solver)
    model = AtmosphereModel(grid; dynamics, formulation, thermodynamic_constants = constants,
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()))

    # Dry below 1 km, unsaturated vapor to 2 km, and enough water to condense above.
    qᵗ(x, y, z) = ifelse(z < 1e3, 0.0, ifelse(z < 2e3, 0.005, 0.03))
    set!(model; ρ = model.dynamics.reference_state.density, θ = 300, qᵗ)
    prepare_acoustic_cache!(model.timestepper.substepper, model)

    Cᴸ = Array(interior(model.timestepper.substepper.linearization_pressure_coefficient))
    ρᵈ = Array(interior(model.dynamics.dry_density))
    ρ  = Array(interior(model.dynamics.total_density))
    ρθ = Array(interior(model.formulation.potential_temperature_density))
    p  = Array(interior(model.dynamics.pressure))
    qᵛ = Array(interior(model.microphysical_fields.qᵛ))
    qˡ = Array(interior(model.microphysical_fields.qˡ))
    @test maximum(qˡ) > 1e-4  # the top layer condensed
    @test minimum(qᵛ) == 0    # the bottom layer is dry

    pˢᵗ = FT(standard_pressure(model.dynamics))
    Rᵈ  = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κᵈ  = Rᵈ / cᵖᵈ
    ε    = FT == Float64 ? FT(1e-4) : FT(1e-2)
    rtol = FT == Float64 ? FT(1e-5) : FT(3e-3)
    solver = model.formulation.temperature_solver

    for I in CartesianIndices(ρθ)
        q = MoistureMassFractions(qᵛ[I], qˡ[I], zero(FT))
        θ = ρθ[I] / ρᵈ[I]
        state(θ′, s = 1) = LiquidIceDensityState(θ′, q, pˢᵗ, s * ρ[I], solver, κ̂)
        slope = (air_pressure(state(θ * (1 + ε)), constants) -
                 air_pressure(state(θ * (1 - ε)), constants)) / (2ε * θ * ρᵈ[I])
        @test Cᴸ[I] ≈ slope rtol = rtol

        # Compress every partial density and ρθ by the same factor, at fixed θ and composition.
        compression_slope = (air_pressure(state(θ, 1 + ε), constants) -
                             air_pressure(state(θ, 1 - ε), constants)) / (2ε * ρᵈ[I])
        if qˡ[I] == 0
            @test Cᴸ[I] * θ ≈ compression_slope rtol = rtol
        elseif temperature_solver === NewtonSolver(FT)
            # In cloud the loop omits ∂p/∂ρᵈ = Cᴸ θ Hᶜ / (cᵖᵐ T − Hᶜ) at the converged root.
            T  = temperature(state(θ), constants)
            Hᶜ = condensate_latent_energy(q, constants)
            omitted = Hᶜ / (mixture_heat_capacity(q, constants) * T - Hᶜ)
            @test omitted > FT(0.01)
            @test Cᴸ[I] * θ * (1 + omitted) ≈ compression_slope rtol = rtol
        end

        if qᵛ[I] == 0
            @test Cᴸ[I] ≈ cᵖᵈ * Rᵈ / (cᵖᵈ - Rᵈ) * (p[I] / pˢᵗ)^κᵈ rtol = 100eps(FT)
        end
    end
    Oceananigans.defaults.FloatType = old_FT
end
