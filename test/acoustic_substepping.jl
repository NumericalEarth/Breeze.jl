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
                               buoyancy_forceᶜᶜᶜ, dynamics_density
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
    pᴸ = CenterField(grid)
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
            ρu′, ρv′, grid, nothing, FT(0.5), ρθ′, Πᴸ, pᴸ, Gρu, Gρv, γRᵐᴸ, false)

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
            # Default damping is ThermalDivergenceDamping(0.1) (Klemp/Skamarock/Ha 2018
            # with the vertical part left to CN off-centering by default); required
            # for stability of the WS-RK3 + substepper coupling at production Δt.
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

        @testset "Invalid damping parameters" begin
            @test_throws ArgumentError SplitExplicitTimeDiscretization(
                damping=(ThermalDivergenceDamping(), NoDivergenceDamping()))
        end
    end

    #####
    ##### Regression for issue #716: nonzero OBC on prognostic momentum must
    ##### not bleed onto the perturbation halo. Build a model with `Bounded`
    ##### x-topology and `OpenBoundaryCondition(ρ·U)` on `ρu`, then confirm
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
        # just needs a nonzero scalar `OpenBoundaryCondition` value.
        ρU = FT(6)

        ρu_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(ρU),
                                         east = OpenBoundaryCondition(ρU))
        boundary_conditions = (; ρu = ρu_bcs)

        model = AtmosphereModel(grid;
                                advection = WENO(),
                                dynamics,
                                timestepper = :AcousticRungeKutta3,
                                boundary_conditions)

        # The perturbation field uses topology defaults — west/east sides on
        # a Bounded XFaceField default to `nothing`, so the prognostic's
        # `OpenBoundaryCondition(ρU)` is not propagated.
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
                                dynamics,
                                timestepper=:AcousticRungeKutta3)

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
                                dynamics,
                                timestepper=:AcousticRungeKutta3)

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
        model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                timestepper=:AcousticRungeKutta3)

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

    function build_igw_model(arch; timestepper=:AcousticRungeKutta3, Ns=8, κᵈ=0.05)
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

        model = AtmosphereModel(grid; advection=WENO(), dynamics, timestepper)

        ref = model.dynamics.reference_state
        set!(model; θ=θᵢ, u=U, qᵗ=0, ρ=ref.density)

        return model
    end

    @testset "IGW stability: WS-RK3 (Δt=12, Ns=8) [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        Oceananigans.defaults.FloatType = FT

        model = build_igw_model(arch; timestepper=:AcousticRungeKutta3, Ns=8, κᵈ=0.10)

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
            timestepper = :AcousticRungeKutta3
        else
            error("Unknown tiny bubble model kind: $kind")
        end

        model = AtmosphereModel(grid; advection = WENO(), dynamics, timestepper)

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
        model = AtmosphereModel(grid; advection=WENO(), dynamics,
                                timestepper=:AcousticRungeKutta3)

        ref = model.dynamics.reference_state
        set!(model; θ=300, u=0, qᵗ=0, ρ=ref.density)

        simulation = Simulation(model; Δt=6, stop_iteration=3, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 3
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
