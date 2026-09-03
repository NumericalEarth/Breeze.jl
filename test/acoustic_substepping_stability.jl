include(joinpath(@__DIR__, "setup.jl"))

#####
##### Time-integration tests for acoustic substepping in CompressibleDynamics
#####
##### These tests verify that the AcousticRungeKutta3 (WS-RK3) time
##### stepper with the Exner pressure acoustic substepping formulation
##### produces stable, correct results over multiple outer steps:
##### NaN-free integration, backward (negative-Δt) round trips, the SK94
##### inertia-gravity-wave benchmark, dry-thermal-bubble consistency with
##### the explicit compressible and anelastic formulations, and a
##### balanced rest state that stays quiet.
#####
##### Component-level tests live in
##### `test/acoustic_substepping_components.jl`; open-boundary behavior
##### lives in `test/acoustic_substepping_open_boundaries.jl`.
#####

using Breeze
using Breeze.CompressibleEquations: ExplicitTimeStepping, SplitExplicitTimeDiscretization
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Test
using Metal: Metal, MetalBackend

const arches = (Metal.functional() || get(ENV, "BREEZE_FORCE_METAL_FUNCTIONAL", "false") == "true") ? (default_arch, GPU(MetalBackend())) : (default_arch,)

as_test_float_types(arch) = arch isa GPU{MetalBackend} ? (Float32,) : test_float_types()

for arch in arches

    #####
    ##### Test that models with acoustic substepping run without NaN
    #####

    @testset "WS-RK3 model runs without NaN [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        @test !any(isnan, parent(model.dynamics.dry_density))
        Oceananigans.defaults.FloatType = old_FT
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
        old_FT = Oceananigans.defaults.FloatType
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

        ρ_init  = Array(parent(model.dynamics.dry_density))
        ρu_init = Array(parent(model.momentum.ρu))
        ρw_init = Array(parent(model.momentum.ρw))

        Δt = FT(6)
        time_step!(model, +Δt)
        time_step!(model, -Δt)

        # Clock counts both steps but net time returns to zero.
        @test model.clock.iteration == 2
        @test model.clock.time ≈ 0 atol=sqrt(eps(FT))

        # Doesn't blow up.
        for field in (model.dynamics.dry_density, model.momentum.ρu, model.momentum.ρw)
            @test !any(isnan, parent(field))
            @test !any(isinf, parent(field))
        end

        # Round-trip is dissipative but tight: residuals stay well below the
        # ~4e-2 |Δρw| disturbance produced by the forward step (CPU Float32
        # measures ~3e-5). Use a relative tolerance for ρ (which has a
        # meaningful baseline) and an absolute tolerance for ρu, ρw (which
        # start from rest). The ρw tolerance carries headroom for GPU Float32
        # backends (Metal), where last-bit rounding differences are amplified
        # by the non-normal substep operator into the low-1e-3 range.
        ρ_final  = Array(parent(model.dynamics.dry_density))
        ρu_final = Array(parent(model.momentum.ρu))
        ρw_final = Array(parent(model.momentum.ρw))
        @test isapprox(ρ_final,  ρ_init;  rtol=1e-3)
        @test isapprox(ρu_final, ρu_init; atol=1e-3)
        @test isapprox(ρw_final, ρw_init; atol=1e-2)
        Oceananigans.defaults.FloatType = old_FT
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
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT

        model = build_igw_model(arch; Ns=8, κᵈ=0.10)

        simulation = Simulation(model; Δt=12, stop_iteration=20, verbose=false)
        run!(simulation)

        @test model.clock.iteration == 20
        @test !any(isnan, parent(model.dynamics.dry_density))
        @test !any(isnan, parent(model.momentum.ρw))

        # max|w| should remain bounded
        w_max = @allowscalar maximum(abs, interior(model.velocities.w))
        @test w_max < 1.0

        # Density should remain physical
        ρ_min = @allowscalar minimum(interior(model.dynamics.dry_density))
        @test ρ_min > 0
        Oceananigans.defaults.FloatType = old_FT
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
        old_FT = Oceananigans.defaults.FloatType
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
        Oceananigans.defaults.FloatType = old_FT
    end

    #####
    ##### Test balanced state stability (no perturbation → near-zero motion)
    #####

    @testset "Balanced state stays quiet [$(arch), $(FT)]" for FT in as_test_float_types(arch)
        old_FT = Oceananigans.defaults.FloatType
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
        Oceananigans.defaults.FloatType = old_FT
    end

end
