include(joinpath(@__DIR__, "setup.jl"))

using Adapt: adapt
using Breeze
using Breeze: PrescribedDensity, PrescribedDynamics, KinematicModel
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, NormalFlowBoundaryCondition
using Oceananigans.Fields: ZeroField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Test

@testset "KinematicDriver [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), extent=(1000, 1000, 2000))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants)

    @testset "PrescribedDynamics construction" begin
        dynamics = PrescribedDynamics(reference_state)
        @test dynamics.density isa PrescribedDensity
        @test dynamics_density(dynamics) === reference_state.density

        dynamics_div = PrescribedDynamics(reference_state; divergence_correction=true)
        @test dynamics_div isa PrescribedDynamics{true}
    end

    @testset "PrescribedDensity adapt and on_architecture" begin
        pd = PrescribedDensity(reference_state.density)

        # Test adapt_structure
        adapted_pd = adapt(CPU(), pd)
        @test adapted_pd isa PrescribedDensity

        # Test on_architecture
        transferred_pd = on_architecture(CPU(), pd)
        @test transferred_pd isa PrescribedDensity
    end

    @testset "KinematicModel with prognostic density" begin
        ρ = CenterField(grid)
        set!(ρ, FT(1))
        model = AtmosphereModel(grid; dynamics=PrescribedDynamics(ρ))
        @test haskey(Oceananigans.prognostic_fields(model), :ρ)
    end

    @testset "Hydrostatic pressure uses the z = 0 datum on a raised domain" begin
        z_bottom = FT(2000)
        raised_grid = RectilinearGrid(default_arch; size=(4, 4, 4),
                                      x=(0, 100), y=(0, 100), z=(z_bottom, FT(3000)))
        ρ_value = FT(1.2)
        p₀ = FT(101325)
        ρ = CenterField(raised_grid)
        set!(ρ, ρ_value)

        dynamics = PrescribedDynamics(PrescribedDensity(ρ); base_pressure=p₀)
        model = AtmosphereModel(raised_grid; dynamics, thermodynamic_constants=ThermodynamicConstants(FT))
        g = model.thermodynamic_constants.gravitational_acceleration
        Δz = FT(250)
        pˢ_expected = p₀ - ρ_value * g * z_bottom
        p¹_expected = pˢ_expected - ρ_value * g * Δz / 2

        @test @allowscalar(model.dynamics.surface_pressure[1, 1, 1]) ≈ pˢ_expected
        @test @allowscalar(interior(model.dynamics.pressure)[1, 1, 1]) ≈ p¹_expected
        @test @allowscalar(ℑzᵃᵃᶠ(1, 1, 1, raised_grid, model.dynamics.pressure)) ≈ pˢ_expected
        @test @allowscalar(Breeze.AtmosphereModels.surface_pressure(model.dynamics)[1, 1, 1]) ≈ pˢ_expected
    end

    # `materialize_pressure` has three methods — no pressure spec, a field, and a number or
    # function — and each derives its own bottom-face anchor unless the user overrides it. Only the
    # first was exercised, so the anchor the other two derive, and the override path itself, went
    # untested.
    @testset "PrescribedDynamics pressure specs and anchor overrides" begin
        z_bottom = FT(2000)
        raised_grid = RectilinearGrid(default_arch; size=(2, 2, 4),
                                      x=(0, 100), y=(0, 100), z=(z_bottom, FT(3000)))
        constants = ThermodynamicConstants(FT)
        g = constants.gravitational_acceleration
        ρ_value = FT(1.2)
        p₀ = FT(101325)
        Δz = FT(250)
        ρ = CenterField(raised_grid)
        set!(ρ, ρ_value)

        materialize(d) = Breeze.AtmosphereModels.materialize_dynamics(d, raised_grid, nothing, constants)
        anchor(d) = @allowscalar d.surface_pressure[1, 1, 1]

        # (1) No pressure spec: the datum is reduced by extending ρ¹ down to the bottom face.
        derived = materialize(PrescribedDynamics(PrescribedDensity(ρ); base_pressure=p₀))
        pˢ_linear = p₀ - ρ_value * g * z_bottom
        @test anchor(derived) ≈ pˢ_linear
        @test derived.surface_pressure isa Field{Center, Center, Nothing}

        # (2) A supplied pressure field: the anchor is diagnosed from it with the local scale
        # height, which is a different rule — it needs p and ρ at one level, not the datum. The two
        # disagree at second order, and that gap is what this pins.
        from_field = materialize(PrescribedDynamics(PrescribedDensity(ρ);
                                                    pressure=derived.pressure, base_pressure=p₀))
        p¹ = @allowscalar interior(derived.pressure)[1, 1, 1]
        @test anchor(from_field) ≈ p¹ * exp(g * Δz * ρ_value / (2 * p¹))
        @test from_field.pressure === derived.pressure          # a supplied field is used as given
        round_trip = anchor(from_field) - anchor(derived)
        @test round_trip ≈ (ρ_value * g * Δz / 2)^2 / (2 * p¹) rtol=1e-2
        @test 0 < round_trip < FT(20)                            # second order, not a discrepancy

        # (3) A number, and a function of z: filled into a field, then the same diagnosis.
        for spec in (FT(90000), (x, y, z) -> FT(90000) - ρ_value * g * (z - z_bottom))
            from_spec = materialize(PrescribedDynamics(PrescribedDensity(ρ);
                                                       pressure=spec, base_pressure=p₀))
            p¹ˢ = @allowscalar interior(from_spec.pressure)[1, 1, 1]
            @test anchor(from_spec) ≈ p¹ˢ * exp(g * Δz * ρ_value / (2 * p¹ˢ))
            # The derived anchor is carried as the field's bottom boundary value.
            @test from_spec.pressure.boundary_conditions.bottom.condition === from_spec.surface_pressure
        end

        # (4) An explicit scalar anchor wins over all of the above and is normalized to a field.
        override = FT(79000)
        scalar_anchor = materialize(PrescribedDynamics(PrescribedDensity(ρ);
                                                       surface_pressure=override, base_pressure=p₀))
        @test scalar_anchor.surface_pressure isa Field{Center, Center, Nothing}
        @test anchor(scalar_anchor) ≈ override
        @test @allowscalar(interior(scalar_anchor.pressure)[1, 1, 1]) ≈ override - ρ_value * g * Δz / 2

        # (5) An explicit field anchor is passed through untouched, so a caller can alias it.
        field_anchor_value = Field{Center, Center, Nothing}(raised_grid)
        set!(field_anchor_value, override)
        field_anchor = materialize(PrescribedDynamics(PrescribedDensity(ρ);
                                                      surface_pressure=field_anchor_value,
                                                      base_pressure=p₀))
        @test field_anchor.surface_pressure === field_anchor_value
        @test anchor(field_anchor) ≈ override

        # `base_pressure` is the scalar datum throughout; only `surface_pressure` became a field.
        @test Breeze.AtmosphereModels.base_pressure(derived) ≈ p₀
        @test Breeze.AtmosphereModels.base_pressure(field_anchor) ≈ p₀
    end

    @testset "KinematicModel with regular fields" begin
        model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state))
        @test model isa KinematicModel
        @test model.pressure_solver === nothing

        set!(model, θ=300, qᵗ=0.01, w=1)
        @test @allowscalar(model.velocities.w[1, 1, 4]) ≈ FT(1)

        time_step!(model, 1)
        @test model.clock.iteration == 1
    end

    @testset "KinematicModel with PrescribedVelocityFields" begin
        w_evolving(x, y, z, t) = (1 - exp(-t / 100)) * sin(π * z / 2000)

        model = AtmosphereModel(grid;
            dynamics = PrescribedDynamics(reference_state),
            velocities = PrescribedVelocityFields(w=w_evolving))

        @test model isa KinematicModel
        set!(model, θ=300, qᵗ=0.01)
        @test_throws ArgumentError set!(model, w=1)

        time_step!(model, 10)
        @test model.clock.time ≈ 10
    end

    @testset "Velocity boundary conditions" begin
        w_inlet(x, y, t) = FT(0.5)
        w_bcs = FieldBoundaryConditions(bottom=NormalFlowBoundaryCondition(w_inlet))
        boundary_conditions = (; w = w_bcs)

        model = AtmosphereModel(grid; dynamics=PrescribedDynamics(reference_state), boundary_conditions)
        @test model isa KinematicModel
        @test model.velocities.w.boundary_conditions.bottom isa Oceananigans.BoundaryConditions.BoundaryCondition

        # AnelasticDynamics does not allow velocity boundary conditions
        @test_throws ArgumentError AtmosphereModel(grid; boundary_conditions)
    end
end

@testset "Gaussian advection (analytical solution) [Float64]" begin
    FT = Float64
    Oceananigans.defaults.FloatType = FT

    Lz, Nz, w₀ = 4000, 64, 10  # Reduced resolution for faster test
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Lz))

    model = AtmosphereModel(grid;
        dynamics = PrescribedDynamics(ReferenceState(grid, ThermodynamicConstants())),
        tracers = :c,
        advection = WENO())

    z₀, σ = 1000, 100
    c_exact(x, y, z, t) = exp(-(z - z₀ - w₀ * t)^2 / (2 * σ^2))

    set!(model, θ=300, qᵗ=0, w=w₀, c=(x, y, z) -> c_exact(x, y, z, 0))

    stop_time = 50
    simulation = Simulation(model; Δt=1, stop_time, verbose=false)
    run!(simulation)

    c_truth = CenterField(grid)
    set!(c_truth, (x, y, z) -> c_exact(x, y, z, stop_time))

    error = @allowscalar maximum(abs, interior(model.tracers.c) .- interior(c_truth))
    @test error < FT(0.1)  # Relaxed tolerance for reduced resolution test
end
