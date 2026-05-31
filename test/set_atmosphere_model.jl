using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

#####
##### Setting potential temperature
#####

@testset "Setting potential temperature (no microphysics) [$(FT)]" for FT in test_float_types(), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation)

    # Initialize with potential temperature and dry air
    θᵢ = CenterField(grid)
    set!(θᵢ, (x, y, z) -> θ₀ + rand())
    set!(model; θ=θᵢ)

    θ_model = liquid_ice_potential_temperature(model) |> Field
    @test θ_model ≈ θᵢ
end

@testset "Setting potential temperature from physical z over terrain" begin
    FT = Float64
    Oceananigans.defaults.FloatType = FT

    Nx, Nz = 16, 8
    Lx, Lz = FT(100000), FT(10000)
    r_faces = collect(range(0, Lz, length=Nz+1))
    z_faces = Breeze.TerrainFollowingDiscretization.TerrainFollowingVerticalDiscretization(
        r_faces; formulation = Breeze.TerrainFollowingDiscretization.SLEVE(
            large_scale_height = Lz / 2,
            small_scale_height = Lz / 4))

    grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(3, 3),
                           x=(-Lx/2, Lx/2), z=z_faces,
                           topology=(Periodic, Flat, Bounded))

    h₀ = FT(1000)
    a = FT(10000)
    hill(x, y) = h₀ * exp(-x^2 / a^2)
    Breeze.TerrainFollowingDiscretization.materialize_terrain!(grid, hill)
    metrics = Breeze.TerrainFollowingDiscretization.build_terrain_metrics(
        grid, Breeze.TerrainFollowingDiscretization.SlopeInsideInterpolation())

    constants = ThermodynamicConstants(FT)
    θ₀ = FT(300)
    N² = FT(1e-4)
    θ_of_z(z) = θ₀ * exp(N² * z / constants.gravitational_acceleration)

    θ_field = CenterField(grid)
    set!(θ_field, (x, z) -> z)

    max_znode_error = zero(FT)
    max_rnode_difference = zero(FT)
    @allowscalar for i in 1:Nx, k in 1:Nz
        z_phys = znode(i, 1, k, grid, Center(), Center(), Center())
        ζ = Oceananigans.Grids.rnode(i, 1, k, grid, Center(), Center(), Center())
        θ_value = θ_field[i, 1, k]
        max_znode_error = max(max_znode_error, abs(θ_value - z_phys))
        max_rnode_difference = max(max_rnode_difference, abs(θ_value - ζ))
    end

    @test max_znode_error < FT(1e-10)
    @test max_rnode_difference > FT(1)

    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    terrain_metrics = metrics,
                                    reference_potential_temperature = θ_of_z,
                                    surface_pressure = FT(101325),
                                    standard_pressure = FT(1e5))
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants)

    set!(model,
         ρ = model.dynamics.terrain_reference_density,
         θ = (x, z) -> θ_of_z(z),
         enforce_mass_conservation = false)

    p = model.dynamics.pressure
    p_ref = model.dynamics.terrain_reference_pressure
    @test @allowscalar maximum(abs, interior(p) .- interior(p_ref)) < FT(1e-6)
end

#####
##### Setting temperature directly
#####

@testset "Setting temperature directly [$(FT), $(formulation)]" for FT in test_float_types(), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 10), x=(0, 1_000), y=(0, 1_000), z=(0, 5_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101500)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)

    # Test with no microphysics first (no saturation adjustment effects)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation)

    # Set a standard lapse rate temperature profile with dry air
    T_profile(x, y, z) = FT(300) - FT(0.0065) * z

    set!(model, T=T_profile, qᵗ=FT(0))  # dry air

    # Check that temperature was set correctly (should match for dry air)
    z_nodes = Oceananigans.Grids.znodes(grid, Center())
    for k in 1:10
        T_expected = T_profile(0, 0, z_nodes[k])
        T_actual = @allowscalar model.temperature[1, 1, k]
        @test T_actual ≈ T_expected rtol=FT(1e-4)
    end

    # Check that potential temperature increases with height (stable atmosphere)
    θ = liquid_ice_potential_temperature(model) |> Field
    θ_prev = @allowscalar θ[1, 1, 1]
    for k in 2:10
        θ_k = @allowscalar θ[1, 1, k]
        @test θ_k > θ_prev  # potential temperature should increase with height
        θ_prev = θ_k
    end

    # Test round-trip consistency: set T, get θ; set θ back, get same T
    set!(model, T=FT(280), qᵗ=FT(0))
    T_after_set = @allowscalar model.temperature[2, 2, 5]
    @test T_after_set ≈ FT(280) rtol=FT(1e-4)

    # Now test with saturation adjustment
    microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
    model_moist = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation, microphysics)

    # Set T with subsaturated moisture (no condensate expected)
    set!(model_moist, T=T_profile, qᵗ=FT(0.001))  # low moisture

    # Temperature should still be close to input for subsaturated air
    T_actual = @allowscalar model_moist.temperature[1, 1, 1]
    T_expected = T_profile(0, 0, z_nodes[1])
    @test T_actual ≈ T_expected rtol=FT(0.02)  # allow 2% tolerance due to moisture effects
end

#####
##### Setting relative humidity
#####

@testset "Setting relative humidity [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(default_arch; size=(1, 1, 8), x=(0, 1e3), y=(0, 1e3), z=(0, 1e3))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())

    @testset "Scalar ℋ (subsaturated)" begin
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        set!(model, θ=FT(300), ℋ=FT(0.5))

        # Verify the moisture was set correctly using the RelativeHumidity diagnostic
        ℋ_field = RelativeHumidityField(model)
        # Convert to host Array because in Julia v1.11 the broadcast with the
        # keyword argument doesn't work.
        @test @allowscalar all(isapprox(FT(0.5); rtol=5e-2), Array(interior(ℋ_field)))
        @test @allowscalar all(x -> x > 0, interior(specific_humidity(model)))
    end

    # Function inputs don't work on GPU (non-bitstype argument error)
    if default_arch isa CPU
        @testset "Function ℋ (spatially-varying)" begin
            model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                    dynamics=AnelasticDynamics(reference_state),
                                    microphysics)

            ℋ_func(x, y, z) = FT(0.8) * exp(-z / FT(500))
            set!(model, θ=FT(300), ℋ=ℋ_func)

            ℋ_field = RelativeHumidityField(model)
            @allowscalar for k in 1:8
                z = znodes(grid, Center())[k]
                @test isapprox(interior(ℋ_field, 1, 1, k)[1], ℋ_func(0, 0, z); rtol=5e-2)
            end
        end
    end

    @testset "Supersaturated ℋ creates cloud liquid" begin
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        set!(model, θ=FT(300), ℋ=FT(1.5))

        # After saturation adjustment, there should be cloud liquid
        @test @allowscalar all(x -> x > 0, interior(model.microphysical_fields.qˡ))
        # And relative humidity should be capped at 1
        ℋ_field = RelativeHumidityField(model)
        @test @allowscalar all(x -> x ≤ 1.01, interior(ℋ_field))
    end
end
