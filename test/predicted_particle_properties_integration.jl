using Test

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics

using Oceananigans: CPU, Center, CenterField, Face, Field, GridFittedBottom,
                     ImmersedBoundaryGrid, RectilinearGrid, set!, time_step!
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Fields: interior, location
using Oceananigans.TimeSteppers: update_state!

@testset "P3 atmosphere integration" begin
    @testset "Hallett–Mossop surface temperature follows the immersed bottom" begin
        FT = Float64
        underlying_grid = RectilinearGrid(default_arch, FT;
                                          size = (2, 1, 4),
                                          x = (0, 2), y = (0, 1), z = (0, 4))
        bottom_height = reshape(FT[0, 2], 2, 1)
        grid = ImmersedBoundaryGrid(underlying_grid,
                                    GridFittedBottom(bottom_height))
        temperature = CenterField(grid)
        surface_temperature = Field{Center, Center, Nothing}(grid)
        set!(temperature, (x, y, z) -> FT(270) + z)

        Breeze.Microphysics.PredictedParticleProperties.compute_p3_surface_temperature!(
            surface_temperature, temperature, grid)

        @test vec(Array(interior(surface_temperature))) ≈ FT[270.5, 272.5]
    end

    @testset "P3 contributes only physical condensate mass to total density" begin
        p3 = PredictedParticlePropertiesMicrophysics(Float64)
        p3_three_moment = PredictedParticlePropertiesMicrophysics(Float64; three_moment_ice = true)
        condensate_names = (:ρqᶜˡ, :ρqʳ, :ρqⁱ, :ρqʷⁱ)

        @test AtmosphereModels.condensate_field_names(p3) == condensate_names
        @test AtmosphereModels.condensate_field_names(p3_three_moment) == condensate_names

        grid = RectilinearGrid(CPU(), Float64; size = (1, 1, 1), extent = (1, 1, 1))
        dry_density = CenterField(grid)
        vapor_density = CenterField(grid)
        μ = AtmosphereModels.materialize_microphysical_fields(p3_three_moment, grid, NamedTuple())

        set!(dry_density, 1.0)
        set!(vapor_density, 0.01)
        set!(μ.ρqᶜˡ, 0.001)
        set!(μ.ρqʳ, 0.002)
        set!(μ.ρqⁱ, 0.003)
        set!(μ.ρqʷⁱ, 0.004)

        expected_density = 1.02
        density = AtmosphereModels.total_density(1, 1, 1, dry_density, p3_three_moment,
                                                  vapor_density, μ)
        @test density ≈ expected_density

        # These moments and properties are not independent material masses. In
        # particular, rime mass is already included in total ice mass.
        set!(μ.ρnᶜˡ, 1e12)
        set!(μ.ρnʳ, 2e12)
        set!(μ.ρnⁱ, 3e12)
        set!(μ.ρqᶠ, 0.5)
        set!(μ.ρbᶠ, 0.6)
        set!(μ.ρz̃ⁱ, 0.7)
        set!(μ.ρsˢᵃᵗ, 0.8)
        set!(μ.ρnᵃ, 4e12)

        density_with_nonmass_moments =
            AtmosphereModels.total_density(1, 1, 1, dry_density, p3_three_moment,
                                           vapor_density, μ)
        @test density_with_nonmass_moments ≈ expected_density
    end

    @testset "P3 cache and fall speeds follow the current RK-stage state" begin
        FT = Float64
        grid = RectilinearGrid(default_arch, FT; size = (2, 2, 2),
                               extent = (100, 100, 100))
        constants = ThermodynamicConstants(FT)
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = FT(101325),
                                         potential_temperature = FT(285))
        dynamics = AnelasticDynamics(reference_state)
        p3 = PredictedParticlePropertiesMicrophysics(FT)
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                                microphysics = p3)

        set!(model; θ = FT(285), qᵛ = FT(0.01), qᶜˡ = FT(0.003),
             nᶜˡ = FT(2e8), enforce_mass_conservation = false)
        update_state!(model)

        # Fall speeds live at z-Faces: index k is the bottom face of cell k, and the top
        # face (k = Nz+1) is held at zero by the impenetrable top boundary condition so
        # nothing sediments in from above the model top.
        Nz = size(grid, 3)
        first_fall_speed = Array(interior(model.microphysical_fields.wᶜˡ, :, :, 1:Nz))
        first_rain_source = Array(interior(model.microphysical_fields.cache_ρqʳ))
        @test all(first_fall_speed .< 0)
        @test all(Array(interior(model.microphysical_fields.wᶜˡ, :, :, Nz+1:Nz+1)) .== 0)
        @test any(first_rain_source .> 0)

        set!(model; qᶜˡ = FT(0.006), enforce_mass_conservation = false)
        update_state!(model)

        second_fall_speed = Array(interior(model.microphysical_fields.wᶜˡ, :, :, 1:Nz))
        second_rain_source = Array(interior(model.microphysical_fields.cache_ρqʳ))
        @test second_fall_speed != first_fall_speed
        @test second_rain_source != first_rain_source
    end

    @testset "P3 sedimentation velocities live at z-Faces" begin
        FT = Float64
        grid = RectilinearGrid(default_arch, FT; size = (2, 2, 4),
                               extent = (100, 100, 200))
        constants = ThermodynamicConstants(FT)
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = FT(101325),
                                         potential_temperature = FT(285))
        dynamics = AnelasticDynamics(reference_state)
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                                microphysics = PredictedParticlePropertiesMicrophysics(FT))
        μ = model.microphysical_fields

        # `div_ρUc` reads these as advecting velocities via `Az_qᶜᶜᶠ`, so they must be
        # located at (Center, Center, Face) like the resolved `w`.
        velocity_names = (:wᶜˡ, :wᶜˡₙ, :wʳ, :wʳₙ, :wⁱ, :wⁱₙ, :wⁱ_z, :wⁱ_z̃)
        for name in velocity_names
            @test location(μ[name]) === (Center, Center, Face)
            @test location(μ[name]) === location(model.velocities.w)
        end
    end

    @testset "P3 surface precipitation boundary condition" begin
        FT = Float64
        Nz = 4

        function rain_model(precipitation_boundary_condition)
            grid = RectilinearGrid(default_arch, FT; size = (2, 2, Nz),
                                   extent = (100, 100, 200))
            constants = ThermodynamicConstants(FT)
            reference_state = ReferenceState(grid, constants;
                                             surface_pressure = FT(101325),
                                             potential_temperature = FT(285))
            dynamics = AnelasticDynamics(reference_state)
            p3 = PredictedParticlePropertiesMicrophysics(FT; precipitation_boundary_condition)
            model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                                    microphysics = p3)
            set!(model; θ = FT(285), qᵛ = FT(0.01), qʳ = FT(1e-3), nʳ = FT(1e6),
                 enforce_mass_conservation = false)
            update_state!(model)
            return model
        end

        total_water(μ, model) = sum(Array(interior(model.moisture_density))) +
                               sum(Array(interior(μ.ρqᶜˡ))) +
                               sum(Array(interior(μ.ρqʳ))) +
                               sum(Array(interior(μ.ρqⁱ))) +
                               sum(Array(interior(μ.ρqʷⁱ)))

        open_model = rain_model(nothing)
        closed_model = rain_model(ImpenetrableBoundaryCondition())

        open_bottom = Array(interior(open_model.microphysical_fields.wʳ, :, :, 1:1))
        closed_bottom = Array(interior(closed_model.microphysical_fields.wʳ, :, :, 1:1))

        # Open surface: rain falls out through the bottom face. Impenetrable: it cannot.
        @test all(open_bottom .< 0)
        @test all(closed_bottom .== 0)

        # Neither boundary condition may admit precipitation through the model top.
        for model in (open_model, closed_model)
            @test all(Array(interior(model.microphysical_fields.wʳ, :, :, Nz+1:Nz+1)) .== 0)
        end

        # With periodic sides, a zero top face and an impenetrable surface, the domain is
        # closed: total water must be conserved to round-off. The open surface must lose it.
        Δt = FT(1)
        closed_before = total_water(closed_model.microphysical_fields, closed_model)
        open_before = total_water(open_model.microphysical_fields, open_model)
        time_step!(closed_model, Δt)
        time_step!(open_model, Δt)
        closed_after = total_water(closed_model.microphysical_fields, closed_model)
        open_after = total_water(open_model.microphysical_fields, open_model)

        @test closed_after ≈ closed_before rtol = 1e-12
        @test open_after < open_before
    end

    @testset "P3 square-root moment uses the mean Z/N fall speed" begin
        FT = Float64
        grid = RectilinearGrid(default_arch, FT; size = (2, 2, 3),
                               extent = (100, 100, 150))
        constants = ThermodynamicConstants(FT)
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = FT(101325),
                                         potential_temperature = FT(268))
        dynamics = AnelasticDynamics(reference_state)
        p3 = PredictedParticlePropertiesMicrophysics(FT; three_moment_ice = true)
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                                microphysics = p3)

        set!(model; θ = FT(268), qᵛ = FT(0.003), qⁱ = FT(1e-4),
             nⁱ = FT(1e5), qᶠ = FT(2e-5), enforce_mass_conservation = false)
        set!(model.microphysical_fields.ρz̃ⁱ, (x, y, z) -> FT(1e-8) * (1 + z / 150))
        update_state!(model)

        μ = model.microphysical_fields
        number_velocity = Array(interior(μ.wⁱₙ))
        reflectivity_velocity = Array(interior(μ.wⁱ_z))
        square_root_velocity = Array(interior(μ.wⁱ_z̃))
        square_root_tendency = Array(interior(model.timestepper.Gⁿ.ρz̃ⁱ))

        @test square_root_velocity ≈ (number_velocity + reflectivity_velocity) / 2
        @test all(isfinite, square_root_tendency)
    end

    @testset "P3 initialization is consistent for total and dry density inputs" begin
        FT = Float64
        grid = RectilinearGrid(default_arch, FT; size = (5, 5, 5), halo = (5, 5, 5),
                               extent = (100, 100, 100))
        constants = ThermodynamicConstants(FT)

        function make_model()
            dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                            surface_pressure = FT(1e5),
                                            standard_pressure = FT(1e5),
                                            reference_potential_temperature = z -> FT(280))
            return AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                                   microphysics = PredictedParticlePropertiesMicrophysics(FT),
                                   timestepper = :AcousticRungeKutta3)
        end

        dry_density = FT(0.9)
        qᵛ = FT(0.05)
        qᶜˡ = FT(0.01)
        qʳ = FT(0.005)
        qⁱ = FT(0.002)
        qʷⁱ = FT(0.001)
        qᶠ = FT(0.0005)
        nᶜˡ = FT(2e8)
        total_water = qᵛ + qᶜˡ + qʳ + qⁱ + qʷⁱ
        total_density = dry_density / (1 - total_water)

        model_with_total = make_model()
        set!(model_with_total; ρ = total_density, T = FT(280), qᵛ, qᶜˡ, qʳ, qⁱ,
             qʷⁱ, qᶠ, nᶜˡ, enforce_mass_conservation = false)

        model_with_dry = make_model()
        set!(model_with_dry; ρᵈ = dry_density, T = FT(280), qᵛ, qᶜˡ, qʳ, qⁱ,
             qʷⁱ, qᶠ, nᶜˡ, enforce_mass_conservation = false)

        cpu(field) = Array(interior(field))
        @test all(≈(total_density), cpu(model_with_dry.dynamics.total_density))
        @test all(≈(dry_density), cpu(model_with_total.dynamics.dry_density))
        @test cpu(model_with_dry.dynamics.total_density) ≈
              cpu(model_with_total.dynamics.total_density)
        @test cpu(model_with_dry.dynamics.dry_density) ≈
              cpu(model_with_total.dynamics.dry_density)
        @test cpu(model_with_dry.temperature) ≈ cpu(model_with_total.temperature)

        for (name, specific_value) in ((:ρqᶜˡ, qᶜˡ), (:ρqʳ, qʳ), (:ρqⁱ, qⁱ),
                                       (:ρqʷⁱ, qʷⁱ), (:ρqᶠ, qᶠ), (:ρnᶜˡ, nᶜˡ))
            dry_field = model_with_dry.microphysical_fields[name]
            total_field = model_with_total.microphysical_fields[name]
            @test all(≈(total_density * specific_value), cpu(dry_field))
            @test cpu(dry_field) ≈ cpu(total_field)
        end

        @test all(≈(total_density * qᵛ), cpu(model_with_dry.moisture_density))
        @test cpu(model_with_dry.moisture_density) ≈ cpu(model_with_total.moisture_density)

        model_with_total_water = make_model()
        set!(model_with_total_water; ρᵈ = dry_density, T = FT(280), qᵗ = total_water,
             qᶜˡ, qʳ, qⁱ, qʷⁱ, qᶠ, nᶜˡ, enforce_mass_conservation = false)
        @test all(≈(total_density), cpu(model_with_total_water.dynamics.total_density))
        @test all(≈(qᵛ), cpu(model_with_total_water.microphysical_fields.qᵛ))
        @test all(≈(total_density * qᵛ), cpu(model_with_total_water.moisture_density))
        @test cpu(model_with_total_water.formulation.potential_temperature_density) ≈
              cpu(model_with_dry.formulation.potential_temperature_density)
        @test cpu(model_with_total_water.temperature) ≈ cpu(model_with_dry.temperature)

        model_with_total_and_total_water = make_model()
        set!(model_with_total_and_total_water; ρ = total_density, T = FT(280),
             qᵗ = total_water, qᶜˡ, qʳ, qⁱ, qʷⁱ, qᶠ, nᶜˡ,
             enforce_mass_conservation = false)
        @test all(≈(dry_density),
                  cpu(model_with_total_and_total_water.dynamics.dry_density))
        @test all(≈(qᵛ), cpu(model_with_total_and_total_water.microphysical_fields.qᵛ))

        model_without_repeated_density = make_model()
        set!(model_without_repeated_density; ρᵈ = dry_density, T = FT(280),
             qᵛ = FT(0), enforce_mass_conservation = false)
        set!(model_without_repeated_density; T = FT(280), qᵛ, qᶜˡ, qʳ, qⁱ,
             qʷⁱ, qᶠ, nᶜˡ, enforce_mass_conservation = false)
        @test cpu(model_without_repeated_density.dynamics.total_density) ≈
              cpu(model_with_dry.dynamics.total_density)
        @test cpu(model_without_repeated_density.moisture_density) ≈
              cpu(model_with_dry.moisture_density)

        conflicting_density_model = make_model()
        @test_throws ArgumentError set!(conflicting_density_model;
                                        ρ = total_density, ρᵈ = dry_density,
                                        enforce_mass_conservation = false)

        relative_humidity = FT(0.5)
        relative_humidity_cloud = FT(0.01)
        relative_humidity_number = FT(2e8)

        model_with_dry_and_relative_humidity = make_model()
        set!(model_with_dry_and_relative_humidity;
             ρᵈ = dry_density, T = FT(280), ℋ = relative_humidity,
             qᶜˡ = relative_humidity_cloud, nᶜˡ = relative_humidity_number,
             enforce_mass_conservation = false)

        dry_relative_humidity_total =
            cpu(model_with_dry_and_relative_humidity.dynamics.total_density)
        dry_relative_humidity_fields = model_with_dry_and_relative_humidity.microphysical_fields
        @test all(≈(dry_density),
                  cpu(model_with_dry_and_relative_humidity.dynamics.dry_density))
        @test cpu(dry_relative_humidity_fields.ρqᶜˡ) ./ dry_relative_humidity_total ≈
              fill(relative_humidity_cloud, size(dry_relative_humidity_total))
        @test cpu(dry_relative_humidity_fields.ρnᶜˡ) ./ dry_relative_humidity_total ≈
              fill(relative_humidity_number, size(dry_relative_humidity_total))
        @test dry_relative_humidity_total ≈
              cpu(model_with_dry_and_relative_humidity.dynamics.dry_density) .+
              cpu(model_with_dry_and_relative_humidity.moisture_density) .+
              cpu(dry_relative_humidity_fields.ρqᶜˡ)
        @test all(≈(FT(280)), cpu(model_with_dry_and_relative_humidity.temperature))
        @test cpu(RelativeHumidityField(model_with_dry_and_relative_humidity)) ≈
              fill(relative_humidity, size(dry_relative_humidity_total)) rtol = 1e-10

        fixed_total_density = FT(1)
        model_with_total_and_relative_humidity = make_model()
        set!(model_with_total_and_relative_humidity;
             ρ = fixed_total_density, T = FT(280), ℋ = relative_humidity,
             qᶜˡ = relative_humidity_cloud, nᶜˡ = relative_humidity_number,
             enforce_mass_conservation = false)

        total_relative_humidity_fields = model_with_total_and_relative_humidity.microphysical_fields
        @test all(≈(fixed_total_density),
                  cpu(model_with_total_and_relative_humidity.dynamics.total_density))
        @test cpu(total_relative_humidity_fields.ρqᶜˡ) ≈
              fill(fixed_total_density * relative_humidity_cloud,
                   size(dry_relative_humidity_total))
        @test cpu(model_with_total_and_relative_humidity.dynamics.dry_density) ≈
              fixed_total_density .-
              cpu(model_with_total_and_relative_humidity.moisture_density) .-
              cpu(total_relative_humidity_fields.ρqᶜˡ)
        @test all(≈(FT(280)), cpu(model_with_total_and_relative_humidity.temperature))
        @test cpu(RelativeHumidityField(model_with_total_and_relative_humidity)) ≈
              fill(relative_humidity, size(dry_relative_humidity_total)) rtol = 1e-10

        # Compressible relative-humidity initialization must not require the optional
        # hydrostatic reference state; it closes against prognostic total density and
        # the equation-of-state pressure.
        dynamics_without_reference =
            CompressibleDynamics(SplitExplicitTimeDiscretization();
                                 surface_pressure = FT(1e5),
                                 standard_pressure = FT(1e5))
        model_without_reference =
            AtmosphereModel(grid; dynamics = dynamics_without_reference,
                            thermodynamic_constants = constants,
                            microphysics = PredictedParticlePropertiesMicrophysics(FT),
                            timestepper = :AcousticRungeKutta3)
        set!(model_without_reference; ρ = fixed_total_density, T = FT(280),
             ℋ = relative_humidity, enforce_mass_conservation = false)
        @test cpu(RelativeHumidityField(model_without_reference)) ≈
              fill(relative_humidity, size(dry_relative_humidity_total)) rtol = 1e-10

        conflicting_moisture_model = make_model()
        @test_throws ArgumentError set!(conflicting_moisture_model;
                                        ρᵈ = dry_density, T = FT(280),
                                        ℋ = relative_humidity, qᵛ = FT(0.01),
                                        enforce_mass_conservation = false)

        hydrostatic_total_water = FT(0.02)
        hydrostatic_model = make_model()
        set!(hydrostatic_model;
             ρ = HydrostaticallyBalancedDensity(surface_pressure = FT(1e5)),
             T = FT(280), qᵗ = hydrostatic_total_water,
             enforce_mass_conservation = false)

        hydrostatic_density = cpu(hydrostatic_model.dynamics.total_density)
        @test cpu(hydrostatic_model.moisture_density) ./ hydrostatic_density ≈
              fill(hydrostatic_total_water, size(hydrostatic_density))
        @test hydrostatic_density ≈
              cpu(hydrostatic_model.dynamics.dry_density) .+
              cpu(hydrostatic_model.moisture_density)

        hydrostatic_condensate_model = make_model()
        @test_throws ArgumentError set!(hydrostatic_condensate_model;
                                        ρ = HydrostaticallyBalancedDensity(surface_pressure = FT(1e5)),
                                        T = FT(280), qᵗ = hydrostatic_total_water,
                                        qᶜˡ = FT(0.005),
                                        enforce_mass_conservation = false)

        hydrostatic_relative_humidity_model = make_model()
        @test_throws ArgumentError set!(hydrostatic_relative_humidity_model;
                                        ρ = HydrostaticallyBalancedDensity(surface_pressure = FT(1e5)),
                                        T = FT(280), ℋ = relative_humidity,
                                        enforce_mass_conservation = false)
    end

    @testset "P3 produces rain on its first RK step" begin
        FT = Float64
        grid = RectilinearGrid(default_arch, FT; size = (2, 2, 2),
                               extent = (100, 100, 100))
        constants = ThermodynamicConstants(FT)
        reference_state = ReferenceState(grid, constants;
                                         surface_pressure = FT(101325),
                                         potential_temperature = FT(285))
        dynamics = AnelasticDynamics(reference_state)
        # The point of the budget check below is that the cloud -> rain conversion conserves
        # water, so the domain has to be closed: with the default open surface, cloud
        # droplets and rain sediment out through the bottom face and the interior total is
        # *supposed* to drop. Periodic sides plus an impenetrable surface plus the zero top
        # face make the box closed, so the remaining budget is purely microphysical.
        p3 = PredictedParticlePropertiesMicrophysics(
            FT; precipitation_boundary_condition = ImpenetrableBoundaryCondition())
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                                microphysics = p3)

        set!(model; θ = FT(285), qᵛ = FT(0.01), qᶜˡ = FT(0.003),
             nᶜˡ = FT(2e8), enforce_mass_conservation = false)
        @test all(Array(interior(model.microphysical_fields.ρqʳ)) .== 0)

        μ = model.microphysical_fields
        total_water_before = sum(Array(interior(model.moisture_density))) +
                             sum(Array(interior(μ.ρqᶜˡ))) +
                             sum(Array(interior(μ.ρqʳ))) +
                             sum(Array(interior(μ.ρqⁱ))) +
                             sum(Array(interior(μ.ρqʷⁱ)))

        time_step!(model, FT(0.01))

        @test any(Array(interior(model.microphysical_fields.ρqʳ)) .> 0)
        total_water_after = sum(Array(interior(model.moisture_density))) +
                            sum(Array(interior(μ.ρqᶜˡ))) +
                            sum(Array(interior(μ.ρqʳ))) +
                            sum(Array(interior(μ.ρqⁱ))) +
                            sum(Array(interior(μ.ρqʷⁱ)))
        @test total_water_after ≈ total_water_before rtol = 1e-12
    end

    @testset "P3 runs under the acoustic RK stepper" begin
        FT = Float64
        grid = RectilinearGrid(
            CPU(), FT; size = (5, 5, 5), halo = (5, 5, 5),
            extent = (100, 100, 100))
        dynamics = CompressibleDynamics(
            SplitExplicitTimeDiscretization();
            surface_pressure = FT(1e5),
            standard_pressure = FT(1e5),
            reference_potential_temperature = z -> FT(280))
        model = AtmosphereModel(
            grid; dynamics,
            microphysics = PredictedParticlePropertiesMicrophysics(FT),
            timestepper = :AcousticRungeKutta3)
        set!(model; ρᵈ = FT(1), T = FT(280), qᵛ = FT(0.01),
             qᶜˡ = FT(1e-4), nᶜˡ = FT(1e8),
             enforce_mass_conservation = false)

        Δt = FT(0.01)
        time_step!(model, Δt)
        μ = model.microphysical_fields

        @test all(isfinite, Array(interior(μ.ρqᶜˡ)))
        @test all(isfinite, Array(interior(model.moisture_density)))
    end
end
