using Breeze
using Oceananigans
using Test

test_thermodynamics = (:StaticEnergy, :LiquidIcePotentialTemperature)

@testset "Time stepping with TurbulenceClosures [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 100), y=(0, 100), z=(0, 100))
    vitd = VerticallyImplicitTimeDiscretization()

    closures = (
        ScalarDiffusivity(ν=1, κ=2),
        ScalarDiffusivity(vitd, ν=1),
        SmagorinskyLilly(),
        AnisotropicMinimumDissipation()
    )

    for closure in closures
        @testset let closure=closure
            model = AtmosphereModel(grid; closure)
            time_step!(model, 1)
            @test true
        end
    end

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants)
    dynamics = AnelasticDynamics(reference_state)
    etd = Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization()
    discretizations = (vitd, etd)

    @testset "AtmosphereModel with $formulation thermodynamics [$FT]" for formulation in test_thermodynamics

        @testset "Implicit diffusion solver with ScalarDiffusivity [$formulation, $(FT), $(typeof(disc))]" for disc in discretizations
            closure = ScalarDiffusivity(disc, ν=1, κ=1)
            model = AtmosphereModel(grid; dynamics, formulation, closure, tracers=:ρc)
            # Set uniform specific energy for no diffusion
            θ₀ = model.dynamics.reference_state.potential_temperature
            cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
            e₀ = cᵖᵈ * θ₀
            set!(model; e=e₀)
            ρe₀ = deepcopy(static_energy_density(model))
            time_step!(model, 1)
            # Use rtol for implicit solver which may have small numerical effects
            @test isapprox(static_energy_density(model), ρe₀, rtol=1e-5)
        end

        @testset "Closure flux affects momentum tendency [$formulation, $(FT)]" begin
            closure = ScalarDiffusivity(ν=1e4)
            model = AtmosphereModel(grid; dynamics, formulation, advection=nothing, closure)
            set!(model; ρu = (x, y, z) -> exp((z - 50)^2 / (2 * 20^2)))
            Breeze.AtmosphereModels.compute_tendencies!(model)
            Gρu = model.timestepper.Gⁿ.ρu
            @test maximum(abs, Gρu) > 0
        end

        @testset "SmagorinskyLilly with velocity gradients [$formulation, $(FT)]" begin
            model = AtmosphereModel(grid; dynamics, formulation, closure=SmagorinskyLilly())
            θ₀ = model.dynamics.reference_state.potential_temperature
            set!(model; θ=θ₀, ρu = (x, y, z) -> z / 100)
            Breeze.AtmosphereModels.update_state!(model)
            @test maximum(abs, model.closure_fields.νₑ) > 0
        end

        @testset "AnisotropicMinimumDissipation with velocity gradients [$formulation, $(FT)]" begin
            model = AtmosphereModel(grid; dynamics, formulation, closure=AnisotropicMinimumDissipation())
            set!(model; ρu = (x, y, z) -> z / 100)
            Breeze.AtmosphereModels.update_state!(model)
            @test haskey(model.closure_fields, :νₑ) || haskey(model.closure_fields, :κₑ)
        end

        # Test LES scalar diffusion with advection=nothing
        # This isolates the effect of the closure on scalar fields
        les_closures = (SmagorinskyLilly(), AnisotropicMinimumDissipation())

        @testset "LES scalar diffusion without advection [$formulation, $(FT), $(nameof(typeof(closure)))]" for closure in les_closures
            model = AtmosphereModel(grid; dynamics, formulation, closure, advection=nothing, tracers=:ρc)

            # Set random velocity field to trigger non-zero eddy diffusivity
            Ξ(x, y, z) = randn()

            # Set scalar gradients for energy, moisture, and passive tracer
            θ₀ = model.dynamics.reference_state.potential_temperature
            z₀, dz = 50, 10
            gaussian(z) = exp(- (z - z₀)^2 / 2dz^2)
            θᵢ(x, y, z) = θ₀ + 10 * gaussian(z)
            qᵗᵢ(x, y, z) = 0.01 + 1e-3 * gaussian(z)
            ρcᵢ(x, y, z) = gaussian(z)
            set!(model; θ = θᵢ, ρqᵗ = qᵗᵢ, ρc = ρcᵢ, ρu = Ξ, ρv = Ξ, ρw = Ξ)

            # Store initial scalar fields (using copy of data to avoid reference issues)
            ρe₀ = static_energy_density(model) |> Field |> interior |> Array
            ρqᵛ₀ = model.moisture_density |> interior |> Array
            ρc₀ = model.tracers.ρc |> interior |> Array

            # Take a time step
            time_step!(model, 1)

            ρe₁ = static_energy_density(model) |> Field |> interior |> Array
            ρqᵛ₁ = model.moisture_density |> interior |> Array
            ρc₁ = model.tracers.ρc |> interior |> Array

            # Scalars should change due to diffusion (not advection since advection=nothing)
            # Use explicit maximum difference check instead of ≈ to handle Float32
            @test maximum(abs, ρe₁ - ρe₀) > 0
            @test maximum(abs, ρqᵛ₁ - ρqᵛ₀) > 0
            @test maximum(abs, ρc₁ - ρc₀) > 0
        end
    end
end

@testset "Surface layer consistency [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Lz = FT(400)
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 400), y=(0, 400), z=(0, Lz))

    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants)
    dynamics = AnelasticDynamics(reference_state)

    Cᴰ = FT(1e-3)
    drag = Breeze.BulkDrag(coefficient=Cᴰ, gustiness=FT(0.5))

    ρu_bcs = FieldBoundaryConditions(bottom=drag)
    ρv_bcs = FieldBoundaryConditions(bottom=drag)

    for closure in (SmagorinskyLilly(), AnisotropicMinimumDissipation())
        @testset "$(nameof(typeof(closure)))" begin
            model = AtmosphereModel(grid; dynamics, closure,
                                    boundary_conditions=(; ρu=ρu_bcs, ρv=ρv_bcs))

            # Set a velocity profile that produces shear and non-zero Smagorinsky νₑ
            set!(model; ρu = (x, y, z) -> z / Lz, ρv = (x, y, z) -> FT(0.5) * z / Lz)

            Breeze.AtmosphereModels.update_state!(model)

            # Check that νₑ at k=1 matches the MOST prediction
            νₑ = model.closure_fields.νₑ
            κ = FT(0.4) # von Kármán constant
            z₁ = Lz / 16 # midpoint of first cell (Nz=8, uniform grid)

            u₁ = interior(model.velocities.u, 2, 2, 1)[]
            v₁ = interior(model.velocities.v, 2, 2, 1)[]
            U = sqrt(u₁^2 + v₁^2)

            gustiness = FT(0.5)
            u_star = sqrt(Cᴰ * (U^2 + gustiness^2))
            νₑ_expected = κ * u_star * z₁

            # Check at an interior point (avoid boundary halo effects)
            νₑ_actual = interior(νₑ, 2, 2, 1)[]
            @test νₑ_actual ≈ νₑ_expected rtol=FT(0.1)

            # Verify it differs from what the LES closure alone would produce at k=2
            νₑ_interior = interior(νₑ, 2, 2, 2)[]
            @test νₑ_actual != νₑ_interior
        end
    end
end
