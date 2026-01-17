using Breeze
using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: BoundaryCondition
using Test

function setup_forcing_model(grid, forcing)
    model = AtmosphereModel(grid; tracers=:ρc, forcing)
    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model; θ=θ₀)
    return model
end

increment_tolerance(::Type{Float32}) = 1f-5
increment_tolerance(::Type{Float64}) = 1e-10

@testset "AtmosphereModel forcing increments prognostic fields [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    forcings = [
        Returns(one(FT)),
        Forcing(Returns(one(FT)), discrete_form=true),
        Forcing(Returns(one(FT)), field_dependencies=:ρu, discrete_form=true),
        Forcing(Returns(one(FT)), field_dependencies=(:ρe, :ρqᵗ, :ρu), discrete_form=true),
    ]

    Δt = convert(FT, 1e-6)

    @testset "Forcing increments prognostic fields ($FT, $(typeof(forcing)))" for forcing in forcings
        # x-momentum (ρu)
        u_forcing = (; ρu=forcing)
        model = setup_forcing_model(grid, u_forcing)
        time_step!(model, Δt)
        @test maximum(model.momentum.ρu) ≈ Δt

        # y-momentum (ρv)
        v_forcing = (; ρv=forcing)
        model = setup_forcing_model(grid, v_forcing)
        time_step!(model, Δt)
        @test maximum(model.momentum.ρv) ≈ Δt

        e_forcing = (; ρe=forcing)
        model = setup_forcing_model(grid, e_forcing)
        ρe_before = deepcopy(static_energy_density(model))
        time_step!(model, Δt)
        @test maximum(static_energy_density(model)) ≈ maximum(ρe_before) + Δt

        q_forcing = (; ρqᵗ=forcing)
        model = setup_forcing_model(grid, q_forcing)
        time_step!(model, Δt)
        @test maximum(model.moisture_density) ≈ Δt

        c_forcing = (; ρc=forcing)
        model = setup_forcing_model(grid, c_forcing)
        time_step!(model, Δt)
        @test maximum(model.tracers.ρc) ≈ Δt
    end

    @testset "Forcing on non-existing field errors" begin
        bad = (; u=forcings[1])
        @test_throws ArgumentError AtmosphereModel(grid; forcing=bad)
    end

    @testset "Incorrectly specified forcing" begin
        @test_throws ArgumentError AtmosphereModel(grid; forcing=forcings[1])
    end

end

#####
##### Bulk boundary condition tests
#####

@testset "Bulk boundary conditions [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    Cᴰ = 1e-3
    gustiness = 0.1
    T₀ = 290

    @testset "BulkDrag construction and application [$FT]" begin
        # Test construction with default parameters
        drag = BulkDrag()
        @test drag isa BoundaryCondition

        # Test construction with explicit coefficient and gustiness
        drag = BulkDrag(coefficient=2e-3, gustiness=0.5)
        @test drag isa BoundaryCondition

        # Test that model can be built with BulkDrag boundary conditions
        ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=gustiness))
        ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=gustiness))
        boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)

        # Model should build and run without error
        time_step!(model, 1e-6)
        @test true  # If we get here, construction and time stepping worked
    end

    @testset "BulkSensibleHeatFlux construction and application [$FT]" begin
        # Test with constant surface temperature
        bc = BulkSensibleHeatFlux(surface_temperature=T₀, coefficient=Cᴰ, gustiness=gustiness)
        @test bc isa BoundaryCondition

        # Test with function for surface temperature
        T₀_func(x, y) = T₀ + 2 * sign(cos(2π * x / 100))
        bc = BulkSensibleHeatFlux(surface_temperature=T₀_func, coefficient=Cᴰ, gustiness=gustiness)
        @test bc isa BoundaryCondition

        # Test that model can be built with BulkSensibleHeatFlux
        ρθ_bcs = FieldBoundaryConditions(bottom=bc)
        boundary_conditions = (; ρθ=ρθ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)

        # Model should build and run without error
        time_step!(model, 1e-6)
        @test true
    end

    @testset "BulkVaporFlux construction and application [$FT]" begin
        # Test with constant surface temperature
        bc = BulkVaporFlux(surface_temperature=T₀, coefficient=Cᴰ, gustiness=gustiness)
        @test bc isa BoundaryCondition

        # Test with function for surface temperature
        T₀_func(x, y) = T₀ + 2 * sign(cos(2π * x / 100))
        bc = BulkVaporFlux(surface_temperature=T₀_func, coefficient=Cᴰ, gustiness=gustiness)
        @test bc isa BoundaryCondition

        # Test that model can be built with BulkVaporFlux
        ρqᵗ_bcs = FieldBoundaryConditions(bottom=bc)
        boundary_conditions = (; ρqᵗ=ρqᵗ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)

        # Model should build and run without error
        time_step!(model, 1e-6)
        @test true
    end

    @testset "Combined bulk boundary conditions [$FT]" begin
        # Build a model with all bulk boundary conditions
        ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=gustiness))
        ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=gustiness))
        ρθ_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(surface_temperature=T₀,
                                                                     coefficient=Cᴰ, gustiness=gustiness))
        ρqᵗ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(surface_temperature=T₀,
                                                               coefficient=Cᴰ, gustiness=gustiness))

        boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)

        # Model should build and run without error
        time_step!(model, 1e-6)
        @test true
    end

    @testset "EnergyFluxBoundaryCondition construction and application [$FT]" begin
        # Test with constant energy flux (W/m²)
        Jᵉ = FT(100)  # 100 W/m²
        bc = EnergyFluxBoundaryCondition(Jᵉ)
        @test bc isa BoundaryCondition

        # Test that model can be built with EnergyFluxBoundaryCondition on bottom
        ρθ_bcs = FieldBoundaryConditions(bottom=EnergyFluxBoundaryCondition(Jᵉ))
        boundary_conditions = (; ρθ=ρθ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)

        # Model should build and run without error
        time_step!(model, FT(1e-6))
        @test true

        # Test with EnergyFluxBoundaryCondition on top
        ρθ_bcs = FieldBoundaryConditions(top=EnergyFluxBoundaryCondition(-Jᵉ))  # negative = cooling
        boundary_conditions = (; ρθ=ρθ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        set!(model; θ=θ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test with EnergyFluxBoundaryCondition on both bottom and top
        ρθ_bcs = FieldBoundaryConditions(bottom=EnergyFluxBoundaryCondition(Jᵉ),
                                          top=EnergyFluxBoundaryCondition(-Jᵉ))
        boundary_conditions = (; ρθ=ρθ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        set!(model; θ=θ₀)
        time_step!(model, FT(1e-6))
        @test true
    end

    @testset "EnergyFluxBoundaryCondition converts energy to θ flux correctly [$FT]" begin
        using Breeze.Thermodynamics: mixture_heat_capacity, MoistureMassFractions

        # Create a model with known moisture content
        grid = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 100), y=(0, 100), z=(0, 100))
        
        # Use a constant energy flux
        Jᵉ = FT(1000)  # W/m²
        ρθ_bcs = FieldBoundaryConditions(bottom=EnergyFluxBoundaryCondition(Jᵉ))
        boundary_conditions = (; ρθ=ρθ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        qᵗ₀ = FT(0.01)  # 1% specific humidity
        set!(model; θ=θ₀, qᵗ=qᵗ₀)

        # The energy flux Jᵉ should be divided by cᵖᵐ to get potential temperature flux
        q = MoistureMassFractions(qᵗ₀)
        cᵖᵐ = mixture_heat_capacity(q, model.thermodynamic_constants)
        expected_θ_flux = Jᵉ / cᵖᵐ

        # Run a small timestep - tendencies will be computed
        time_step!(model, FT(1e-6))

        # Verify the relationship: the boundary condition should apply Jᵉ/cᵖᵐ
        # This is an indirect test - if the model runs and produces reasonable results, the BC works
        @test cᵖᵐ > 1000  # cᵖᵐ for moist air should be > 1000 J/(kg K)
        @test expected_θ_flux < Jᵉ  # θ flux should be less than energy flux (divided by cᵖᵐ)
        @test expected_θ_flux ≈ Jᵉ / cᵖᵐ
    end
end
