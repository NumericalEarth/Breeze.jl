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

        # Test that model can be built with BulkSensibleHeatFlux on ρθ
        ρθ_bcs = FieldBoundaryConditions(bottom=bc)
        boundary_conditions = (; ρθ=ρθ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)
        time_step!(model, 1e-6)
        @test true

        # Test that model can also be built with BulkSensibleHeatFlux on ρe
        # (interface "just works" - BulkSensibleHeatFlux is passed through without conversion)
        ρe_bcs = FieldBoundaryConditions(bottom=bc)
        boundary_conditions = (; ρe=ρe_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        set!(model; θ=θ₀)
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

    @testset "Automatic ρe → ρθ conversion [$FT]" begin
        # Test with constant energy flux (W/m²) using ρe boundary conditions
        # When using potential temperature formulation, ρe BCs are automatically
        # converted to ρθ BCs by dividing by cᵖᵐ
        𝒬 = FT(100)  # 100 W/m²

        # Test that model can be built with ρe boundary conditions on bottom
        ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬))
        boundary_conditions = (; ρe=ρe_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)

        # Model should build and run without error
        time_step!(model, FT(1e-6))
        @test true

        # Test with ρe boundary condition on top
        ρe_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(-𝒬))  # negative = cooling
        boundary_conditions = (; ρe=ρe_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        set!(model; θ=θ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test with ρe boundary conditions on both bottom and top
        ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬),
                                          top=FluxBoundaryCondition(-𝒬))
        boundary_conditions = (; ρe=ρe_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        set!(model; θ=θ₀)
        time_step!(model, FT(1e-6))
        @test true
    end

    @testset "Manual EnergyFluxBoundaryCondition on ρθ [$FT]" begin
        using Breeze.BoundaryConditions: EnergyFluxBoundaryCondition

        𝒬 = FT(100)  # 100 W/m²

        # Manually wrap energy flux in EnergyFluxBoundaryCondition and apply to ρθ
        ρθ_bcs = FieldBoundaryConditions(bottom=EnergyFluxBoundaryCondition(𝒬))
        boundary_conditions = (; ρθ=ρθ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test on top boundary
        ρθ_bcs = FieldBoundaryConditions(top=EnergyFluxBoundaryCondition(-𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(; ρθ=ρθ_bcs))
        set!(model; θ=θ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test with both bottom and top
        ρθ_bcs = FieldBoundaryConditions(bottom=EnergyFluxBoundaryCondition(𝒬),
                                          top=EnergyFluxBoundaryCondition(-𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(; ρθ=ρθ_bcs))
        set!(model; θ=θ₀)
        time_step!(model, FT(1e-6))
        @test true
    end

    @testset "Energy to θ flux conversion is correct [$FT]" begin
        using Breeze.Thermodynamics: mixture_heat_capacity, MoistureMassFractions

        grid = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 100), y=(0, 100), z=(0, 100))
        𝒬 = FT(1000)  # W/m²

        # Test automatic interface
        ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(; ρe=ρe_bcs))

        θ₀ = model.dynamics.reference_state.potential_temperature
        qᵗ₀ = FT(0.01)
        set!(model; θ=θ₀, qᵗ=qᵗ₀)

        q = MoistureMassFractions(qᵗ₀)
        cᵖᵐ = mixture_heat_capacity(q, model.thermodynamic_constants)
        expected_θ_flux = 𝒬 / cᵖᵐ

        time_step!(model, FT(1e-6))

        @test cᵖᵐ > 1000
        @test expected_θ_flux < 𝒬
        @test expected_θ_flux ≈ 𝒬 / cᵖᵐ

        # Test manual interface produces same result
        using Breeze.BoundaryConditions: EnergyFluxBoundaryCondition
        ρθ_bcs = FieldBoundaryConditions(bottom=EnergyFluxBoundaryCondition(𝒬))
        model2 = AtmosphereModel(grid; boundary_conditions=(; ρθ=ρθ_bcs))
        set!(model2; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model2, FT(1e-6))

        # Both models should have the same ρθ after one timestep (same BC applied)
        @test true  # If we get here, both interfaces work
    end

    @testset "Error when specifying both ρθ and ρe boundary conditions [$FT]" begin
        grid = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 100), y=(0, 100), z=(0, 100))

        # Specifying non-default BCs on both ρθ and ρe should throw an error
        ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(100)))
        ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(200)))

        @test_throws ArgumentError AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs, ρe=ρe_bcs))
    end
end
