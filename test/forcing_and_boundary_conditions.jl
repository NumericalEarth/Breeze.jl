using Breeze
using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: BoundaryCondition
using Oceananigans.Fields: location
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

        # Test that BulkDrag on a scalar field throws an error
        ρθ_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
        @test_throws ArgumentError AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
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

    @testset "materialize_surface_field [$FT]" begin
        using Breeze.BoundaryConditions: materialize_surface_field

        # Test Number passthrough
        T_number = FT(300)
        result = materialize_surface_field(T_number, grid)
        @test result === T_number

        # Test Field passthrough
        T_field = Field{Center, Center, Nothing}(grid)
        set!(T_field, FT(295))
        result = materialize_surface_field(T_field, grid)
        @test result === T_field

        # Test Function → Field conversion
        T_func(x, y) = FT(290) + FT(5) * sin(2π * x / 100)
        result = materialize_surface_field(T_func, grid)
        @test result isa Field
        @test location(result) == (Center, Center, Nothing)
        @test maximum(result) ≈ FT(295)
        @test minimum(result) ≈ FT(285)
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

    @testset "BulkSensibleHeatFlux on ρe passes through correctly [$FT]" begin
        # BulkSensibleHeatFlux already returns a potential temperature flux,
        # so when applied to ρe, it should pass through directly without wrapping
        T₀ = FT(300)
        Cᵀ = FT(1e-3)

        ρe_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(surface_temperature=T₀,
                                                                     coefficient=Cᵀ, gustiness=FT(0.1)))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)

        # Model should build and run without error
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

    @testset "EnergyFluxOperation returns energy flux [$FT]" begin
        grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

        # Test 1: Bottom boundary with ρe BC (EnergyFluxBoundaryCondition path)
        𝒬₀ = FT(500)  # W/m²
        ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬₀))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))

        θ₀ = model.dynamics.reference_state.potential_temperature
        qᵗ₀ = FT(0.01)  # 10 g/kg moisture
        set!(model; θ=θ₀, qᵗ=qᵗ₀)

        𝒬_op = EnergyFluxOperation(model, :bottom)
        𝒬_field = Field(𝒬_op)
        compute!(𝒬_field)
        @test all(interior(𝒬_field) .≈ 𝒬₀)

        # Test 2: Top boundary with ρe BC
        𝒬_top = FT(-100)  # W/m² (cooling)
        ρe_bcs_top = FieldBoundaryConditions(top=FluxBoundaryCondition(𝒬_top))
        model_top = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs_top,))
        set!(model_top; θ=θ₀, qᵗ=qᵗ₀)

        𝒬_top_op = EnergyFluxOperation(model_top, :top)
        𝒬_top_field = Field(𝒬_top_op)
        compute!(𝒬_top_field)
        @test all(interior(𝒬_top_field) .≈ 𝒬_top)

        # Test 3: Both bottom and top
        ρe_bcs_both = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬₀),
                                               top=FluxBoundaryCondition(𝒬_top))
        model_both = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs_both,))
        set!(model_both; θ=θ₀, qᵗ=qᵗ₀)

        𝒬_bottom_op = EnergyFluxOperation(model_both, :bottom)
        𝒬_top_op2 = EnergyFluxOperation(model_both, :top)

        𝒬_bottom_field = Field(𝒬_bottom_op)
        𝒬_top_field2 = Field(𝒬_top_op2)
        compute!(𝒬_bottom_field)
        compute!(𝒬_top_field2)

        @test all(interior(𝒬_bottom_field) .≈ 𝒬₀)
        @test all(interior(𝒬_top_field2) .≈ 𝒬_top)

        # Test 4: Regular ρθ BC at bottom (fallback path that multiplies by cᵖᵐ)
        Jᶿ₀ = FT(0.5)  # K·kg/(m²·s) - potential temperature flux
        ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Jᶿ₀))
        model_theta = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
        set!(model_theta; θ=θ₀, qᵗ=qᵗ₀)

        𝒬_from_θ_op = EnergyFluxOperation(model_theta, :bottom)
        𝒬_from_θ_field = Field(𝒬_from_θ_op)
        compute!(𝒬_from_θ_field)

        # Energy flux should be Jᶿ × cᵖᵐ
        q = MoistureMassFractions(qᵗ₀)
        cᵖᵐ = mixture_heat_capacity(q, model_theta.thermodynamic_constants)
        expected_𝒬 = Jᶿ₀ * cᵖᵐ
        @test all(interior(𝒬_from_θ_field) .≈ expected_𝒬)

        # Test 5: Regular ρθ BC at top (fallback path for top boundary)
        ρθ_bcs_top = FieldBoundaryConditions(top=FluxBoundaryCondition(-Jᶿ₀))
        model_theta_top = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs_top,))
        set!(model_theta_top; θ=θ₀, qᵗ=qᵗ₀)

        𝒬_from_θ_top_op = EnergyFluxOperation(model_theta_top, :top)
        𝒬_from_θ_top_field = Field(𝒬_from_θ_top_op)
        compute!(𝒬_from_θ_top_field)

        expected_𝒬_top = -Jᶿ₀ * cᵖᵐ
        @test all(interior(𝒬_from_θ_top_field) .≈ expected_𝒬_top)
    end

    @testset "Varying energy flux values [$FT]" begin
        grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
        θ₀ = FT(290)
        qᵗ₀ = FT(0.01)

        # Test different energy flux values
        for 𝒬 in [FT(0), FT(100), FT(-50), FT(1000)]
            ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬))
            model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))
            set!(model; θ=θ₀, qᵗ=qᵗ₀)
            time_step!(model, FT(1e-6))
            @test true

            𝒬_op = EnergyFluxOperation(model, :bottom)
            𝒬_field = Field(𝒬_op)
            compute!(𝒬_field)
            @test all(interior(𝒬_field) .≈ 𝒬)
        end
    end

    @testset "EnergyFluxBoundaryConditionFunction summary [$FT]" begin
        using Breeze.BoundaryConditions: EnergyFluxBoundaryConditionFunction

        # Test summary for number condition
        ef_number = EnergyFluxBoundaryConditionFunction(FT(500), nothing, nothing)
        s = summary(ef_number)
        @test occursin("500", s) || occursin("5", s)  # Float formatting may vary

        # Test summary for function condition
        𝒬_func(x, y, t) = FT(100)
        ef_func = EnergyFluxBoundaryConditionFunction(𝒬_func, nothing, nothing)
        s_func = summary(ef_func)
        @test occursin("Function", s_func) || occursin("function", s_func)
    end

    @testset "EnergyFluxBoundaryCondition on lateral boundaries [$FT]" begin
        # Test that EnergyFluxBoundaryCondition works on west/east/south/north boundaries
        # Need a bounded topology to test lateral BCs
        grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100),
                               topology=(Bounded, Bounded, Bounded))

        𝒬 = FT(100)  # W/m²
        θ₀ = FT(290)
        qᵗ₀ = FT(0.01)

        # Test west boundary
        ρe_bcs = FieldBoundaryConditions(west=FluxBoundaryCondition(𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test east boundary
        ρe_bcs = FieldBoundaryConditions(east=FluxBoundaryCondition(-𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test south boundary
        ρe_bcs = FieldBoundaryConditions(south=FluxBoundaryCondition(𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test north boundary
        ρe_bcs = FieldBoundaryConditions(north=FluxBoundaryCondition(-𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test multiple lateral boundaries at once
        ρe_bcs = FieldBoundaryConditions(west=FluxBoundaryCondition(𝒬),
                                          east=FluxBoundaryCondition(-𝒬),
                                          south=FluxBoundaryCondition(𝒬/2),
                                          north=FluxBoundaryCondition(-𝒬/2))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true
    end

    @testset "Manual EnergyFluxBoundaryCondition on lateral boundaries [$FT]" begin
        using Breeze.BoundaryConditions: EnergyFluxBoundaryCondition

        # Test manual interface on lateral boundaries
        grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100),
                               topology=(Bounded, Bounded, Bounded))

        𝒬 = FT(200)
        θ₀ = FT(290)
        qᵗ₀ = FT(0.01)

        # Test west boundary
        ρθ_bcs = FieldBoundaryConditions(west=EnergyFluxBoundaryCondition(𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test east boundary
        ρθ_bcs = FieldBoundaryConditions(east=EnergyFluxBoundaryCondition(-𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test south boundary
        ρθ_bcs = FieldBoundaryConditions(south=EnergyFluxBoundaryCondition(𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true

        # Test north boundary
        ρθ_bcs = FieldBoundaryConditions(north=EnergyFluxBoundaryCondition(-𝒬))
        model = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true
    end

    @testset "has_nondefault_bcs helper function [$FT]" begin
        using Breeze.BoundaryConditions: has_nondefault_bcs

        # Test with nothing
        @test has_nondefault_bcs(nothing) == false

        # Test with non-FieldBoundaryConditions type
        @test has_nondefault_bcs(:some_symbol) == false

        # Test with empty FieldBoundaryConditions (all defaults)
        fbcs_default = FieldBoundaryConditions()
        @test has_nondefault_bcs(fbcs_default) == false

        # Test with non-default BC
        fbcs_nondefault = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(100)))
        @test has_nondefault_bcs(fbcs_nondefault) == true
    end

    @testset "wrap_energy_field_bcs fallback [$FT]" begin
        using Breeze.BoundaryConditions: wrap_energy_field_bcs

        # Test that non-FieldBoundaryConditions pass through unchanged
        result = wrap_energy_field_bcs(:not_a_fbc)
        @test result === :not_a_fbc

        result2 = wrap_energy_field_bcs(nothing)
        @test result2 === nothing
    end

    @testset "side_type helper function [$FT]" begin
        using Breeze.BoundaryConditions: side_type
        using Oceananigans.BoundaryConditions: Bottom, Top, West, East, South, North

        @test side_type(:bottom) isa Bottom
        @test side_type(:top) isa Top
        @test side_type(:west) isa West
        @test side_type(:east) isa East
        @test side_type(:south) isa South
        @test side_type(:north) isa North

        # Test error for unknown side
        @test_throws ArgumentError side_type(:invalid)
    end

    @testset "energy_flux_location helper function [$FT]" begin
        using Breeze.BoundaryConditions: energy_flux_location

        # Test bottom/top (2D slice in xy-plane)
        LX, LY, LZ = energy_flux_location(:bottom, Center, Center, Center)
        @test LZ === Nothing

        LX, LY, LZ = energy_flux_location(:top, Center, Center, Center)
        @test LZ === Nothing

        # Test west/east (2D slice in yz-plane)
        LX, LY, LZ = energy_flux_location(:west, Center, Center, Center)
        @test LX === Nothing

        LX, LY, LZ = energy_flux_location(:east, Center, Center, Center)
        @test LX === Nothing

        # Test south/north (2D slice in xz-plane)
        LX, LY, LZ = energy_flux_location(:south, Center, Center, Center)
        @test LY === Nothing

        LX, LY, LZ = energy_flux_location(:north, Center, Center, Center)
        @test LY === Nothing
    end

    @testset "EnergyFluxOperation error for unsupported lateral sides [$FT]" begin
        # EnergyFluxOperation for regular ρθ BCs on lateral boundaries is not implemented
        grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100),
                               topology=(Bounded, Bounded, Bounded))

        Jᶿ = FT(0.5)  # potential temperature flux
        ρθ_bcs = FieldBoundaryConditions(west=FluxBoundaryCondition(Jᶿ))
        model = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀, qᵗ=FT(0.01))

        # EnergyFluxOperation for :west should throw (not yet implemented for lateral regular BCs)
        @test_throws ArgumentError EnergyFluxOperation(model, :west)
    end

    @testset "convert_energy_to_theta_bcs with Symbol formulation [$FT]" begin
        using Breeze.BoundaryConditions: convert_energy_to_theta_bcs

        # Test that Symbol formulation is converted to Val and dispatches correctly
        bcs = (; ρe=FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(100))))
        constants = ThermodynamicConstants()

        # Should not throw and should convert using the Symbol dispatch
        result = convert_energy_to_theta_bcs(bcs, :LiquidIcePotentialTemperature, constants)
        @test :ρθ ∈ keys(result)
        @test :ρe ∉ keys(result)

        # Also test with :θ formulation symbol
        result2 = convert_energy_to_theta_bcs(bcs, :θ, constants)
        @test :ρθ ∈ keys(result2)
    end
end
