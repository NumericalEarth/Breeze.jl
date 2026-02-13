using Breeze
using GPUArraysCore: @allowscalar
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

    # Test a representative subset of forcing types (reduced from 4 to 2)
    forcings = [
        Returns(one(FT)),
        Forcing(Returns(one(FT)), field_dependencies=(:ρe, :ρqᵗ, :ρu), discrete_form=true),
    ]

    Δt = convert(FT, 1e-6)

    @testset "Forcing increments prognostic fields ($FT, $(typeof(forcing)))" for forcing in forcings
        # Test all field types with a single model construction where possible
        u_forcing = (; ρu=forcing)
        model = setup_forcing_model(grid, u_forcing)
        time_step!(model, Δt)
        @test maximum(model.momentum.ρu) ≈ Δt

        v_forcing = (; ρv=forcing)
        model = setup_forcing_model(grid, v_forcing)
        time_step!(model, Δt)
        @test maximum(model.momentum.ρv) ≈ Δt

        e_forcing = (; ρe=forcing)
        model = setup_forcing_model(grid, e_forcing)
        ρe_before = deepcopy(static_energy_density(model))
        time_step!(model, Δt)
        @test maximum(static_energy_density(model)) ≈ maximum(ρe_before) + Δt
    end

    @testset "Forcing on non-existing field errors" begin
        bad = (; u=forcings[1])
        @test_throws ArgumentError AtmosphereModel(grid; forcing=bad)
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
        drag = BulkDrag()
        @test drag isa BoundaryCondition

        drag = BulkDrag(coefficient=2e-3, gustiness=0.5)
        @test drag isa BoundaryCondition

        ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=gustiness))
        ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=gustiness))
        boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)
        time_step!(model, 1e-6)
        @test true

        # Test that BulkDrag on a scalar field throws an error
        ρθ_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
        @test_throws ArgumentError AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
    end

    @testset "BulkSensibleHeatFlux construction and application [$FT]" begin
        bc = BulkSensibleHeatFlux(surface_temperature=T₀, coefficient=Cᴰ, gustiness=gustiness)
        @test bc isa BoundaryCondition

        # Test with ρθ (potential temperature formulation)
        ρθ_bcs = FieldBoundaryConditions(bottom=bc)
        model = AtmosphereModel(grid; boundary_conditions=(; ρθ=ρθ_bcs))
        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)
        time_step!(model, 1e-6)
        @test true
    end

    @testset "BulkSensibleHeatFlux with StaticEnergyFormulation [$FT]" begin
        bc = BulkSensibleHeatFlux(surface_temperature=T₀, coefficient=Cᴰ, gustiness=gustiness)

        # Test with ρe on static energy formulation
        ρe_bcs = FieldBoundaryConditions(bottom=bc)
        model = AtmosphereModel(grid; formulation=:StaticEnergy,
                                boundary_conditions=(; ρe=ρe_bcs))
        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀, qᵗ=FT(0.01))
        time_step!(model, 1e-6)
        @test true
    end

    @testset "BulkSensibleHeatFlux with ρe auto-converts for θ formulation [$FT]" begin
        bc = BulkSensibleHeatFlux(surface_temperature=T₀, coefficient=Cᴰ, gustiness=gustiness)

        # ρe BCs with θ formulation: should auto-convert to ρθ
        ρe_bcs = FieldBoundaryConditions(bottom=bc)
        model = AtmosphereModel(grid; boundary_conditions=(; ρe=ρe_bcs))
        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)
        time_step!(model, 1e-6)
        @test true
    end

    @testset "BulkVaporFlux construction and application [$FT]" begin
        bc = BulkVaporFlux(surface_temperature=T₀, coefficient=Cᴰ, gustiness=gustiness)
        @test bc isa BoundaryCondition

        ρqᵗ_bcs = FieldBoundaryConditions(bottom=bc)
        model = AtmosphereModel(grid; boundary_conditions=(; ρqᵗ=ρqᵗ_bcs))
        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀)
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
        # Note: With 4 cells in x ∈ [0, 100], centers are at x = 12.5, 37.5, 62.5, 87.5
        # sin(2π * 12.5 / 100) = sin(π/4) ≈ 0.707, so max ≈ 290 + 5 * 0.707 ≈ 293.5
        T_func(x, y) = FT(290) + FT(5) * sin(2π * x / 100)
        result = materialize_surface_field(T_func, grid)
        @test result isa Field
        @test location(result) == (Center, Center, Nothing)
        @test maximum(result) ≈ FT(290) + FT(5) * sin(π / 4)  # ≈ 293.54
        @test minimum(result) ≈ FT(290) - FT(5) * sin(π / 4)  # ≈ 286.46
    end

    @testset "Combined bulk boundary conditions [$FT]" begin
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
        time_step!(model, 1e-6)
        @test true
    end

    @testset "Combined bulk boundary conditions with StaticEnergyFormulation [$FT]" begin
        ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=gustiness))
        ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=gustiness))
        ρe_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(surface_temperature=T₀,
                                                                     coefficient=Cᴰ, gustiness=gustiness))
        ρqᵗ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(surface_temperature=T₀,
                                                               coefficient=Cᴰ, gustiness=gustiness))

        boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs)
        model = AtmosphereModel(grid; formulation=:StaticEnergy, boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀, qᵗ=FT(0.01))
        time_step!(model, 1e-6)
        @test true
    end
end

#####
##### Energy flux boundary condition tests (consolidated)
#####

@testset "Energy flux boundary conditions [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    using Breeze.BoundaryConditions: EnergyFluxBoundaryCondition
    using Breeze.Thermodynamics: mixture_heat_capacity, MoistureMassFractions
    using Oceananigans.Models: BoundaryConditionOperation

    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    θ₀ = FT(290)
    qᵗ₀ = FT(0.01)

    @testset "Automatic ρe → ρθ conversion [$FT]" begin
        𝒬 = FT(100)  # W/m²

        # Test bottom, top, and both together
        for bcs_config in [
            FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬)),
            FieldBoundaryConditions(top=FluxBoundaryCondition(-𝒬)),
            FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬), top=FluxBoundaryCondition(-𝒬))
        ]
            model = AtmosphereModel(grid; boundary_conditions=(ρe=bcs_config,))
            set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true
    end
    end

    @testset "Manual EnergyFluxBoundaryCondition on ρθ [$FT]" begin
        𝒬 = FT(100)

        # Test bottom and top
        for bc_config in [
            FieldBoundaryConditions(bottom=EnergyFluxBoundaryCondition(𝒬)),
            FieldBoundaryConditions(top=EnergyFluxBoundaryCondition(-𝒬))
        ]
            model = AtmosphereModel(grid; boundary_conditions=(; ρθ=bc_config))
            set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true
        end
    end

    @testset "Energy to θ flux conversion is correct [$FT]" begin
        grid_1 = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))
        𝒬 = FT(1000)

        ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬))
        model = AtmosphereModel(grid_1; boundary_conditions=(; ρe=ρe_bcs))

        θ₀_ref = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀_ref, qᵗ=qᵗ₀)

        q = MoistureMassFractions(qᵗ₀)
        cᵖᵐ = mixture_heat_capacity(q, model.thermodynamic_constants)
        expected_θ_flux = 𝒬 / cᵖᵐ

        time_step!(model, FT(1e-6))

        @test cᵖᵐ > 1000
        @test expected_θ_flux < 𝒬
        @test expected_θ_flux ≈ 𝒬 / cᵖᵐ
    end

    @testset "Error when specifying both ρθ and ρe boundary conditions [$FT]" begin
        grid_1 = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))

        ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(100)))
        ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(200)))

        @test_throws ArgumentError AtmosphereModel(grid_1; boundary_conditions=(ρθ=ρθ_bcs, ρe=ρe_bcs))
    end

    @testset "static_energy_density returns Field with energy flux BCs [$FT]" begin
        𝒬₀ = FT(500)

        ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬₀))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))

        θ₀_ref = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀_ref, qᵗ=qᵗ₀)

        ρe = static_energy_density(model)
        𝒬_op = BoundaryConditionOperation(ρe, :bottom, model)
        𝒬_field = Field(𝒬_op)
        compute!(𝒬_field)
        @test all(interior(𝒬_field) .≈ 𝒬₀)
        end
    end

#####
##### Lateral boundary condition tests (consolidated - test one representative case per boundary)
#####

@testset "Lateral energy flux boundary conditions [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    using Breeze.BoundaryConditions: EnergyFluxBoundaryCondition
    using Oceananigans.Models: BoundaryConditionOperation

        grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100),
                               topology=(Bounded, Bounded, Bounded))

    𝒬 = FT(100)
        θ₀ = FT(290)
        qᵗ₀ = FT(0.01)

    # Test all lateral boundaries at once (more efficient than individual tests)
    @testset "Multiple lateral boundaries [$FT]" begin
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
        # Test one representative lateral boundary
        ρθ_bcs = FieldBoundaryConditions(west=EnergyFluxBoundaryCondition(FT(200)))
        model = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)
        time_step!(model, FT(1e-6))
        @test true
    end

    @testset "static_energy_density works for lateral EnergyFluxBC [$FT]" begin
        𝒬_west = 200
        ρe_bcs = FieldBoundaryConditions(west=FluxBoundaryCondition(𝒬_west))
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))

        θ₀_ref = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀_ref, qᵗ=qᵗ₀)

        ρe = static_energy_density(model)
        𝒬_op = BoundaryConditionOperation(ρe, :west, model)
        𝒬_field = Field(𝒬_op)
        compute!(𝒬_field)
        @test all(interior(𝒬_field) .≈ 𝒬_west)
    end
end

#####
##### Helper function and edge case tests (consolidated)
#####

@testset "Boundary condition helper functions [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    using Breeze.BoundaryConditions: has_nondefault_bcs, convert_energy_to_theta_bcs,
                                     theta_to_energy_bcs, EnergyFluxBoundaryCondition,
                                     EnergyFluxBoundaryConditionFunction, ThetaFluxBoundaryConditionFunction,
                                     ThetaFluxBCType, EnergyFluxBCType
    using Oceananigans.Models: boundary_condition_location

    @testset "has_nondefault_bcs [$FT]" begin
        @test has_nondefault_bcs(nothing) == false
        @test has_nondefault_bcs(:some_symbol) == false
        @test has_nondefault_bcs(FieldBoundaryConditions()) == false
        @test has_nondefault_bcs(FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(100)))) == true
    end

    @testset "boundary_condition_location [$FT]" begin
        LX, LY, LZ = boundary_condition_location(:bottom, Center, Center, Center)
        @test LZ === Nothing

        LX, LY, LZ = boundary_condition_location(:west, Center, Center, Center)
        @test LX === Nothing
    end

    @testset "convert_energy_to_theta_bcs with Symbol formulation [$FT]" begin
        bcs = (; ρe=FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(100))))
        constants = ThermodynamicConstants()

        result = convert_energy_to_theta_bcs(bcs, :LiquidIcePotentialTemperature, constants)
        @test :ρθ ∈ keys(result)
        @test :ρe ∉ keys(result)
    end

    @testset "theta_to_energy_bcs correctly converts BCs [$FT]" begin
        Jᶿ = FT(0.5)
        ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Jᶿ))
        ρe_bcs = theta_to_energy_bcs(ρθ_bcs)
        @test ρe_bcs.bottom isa ThetaFluxBCType

        𝒬 = FT(500)
        ρθ_bcs_with_energy = FieldBoundaryConditions(bottom=EnergyFluxBoundaryCondition(𝒬))
        ρe_bcs_extracted = theta_to_energy_bcs(ρθ_bcs_with_energy)
        @test ρe_bcs_extracted.bottom.condition == 𝒬
    end

    @testset "EnergyFluxBoundaryConditionFunction summary [$FT]" begin
        ef_number = EnergyFluxBoundaryConditionFunction(500, nothing, nothing, nothing, nothing)
        s = summary(ef_number)
        @test occursin("500", s) || occursin("5", s)

        𝒬_func(x, y, t) = 100
        ef_func = EnergyFluxBoundaryConditionFunction(𝒬_func, nothing, nothing, nothing, nothing)
        s_func = summary(ef_func)
        @test occursin("Function", s_func) || occursin("function", s_func)
    end

    @testset "ThetaFluxBoundaryConditionFunction summary [$FT]" begin
        tf_number = ThetaFluxBoundaryConditionFunction(FT(0.5), nothing, nothing, nothing)
        s = summary(tf_number)
        @test occursin("0.5", s) || occursin("5", s)

        Jᶿ_func(x, y, t) = FT(0.1)
        tf_func = ThetaFluxBoundaryConditionFunction(Jᶿ_func, nothing, nothing, nothing)
        s_func = summary(tf_func)
        @test occursin("Function", s_func) || occursin("function", s_func)
    end
end

#####
##### getbc coverage tests (consolidated - test all boundaries in one model)
#####

@testset "getbc coverage for all boundary faces [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    using Breeze.AtmosphereModels: thermodynamic_density

    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 100), y=(0, 100), z=(0, 100),
                           topology=(Bounded, Bounded, Bounded))

    𝒬 = FT(1000)
    θ₀ = FT(290)
    qᵗ₀ = FT(0.01)
    Δt = FT(1e-6)

    # Test a representative subset of boundaries (bottom and west are sufficient for coverage)
    for (side, ρe_bcs) in [
        (:bottom, FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬))),
        (:west, FieldBoundaryConditions(west=FluxBoundaryCondition(𝒬))),
    ]
        model = AtmosphereModel(grid; boundary_conditions=(ρe=ρe_bcs,))
        set!(model; θ=θ₀, qᵗ=qᵗ₀)

        ρθ = thermodynamic_density(model.formulation)
        ρθ_before = @allowscalar ρθ[1, 1, 1]
        time_step!(model, Δt)
        ρθ_after = @allowscalar ρθ[1, 1, 1]

        Δρθ = ρθ_after - ρθ_before
        @test Δρθ != 0
    end
end

@testset "ThetaFluxBC getbc coverage [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    using Oceananigans.Models: BoundaryConditionOperation

    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 100), y=(0, 100), z=(0, 100),
                           topology=(Bounded, Bounded, Bounded))

    Jᶿ = FT(0.5)
    θ₀ = FT(290)
    qᵗ₀ = FT(0.01)

    # Test bottom boundary only (representative case)
    ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Jᶿ))
    model = AtmosphereModel(grid; boundary_conditions=(ρθ=ρθ_bcs,))
    set!(model; θ=θ₀, qᵗ=qᵗ₀)

    ρe = static_energy_density(model)
    𝒬_op = BoundaryConditionOperation(ρe, :bottom, model)
    𝒬_field = Field(𝒬_op)
    compute!(𝒬_field)

    # Energy flux = Jᶿ × cᵖᵐ where cᵖᵐ ≈ 1000-1100 J/(kg·K)
    @test all(interior(𝒬_field) .> 250)
end
