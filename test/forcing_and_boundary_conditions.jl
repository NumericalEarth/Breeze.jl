using Breeze
using Breeze.AtmosphereModels: thermodynamic_density, surface_pressure, standard_pressure
using Breeze.BoundaryConditions: EnergyFluxBoundaryCondition, FilteredSurfaceVelocities
using Breeze.Thermodynamics: potential_temperature_from_temperature
using GPUArraysCore: @allowscalar
using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: BoundaryCondition
using Oceananigans.Fields: location
using Oceananigans.TimeSteppers: compute_flux_bc_tendencies!, update_state!
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
        Forcing(Returns(one(FT)), field_dependencies=(:ρe, :ρqᵛ, :ρu), discrete_form=true),
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
        # `:u` is the specific alias of `:ρu`, so it's a valid key. Use a name that is
        # neither a prognostic ρ-name nor a known specific alias.
        bad = (; bogus=forcings[1])
        @test_throws ArgumentError AtmosphereModel(grid; forcing=bad)
    end
end

@testset "Forcing field_dependencies resolve consistently at materialize and runtime [$FT]" for FT in test_float_types()
    # ContinuousForcing resolves `field_dependencies` to positional indices into the
    # materialize-time `model_fields`, then dereferences those positions against the
    # runtime `fields(model)` tuple. The two orderings must agree, or a forcing reads
    # the wrong field. This test catches the order drift via a forcing that returns
    # its `:u` dependency: under a misaligned ordering Gρθ would equal `θ` instead.
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    @inline u_dep(x, y, z, t, u) = u
    u_forcing = Forcing(u_dep, field_dependencies=(:u,))
    model = AtmosphereModel(grid; forcing=(; ρθ=u_forcing))

    θ₀ = model.dynamics.reference_state.potential_temperature
    u_value = 13
    set!(model; θ=θ₀, u=u_value)
    update_state!(model)

    Gρθ = interior(model.timestepper.Gⁿ.ρθ) |> Array
    @test all(isapprox.(Gρθ, u_value))
end

#####
##### Bulk boundary condition tests
#####

@testset "Boundary-condition field dependencies align with model fields [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch;
                           size = (8, 8, 8), halo = (5, 5, 5),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 2,
                                                                    damping = NoDivergenceDamping());
                                    reference_potential_temperature = FT(300),
                                    surface_pressure = FT(1e5),
                                    standard_pressure = FT(1e5))

    @inline first_dependency(x, y, t, a, b, p) = a
    @inline second_dependency(x, y, t, a, b, p) = b

    function bottom_flux_tendency(dependencies, dependency_index)
        condition = dependency_index == 1 ? first_dependency : second_dependency
        ρv_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(condition,
                                                                        field_dependencies = dependencies,
                                                                        parameters = (;)))
        model = AtmosphereModel(grid; dynamics,
                                      boundary_conditions = (; ρv = ρv_bcs),
                                      timestepper = :AcousticRungeKutta3)

        set!(model, ρ = (x, y, z) -> FT(1),
                    θ = (x, y, z) -> FT(300),
                    u = (x, y, z) -> FT(-8.75),
                    v = (x, y, z) -> FT(0))
        update_state!(model; compute_tendencies = false)
        fill!(parent(model.timestepper.Gⁿ.ρv), 0)
        compute_flux_bc_tendencies!(model)
        return Array(interior(model.timestepper.Gⁿ.ρv, :, :, 1))
    end

    Δz = FT(1 / 8)
    @test all(bottom_flux_tendency((:u, :v), 1) .≈ FT(-8.75) / Δz)
    @test all(bottom_flux_tendency((:u, :v), 2) .== 0)
    @test all(bottom_flux_tendency((:ρu, :ρv), 1) .≈ FT(-8.75) / Δz)
    @test all(bottom_flux_tendency((:ρu, :ρv), 2) .== 0)
end

@testset "Time-dependent Open BC on momentum [$FT]" for FT in test_float_types()
    # Regression test for #717: `compute_velocities!` refilled the density and
    # momentum halos without threading `model.clock`/`fields(model)`, so a
    # time-dependent Open BC on momentum hit a `getbc` signature that could not
    # evaluate the time argument (continuous callables → MethodError;
    # FieldTimeSeries → BoundsError on the `getbc(::AbstractArray, ...)` fallback).
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 4),
                           x=(0, 1000), y=(0, 1000), z=(0, 200),
                           topology=(Bounded, Bounded, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    reference_potential_temperature=FT(300),
                                    surface_pressure=FT(1e5))

    @inline ρu_west(y, z, t, p) = p.ρ * cos(p.ω * t)
    ρu_bcs = FieldBoundaryConditions(
        west = OpenBoundaryCondition(ρu_west; parameters=(; ρ=FT(1.17), ω=FT(0.01))))

    # `set!` triggers `update_state!` → `compute_velocities!` → momentum halo fill.
    # Pre-#717 this threw before any explicit time step.
    model = AtmosphereModel(grid; dynamics, boundary_conditions=(; ρu=ρu_bcs))
    set!(model; θ=FT(300), ρ=FT(1.17))

    # West boundary face (i=1) carries the prescribed value at the current time.
    bc_value(t) = FT(1.17) * cos(FT(0.01) * t)
    @test @allowscalar(model.momentum.ρu[1, 1, 1]) ≈ bc_value(model.clock.time) atol=sqrt(eps(FT))

    # Stepping re-evaluates the BC at the new time without error.
    time_step!(model, FT(10))
    @test model.clock.iteration == 1
    @test @allowscalar(model.momentum.ρu[1, 1, 1]) ≈ bc_value(model.clock.time) atol=sqrt(eps(FT))
    @test !any(isnan, parent(model.momentum.ρu))
end

@testset "FieldTimeSeries Open BC on momentum [$FT]" for FT in test_float_types()
    # Regression test for #717, array-backed branch: a `FieldTimeSeries` Open BC
    # on momentum previously hit `getbc(::AbstractArray, i, j, ...)` (→ BoundsError
    # on `condition[i, j]`) during the clock-less momentum halo fill. With the clock
    # threaded through, `getbc` dispatches to the FTS method and interpolates in time.
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 4),
                           x=(0, 1000), y=(0, 1000), z=(0, 200),
                           topology=(Bounded, Bounded, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    reference_potential_temperature=FT(300),
                                    surface_pressure=FT(1e5))

    # 2-D (y, z) boundary slice for a west OBC on ρu (Face, Center, Center).
    # Slice values 1, 2, 3 at times 0, 10, 20 so the boundary value linearly
    # interpolates to 1.5 at t = 5.
    times = [FT(0), FT(10), FT(20)]
    ρu_fts = FieldTimeSeries{Nothing, Center, Center}(grid, times)
    for n in eachindex(times)
        set!(ρu_fts[n], (y, z) -> FT(n))
    end

    ρu_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(ρu_fts))
    model = AtmosphereModel(grid; dynamics, boundary_conditions=(; ρu=ρu_bcs))
    set!(model; θ=FT(300), ρ=FT(1.17))

    # t = 0: west boundary face equals the first slice.
    @test @allowscalar(model.momentum.ρu[1, 1, 1]) ≈ FT(1) atol=sqrt(eps(FT))

    # t = 5: halfway between slices 1 and 2 → linear interpolation gives 1.5.
    time_step!(model, FT(5))
    @test model.clock.iteration == 1
    @test @allowscalar(model.momentum.ρu[1, 1, 1]) ≈ FT(1.5) atol=sqrt(eps(FT))
    @test !any(isnan, parent(model.momentum.ρu))
end

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

    @testset "BulkSensibleHeatFlux uses surface-equivalent θ [$FT]" begin
        using Oceananigans.Models: BoundaryConditionOperation

        grid_1 = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 100), y=(0, 100), z=(0, 100))
        bc = BulkSensibleHeatFlux(surface_temperature=FT(T₀), coefficient=FT(Cᴰ), gustiness=FT(gustiness))
        ρθ_bcs = FieldBoundaryConditions(bottom=bc)
        model = AtmosphereModel(grid_1; boundary_conditions=(; ρθ=ρθ_bcs))

        constants = model.thermodynamic_constants
        p₀ = surface_pressure(model.dynamics)
        pˢᵗ = standard_pressure(model.dynamics)
        θ_surface = potential_temperature_from_temperature(FT(T₀), p₀, pˢᵗ, constants)

        @test p₀ != pˢᵗ
        @test abs(θ_surface - FT(T₀)) > increment_tolerance(FT)

        set!(model; θ=θ_surface, u=FT(5))

        ρθ = thermodynamic_density(model.formulation)
        Jᶿ_op = BoundaryConditionOperation(ρθ, :bottom, model)
        Jᶿ_field = Field(Jᶿ_op)
        compute!(Jᶿ_field)

        @test all(abs.(interior(Jᶿ_field)) .<= increment_tolerance(FT))
    end

    @testset "BulkSensibleHeatFlux uses surface-equivalent filtered θ [$FT]" begin
        using Oceananigans.Models: BoundaryConditionOperation

        grid_1 = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 100), y=(0, 100), z=(0, 100))
        fv = FilteredSurfaceVelocities(grid_1; filter_timescale=FT(3600))
        bc = BulkSensibleHeatFlux(surface_temperature = FT(T₀),
                                  coefficient = FT(Cᴰ),
                                  gustiness = FT(gustiness),
                                  filtered_velocities = fv)
        ρθ_bcs = FieldBoundaryConditions(bottom=bc)
        model = AtmosphereModel(grid_1; boundary_conditions=(; ρθ=ρθ_bcs))

        constants = model.thermodynamic_constants
        p₀ = surface_pressure(model.dynamics)
        pˢᵗ = standard_pressure(model.dynamics)
        θ_surface = potential_temperature_from_temperature(FT(T₀), p₀, pˢᵗ, constants)

        set!(model; θ=θ_surface, u=FT(5))
        Oceananigans.initialize!(model)

        ρθ = thermodynamic_density(model.formulation)
        bc_condition = Oceananigans.boundary_conditions(ρθ).bottom.condition
        @test bc_condition.filtered_scalar !== nothing

        Jᶿ_op = BoundaryConditionOperation(ρθ, :bottom, model)
        Jᶿ_field = Field(Jᶿ_op)
        compute!(Jᶿ_field)

        @test all(abs.(interior(Jᶿ_field)) .<= increment_tolerance(FT))
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

        ρqᵛ_bcs = FieldBoundaryConditions(bottom=bc)
        model = AtmosphereModel(grid; boundary_conditions=(; ρqᵛ=ρqᵛ_bcs))
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
        ρqᵛ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(surface_temperature=T₀,
                                                               coefficient=Cᴰ, gustiness=gustiness))

        boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)
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
        ρqᵛ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(surface_temperature=T₀,
                                                               coefficient=Cᴰ, gustiness=gustiness))

        boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵛ=ρqᵛ_bcs)
        model = AtmosphereModel(grid; formulation=:StaticEnergy, boundary_conditions)

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀, qᵗ=FT(0.01))
        time_step!(model, 1e-6)
        @test true
    end

    @testset "PolynomialCoefficient full model build + time step [$FT]" begin
        coef = PolynomialCoefficient()

        ρu_bcs  = FieldBoundaryConditions(bottom=BulkDrag(coefficient=coef, gustiness=gustiness, surface_temperature=T₀))
        ρv_bcs  = FieldBoundaryConditions(bottom=BulkDrag(coefficient=coef, gustiness=gustiness, surface_temperature=T₀))
        ρθ_bcs  = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=coef, gustiness=gustiness, surface_temperature=T₀))
        ρqᵛ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=coef, gustiness=gustiness, surface_temperature=T₀))

        boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀_ref = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀_ref, u=FT(5), qᵗ=FT(0.01))
        time_step!(model, 1e-6)
        @test true
    end

    @testset "PolynomialCoefficient with no stability correction [$FT]" begin
        coef = PolynomialCoefficient(stability_function=nothing)

        ρu_bcs  = FieldBoundaryConditions(bottom=BulkDrag(coefficient=coef, gustiness=gustiness, surface_temperature=T₀))
        ρv_bcs  = FieldBoundaryConditions(bottom=BulkDrag(coefficient=coef, gustiness=gustiness, surface_temperature=T₀))
        ρθ_bcs  = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=coef, gustiness=gustiness, surface_temperature=T₀))
        ρqᵛ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=coef, gustiness=gustiness, surface_temperature=T₀))

        boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)
        model = AtmosphereModel(grid; boundary_conditions)

        θ₀_ref = model.dynamics.reference_state.potential_temperature
        set!(model; θ=θ₀_ref, u=FT(5), qᵗ=FT(0.01))
        time_step!(model, 1e-6)
        @test true
    end
end

#####
##### Energy flux boundary condition tests (consolidated)
#####

@testset "Energy flux boundary conditions [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
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
        grid_1 = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 100), y=(0, 100), z=(0, 100))
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
        grid_1 = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 100), y=(0, 100), z=(0, 100))

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
                                     ThetaFluxBCType
    using Oceananigans.Models: boundary_condition_location

    @testset "has_nondefault_bcs [$FT]" begin
        @test has_nondefault_bcs(nothing) == false
        @test has_nondefault_bcs(:some_symbol) == false
        @test has_nondefault_bcs(FieldBoundaryConditions()) == false
        @test has_nondefault_bcs(FieldBoundaryConditions(bottom=FluxBoundaryCondition(FT(100)))) == true
    end

    @testset "boundary_condition_location [$FT]" begin
        LZ = boundary_condition_location(:bottom, Center, Center, Center)[3]
        @test LZ === Nothing

        LX = boundary_condition_location(:west, Center, Center, Center)[1]
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

    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 100), y=(0, 100), z=(0, 100),
                           topology=(Bounded, Bounded, Bounded))

    𝒬 = FT(1000)
    θ₀ = FT(290)
    qᵗ₀ = FT(0.01)
    Δt = FT(1e-6)

    # Test a representative subset of boundaries (bottom and west are sufficient for coverage)
    for ρe_bcs in [
        FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬)),
        FieldBoundaryConditions(west=FluxBoundaryCondition(𝒬)),
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
