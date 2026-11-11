include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: materialize_closure, closure_scalar_names
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: SHOCParameters, trainable_parameter_names
using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Test

# A convective boundary layer with moisture and wind: a quick exercise of every part of SHOC
function shoc_boundary_layer_model(grid; closure = SimplifiedHigherOrderClosure(eltype(grid);
                                                                               surface_temperature_flux = 0.1,
                                                                               surface_moisture_flux = 5e-5,
                                                                               friction_velocity = 0.3),
                                   boundary_conditions = NamedTuple())
    FT = eltype(grid)
    θᵢ(z) = 300 + 3e-3 * z
    qᵢ(z) = 0.016 * exp(-z / 2000)
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=θᵢ)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, closure, boundary_conditions)
    set!(model, θ = (args...) -> θᵢ(last(args)), qᵗ = (args...) -> qᵢ(last(args)), u = 5)
    set!(model.tracers.ρe, reference_state.density * FT(0.1))
    return model
end

column_data(field) = Array(interior(field))

@testset "SimplifiedHigherOrderClosure [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 24

    @testset "Components and parameters" begin
        closure = SimplifiedHigherOrderClosure(FT; surface_temperature_flux = 0.1)
        @test isbits(closure)
        @test SHOC === SimplifiedHigherOrderClosure
        @test SimplifiedHigherOrderClosure(; surface_temperature_flux = 0.1) === closure
        @test_throws MethodError SimplifiedHigherOrderClosure(FT; core = nothing)
        @test closure.mixing_length isa BlendedMixingLength{FT}
        @test closure.turbulence_time_scale isa ColumnStabilityDamping{FT}
        @test closure.tke_update isa ForwardEulerTKE{FT}
        @test closure.buoyancy_source isa PDFBuoyancyFlux
        @test closure.second_moments isa DowngradientSecondMoments{FT}
        @test closure.third_moment isa CanutoThirdMoment{ResetBound{FT}, FT}
        @test closure.assumed_pdf isa DoubleGaussianPDF{FT}
        @test closure.clipping isa HardClipping
        @test closure.constants.eps ≈ closure.constants.rv / closure.constants.rgas - 1

        # The flat view of the coefficients, and the snapshot of the numbers of shoc.F90
        ℂ = closure_parameters(closure)
        @test length(ℂ) == 49
        @test all(value -> value isa FT, values(ℂ))
        @test ℂ.ℂᶜ == FT(0.1) && ℂ.ℂˡ₂ == FT(2.8284) && ℂ.ℂʷ₂ == FT(7) && ℂ.ℂᴴ₃ == FT(8.5)
        @test length(trainable_parameter_names(closure)) == 40
        @test :ℂˡ₅ ∉ trainable_parameter_names(closure)
        @test :ℂᶠ₅ ∉ trainable_parameter_names(closure)

        # The literals of shoc.F90 are fields with the values of shoc.F90
        @test ℂ.ℂᴰ₃ == FT(0.19) && ℂ.ℂᴰ₄ == FT(0.51) && ℂ.ℂᴰ₅ == FT(0.7)
        @test ℂ.ℂᶠ₂ == FT(0.4) && ℂ.ℂᶠ₃ == FT(0.2) && ℂ.ℂᶠ₄ == FT(0.3) && ℂ.ℂᶠ₅ == 1
        @test ℂ.ℂʷ₃ == FT(0.52) && ℂ.ℂʷ₄ == FT(0.87) && ℂ.ℂʷ₅ == FT(0.5) && ℂ.ℂʷ₆ == FT(0.6) && ℂ.ℂʷ₇ == FT(2.4)
        @test ℂ.ℂʷ₈ == FT(0.6) && ℂ.ℂʷ₉ == FT(1.2) && ℂ.ℂʷ₁₀ == FT(1.5) && ℂ.ℂʷ₁₁ == FT(1.25)
        @test ℂ.ℂᴴ₅ == FT(0.1) && ℂ.ℂᴴ₆ == 700 && ℂ.ℂᴴ₇ == 50
        @test closure.turbulence_time_scale.temp_crit == 182 && closure.turbulence_time_scale.pbl_trans == 200
        @test closure.third_moment.bound.w3_reset == FT(0.02)
        @test closure.constants.wsmin == 1 && closure.constants.ksrfmin == FT(1e-4)
        @test closure.assumed_pdf.plume_fraction_min == FT(0.01) && closure.assumed_pdf.plume_fraction_max == FT(0.99)
        @test with_parameters(closure; w3_reset = FT(0.01)).third_moment.bound.w3_reset == FT(0.01)
        @test with_parameters(closure; ℂʷ₁₁ = FT(1)).third_moment.ℂʷ₁₁ == 1
        p = SHOCParameters(closure)
        @test p isa SHOCParameters{FT}
        @test p.ℂᶜ == ℂ.ℂᶜ && p.mintke == closure.constants.mintke && p.w3clip == closure.third_moment.w3clip

        # Coefficients are replaced through the closure, whichever component holds them
        changed = with_parameters(closure; ℂᶜ = FT(0.2), ℂˡ₁ = FT(0.4), ℂʷ₂ = FT(6), mintke = FT(1e-4))
        @test typeof(changed) === typeof(closure)
        @test changed.turbulence_time_scale.ℂᶜ == FT(0.2)
        @test changed.mixing_length.ℂˡ₁ == FT(0.4)
        @test changed.third_moment.ℂʷ₂ == FT(6)
        @test changed.constants.mintke == FT(1e-4)
        @test changed.turbulence_time_scale.ℂᵘ == closure.turbulence_time_scale.ℂᵘ
        @test changed.surface_temperature_flux == closure.surface_temperature_flux
        @test_throws ArgumentError with_parameters(closure; Ckh = FT(0.2))

        # A replacement of another type promotes every component
        promoted = with_parameters(SimplifiedHigherOrderClosure(Float32); ℂᶜ = 0.2)
        @test all(value -> value isa Float64, values(closure_parameters(promoted)))

        # Coefficients may also be given with the component
        closure = SimplifiedHigherOrderClosure(FT; turbulence_time_scale = ColumnStabilityDamping(ℂᶜ = 0.2))
        @test closure.turbulence_time_scale.ℂᶜ == FT(0.2)
        @test_throws ArgumentError ColumnStabilityDamping(Ckh = 0.2)

        # A component is refused in a slot that is not its own
        @test_throws ArgumentError SimplifiedHigherOrderClosure(FT; second_moments = ForwardEulerTKE())
    end

    @testset "Single column" begin
        grid = RectilinearGrid(default_arch, FT; size=Nz, z=(0, 3000), topology=(Flat, Flat, Bounded))
        model = shoc_boundary_layer_model(grid)
        @test :ρe ∈ keys(model.tracers)

        ρ = column_data(model.dynamics.reference_state.density)
        Δz = 3000 / Nz
        water₀ = sum(column_data(model.moisture_density)) * Δz
        Δt = 60

        for _ in 1:10
            time_step!(model, Δt)
        end

        d = model.closure_fields.diagnostics
        e = column_data(model.tracers.ρe) ./ ρ
        @test all(isfinite, column_data(model.formulation.potential_temperature_density))
        @test all(isfinite, column_data(model.moisture_density))
        @test all(e .≥ FT(0.0004) * (1 - 10eps(FT)))
        @test maximum(e) > FT(0.01)
        @test all(x -> 0 ≤ x ≤ 1, column_data(d.shoc_cldfrac))
        @test all(≥(0), column_data(d.shoc_ql))
        @test all(>(0), column_data(d.tkh))
        @test all(isfinite, column_data(d.w3))
        @test @allowscalar(d.pblh[1, 1, 1]) > 0

        # The surface moisture flux is the only source of water: ∫ρqᵗ dz grows by ρˢ w'q' Δt per step
        water₁ = sum(column_data(model.moisture_density)) * Δz
        ρˢ = @allowscalar Breeze.Thermodynamics.surface_density(model.dynamics.reference_state)
        expected = ρˢ * 5e-5 * 10 * Δt
        @test isapprox(water₁ - water₀, expected; rtol = FT == Float64 ? 2e-2 : 5e-2)

        # The drag decelerates the wind of the lowest level
        @test @allowscalar(model.momentum.ρu[1, 1, 1] / ρ[1, 1, 1]) < 5
    end

    @testset "Surface fluxes from the model's bottom boundary conditions" begin
        grid = RectilinearGrid(default_arch, FT; size=Nz, z=(0, 3000), topology=(Flat, Flat, Bounded))
        Δz = 3000 / Nz
        Δt = 60
        Nt = 10
        𝒬ᵀ = 120    # sensible heat flux, W m⁻²
        Jᵗ = 6e-5   # moisture flux, kg m⁻² s⁻¹
        𝒯ˣ = -0.1   # x-momentum flux, kg m⁻¹ s⁻²

        flux(J; kw...) = FieldBoundaryConditions(bottom = FluxBoundaryCondition(J; kw...))
        boundary_conditions = (ρE = flux(𝒬ᵀ), ρqᵗ = flux(Jᵗ), ρu = flux(𝒯ˣ))
        model = shoc_boundary_layer_model(grid; closure = SimplifiedHigherOrderClosure(FT), boundary_conditions)

        ρθ = model.formulation.potential_temperature_density
        water₀ = sum(column_data(model.moisture_density)) * Δz
        heat₀ = sum(column_data(ρθ)) * Δz
        for _ in 1:Nt
            time_step!(model, Δt)
        end
        water₁ = sum(column_data(model.moisture_density)) * Δz
        heat₁ = sum(column_data(ρθ)) * Δz

        # The boundary-condition flux enters the column exactly once: SHOC applies it and the
        # Runge–Kutta stages do not.
        @test isapprox(water₁ - water₀, Jᵗ * Nt * Δt; rtol = FT == Float64 ? 1e-9 : 1e-3)

        # The energy flux arrives as the potential temperature flux 𝒬ᵀ / (cᵖ Π)
        cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
        @test isapprox(heat₁ - heat₀, 𝒬ᵀ / cᵖ * Nt * Δt; rtol = 5e-2)

        # SHOC used the kinematic fluxes: the density-weighted ones over the surface density
        surface = model.closure_fields.surface
        ρ₁ = @allowscalar model.dynamics.reference_state.density[1, 1, 1]
        @test isapprox(@allowscalar(surface.wqw_sfc[1, 1, 1]), Jᵗ / ρ₁; rtol = 5e-2)
        @test isapprox(@allowscalar(surface.uw_sfc[1, 1, 1]), 𝒯ˣ / ρ₁; rtol = 5e-2)
        @test @allowscalar(surface.wthl_sfc[1, 1, 1]) > 0

        # Boundary-condition fluxes and the closure-owned kinematic fluxes are the same thing
        by_condition = shoc_boundary_layer_model(grid; closure = SimplifiedHigherOrderClosure(FT),
                                                 boundary_conditions = (ρqᵗ = flux(Jᵗ), ρu = flux(𝒯ˣ)))
        time_step!(by_condition, Δt)
        kinematic = by_condition.closure_fields.surface
        surface_moisture_flux = @allowscalar kinematic.wqw_sfc[1, 1, 1]
        surface_x_momentum_flux = @allowscalar kinematic.uw_sfc[1, 1, 1]
        by_closure = shoc_boundary_layer_model(grid; closure = SimplifiedHigherOrderClosure(FT; surface_moisture_flux,
                                                                                            surface_x_momentum_flux))
        time_step!(by_closure, Δt)
        for _ in 2:Nt
            time_step!(by_condition, Δt)
            time_step!(by_closure, Δt)
        end
        @test column_data(by_condition.moisture_density) == column_data(by_closure.moisture_density)
        @test column_data(by_condition.momentum.ρu) == column_data(by_closure.momentum.ρu)
        @test column_data(by_condition.tracers.ρe) == column_data(by_closure.tracers.ρe)

        # The same fluxes as a discrete-form function and as a field give the same answer
        discrete_flux(i, j, grid, clock, fields, J) = J
        surface_flux_field = Field{Center, Center, Nothing}(grid)
        set!(surface_flux_field, Jᵗ)
        other_conditions = (ρE = flux(discrete_flux; discrete_form=true, parameters=FT(𝒬ᵀ)),
                            ρqᵗ = flux(surface_flux_field), ρu = flux(𝒯ˣ))

        # A wall flux of turbulent kinetic energy is replaced by SHOC's own surface source
        wall_conditions = merge(boundary_conditions, (; ρe = flux(10)))

        for conditions in (other_conditions, wall_conditions)
            other = shoc_boundary_layer_model(grid; closure = SimplifiedHigherOrderClosure(FT),
                                              boundary_conditions = conditions)
            for _ in 1:Nt
                time_step!(other, Δt)
            end
            @test column_data(other.moisture_density) == column_data(model.moisture_density)
            @test column_data(other.formulation.potential_temperature_density) == column_data(ρθ)
            @test column_data(other.tracers.ρe) == column_data(model.tracers.ρe)
        end
    end

    @testset "Rebuilding and materializing the closure" begin
        grid = RectilinearGrid(default_arch, FT; size=Nz, z=(0, 3000), topology=(Flat, Flat, Bounded))
        model = shoc_boundary_layer_model(grid)
        closure = with_parameters(model.closure; ℂᶜ = FT(0.05), ℂᵘ = FT(0.05))
        materialized = materialize_closure(closure, closure_scalar_names(model), grid.architecture)
        @test materialized.turbulence_time_scale.ℂᶜ == FT(0.05)
        @test materialized.surface_temperature_flux == model.closure.surface_temperature_flux

        # The rebuilt closure fits the closure fields of the model
        model.closure = materialized
        time_step!(model, 60)
        @test all(isfinite, column_data(model.tracers.ρe))
        @test_throws ArgumentError materialize_closure(closure, (:ρθ, :ρqᵛ), grid.architecture)
    end

    @testset "Column forest and three-dimensional grid" begin
        column_grid = RectilinearGrid(default_arch, FT; size=Nz, z=(0, 3000), topology=(Flat, Flat, Bounded))
        forest_grid = RectilinearGrid(default_arch, FT; size=ColumnEnsembleSize(Nz=Nz, ensemble=(3, 2), Hz=3),
                                      z=(0, 3000), topology=(Flat, Flat, Bounded))
        box_grid = RectilinearGrid(default_arch, FT; size=(4, 4, Nz), x=(0, 4e5), y=(0, 4e5), z=(0, 3000),
                                   topology=(Periodic, Periodic, Bounded))

        models = map(shoc_boundary_layer_model, (column_grid, forest_grid, box_grid))
        for model in models, _ in 1:4
            time_step!(model, 60)
        end

        column, forest, box = models
        θ_column = column_data(column.formulation.potential_temperature_density)
        θ_forest = column_data(forest.formulation.potential_temperature_density)
        θ_box = column_data(box.formulation.potential_temperature_density)
        e_column = column_data(column.tracers.ρe)
        e_forest = column_data(forest.tracers.ρe)
        e_box = column_data(box.tracers.ρe)

        # Every column of the forest is the single column
        @test all(θ_forest .== θ_column)
        @test all(e_forest .== e_column)
        @test all(column_data(forest.closure_fields.diagnostics.shoc_mix) .==
                  column_data(column.closure_fields.diagnostics.shoc_mix))

        # A horizontally uniform three-dimensional model stays uniform and follows the column
        @test all(isfinite, θ_box)
        @test maximum(abs, θ_box .- θ_box[1:1, 1:1, :]) ≤ 100eps(FT) * maximum(abs, θ_box)
        @test isapprox(θ_box[1:1, 1:1, :], θ_column; rtol = sqrt(eps(FT)))
        @test isapprox(e_box[1:1, 1:1, :], e_column; rtol = FT == Float64 ? 1e-6 : 1e-2)
    end

    @testset "Formulation check" begin
        grid = RectilinearGrid(default_arch, FT; size=Nz, z=(0, 3000), topology=(Flat, Flat, Bounded))
        model = AtmosphereModel(grid; closure=SimplifiedHigherOrderClosure(FT), formulation=:StaticEnergy)
        @test_throws ArgumentError time_step!(model, 1)
    end
end
