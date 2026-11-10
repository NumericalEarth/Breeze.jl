include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: materialize_closure, closure_scalar_names
using Breeze.TurbulenceClosures.SHOC: SHOC, with_parameters
using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Test

# A convective boundary layer with moisture and wind: a quick exercise of every part of SHOC
function shoc_boundary_layer_model(grid; closure = SimplifiedHigherOrderClosure(eltype(grid);
                                                                               surface_temperature_flux = 0.1,
                                                                               surface_moisture_flux = 5e-5,
                                                                               friction_velocity = 0.3))
    FT = eltype(grid)
    θᵢ(z) = 300 + 3e-3 * z
    qᵢ(z) = 0.016 * exp(-z / 2000)
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=θᵢ)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, closure)
    set!(model, θ = (args...) -> θᵢ(last(args)), qᵗ = (args...) -> qᵢ(last(args)), u = 5)
    set!(model.tracers.ρe, reference_state.density * FT(0.1))
    return model
end

column_data(field) = Array(interior(field))

@testset "SimplifiedHigherOrderClosure [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 24

    @testset "Parameters" begin
        p = SHOCParameters(FT)
        @test p isa SHOCParameters{FT}
        @test isbits(p)
        @test p.eps ≈ p.rv / p.rgas - 1
        q = with_parameters(p; ℂᶜ = FT(0.2), ℂˡ₁ = FT(0.4))
        @test q isa SHOCParameters{FT}
        @test q.ℂᶜ == FT(0.2) && q.ℂˡ₁ == FT(0.4) && q.ℂᵘ == p.ℂᵘ
        @test_throws ArgumentError with_parameters(p; Ckh = FT(0.2))

        closure = SimplifiedHigherOrderClosure(FT; parameters = q, surface_temperature_flux = 0.1)
        @test isbits(closure)
        @test closure.parameters.ℂᶜ == FT(0.2)
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

    @testset "Rebuilding and materializing the closure" begin
        grid = RectilinearGrid(default_arch, FT; size=Nz, z=(0, 3000), topology=(Flat, Flat, Bounded))
        model = shoc_boundary_layer_model(grid)
        parameters = with_parameters(model.closure.parameters; ℂᶜ = FT(0.05), ℂᵘ = FT(0.05))
        closure = SimplifiedHigherOrderClosure(FT; parameters, surface_temperature_flux = 0.1)
        materialized = materialize_closure(closure, closure_scalar_names(model), grid.architecture)
        @test materialized.parameters.ℂᶜ == FT(0.05)
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
