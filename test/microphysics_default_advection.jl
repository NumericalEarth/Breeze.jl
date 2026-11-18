include(joinpath(@__DIR__, "setup.jl"))

using Test
using Breeze
using Breeze.AtmosphereModels: default_microphysics_advection, non_negative_microphysics_advection
using Oceananigans.Advection: WENO, Centered
using Oceananigans.Utils: AdaptiveVerticallyImplicitDiscretization

is_bounded_weno(scheme) = scheme isa WENO && !isnothing(scheme.bounds) && iszero(scheme.bounds.minimum_value)
is_adaptive_implicit(scheme) = scheme.time_discretization isa AdaptiveVerticallyImplicitDiscretization

@testset "Default advection of microphysical fields [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    column = RectilinearGrid(default_arch, FT; size = 50, z = (0, 4000), topology = (Flat, Flat, Bounded))

    @testset "P3 fields default to bounded WENO with adaptive implicit vertical advection" begin
        model = AtmosphereModel(column; microphysics = P3Microphysics(FT))
        for name in (:ρqᵛ, :ρqᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)
            scheme = model.advection[name]
            @test is_bounded_weno(scheme)
            @test is_adaptive_implicit(scheme)
            @test scheme.time_discretization.cfl == FT(1/4)
        end
        # Non-microphysical scalars keep the model-wide default
        @test model.advection.ρθ isa Centered
        @test model.advection.momentum isa Centered
    end

    @testset "Saturation adjustment moisture and dry models" begin
        grid = RectilinearGrid(default_arch, FT; size = (4, 4, 4), extent = (1, 1, 1))  # halo 3: order 3
        model = AtmosphereModel(grid; microphysics = SaturationAdjustment(FT))
        @test is_bounded_weno(model.advection.ρqᵉ)
        @test model.advection.ρqᵉ isa WENO{2}  # order 3
        @test default_microphysics_advection(nothing, grid) == NamedTuple()
        dry = AtmosphereModel(grid)
        @test all(scheme -> scheme isa Centered, values(dry.advection))

        tall_halo = RectilinearGrid(default_arch, FT; size = (4, 4, 4), halo = (4, 4, 4), extent = (1, 1, 1))
        @test non_negative_microphysics_advection(tall_halo) isa WENO{3}  # order 5
        small_halo = RectilinearGrid(default_arch, FT; size = (4, 4, 4), halo = (1, 1, 1), extent = (1, 1, 1))
        @test isnothing(non_negative_microphysics_advection(small_halo))
    end

    @testset "User-specified schemes override the defaults" begin
        p3 = P3Microphysics(FT)
        model = AtmosphereModel(column; microphysics = p3, scalar_advection = (; ρqʳ = Centered(order = 2)))
        @test model.advection.ρqʳ isa Centered
        @test is_bounded_weno(model.advection.ρnʳ)

        for kw in ((; advection = Centered(order = 2)), (; scalar_advection = Centered(order = 2)))
            model = AtmosphereModel(column; microphysics = p3, kw...)
            @test all(name -> model.advection[name] isa Centered, (:ρqᵛ, :ρqʳ, :ρnʳ, :ρqⁱ))
        end
    end

    @testset "A rain column stays non-negative and finite at large Δt without a user scheme" begin
        # Rain falls at several m/s: many 80 m cells per 120 s step.
        model = AtmosphereModel(column; microphysics = P3Microphysics(FT))
        set!(model; θ = 290, qᵗ = 1e-2,
             ρqʳ = z -> ifelse(z > 3000, 3e-3, 0), ρnʳ = z -> ifelse(z > 3000, 3e4, 0))
        for _ in 1:5
            time_step!(model, 120)
        end
        for name in (:ρqʳ, :ρnʳ, :ρqᶜˡ, :ρqⁱ)
            c = Array(interior(model.microphysical_fields[name]))
            @test all(isfinite, c)
            @test minimum(c) ≥ 0
        end
    end
end
