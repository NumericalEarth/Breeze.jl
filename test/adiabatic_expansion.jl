include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, mixture_gas_constant,
                             supersaturation, reference_value, MoistureMassFractions, PlanarLiquidSurface
using Oceananigans
using Statistics: mean
using Test

# Exact column test of AdiabaticExpansion. A frame rising at W holds Nc one-cell columns with the steady
# flow w′ = w₁ sin(2πx/Lx), u = v = 0, no closure and no buoyancy, so the columns exchange nothing and
# each column is an adiabatic parcel rising at W + w′. Its truth is a one-cell box whose frame rises at W + w′.

const p₀_column = 90000
const T₀_column = 281
const W_column = 2

function column_vapor_mass_fraction(constants, saturation_ratio)
    pᵛ = saturation_ratio * saturation_vapor_pressure(T₀_column, constants, PlanarLiquidSurface())
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    return ε * pᵛ / (p₀_column - (1 - ε) * pᵛ)
end

function column_model(Nc, W, w′; adiabatic_expansion = nothing, Δ = 0.01, Nd = 20)
    grid = RectilinearGrid(CPU(); size=(Nc, 1, 1), halo=(1, 1, 1), x=(0, Nc * Δ), y=(0, Δ), z=(0, Δ),
                           topology=(Periodic, Periodic, Periodic))

    # The frame rises hydrostatically under gravity; buoyancy is off so w′ stays steady
    dynamics = BoussinesqDynamics(grid; pressure=p₀_column, temperature=T₀_column, frame_vertical_velocity=W,
                                  adiabatic_expansion)
    constants = ThermodynamicConstants(gravitational_acceleration=0)

    n = 1:Nd
    droplets = (x = repeat([(i - 1/2) * Δ for i in 1:Nc], inner=Nd), y = fill(Δ / 2, Nd * Nc),
                z = repeat(Δ .* (n .- 1/2) ./ Nd, Nc), r = fill(5e-6, Nd * Nc), rᵈ = fill(5e-8, Nd * Nc),
                κ = fill(0.6, Nd * Nc))
    microphysics = LagrangianMicrophysics(; droplets)

    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants=constants, closure=nothing,
                            advection=Centered(order=2))

    qᵛ₀ = column_vapor_mass_fraction(constants, 1.002)
    set!(model; T=T₀_column, qᵗ=qᵛ₀, w=reshape(w′, Nc, 1, 1))
    return model
end

function column_supersaturation(model)
    p = reference_value(model.dynamics.reference_state.pressure)
    constants = model.thermodynamic_constants
    T = Array(interior(model.temperature))[:, 1, 1]
    qᵛ = Array(interior(model.microphysical_fields.qᵛ))[:, 1, 1]
    return map(T, qᵛ) do T, qᵛ
        q = MoistureMassFractions(qᵛ)
        ρ = p / (mixture_gas_constant(q, constants) * T)
        supersaturation(T, ρ, q, constants, PlanarLiquidSurface())
    end
end

function run_columns!(model; Δt = 1e-2, Nt = 100)
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    return column_supersaturation(model)
end

contains_expansion(forcing) = forcing isa AdiabaticExpansion
contains_expansion(forcing::SpecificForcing) = contains_expansion(forcing.forcing)
contains_expansion(forcing::Oceananigans.Forcings.MultipleForcings) = any(contains_expansion, forcing.forcings)

@testset "AdiabaticExpansion" begin
    @test summary(AdiabaticExpansion()) == "AdiabaticExpansion"

    periodic = RectilinearGrid(CPU(); size=(2, 2, 2), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
    bounded = RectilinearGrid(CPU(); size=(2, 2, 2), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))

    @testset "Default resolution" begin
        # A moving frame, rising or fixed, carries the expansion whatever the vertical topology
        for grid in (periodic, bounded), W in (2, 0)
            dynamics = BoussinesqDynamics(grid; frame_vertical_velocity=W)
            @test dynamics.adiabatic_expansion isa AdiabaticExpansion
            @test occursin("AdiabaticExpansion", summary(dynamics))
            model = AtmosphereModel(grid; dynamics)
            @test model.dynamics.adiabatic_expansion.gradients isa Breeze.AnelasticEquations.ExpansionGradients
            @test contains_expansion(model.forcing.ρθ)
            @test contains_expansion(model.forcing.ρqᵛ)
            time_step!(model, 1e-3)
        end

        # No frame, or explicitly disabled: no expansion
        @test isnothing(BoussinesqDynamics(periodic).adiabatic_expansion)
        disabled = BoussinesqDynamics(periodic; frame_vertical_velocity=2, adiabatic_expansion=nothing)
        @test isnothing(disabled.adiabatic_expansion)
        @test !occursin("AdiabaticExpansion", summary(disabled))
        @test AtmosphereModel(periodic; dynamics=disabled).forcing.ρθ isa Returns

        # The expansion needs a frame, and is not a user forcing
        @test_throws ArgumentError BoussinesqDynamics(periodic; adiabatic_expansion=AdiabaticExpansion())
        @test_throws ArgumentError AtmosphereModel(periodic; dynamics=BoussinesqDynamics(periodic),
                                                   forcing=(; θ=AdiabaticExpansion()))

        # The expansion composes with a user forcing on the same variable
        dynamics = BoussinesqDynamics(periodic; frame_vertical_velocity=2)
        model = AtmosphereModel(periodic; dynamics, forcing=(; θ=(x, y, z, t) -> 0))
        @test model.forcing.ρθ isa Oceananigans.Forcings.MultipleForcings
        @test contains_expansion(model.forcing.ρθ)
        time_step!(model, 1e-3)
    end

    # Moist neutrality and the adiabatic supersaturation source
    model = column_model(2, W_column, [0.5, -0.5]; adiabatic_expansion=AdiabaticExpansion())
    time_step!(model, 1e-3)
    gradients = model.dynamics.adiabatic_expansion.gradients
    @test gradients.vapor_mass_fraction < 0
    @test gradients.potential_temperature > 0
    @test gradients.supersaturation_source ≈ gradients.A₁ rtol=1e-6
    @test 4e-4 < gradients.A₁ < 7e-4 # m⁻¹, near 281 K and 900 hPa
    @test startswith(summary(model.dynamics.adiabatic_expansion), "AdiabaticExpansion(Gᶿ=")

    Nc = 4
    w′ = [sin(2π * (i - 1/2) / Nc) for i in 1:Nc]

    frame = run_columns!(column_model(Nc, W_column, w′))
    expansion = run_columns!(column_model(Nc, W_column, w′; adiabatic_expansion=AdiabaticExpansion()))
    truth = [run_columns!(column_model(1, W_column + w′[i], [0.0]))[1] for i in 1:Nc]

    # Without the expansion every column equals the frame
    @test all(frame .≈ frame[1])

    effect = abs.(truth .- frame)
    @test all(effect .> 1e-5)
    @test all(abs.(expansion .- truth) .< 0.01 .* effect)
end
