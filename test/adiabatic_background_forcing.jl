include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, mixture_gas_constant,
                             supersaturation, reference_value, MoistureMassFractions, PlanarLiquidSurface
using Oceananigans
using Statistics: mean
using Test

# Exact column test of AdiabaticBackgroundForcing. A frame rising at W holds Nc one-cell columns with the steady
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

function column_model(Nc, W, w′; background = nothing, Δ = 0.01, Nd = 20)
    grid = RectilinearGrid(CPU(); size=(Nc, 1, 1), halo=(1, 1, 1), x=(0, Nc * Δ), y=(0, Δ), z=(0, Δ),
                           topology=(Periodic, Periodic, Periodic))

    # The frame rises hydrostatically under gravity; buoyancy is off so w′ stays steady
    dynamics = BoussinesqDynamics(grid; pressure=p₀_column, temperature=T₀_column, frame_vertical_velocity=W)
    constants = ThermodynamicConstants(gravitational_acceleration=0)

    n = 1:Nd
    droplets = (x = repeat([(i - 1/2) * Δ for i in 1:Nc], inner=Nd), y = fill(Δ / 2, Nd * Nc),
                z = repeat(Δ .* (n .- 1/2) ./ Nd, Nc), r = fill(5e-6, Nd * Nc), rᵈ = fill(5e-8, Nd * Nc),
                κ = fill(0.6, Nd * Nc))
    microphysics = LagrangianMicrophysics(; droplets)

    forcing = isnothing(background) ? NamedTuple() :
              (; θ = AdiabaticBackgroundForcing(; background, gravitational_acceleration=9.81),
                 qᵛ = AdiabaticBackgroundForcing(; background, gravitational_acceleration=9.81))

    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants=constants, closure=nothing,
                            advection=Centered(order=2), forcing)

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

@testset "AdiabaticBackgroundForcing" begin
    background = AdiabaticBackgroundForcing()
    @test occursin("MoistAdiabaticBackground", summary(background))
    @test_throws ArgumentError AdiabaticBackgroundForcing(; background=:dry)

    grid = RectilinearGrid(CPU(); size=(2, 2, 2), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
    @test_throws ArgumentError AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid), forcing=(; u=background))

    # Moist neutrality and the adiabatic supersaturation source
    model = column_model(2, W_column, [0.5, -0.5]; background=:moist)
    time_step!(model, 1e-3)
    gradients = model.forcing.ρθ.forcing.gradients
    @test gradients.vapor_mass_fraction < 0
    @test gradients.potential_temperature > 0
    @test gradients.supersaturation_source ≈ gradients.A₁ rtol=1e-6
    @test 4e-4 < gradients.A₁ < 7e-4 # m⁻¹, near 281 K and 900 hPa

    Nc = 4
    w′ = [sin(2π * (i - 1/2) / Nc) for i in 1:Nc]

    frame = run_columns!(column_model(Nc, W_column, w′))
    moist = run_columns!(column_model(Nc, W_column, w′; background=:moist))
    isothermal = run_columns!(column_model(Nc, W_column, w′; background=:isothermal))
    truth = [run_columns!(column_model(1, W_column + w′[i], [0.0]))[1] for i in 1:Nc]

    # Without the forcing every column equals the frame
    @test all(frame .≈ frame[1])

    effect = abs.(truth .- frame)
    @test all(effect .> 1e-5)
    @test all(abs.(moist .- truth) .< 0.01 .* effect)
    @test all(abs.(isothermal .- truth) .> 0.1 .* effect)
end
