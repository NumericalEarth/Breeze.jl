include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.YinYangConservation: MassFixer
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.Units
using Test

function crossing_flow_model(grid, time_discretization)
    model = AtmosphereModel(grid; tracers = :c,
                            dynamics = CompressibleDynamics(time_discretization),
                            coriolis = SphericalCoriolis(),
                            closure = ScalarDiffusivity(HorizontalFormulation(); ν = 1e5, κ = 1e5),
                            momentum_advection = WENO(order=5),
                            scalar_advection = (; ρθ = WENO(order=5), ρqᵛ = WENO(order=5, bounds=(0, 1)), c = WENO(order=5)))

    θᵢ(λ, φ, z) = 300 + 3z / 1000 + 2cosd(φ) * cosd(λ)
    qᵢ(λ, φ, z) = 0.01 * exp(-((λ - 30)^2 + φ^2) / 400 - z / 3000)
    cᵢ(λ, φ, z) = exp(-((λ + 60)^2 + (φ - 20)^2) / 400)
    uᵢ(λ, φ, z) = 20cosd(φ) + 5sind(λ)
    set!(model; θ = θᵢ, qᵗ = qᵢ, c = cᵢ, u = uᵢ, ρ = HydrostaticallyBalancedDensity())

    return model
end

# ρᵈ, ρθ, ρqᵛ and c
densities(model) = values(Base.structdiff(prognostic_fields(model), model.momentum))

# Integrals on a Yin-Yang grid weight every cell by the fraction of it that its panel owns
content(c) = @allowscalar Field(Integral(c))[1, 1, 1]
contents(model) = map(content, densities(model))

# The explicit time step resolves the vertical sound waves
time_step_size(::ExplicitTimeStepping) = 2
time_step_size(::SplitExplicitTimeDiscretization) = 5minutes

function run_steps!(model, Δt, fixer)
    simulation = Simulation(model; Δt, stop_iteration=5, verbose=false)
    fixer && add_conservation_fixer!(simulation)
    run!(simulation)
    return simulation
end

grid = YinYangGrid(default_arch, Float64; size = (48, 16, 4), z = (0, 10kilometers), halo = (4, 4, 3))

@testset "Yin-Yang conservation [$(nameof(typeof(td)))]" for td in (SplitExplicitTimeDiscretization(), ExplicitTimeStepping())
    Δt = time_step_size(td)

    model = crossing_flow_model(grid, td)
    M₀ = contents(model)
    simulation = run_steps!(model, Δt, true)
    @test simulation.callbacks[:conservation_fixer].func isa MassFixer
    @test all(isapprox.(contents(model), M₀; rtol=1e-13))

    # Without the fixer the contents drift by far more than rounding error
    drifting = crossing_flow_model(grid, td)
    run_steps!(drifting, Δt, false)
    @test all(abs.(contents(drifting) .- M₀) .> 1e-10 .* abs.(M₀))
end
