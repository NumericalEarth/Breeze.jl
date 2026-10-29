include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.RoundingTreatments: rounded_add, rounding_key
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: time_step!
using Statistics: mean
using Test

#####
##### Scalar emulation of the SSP RK3 stage updates (issue #1122)
#####
##### du/dt = G with u₀ = 300, Δt = 2.5 s, over 10 days. In Float32 half an ulp of 300 is
##### 1.5e-5, and a heating of 1.5 K/day adds at most 4.3e-5 per stage, so round-to-nearest
##### loses almost all of it; stochastic rounding is exact in expectation.
#####

function emulate_ssp_rk3(rounding, ::Type{FT}, G, Δt, Nsteps; u₀ = 300) where FT
    α = (FT(1), FT(1//4), FT(2//3))
    G = FT(G)
    Δt = FT(Δt)
    u = FT(u₀)
    for n in 0:Nsteps-1
        u⁰ = u
        for stage in 1:3
            key = rounding_key(rounding, n, stage, 1)
            u = ssp_stage(rounding, u, u⁰, G, Δt, α[stage], key)
        end
    end
    return u - FT(u₀)
end

# The same arithmetic as the kernels `_ssp_rk3_substep!` and `_rounded_ssp_rk3_substep!`
ssp_stage(::RoundToNearest, u, u⁰, G, Δt, α, key) = (1 - α) * u⁰ + α * (u + Δt * G)
ssp_stage(rounding, u, u⁰, G, Δt, α, key) = rounded_add(rounding, u⁰, α * ((u - u⁰) + Δt * G), 1, 1, 1, key)

@testset "Scalar emulation: sub-ulp increments" begin
    day = 86400
    Δt = 2.5
    Nsteps = round(Int, 10day / Δt)
    gap = eps(300f0)

    # A random walk of about one gap per step: σ ≲ gap √N. Allow 3σ.
    tolerance = 3 * gap * sqrt(Nsteps)

    for (heating, exact) in ((1.5, 15.0), (0.3, 3.0), (-2.0, -20.0))
        G = heating / day

        # Float64 resolves the increments
        @test emulate_ssp_rk3(RoundToNearest(), Float64, G, Δt, Nsteps) ≈ exact rtol=1e-6

        # Round to nearest loses (or, for cooling, distorts) the increments in Float32...
        rtn = emulate_ssp_rk3(RoundToNearest(), Float32, G, Δt, Nsteps)
        @test abs(rtn - exact) > 0.5

        # ... and stochastic rounding recovers them within the random walk.
        stochastic = emulate_ssp_rk3(StochasticRounding(seed = 1), Float32, G, Δt, Nsteps)
        @test abs(stochastic - exact) < tolerance
        @info "Scalar emulation, $(heating) K/day over 10 days" exact rtn stochastic tolerance
    end

    # In the 1.5 K/day case of #1122, round to nearest keeps essentially nothing
    @test abs(emulate_ssp_rk3(RoundToNearest(), Float32, 1.5 / day, Δt, Nsteps)) < 0.01
end

@testset "rounded_add" begin
    for FT in (Float32, Float64)
        a = FT(300)
        rounding = StochasticRounding(seed = 7)
        key = rounding_key(rounding, 3, 2, 1)

        # Exactly representable sums are not perturbed, whatever the random number
        @test all(rounded_add(rounding, a, FT(1), i, 1, 1, key) == a + 1 for i in 1:1000)
        @test all(rounded_add(rounding, -a, FT(0), i, 1, 1, key) == -a for i in 1:1000)
        @test rounded_add(rounding, zero(FT), zero(FT), 1, 1, 1, key) == 0

        # A sub-ulp increment rounds to one of the two neighbors, with the right mean
        δ = eps(a) / 4
        sums = [rounded_add(rounding, a, δ, i, j, 1, key) for i in 1:200, j in 1:200]
        @test all(s -> s == a || s == nextfloat(a), sums)
        @test mean(Float64.(sums) .- Float64(a)) ≈ Float64(δ) rtol=0.05
        sums = [rounded_add(rounding, a, -δ, i, j, 1, key) for i in 1:200, j in 1:200]
        @test all(s -> s == a || s == prevfloat(a), sums)
        @test mean(Float64.(sums) .- Float64(a)) ≈ -Float64(δ) rtol=0.05

        # TwoSum also handles increments larger than the state (momentum crossing zero)
        sums = [rounded_add(rounding, FT(1e-3), FT(-1) + eps(FT(1)) / 3, i, 1, 1, key) for i in 1:10000]
        exact = big(FT(1e-3)) + big(FT(-1) + eps(FT(1)) / 3)
        @test all(s -> abs(big(s) - exact) < eps(FT(1)), sums)

        # Round to nearest is plain addition
        @test rounded_add(RoundToNearest(), a, δ, 1, 1, 1, UInt32(0)) === a + δ

        # Keys differ by iteration, stage, field and seed
        keys = [rounding_key(StochasticRounding(seed = s), n, m, f) for s in 0:1, n in 0:3, m in 1:3, f in 0:4]
        @test allunique(keys)
        @test rounding_key(RoundToNearest(), 5, 2, 1) === UInt32(0)
    end

    @test StochasticRounding().seed === UInt32(0)
    @test StochasticRounding(seed = -1).seed === typemax(UInt32)
    @test sprint(show, StochasticRounding(seed = 3)) == "StochasticRounding(seed=3)"
    @test sprint(show, RoundToNearest()) == "RoundToNearest()"
end

#####
##### Small models
#####

θᵢ(x, y, z) = 300 + z / 1000 + sin(2π * x / 1000) * cos(2π * y / 1000) * exp(-z / 500) / 10
uᵢ(x, y, z) = cos(2π * x / 1000) * sin(π * z / 1000)

function rounding_test_model(FT, dynamics_kind; rounding = nothing, forcing = NamedTuple())
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 6), halo=(3, 3, 3),
                           x=(0, 1000), y=(0, 1000), z=(0, 1000))

    timestepper_kwargs = isnothing(rounding) ? NamedTuple() : (; rounding)

    if dynamics_kind === :anelastic
        model = AtmosphereModel(grid; advection=WENO(order=5), tracers=:c, forcing, timestepper_kwargs)
        set!(model; θ=θᵢ, u=uᵢ, qᵗ=1e-3, c=1)
    else
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6);
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid; advection=WENO(order=5), dynamics, tracers=:c, forcing, timestepper_kwargs)
        set!(model; ρ=model.dynamics.reference_state.density, θ=θᵢ, u=uᵢ, qᵗ=1e-3, c=1)
    end

    return model
end

snapshot(model) = map(field -> Array(parent(field)), prognostic_fields(model))

function run_and_snapshot(FT, dynamics_kind; Nsteps = 4, Δt = 1, kw...)
    model = rounding_test_model(FT, dynamics_kind; kw...)
    for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return snapshot(model)
end

bitwise_equal(a, b) = all(map((x, y) -> isequal(x, y), a, b))

# The model tests below set `Oceananigans.defaults.FloatType`; restore it afterwards so that
# tests sharing this worker process keep the default precision.
default_FT = Oceananigans.defaults.FloatType

@testset "Round-off treatment of $(dynamics_kind) time stepping [$FT]" for dynamics_kind in (:anelastic, :compressible),
                                                                           FT in (Float32, Float64)

    model = rounding_test_model(FT, dynamics_kind)
    @test model.timestepper.rounding isa RoundToNearest

    model = rounding_test_model(FT, dynamics_kind; rounding = StochasticRounding(seed = 3))
    @test model.timestepper.rounding === StochasticRounding(seed = 3)

    default = run_and_snapshot(FT, dynamics_kind)
    nearest = run_and_snapshot(FT, dynamics_kind; rounding = RoundToNearest())
    stochastic = run_and_snapshot(FT, dynamics_kind; rounding = StochasticRounding(seed = 1))
    repeated = run_and_snapshot(FT, dynamics_kind; rounding = StochasticRounding(seed = 1))
    reseeded = run_and_snapshot(FT, dynamics_kind; rounding = StochasticRounding(seed = 2))

    # The default is round to nearest
    @test bitwise_equal(default, nearest)

    # Stochastic rounding is deterministic for a given seed...
    @test bitwise_equal(stochastic, repeated)

    # ... changes the result at round-off level, and depends on the seed.
    @test !bitwise_equal(stochastic, nearest)
    @test !bitwise_equal(stochastic, reseeded)

    # The acoustic substeps amplify round-off differences somewhat, so allow 1000 ulps
    for name in keys(stochastic)
        scale = max(1, maximum(abs, nearest[name]))
        @test maximum(abs, stochastic[name] .- nearest[name]) < 1000 * eps(FT) * scale
    end

    Oceananigans.defaults.FloatType = default_FT
end

#####
##### A slow, uniform heating in a 3D Float32 model
#####
##### 1.5 K/day at Δt = 2.5 s puts each stage increment of ρθ below one ulp, as for radiative
##### heating in the upper troposphere. Round to nearest loses most of it; stochastic rounding
##### matches Float64 to within its random walk.
#####

@testset "Slow uniform heating in Float32" begin
    day = 86400
    heating = 1.5 / day
    Δt = 2.5
    Nsteps = 1000

    function mean_temperature_change(FT; rounding = RoundToNearest())
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1000), y=(0, 1000), z=(0, 10000))
        model = AtmosphereModel(grid; forcing = (; θ = Returns(FT(heating))), timestepper_kwargs = (; rounding))
        set!(model; θ = 300)
        T₀ = mean(Float64.(Array(interior(model.temperature))))
        for _ in 1:Nsteps
            time_step!(model, Δt)
        end
        return mean(Float64.(Array(interior(model.temperature)))) - T₀
    end

    exact = Float64(mean_temperature_change(Float64))
    nearest = mean_temperature_change(Float32)
    stochastic = mean_temperature_change(Float32; rounding = StochasticRounding(seed = 5))
    @info "Mean temperature change over $(Nsteps) steps of $(Δt) s" exact nearest stochastic

    @test exact > 0

    # Round to nearest loses more than 20% of the heating; stochastic rounding recovers it
    @test !isapprox(nearest, exact, rtol = 0.2)
    @test stochastic ≈ exact rtol = 0.02

    Oceananigans.defaults.FloatType = default_FT
end
