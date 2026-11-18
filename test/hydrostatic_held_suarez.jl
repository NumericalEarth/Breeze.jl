include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Forcings: HeldSuarezForcing, DensityTendencyForcing, equilibrium_temperature,
                       thermal_relaxation_rate, momentum_damping_rate
using Breeze.HydrostaticEquations: HydrostaticModel, VerletSubsteps
using Breeze.TimeSteppers: SSPRungeKutta3, GravityWaveMultirateInfinitesimalStep
using Oceananigans
using Oceananigans: fields
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: day
using Printf
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

tendencies(model) = (update_state!(model); map(G -> Array(interior(G)), model.timestepper.Gⁿ))

#####
##### Forcing of the prognostic velocities
#####
#
# `HydrostaticDynamics` evolves the velocities u and v. A forcing under `u` or `v` is an
# acceleration and enters the tendency as it is; a forcing under `ρu` or `ρv` is a force per unit
# volume, as for the dynamics that evolve momentum densities, and enters divided by the layer mass
# at the velocity point. Tolerance: the forcing is added to a tendency of ~ 1e-3 m s⁻², so the
# difference of two tendencies recovers it to rounding, 1e-12 relative to the forcing.

@testset "Velocity forcing of HydrostaticDynamics" begin
    grid = RectilinearGrid(default_arch, FT; size=(8, 6, 5), x=(0, 8e5), y=(0, 6e5), z=(0, 1))
    kˣ, kʸ = 2π / 8e5, 2π / 6e5

    acceleration(x, y, z, t) = 1e-4 * (1 + z) * cos(kˣ * x)
    force(x, y, z, t) = 1e-1 * (2 - z) * sin(kʸ * y)

    function model_with(forcing)
        model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(top_pressure=200), forcing)
        set!(model; pˢ = (x, y) -> 1e5 + 500 * cos(kˣ * x) * cos(kʸ * y),
                    θ = (x, y, z) -> 290 + 60 * z,
                    u = (x, y, z) -> 5 * sin(kʸ * y),
                    v = (x, y, z) -> 3 * sin(kˣ * x))
        return model
    end

    unforced = model_with(NamedTuple())
    G⁰ = tendencies(unforced)

    xᶠ, yᶜ, zᶜ = Array(xnodes(grid, Face())), Array(ynodes(grid, Center())), Array(znodes(grid, Center()))
    xᶜ, yᶠ = Array(xnodes(grid, Center())), Array(ynodes(grid, Face()))[1:6]
    ρ = Array(interior(continuity_density(unforced.dynamics)))
    ρᶠᶜᶜ = (ρ .+ circshift(ρ, (1, 0, 0))) ./ 2
    ρᶜᶠᶜ = (ρ .+ circshift(ρ, (0, 1, 0))) ./ 2

    # An acceleration under the velocity's name
    model = model_with((; u=acceleration, v=acceleration))
    @test model isa HydrostaticModel
    G = tendencies(model)
    expected_u = [acceleration(x, y, z, 0) for x in xᶠ, y in yᶜ, z in zᶜ]
    expected_v = [acceleration(x, y, z, 0) for x in xᶜ, y in yᶠ, z in zᶜ]
    @test maximum(abs, G.u .- G⁰.u .- expected_u) < 1e-12 * maximum(abs, expected_u)
    @test maximum(abs, G.v .- G⁰.v .- expected_v) < 1e-12 * maximum(abs, expected_v)
    @test G.ρθ == G⁰.ρθ
    @test G.pˢ == G⁰.pˢ

    # A force per unit volume under the momentum density's name, divided by the layer mass
    model = model_with((; ρu=force, ρv=force))
    @test model.forcing.u isa DensityTendencyForcing
    G = tendencies(model)
    expected_u = [force(x, y, z, 0) for x in xᶠ, y in yᶜ, z in zᶜ] ./ ρᶠᶜᶜ
    expected_v = [force(x, y, z, 0) for x in xᶜ, y in yᶠ, z in zᶜ] ./ ρᶜᶠᶜ
    @test maximum(abs, G.u .- G⁰.u .- expected_u) < 1e-12 * maximum(abs, expected_u)
    @test maximum(abs, G.v .- G⁰.v .- expected_v) < 1e-12 * maximum(abs, expected_v)

    # Both at once add
    model = model_with((; u=acceleration, ρu=force))
    G = tendencies(model)
    expected_u = [acceleration(x, y, z, 0) for x in xᶠ, y in yᶜ, z in zᶜ] .+
                 [force(x, y, z, 0) for x in xᶠ, y in yᶜ, z in zᶜ] ./ ρᶠᶜᶜ
    @test maximum(abs, G.u .- G⁰.u .- expected_u) < 1e-12 * maximum(abs, expected_u)
    @test G.v == G⁰.v
end

#####
##### Held-Suarez forcing: pointwise
#####
#
# The friction is - kᵘ u on the velocities, with σ = p / pˢ from the prognostic surface pressure
# and the layer pressure. The temperature relaxation is the enthalpy tendency
# - ρ cᵖᵈ kᵀ (T - Tᵉ) under `ρE`, which the potential temperature density takes divided by cᵖᵈ Π:
# ∂ₜ(ρθ) = - ρ kᵀ (T - Tᵉ) / Π. Without the Exner function the tendency would be that of ρT, too
# small by the factor Π, which is 0.61 at the highest level used here.
#
# Tolerances: 1e-10 relative. The friction is the difference of two velocity tendencies that are
# larger than it by at most 1e3; the relaxation is evaluated at rest, where it is the whole
# tendency of ρθ.

held_suarez_grid(; size=(90, 38, 8)) =
    LatitudeLongitudeGrid(default_arch, FT; size, longitude=(0, 360), latitude=(-76, 76), z=(0, 1))

function held_suarez_model(grid; forcing=held_suarez_forcings(FT), T=260, kw...)
    dynamics = HydrostaticDynamics(surface_pressure=pˢᵗ, standard_pressure=pˢᵗ, top_pressure=5000)
    model = hydrostatic_model(grid; dynamics, forcing, coriolis=HydrostaticSphericalCoriolis(FT),
                              thermodynamic_constants=reference_model_constants(), kw...)

    # Isothermal and at rest
    p = Array(interior(dynamics_pressure(model.dynamics)))
    set!(model; θ = T .* (pˢᵗ ./ p) .^ κ)
    return model
end

@testset "Held-Suarez forcing of HydrostaticDynamics" begin
    grid = held_suarez_grid(size=(8, 10, 8))
    model = held_suarez_model(grid)

    @test model.forcing.u isa HeldSuarezForcing{Val{:u}}
    @test model.forcing.v isa HeldSuarezForcing{Val{:v}}
    @test model.forcing.ρE isa HeldSuarezForcing{Val{:ρE}}
    @test model.forcing.ρE.surface_pressure === model.dynamics.surface_pressure

    # A surface pressure that varies, so that σ does. At rest the potential temperature has no
    # other tendency than the relaxation.
    unforced = held_suarez_model(grid; forcing=NamedTuple())
    surface_pressure(λ, φ) = 1e5 - 2e3 * cosd(λ) * cosd(3φ)
    set!(model; pˢ=surface_pressure)
    set!(unforced; pˢ=surface_pressure)
    Gρθ = tendencies(model).ρθ
    @test all(iszero, tendencies(unforced).ρθ)

    # With flow, for the friction
    flow = (u = (λ, φ, z) -> 10 * cosd(φ), v = (λ, φ, z) -> 2 * sind(λ) * cosd(3φ))
    set!(model; flow...)
    set!(unforced; flow...)
    G = tendencies(model)
    G⁰ = tendencies(unforced)

    parameters = model.forcing.ρE.parameters
    p = Array(interior(dynamics_pressure(model.dynamics)))
    pˢ = Array(interior(model.dynamics.surface_pressure))
    ρ = Array(interior(continuity_density(model.dynamics)))
    Π = Array(interior(model.dynamics.exner_function))
    T = Array(interior(model.temperature))
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))
    φ = Array(φnodes(grid, Center()))
    σ = p ./ pˢ

    @test minimum(σ) < 0.7 < maximum(σ) # levels in and above the boundary layer
    @test minimum(Π) < 0.65             # the Exner function is far from 1 aloft

    # Friction: σ averaged to the velocity points
    σᶠᶜᶜ = (σ .+ circshift(σ, (1, 0, 0))) ./ 2
    expected_u = [- momentum_damping_rate(parameters, σᶠᶜᶜ[i, j, k]) * u[i, j, k] for i in axes(u, 1), j in axes(u, 2), k in axes(u, 3)]
    @test maximum(abs, expected_u) > 1e-5
    @test maximum(abs, G.u .- G⁰.u .- expected_u) < 1e-10 * maximum(abs, expected_u)

    # v-faces 2 … Ny lie between cell centers; the first face is a wall
    σᶜᶠᶜ = (σ[:, 2:end, :] .+ σ[:, 1:end-1, :]) ./ 2
    expected_v = [- momentum_damping_rate(parameters, σᶜᶠᶜ[i, j, k]) * v[i, j+1, k] for i in axes(v, 1), j in axes(σᶜᶠᶜ, 2), k in axes(v, 3)]
    @test maximum(abs, expected_v) > 1e-6
    @test maximum(abs, (G.v .- G⁰.v)[:, 2:size(σ, 2), :] .- expected_v) < 1e-10 * maximum(abs, expected_v)

    # Temperature relaxation into ρθ, with the Exner function
    expected_ρθ = [- ρ[i, j, k] * thermal_relaxation_rate(parameters, φ[j], σ[i, j, k]) *
                   (T[i, j, k] - equilibrium_temperature(parameters, φ[j], p[i, j, k])) / Π[i, j, k]
                   for i in axes(ρ, 1), j in axes(ρ, 2), k in axes(ρ, 3)]
    @test maximum(abs, expected_ρθ) > 0
    @test maximum(abs, Gρθ .- expected_ρθ) < 1e-10 * maximum(abs, expected_ρθ)

    # Without the Exner function the tendency would be wrong by far more than the tolerance
    @test maximum(abs, Gρθ .- expected_ρθ .* Π) > 1e-2 * maximum(abs, expected_ρθ)

    # The surface pressure is not forced
    @test G.pˢ == G⁰.pˢ
end

#####
##### Held-Suarez spin-up: 10 days on a 4° grid
#####
#
# From an isothermal atmosphere at rest, the temperature relaxes toward the Held-Suarez
# equilibrium, which is warm at the equator and cold toward the walls at 76°. The response is
# zonally symmetric and close to thermal-wind balance, so a westerly jet grows aloft in each
# hemisphere as the temperature gradient does.
#
# Criteria, written before the first run. The free atmosphere relaxes at 1 / 40 days, so after
# 10 days the temperature has covered 1 - exp(-1/4) = 22 % of the way to equilibrium, whose
# difference between the equator and 60° is about 45 K (p / p₀)^κ: a gradient of about 10 K over
# 6.7e6 m. The thermal wind between the surface and the upper levels, over ln(pˢ / p) ≈ 1.5, is
# u ≈ (Rᵈ / f) (ΔT / L) ln(pˢ / p) ≈ 287 / 1e-4 × 1.5e-6 × 1.5 ≈ 6 m s⁻¹. The test requires, in
# each hemisphere, a zonal-mean westerly jet of at least 3 m s⁻¹ in the upper half of the levels
# poleward of 20°, stronger than the zonal-mean wind of the lowest level at the same latitude,
# which friction holds back; and that every field is a number, that the temperature stays between
# 150 and 350 K, and that the surface pressure stays within 5 % of 1e5 Pa.

function spin_up(model, Δt, days)
    for _ in 1:round(Int, days * day / Δt)
        time_step!(model, Δt)
    end

    u = Array(interior(model.velocities.u))
    zonal_mean_u = dropdims(sum(u, dims=1), dims=1) ./ size(u, 1) # (latitude, level)
    φ = Array(φnodes(model.grid, Center()))
    upper = size(u, 3) ÷ 2 + 1 : size(u, 3)

    jet(latitudes) = maximum((zonal_mean_u[j, k], j, k) for j in findall(latitudes, φ), k in upper)

    return (; φ, zonal_mean_u, north = jet(>(20)), south = jet(<(-20)),
              T = extrema(Array(interior(model.temperature))),
              pˢ = extrema(Array(interior(model.dynamics.surface_pressure))),
              v = max_abs(model.velocities.v),
              finite = all(isfinite, u) && all(isfinite, Array(interior(model.temperature))))
end

held_suarez_steppers = (("SSPRungeKutta3", SSPRungeKutta3, 200, NamedTuple()),
                        ("MIS3C + VerletSubsteps()", GravityWaveMultirateInfinitesimalStep, 1200,
                         (timestepper = :GravityWaveMultirateInfinitesimalStep,
                          timestepper_kwargs = (tableau = :MIS3C, inner_solver = VerletSubsteps()))))

@testset "Held-Suarez spin-up of HydrostaticDynamics [$name]" for (name, TimeStepper, Δt, kw) in held_suarez_steppers
    model = held_suarez_model(held_suarez_grid(); kw...)
    @test model.timestepper isa TimeStepper
    result = spin_up(model, Δt, 10)

    for (hemisphere, (speed, j, k)) in (("north", result.north), ("south", result.south))
        @info @sprintf("Held-Suarez spin-up, %s, day 10, %s: zonal-mean jet %.6f m s⁻¹ at %.0f° on level %d; lowest level there %.6f m s⁻¹",
                       name, hemisphere, speed, result.φ[j], k, result.zonal_mean_u[j, 1])
        @test speed > 3
        @test speed > result.zonal_mean_u[j, 1]
    end

    @info @sprintf("Held-Suarez spin-up, %s, day 10: T from %.4f to %.4f K, pˢ from %.2f to %.2f Pa, largest |v| %.6f m s⁻¹",
                   name, result.T..., result.pˢ..., result.v)

    @test result.finite
    @test 150 < result.T[1] && result.T[2] < 350
    @test 0.95e5 < result.pˢ[1] && result.pˢ[2] < 1.05e5
    @test model.clock.time ≈ 10day
end
