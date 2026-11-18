include(joinpath(@__DIR__, "setup.jl"))

#####
##### A unit-scale version of the M-INT1 land smoke, without NumericalEarth: a resting standard
##### atmosphere on the 30 hybrid levels of the planned configuration (lowest layer 14 hPa) over a
##### 4° latitude-longitude grid, with a prescribed diurnal energy flux of up to 220 W m⁻² over a
##### "continent" and none elsewhere, for two days
#####
##### M-INT1 found that without vertical mixing the lowest layer over sunlit dry land left the
##### bounds of the state check after 1.14 days (372 K at day 1.5) and that the winds reached
##### 86 m s⁻¹ by day 2. Here the same heating is given to the core without vertical diffusion
##### (the control, which must reproduce the failure for the test to mean anything) and with the
##### implicit vertical diffusion and the Richardson-number boundary-layer scheme, plus the
##### ∇⁴ hyperdiffusion: the lowest layer must stay physically valid.
#####

using Breeze
using Breeze.HydrostaticEquations: GridScaleHyperdiffusion, VerticalDiffusion, RichardsonNumberBoundaryLayer,
                                   VerletSubsteps, hydrostatic_state_check, accumulate_state_checks
using Oceananigans
using Oceananigans.Units
using Printf
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

const Nz = 30
const rotation_rate = 7.29212e-5

# The sun is overhead at longitude 0° at t = 6 h, and the flux follows the cosine of the hour
# angle by day and vanishes by night. The continent spans 100° of longitude and 60° of latitude.
@inline over_land(λ, φ) = (-40 ≤ λ ≤ 60) & (-20 ≤ φ ≤ 40)
@inline surface_energy_flux(λ, φ, t) = ifelse(over_land(λ, φ), 220 * max(0, cos(2π * (t - 6 * 3600) / 86400 + λ * π / 180)) * cosd(φ), 0.0)

function heated_model(; mixing, kw...)
    grid = LatitudeLongitudeGrid(default_arch, FT; size=(90, 35, Nz), longitude=(-180, 180), latitude=(-70, 70), z=(0, 1))
    A, B = hybrid_coefficients(Nz; top_pressure=200)

    dissipation = mixing ? (hyperdiffusion = GridScaleHyperdiffusion(FT; damping_time=43200),
                            vertical_diffusion = VerticalDiffusion(boundary_layer=RichardsonNumberBoundaryLayer(FT))) : (;)

    dynamics = HydrostaticDynamics(; pressure_coefficient=A, surface_pressure_coefficient=B, dissipation...)
    # The flux is a field that is filled before every time step, as a coupler fills it
    energy_flux = Field{Center, Center, Nothing}(grid)
    boundary_conditions = (; ρE = FieldBoundaryConditions(bottom=FluxBoundaryCondition(energy_flux)))

    model = hydrostatic_model(grid; dynamics, boundary_conditions,
                              coriolis=HydrostaticSphericalCoriolis(FT; rotation_rate),
                              thermodynamic_constants=reference_model_constants(), kw...)

    p = dynamics_pressure(model.dynamics)
    T = Array(interior(p))
    T .= standard_temperature.(T)
    set!(model; θ = T ./ Array(interior(model.dynamics.exner_function)))
    return model, energy_flux
end

function set_energy_flux!(energy_flux, t)
    set!(energy_flux, (λ, φ) -> surface_energy_flux(λ, φ, t))
    return nothing
end

lowest_layer_temperature(model) = maximum(Array(interior(model.temperature))[:, :, 1])
largest_wind(model) = max(maximum(abs, Array(interior(model.velocities.u))), maximum(abs, Array(interior(model.velocities.v))))

# Two days in steps of Δt, with a state check every six hours. An integration that stops with an
# error has failed, and is reported as such.
function run_two_days!(model, energy_flux, Δt)
    checks = [hydrostatic_state_check(model)]
    series = [(0.0, lowest_layer_temperature(model), largest_wind(model))]
    completed = true

    try
        for n in 1:round(Int, 2days / Δt)
            set_energy_flux!(energy_flux, model.clock.time)
            time_step!(model, Δt)
            if n % round(Int, 6hours / Δt) == 0
                push!(checks, hydrostatic_state_check(model))
                push!(series, (model.clock.time / 86400, lowest_layer_temperature(model), largest_wind(model)))
            end
        end
    catch err
        err isa Union{DomainError, ArgumentError} || rethrow()
        completed = false
    end

    return (; completed, check = accumulate_state_checks(checks), series)
end

figdata_path(name) = (dir = get(ENV, "BREEZE_P3B_FIGDATA", ""); isempty(dir) ? nothing : joinpath(dir, name))

#####
##### Tolerances, written before these runs
#####
#
# The heat of one day is 220 (2 / π) 43200 cos φ = 6.0e6 J m⁻² at the equator. In the lowest layer
# alone (14 hPa, 143 kg m⁻²) that is 42 K per day; mixed over the 1.5 to 2 km that the convective
# column test finds, a few K.
#
#   1. Control (no vertical diffusion, no hyperdiffusion): the lowest layer leaves the bounds of
#      the state check (150 to 350 K) within the two days, or the integration stops: the failure
#      of M-INT1 is reproduced at this scale.
#   2. With vertical diffusion, the boundary-layer scheme and hyperdiffusion, for SSP RK3
#      (Δt = 300 s) and MIS3C + Verlet (Δt = 600 s): the integration completes; every six-hourly
#      state check is valid (all values finite, all layers with mass, 150 K ≤ T ≤ 350 K); the
#      lowest-layer temperature never exceeds 320 K; and no wind component exceeds 25 m s⁻¹
#      (a heat low of a few K over 1.5 km: thermal wind g ΔT H / (T f L) ≈ 6 m s⁻¹ at 30°, more
#      near the equator; M-INT1's control without land kept its 35 m s⁻¹ jet, and its failure
#      reached 86 m s⁻¹).

@testset "Heated continent without vertical mixing: the M-INT1 failure [$FT]" begin
    model, energy_flux = heated_model(mixing=false)
    result = run_two_days!(model, energy_flux, 300)
    T₁ = maximum(s -> s[2], result.series)

    @test !result.completed || !result.check.valid                                    # gate 1
    @info @sprintf("Control: completed = %s, valid = %s, warmest lowest layer %.1f K, largest wind %.1f m/s",
                   result.completed, result.check.valid, T₁, maximum(s -> s[3], result.series))

    path = figdata_path("surface_heating_smoke_control.csv")
    isnothing(path) || open(path, "w") do io
        println(io, "time_days,warmest_lowest_layer_temperature_K,largest_wind_component_m_per_s")
        foreach(s -> println(io, @sprintf("%.4f,%.4f,%.4f", s...)), result.series)
    end
end

@testset "Heated continent with vertical diffusion [$FT, $name]" for (name, kw, Δt) in
        (("SSP RK3", (;), 300), ("MIS3C + Verlet", split_explicit(:MIS3C, VerletSubsteps()), 600))

    model, energy_flux = heated_model(; mixing=true, kw...)
    result = run_two_days!(model, energy_flux, Δt)
    T₁ = maximum(s -> s[2], result.series)
    wind = maximum(s -> s[3], result.series)

    @test result.completed                                                             # gate 2
    @test result.check.valid
    @test result.check.finite && result.check.positive_layers && result.check.physical_temperatures
    @test T₁ < 320
    @test wind < 25
    @test T₁ > 288.5 # the land was heated

    @info @sprintf("With mixing [%s]: valid = %s, T between %.1f and %.1f K, warmest lowest layer %.1f K, largest wind %.1f m/s",
                   name, result.check.valid, result.check.extrema.T..., T₁, wind)

    path = figdata_path("surface_heating_smoke_$(replace(lowercase(name), " " => "_", "+" => "and")).csv")
    isnothing(path) || open(path, "w") do io
        println(io, "time_days,warmest_lowest_layer_temperature_K,largest_wind_component_m_per_s")
        foreach(s -> println(io, @sprintf("%.4f,%.4f,%.4f", s...)), result.series)
    end
end
