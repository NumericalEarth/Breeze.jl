include(joinpath(@__DIR__, "setup.jl"))

using Breeze, Oceananigans, Test
using Oceananigans.TimeSteppers: update_state!, time_step!

const AM = Breeze.AtmosphereModels
const TH = Breeze.Thermodynamics

# A test-only scheme: one condensate, prescribed velocity, no phase changes/sources.
struct SedimentationOnly{P, FT}
    phase :: P
    fall_speed :: FT
    closed_bottom :: Bool
end

AM.prognostic_field_names(::SedimentationOnly) = (:ρqʳ,)
AM.moisture_prognostic_name(::SedimentationOnly) = :ρqᵛ
AM.condensate_liquid_fraction(::SedimentationOnly{Val{:liquid}}, ::Val{:ρqʳ}) = 1
AM.condensate_liquid_fraction(::SedimentationOnly{Val{:ice}}, ::Val{:ρqʳ}) = 0
AM.sedimentation_velocity(::SedimentationOnly, μ, ::Val{:ρqʳ}) = μ.wʳ
AM.compute_microphysical_tendencies!(::SedimentationOnly, model) = nothing
AM.microphysics_model_update!(::SedimentationOnly, model) = nothing
@inline AM.maybe_adjust_thermodynamic_state(state, ::SedimentationOnly, vapor, constants) = state
@inline AM.microphysical_state(::SedimentationOnly, ρ, μ::NamedTuple, state, velocities) = (; qʳ=μ.ρqʳ / ρ)
@inline AM.moisture_fractions(::SedimentationOnly{Val{:liquid}}, state::NamedTuple, vapor) = TH.MoistureMassFractions(vapor, state.qʳ, oftype(vapor, 0))
@inline AM.moisture_fractions(::SedimentationOnly{Val{:ice}}, state::NamedTuple, vapor) = TH.MoistureMassFractions(vapor, oftype(vapor, 0), state.qʳ)
AM.liquid_mass_fraction(::SedimentationOnly{Val{:liquid}}, model) = model.microphysical_fields.qʳ
AM.liquid_mass_fraction(::SedimentationOnly{Val{:ice}}, model) = nothing
AM.ice_mass_fraction(::SedimentationOnly{Val{:ice}}, model) = model.microphysical_fields.qʳ
AM.ice_mass_fraction(::SedimentationOnly{Val{:liquid}}, model) = nothing

function AM.materialize_microphysical_fields(::SedimentationOnly, grid, bcs)
    return (; ρqʳ=CenterField(grid; boundary_conditions=bcs.ρqʳ),
              qʳ=CenterField(grid), qᵛ=CenterField(grid), wʳ=AM.sedimentation_velocity_field(grid))
end

@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, m::SedimentationOnly, state, ρ, thermo, constants)
    @inbounds μ.qʳ[i, j, k] = state.qʳ
    @inbounds μ.qᵛ[i, j, k] = thermo.moisture_mass_fractions.vapor
    speed = ifelse(m.closed_bottom & (k == 1), 0, -m.fall_speed)
    @inbounds μ.wʳ[i, j, k] = speed
    return nothing
end

# No resolved flow, phase changes, or gravity. The only scalar/thermal tendencies are
# sedimentation. Both open and closed bottoms distinguish inflow from net mass gain.
function sedimentation_column(FT, phase, donor_temperature, closed, speed; compressible=true)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(1, 1, 2), x=(0, 1), y=(0, 1), z=(0, 200))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    reference = ReferenceState(grid, constants; base_pressure=1e5, potential_temperature=280)
    dynamics = compressible ? CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=280) :
                             AnelasticDynamics(reference)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants,
                            microphysics=SedimentationOnly(Val(phase), FT(speed), closed),
                            scalar_advection=(; ρqʳ=UpwindBiased(FT; order=1)))
    condensate(x, y, z) = ifelse(z > 100, FT(0.01), FT(0.001))
    temperature(x, y, z) = ifelse(z > 100, FT(donor_temperature), FT(280))
    if compressible
        density(x, y, z) = FT(1e5) / (((1 - FT(0.005) - condensate(x, y, z)) * TH.dry_air_gas_constant(constants) +
                                       FT(0.005) * TH.vapor_gas_constant(constants)) * temperature(x, y, z))
        set!(model; ρ=density, T=temperature, qᵛ=FT(0.005), qʳ=condensate)
    else
        set!(model; T=temperature, qᵛ=FT(0.005), qʳ=condensate)
    end
    update_state!(model)
    AM.compute_tendencies!(model)
    return model
end

sedimentation_column_values(f) = Float64.(vec(Array(interior(f))))

# Independent host Float64 differences of the executable state definition, not of the
# callback's closed-form coefficient. The tendencies themselves use the selected FT/device.
function sedimentation_temperature_rates(model, phase, compressible)
    column = sedimentation_column_values
    constants = ThermodynamicConstants(Float64; gravitational_acceleration=0)
    ρ, ρv, ρx = column(AM.total_density(model.dynamics)), column(model.moisture_density), column(model.microphysical_fields.ρqʳ)
    T = column(model.temperature)
    mass_rate, theta_rate = column(model.timestepper.Gⁿ.ρqʳ), column(model.timestepper.Gⁿ.ρθ)
    properties = phase === :liquid ? constants.liquid : constants.ice
    cx, L = properties.heat_capacity, TH.absolute_zero_latent_heat(constants, properties)
    cd, cv = constants.dry_air.heat_capacity, constants.vapor.heat_capacity
    gas = (ρ .- ρv .- ρx) .* TH.dry_air_gas_constant(constants) .+ ρv .* TH.vapor_gas_constant(constants)
    capacity = (ρ .- ρv .- ρx) .* cd .+ ρv .* cv .+ ρx .* cx .- (compressible ? gas : zero(gas))
    pressure = compressible ? gas .* T : column(model.dynamics.reference_state.pressure)
    carrier = column(AM.dynamics_density(model.dynamics))
    function theta_density(k, temperature, condensate)
        density = compressible ? ρ[k] + condensate - ρx[k] : ρ[k]
        q = phase === :liquid ? TH.MoistureMassFractions(ρv[k] / density, condensate / density, 0.0) :
                               TH.MoistureMassFractions(ρv[k] / density, 0.0, condensate / density)
        p = compressible ? gas[k] * temperature : pressure[k]
        state = TH.LiquidIcePotentialTemperatureState(0.0, q, Float64(AM.standard_pressure(model.dynamics)), p)
        return carrier[k] * TH.with_temperature(state, temperature, constants).potential_temperature
    end
    derivative_T = [(theta_density(k, T[k] + 1e-3, ρx[k]) - theta_density(k, T[k] - 1e-3, ρx[k])) / 2e-3 for k in 1:2]
    derivative_mass = [(theta_density(k, T[k], ρx[k] + 1e-6) - theta_density(k, T[k], ρx[k] - 1e-6)) / 2e-6 for k in 1:2]
    rate = (theta_rate .- derivative_mass .* mass_rate) ./ derivative_T
    # Self-temperature outflow gives no sensible excess; use donor loss for incoming mass.
    heat_capacity = compressible ? cx : cx - cd
    reference_rate = [heat_capacity * (T[2] - T[1]) * (-mass_rate[2]) / capacity[1], 0.0]
    enthalpy = heat_capacity .* T .- L
    energy_rate = capacity .* rate .+ enthalpy .* mass_rate
    boundary_energy_rate = enthalpy[1] * sum(mass_rate) * 100
    return (; rate, reference_rate, mass_rate, energy_rate, boundary_energy_rate, capacity)
end

@testset "Isolated sedimentation temperature and thermal budget [$FT]" for FT in test_float_types()
    for compressible in (true, false), phase in (:liquid, :ice), donor_temperature in (275, 280, 285),
        closed in (true, false), speed in (0, 1)
        model = sedimentation_column(FT, phase, donor_temperature, closed, speed; compressible)
        data = sedimentation_temperature_rates(model, phase, compressible)
        # The reference rate is exactly zero in the second cell and whenever nothing falls
        # (speed = 0), so the comparison needs an absolute tolerance: the FT roundoff of the
        # device tendencies plus the error of the host finite differences, in K/s.
        tolerance = FT === Float32 ? 2e-6 : 2e-9
        @test maximum(abs.(data.rate .- data.reference_rate)) < tolerance
        @test abs(sum(data.energy_rate) * 100 - data.boundary_energy_rate) < sum(data.capacity) * 100 * tolerance
        if closed
            @test abs(sum(data.mass_rate)) < 100eps(FT) * max(maximum(abs, data.mass_rate), eps(FT))
        end
        if speed == 0
            @test maximum(abs, data.rate) < tolerance
        end
    end
end

# Rain falling through a column without gravity at uniform pressure. The acoustic substep loop
# forms its pressure perturbation from (ρθ)′ alone, so the composition content the rain moves,
# which changes ρθˡⁱ at fixed temperature and pressure, must not reach it (see
# `remove_sedimentation_content!`). Under adaptive implicit vertical advection it once did, and
# the spurious pressure drove motion and adiabatic heating and cooling of several kelvin in a few
# minutes, concentrated where the fall Courant number crosses the split's CFL target.
function acoustic_sedimentation_column(FT, scheme; fall_speed = 8, Δt = 2, stop_time = 150,
                                       temperature = z -> 290)
    Oceananigans.defaults.FloatType = FT

    # 10 m at the surface, stretched by 15 % per level to 1.2 km: at Δt = 2 s the fall Courant
    # number exceeds 0.5 below about 200 m, so the adaptive split engages near the surface.
    z = [0.0]
    while z[end] < 1200
        push!(z, z[end] + 10 * 1.15^(length(z) - 1))
    end

    grid = RectilinearGrid(default_arch, FT; size=length(z) - 1, z, topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=290)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants,
                            microphysics=SedimentationOnly(Val(:liquid), FT(fall_speed), false),
                            scalar_advection=(; ρqʳ=scheme))

    # The gas pressure p₀ = (ρᵈ Rᵈ + ρᵛ Rᵛ) T is uniform, with vapor a fixed fraction ε of the dry
    # air by density. The rain only adds mass, so the vapor's fraction of the total mass falls
    # where the rain is: qᵛ = ε (1 − qʳ) / (1 + ε).
    p₀, ε = 1e5, 0.005
    Rᵈ, Rᵛ = TH.dry_air_gas_constant(constants), TH.vapor_gas_constant(constants)
    rain(z) = ifelse(700 < z < 1000, 0.01, 0.0)
    set!(model; ρᵈ = z -> p₀ / ((Rᵈ + ε * Rᵛ) * temperature(z)), T = temperature,
         qᵛ = z -> ε * (1 - rain(z)) / (1 + ε), qʳ = rain)

    p = sedimentation_column_values(model.dynamics.pressure)
    T⁰ = sedimentation_column_values(model.temperature)
    rain⁰ = sum(sedimentation_column_values(model.microphysical_fields.ρqʳ) .* diff(z))
    largest_departure = 0.0
    for _ in 1:round(Int, stop_time / Δt)
        time_step!(model, FT(Δt))
        T = sedimentation_column_values(model.temperature)
        largest_departure = max(largest_departure, maximum(abs, T .- T⁰))
    end
    rain = sum(sedimentation_column_values(model.microphysical_fields.ρqʳ) .* diff(z))

    return (; model, pressure_spread = (maximum(p) - minimum(p)) / maximum(p), largest_departure,
              rain_left = rain / rain⁰, temperature = sedimentation_column_values(model.temperature), T⁰)
end

@testset "Sedimentation keeps an isothermal column's temperature on the acoustic stepper [$FT]" for FT in test_float_types()
    adaptive = UpwindBiased(FT; order=1, time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5))

    # The fixture starts at rest and stays there while nothing falls.
    resting = acoustic_sedimentation_column(FT, adaptive; fall_speed = 0)
    @test resting.pressure_spread < 10eps(FT)
    @test resting.largest_departure < 1e-6

    # An isothermal column moves no sensible heat, so the temperature must not change.
    for (scheme, Δt) in ((UpwindBiased(FT; order=1), 1), (adaptive, 2), (adaptive, 5))
        column = acoustic_sedimentation_column(FT, scheme; Δt)
        @test column.model.timestepper isa Breeze.TimeSteppers.AcousticRungeKutta3
        @test column.pressure_spread < 10eps(FT)
        @test column.rain_left < 0.2 # most of the rain has fallen out of the column
        @test column.largest_departure < 0.05
    end
end

# Through a column cooling with height, the falling rain carries its sensible heat down, a real
# heating that changes the pressure. It joins the slow tendency the substep loop integrates while
# only the composition content bypasses the loop, so at any time step the adaptive split reaches
# the temperature an explicit run reaches at a small one.
@testset "Sedimenting rain carries its sensible heat at any time step on the acoustic stepper [$FT]" for FT in test_float_types()
    adaptive = UpwindBiased(FT; order=1, time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5))
    cooling_with_height(z) = 300 - z / 100
    reference = acoustic_sedimentation_column(FT, UpwindBiased(FT; order=1); Δt = 0.5, temperature = cooling_with_height)
    signal = maximum(abs, reference.temperature .- reference.T⁰)
    @test reference.pressure_spread < 10eps(FT)
    @test signal > 0.1
    for Δt in (2, 5)
        column = acoustic_sedimentation_column(FT, adaptive; Δt, temperature = cooling_with_height)
        @test maximum(abs, column.temperature .- reference.temperature) < signal / 5
    end
end
