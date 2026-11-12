include(joinpath(@__DIR__, "setup.jl"))

using Breeze, Oceananigans, Test
using Oceananigans: fields
using Oceananigans.TimeSteppers: update_state!, time_step!, implicit_step!

const AM = Breeze.AtmosphereModels
const TH = Breeze.Thermodynamics
const TS = Breeze.TimeSteppers

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
function sedimentation_column(FT, phase, donor_temperature, closed, speed;
                               compressible = true,
                               time_discretization = ExplicitTimeStepping(),
                               advection = UpwindBiased(FT; order=1))
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(1, 1, 2), x=(0, 1), y=(0, 1), z=(0, 200))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    reference = ReferenceState(grid, constants; base_pressure=1e5, potential_temperature=280)
    dynamics = compressible ? CompressibleDynamics(time_discretization; reference_potential_temperature=280) :
                             AnelasticDynamics(reference)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants,
                            microphysics=SedimentationOnly(Val(phase), FT(speed), closed),
                            scalar_advection=(; ρqʳ=advection))
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

# First-order upwind with the adaptive implicit split engaging above a fall Courant number of 0.5.
adaptive_upwind(FT) = UpwindBiased(FT; order=1, time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5))

# The fields of the acoustic stepper's sedimentation content bypass.
sedimentation_content(model) = TS.sedimentation_bypass(model).content
frozen_thermodynamic_density(model) = TS.sedimentation_bypass(model).frozen_thermodynamic_density

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
    weighting_density = column(AM.continuity_density(model.dynamics))
    function theta_density(k, temperature, condensate)
        density = compressible ? ρ[k] + condensate - ρx[k] : ρ[k]
        q = phase === :liquid ? TH.MoistureMassFractions(ρv[k] / density, condensate / density, 0.0) :
                               TH.MoistureMassFractions(ρv[k] / density, 0.0, condensate / density)
        p = compressible ? gas[k] * temperature : pressure[k]
        state = TH.LiquidIcePotentialTemperatureState(0.0, q, Float64(AM.standard_pressure(model.dynamics)), p)
        return weighting_density[k] * TH.with_temperature(state, temperature, constants).potential_temperature
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
    return (; rate, reference_rate, mass_rate, energy_rate, boundary_energy_rate, capacity,
              derivative_T, derivative_mass, heat_capacity)
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

# The composition and sensible parts the implicit remainder should move, from the independent
# derivatives of `sedimentation_temperature_rates` and the mass rate the solve actually applied:
# the content of the mass, and the conversion of the sensible heat the donor cell's mass brings
# to the receiving cell, where the donor's temperature differs from 280 K.
function expected_implicit_parts(reference, donor_temperature, implicit_mass_rate)
    temperature_rate = reference.heat_capacity * (donor_temperature - 280) * implicit_mass_rate[1] / reference.capacity[1]
    composition = reference.derivative_mass .* implicit_mass_rate
    sensible = reference.derivative_T .* [temperature_rate, 0]
    return composition, sensible
end

# The implicit mass flux must use the solved rain/ice density, but its thermal coefficients
# must come from one consistent state. Compare the split against independent derivatives of
# the stage-entry thermodynamic definition and the mass the solve actually moved. In particular,
# equal donor and receiver temperatures give exactly zero sensible heat at large fall CFL.
@testset "Implicit acoustic sedimentation uses consistent thermal coefficients [$FT]" for FT in test_float_types()
    column = sedimentation_column_values

    for phase in (:liquid, :ice), donor_temperature in (275, 280, 285), Δt in FT.((5, 20))
        model = sedimentation_column(FT, phase, donor_temperature, true, 80;
                                     time_discretization=SplitExplicitTimeDiscretization(), advection=adaptive_upwind(FT))
        Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(model, Δt, [])
        TS.store_initial_state!(model)
        TS.remove_sedimentation_content!(model, 1)
        TS.cache_advecting_state!(model)
        reference = sedimentation_temperature_rates(model, phase, true)
        Δt_stage = model.timestepper.β₁ * Δt
        predicted_mass = column(model.timestepper.U⁰.ρqʳ) .+ Δt_stage .* column(model.timestepper.Gⁿ.ρqʳ)

        TS.add_explicit_sedimentation_content!(model, Δt_stage)
        sensible_before = column(model.timestepper.Gⁿ.ρθ)
        composition_before = column(sedimentation_content(model))
        TS.scalar_rk3_substep!(model, Δt_stage)

        implicit_mass_rate = (column(model.microphysical_fields.ρqʳ) .- predicted_mass) ./ Δt_stage
        sensible = column(model.timestepper.Gⁿ.ρθ) .- sensible_before
        composition = (column(sedimentation_content(model)) .- composition_before) ./ Δt_stage
        expected_composition, expected_sensible = expected_implicit_parts(reference, donor_temperature, implicit_mass_rate)
        tolerance = FT === Float32 ? 3e-5 : 3e-8

        @test implicit_mass_rate[1] > 0
        @test implicit_mass_rate[1] ≈ -implicit_mass_rate[2] rtol=tolerance
        @test composition ≈ expected_composition rtol=tolerance
        @test isapprox(sensible, expected_sensible; rtol=tolerance, atol=100eps(FT))
        if donor_temperature == 280
            @test maximum(abs, sensible) < 100eps(FT)
        else
            @test sign(sensible[1]) == sign(donor_temperature - 280)
        end
    end
end

# The SSP stepper moves the same content in one piece, between the tracers' solves and the
# thermodynamic variable's own (`implicit_sedimentation_step!`): the mass fluxes from the solved
# rain, the content per unit mass from the stage-entry state. Equal donor and receiver temperatures
# must then move composition content alone. A temperature rediagnosed from the stage-entry θ and
# the solved rain is off by about ℒ Δq / cᵖ in the receiving cell, and would turn that fictitious
# contrast into sensible heat.
@testset "Implicit SSP sedimentation uses consistent thermal coefficients [$FT]" for FT in test_float_types()
    column = sedimentation_column_values

    for phase in (:liquid, :ice), donor_temperature in (275, 280, 285), Δt in FT.((5, 20))
        model = sedimentation_column(FT, phase, donor_temperature, true, 80; compressible=false, advection=adaptive_upwind(FT))
        @test model.timestepper isa TS.SSPRungeKutta3
        Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(model, Δt, [])
        TS.store_initial_state!(model)
        reference = sedimentation_temperature_rates(model, phase, false)
        ρθ = model.formulation.potential_temperature_density
        explicit_ρθ = column(ρθ) .+ Δt .* column(model.timestepper.Gⁿ.ρθ)
        predicted_mass = column(model.microphysical_fields.ρqʳ) .+ Δt .* column(model.timestepper.Gⁿ.ρqʳ)

        TS.ssp_rk3_substep!(model, Δt, one(FT))

        implicit_mass_rate = (column(model.microphysical_fields.ρqʳ) .- predicted_mass) ./ Δt
        implicit_content_rate = (column(ρθ) .- explicit_ρθ) ./ Δt
        composition, sensible = expected_implicit_parts(reference, donor_temperature, implicit_mass_rate)
        tolerance = FT === Float32 ? 3e-5 : 3e-8

        @test implicit_mass_rate[1] > 0
        @test implicit_mass_rate[1] ≈ -implicit_mass_rate[2] rtol=tolerance
        @test implicit_content_rate ≈ composition .+ sensible rtol=tolerance
        if donor_temperature == 280
            @test implicit_content_rate ≈ composition rtol=tolerance
        else
            @test sign(implicit_content_rate[1] - composition[1]) == sign(donor_temperature - 280)
        end
    end
end

# Rain falling through a column at rest: without gravity at uniform pressure, or under gravity in
# the substep loop's discrete hydrostatic balance. The acoustic substep loop forms its pressure
# perturbation from (ρθ)′ alone, so the composition content the rain moves, which changes ρθˡⁱ at
# fixed temperature and pressure, must not reach it (see the section comment on the bypass in
# `acoustic_runge_kutta_3.jl`). Under adaptive implicit vertical advection it once did, and the
# spurious pressure drove motion and adiabatic heating and cooling of several kelvin in a few
# minutes, concentrated where the fall Courant number crosses the split's CFL target.
function acoustic_sedimentation_column(FT, scheme; fall_speed = 8, Δt = 2, stop_time = 150,
                                       temperature = z -> 290, gravity = false)
    Oceananigans.defaults.FloatType = FT

    # 10 m at the surface, stretched by 15 % per level to 1.2 km: at Δt = 2 s the fall Courant
    # number exceeds 0.5 below about 200 m, so the adaptive split engages near the surface.
    z = [0.0]
    while z[end] < 1200
        push!(z, z[end] + 10 * 1.15^(length(z) - 1))
    end

    grid = RectilinearGrid(default_arch, FT; size=length(z) - 1, z, topology=(Flat, Flat, Bounded))
    g = gravity ? 9.80665 : 0
    constants = ThermodynamicConstants(FT; gravitational_acceleration=g)
    # Under gravity, an isothermal reference in the substep loop's exact discrete hydrostatic
    # balance (`substepper_rest_state.jl`), which anchors the column's own balance below.
    dynamics = gravity ?
        CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_temperature=290, base_pressure=1e5, standard_pressure=1e5) :
        CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=290)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants,
                            microphysics=SedimentationOnly(Val(:liquid), FT(fall_speed), false),
                            scalar_advection=(; ρqʳ=scheme))

    # The gas pressure p = (ρᵈ Rᵈ + ρᵛ Rᵛ) T, with vapor a fixed fraction ε of the dry air by
    # density. The rain only adds mass, so the vapor's fraction of the total mass falls where the
    # rain is, qᵛ = ε (1 − qʳ) / (1 + ε), as does the dry air's, qᵈ = (1 − qʳ) / (1 + ε).
    p₀, ε = 1e5, 0.005
    Rᵈ, Rᵛ = TH.dry_air_gas_constant(constants), TH.vapor_gas_constant(constants)
    rain(z) = ifelse(700 < z < 1000, 0.01, 0.0)
    moisture = (; qᵛ = z -> ε * (1 - rain(z)) / (1 + ε), qʳ = rain)
    if gravity
        # The loop's buoyancy weighs the total density, rain included, so the column is set in
        # the discrete hydrostatic balance of its full state with the loop's own operators,
        # p(k+1) − p(k) = −g Δzᶠ (ρ(k) + ρ(k+1)) / 2 between cell centers, with p = ρᵈ (Rᵈ + ε Rᵛ) T
        # at a uniform T and ρ = ρᵈ / qᵈ. The bottom cell, which holds no rain, is anchored on
        # the reference, whose own balance has the same form, so the `temperature` argument is
        # unused and the loop's slow vertical momentum tendency, which differences the full
        # state's pressure and density against the reference, vanishes to roundoff.
        zᶜ = Float64.(Array(znodes(grid, Center())))
        ρᵣ = sedimentation_column_values(model.dynamics.reference_state.density)
        pᵣ = sedimentation_column_values(model.dynamics.reference_state.pressure)
        qᵈ = (1 .- rain.(zᶜ)) ./ (1 + ε)
        T = pᵣ[1] / (ρᵣ[1] * qᵈ[1] * (Rᵈ + ε * Rᵛ))
        a = (Rᵈ + ε * Rᵛ) * T
        ρᵈ = similar(zᶜ)
        ρᵈ[1] = ρᵣ[1] * qᵈ[1]
        for k in 1:length(zᶜ)-1
            Δzᶠ = zᶜ[k+1] - zᶜ[k]
            ρᵈ[k+1] = ρᵈ[k] * (a - g * Δzᶠ / (2qᵈ[k])) / (a + g * Δzᶠ / (2qᵈ[k+1]))
        end
        set!(model; ρᵈ = reshape(FT.(ρᵈ), 1, 1, :), T = FT(T), moisture...)
    else
        # The gas pressure p₀ is uniform.
        set!(model; ρᵈ = z -> p₀ / ((Rᵈ + ε * Rᵛ) * temperature(z)), T = temperature, moisture...)
    end

    spread(p) = (maximum(p) - minimum(p)) / maximum(p)
    p⁰ = sedimentation_column_values(model.dynamics.pressure)
    T⁰ = sedimentation_column_values(model.temperature)
    rain⁰ = sum(sedimentation_column_values(model.microphysical_fields.ρqʳ) .* diff(z))
    largest_departure = 0.0
    for _ in 1:round(Int, stop_time / Δt)
        time_step!(model, FT(Δt))
        T = sedimentation_column_values(model.temperature)
        largest_departure = max(largest_departure, maximum(abs, T .- T⁰))
    end
    rain_mass = sum(sedimentation_column_values(model.microphysical_fields.ρqʳ) .* diff(z))

    # The pressure and the velocity the run leaves, diagnosed by the last step's `update_state!`:
    # a spurious pressure drives motion, and the motion changes the pressure.
    p = sedimentation_column_values(model.dynamics.pressure)
    w = sedimentation_column_values(model.velocities.w)

    return (; model, initial_pressure_spread = spread(p⁰), pressure_spread = spread(p),
              pressure_change = maximum(abs, p .- p⁰) / maximum(p⁰), pressure = p, vertical_velocity = w,
              largest_vertical_velocity = maximum(abs, w), largest_departure, rain_left = rain_mass / rain⁰,
              temperature = sedimentation_column_values(model.temperature), T⁰)
end

@testset "Sedimentation keeps an isothermal column's temperature on the acoustic stepper [$FT]" for FT in test_float_types()
    # The fixture starts at rest at a uniform pressure and stays there, to roundoff, while
    # nothing falls.
    resting = acoustic_sedimentation_column(FT, adaptive_upwind(FT); fall_speed = 0)
    @test resting.model.timestepper isa TS.AcousticRungeKutta3
    @test resting.initial_pressure_spread < 10eps(FT)
    @test resting.pressure_spread < 10eps(FT)
    @test resting.largest_vertical_velocity < 100eps(FT)
    @test resting.largest_departure < 100eps(FT) * maximum(resting.T⁰)

    # An isothermal column moves no sensible heat, so the temperature must not change; the rain
    # moves no gas, so the pressure must stay uniform and the column at rest. The spurious
    # pressure moved the column at meters per second; the splitting leaves, at Δt = 5, a
    # departure of 8 mK, a relative pressure change of 5e-6 and a velocity of 3e-6 m/s at the
    # end of the run.
    for (scheme, Δt) in ((UpwindBiased(FT; order=1), 1), (adaptive_upwind(FT), 2), (adaptive_upwind(FT), 5))
        column = acoustic_sedimentation_column(FT, scheme; Δt)
        @test column.rain_left < 0.2 # most of the rain has fallen out of the column
        @test column.largest_departure < 0.02
        @test column.pressure_spread < 5e-5
        @test column.largest_vertical_velocity < 1e-3
    end
end

# The same column under gravity, in the discrete hydrostatic balance of its full state, rain
# included. The bypass removes the content from the base the loop measures its perturbations
# from, while the loop's pressure gradient and buoyancy difference the full state's pressure and
# density against the reference: the balance must survive the removal, to roundoff, while nothing
# falls. The falling rain unloads the column, a real change of the pressure and a real, small,
# motion, so the adaptive split at a large time step is held to the explicit run at a small one,
# within the splitting difference the column shows without gravity (8 mK at Δt = 5).
@testset "Sedimentation keeps a hydrostatic isothermal column at rest on the acoustic stepper [$FT]" for FT in test_float_types()
    resting = acoustic_sedimentation_column(FT, adaptive_upwind(FT); fall_speed = 0, gravity = true)
    @test resting.pressure_change < 100eps(FT)
    @test resting.largest_vertical_velocity < 1e4 * eps(FT)
    @test resting.largest_departure < 100eps(FT) * maximum(resting.T⁰)

    reference = acoustic_sedimentation_column(FT, UpwindBiased(FT; order=1); Δt = 0.5, gravity = true)
    @test reference.rain_left < 0.2
    @test reference.pressure_change > 1e-4 # the rain's weight has left the column
    for (scheme, Δt) in ((UpwindBiased(FT; order=1), 1), (adaptive_upwind(FT), 2), (adaptive_upwind(FT), 5))
        column = acoustic_sedimentation_column(FT, scheme; Δt, gravity = true)
        @test column.rain_left < 0.2
        @test maximum(abs, column.temperature .- reference.temperature) < 0.02
        @test maximum(abs, column.pressure .- reference.pressure) / maximum(reference.pressure) < 5e-5
        @test maximum(abs, column.vertical_velocity .- reference.vertical_velocity) < 1e-3
    end
end

# The thermodynamic variable's post-loop operator in full: the implicit part of a boundary flux,
# which reads the variable's boundary conditions and boundary values, and coefficients that read
# the variable, here from the cells on both sides of each face, which the variable's own solve
# overwrites as it sweeps up the column.
@inline diffusivity_of_ρθ(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields) =
    @inbounds ((fields.ρθ[i, j, k-1] + fields.ρθ[i, j, k]) / 600)^4 / 2

# A model without gravity on `grid` whose rain falls at `fall_speed`, for the stage mechanics
# tests below; `advection` is the rain's scheme unless `scalar_advection` names every scheme.
function rain_model(FT, grid; fall_speed, closed_bottom = false, advection = UpwindBiased(FT; order=1),
                    scalar_advection = (; ρqʳ = advection), open_boundary_relaxation = 0.5, kwargs...)
    Oceananigans.defaults.FloatType = FT
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(; open_boundary_relaxation);
                                    reference_potential_temperature=290)
    return AtmosphereModel(grid; dynamics, scalar_advection, thermodynamic_constants=constants,
                           microphysics=SedimentationOnly(Val(:liquid), FT(fall_speed), closed_bottom), kwargs...)
end

# A four-cell column for that operator: the closures to try, one of them the state-dependent
# diffusivity above, and an implicit flux −ρθ through the bottom, which halves the bottom cell
# over Δt = Δz = 1. The rain's adaptive scheme gives the model its implicit solver.
function postloop_operator_fixture(FT)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=4, z=(0, 4), topology=(Flat, Flat, Bounded))
    boundary_conditions = (; ρθ=FieldBoundaryConditions(bottom=IMEXFluxBoundaryCondition(0, -1)))
    state_dependent = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=diffusivity_of_ρθ, discrete_form=true)
    return (; grid, boundary_conditions, closures = (nothing, state_dependent))
end

# A model on that column with rain above 2 m, its state diagnosed and its stage-entry advecting
# state frozen, as the stage leaves it for the post-loop solves.
function postloop_model(fixture; temperature = 290, kwargs...)
    FT = eltype(fixture.grid)
    model = rain_model(FT, fixture.grid; fall_speed = 1, advection = adaptive_upwind(FT), kwargs...)
    set!(model; ρᵈ=1.2, T=temperature, qᵛ=0.005, qʳ=z -> ifelse(z > 2, 0.001, 0.0))
    update_state!(model)
    TS.cache_advecting_state!(model)
    return model
end

# Oceananigans' solve of the thermodynamic variable with its post-loop operator, reading the live
# model fields, over Δt: the post-loop solve when no condensate sediments.
function oceananigans_postloop_solve!(model, Δt)
    TS.thermodynamic_implicit_step!(model, model.timestepper.implicit_solver, Δt)
    return sedimentation_column_values(AM.thermodynamic_density(model.formulation))
end

# The composition content takes the thermodynamic variable's post-loop solve on its own, so it
# must take the variable's operator in full. A content proportional to the variable, s ρθ, then
# adds s A⁻¹ρθ, where A⁻¹ρθ is what the variable's own solve gives, so the increment δ and the
# result ρθ¹ = A⁻¹ρθ + δ satisfy δ = s (ρθ¹ − δ).
@testset "The sedimentation content takes the thermodynamic variable's post-loop operator [$FT]" for FT in test_float_types()
    fixture = postloop_operator_fixture(FT)
    s = FT(0.3)

    for closure in fixture.closures
        model = postloop_model(fixture; closure, fixture.boundary_conditions)
        ρθ = AM.thermodynamic_density(model.formulation)
        content = sedimentation_content(model)
        interior(content) .= s .* interior(ρθ)
        TS.apply_sedimentation_content!(model)
        TS.implicit_substep!(model, FT(1))

        δ = sedimentation_column_values(content)
        ρθ¹ = sedimentation_column_values(ρθ)
        @test δ ≈ s .* (ρθ¹ .- δ) rtol=100eps(FT)
    end
end

# With a bypass, the variable and the content take one call that mirrors Oceananigans'
# `implicit_step!` call for call, with the variable's boundary conditions and boundary value in
# place of the solved field's and the frozen variable as the fields the coefficients read. The
# variable, and a content equal to it, must then solve to what Oceananigans' own solve of the
# variable gives, for a closure whose coefficients do not read the variable. The thermodynamic
# scheme is adaptive implicit, so every solve leaves the advective half to the loop. A coefficient
# that reads the variable is the reason for the frozen copy: Oceananigans' solve reads it live,
# below the row its sweep has already overwritten, and gives something else.
@testset "The post-loop solves are Oceananigans' implicit_step! of the thermodynamic variable [$FT]" for FT in test_float_types()
    fixture = postloop_operator_fixture(FT)
    constant = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=1)

    for closure in (nothing, constant, fixture.closures[2])
        model = postloop_model(fixture; closure, fixture.boundary_conditions, temperature = z -> 290 - z,
                               scalar_advection = (; ρθ = adaptive_upwind(FT), ρqʳ = adaptive_upwind(FT)))
        ρθ = AM.thermodynamic_density(model.formulation)
        content = sedimentation_content(model)
        ρθ⁰ = sedimentation_column_values(ρθ)
        ρθ⁰_with_halos = copy(parent(ρθ))
        interior(content) .= interior(ρθ)
        TS.implicit_substep!(model, FT(1))
        ρθ¹ = sedimentation_column_values(ρθ)
        content¹ = sedimentation_column_values(content)
        @test maximum(abs, ρθ¹ .- ρθ⁰) > 0
        @test content¹ == ρθ¹

        parent(ρθ) .= ρθ⁰_with_halos
        oceananigans = oceananigans_postloop_solve!(model, FT(1))
        if closure === fixture.closures[2]
            @test ρθ¹ != oceananigans
        else
            @test ρθ¹ == oceananigans
        end
    end
end

# The operator's coefficients read the state the solves advance: the variable with the stage's
# content in it, not without. A flux −J through the bottom split by Patankar's rule puts
# λ = −J/ρθ₁ on the bottom cell's diagonal, with ρθ₁ the full bottom value, so the solve leaves
# ρθ₁² / (ρθ₁ + J Δt/Δz) there. Oceananigans' solve of the full variable reads ρθ₁ live, before
# its sweep overwrites it, and must agree.
@testset "The post-loop operator reads the thermodynamic variable with the sedimentation content in it [$FT]" for FT in test_float_types()
    fixture = postloop_operator_fixture(FT)
    J = FT(30)
    model = postloop_model(fixture; closure=nothing,
                           boundary_conditions = (; ρθ=FieldBoundaryConditions(bottom=IMEXFluxBoundaryCondition(-J))))

    ρθ = AM.thermodynamic_density(model.formulation)
    content = sedimentation_content(model)
    interior(content) .= -FT(0.1) .* interior(ρθ)
    TS.apply_sedimentation_content!(model)
    full = sedimentation_column_values(ρθ)
    full_with_halos = copy(parent(ρθ))
    TS.implicit_substep!(model, FT(1))
    ρθ¹ = sedimentation_column_values(ρθ)
    expected = copy(full)
    expected[1] = full[1]^2 / (full[1] + J)
    @test ρθ¹ ≈ expected rtol=100eps(FT)
    @test !(ρθ¹ ≈ [full[1] / (1 + J / (full[1] / 0.9)); full[2:end]])

    parent(ρθ) .= full_with_halos
    @test oceananigans_postloop_solve!(model, FT(1)) ≈ ρθ¹ rtol=10eps(FT)
end

# Opening a stage keeps the full variable, halos included, and takes the content out of the halos
# too, so that the reduced variable's halos are the full variable's less the content: the vertical
# operators the stage applies to the reduced variable see no step at the boundaries, and a wall
# value of the full variable is not held against the reduced interior.
@testset "Removing the sedimentation content takes it out of the halos too [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(4, 4), x=(0, 4), z=(0, 4), topology=(Bounded, Flat, Bounded))
    model = rain_model(FT, grid; fall_speed = 1,
                       boundary_conditions = (; ρθ=FieldBoundaryConditions(west=ValueBoundaryCondition(FT(360)))))
    set!(model; ρᵈ=1.2, T=290, qᵛ=0.005, qʳ=0.001)
    update_state!(model)

    ρθ = AM.thermodynamic_density(model.formulation)
    content = sedimentation_content(model)
    full_with_halos = Array(parent(ρθ))
    c = FT(3)
    interior(content) .= c
    @allowscalar begin
        ρθ_cell⁰ = ρθ[1, 1, 1]
        step⁰ = ρθ[0, 1, 1] - ρθ[1, 1, 1]
        @test step⁰ / 2 == 360 - ρθ_cell⁰
        TS.remove_sedimentation_content!(model, 2)
        @test ρθ[1, 1, 1] == ρθ_cell⁰ - c
        @test ρθ[0, 1, 1] - ρθ[1, 1, 1] ≈ step⁰ rtol=10eps(FT)
    end
    @test all(iszero, Array(parent(content)))
    @test Array(parent(frozen_thermodynamic_density(model))) == full_with_halos
end

# The content is kept out of the substep loop and added after it, so the loop's open-boundary
# relaxation must target the wall value for the full variable: the stage-entry variable's wall
# value less the content the stage adds. Targeting the base it reads would end each stage at
# the wall value plus this stage's content less the previous stage's.
@testset "The open-boundary relaxation lands the full thermodynamic variable on the wall value [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(4, 2), x=(0, 4000), z=(0, 200), topology=(Bounded, Flat, Bounded))
    v = FT(360)
    boundary_conditions = (; ρθ=FieldBoundaryConditions(west=ValueBoundaryCondition(v)),
                             ρu=FieldBoundaryConditions(west=NormalFlowBoundaryCondition(FT(0)),
                                                        east=NormalFlowBoundaryCondition(FT(0))))
    model = rain_model(FT, grid; fall_speed = 5, open_boundary_relaxation = 1, boundary_conditions)
    set!(model; ρᵈ=1.2, T=290, qᵛ=0.005, qʳ=(x, z) -> ifelse(z > 100, FT(0.01), FT(0.001)))
    time_step!(model, 1)
    ρθ = AM.thermodynamic_density(model.formulation)
    @test all(!iszero, Array(interior(sedimentation_content(model), 1, 1, :)))
    @test Array(interior(ρθ, 1, 1, :)) ≈ fill(v, 2) rtol=100eps(FT)
end

# Open velocity boundaries do not prescribe a thermodynamic value when ρθ has a flux or
# gradient boundary condition. A horizontally uniform rain column must not develop boundary
# hot/cold spots as its composition changes. Exercise both horizontal directions of the
# relaxation kernel and the default as well as full relaxation strength; the value-boundary test
# above still applies. The relaxation still holds the pressure-changing part of ρθ in the
# boundary cells at its stage-entry value, so the small pressure-changing heating the interior
# takes over the step leaves a contrast of a few mK and winds of a few cm/s. Targeting the
# stage-entry variable with its content (or taking the content off a target without a wall
# value) gives contrasts of tens of K and winds of tens of m/s.
@testset "Open boundaries allow sedimentation content without a prescribed thermodynamic value [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size=(3, 3, 2), x=(0, 3000), y=(0, 3000), z=(0, 20),
                           topology=(Bounded, Bounded, Bounded))
    rain(x, y, z) = ifelse(z > 10, FT(0.01), FT(0.001))
    ε = FT(0.005)

    for (relaxation, bc) in ((FT(0.5), nothing), (FT(1), GradientBoundaryCondition(FT(0))), (FT(1), FluxBoundaryCondition(FT(0))))
        thermodynamic_bcs = isnothing(bc) ? (;) : (; ρθ=FieldBoundaryConditions(west=bc, east=bc, south=bc, north=bc))
        boundary_conditions = (; ρu=FieldBoundaryConditions(west=NormalFlowBoundaryCondition(FT(0)),
                                                            east=NormalFlowBoundaryCondition(FT(0))),
                                 ρv=FieldBoundaryConditions(south=NormalFlowBoundaryCondition(FT(0)),
                                                            north=NormalFlowBoundaryCondition(FT(0))),
                                 thermodynamic_bcs...)
        model = rain_model(FT, grid; fall_speed = 8, closed_bottom = true, advection = adaptive_upwind(FT),
                           open_boundary_relaxation = relaxation, boundary_conditions)
        set!(model; ρᵈ=FT(1.2), T=FT(290), qʳ=rain, qᵛ=(x, y, z) -> ε * (1 - rain(x, y, z)) / (1 + ε))
        initial_ρθ = Array(interior(AM.thermodynamic_density(model.formulation)))
        time_step!(model, FT(5))

        T = Array(interior(model.temperature))
        horizontal_spread = maximum(T; dims=(1, 2)) .- minimum(T; dims=(1, 2))
        change = Array(interior(AM.thermodynamic_density(model.formulation))) .- initial_ρθ
        @test maximum(horizontal_spread) < 0.02
        @test maximum(abs, Array(interior(model.velocities.u))) < 0.1
        @test maximum(abs, Array(interior(model.velocities.v))) < 0.1
        @test minimum(abs, change) > 1 # the content reaches every column, including all four sides
    end
end

# The moisture and tracer solves run while the previous stage's content is out of the live
# thermodynamic variable. Their coefficients read the variable from its stage-entry copy, so a
# diffusivity that depends on it diffuses the rain as with the content in the variable, not as
# from the reduced variable.
@testset "The tracer solves' coefficients read the full thermodynamic variable [$FT]" for FT in test_float_types()
    fixture = postloop_operator_fixture(FT)
    c = FT(30)

    # The rain the solves leave with the content removed from the variable or not, and with the
    # stage-entry copy overwritten by the reduced variable or not.
    function solved_rain(; remove, stale)
        model = postloop_model(fixture; closure=fixture.closures[2], temperature = z -> 290 - z)
        TS.store_initial_state!(model)
        foreach(G -> fill!(G, 0), model.timestepper.Gⁿ)
        interior(sedimentation_content(model)) .= c
        TS.remove_sedimentation_content!(model, remove ? 2 : 1)
        stale && (parent(frozen_thermodynamic_density(model)) .= parent(AM.thermodynamic_density(model.formulation)))
        TS.scalar_rk3_substep!(model, FT(1))
        return sedimentation_column_values(model.microphysical_fields.ρqʳ)
    end

    full = solved_rain(; remove = false, stale = false)
    @test solved_rain(; remove = true, stale = false) == full
    @test solved_rain(; remove = true, stale = true) != full
end

# The post-loop solves of the variable and the content run only when something about the
# variable is vertically implicit: closure diffusion or an implicit-explicit boundary flux, not a
# solver the moisture and tracers alone call for; without it the variable stays as the loop left
# it. When they run, the copy they read their coefficients from is taken after the content's
# addition refilled the variable's halos, so a coefficient reading a neighboring cell sees the
# full variable there too.
@testset "The post-loop solves run only with something implicit about the variable, and read it with fresh halos [$FT]" for FT in test_float_types()
    fixture = postloop_operator_fixture(FT)
    explicit = ScalarDiffusivity(; κ=1)
    implicit = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=1)

    tracers_only = postloop_model(fixture; closure=explicit, temperature = z -> 290 - z)
    @test !isnothing(tracers_only.timestepper.implicit_solver)
    @test !isnothing(TS.sedimentation_bypass(tracers_only))
    @test !TS.thermodynamic_variable_implicit(tracers_only)
    ρθ = AM.thermodynamic_density(tracers_only.formulation)
    ρθ⁰ = sedimentation_column_values(ρθ)
    TS.implicit_substep!(tracers_only, FT(1))
    @test sedimentation_column_values(ρθ) == ρθ⁰

    for kwargs in ((; closure=implicit), (; closure=explicit, fixture.boundary_conditions))
        model = postloop_model(fixture; kwargs...)
        @test TS.thermodynamic_variable_implicit(model)
        ρθ = AM.thermodynamic_density(model.formulation)
        interior(sedimentation_content(model)) .= FT(3)
        TS.apply_sedimentation_content!(model)
        @allowscalar @test ρθ[1, 1, 0] == ρθ[1, 1, 1]
        full_with_halos = Array(parent(ρθ))
        full = sedimentation_column_values(ρθ)
        TS.implicit_substep!(model, FT(1))
        @test Array(parent(frozen_thermodynamic_density(model))) == full_with_halos
        @test sedimentation_column_values(ρθ) != full
    end
end

# Through a column cooling with height, the falling rain carries its sensible heat down, a real
# heating that changes the pressure. It joins the slow tendency the substep loop integrates while
# only the composition content bypasses the loop, so at any time step the adaptive split reaches
# the temperature an explicit run reaches at a small one: within 0.6 % of the signal at Δt = 2
# and 3 % at Δt = 5.
@testset "Sedimenting rain carries its sensible heat at any time step on the acoustic stepper [$FT]" for FT in test_float_types()
    cooling_with_height(z) = 300 - z / 100
    reference = acoustic_sedimentation_column(FT, UpwindBiased(FT; order=1); Δt = 0.5, temperature = cooling_with_height)
    signal = maximum(abs, reference.temperature .- reference.T⁰)
    @test reference.initial_pressure_spread < 10eps(FT)
    @test signal > 0.1
    for Δt in (2, 5)
        column = acoustic_sedimentation_column(FT, adaptive_upwind(FT); Δt, temperature = cooling_with_height)
        @test maximum(abs, column.temperature .- reference.temperature) < signal / 10
    end
end
