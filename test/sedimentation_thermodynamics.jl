include(joinpath(@__DIR__, "setup.jl"))

using Breeze, Oceananigans, Test
using Oceananigans: fields
using Oceananigans.TimeSteppers: update_state!, time_step!, implicit_step!

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
    rain = sum(sedimentation_column_values(model.microphysical_fields.ρqʳ) .* diff(z))

    # The pressure and the velocity the run leaves, diagnosed by the last step's `update_state!`:
    # a spurious pressure drives motion, and the motion changes the pressure.
    p = sedimentation_column_values(model.dynamics.pressure)
    w = sedimentation_column_values(model.velocities.w)

    return (; model, initial_pressure_spread = spread(p⁰), pressure_spread = spread(p),
              largest_vertical_velocity = maximum(abs, w), largest_departure, rain_left = rain / rain⁰,
              temperature = sedimentation_column_values(model.temperature), T⁰)
end

@testset "Sedimentation keeps an isothermal column's temperature on the acoustic stepper [$FT]" for FT in test_float_types()
    adaptive = UpwindBiased(FT; order=1, time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5))

    # The fixture starts at rest at a uniform pressure and stays there, to roundoff, while
    # nothing falls.
    resting = acoustic_sedimentation_column(FT, adaptive; fall_speed = 0)
    @test resting.initial_pressure_spread < 10eps(FT)
    @test resting.pressure_spread < 10eps(FT)
    @test resting.largest_vertical_velocity < 100eps(FT)
    @test resting.largest_departure < 100eps(FT) * maximum(resting.T⁰)

    # An isothermal column moves no sensible heat, so the temperature must not change; the rain
    # moves no gas, so the pressure must stay uniform and the column at rest. The spurious
    # pressure moved the column at meters per second; the splitting leaves, in Float64 at Δt = 5,
    # a departure of 0.02 K, a relative pressure spread of 4e-6 and a velocity of 2e-4 m/s.
    for (scheme, Δt) in ((UpwindBiased(FT; order=1), 1), (adaptive, 2), (adaptive, 5))
        column = acoustic_sedimentation_column(FT, scheme; Δt)
        @test column.model.timestepper isa Breeze.TimeSteppers.AcousticRungeKutta3
        @test column.initial_pressure_spread < 10eps(FT)
        @test column.rain_left < 0.2 # most of the rain has fallen out of the column
        @test column.largest_departure < 0.05
        @test column.pressure_spread < 1e-4
        @test column.largest_vertical_velocity < 0.01
    end
end

# The thermodynamic variable's post-loop operator in full: the implicit part of a boundary flux,
# which reads the variable's boundary conditions and boundary values, and coefficients that read
# the variable, here from the cells on both sides of each face, which the variable's own solve
# overwrites as it sweeps up the column.
@inline diffusivity_of_ρθ(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields) =
    @inbounds ((fields.ρθ[i, j, k-1] + fields.ρθ[i, j, k]) / 600)^4 / 2

# A four-cell column without gravity for that operator: the closures to try, one of them the
# state-dependent diffusivity above, and an implicit flux −ρθ through the bottom, which halves the
# bottom cell over Δt = Δz = 1. The rain's adaptive scheme gives the model its implicit solver.
function postloop_operator_fixture(FT)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=4, z=(0, 4), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=290)
    adaptive = UpwindBiased(FT; order=1, time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5))
    boundary_conditions = (; ρθ=FieldBoundaryConditions(bottom=IMEXFluxBoundaryCondition(0, -1)))
    state_dependent = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=diffusivity_of_ρθ, discrete_form=true)
    microphysics = SedimentationOnly(Val(:liquid), FT(1), false)
    return (; grid, constants, dynamics, adaptive, boundary_conditions, microphysics, closures = (nothing, state_dependent))
end

# A model on that column with rain above 2 m, its state diagnosed and its stage-entry advecting
# state frozen, as the stage leaves it for the post-loop solves.
function postloop_model(fixture; temperature = 290, scalar_advection = (; ρqʳ = fixture.adaptive), kwargs...)
    model = AtmosphereModel(fixture.grid; fixture.dynamics, fixture.microphysics, scalar_advection,
                            thermodynamic_constants = fixture.constants, kwargs...)
    set!(model; ρᵈ=1.2, T=temperature, qᵛ=0.005, qʳ=z -> ifelse(z > 2, 0.001, 0.0))
    update_state!(model)
    Breeze.TimeSteppers.cache_advecting_state!(model)
    return model
end

# Oceananigans' solve of the thermodynamic variable with its post-loop operator, reading the live
# model fields, over Δt.
function oceananigans_postloop_solve!(model, Δt)
    ρθ = AM.thermodynamic_density(model.formulation)
    w, ρᵈ = Breeze.TimeSteppers.advecting_state(model)
    scheme = Breeze.TimeSteppers.postloop_thermodynamic_scheme(model.advection.ρθ, AM.dynamics_density(model.dynamics))
    implicit_step!(ρθ, model.timestepper.implicit_solver, model.closure, model.closure_fields,
                   AM.closure_scalar_index(model, :ρθ), model.clock, fields(model), Δt, scheme, (; w), ρᵈ)
    return Array(interior(ρθ, 1, 1, :))
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
        content = model.timestepper.sedimentation_content
        interior(content) .= s .* interior(ρθ)
        Breeze.TimeSteppers.apply_sedimentation_content!(model)
        Breeze.TimeSteppers.implicit_substep!(model, FT(1))

        δ = Array(interior(content, 1, 1, :))
        ρθ¹ = Array(interior(ρθ, 1, 1, :))
        @test δ ≈ s .* (ρθ¹ .- δ) rtol=100eps(FT)
    end
end

# The content's solve mirrors Oceananigans' `implicit_step!` call for call, with the variable's
# boundary conditions and boundary value in place of the content's and the frozen variable as the
# fields the coefficients read, while the variable itself takes Oceananigans' solve with those
# fields. A content equal to the variable must then solve to what the variable solves to. The
# thermodynamic scheme is adaptive implicit, so both solves leave the advective half to the loop.
@testset "The sedimentation content's solve is Oceananigans' implicit_step! of the thermodynamic variable [$FT]" for FT in test_float_types()
    fixture = postloop_operator_fixture(FT)

    for closure in fixture.closures
        model = postloop_model(fixture; closure, fixture.boundary_conditions, temperature = z -> 290 - z,
                               scalar_advection = (; ρθ = fixture.adaptive, ρqʳ = fixture.adaptive))
        ρθ = AM.thermodynamic_density(model.formulation)
        content = model.timestepper.sedimentation_content
        ρθ⁰ = Array(interior(ρθ, 1, 1, :))
        interior(content) .= interior(ρθ)
        Breeze.TimeSteppers.implicit_substep!(model, FT(1))
        ρθ¹ = Array(interior(ρθ, 1, 1, :))
        @test maximum(abs, ρθ¹ .- ρθ⁰) > 0
        @test Array(interior(content, 1, 1, :)) == ρθ¹
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
    content = model.timestepper.sedimentation_content
    interior(content) .= -FT(0.1) .* interior(ρθ)
    Breeze.TimeSteppers.apply_sedimentation_content!(model)
    full = Array(interior(ρθ, 1, 1, :))
    full_with_halos = copy(parent(ρθ))
    Breeze.TimeSteppers.implicit_substep!(model, FT(1))
    ρθ¹ = Array(interior(ρθ, 1, 1, :))
    expected = copy(full)
    expected[1] = full[1]^2 / (full[1] + J)
    @test ρθ¹ ≈ expected rtol=100eps(FT)
    @test !(ρθ¹ ≈ [full[1] / (1 + J / (full[1] / FT(0.9))); full[2:end]])

    parent(ρθ) .= full_with_halos
    @test oceananigans_postloop_solve!(model, FT(1)) ≈ ρθ¹ rtol=10eps(FT)
end

# The content comes out of the halos too, so that the reduced variable's halos are the full
# variable's less the content: the vertical operators the stage applies to the reduced variable
# see no step at the boundaries, and a wall value of the full variable is not held against the
# reduced interior.
@testset "Removing the sedimentation content takes it out of the halos too [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(4, 4), x=(0, 4), z=(0, 4), topology=(Bounded, Flat, Bounded))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=290)
    boundary_conditions = (; ρθ=FieldBoundaryConditions(west=ValueBoundaryCondition(FT(360))))
    model = AtmosphereModel(grid; dynamics, boundary_conditions, thermodynamic_constants=constants,
                            microphysics=SedimentationOnly(Val(:liquid), FT(1), false),
                            scalar_advection=(; ρqʳ=UpwindBiased(FT; order=1)))
    set!(model; ρᵈ=1.2, T=290, qᵛ=0.005, qʳ=0.001)
    update_state!(model)

    ρθ = AM.thermodynamic_density(model.formulation)
    content = model.timestepper.sedimentation_content
    c = FT(3)
    interior(content) .= c
    @allowscalar begin
        ρθ_cell⁰ = ρθ[1, 1, 1]
        step⁰ = ρθ[0, 1, 1] - ρθ[1, 1, 1]
        @test step⁰ / 2 == 360 - ρθ_cell⁰
        Breeze.TimeSteppers.remove_sedimentation_content!(model, 2)
        @test ρθ[1, 1, 1] == ρθ_cell⁰ - c
        @test ρθ[0, 1, 1] - ρθ[1, 1, 1] ≈ step⁰ rtol=10eps(FT)
    end
    @test all(iszero, Array(parent(content)))
end

# The content is kept out of the substep loop and added after it, so the loop's open-boundary
# relaxation must target the wall value for the full variable: the stage-entry variable's wall
# value less the content the stage adds. Targeting the base it reads would end each stage at
# the wall value plus this stage's content less the previous stage's.
@testset "The open-boundary relaxation lands the full thermodynamic variable on the wall value [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(4, 2), x=(0, 4000), z=(0, 200), topology=(Bounded, Flat, Bounded))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(; open_boundary_relaxation=1);
                                    reference_potential_temperature=290)
    v = FT(360)
    boundary_conditions = (; ρθ=FieldBoundaryConditions(west=ValueBoundaryCondition(v)),
                             ρu=FieldBoundaryConditions(west=NormalFlowBoundaryCondition(FT(0)),
                                                        east=NormalFlowBoundaryCondition(FT(0))))
    model = AtmosphereModel(grid; dynamics, boundary_conditions, thermodynamic_constants=constants,
                            microphysics=SedimentationOnly(Val(:liquid), FT(5), false),
                            scalar_advection=(; ρqʳ=UpwindBiased(FT; order=1)))
    set!(model; ρᵈ=1.2, T=290, qᵛ=0.005, qʳ=(x, z) -> ifelse(z > 100, FT(0.01), FT(0.001)))
    time_step!(model, 1)
    ρθ = AM.thermodynamic_density(model.formulation)
    content = model.timestepper.sedimentation_content
    @test all(!iszero, Array(interior(content, 1, 1, :)))
    @test Array(interior(ρθ, 1, 1, :)) ≈ fill(v, 2) rtol=100eps(FT)
end

# The moisture and tracer solves run while the previous stage's content is out of the live
# thermodynamic variable. Their coefficients read the variable from its stage-entry copy, so a
# diffusivity that depends on it diffuses the rain as with the content in the variable, not as
# from the reduced variable.
@testset "The tracer solves' coefficients read the full thermodynamic variable [$FT]" for FT in test_float_types()
    fixture = postloop_operator_fixture(FT)
    c = FT(30)
    rain = NamedTuple{(:full, :reduced, :stale)}(map((:full, :reduced, :stale)) do case
        model = postloop_model(fixture; closure=fixture.closures[2], temperature = z -> 290 - z)
        ρθ = AM.thermodynamic_density(model.formulation)
        for (u⁰, u) in zip(model.timestepper.U⁰, Oceananigans.prognostic_fields(model))
            parent(u⁰) .= parent(u)
        end
        foreach(G -> fill!(G, 0), model.timestepper.Gⁿ)
        Breeze.TimeSteppers.freeze_thermodynamic_density!(model)
        if case != :full
            interior(model.timestepper.sedimentation_content) .= c
            Breeze.TimeSteppers.remove_sedimentation_content!(model, 2)
        end
        case == :stale && (parent(model.timestepper.frozen_thermodynamic_density) .= parent(ρθ))
        Breeze.TimeSteppers.scalar_rk3_substep!(model, FT(1))
        Array(interior(model.microphysical_fields.ρqʳ, 1, 1, :))
    end)
    @test rain.reduced == rain.full
    @test rain.stale != rain.full
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
    @test !isnothing(tracers_only.timestepper.sedimentation_content)
    @test !isnothing(tracers_only.timestepper.frozen_thermodynamic_density)
    @test !Breeze.TimeSteppers.thermodynamic_variable_implicit(tracers_only)
    ρθ = AM.thermodynamic_density(tracers_only.formulation)
    ρθ⁰ = Array(interior(ρθ, 1, 1, :))
    Breeze.TimeSteppers.implicit_substep!(tracers_only, FT(1))
    @test Array(interior(ρθ, 1, 1, :)) == ρθ⁰

    for kwargs in ((; closure=implicit), (; closure=explicit, fixture.boundary_conditions))
        model = postloop_model(fixture; kwargs...)
        @test Breeze.TimeSteppers.thermodynamic_variable_implicit(model)
        ρθ = AM.thermodynamic_density(model.formulation)
        interior(model.timestepper.sedimentation_content) .= FT(3)
        Breeze.TimeSteppers.apply_sedimentation_content!(model)
        @allowscalar @test ρθ[1, 1, 0] == ρθ[1, 1, 1]
        full_with_halos = Array(parent(ρθ))
        Breeze.TimeSteppers.implicit_substep!(model, FT(1))
        @test Array(parent(model.timestepper.frozen_thermodynamic_density)) == full_with_halos
        @test Array(interior(ρθ, 1, 1, :)) != full_with_halos[1, 1, :][end-1:end]
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
    @test reference.initial_pressure_spread < 10eps(FT)
    @test signal > 0.1
    for Δt in (2, 5)
        column = acoustic_sedimentation_column(FT, adaptive; Δt, temperature = cooling_with_height)
        @test maximum(abs, column.temperature .- reference.temperature) < signal / 5
    end
end
