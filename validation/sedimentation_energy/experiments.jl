# Transport-only probes; no production thermodynamic methods are replaced.
using Breeze, Oceananigans
using Oceananigans.TimeSteppers: update_state!, implicit_step!, time_discretization
using Oceananigans.Fields: compute!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!
using Breeze.AtmosphereModels: @kernel, @index
using Printf, Test

const AM = Breeze.AtmosphereModels
const TH = Breeze.Thermodynamics
const PT = Breeze.PotentialTemperatureFormulations
const SE = Breeze.StaticEnergyFormulations

# A test-only scheme: one condensate, prescribed velocity, no phase changes/sources.
struct TransportOnly{P, FT}
    phase :: P
    fall_speed :: FT
    closed_bottom :: Bool
end

AM.prognostic_field_names(::TransportOnly) = (:ρqʳ,)
AM.moisture_prognostic_name(::TransportOnly) = :ρqᵛ
AM.condensate_phase(m::TransportOnly, ::Val{:ρqʳ}) = m.phase
AM.sedimentation_velocity(::TransportOnly, μ, ::Val{:ρqʳ}) = μ.wʳ
AM.compute_microphysical_tendencies!(::TransportOnly, model) = nothing
AM.microphysics_model_update!(::TransportOnly, model) = nothing
@inline AM.maybe_adjust_thermodynamic_state(state, ::TransportOnly, vapor, constants) = state
@inline AM.microphysical_state(::TransportOnly, ρ, μ::NamedTuple, state, velocities) = (; qʳ=μ.ρqʳ / ρ)
@inline AM.moisture_fractions(::TransportOnly{Val{:liquid}}, state::NamedTuple, vapor) = TH.MoistureMassFractions(vapor, state.qʳ, oftype(vapor, 0))
@inline AM.moisture_fractions(::TransportOnly{Val{:ice}}, state::NamedTuple, vapor) = TH.MoistureMassFractions(vapor, oftype(vapor, 0), state.qʳ)
AM.liquid_mass_fraction(::TransportOnly{Val{:liquid}}, model) = model.microphysical_fields.qʳ
AM.liquid_mass_fraction(::TransportOnly{Val{:ice}}, model) = nothing
AM.ice_mass_fraction(::TransportOnly{Val{:ice}}, model) = model.microphysical_fields.qʳ
AM.ice_mass_fraction(::TransportOnly{Val{:liquid}}, model) = nothing

function AM.materialize_microphysical_fields(::TransportOnly, grid, bcs)
    return (; ρqʳ=CenterField(grid; boundary_conditions=bcs.ρqʳ),
              qʳ=CenterField(grid), qᵛ=CenterField(grid), wʳ=AM.sedimentation_velocity_field(grid))
end

@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, m::TransportOnly, state, ρ, thermo, constants)
    @inbounds μ.qʳ[i, j, k] = state.qʳ
    @inbounds μ.qᵛ[i, j, k] = thermo.moisture_mass_fractions.vapor
    speed = ifelse(m.closed_bottom & (k == 1), 0, -m.fall_speed)
    @inbounds μ.wʳ[i, j, k] = speed
    return nothing
end

function make_column(; core=:anelastic, formulation=:LiquidIcePotentialTemperature,
                       phase=:liquid, implicit=false, Nz=2, dz=20.0, speed=8.0,
                       closed=false, upper_temperature=280.0, equal_composition=false,
                       resolved_velocity=0.0, FT=Float64, arch=CPU(), scheme_kind=:upwind,
                       smooth_profile=false, bounds=(0, 1), horizontal_size=1)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(arch, FT; size=(horizontal_size, horizontal_size, Nz), x=(0, 1), y=(0, 1), z=(0, Nz * dz))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    reference = ReferenceState(grid, constants; surface_pressure=1e5, potential_temperature=280)
    dynamics = if core === :anelastic
        AnelasticDynamics(reference)
    elseif core === :prescribed
        PrescribedDynamics(reference)
    elseif core === :acoustic
        CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=280)
    else
        CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=280)
    end
    scheme = implicit ? WENO(FT; order=3, time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5)) : UpwindBiased(FT; order=1)
    if !implicit && scheme_kind !== :upwind
        scheme = scheme_kind === :bounded ? WENO(FT; order=3, bounds=FT.(bounds)) : WENO(FT; order=3)
    end
    microphysics = TransportOnly(Val(phase), FT(speed), closed)
    model = AtmosphereModel(grid; dynamics, formulation, microphysics, thermodynamic_constants=constants,
                            scalar_advection=(; ρqʳ=scheme))
    condensate(x, y, z) = FT(ifelse(smooth_profile, 0.0055 + 0.0045 * cos(2π * (z / dz - Nz / 2 - 0.5) / Nz),
                                  ifelse(equal_composition, 0.001, ifelse(z > Nz * dz / 2, 0.01, 0.001))))
    temperature(x, y, z) = FT(ifelse(z > Nz * dz / 2, upper_temperature, 280.0))
    if core in (:compressible, :acoustic)
        Rv = TH.vapor_gas_constant(constants)
        Rd = TH.dry_air_gas_constant(constants)
        density(x, y, z) = FT(1e5) / (((1 - FT(0.005) - condensate(x, y, z)) * Rd + FT(0.005) * Rv) * temperature(x, y, z))
        set!(model; ρ=density, T=temperature, qᵛ=FT(0.005), qʳ=condensate)
    else
        set!(model; T=temperature, qᵛ=FT(0.005), qʳ=condensate)
    end
    if resolved_velocity != 0
        set!(model; w=(x, y, z) -> FT(ifelse(0 < z < Nz * dz, resolved_velocity, 0)), enforce_mass_conservation=false)
    end
    update_state!(model)
    return model
end

host_column(array) = vec(sum(Array(array); dims=(1, 2))) ./ (size(array, 1) * size(array, 2))
column(field) = host_column(interior(field))

# Measure actual 0.111/PR reconstructions; do not infer limiter activation from bounds alone.
@kernel function _reconstruction_changes!(limiter_change, high_order_change, grid, scheme, q)
    i, j, k = @index(Global, NTuple)
    raw = Oceananigans.Advection._biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, Oceananigans.Advection.RightBias, q)
    _, limited, _, _ = Breeze.Advection.bounded_face_reconstructions(i, j, k, grid, scheme, q)
    @inbounds limiter_change[i, j, k] = abs(limited - raw)
    @inbounds high_order_change[i, j, k] = abs(raw - q[i, j, k])
end

function reconstruction_changes(model)
    limiter_change = CenterField(model.grid)
    high_order_change = CenterField(model.grid)
    scheme = Oceananigans.Advection.vertical_scheme(model.advection.ρqʳ)
    launch!(model.grid.architecture, model.grid, :xyz, _reconstruction_changes!,
            limiter_change, high_order_change, model.grid, scheme, model.microphysical_fields.qʳ)
    return (; limiter_change=Float64.(column(limiter_change)), high_order_change=Float64.(column(high_order_change)))
end

@kernel function _sedimentation_tendency!(G, grid, constituents, w, dynamics, callback, args)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] = -AM.condensate_sedimentation_divergence(i, j, k, grid, constituents,
        w, dynamics, AM.ExplicitSedimentationFluxes(), callback, args...)
end

function thermal_tendency(model; corrected=false, volume_corrected=false)
    G = CenterField(model.grid)
    if model.formulation isa PT.LiquidIcePotentialTemperatureFormulation
        callback = corrected ? phase_enthalpy_coefficients : PT.potential_temperature_condensate_content
        callback = volume_corrected ? phase_enthalpy_volume_coefficients : callback
        args = (model.formulation, model.dynamics, model.thermodynamic_constants,
                model.microphysics, model.microphysical_fields, AM.specific_prognostic_moisture(model))
    else
        callback = SE.static_energy_condensate_content
        args = (model.dynamics, model.thermodynamic_constants, model.microphysics,
                model.microphysical_fields, AM.specific_prognostic_moisture(model), model.temperature)
    end
    launch!(model.grid.architecture, model.grid, :xyz, _sedimentation_tendency!, G, model.grid,
            model.sedimentation_constituents, model.velocities.w, model.dynamics, callback, args)
    return G
end

@inline function phase_enthalpy_volume_coefficients(i, j, k, grid, formulation, dynamics, constants, microphysics, μ, vapor)
    content = phase_enthalpy_coefficients(i, j, k, grid, formulation, dynamics, constants, microphysics, μ, vapor)
    state = PT.grid_thermodynamic_state(i, j, k, grid, formulation, dynamics, microphysics, μ, vapor)
    T = TH.temperature(state, constants)
    q = state.moisture_mass_fractions
    cp = TH.mixture_heat_capacity(q, constants)
    gas_constant = TH.mixture_gas_constant(q, constants)
    latent = (constants.liquid.reference_latent_heat * q.liquid + constants.ice.reference_latent_heat * q.ice) / cp
    b = 1 - gas_constant / cp * (1 - latent / T)
    beta = b / ((cp - gas_constant) * TH.exner_function(state, constants))
    return (; χ=content.χ, h=content.h, ∂φ∂h=beta)
end

@inline function phase_enthalpy_coefficients(i, j, k, grid, formulation, dynamics, constants, microphysics, μ, vapor)
    original = PT.potential_temperature_condensate_content(i, j, k, grid, formulation, dynamics, constants, microphysics, μ, vapor)
    state = PT.grid_thermodynamic_state(i, j, k, grid, formulation, dynamics, microphysics, μ, vapor)
    T = TH.temperature(state, constants)
    liquid = constants.liquid.heat_capacity * T - constants.liquid.reference_latent_heat
    ice = constants.ice.heat_capacity * T - constants.ice.reference_latent_heat
    return (; χ=original.χ, h=(liquid, ice), ∂φ∂h=original.∂φ∂h)
end

function rain_tendency(model)
    AM.compute_tendencies!(model)
    with_fall = copy(interior(model.timestepper.Gⁿ.ρqʳ))
    set!(model.microphysical_fields.wʳ, 0)
    AM.compute_tendencies!(model)
    without_fall = copy(interior(model.timestepper.Gⁿ.ρqʳ))
    update_state!(model)
    return with_fall .- without_fall
end

function explicit_case(; Δt=1e-4, corrected=false, kwargs...)
    model = make_column(; implicit=false, kwargs...)
    Δt = eltype(model.grid)(Δt)
    T0 = column(model.temperature)
    Gq = rain_tendency(model)
    G = thermal_tendency(model; corrected)
    interior(AM.thermodynamic_density(model.formulation)) .+= Δt .* interior(G)
    interior(model.microphysical_fields.ρqʳ) .+= Δt .* Gq
    update_state!(model; compute_tendencies=false)
    return (; temperature_rate=(column(model.temperature) .- T0) ./ Δt,
              mass_rate=host_column(Gq), model)
end

function state_arrays(model)
    constants = model.thermodynamic_constants
    ρ = Float64.(column(AM.total_density(model.dynamics)))
    ρv = Float64.(column(model.moisture_density))
    ρr = Float64.(column(model.microphysical_fields.ρqʳ))
    T = Float64.(column(model.temperature))
    phase = model.microphysics.phase
    properties = phase isa Val{:liquid} ? constants.liquid : constants.ice
    cd = constants.dry_air.heat_capacity
    cv = constants.vapor.heat_capacity
    cx = properties.heat_capacity
    L = properties.reference_latent_heat
    C = (ρ .- ρv .- ρr) .* cd .+ ρv .* cv .+ ρr .* cx
    E = C .* T .- ρr .* L # gravity is zero in these experiments
    return (; ρ, ρv, ρr, T, C, E, cd, cv, cx, L)
end

function solve_rain!(model, Δt)
    μ = model.microphysical_fields
    implicit_step!(μ.ρqʳ, model.timestepper.implicit_solver, model.closure, model.closure_fields,
        AM.closure_scalar_index(model, :ρqʳ), model.clock, Oceananigans.fields(model), Δt,
        AM.implicit_step_scheme(model.advection.ρqʳ),
        AM.implicit_advection_velocities(model.dynamics, model.velocities, :ρqʳ, model.microphysics, μ),
        AM.implicit_advection_density(model.dynamics, model.formulation, :ρqʳ))
end

function implicit_case(; Δt=10.0, formulation=:LiquidIcePotentialTemperature, phase=:liquid,
                        upper_temperature=280.0, closed=false, FT=Float64, arch=CPU())
    model = make_column(; implicit=true, formulation, phase, upper_temperature, closed, FT, arch)
    Δt = FT(Δt)
    time_discretization(model.advection.ρqʳ).Δt[] = Δt
    initial = state_arrays(model)
    solve_rain!(model, Δt)
    solved_rain = column(model.microphysical_fields.ρqʳ)

    # Independent backward-Euler mass and finite energy balances, for uniform rho.
    implicit_speed = max(0, 8.0 - 0.5 * 20.0 / Δt)
    bottom_mask = vcat(closed ? 0.0 : 1.0, ones(length(solved_rain) - 1))
    mass_faces = vcat(-implicit_speed .* solved_rain .* bottom_mask, 0.0)
    energy_faces = mass_faces .* vcat((initial.cx - initial.cd) .* initial.T .- initial.L, 0.0)
    predicted_rain = initial.ρr .- Δt .* diff(mass_faces) ./ 20.0
    predicted_energy = initial.E .- Δt .* diff(energy_faces) ./ 20.0
    new_capacity = initial.C .+ (solved_rain .- initial.ρr) .* (initial.cx - initial.cd)
    reference_temperature = (predicted_energy .+ solved_rain .* initial.L) ./ new_capacity
    @test solved_rain ≈ predicted_rain rtol=100eps(FT) atol=0

    AM.implicit_sedimentation_step!(model, Δt, model.velocities)
    update_state!(model; compute_tendencies=false)
    final = state_arrays(model)
    return (; Δt, courant=8 * Δt / 20, formulation=string(formulation), phase=string(phase),
              initial_temperature=initial.T, temperature=final.T, reference_temperature,
              temperature_error=final.T .- reference_temperature,
              initial_column_mass=sum(initial.ρr) * 20,
              initial_column_thermal_energy=sum(initial.E) * 20,
              mass_residual=sum(final.ρr .- initial.ρr) * 20 - Δt * mass_faces[1],
              energy_residual=sum(final.E .- initial.E) * 20 - Δt * energy_faces[1])
end

function coupled_case(; Δt, steps, implicit=false, speed=8.0, FT, arch)
    model = make_column(; core=:acoustic, implicit, Nz=8, closed=true, speed, FT, arch)
    initial = state_arrays(model)
    times = [0.0]
    profiles = [initial.T]
    maximum_drift = [0.0]
    mass_changes = [0.0]
    for n in 1:steps
        time_step!(model, FT(Δt))
        state = state_arrays(model)
        @test all(isfinite, state.T)
        push!(times, n * Δt)
        push!(profiles, state.T)
        push!(maximum_drift, maximum(abs.(state.T .- initial.T)))
        push!(mass_changes, sum(state.ρr .- initial.ρr) * 20)
    end
    return (; Δt, steps, implicit, speed, times, profiles, maximum_drift, mass_changes,
              maximum_velocity=maximum(abs, column(model.velocities.w)))
end
