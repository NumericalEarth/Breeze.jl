#####
##### MoistEntropyFormulation on CompressibleDynamics
#####

AtmosphereModels.materialize_formulation(formulation::MoistEntropyFormulation, ::CompressibleDynamics, grid, boundary_conditions) =
    MoistEntropyFormulations.materialize_moist_entropy_formulation(formulation, grid, boundary_conditions)

const MoistEntropyCompressibleModel = AtmosphereModel{<:CompressibleDynamics, <:MoistEntropyFormulation}

AtmosphereModels.initialize_model_thermodynamics!(model::MoistEntropyCompressibleModel) =
    MoistEntropyFormulations.validate_moist_entropy_microphysics(model.microphysics)

# Temperature in closed form from the entropy per unit total mass, η = ρη / ρ, at the total density
# and composition (see `temperature_from_moist_entropy`), and pressure from the ideal gas law.
@inline function temperature_and_pressure(i, j, k, grid, formulation::MoistEntropyFormulation,
                                          dynamics, ρᵈ, ρ, Rᵐ, q, constants)
    η = @inbounds formulation.entropy_density[i, j, k] / ρ
    η₀ = MoistEntropyFormulations.offset_free_entropy(formulation, η, q)
    T = temperature_from_moist_entropy(η₀, ρ, q, constants)
    return T, ρ * Rᵐ * T
end

#####
##### Acoustic linearization
#####
#
# Below is the entropy density's own pressure response, p′ = C[S′ + Σₓ Eˣ ρˣ′]. It defines the slow
# rate of the adiabatic pressure variable the loop runs on (see `adiabatic_pressure_variable`).
#
# The entropy density S = ρη, the dry density ρᵈ and the water densities ρˣ determine the temperature
# through
#
#   ln(T / Tᵗʳ) = (S + F) / Cᵛ ,   Cᵛ = Σ ρˣ cᵛˣ ,
#   F = ρᵈ Rᵈ ln(ρᵈ / ρᵈᵗʳ) + ρᵛ Rᵛ ln(ρᵛ / ρᵛᵗʳ) − ρᵛ ℒˡ(Tᵗʳ) / Tᵗʳ + ρⁱ ℒᶠ(Tᵗʳ) / Tᵗʳ − b ρʷ ,
#
# with ρʷ the total water density and b the water entropy offset, and the pressure through
# p = (ρᵈ Rᵈ + ρᵛ Rᵛ) T. Differentiating,
#
#   C = ∂p/∂S = p / Cᵛ = Rᵐ T / cᵛᵐ ,
#   Eᵈ = (∂p/∂ρᵈ) / C = Rᵈ cᵖᵐ / Rᵐ − ηᵈ ,   Eᵛ = (∂p/∂ρᵛ) / C = Rᵛ cᵖᵐ / Rᵐ − ηᵛ ,
#   Eˡ = −ηˡ ,   Eⁱ = −ηⁱ ,
#
# where ηˣ are the partial specific entropies, those of the water species including b. The acoustic
# loop evolves ρᵈ′ and S′ and leaves the water to the tracer substep that follows, which advances it
# with the same time-averaged dry-mass flux. The loop's pressure anticipates the compression of
# the water with the dry air by holding the mixing ratios rˣ = ρˣ / ρᵈ fixed rather than the
# water densities, E = Eᵈ + Σₓ Eˣ rˣ. For dry air E = cᵖᵈ − η and
# p′ = (R T / cᵛ)[S′ + (cᵖ − η) ρ′] = c² ρ′ + (p / cᵛ) η′, the familiar p′ = c² ρ′ + (∂p/∂η)_ρ η′.
# The flux of S carries χ̂ = S / ρᵈ per unit dry mass.
#
# A loop on (S′, ρᵈ′) freezes the water densities against *advection*: a parcel displaced across a
# gradient of rˣ would carry its entropy but not its water, which perturbs the pressure by
# −C Σₓ Eˣ M′ ⋅ ∇rˣ, and the slow tendency and the rewind of S would act without those of the water.
# With b = 0, Eᵛ ≈ −7000 J/kg/K for boundary-layer vapor; with b = −7000 J/kg/K, Eᵛ ≈ 0 and Eˡ ≈ 7000.
#
# S′ + E ρ′ = ρ η′ + (E + η) ρ′, where E + η ≈ cᵖ: with this entropy origin η ≈ −1300 J/kg/K near the
# ground and E ≈ 2400 J/kg/K, so for an acoustic perturbation (η′ ≈ 0) the two terms cancel to within
# a factor of about two, a loss of about one bit.
@inline function acoustic_linearization(i, j, k, grid, formulation::MoistEntropyFormulation,
                                        dynamics, constants, microphysics, μ, qᵛ, velocities)
    @inbounds begin
        ρ = dynamics.total_density[i, j, k]
        ρᵈ = dynamics.dry_density[i, j, k]
        p = dynamics.pressure[i, j, k]
        S = formulation.entropy_density[i, j, k]
        qᵛᵢ = qᵛ[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵢ, μ)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵛᵐ = cᵖᵐ - Rᵐ
    T = p / (ρ * Rᵐ)

    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵛᵈ = constants.dry_air.heat_capacity - Rᵈ
    Tᵗʳ = constants.triple_point_temperature
    ρᵈᵗʳ = constants.triple_point_pressure / (Rᵈ * Tᵗʳ)
    b = formulation.water_entropy_offset

    ηᵈ = cᵛᵈ * log(T / Tᵗʳ) - Rᵈ * log(ρᵈ / ρᵈᵗʳ)
    ηᵛ = vapor_entropy(T, ρ * q.vapor, constants) + b
    ηˡ = liquid_entropy(T, constants) + b
    ηⁱ = ice_entropy(T, constants) + b

    Eᵈ = Rᵈ * cᵖᵐ / Rᵐ - ηᵈ
    Eᵛ = Rᵛ * cᵖᵐ / Rᵐ - ηᵛ
    r = ρ / ρᵈ # rˣ = qˣ ρ / ρᵈ

    C = Rᵐ * T / cᵛᵐ
    E = Eᵈ + r * (Eᵛ * q.vapor - ηˡ * q.liquid - ηⁱ * q.ice)
    χ̂ = S / ρᵈ

    return C, χ̂, E
end

"""
$(TYPEDSIGNATURES)

Return `Y = (p / pˢᵗ)^(1 / γ) pˢᵗ / Rᵈ` and `C = ∂p/∂Y = γ p / Y`, with `γ = cᵖᵐ / cᵛᵐ` of the local
composition: the variable the acoustic loop of the moist entropy formulation evolves.

Adiabatic motion obeys `D ln p / Dt = γ D ln ρᵈ / Dt` at fixed composition, whatever the water does, so
`Θ = Y / ρᵈ` is carried by the air and

```math
∂ₜ Y + ∇ ⋅ (Θ ρᵈ 𝐮) = 0
```

with no reference to the entropy, its origin, or the water densities: the same loop as for ``ρθ`` (for
dry air `Y = ρθ`). The loop's `p′ = C Y′` therefore sees a displaced parcel's water with its entropy. The
rates of change of `ρη`, `ρᵈ` and the water that the loop does not resolve enter through `Gᵞ`
(`initialize_pressure_variable!`), which holds the water's tendency together with the entropy's.
"""
@inline function adiabatic_pressure_variable(p, cᵖᵐ, cᵛᵐ, pˢᵗ, Rᵈ)
    γ = cᵖᵐ / cᵛᵐ
    Y = (p / pˢᵗ)^(1 / γ) * pˢᵗ / Rᵈ
    return Y, γ * p / Y
end

function refresh_linearization_basic_state!(formulation::MoistEntropyFormulation, substepper, model)
    refresh_moist_entropy_linearization!(substepper.pressure_variable_loop, formulation, substepper, model)
    return nothing
end

refresh_moist_entropy_linearization!(::Nothing, formulation, substepper, model) =
    refresh_general_linearization!(formulation, substepper, model)

function refresh_moist_entropy_linearization!(loop::AdiabaticPressureLoop, formulation, substepper, model)
    grid = model.grid
    launch!(architecture(grid), grid, :xyz, _compute_adiabatic_pressure_linearization!,
            substepper.linearization_exner,
            substepper.linearization_potential_temperature,
            substepper.linearization_gamma_R_mixture,
            loop.flux_weight,
            grid, formulation, model.dynamics, model.thermodynamic_constants,
            model.microphysics, model.microphysical_fields, specific_prognostic_moisture(model))

    fill_halo_regions!(substepper.linearization_exner)
    fill_halo_regions!(substepper.linearization_potential_temperature)
    fill_halo_regions!(substepper.linearization_gamma_R_mixture)
    fill_halo_regions!(loop.flux_weight)
    return nothing
end

@kernel function _compute_adiabatic_pressure_linearization!(Cʸ, Θ, one_field, χ̂, grid, formulation, dynamics, constants,
                                                            microphysics, μ, qᵛ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ = dynamics.total_density[i, j, k]
        ρᵈ = dynamics.dry_density[i, j, k]
        p = dynamics.pressure[i, j, k]
        S = formulation.entropy_density[i, j, k]
        qᵛᵢ = qᵛ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵢ, μ)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵛᵐ = cᵖᵐ - mixture_gas_constant(q, constants)
    pˢᵗ = convert(typeof(p), dynamics.standard_pressure)
    Y, Cᴸ = adiabatic_pressure_variable(p, cᵖᵐ, cᵛᵐ, pˢᵗ, dry_air_gas_constant(constants))
    @inbounds begin
        Cʸ[i, j, k] = Cᴸ
        Θ[i, j, k] = Y / ρᵈ
        one_field[i, j, k] = 1
        χ̂[i, j, k] = S / ρᵈ
    end
end

# The water densities and tendencies of the loop's pressure, as (live density, outer-step density,
# tendency, liquid fraction): the vapor is separate, and the condensate masses are those with a liquid
# fraction (number moments and rime volume do not change the pressure at fixed masses).
water_mass_fields(model, U⁰, Gⁿ) = water_mass_fields(model.microphysics, prognostic_field_names(model.microphysics), model, U⁰, Gⁿ)
water_mass_fields(microphysics, ::Tuple{}, model, U⁰, Gⁿ) = ()

function water_mass_fields(microphysics, names::Tuple, model, U⁰, Gⁿ)
    name = first(names)
    rest = water_mass_fields(microphysics, Base.tail(names), model, U⁰, Gⁿ)
    fˡ = condensate_liquid_fraction(microphysics, Val(name))
    isnothing(fˡ) && return rest
    return ((model.microphysical_fields[name], U⁰[name], Gⁿ[name], fˡ), rest...)
end

# Σₓ Eˣ δρˣ and Σₓ Eˣ Gˣ over the condensate masses
@inline condensate_pressure_sums(i, j, k, ::Tuple{}, Eˡ, Eⁱ) = (0, 0)

@inline function condensate_pressure_sums(i, j, k, water::Tuple, Eˡ, Eⁱ)
    ρˣ, ρˣ⁰, Gˣ, fˡ = first(water)
    Eˣ = fˡ * Eˡ + (1 - fˡ) * Eⁱ
    δ, G = condensate_pressure_sums(i, j, k, Base.tail(water), Eˡ, Eⁱ)
    @inbounds return δ + Eˣ * (ρˣ⁰[i, j, k] - ρˣ[i, j, k]), G + Eˣ * Gˣ[i, j, k]
end

# The loop starts from the rewind to the outer-step state and is forced by Gᵞ; the water is advanced
# after the loop, but the pressure it will have is the pressure the loop should feel.
function initialize_pressure_variable!(loop::AdiabaticPressureLoop, substepper, model, U⁰)
    grid = model.grid
    Gⁿ = model.timestepper.Gⁿ
    moisture_name = moisture_prognostic_name(model.microphysics)
    launch!(architecture(grid), grid, :xyz, _initialize_pressure_variable!,
            substepper.density_potential_temperature_perturbation, loop.slow_tendency, loop.perturbation,
            substepper.linearization_exner,
            grid, model.formulation, model.dynamics, model.thermodynamic_constants, model.microphysics,
            model.microphysical_fields, specific_prognostic_moisture(model), model.velocities,
            U⁰.ρη, U⁰.ρᵈ, getproperty(U⁰, moisture_name), model.moisture_density,
            Gⁿ.ρη, Gⁿ.ρᵈ, getproperty(Gⁿ, moisture_name), water_mass_fields(model, U⁰, Gⁿ))
    fill_halo_regions!(substepper.density_potential_temperature_perturbation)
    return nothing
end

@kernel function _initialize_pressure_variable!(Y′, Gᵞ, S′, Cʸ, grid, formulation, dynamics, constants, microphysics,
                                                μ, qᵛ, velocities, S⁰, ρᵈ⁰, ρᵛ⁰, ρᵛᴸ, GS, Gρᵈ, Gρᵛ, water)
    i, j, k = @index(Global, NTuple)
    C, χ̂, _ = acoustic_linearization(i, j, k, grid, formulation, dynamics, constants, microphysics, μ, qᵛ, velocities)
    Eᵈ, Eᵛ, Eˡ, Eⁱ = entropy_pressure_weights(i, j, k, grid, formulation, dynamics, constants, microphysics, μ, qᵛ)
    δ, G = condensate_pressure_sums(i, j, k, water, Eˡ, Eⁱ)
    @inbounds begin
        weight = C / Cʸ[i, j, k]
        δS = S⁰[i, j, k] - formulation.entropy_density[i, j, k]
        S′[i, j, k] = δS
        Y′[i, j, k] = weight * (δS + Eᵈ * (ρᵈ⁰[i, j, k] - dynamics.dry_density[i, j, k]) +
                                Eᵛ * (ρᵛ⁰[i, j, k] - ρᵛᴸ[i, j, k]) + δ)
        Gᵞ[i, j, k] = weight * (GS[i, j, k] + Eᵈ * Gρᵈ[i, j, k] + Eᵛ * Gρᵛ[i, j, k] + G)
    end
end

# The partial derivatives of the pressure Eˣ = (∂p/∂ρˣ) / (∂p/∂S) of the block comment above
@inline function entropy_pressure_weights(i, j, k, grid, formulation, dynamics, constants, microphysics, μ, qᵛ)
    @inbounds begin
        ρ = dynamics.total_density[i, j, k]
        ρᵈ = dynamics.dry_density[i, j, k]
        p = dynamics.pressure[i, j, k]
        qᵛᵢ = qᵛ[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵢ, μ)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    T = p / (ρ * Rᵐ)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵛᵈ = constants.dry_air.heat_capacity - Rᵈ
    Tᵗʳ = constants.triple_point_temperature
    ρᵈᵗʳ = constants.triple_point_pressure / (Rᵈ * Tᵗʳ)
    b = formulation.water_entropy_offset
    ηᵈ = cᵛᵈ * log(T / Tᵗʳ) - Rᵈ * log(ρᵈ / ρᵈᵗʳ)
    return (Rᵈ * cᵖᵐ / Rᵐ - ηᵈ,
            Rᵛ * cᵖᵐ / Rᵐ - vapor_entropy(T, ρ * q.vapor, constants) - b,
            - liquid_entropy(T, constants) - b,
            - ice_entropy(T, constants) - b)
end
