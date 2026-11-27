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
# The water densities are still frozen against *advection* within the loop: a parcel displaced
# across a gradient of rˣ carries its entropy but not its water, which perturbs the pressure by
# −C Σₓ Eˣ M′ ⋅ ∇rˣ. With b = 0, Eᵛ ≈ −7000 J/kg/K for boundary-layer vapor, and this term makes
# moist air with vapor decreasing upward strongly unstable to the loop; the default offset
# b ≈ −7000 J/kg/K reduces Eᵛ to a few hundred (see `MoistEntropyFormulation`).
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
