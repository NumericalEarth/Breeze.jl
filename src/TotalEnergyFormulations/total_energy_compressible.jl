#####
##### Compressible dynamics with a prognostic total energy
#####

const CompressibleTotalEnergyModel = AtmosphereModel{<:CompressibleDynamics, <:TotalEnergyFormulation}

"""
$(TYPEDSIGNATURES)

Diagnose temperature, pressure, the advected total enthalpy per unit dry mass
``Ĥ = (ρE + p) / ρᵈ`` and the diffused static energy ``ŝ = cᵖᵐ T + g z + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ``
of a compressible total-energy model. The temperature is the closed form
``T = (𝓊 - ℒˡ₀ qᵛ - (ℒˡ₀ - ℒⁱ₀) qⁱ) / cᵛᵐ`` at the moisture mass fractions left by any saturation
adjustment, and the pressure ``p = ρ Rᵐ T``.
"""
function AtmosphereModels.compute_auxiliary_dynamics_variables!(model::CompressibleTotalEnergyModel)
    grid = model.grid
    dynamics = model.dynamics
    formulation = model.formulation

    launch!(grid.architecture, grid, :xyz, _compute_total_energy_auxiliaries!,
            model.temperature, dynamics.pressure, formulation, dynamics, grid,
            specific_prognostic_moisture(model), model.microphysics, model.microphysical_fields,
            model.thermodynamic_constants)

    fill_halo_regions!(model.temperature)
    fill_halo_regions!(dynamics.pressure)
    fill_halo_regions!(formulation)

    return nothing
end

@kernel function _compute_total_energy_auxiliaries!(temperature_field, pressure_field, formulation, dynamics, grid,
                                                    specific_prognostic_moisture, microphysics,
                                                    microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = dynamics.total_density[i, j, k]
        ρᵈ = dynamics.dry_density[i, j, k]
        ρE = formulation.energy_density[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    𝒰 = AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
    T = temperature(𝒰, constants)
    p = ρ * mixture_gas_constant(q, constants) * T

    g = constants.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)
    ŝ = mixture_heat_capacity(q, constants) * T + g * z + liquid_referenced_latent_energy(q, constants)

    @inbounds begin
        temperature_field[i, j, k] = T
        pressure_field[i, j, k] = p
        formulation.total_enthalpy[i, j, k] = (ρE + p) / ρᵈ
        formulation.static_energy[i, j, k] = ŝ
    end
end

#####
##### Acoustic linearization
#####
#
# With the dry density ρᵈ, the momentum 𝐦 = ρᵈ𝐮 and the water densities ρˣ held fixed, as in the
# acoustic loop, the pressure p = ρRᵐT closes on ρE through
#
#   ρE = ρcᵛᵐ T + Λ + ρK + ρ g z,   Λ = ℒˡ₀ ρqᵛ + (ℒˡ₀ − ℒⁱ₀) ρqⁱ,   K = |𝐦|² / (2ρᵈ²),
#
# with ρ = ρᵈ + Σρˣ, ρRᵐ = ρᵈRᵈ + ρqᵛRᵛ and ρcᵛᵐ = ρᵈcᵛᵈ + Σ ρˣ cᵛˣ. So p = C (ρE − ρK − ρgz − Λ) with
#
#   C = ∂p/∂(ρE) = Rᵐ / cᵛᵐ ,
#
# and, since ∂(ρK)/∂ρᵈ = K − 2ρK/ρᵈ at fixed momentum (the velocity falls as ρᵈ grows) and
# ∂(ρgz)/∂ρᵈ = gz,
#
#   ∂p/∂ρᵈ = Rᵈ T − C cᵛᵈ T + C [K (2ρ/ρᵈ − 1) − g z] ,
#   E = (∂p/∂ρᵈ) / C = (Rᵈ cᵛᵐ / Rᵐ − cᵛᵈ) T + K (2ρ/ρᵈ − 1) − g z .
#
# The perturbation of ρE is carried by the dry-mass flux with χ̂ = Ĥ = (ρE + p)/ρᵈ. In dry air at
# rest C (Ĥ + E) = (Rᵈ/cᵛᵈ) cᵖᵈ T = γRᵈT, the squared sound speed, as for ρθ.

@inline function CompressibleEquations.acoustic_linearization(i, j, k, grid, formulation::TotalEnergyFormulation,
                                                              dynamics, constants, microphysics, μ, qᵛ, velocities)
    @inbounds begin
        ρ = dynamics.total_density[i, j, k]
        ρᵈ = dynamics.dry_density[i, j, k]
        p = dynamics.pressure[i, j, k]
        qᵛᵉ = qᵛ[i, j, k]
        Ĥ = formulation.total_enthalpy[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, μ)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵛᵐ = mixture_isochoric_heat_capacity(q, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵛᵈ = constants.dry_air.heat_capacity - Rᵈ
    T = p / (ρ * Rᵐ)

    K = kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities)
    g = constants.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)

    C = Rᵐ / cᵛᵐ
    E = (Rᵈ * cᵛᵐ / Rᵐ - cᵛᵈ) * T + K * (2ρ / ρᵈ - 1) - g * z

    return C, Ĥ, E
end
