#####
##### Compressible dynamics with a prognostic total energy
#####

const CompressibleTotalEnergyModel = AtmosphereModel{<:FlatCompressibleDynamics, <:TotalEnergyFormulation}

"""
$(TYPEDSIGNATURES)

Diagnose temperature, pressure, the advected total enthalpy per unit dry mass
``Ĥ = (ρE + p) / ρᵈ`` and the diffused static energy ``ŝ = cᵖᵐ T + g z + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ``
of a compressible total-energy model, both measured from the formulation's energy origin ``T₀``
(``ŝ`` less ``T₀ (qᵈ cᵖᵈ + qᵗ cˡ)``). The temperature is the closed form
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
    ŝ₀ = mixture_heat_capacity(q, constants) * T + g * z + liquid_referenced_latent_energy(q, constants)
    ŝ = ŝ₀ - energy_origin_offset(formulation.reference_temperature, ρᵈ, ρ, constants) / ρ

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
# With the momentum 𝐦 = ρᵈ𝐮 held fixed, as in the acoustic loop, the pressure p = ρRᵐT closes on
# ρE through
#
#   ρE = ρcᵛᵐ T + Λ + ρK + ρ g z − T₀ (ρᵈ cᵖᵈ + ρqᵗ cˡ),   Λ = ℒˡ₀ ρqᵛ + (ℒˡ₀ − ℒⁱ₀) ρqⁱ,   K = |𝐦|² / (2ρᵈ²),
#
# with ρ = ρᵈ + ρqᵗ, ρRᵐ = ρᵈRᵈ + ρqᵛRᵛ and ρcᵛᵐ = ρᵈcᵛᵈ + Σ ρqˣ cᵛˣ. So
#
#   C = ∂p/∂(ρE) = Rᵐ / cᵛᵐ .
#
# The perturbation mass flux of the loop moves the water with the dry air: the tracers' update applies
# the loop's mean mass flux to them, while (ρE)′ is carried by χ̂ = Ĥ = (ρE + p)/ρᵈ, which includes the
# water's energy. E is therefore the derivative of p along that motion, at fixed water per unit dry
# mass, δρqˣ = (ρqˣ/ρᵈ) δρᵈ. Since ∂(ρK)/∂ρᵈ = K − 2ρK/ρᵈ at fixed momentum, the derivative is
#
#   ∂p/∂ρᵈ|_{qˣ/qᵈ} = C (ρ/ρᵈ) [K − g z − ℒˡ₀ qᵛ − (ℒˡ₀ − ℒⁱ₀) qⁱ + T₀ (qᵈ cᵖᵈ + qᵗ cˡ)] ,
#
# the temperature terms cancelling exactly (ρRᵐ/C = ρcᵛᵐ), so that
#
#   E = (ρ/ρᵈ) [K − g z − ℒˡ₀ qᵛ − (ℒˡ₀ − ℒⁱ₀) qⁱ + T₀ (qᵈ cᵖᵈ + qᵗ cˡ)] ,
#   C (Ĥ + E) = (γᵐ Rᵐ T + 2 C K) / qᵈ ,
#
# the squared sound speed of moist air at frozen composition per unit dry mass (plus a kinetic term of
# order the Mach number squared), independent of the energy origin T₀. Holding the water densities
# fixed instead would add the latent energy the flux of Ĥ carries, about ℒˡ₀ qᵛ, to the pressure
# response without the water that carries it, raising the sound speed of moist air by ~10%, and would
# make the loop depend on T₀.

@inline function CompressibleEquations.acoustic_linearization(i, j, k, grid, formulation::TotalEnergyFormulation,
                                                              dynamics, constants, microphysics, μ, qᵛ, velocities)
    @inbounds begin
        ρ = dynamics.total_density[i, j, k]
        ρᵈ = dynamics.dry_density[i, j, k]
        qᵛᵉ = qᵛ[i, j, k]
        Ĥ = formulation.total_enthalpy[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, μ)
    C = mixture_gas_constant(q, constants) / mixture_isochoric_heat_capacity(q, constants)

    K = kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities)
    g = constants.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)
    T₀ = formulation.reference_temperature
    offset = energy_origin_offset(T₀, ρᵈ, ρ, constants) / ρ

    E = ρ / ρᵈ * (K - g * z - liquid_referenced_latent_energy(q, constants) + offset)

    return C, Ĥ, E
end
