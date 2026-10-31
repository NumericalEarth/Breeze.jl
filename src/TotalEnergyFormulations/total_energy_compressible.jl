#####
##### Compressible dynamics with a prognostic total energy
#####

const CompressibleTotalEnergyModel = AtmosphereModel{<:FlatCompressibleDynamics, <:TotalEnergyFormulation}

"""
$(TYPEDSIGNATURES)

Diagnose temperature, pressure, the enthalpy of the dry air ``ĥᵈ = cᵖᵈ (T - T₀) + K + g z``, the
enthalpy ``eᵛᵉ`` of the prognostic moisture less ``K + g z``, and the diffused static energy
``ŝ = cᵖᵐ T + g z + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ`` of a compressible total-energy model, each measured from
the formulation's energy origin ``T₀`` (``ŝ`` less ``T₀ (qᵈ cᵖᵈ + qᵗ cˡ)``). The temperature is the closed
form ``T = (𝓊 - ℒˡ₀ qᵛ - (ℒˡ₀ - ℒⁱ₀) qⁱ) / cᵛᵐ`` at the moisture mass fractions left by any saturation
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
    T₀ = formulation.reference_temperature
    offset = energy_origin_offset(T₀, ρᵈ, ρ, constants)
    ŝ = ŝ₀ - offset / ρ

    # K + g z, as the internal energy was diagnosed with it
    @inbounds ℬ = (ρE + offset) / ρ - formulation.internal_energy[i, j, k]

    @inbounds begin
        temperature_field[i, j, k] = T
        pressure_field[i, j, k] = p
        formulation.dry_enthalpy[i, j, k] = constants.dry_air.heat_capacity * (T - T₀) + ℬ
        formulation.moisture_enthalpy[i, j, k] = moisture_enthalpy(q, qᵛᵉ, T, T₀, constants)
        formulation.static_energy[i, j, k] = ŝ
    end
end

# The fraction of the prognostic moisture qᵛᵉ that is vapor: all of it unless a saturation adjustment
# holds some of it as cloud.
@inline moisture_vapor_fraction(q, qᵛᵉ) = ifelse(qᵛᵉ > 0, min(one(qᵛᵉ), q.vapor / qᵛᵉ), one(qᵛᵉ))

# The enthalpy per unit mass of the prognostic moisture: that of vapor, and of the cloud a saturation
# adjustment holds in it, which is taken to have the phase partition of the cell's condensate.
@inline function moisture_enthalpy(q, qᵛᵉ, T, T₀, constants)
    λᵛ = moisture_vapor_fraction(q, qᵛᵉ)
    qᶜ = q.liquid + q.ice
    λˡ = ifelse(qᶜ > 0, q.liquid / max(qᶜ, eps(typeof(qᶜ))), one(qᶜ))
    hᵛ = phase_enthalpy(constants.vapor.heat_capacity, vapor_latent_energy(constants), T)
    hˡ = phase_enthalpy(constants.liquid.heat_capacity, zero(T), T)
    hⁱ = phase_enthalpy(constants.ice.heat_capacity, ice_latent_energy(constants), T)
    return λᵛ * hᵛ + (1 - λᵛ) * (λˡ * hˡ + (1 - λˡ) * hⁱ) - constants.liquid.heat_capacity * T₀
end

@inline phase_enthalpy(cˣ, ℓˣ, T) = cˣ * T + ℓˣ
@inline vapor_latent_energy(constants) = absolute_zero_latent_heat(constants, constants.liquid)
@inline ice_latent_energy(constants) = absolute_zero_latent_heat(constants, constants.liquid) -
                                       absolute_zero_latent_heat(constants, constants.ice)

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
#   C = ∂p/∂(ρE) = Rᵐ / cᵛᵐ ,
#
# The loop's mass flux moves the dry air, and the water does not move inside it: each species moves
# by its own flux in the stage's scalar update. (ρE)′ therefore carries only the dry air's enthalpy,
# χ̂ = ĥᵈ, and E is the derivative of p at fixed water densities. Since ∂(ρK)/∂ρᵈ = K − 2ρK/ρᵈ,
#
#   E = (∂p/∂ρᵈ) / C = Rᵈ T / C − cᵛᵈ T − g z + T₀ cᵖᵈ + K (2ρ/ρᵈ − 1) ,
#   C (ĥᵈ + E) = (1 + C) Rᵈ T + 2 C K ρ/ρᵈ ,
#
# the squared sound speed per unit dry mass of air compressed through motionless water, independent
# of the energy origin T₀ and of the water's latent energy. (Moving the water with the air would add
# Rᵛ rᵛ (1 + C) T + c²_water, about 3%, to the squared sound speed of moist air at 20 g/kg.)
#
# Carrying Ĥ = (ρE + p) / ρᵈ in the loop instead, as the sum of all constituents' enthalpies, would
# advect the water's energy through the loop without its water. The loop's stratification would then
# be the moist static energy's, which falls with height through the moist troposphere, with a growing
# acoustic-gravity mode that only the lagged update of the water corrects. That was issue #1093's
# mode for E, and its noise grew at any T₀ below about 770 K, where ℒˡ₀ − cˡ T₀ changes sign.
#
# The water still changes the pressure, by
#
#   Ĉˣ = (∂p/∂ρqˣ) / C = Rˣ T (1 + 1 / C) − ĥˣ ,   ĥˣ = hˣ + K + g z  (Rˣ = 0 for condensate),
#
# so that moving a mass of water with its own enthalpy changes the pressure by Rˣ T (1 + C): the
# latent energy, the origin and K + g z cancel, and condensate transport leaves the pressure alone.
# The stage's water update is known before the loop (its tendency does not depend on the loop), so
# the loop integrates p′/C = (ρE)′ + Eρ′ + Σ Ĉˣ (ρqˣ)′ with (ρqˣ)′ the prescribed rewind-plus-tendency
# water perturbation. Without this the loop would see the latent energy of a surface vapor flux, or
# of the water's advective flux divergence, without the vapor, and phase changes not at all, and
# each stage's state would be offset from the loop's until the next stage. This is carried by
# adding Σ Ĉˣ (ρqˣ)′ to the loop's (ρE)′, and removing its final value on recovery.

@inline function CompressibleEquations.acoustic_linearization(i, j, k, grid, formulation::TotalEnergyFormulation,
                                                              dynamics, constants, microphysics, μ, qᵛ, velocities)
    @inbounds begin
        ρ = dynamics.total_density[i, j, k]
        ρᵈ = dynamics.dry_density[i, j, k]
        p = dynamics.pressure[i, j, k]
        qᵛᵉ = qᵛ[i, j, k]
        ĥᵈ = formulation.dry_enthalpy[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, μ)
    Rᵐ = mixture_gas_constant(q, constants)
    C = Rᵐ / mixture_isochoric_heat_capacity(q, constants)
    T = p / (ρ * Rᵐ)

    K = kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities)
    g = constants.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)
    T₀ = formulation.reference_temperature
    cᵖᵈ = constants.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(constants)
    cᵛᵈ = cᵖᵈ - Rᵈ

    E = Rᵈ * T / C - cᵛᵈ * T - g * z + T₀ * cᵖᵈ + K * (2ρ / ρᵈ - 1)

    return C, ĥᵈ, E
end

#####
##### The stage's water update in the acoustic loop
#####

function CompressibleEquations.initialize_water_pressure_perturbation!(formulation::TotalEnergyFormulation,
                                                                       substepper, model, U⁰, Δt)
    grid = model.grid
    arch = grid.architecture
    W = formulation.water_pressure_perturbation
    fill!(parent(W), 0)
    ρE′ = substepper.density_potential_temperature_perturbation
    C = substepper.linearization_exner
    Gⁿ = model.timestepper.Gⁿ
    constants = model.thermodynamic_constants
    FT = eltype(grid)
    kernel_Δt = convert(FT, Δt)

    moisture_name = moisture_prognostic_name(model.microphysics)
    launch!(arch, grid, :xyz, _add_moisture_pressure_perturbation!,
            W, ρE′, Gⁿ.ρE, grid, kernel_Δt, C, formulation, model.temperature,
            specific_prognostic_moisture(model), model.dynamics, model.microphysics, model.microphysical_fields,
            getproperty(U⁰, moisture_name), getproperty(prognostic_fields(model), moisture_name),
            getproperty(Gⁿ, moisture_name), constants)

    add_condensate_pressure_perturbations!(condensate_field_names(model.microphysics), W, ρE′, Gⁿ, grid, kernel_Δt,
                                           formulation, model, U⁰)
    fill_halo_regions!(ρE′)
    return nothing
end

add_condensate_pressure_perturbations!(::Tuple{}, args...) = nothing

function add_condensate_pressure_perturbations!(names::Tuple, W, ρE′, Gⁿ, grid, Δt, formulation, model, U⁰)
    name = first(names)
    launch!(grid.architecture, grid, :xyz, _add_condensate_pressure_perturbation!,
            W, ρE′, Gⁿ.ρE, Δt, formulation, model.temperature,
            condensate_enthalpy(model.microphysics, Val(name), formulation.reference_temperature,
                                model.thermodynamic_constants),
            getproperty(U⁰, name), getproperty(prognostic_fields(model), name), getproperty(Gⁿ, name),
            model.thermodynamic_constants)
    return add_condensate_pressure_perturbations!(Base.tail(names), W, ρE′, Gⁿ, grid, Δt, formulation, model, U⁰)
end

# K + g z = ĥᵈ − cᵖᵈ (T − T₀)
@inline mechanical_energy(ĥᵈ, T, T₀, constants) = ĥᵈ - constants.dry_air.heat_capacity * (T - T₀)

@kernel function _add_moisture_pressure_perturbation!(W, ρE′, GρE, grid, Δt, C, formulation, temperature_field,
                                                      specific_moisture, dynamics, microphysics, μ,
                                                      ρq⁰, ρq, Gρq, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = temperature_field[i, j, k]
        ĥᵛᵉ = mechanical_energy(formulation.dry_enthalpy[i, j, k], T, formulation.reference_temperature, constants) +
              formulation.moisture_enthalpy[i, j, k]
        qᵛᵉ = specific_moisture[i, j, k]
        ρᵢ = dynamics.total_density[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵢ, qᵛᵉ, μ)
    Rᵛᵉ = moisture_vapor_fraction(q, qᵛᵉ) * vapor_gas_constant(constants)
    @inbounds begin
        Ĉ = Rᵛᵉ * T * (1 + 1 / C[i, j, k]) - ĥᵛᵉ
        rewind = Ĉ * (ρq⁰[i, j, k] - ρq[i, j, k])
        ρE′[i, j, k] += rewind
        GρE[i, j, k] += Ĉ * Gρq[i, j, k]
        W[i, j, k] += rewind + Δt * Ĉ * Gρq[i, j, k]
    end
end

@kernel function _add_condensate_pressure_perturbation!(W, ρE′, GρE, Δt, formulation, temperature_field, enthalpy,
                                                        ρq⁰, ρq, Gρq, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = temperature_field[i, j, k]
        ĥˣ = mechanical_energy(formulation.dry_enthalpy[i, j, k], T, formulation.reference_temperature, constants) +
             enthalpy.heat_capacity * T + enthalpy.latent_energy
        Ĉ = - ĥˣ
        rewind = Ĉ * (ρq⁰[i, j, k] - ρq[i, j, k])
        ρE′[i, j, k] += rewind
        GρE[i, j, k] += Ĉ * Gρq[i, j, k]
        W[i, j, k] += rewind + Δt * Ĉ * Gρq[i, j, k]
    end
end

function CompressibleEquations.remove_water_pressure_perturbation!(formulation::TotalEnergyFormulation, substepper, model)
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _subtract!, formulation.energy_density, formulation.water_pressure_perturbation)
    return nothing
end

@kernel function _subtract!(a, b)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] -= b[i, j, k]
end
