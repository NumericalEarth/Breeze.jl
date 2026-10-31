#####
##### Dry potential temperature on CompressibleDynamics
#####

const DryPotentialTemperatureCompressibleModel = AtmosphereModel{<:CompressibleDynamics, <:DryPotentialTemperatureFormulation}

# The dry θ formulation is faithful to CM1, whose pressure-gradient force acts on the total mass
# (it uses the density potential temperature), so it weights the forces on the dry-coupled momentum.
function AtmosphereModels.materialize_dynamics(dynamics::CompressibleDynamics, grid, boundary_conditions,
                                               thermodynamic_constants, microphysics,
                                               ::DryPotentialTemperatureFormulation)
    materialized = AtmosphereModels.materialize_dynamics(dynamics, grid, boundary_conditions, thermodynamic_constants, microphysics)
    materialized.terrain_metrics === nothing ||
        throw(ArgumentError("DryPotentialTemperatureFormulation does not support terrain-following grids yet"))
    return with_momentum_force_weighting(materialized, MomentumForceWeighting())
end

AtmosphereModels.materialize_dynamics(dynamics::CompressibleDynamics, grid, boundary_conditions,
                                      thermodynamic_constants, microphysics, name::Symbol) =
    AtmosphereModels.materialize_dynamics(dynamics, grid, boundary_conditions, thermodynamic_constants, microphysics, Val(name))

AtmosphereModels.materialize_dynamics(dynamics::CompressibleDynamics, grid, boundary_conditions,
                                      thermodynamic_constants, microphysics, ::Val) =
    AtmosphereModels.materialize_dynamics(dynamics, grid, boundary_conditions, thermodynamic_constants, microphysics)

AtmosphereModels.materialize_dynamics(dynamics::CompressibleDynamics, grid, boundary_conditions,
                                      thermodynamic_constants, microphysics, ::Val{:DryPotentialTemperature}) =
    AtmosphereModels.materialize_dynamics(dynamics, grid, boundary_conditions, thermodynamic_constants, microphysics,
                                          DryPotentialTemperatureFormulation())

function AtmosphereModels.initialize_model_thermodynamics!(model::DryPotentialTemperatureCompressibleModel)
    microphysics = model.microphysics
    AtmosphereModels.dry_potential_temperature_compatible(microphysics) && return nothing
    throw(ArgumentError("DryPotentialTemperatureFormulation does not support $(summary(microphysics)). It needs " *
                        "microphysics that changes the water species through tendencies, or that updates the " *
                        "temperature itself (OperatorSplitSaturationAdjustment). Diagnostic saturation adjustment " *
                        "re-derives the condensate from a phase-invariant variable, which releases no heat in θ."))
end

#####
##### Temperature and pressure
#####

# Closed form of the equation of state: T = θ^γᵈ (ρ Rᵐ / pˢᵗ)^(γᵈ - 1), and p = ρ Rᵐ T. As CM1's
# Π = (ρᵈ Rᵈ θ (1 + rᵛ / ε) / pˢᵗ)^(Rᵈ / cᵛᵈ) (`solve1.F:1733`), since ρ Rᵐ = ρᵈ (Rᵈ + Rᵛ rᵛ).
@inline function temperature_and_pressure(i, j, k, grid, formulation::DryPotentialTemperatureFormulation,
                                          dynamics, ρᵈ, ρ, Rᵐ, q, constants)
    ρθ = @inbounds formulation.potential_temperature_density[i, j, k]
    θ = ρθ / ρᵈ
    𝒰 = DryDensityState(θ, q, standard_pressure(dynamics), ρ)
    T = temperature(𝒰, constants)
    return T, ρ * Rᵐ * T
end

@inline function AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid,
                                                               formulation::DryPotentialTemperatureFormulation,
                                                               dynamics::CompressibleDynamics, q)
    θ = @inbounds formulation.potential_temperature[i, j, k]
    ρ = @inbounds dynamics.total_density[i, j, k]
    return DryDensityState(θ, q, standard_pressure(dynamics), ρ)
end

#####
##### The expansion term is applied after the acoustic loop on the split-explicit core
#####

@inline PotentialTemperatureFormulations.explicit_velocity_divergence(i, j, k, grid,
                                                                      ::CompressibleDynamics{<:SplitExplicitTimeDiscretization},
                                                                      velocities) = 0

add_post_acoustic_loop_source!(formulation, substepper, model, Δt) = nothing

add_post_acoustic_loop_source!(formulation::DryPotentialTemperatureFormulation, substepper, model, Δt) =
    add_post_acoustic_loop_source!(formulation.flavor.pressure_work, formulation, substepper, model, Δt)

add_post_acoustic_loop_source!(::IsobaricPressureWork, formulation, substepper, model, Δt) = nothing

# CM1 applies Dθ/Dt ∋ − θ (Rᵐ − κᵈ cᵖᵐ) / cᵛᵐ ∇⋅⟨𝐮⟩ after its acoustic loop, with the coefficient of the
# stage-entry state and the velocity averaged over that loop (`solve2.F:410, 1278-1281, 1327`).
# The model fields are still the stage-entry state here, and adding the increment to the
# perturbation (ρθ)′ makes it part of the recovery of ρθ.
function add_post_acoustic_loop_source!(::IsochoricPressureWork, formulation, substepper, model, Δt)
    grid = model.grid
    launch!(architecture(grid), grid, :xyz, _add_divergence_expansion!,
            substepper.density_potential_temperature_perturbation,
            grid, convert(eltype(grid), Δt),
            substepper.time_averaged_velocities,
            formulation.potential_temperature,
            model.dynamics.dry_density,
            model.dynamics.total_density,
            model.microphysics,
            specific_prognostic_moisture(model),
            model.microphysical_fields,
            model.thermodynamic_constants)
    return nothing
end

@kernel function _add_divergence_expansion!(ρθ′, grid, Δt, ū, θ, ρᵈ, ρ, microphysics, qᵛᵉ, μ, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρₖ = ρ[i, j, k]
        q = grid_moisture_fractions(i, j, k, grid, microphysics, ρₖ, qᵛᵉ[i, j, k], μ)
        S = divergence_expansion_coefficient(θ[i, j, k], q, constants)
        ρθ′[i, j, k] -= Δt * ρᵈ[i, j, k] * S * divᶜᶜᶜ(i, j, k, grid, ū.u, ū.v, ū.w)
    end
end

#####
##### Acoustic pressure coefficient
#####

# The loop perturbs p′ = Cᴸ (ρθ)′ and moves (ρθ)′ with the flux θᴸ (ρ𝐮)′ under the force − qᵈ ∇p′, so
# its squared sound speed is qᵈ Cᴸ θᴸ. With the expansion term applied after the loop, CM1's value
# c² = γᵐ Rᵐ T (`sound.F`: cᵖ θ_ρ ppterm) needs Cᴸ = γᵐ Rᵐ Π / qᵈ, with Π = (p / pˢᵗ)^κᵈ and
# γᵐ = cᵖᵐ / cᵛᵐ, and the increment of the recovered ρθ, including the expansion source, then gives the
# same pressure through the equation of state, whose ∂p/∂(ρθ) is γᵈ Rᵐ Π / qᵈ, since
# θ + θ (Rᵐ − κᵈ cᵖᵐ) / cᵛᵐ = θ γᵐ / γᵈ.
#
# Without the source in the loop's accounting (`IsobaricPressureWork`) the coefficient is the
# derivative of the equation of state itself.
function compute_linearization_pressure_coefficient!(formulation::DryPotentialTemperatureFormulation, substepper, model)
    launch!(architecture(model.grid), model.grid, :xyz, _compute_dry_theta_pressure_coefficient!,
            substepper.linearization_gamma_R_mixture,
            model.grid,
            formulation.flavor.pressure_work,
            model.microphysics,
            model.dynamics.total_density,
            model.dynamics.dry_density,
            specific_prognostic_moisture(model),
            model.microphysical_fields,
            model.thermodynamic_constants)
    return nothing
end

@inline acoustic_heat_capacity_ratio(::IsochoricPressureWork, cᵖᵐ, cᵛᵐ, constants) = cᵖᵐ / cᵛᵐ

@inline function acoustic_heat_capacity_ratio(::IsobaricPressureWork, cᵖᵐ, cᵛᵐ, constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    return cᵖᵈ / (cᵖᵈ - dry_air_gas_constant(constants))
end

@kernel function _compute_dry_theta_pressure_coefficient!(γRᵐ, grid, pressure_work, microphysics, ρ, ρᵈ, qᵛ, μ, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds ρᵢ = ρ[i, j, k]
    @inbounds ρᵈᵢ = ρᵈ[i, j, k]
    @inbounds q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵢ, qᵛ[i, j, k], μ)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    γ = acoustic_heat_capacity_ratio(pressure_work, cᵖᵐ, cᵖᵐ - Rᵐ, constants)
    # Divide rather than multiply by qᵈ, so that dry air (ρ = ρᵈ) is exact; a state before `set!` has ρᵈ = 0
    @inbounds γRᵐ[i, j, k] = γ * Rᵐ * (ρᵢ / ifelse(ρᵈᵢ == 0, one(ρᵈᵢ), ρᵈᵢ))
end
