#####
##### Helper accessors
#####

AtmosphereModels.liquid_ice_potential_temperature(model::TotalEnergyModel) =
    Diagnostics.LiquidIcePotentialTemperature(model, :specific)
AtmosphereModels.liquid_ice_potential_temperature_density(model::TotalEnergyModel) =
    Diagnostics.LiquidIcePotentialTemperature(model, :density)
AtmosphereModels.static_energy(model::TotalEnergyModel) = Diagnostics.StaticEnergy(model, :specific)

"""
$(TYPEDSIGNATURES)

Return the static energy density of a total-energy model as a `Field` whose boundary conditions
return the energy fluxes supplied for `ρE` — the sensible heat flux — without the enthalpy of the
water crossing the walls, which the prognostic `ρE` also receives (see
`BoundaryConditions.VaporEnthalpyFluxFunction`).
"""
function AtmosphereModels.static_energy_density(model::TotalEnergyModel)
    ρE_bcs = model.formulation.energy_density.boundary_conditions
    return Field(Diagnostics.StaticEnergy(model, :density); boundary_conditions=energy_boundary_conditions(ρE_bcs))
end

#####
##### Tendency computation
#####

function AtmosphereModels.compute_thermodynamic_tendency!(model::TotalEnergyModel, common_args, tracer_transport_velocity)
    grid = model.grid
    arch = grid.architecture

    advection = model.advection.ρE
    validate_total_energy_advection(advection)

    ρE_args = (Val(1),
               model.forcing.ρE,
               advection,
               radiation_flux_divergence(model.radiation),
               common_args...)

    GρE = model.timestepper.Gⁿ.ρE
    launch!(arch, grid, :xyz, _compute_total_energy_tendency!, GρE, grid, ρE_args)
    add_total_energy_sedimentation_tendency!(GρE, model, tracer_transport_velocity)
    return nothing
end

validate_total_energy_advection(advection) = nothing

# The implicit half of the split would transport ρE / ρᵈ rather than Ĥ = (ρE + p) / ρᵈ.
validate_total_energy_advection(::AdaptiveImplicitVerticalAdvection) =
    throw(ArgumentError("TotalEnergyFormulation does not support adaptive implicit vertical " *
                        "advection of ρE; use an explicit scheme such as WENO(order=5)."))

@kernel function _compute_total_energy_tendency!(GρE, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds GρE[i, j, k] = total_energy_tendency(i, j, k, grid, args...)
end

# ∂ₜ(ρE) = - ∇⋅(ρᵈ 𝐮 Ĥ) - ∇⋅(ρᵈ 𝐉ˢ) + Fᴱ + Q_rad. The advective flux carries the total enthalpy per
# unit dry mass with the dry-air mass flux, and the closure flux is that of the static energy ŝ, so
# that turbulent mixing at fixed composition and pressure conserves enthalpy. The vertical part of a
# vertically implicit closure is applied by `implicit_thermodynamic_solve!`. There is no source for
# phase change, expansion or dissipation: they exchange energy within E.
@inline function total_energy_tendency(i, j, k, grid,
                                       id,
                                       ρE_forcing,
                                       advection,
                                       radiation_flux_divergence_field,
                                       dynamics,
                                       formulation,
                                       constants,
                                       specific_prognostic_moisture,
                                       velocities,
                                       microphysics,
                                       microphysical_fields,
                                       closure,
                                       closure_fields,
                                       clock,
                                       model_fields)

    ρᵈ = dynamics_density(dynamics)
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants)

    return ( - div_ρUc(i, j, k, grid, advection, ρᵈ, velocities, formulation.total_enthalpy)
             - ∇_dot_Jᶜ(i, j, k, grid, ρᵈ, closure, closure_fields, id, formulation.static_energy,
                        clock, model_fields, closure_buoyancy)
             + ρE_forcing(i, j, k, grid, clock, model_fields)
             + radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field))
end

#####
##### Sedimentation
#####
#
# Condensate falling relative to the air carries its enthalpy hˣ(T) and its potential energy g z.
# Through a face the energy flux is ρᶠ Σₓ eˣ Fˣ, with Fˣ the sedimentation part of the condensate's
# mass flux per unit density — the flux at the total velocity Wˣ = wᵗ + wˣ less the flux at the
# transport velocity wᵗ, with the condensate's own advection scheme, exactly the mass the tracer
# tendency moves (`sedimentation_mass_fluxes`) — and
#
#   eˣ = hˣ(T_up) + g z_face,   hˡ = cˡ T,   hⁱ = cⁱ T + ℒˡ₀ − ℒⁱ₀,
#
# taken from the cell the flux drains (upwind of its velocity) at the height of the face. The
# tendency is the divergence of that flux, so ∫ρE changes only by what leaves through the bottom:
# cˡ T₁ (or cⁱ T₁ + ℒˡ₀ − ℒⁱ₀) per unit precipitated mass, since g z = 0 at the surface. A cell the
# condensate passes through gains g Δz per unit mass, the frictional dissipation of the fall.

function add_total_energy_sedimentation_tendency!(GρE, model, tracer_transport_velocity)
    condensates = values(model.sedimentation)
    isempty(condensates) && return nothing
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _add_total_energy_sedimentation_tendency!,
            GρE, grid, condensates, tracer_transport_velocity, explicit_condensate_mass_fluxes,
            total_density(model.dynamics), model.temperature, model.thermodynamic_constants)
    return nothing
end

@kernel function _add_total_energy_sedimentation_tendency!(GρE, grid, condensates, wᵗ, mass_fluxes, ρ, T, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds GρE[i, j, k] += total_energy_sedimentation_tendency(i, j, k, grid, condensates, wᵗ,
                                                                   mass_fluxes, ρ, T, constants)
end

@inline function total_energy_sedimentation_tendency(i, j, k, grid, condensates, wᵗ, mass_fluxes, ρ, T, constants)
    Φ⁻, Φ⁺ = condensate_energy_fluxes(i, j, k, grid, condensates, wᵗ, mass_fluxes, ρ, T, constants)
    ρᶠ⁻ = ℑzᵃᵃᶠ(i, j, k,     grid, ρ)
    ρᶠ⁺ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ)
    return - V⁻¹ᶜᶜᶜ(i, j, k, grid) * (ρᶠ⁺ * Φ⁺ - ρᶠ⁻ * Φ⁻)
end

@inline condensate_energy_fluxes(i, j, k, grid, ::Tuple{}, wᵗ, mass_fluxes, ρ, T, constants) = (zero(grid), zero(grid))

@inline function condensate_energy_fluxes(i, j, k, grid, condensates::Tuple, wᵗ, mass_fluxes, ρ, T, constants)
    condensate = first(condensates)
    F⁻, F⁺ = mass_fluxes(i, j, k, grid, condensate, wᵗ, ρ)
    f = condensate.liquid_fraction
    Φ⁻ = condensate_energy_flux(i, j, k,     grid, wᵗ, condensate.velocity, F⁻, f, T, constants)
    Φ⁺ = condensate_energy_flux(i, j, k + 1, grid, wᵗ, condensate.velocity, F⁺, f, T, constants)
    rest⁻, rest⁺ = condensate_energy_fluxes(i, j, k, grid, Base.tail(condensates), wᵗ, mass_fluxes, ρ, T, constants)
    return Φ⁻ + rest⁻, Φ⁺ + rest⁺
end

# The energy carried through face k by the mass fluxes F = (F(W), F(wᵗ)) of a condensate with
# liquid fraction f: the sedimentation part of the flux, each mass flux with the energy of the cell
# it drains.
@inline function condensate_energy_flux(i, j, k, grid, wᵗ, w, F, f, T, constants)
    Fᵂ, Fᵗ = F
    @inbounds wᵗₖ = wᵗ[i, j, k]
    @inbounds Wₖ = wᵗₖ + w[i, j, k]
    z = znode(i, j, k, grid, c, c, Face())
    eᵂ = falling_condensate_energy(i, j, k, Wₖ, f, T, z, constants)
    eᵗ = falling_condensate_energy(i, j, k, wᵗₖ, f, T, z, constants)
    return eᵂ * Fᵂ - eᵗ * Fᵗ
end

# Enthalpy plus potential energy per unit mass of condensate crossing face k at velocity w, with
# the temperature of the cell below the face for an upward velocity and above it otherwise.
@inline function falling_condensate_energy(i, j, k, w, f, T, z, constants)
    k_upwind = ifelse(w > 0, k - 1, k)
    @inbounds T_upwind = T[i, j, k_upwind]
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    hˡ = cˡ * T_upwind
    hⁱ = cⁱ * T_upwind + ℒˡ₀ - ℒⁱ₀
    g = constants.gravitational_acceleration
    return f * hˡ + (1 - f) * hⁱ + g * z
end

# The remainder of the sedimentation flux that the adaptive implicit solve applies to the condensate
# tracers carries its energy in the same flux form.
function AtmosphereModels.implicit_sedimentation_step!(model, Δt, velocities, formulation::TotalEnergyFormulation)
    condensates = values(model.sedimentation)
    implicit_sedimentation(condensates) || return nothing
    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, :xyz, _implicit_total_energy_sedimentation_step!,
            formulation.energy_density, grid, kernel_time_step(arch, grid, Δt), condensates, velocities.w,
            implicit_condensate_mass_fluxes, total_density(model.dynamics), model.temperature,
            model.thermodynamic_constants)
    return nothing
end

@kernel function _implicit_total_energy_sedimentation_step!(ρE, grid, Δt, condensates, wᵗ, mass_fluxes, ρ, T, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds ρE[i, j, k] += Δt * total_energy_sedimentation_tendency(i, j, k, grid, condensates, wᵗ,
                                                                      mass_fluxes, ρ, T, constants)
end

#####
##### Vertically implicit diffusion of the static energy
#####
#
# The closure diffuses the static energy ŝ, not the total energy: diffusing ρE would diffuse g z and
# K too, a spurious downgradient flux of potential energy. The vertically implicit solve therefore acts
# on X = ρᵈ ŝ (the `diffusion_workspace`), which is diffused as ∂ₜX = ∂z(ρᵈ κ ∂z(X / ρᵈ)) like every
# other coupling-weighted scalar, and the increment X⁺ − X is added to ρE. The solve is in flux form
# with no-flux walls (the wall fluxes of ρE enter its explicit tendency), so the increment sums to
# zero over each column and ∫ρE is conserved exactly. ŝ is the value diagnosed at the start of the
# stage; the solve uses the live dry density, as for ρθ.

function AtmosphereModels.implicit_thermodynamic_solve!(solve!, formulation::TotalEnergyFormulation, model)
    isnothing(model.timestepper.implicit_solver) && return nothing
    grid = model.grid
    arch = grid.architecture
    X = formulation.diffusion_workspace
    ρᵈ = dynamics_density(model.dynamics)
    launch!(arch, grid, :xyz, _set_diffusion_workspace!, X, ρᵈ, formulation.static_energy)
    solve!(X)
    launch!(arch, grid, :xyz, _add_diffusion_increment!, formulation.energy_density, X, ρᵈ, formulation.static_energy)
    return nothing
end

@kernel function _set_diffusion_workspace!(X, ρᵈ, ŝ)
    i, j, k = @index(Global, NTuple)
    @inbounds X[i, j, k] = ρᵈ[i, j, k] * ŝ[i, j, k]
end

@kernel function _add_diffusion_increment!(ρE, X, ρᵈ, ŝ)
    i, j, k = @index(Global, NTuple)
    @inbounds ρE[i, j, k] += X[i, j, k] - ρᵈ[i, j, k] * ŝ[i, j, k]
end

#####
##### Set thermodynamic variables
#####

AtmosphereModels.set_thermodynamic_variable!(model::TotalEnergyModel, ::Val{:ρE}, value) =
    set!(model.formulation.energy_density, value)

"""
$(TYPEDSIGNATURES)

Set the total energy density from the temperature ``T``, the moisture, the total density and the
velocities: ``ρE = ρ (cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ + K + g z)``. `set!` applies the velocities
before the temperature, so that the kinetic energy is included; setting the velocities in a later
call to `set!` leaves ``ρE`` unchanged and so changes the temperature.
"""
function AtmosphereModels.set_thermodynamic_variable!(model::TotalEnergyModel, ::Val{:T}, value)
    T = model.temperature # scratch space
    set!(T, value)
    set_energy_density_from_temperature!(model)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Set the total energy density from the liquid-ice potential temperature ``θˡⁱ``, converted to
temperature at the model's total density.
"""
function AtmosphereModels.set_thermodynamic_variable!(model::TotalEnergyModel, ::Union{Val{:θ}, Val{:θˡⁱ}}, value)
    T = model.temperature # scratch space
    set!(T, value)
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _temperature_from_potential_temperature!,
            T, grid, specific_prognostic_moisture(model), model.dynamics, model.microphysics,
            model.microphysical_fields, model.thermodynamic_constants)
    set_energy_density_from_temperature!(model)
    return nothing
end

@kernel function _temperature_from_potential_temperature!(T, grid, specific_prognostic_moisture, dynamics,
                                                          microphysics, microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        θ = T[i, j, k]
        ρ = total_density(dynamics)[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    pˢᵗ = standard_pressure(dynamics)
    𝒰 = LiquidIceDensityState(θ, q, pˢᵗ, ρ, NewtonSolver(typeof(θ)), VaporEquivalentExponent())
    @inbounds T[i, j, k] = temperature(𝒰, constants)
end

function set_energy_density_from_temperature!(model)
    grid = model.grid

    # The kinetic energy is that of the velocities `update_state!` will diagnose from the momentum,
    # with the wall-normal velocity held at zero on impenetrable walls.
    compute_velocities!(model)
    launch!(grid.architecture, grid, :xyz, _energy_density_from_temperature!,
            model.formulation.energy_density, grid, model.temperature, specific_prognostic_moisture(model),
            model.dynamics, model.microphysics, model.microphysical_fields, model.velocities,
            model.thermodynamic_constants)
    return nothing
end

@kernel function _energy_density_from_temperature!(ρE, grid, temperature_field, specific_prognostic_moisture,
                                                   dynamics, microphysics, microphysical_fields, velocities,
                                                   constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ = total_density(dynamics)[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        T = temperature_field[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    𝒰 = with_temperature(InternalEnergyDensityState(zero(T), q, ρ), T, constants)
    K = kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities)
    g = constants.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)
    @inbounds ρE[i, j, k] = ρ * (𝒰.internal_energy + K + g * z)
end
