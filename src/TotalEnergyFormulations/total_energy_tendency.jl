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
    add_water_enthalpy_transport_tendency!(GρE, model)
    add_closure_water_enthalpy_tendency!(GρE, model, model.closure, model.closure_fields)
    add_total_energy_sedimentation_tendency!(GρE, model, tracer_transport_velocity)
    return nothing
end

validate_total_energy_advection(advection) = nothing

# The implicit half of the split would transport ρE / ρᵈ rather than the enthalpies.
validate_total_energy_advection(::AdaptiveImplicitVerticalAdvection) =
    throw(ArgumentError("TotalEnergyFormulation does not support adaptive implicit vertical " *
                        "advection of ρE; use an explicit scheme such as WENO(order=5)."))

@kernel function _compute_total_energy_tendency!(GρE, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds GρE[i, j, k] = total_energy_tendency(i, j, k, grid, args...)
end

# ∂ₜ(ρE) = - ∇⋅(ρᵈ 𝐮 ĥᵈ) - Σₓ ∇⋅(𝐅ˣ ĥˣ) - ∇⋅(ρᵈ 𝐉ˢ) + Fᴱ + Q_rad. The advective flux carries each
# constituent's total enthalpy with its own mass flux (the water's part is added by
# `add_water_enthalpy_transport_tendency!`), and the closure flux is that of the static energy ŝ, so
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
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants, microphysics, microphysical_fields)

    return ( - div_ρUc(i, j, k, grid, advection, ρᵈ, velocities, formulation.dry_enthalpy)
             - ∇_dot_Jᶜ(i, j, k, grid, total_density(dynamics), closure, closure_fields, id, formulation.static_energy,
                        clock, model_fields, closure_buoyancy)
             + ρE_forcing(i, j, k, grid, clock, model_fields)
             + radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field))
end

#####
##### Transport of the water's enthalpy
#####
#
# The advective flux of each water species carries that species' total enthalpy per unit mass,
#
#   ĥˣ = hˣ + K + g z = ĥᵈ + (cˣ − cᵖᵈ) T + ℓˣ + T₀ (cᵖᵈ − cˡ),  hᵛ = cᵖᵛ T + ℒˡ₀ − cˡ T₀, hˡ = cˡ (T − T₀), ...
#
# with the mass flux 𝐅ˣ taken from exactly the flux the species' own tendency applies: its scheme, its
# transport velocity and the total density. So the latent energy ℓˣ and the origin T₀ cˡ of the water
# never move apart from its mass, however differently the species is reconstructed or however its
# transport velocity differs from the dry air's. The solution is independent of T₀ for this reason:
# T₀ (ρᵈ cᵖᵈ + ρqᵗ cˡ) is carried by the dry air's flux and the water's flux with the mass it counts.
#
# The enthalpy at a face is the mean of the two cells', since ĥˣ varies smoothly, like T and z, where
# the mass flux varies sharply and its reconstruction is non-oscillatory. The term runs in kernels of
# its own, one per species, for the same reason as the sedimentation term.

# The part of a species' enthalpy per unit mass other than K + g z: a cell field for the prognostic
# moisture (`moisture_enthalpy`), or cˣ T + ℓˣ for a condensate.
struct CondensateEnthalpy{FT}
    heat_capacity :: FT
    latent_energy :: FT # ℓˣ − cˡ T₀
end

# A liquid fraction f of a mixed-phase mass; a condensate that declares none is liquid.
@inline condensate_enthalpy(microphysics, name, T₀, constants) =
    condensate_enthalpy(something(condensate_liquid_fraction(microphysics, name), 1), T₀, constants)

@inline function condensate_enthalpy(f::Number, T₀, constants)
    FT = typeof(T₀)
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    heat_capacity = f * cˡ + (1 - f) * cⁱ
    latent_energy = (1 - f) * ice_latent_energy(constants) - cˡ * T₀
    return CondensateEnthalpy{FT}(heat_capacity, latent_energy)
end

@inline carried_energyᶠᶜᶜ(i, j, k, grid, e, T) = ℑxᶠᵃᵃ(i, j, k, grid, e)
@inline carried_energyᶜᶠᶜ(i, j, k, grid, e, T) = ℑyᵃᶠᵃ(i, j, k, grid, e)
@inline carried_energyᶜᶜᶠ(i, j, k, grid, e, T) = ℑzᵃᵃᶠ(i, j, k, grid, e)
@inline carried_energyᶠᶜᶜ(i, j, k, grid, e::CondensateEnthalpy, T) = e.heat_capacity * ℑxᶠᵃᵃ(i, j, k, grid, T) + e.latent_energy
@inline carried_energyᶜᶠᶜ(i, j, k, grid, e::CondensateEnthalpy, T) = e.heat_capacity * ℑyᵃᶠᵃ(i, j, k, grid, T) + e.latent_energy
@inline carried_energyᶜᶜᶠ(i, j, k, grid, e::CondensateEnthalpy, T) = e.heat_capacity * ℑzᵃᵃᶠ(i, j, k, grid, T) + e.latent_energy

# ĥᵈ − cᵖᵈ T + cᵖᵈ T₀ = K + g z at a face
@inline water_enthalpy_flux_x(i, j, k, grid, advection, ρ, U, q, e, ĥᵈ, T, cᵖᵈ, T₀) =
    tracer_mass_flux_x(i, j, k, grid, ρ, advection, U.u, q) *
    (ℑxᶠᵃᵃ(i, j, k, grid, ĥᵈ) - cᵖᵈ * (ℑxᶠᵃᵃ(i, j, k, grid, T) - T₀) + carried_energyᶠᶜᶜ(i, j, k, grid, e, T))

@inline water_enthalpy_flux_y(i, j, k, grid, advection, ρ, U, q, e, ĥᵈ, T, cᵖᵈ, T₀) =
    tracer_mass_flux_y(i, j, k, grid, ρ, advection, U.v, q) *
    (ℑyᵃᶠᵃ(i, j, k, grid, ĥᵈ) - cᵖᵈ * (ℑyᵃᶠᵃ(i, j, k, grid, T) - T₀) + carried_energyᶜᶠᶜ(i, j, k, grid, e, T))

@inline water_enthalpy_flux_z(i, j, k, grid, advection, ρ, U, q, e, ĥᵈ, T, cᵖᵈ, T₀) =
    tracer_mass_flux_z(i, j, k, grid, ρ, advection, U.w, q) *
    (ℑzᵃᵃᶠ(i, j, k, grid, ĥᵈ) - cᵖᵈ * (ℑzᵃᵃᶠ(i, j, k, grid, T) - T₀) + carried_energyᶜᶜᶠ(i, j, k, grid, e, T))

@kernel function _add_water_enthalpy_transport_tendency!(GρE, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds GρE[i, j, k] -= V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, water_enthalpy_flux_x, args...) +
                                                       δyᵃᶜᵃ(i, j, k, grid, water_enthalpy_flux_y, args...) +
                                                       δzᵃᵃᶜ(i, j, k, grid, water_enthalpy_flux_z, args...))
end

function add_water_enthalpy_transport_tendency!(GρE, model)
    grid = model.grid
    formulation = model.formulation
    T₀ = formulation.reference_temperature
    constants = model.thermodynamic_constants
    ρ = total_density(model.dynamics)
    U = transport_velocities(model)
    moisture_name = moisture_prognostic_name(model.microphysics)
    common = (ρ, U)
    closing = (formulation.dry_enthalpy, model.temperature, constants.dry_air.heat_capacity, T₀)

    moisture_args = (field_advection_scheme(model.advection, moisture_name), common...,
                     specific_prognostic_moisture(model), formulation.moisture_enthalpy, closing...)
    launch!(grid.architecture, grid, :xyz, _add_water_enthalpy_transport_tendency!, GρE, grid, moisture_args)

    foreach(condensate_field_names(model.microphysics)) do name
        q = getproperty(model.microphysical_fields, specific_field_name(name))
        e = condensate_enthalpy(model.microphysics, Val(name), T₀, constants)
        args = (field_advection_scheme(model.advection, name), common..., q, e, closing...)
        launch!(grid.architecture, grid, :xyz, _add_water_enthalpy_transport_tendency!, GρE, grid, args)
    end

    return nothing
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
#   eˣ = hˣ(T_up) + g z_face,   hˡ = cˡ (T − T₀),   hⁱ = cⁱ T − cˡ T₀ + ℒˡ₀ − ℒⁱ₀,
#
# taken from the cell the flux drains (upwind of its velocity) at the height of the face. The
# tendency is the divergence of that flux, so ∫ρE changes only by what leaves through the bottom:
# cˡ (T₁ − T₀) (or cⁱ T₁ − cˡ T₀ + ℒˡ₀ − ℒⁱ₀) per unit precipitated mass, with T₀ the energy origin, since g z = 0 at the surface. A cell the
# condensate passes through gains g Δz per unit mass, the frictional dissipation of the fall.

function add_total_energy_sedimentation_tendency!(GρE, model, tracer_transport_velocity)
    condensates = values(model.sedimentation)
    isempty(condensates) && return nothing
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _add_total_energy_sedimentation_tendency!,
            GρE, grid, condensates, tracer_transport_velocity, explicit_condensate_mass_fluxes,
            total_density(model.dynamics), model.temperature, model.formulation.reference_temperature,
            model.thermodynamic_constants)
    return nothing
end

@kernel function _add_total_energy_sedimentation_tendency!(GρE, grid, condensates, wᵗ, mass_fluxes, ρ, T, T₀, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds GρE[i, j, k] += total_energy_sedimentation_tendency(i, j, k, grid, condensates, wᵗ,
                                                                   mass_fluxes, ρ, T, T₀, constants)
end

@inline function total_energy_sedimentation_tendency(i, j, k, grid, condensates, wᵗ, mass_fluxes, ρ, T, T₀, constants)
    Φ⁻, Φ⁺ = condensate_energy_fluxes(i, j, k, grid, condensates, wᵗ, mass_fluxes, ρ, T, T₀, constants)
    ρᶠ⁻ = ℑzᵃᵃᶠ(i, j, k,     grid, ρ)
    ρᶠ⁺ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ)
    return - V⁻¹ᶜᶜᶜ(i, j, k, grid) * (ρᶠ⁺ * Φ⁺ - ρᶠ⁻ * Φ⁻)
end

@inline condensate_energy_fluxes(i, j, k, grid, ::Tuple{}, wᵗ, mass_fluxes, ρ, T, T₀, constants) = (zero(grid), zero(grid))

@inline function condensate_energy_fluxes(i, j, k, grid, condensates::Tuple, wᵗ, mass_fluxes, ρ, T, T₀, constants)
    condensate = first(condensates)
    F⁻, F⁺ = mass_fluxes(i, j, k, grid, condensate, wᵗ, ρ)
    f = condensate.liquid_fraction
    Φ⁻ = condensate_energy_flux(i, j, k,     grid, wᵗ, condensate.velocity, F⁻, f, T, T₀, constants)
    Φ⁺ = condensate_energy_flux(i, j, k + 1, grid, wᵗ, condensate.velocity, F⁺, f, T, T₀, constants)
    rest⁻, rest⁺ = condensate_energy_fluxes(i, j, k, grid, Base.tail(condensates), wᵗ, mass_fluxes, ρ, T, T₀, constants)
    return Φ⁻ + rest⁻, Φ⁺ + rest⁺
end

# The energy carried through face k by the mass fluxes F = (F(W), F(wᵗ)) of a condensate with
# liquid fraction f: the sedimentation part of the flux, each mass flux with the energy of the cell
# it drains.
@inline function condensate_energy_flux(i, j, k, grid, wᵗ, w, F, f, T, T₀, constants)
    Fᵂ, Fᵗ = F
    @inbounds wᵗₖ = wᵗ[i, j, k]
    @inbounds Wₖ = wᵗₖ + w[i, j, k]
    z = znode(i, j, k, grid, c, c, Face())
    eᵂ = falling_condensate_energy(i, j, k, Wₖ, f, T, T₀, z, constants)
    eᵗ = falling_condensate_energy(i, j, k, wᵗₖ, f, T, T₀, z, constants)
    return eᵂ * Fᵂ - eᵗ * Fᵗ
end

# Enthalpy plus potential energy per unit mass of condensate crossing face k at velocity w, with
# the temperature of the cell below the face for an upward velocity and above it otherwise.
@inline function falling_condensate_energy(i, j, k, w, f, T, T₀, z, constants)
    k_upwind = ifelse(w > 0, k - 1, k)
    @inbounds T_upwind = T[i, j, k_upwind]
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    hˡ = cˡ * T_upwind
    hⁱ = cⁱ * T_upwind + ℒˡ₀ - ℒⁱ₀
    g = constants.gravitational_acceleration
    return f * hˡ + (1 - f) * hⁱ - cˡ * T₀ + g * z
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
            formulation.reference_temperature, model.thermodynamic_constants)
    return nothing
end

@kernel function _implicit_total_energy_sedimentation_step!(ρE, grid, Δt, condensates, wᵗ, mass_fluxes, ρ, T, T₀, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds ρE[i, j, k] += Δt * total_energy_sedimentation_tendency(i, j, k, grid, condensates, wᵗ,
                                                                      mass_fluxes, ρ, T, T₀, constants)
end

#####
##### Closure transport of the static energy and of the water's dry-air-equivalent enthalpy
#####
#
# The closure mixes each water species by a mass flux 𝐉ˣ = − ρ κ ∇qˣ, which moves water and no dry air
# (the dry density has no closure flux), so the water's whole enthalpy has to move with it. The flux
# of the static energy, − ρ κ ∇ŝ, with ŝ = qᵈ (ĥᵈ − K) + Σₓ qˣ (hˣ + g z), carries only
#
#   − ρ κ [ qᵈ ∇(ĥᵈ − K) + Σₓ qˣ ∇(hˣ + g z) ] + Σₓ (hˣ + g z − ĥᵈ + K) 𝐉ˣ ,
#
# a mixing of air and water at fixed mass and pressure: the water displaces an equal mass of dry
# air. Since the dry air does not actually move, its enthalpy remains in the wrong cell, warming the
# source of the water flux and cooling the sink by about cᵖᵈ (T − T₀) per unit mass, and making the
# solution depend on T₀. The term Σₓ ĥᵈ 𝐉ˣ supplies the difference: it is the enthalpy of the dry air
# the water was assumed to displace.

function add_closure_water_enthalpy_tendency!(GρE, model, closure, closure_fields)
    grid = model.grid
    ρ = total_density(model.dynamics)
    ĥᵈ = model.formulation.dry_enthalpy
    disc = time_discretization(closure)
    constants = model.thermodynamic_constants
    buoyancy = AtmosphereModelBuoyancy(model.dynamics, model.formulation, constants, model.microphysics,
                                       model.microphysical_fields)
    foreach(water_species_fields(model)) do (name, q)
        id = closure_scalar_index(model, name)
        args = (ĥᵈ, ρ, disc, closure, closure_fields, id, q, model.clock, Oceananigans.fields(model), buoyancy)
        launch!(grid.architecture, grid, :xyz, _add_closure_water_enthalpy_tendency!, GρE, grid, args)
    end
    return nothing
end

add_closure_water_enthalpy_tendency!(GρE, model, ::Nothing, closure_fields) = nothing

function add_closure_water_enthalpy_tendency!(GρE, model, closures::Tuple, closure_fields)
    foreach((closure, K) -> add_closure_water_enthalpy_tendency!(GρE, model, closure, K), closures, closure_fields)
    return nothing
end

@inline water_species_fields(model) = (model_moisture_name_field(model), condensate_name_fields(model)...)
@inline model_moisture_name_field(model) = (moisture_prognostic_name(model.microphysics), specific_prognostic_moisture(model))
@inline condensate_name_fields(model) =
    map(name -> (name, getproperty(model.microphysical_fields, specific_field_name(name))),
        condensate_field_names(model.microphysics))

@inline closure_water_enthalpy_flux_x(i, j, k, grid, ĥᵈ, ρ, args...) =
    Axᶠᶜᶜ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, ĥᵈ) * Jᶜx(i, j, k, grid, ρ, args...)
@inline closure_water_enthalpy_flux_y(i, j, k, grid, ĥᵈ, ρ, args...) =
    Ayᶜᶠᶜ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, ĥᵈ) * Jᶜy(i, j, k, grid, ρ, args...)
@inline closure_water_enthalpy_flux_z(i, j, k, grid, ĥᵈ, ρ, args...) =
    Azᶜᶜᶠ(i, j, k, grid) * ℑzᵃᵃᶠ(i, j, k, grid, ĥᵈ) * Jᶜz(i, j, k, grid, ρ, args...)

@kernel function _add_closure_water_enthalpy_tendency!(GρE, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds GρE[i, j, k] -= V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, closure_water_enthalpy_flux_x, args...) +
                                                       δyᵃᶜᵃ(i, j, k, grid, closure_water_enthalpy_flux_y, args...) +
                                                       δzᵃᵃᶜ(i, j, k, grid, closure_water_enthalpy_flux_z, args...))
end

#####
##### Vertically implicit diffusion of the static energy
#####
#
# The closure diffuses the static energy ŝ, not the total energy: diffusing ρE would diffuse g z and
# K too, a spurious downgradient flux of potential energy. The vertically implicit solve therefore acts
# on X = ρ ŝ (the `diffusion_workspace`), which is diffused as ∂ₜX = ∂z(ρ κ ∂z(X / ρ)) with the total
# density ρ the water species' solves are weighted with, and the increment X⁺ − X is added to ρE. The
# solve is in flux form with no-flux walls (the wall fluxes of ρE enter its explicit tendency), so
# the increment sums to zero over each column and ∫ρE is conserved exactly. ŝ is the value diagnosed
# at the start of the stage. The water's solves ran first, and the dry-air-equivalent enthalpy of
# their mass fluxes (see above) is taken from the solved water, the flux they actually applied.

AtmosphereModels.implicit_diffusion_density(::TotalEnergyFormulation, model) = total_density(model.dynamics)

AtmosphereModels.implicit_advection_density(dynamics, ::TotalEnergyFormulation, name::Symbol) =
    name === :ρu || name === :ρv || name === :ρw ? dynamics_density(dynamics) : total_density(dynamics)

function AtmosphereModels.implicit_thermodynamic_solve!(solve!, formulation::TotalEnergyFormulation, model, Δt)
    isnothing(model.timestepper.implicit_solver) && return nothing
    grid = model.grid
    arch = grid.architecture
    X = formulation.diffusion_workspace
    ρ = total_density(model.dynamics)
    launch!(arch, grid, :xyz, _set_diffusion_workspace!, X, ρ, formulation.static_energy)
    solve!(X)
    launch!(arch, grid, :xyz, _add_diffusion_increment!, formulation.energy_density, X, ρ, formulation.static_energy)
    add_implicit_water_enthalpy_flux!(model, model.closure, model.closure_fields, Δt)
    return nothing
end

add_implicit_water_enthalpy_flux!(model, ::Nothing, closure_fields, Δt) = nothing

function add_implicit_water_enthalpy_flux!(model, closures::Tuple, closure_fields, Δt)
    foreach((closure, K) -> add_implicit_water_enthalpy_flux!(model, closure, K, Δt), closures, closure_fields)
    return nothing
end

add_implicit_water_enthalpy_flux!(model, closure, closure_fields, Δt) =
    add_implicit_water_enthalpy_flux!(model, time_discretization(closure), closure, closure_fields, Δt)

add_implicit_water_enthalpy_flux!(model, discretization, closure, closure_fields, Δt) = nothing

function add_implicit_water_enthalpy_flux!(model, ::VerticallyImplicitTimeDiscretization, closure, closure_fields, Δt)
    grid = model.grid
    arch = grid.architecture
    formulation = model.formulation
    ρ = total_density(model.dynamics)
    kernel_Δt = kernel_time_step(arch, grid, Δt)
    constants = model.thermodynamic_constants
    buoyancy = AtmosphereModelBuoyancy(model.dynamics, formulation, constants, model.microphysics,
                                       model.microphysical_fields)
    prognostic = prognostic_fields(model)
    names = (moisture_prognostic_name(model.microphysics), condensate_field_names(model.microphysics)...)
    foreach(names) do name
        q = SolvedSpecificWater(getproperty(prognostic, name), ρ)
        id = closure_scalar_index(model, name)
        launch!(arch, grid, :xyz, _add_implicit_water_enthalpy_flux!, formulation.energy_density, grid, kernel_Δt,
                formulation.dry_enthalpy, ρ, closure, closure_fields, id, q, model.clock, Oceananigans.fields(model), buoyancy)
    end
    return nothing
end

# The specific humidity of a species that the implicit solve weights with ρ: the solved density over ρ
struct SolvedSpecificWater{Q, R}
    density :: Q
    total_density :: R
end

Adapt.adapt_structure(to, q::SolvedSpecificWater) = SolvedSpecificWater(adapt(to, q.density), adapt(to, q.total_density))

@inline Base.getindex(q::SolvedSpecificWater, i, j, k) = @inbounds q.density[i, j, k] / q.total_density[i, j, k]

@inline function implicit_water_enthalpy_flux(i, j, k, grid, ĥᵈ, ρ, closure, K, id, q, clock, model_fields, buoyancy)
    J = ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, K, id, q,
                                                     clock, model_fields, buoyancy)
    interior_face = (k > 1) & (k <= size(grid, 3))
    return ifelse(interior_face, Azᶜᶜᶠ(i, j, k, grid) * ℑzᵃᵃᶠ(i, j, k, grid, ĥᵈ) * J, zero(J))
end

@kernel function _add_implicit_water_enthalpy_flux!(ρE, grid, Δt, ĥᵈ, ρ, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds ρE[i, j, k] -= Δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, implicit_water_enthalpy_flux, ĥᵈ, ρ, args...)
end

@kernel function _set_diffusion_workspace!(X, ρ, ŝ)
    i, j, k = @index(Global, NTuple)
    @inbounds X[i, j, k] = ρ[i, j, k] * ŝ[i, j, k]
end

@kernel function _add_diffusion_increment!(ρE, X, ρ, ŝ)
    i, j, k = @index(Global, NTuple)
    @inbounds ρE[i, j, k] += X[i, j, k] - ρ[i, j, k] * ŝ[i, j, k]
end

#####
##### Set thermodynamic variables
#####

AtmosphereModels.set_thermodynamic_variable!(model::TotalEnergyModel, ::Val{:ρE}, value) =
    set!(model.formulation.energy_density, value)

"""
$(TYPEDSIGNATURES)

Set the total energy density from the temperature ``T``, the moisture, the total density and the
velocities: ``ρE = ρ (cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ + K + g z) - T₀ (ρᵈ cᵖᵈ + ρqᵗ cˡ)``, with ``T₀`` the
energy origin of the formulation. `set!` applies the velocities
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
            model.formulation.reference_temperature, model.thermodynamic_constants)
    return nothing
end

@kernel function _energy_density_from_temperature!(ρE, grid, temperature_field, specific_prognostic_moisture,
                                                   dynamics, microphysics, microphysical_fields, velocities,
                                                   T₀, constants)
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
    @inbounds ρᵈ = dynamics_density(dynamics)[i, j, k]
    @inbounds ρE[i, j, k] = ρ * (𝒰.internal_energy + K + g * z) - energy_origin_offset(T₀, ρᵈ, ρ, constants)
end
