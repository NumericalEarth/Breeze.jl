const MoistEntropyModel = AtmosphereModel{<:Any, <:MoistEntropyFormulation}

#####
##### Helper accessors
#####

AtmosphereModels.liquid_ice_potential_temperature(model::MoistEntropyModel) = Diagnostics.LiquidIcePotentialTemperature(model, :specific)
AtmosphereModels.liquid_ice_potential_temperature_density(model::MoistEntropyModel) = Diagnostics.LiquidIcePotentialTemperature(model, :density)
AtmosphereModels.static_energy(model::MoistEntropyModel) = Diagnostics.StaticEnergy(model, :specific)

"""
$(TYPEDSIGNATURES)

Return the static energy density as a `Field` whose boundary conditions are the energy fluxes
`𝒬ᵀ` carried inside the entropy fluxes of `ρη` (see `EntropyFluxBoundaryConditionFunction`), so
that `BoundaryConditionOperation` extracts the sensible heat flux. The vapor flux's part is not
included: it is the boundary condition of the moisture density.
"""
function AtmosphereModels.static_energy_density(model::MoistEntropyModel)
    ρη_bcs = model.formulation.entropy_density.boundary_conditions
    ρs_bcs = map_field_boundary_conditions(entropy_to_energy_bc, ρη_bcs)
    return Field(Diagnostics.StaticEnergy(model, :density); boundary_conditions=ρs_bcs)
end

#####
##### Tendency computation
#####

function AtmosphereModels.compute_thermodynamic_tendency!(model::MoistEntropyModel, common_args, tracer_transport_velocity)
    grid = model.grid
    arch = grid.architecture
    microphysics = model.microphysics

    # The water species the microphysics changes, as a compile-time tuple: the vapor density
    # first, then the prognostic microphysical fields.
    names = (moisture_prognostic_name(microphysics), prognostic_field_names(microphysics)...)

    ρη_args = (
        Val(1),
        model.forcing.ρη,
        model.forcing.ρE,
        model.advection.ρη,
        radiation_flux_divergence(model.radiation),
        Val(names),
        common_args...,
        model.temperature)

    Gρη = model.timestepper.Gⁿ.ρη
    launch!(arch, grid, :xyz, compute_moist_entropy_tendency!, Gρη, grid, ρη_args)
    add_sedimentation_tendency!(Gρη, model, tracer_transport_velocity)
    add_precipitation_drag_heating!(Gρη, model, tracer_transport_velocity)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Tendency of the moist entropy density ``ρη`` at cell `(i, j, k)`, apart from the sedimentation of
condensate (`add_sedimentation_tendency!` and `add_precipitation_drag_heating!`):

```math
∂_t (ρη) = - ∇ ⋅ (ρ \\boldsymbol{u} η) - ∇ ⋅ \\boldsymbol{J}^η + F^{ρη}
          + \\frac{F^{ρE} - ∇ ⋅ \\boldsymbol{ℐ}}{T}
          - \\frac{1}{T} \\left[ (𝒢ᵛ - 𝒢ˡ) Ṡᵛ + (𝒢ⁱ - 𝒢ˡ) Ṡⁱ \\right] ,
```

with ``ρ`` the total density, the subgrid flux ``\\boldsymbol{J}^η`` of the specific entropy, the
heating ``F^{ρE}`` supplied under the energy key and the radiative flux divergence entering at the
local temperature, and the entropy of the microphysical phase changes evaluated from the
microphysical tendencies of the vapor (``Ṡᵛ``) and ice (``Ṡⁱ``, cloud ice and snow) densities.
"""
@inline function moist_entropy_tendency(i, j, k, grid,
                                        id,
                                        ρη_forcing,
                                        ρE_forcing,
                                        advection,
                                        radiation_flux_divergence_field,
                                        microphysical_names,
                                        dynamics,
                                        formulation::MoistEntropyFormulation,
                                        constants,
                                        specific_prognostic_moisture,
                                        velocities,
                                        microphysics,
                                        microphysical_fields,
                                        closure,
                                        closure_fields,
                                        clock,
                                        model_fields,
                                        temperature_field)

    η = formulation.specific_entropy
    ρ_field = total_density(dynamics) # η is per unit total mass, carried like the water
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants, microphysics, microphysical_fields)

    @inbounds T = temperature_field[i, j, k]

    FρE = ρE_forcing(i, j, k, grid, clock, model_fields)
    div_ℐ = radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field)

    σᵐ = microphysical_entropy_source(i, j, k, grid, microphysics, microphysical_names, formulation,
                                      dynamics, constants, specific_prognostic_moisture, velocities,
                                      microphysical_fields, T)

    return ( - div_ρUc(i, j, k, grid, advection, ρ_field, velocities, η)
             + c_div_ρU(i, j, k, grid, dynamics, velocities, η)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, η, clock, model_fields, closure_buoyancy)
             + ρη_forcing(i, j, k, grid, clock, model_fields)
             # a volumetric heating Q changes the entropy density at Q / T
             + (FρE + div_ℐ) / T
             + σᵐ)
end

#####
##### Entropy of microphysical phase changes
#####
#
# At fixed internal energy and density the fundamental relation T dS = dU + p dV − Σₓ 𝒢ˣ dmˣ gives the
# entropy source of microphysical transfers between the water species, −Σₓ 𝒢ˣ Ṡˣ / T per unit volume,
# where Ṡˣ is the microphysical tendency of the partial density ρqˣ and 𝒢ˣ = hˣ − T ηˣ the specific
# Gibbs function consistent with `moist_entropy`. The same source holds at fixed enthalpy and
# pressure. The tendencies of a phase change sum to zero, so with the liquid as reference,
#
#   σᵐ = −[(𝒢ᵛ − 𝒢ˡ) Ṡᵛ + (𝒢ⁱ − 𝒢ˡ) Ṡⁱ] / T = −Rᵛ [ln(pᵛ / pᵛ⁺ˡ) Ṡᵛ + ln(pᵛ⁺ⁱ / pᵛ⁺ˡ) Ṡⁱ] ,
#
# which is zero for condensation at saturation, positive for relaxation of a supersaturation or
# evaporation into subsaturated air, and zero for freezing at the triple point. Ṡⁱ is the total
# tendency of the ice-phase masses, each weighted by its ice fraction 1 − `condensate_liquid_fraction`.
# The microphysics reports the latent heating only through the water tendencies (the
# temperature then follows from the conserved internal energy), so nothing is counted twice.

@inline microphysical_entropy_source(i, j, k, grid, ::Nothing, ::Val, args...) = zero(grid)

@inline function microphysical_entropy_source(i, j, k, grid, microphysics, ::Val{names}, formulation,
                                              dynamics, constants, specific_prognostic_moisture,
                                              velocities, microphysical_fields, T) where names
    @inbounds ρ = total_density(dynamics)[i, j, k]
    @inbounds qᵛ = specific_prognostic_moisture[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛ, microphysical_fields)
    η = @inbounds formulation.specific_entropy[i, j, k]
    𝒰 = MoistEntropyState(offset_free_entropy(formulation, η, q), q, ρ)
    ℳ = grid_microphysical_state(i, j, k, grid, microphysics, microphysical_fields, ρ, 𝒰, velocities)
    Ṡ = microphysical_tendencies(microphysics, names, ρ, ℳ, 𝒰, constants)

    Ṡᵛ = first(Ṡ)
    Ṡⁱ = ice_phase_tendency(microphysics, Base.tail(names), Base.tail(Ṡ))

    Δ𝒢ᵛ = vapor_liquid_gibbs_difference(T, ρ * q.vapor, constants)
    Δ𝒢ⁱ = ice_liquid_gibbs_difference(T, constants)

    return - (Δ𝒢ᵛ * Ṡᵛ + Δ𝒢ⁱ * Ṡⁱ) / T
end

# Σ (1 − fˣ) Ṡˣ over the condensate masses, with fˣ the liquid fraction; fields that are not
# condensate masses (number concentrations, rime volume) have no liquid fraction and add nothing.
@inline ice_phase_tendency(microphysics, ::Tuple{}, ::Tuple{}) = 0

@inline function ice_phase_tendency(microphysics, names::Tuple, Ṡ::Tuple)
    fˡ = condensate_liquid_fraction(microphysics, Val(first(names)))
    rest = ice_phase_tendency(microphysics, Base.tail(names), Base.tail(Ṡ))
    return ice_fraction_tendency(fˡ, first(Ṡ)) + rest
end

@inline ice_fraction_tendency(::Nothing, Ṡ) = 0
@inline ice_fraction_tendency(fˡ, Ṡ) = (1 - fˡ) * Ṡ

#####
##### Sedimentation
#####
#
# Falling condensate carries its own entropy. `add_sedimentation_tendency!` moves it with the
# content `condensate_content` returns: a cell losing condensate mass at its temperature T loses
# ηˣ(T) per unit mass (the entropy at fixed temperature and gas densities, which leaves T unchanged),
# and a cell receiving mass from a cell at T′ gains ηˣ(T) + [hˣ(T′) − hˣ(T)] / T, the entropy of
# taking in the enthalpy hˣ(T′) at fixed volume. Precipitation leaving through the bottom carries
# ηˣ(T₁) with its enthalpy hˣ(T₁).
#
# `sedimentation_content_tendency` returns the change of a coupling-weighted (ρᵈ-weighted) variable
# and so multiplies the content flux by the cell's ρᵈ / ρ; the prognostic ρη is the entropy
# density itself, so the content is returned multiplied by ρ / ρᵈ, which cancels that factor.

@inline function AtmosphereModels.condensate_content(i, j, k, grid, formulation::MoistEntropyFormulation,
                                                     dynamics, constants, microphysics, microphysical_fields,
                                                     specific_prognostic_moisture, temperature_field)
    @inbounds begin
        T = temperature_field[i, j, k]
        r = total_density(dynamics)[i, j, k] / dynamics_density(dynamics)[i, j, k]
    end
    b = formulation.water_entropy_offset
    χˡ = r * (liquid_entropy(T, constants) + b)
    χⁱ = r * (ice_entropy(T, constants) + b)
    # Only enthalpy differences within a phase enter, so the constant parts of hˣ are omitted
    hˡ = constants.liquid.heat_capacity * T
    hⁱ = constants.ice.heat_capacity * T
    return (; χ = (χˡ, χⁱ), h = (hˡ, hⁱ), ∂φ∂h = r / T)
end

#####
##### Drag heating of falling condensate
#####
#
# Condensate falling at its terminal velocity relative to the air converts its potential energy into
# heat by drag (Pauluis et al. 2000; Pressel et al. 2015, eq. 54), at the rate g |w| ρqˣ per unit
# volume. For the discrete mass that the tracer tendency moves through a face, Φ = ρᶠ [F(wᵗ + wˣ) − F(wᵗ)]
# (positive upward), the potential energy released between the two cell centers is −Φ g Δz; each
# cell takes the part released within it, −Φ g (zᶜ − zᶠ) through its bottom face and
# −Φ g (zᶠ − zᶜ) through its top face. The bottom cell takes the whole fall to the ground of the
# condensate that leaves the domain. The heating enters the entropy at the local temperature.

function add_precipitation_drag_heating!(G, model, tracer_transport_velocity)
    condensates = values(model.sedimentation)
    isempty(condensates) && return nothing
    grid = model.grid
    g = model.thermodynamic_constants.gravitational_acceleration
    launch!(grid.architecture, grid, :xyz, _add_precipitation_drag_heating!,
            G, grid, condensates, tracer_transport_velocity, model.dynamics, model.temperature, g)
    return nothing
end

@kernel function _add_precipitation_drag_heating!(G, grid, condensates, wᵗ, dynamics, temperature_field, g)
    i, j, k = @index(Global, NTuple)
    ρ = total_density(dynamics)
    Φ⁻, Φ⁺ = sedimentation_mass_fluxes(i, j, k, grid, condensates, wᵗ, ρ)
    ρᶠ⁻ = ℑzᵃᵃᶠ(i, j, k,     grid, ρ)
    ρᶠ⁺ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ)
    zᶜ = znode(i, j, k,     grid, c, c, c)
    z⁻ = znode(i, j, k,     grid, c, c, f)
    z⁺ = znode(i, j, k + 1, grid, c, c, f)
    ε = - g * V⁻¹ᶜᶜᶜ(i, j, k, grid) * (ρᶠ⁻ * Φ⁻ * (zᶜ - z⁻) + ρᶠ⁺ * Φ⁺ * (z⁺ - zᶜ))
    @inbounds G[i, j, k] += ε / temperature_field[i, j, k]
end

# The sedimentation parts F(wᵗ + wˣ) − F(wᵗ) of the condensates' volume fluxes through the lower and
# upper faces of cell (i, j, k), summed over the condensates
@inline sedimentation_mass_fluxes(i, j, k, grid, ::Tuple{}, wᵗ, ρ) = (zero(grid), zero(grid))

@inline function sedimentation_mass_fluxes(i, j, k, grid, condensates::Tuple, wᵗ, ρ)
    F⁻, F⁺ = explicit_condensate_mass_fluxes(i, j, k, grid, first(condensates), wᵗ, ρ)
    rest⁻, rest⁺ = sedimentation_mass_fluxes(i, j, k, grid, Base.tail(condensates), wᵗ, ρ)
    return F⁻[1] - F⁻[2] + rest⁻, F⁺[1] - F⁺[2] + rest⁺
end

#####
##### Set thermodynamic variables
#####

AtmosphereModels.set_thermodynamic_variable!(model::MoistEntropyModel, ::Val{:ρη}, value) =
    set!(model.formulation.entropy_density, value)

function AtmosphereModels.set_thermodynamic_variable!(model::MoistEntropyModel, ::Val{:η}, value)
    η = model.formulation.specific_entropy
    set!(η, value)
    ρ = total_density(model.dynamics)
    set!(model.formulation.entropy_density, ρ * η)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Set the moist entropy from the in-situ temperature ``T``, at the model's total density and
moisture mass fractions: ``ρη = ρ η(T, ρ, q)``.
"""
function AtmosphereModels.set_thermodynamic_variable!(model::MoistEntropyModel, ::Val{:T}, value)
    T = model.temperature # scratch
    set!(T, value)
    launch_entropy_from_temperature!(model, T, Val(:T))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Set the moist entropy from the liquid-ice potential temperature ``θˡⁱ`` (with the vapor-equivalent
Exner exponent), inverted for the temperature at the model's total density.
"""
function AtmosphereModels.set_thermodynamic_variable!(model::MoistEntropyModel, ::Union{Val{:θ}, Val{:θˡⁱ}}, value)
    θ = model.temperature # scratch
    set!(θ, value)
    launch_entropy_from_temperature!(model, θ, Val(:θ))
    return nothing
end

function launch_entropy_from_temperature!(model, field, variable)
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _entropy_from_temperature!,
            model.formulation.entropy_density, model.formulation.specific_entropy, grid, field,
            variable, model.formulation, specific_prognostic_moisture(model), model.dynamics, model.microphysics,
            model.microphysical_fields, model.thermodynamic_constants)
    return nothing
end

@kernel function _entropy_from_temperature!(entropy_density, specific_entropy, grid, field, variable,
                                            formulation, specific_prognostic_moisture, dynamics, microphysics,
                                            microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ = total_density(dynamics)[i, j, k]
        qᵛ = specific_prognostic_moisture[i, j, k]
        ϕ = field[i, j, k]
    end
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛ, microphysical_fields)
    T = input_temperature(variable, ϕ, ρ, q, dynamics, constants)
    η = offset_entropy(formulation, moist_entropy(T, ρ, q, constants), q)
    @inbounds specific_entropy[i, j, k] = η
    @inbounds entropy_density[i, j, k] = ρ * η
end

@inline input_temperature(::Val{:T}, T, ρ, q, dynamics, constants) = T

@inline function input_temperature(::Val{:θ}, θ, ρ, q, dynamics, constants)
    pˢᵗ = convert(typeof(θ), standard_pressure(dynamics))
    𝒰 = LiquidIceDensityState(θ, q, pˢᵗ, ρ, NewtonSolver(typeof(θ)), VaporEquivalentExponent())
    return temperature(𝒰, constants)
end
