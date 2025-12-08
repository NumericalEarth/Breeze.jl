using Breeze.Thermodynamics: StaticEnergyState, with_temperature

struct StaticEnergyThermodynamics{E, S}
    energy_density :: E
    specific_energy :: S
end

Adapt.adapt_structure(to, thermo::StaticEnergyThermodynamics) =
    StaticEnergyThermodynamics(adapt(to, thermo.energy_density),
                               adapt(to, thermo.specific_energy))

function fill_halo_regions!(thermo::StaticEnergyThermodynamics)
    fill_halo_regions!(thermo.energy_density)
    fill_halo_regions!(thermo.specific_energy)
    return nothing
end

const ASEF = AnelasticFormulation{<:StaticEnergyThermodynamics}

prognostic_field_names(formulation::ASEF) = tuple(:ρe)
additional_field_names(formulation::ASEF) = tuple(:e)
thermodynamic_density_name(::ASEF) = :ρe
thermodynamic_density(formulation::ASEF) = formulation.thermodynamics.energy_density
fields(formulation::ASEF) = (; e=formulation.thermodynamics.specific_energy)
prognostic_fields(formulation::ASEF) = (; ρe=formulation.thermodynamics.energy_density)

function materialize_thermodynamics(::Val{:StaticEnergy}, grid, boundary_conditions)
    energy_density = CenterField(grid, boundary_conditions=boundary_conditions.ρe)
    specific_energy = CenterField(grid) # e = ρe / ρᵣ (diagnostic per-mass energy)
    return StaticEnergyThermodynamics(energy_density, specific_energy)
end

function compute_auxiliary_thermodynamic_variables!(formulation::ASEF, i, j, k, grid)
    @inbounds begin
        ρᵣ = formulation.reference_state.density[i, j, k]
        ρe = formulation.thermodynamics.energy_density[i, j, k]
        formulation.thermodynamics.specific_energy[i, j, k] = ρe / ρᵣ
    end
    return nothing
end

function diagnose_thermodynamic_state(i, j, k, grid, formulation::ASEF,
                                      microphysics,
                                      microphysical_fields,
                                      constants,
                                      specific_moisture)
  
    e = @inbounds formulation.thermodynamics.specific_energy[i, j, k]
    pᵣ = @inbounds formulation.reference_state.pressure[i, j, k]
    ρᵣ = @inbounds formulation.reference_state.density[i, j, k]
    qᵗ = @inbounds specific_moisture[i, j, k]

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    z = znode(i, j, k, grid, c, c, c)

    return StaticEnergyState(e, q, z, pᵣ)
end

function collect_prognostic_fields(formulation::ASEF,
                                   momentum,
                                   moisture_density,
                                   microphysical_fields,
                                   tracers)
    ρe = formulation.thermodynamics.energy_density
    thermodynamic_variables = (ρe=ρe, ρqᵗ=moisture_density)
    return merge(momentum, thermodynamic_variables, microphysical_fields, tracers)
end

const StaticEnergyAnelasticModel = AtmosphereModel{<:ASEF}
const SEAM = StaticEnergyAnelasticModel

liquid_ice_potential_temperature(model::SEAM) = LiquidIcePotentialTemperature(model, :specific)
potential_temperature_density(model::SEAM) = LiquidIcePotentialTemperature(model, :density)
static_energy(model::SEAM) = model.formulation.thermodynamics.specific_energy
static_energy_density(model::SEAM) = model.formulation.thermodynamics.energy_density

function compute_thermodynamic_tendency!(model::StaticEnergyAnelasticModel, common_args)
    grid = model.grid
    arch = grid.architecture

    ρe_args = (
        Val(1),
        model.forcing.ρe,
        model.advection.ρe,
        common_args...,
        model.temperature)

    Gρe = model.timestepper.Gⁿ.ρe
    launch!(arch, grid, :xyz, compute_static_energy_tendency!, Gρe, grid, ρe_args)
    return nothing
end

@inline function static_energy_tendency(i, j, k, grid,
                                        id,
                                        ρe_forcing,
                                        advection,
                                        formulation,
                                        constants,
                                        specific_moisture,
                                        velocities,
                                        microphysics,
                                        microphysical_fields,
                                        closure,
                                        closure_fields,
                                        clock,
                                        model_fields,
                                        temperature)

    specific_energy = formulation.thermodynamics.specific_energy

    𝒰 = diagnose_thermodynamic_state(i, j, k, grid,
                                     formulation,
                                     microphysics,
                                     microphysical_fields,
                                     constants,
                                     specific_moisture)

    ρ = formulation.reference_state.density

    # Compute the buoyancy flux term, ρᵣ w b
    buoyancy_flux = ℑzᵃᵃᶜ(i, j, k, grid, ρ_w_bᶜᶜᶠ,
                          velocities.w, formulation, ρ, temperature, specific_moisture,
                          microphysics, microphysical_fields, constants)

    closure_buoyancy = AtmosphereModelBuoyancy(formulation, constants)

    return ( - div_ρUc(i, j, k, grid, advection, ρ, velocities, specific_energy)
             + buoyancy_flux
             - ∇_dot_Jᶜ(i, j, k, grid, ρ, closure, closure_fields, id, specific_energy, clock, model_fields, closure_buoyancy)
             + microphysical_tendency(i, j, k, grid, microphysics, Val(:ρe), microphysical_fields, 𝒰, constants)
             + ρe_forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Dispatch for setting thermodynamic variables
#####

# StaticEnergyThermodynamics: :ρe sets energy density directly
set_thermodynamic_variable!(model::StaticEnergyAnelasticModel, ::Val{:ρe}, value) =
    set!(model.formulation.thermodynamics.energy_density, value)

function set_thermodynamic_variable!(model::StaticEnergyAnelasticModel, ::Val{:e}, value)
    set!(model.formulation.thermodynamics.specific_energy, value)
    ρᵣ = model.formulation.reference_state.density
    e = model.formulation.thermodynamics.specific_energy
    set!(model.formulation.thermodynamics.energy_density, ρᵣ * e)
    return nothing
end

# Setting :θ (potential temperature)
function set_thermodynamic_variable!(model::StaticEnergyAnelasticModel, ::Val{:θ}, value)
    thermo = model.formulation.thermodynamics
    θ = model.temperature # scratch space
    set!(θ, value)

    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, :xyz,
            _energy_density_from_potential_temperature!,
            thermo.energy_density,
            thermo.specific_energy,
            grid,
            θ,
            model.specific_moisture,
            model.formulation,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

@kernel function _energy_density_from_potential_temperature!(energy_density,
                                                             specific_energy,
                                                             grid,
                                                             potential_temperature,
                                                             specific_moisture,
                                                             formulation::AnelasticFormulation,
                                                             microphysics,
                                                             microphysical_fields,
                                                             constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = formulation.reference_state.pressure[i, j, k]
        ρᵣ = formulation.reference_state.density[i, j, k]
        qᵗ = specific_moisture[i, j, k]
        θ = potential_temperature[i, j, k]
    end

    p₀ = formulation.reference_state.base_pressure
    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    𝒰θ₀ = LiquidIcePotentialTemperatureState(θ, q, p₀, pᵣ)
    𝒰θ₁ = maybe_adjust_thermodynamic_state(𝒰θ₀, microphysics, microphysical_fields, qᵗ, constants)
    T = temperature(𝒰θ₁, constants)

    z = znode(i, j, k, grid, c, c, c)
    q₁ = 𝒰θ₁.moisture_mass_fractions
    𝒰e₀ = StaticEnergyState(zero(T), q₁, z, pᵣ)
    𝒰e₁ = with_temperature(𝒰e₀, T, constants)
    e = 𝒰e₁.static_energy

    @inbounds specific_energy[i, j, k] = e
    @inbounds energy_density[i, j, k] = ρᵣ * e
end
