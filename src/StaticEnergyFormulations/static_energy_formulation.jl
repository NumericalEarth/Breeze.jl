#####
##### StaticEnergyFormulation
#####

"""
$(TYPEDEF)

`StaticEnergyFormulation` uses moist static energy density `ρs` as the prognostic thermodynamic variable.

Moist static energy is a conserved quantity in adiabatic, frictionless flow that combines
sensible heat, gravitational potential energy, and latent heat:

```math
s = cᵖᵐ T + g z - ℒˡᵣ qˡ - ℒⁱᵣ qⁱ
```

The energy density equation includes a buoyancy flux term following [Pauluis2008](@citet).
"""
struct StaticEnergyFormulation{E, S}
    energy_density :: E    # ρs (prognostic)
    specific_energy :: S   # s = ρs / ρ (diagnostic)
end

Adapt.adapt_structure(to, formulation::StaticEnergyFormulation) =
    StaticEnergyFormulation(adapt(to, formulation.energy_density),
                            adapt(to, formulation.specific_energy))

function BoundaryConditions.fill_halo_regions!(formulation::StaticEnergyFormulation)
    fill_halo_regions!(formulation.specific_energy)
    return nothing
end

#####
##### Field naming interface
#####

AtmosphereModels.prognostic_thermodynamic_field_names(::StaticEnergyFormulation) = tuple(:ρs)
AtmosphereModels.additional_thermodynamic_field_names(::StaticEnergyFormulation) = tuple(:s)
AtmosphereModels.thermodynamic_density_name(::StaticEnergyFormulation) = :ρs
AtmosphereModels.thermodynamic_density(formulation::StaticEnergyFormulation) = formulation.energy_density
AtmosphereModels.with_thermodynamic_density(f::StaticEnergyFormulation, ρs) =
    StaticEnergyFormulation(ρs, f.specific_energy)

AtmosphereModels.prognostic_thermodynamic_field_names(::Val{:StaticEnergy}) = tuple(:ρs)
AtmosphereModels.additional_thermodynamic_field_names(::Val{:StaticEnergy}) = tuple(:s)
AtmosphereModels.thermodynamic_density_name(::Val{:StaticEnergy}) = :ρs

Oceananigans.fields(formulation::StaticEnergyFormulation) = (; s=formulation.specific_energy)
Oceananigans.prognostic_fields(formulation::StaticEnergyFormulation) = (; ρs=formulation.energy_density)

#####
##### Materialization
#####

function AtmosphereModels.materialize_formulation(::Val{:StaticEnergy}, dynamics, grid, boundary_conditions)
    energy_density = CenterField(grid, boundary_conditions=boundary_conditions.ρs)
    specific_energy = CenterField(grid)  # s = ρs / ρ (diagnostic per-mass energy)
    return StaticEnergyFormulation(energy_density, specific_energy)
end

#####
##### Auxiliary variable computation
#####

function AtmosphereModels.compute_auxiliary_thermodynamic_variables!(formulation::StaticEnergyFormulation, dynamics, i, j, k, grid)
    ρ = dynamics_density(dynamics)
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        ρs = formulation.energy_density[i, j, k]
        formulation.specific_energy[i, j, k] = ρs / ρᵢ
    end
    return nothing
end

#####
##### Thermodynamic state diagnosis
#####

"""
$(TYPEDSIGNATURES)

Build a `StaticEnergyState` at grid point `(i, j, k)` from the given
`formulation`, `dynamics`, and pre-computed moisture mass fractions `q`.
"""
function AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid,
                                                       formulation::StaticEnergyFormulation,
                                                       dynamics,
                                                       q)

    s = @inbounds formulation.specific_energy[i, j, k]
    pᵣ = @inbounds dynamics_pressure(dynamics)[i, j, k]
    z = znode(i, j, k, grid, c, c, c)

    return StaticEnergyState(s, q, z, pᵣ)
end

#####
##### Prognostic field collection
#####

function AtmosphereModels.collect_prognostic_fields(formulation::StaticEnergyFormulation,
                                                    dynamics,
                                                    momentum,
                                                    moisture_density,
                                                    moisture_name,
                                                    microphysical_fields,
                                                    tracers)
    ρs = formulation.energy_density
    thermodynamic_variables = merge((ρs=ρs,), NamedTuple{(moisture_name,)}((moisture_density,)))
    dynamics_fields = dynamics_prognostic_fields(dynamics)
    return merge(dynamics_fields, momentum, thermodynamic_variables, microphysical_fields, tracers)
end

#####
##### Show methods
#####

function Base.summary(::StaticEnergyFormulation)
    return "StaticEnergyFormulation"
end

function Base.show(io::IO, formulation::StaticEnergyFormulation)
    print(io, summary(formulation))
    if formulation.energy_density !== nothing
        print(io, '\n')
        print(io, "├── energy_density: ", prettysummary(formulation.energy_density), '\n')
        print(io, "└── specific_energy: ", prettysummary(formulation.specific_energy))
    end
end
