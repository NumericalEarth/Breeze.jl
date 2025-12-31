#####
##### StaticEnergyFormulation
#####

"""
$(TYPEDSIGNATURES)

`StaticEnergyFormulation` uses moist static energy density `ρe` as the prognostic thermodynamic variable.

Moist static energy is a conserved quantity in adiabatic, frictionless flow that combines
sensible heat, gravitational potential energy, and latent heat:

```math
e = cᵖᵐ T + g z - ℒˡᵣ qˡ - ℒⁱᵣ qⁱ
```

The energy density equation includes a buoyancy flux term following [Pauluis2008](@citet).
"""
struct StaticEnergyFormulation{E, S}
    energy_density :: E    # ρe (prognostic)
    specific_energy :: S   # e = ρe / ρ (diagnostic)
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

prognostic_thermodynamic_field_names(::StaticEnergyFormulation) = tuple(:ρe)
additional_thermodynamic_field_names(::StaticEnergyFormulation) = tuple(:e)
thermodynamic_density_name(::StaticEnergyFormulation) = :ρe
thermodynamic_density(formulation::StaticEnergyFormulation) = formulation.energy_density

prognostic_thermodynamic_field_names(::Val{:StaticEnergy}) = tuple(:ρe)
additional_thermodynamic_field_names(::Val{:StaticEnergy}) = tuple(:e)
thermodynamic_density_name(::Val{:StaticEnergy}) = :ρe

Oceananigans.fields(formulation::StaticEnergyFormulation) = (; e=formulation.specific_energy)
Oceananigans.prognostic_fields(formulation::StaticEnergyFormulation) = (; ρe=formulation.energy_density)

#####
##### Materialization
#####

function materialize_formulation(::Val{:StaticEnergy}, dynamics, grid, boundary_conditions)
    energy_density = CenterField(grid, boundary_conditions=boundary_conditions.ρe)
    specific_energy = CenterField(grid)  # e = ρe / ρ (diagnostic per-mass energy)
    return StaticEnergyFormulation(energy_density, specific_energy)
end

#####
##### Auxiliary variable computation
#####

function compute_auxiliary_thermodynamic_variables!(formulation::StaticEnergyFormulation, dynamics, i, j, k, grid)
    ρ = dynamics_density(dynamics)
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        ρe = formulation.energy_density[i, j, k]
        formulation.specific_energy[i, j, k] = ρe / ρᵢ
    end
    return nothing
end

#####
##### Thermodynamic state diagnosis
#####

"""
$(TYPEDSIGNATURES)

Build a `StaticEnergyState` at grid point `(i, j, k)` from the given `formulation`, `dynamics`,
and pre-computed moisture mass fractions `q`.
"""
function diagnose_thermodynamic_state(i, j, k, grid,
                                      formulation::StaticEnergyFormulation,
                                      dynamics,
                                      q)

    e = @inbounds formulation.specific_energy[i, j, k]
    pᵣ = @inbounds dynamics_pressure(dynamics)[i, j, k]
    z = znode(i, j, k, grid, c, c, c)

    return StaticEnergyState(e, q, z, pᵣ)
end

#####
##### Prognostic field collection
#####

function collect_prognostic_fields(formulation::StaticEnergyFormulation,
                                   dynamics,
                                   momentum,
                                   moisture_density,
                                   microphysical_fields,
                                   tracers)
    ρe = formulation.energy_density
    thermodynamic_variables = (ρe=ρe, ρqᵗ=moisture_density)
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

