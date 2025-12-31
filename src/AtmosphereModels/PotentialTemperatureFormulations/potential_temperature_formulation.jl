#####
##### LiquidIcePotentialTemperatureFormulation
#####

"""
$(TYPEDSIGNATURES)

`LiquidIcePotentialTemperatureFormulation` uses liquid-ice potential temperature density `ρθ`
as the prognostic thermodynamic variable.

Liquid-ice potential temperature is a conserved quantity in moist adiabatic processes and is defined as:

```math
θˡⁱ = T \\left( \\frac{p^{st}}{p} \\right)^{Rᵐ/cᵖᵐ} \\exp\\left( \\frac{ℒˡᵣ qˡ + ℒⁱᵣ qⁱ}{cᵖᵐ T} \\right)
```
"""
struct LiquidIcePotentialTemperatureFormulation{F, T}
    potential_temperature_density :: F  # ρθ (prognostic)
    potential_temperature :: T          # θ = ρθ / ρ (diagnostic)
end

Adapt.adapt_structure(to, formulation::LiquidIcePotentialTemperatureFormulation) =
    LiquidIcePotentialTemperatureFormulation(adapt(to, formulation.potential_temperature_density),
                                             adapt(to, formulation.potential_temperature))

function BoundaryConditions.fill_halo_regions!(formulation::LiquidIcePotentialTemperatureFormulation)
    fill_halo_regions!(formulation.potential_temperature)
    return nothing
end

#####
##### Field naming interface
#####

prognostic_thermodynamic_field_names(::LiquidIcePotentialTemperatureFormulation) = tuple(:ρθ)
additional_thermodynamic_field_names(::LiquidIcePotentialTemperatureFormulation) = tuple(:θ)
thermodynamic_density_name(::LiquidIcePotentialTemperatureFormulation) = :ρθ
thermodynamic_density(formulation::LiquidIcePotentialTemperatureFormulation) = formulation.potential_temperature_density

# Val-based versions for pre-materialization (called via Symbol fallback in interface)
prognostic_thermodynamic_field_names(::Val{:LiquidIcePotentialTemperature}) = tuple(:ρθ)
additional_thermodynamic_field_names(::Val{:LiquidIcePotentialTemperature}) = tuple(:θ)
thermodynamic_density_name(::Val{:LiquidIcePotentialTemperature}) = :ρθ

Oceananigans.fields(formulation::LiquidIcePotentialTemperatureFormulation) = (; θ=formulation.potential_temperature)
Oceananigans.prognostic_fields(formulation::LiquidIcePotentialTemperatureFormulation) = (; ρθ=formulation.potential_temperature_density)

#####
##### Materialization
#####

function materialize_formulation(::Val{:LiquidIcePotentialTemperature}, dynamics, grid, boundary_conditions)
    potential_temperature_density = CenterField(grid, boundary_conditions=boundary_conditions.ρθ)
    potential_temperature = CenterField(grid)  # θ = ρθ / ρ (diagnostic)
    return LiquidIcePotentialTemperatureFormulation(potential_temperature_density, potential_temperature)
end

#####
##### Auxiliary variable computation
#####

function compute_auxiliary_thermodynamic_variables!(formulation::LiquidIcePotentialTemperatureFormulation, dynamics, i, j, k, grid)
    ρ = dynamics_density(dynamics)
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        ρθ = formulation.potential_temperature_density[i, j, k]
        formulation.potential_temperature[i, j, k] = ρθ / ρᵢ
    end
    return nothing
end

#####
##### Thermodynamic state diagnosis
#####

"""
$(TYPEDSIGNATURES)

Build a `LiquidIcePotentialTemperatureState` at grid point `(i, j, k)` from the given `formulation`,
`dynamics`, and pre-computed moisture mass fractions `q`.
"""
function diagnose_thermodynamic_state(i, j, k, grid,
                                      formulation::LiquidIcePotentialTemperatureFormulation,
                                      dynamics,
                                      q)

    θ = @inbounds formulation.potential_temperature[i, j, k]
    pᵣ = @inbounds dynamics_pressure(dynamics)[i, j, k]
    pˢᵗ = standard_pressure(dynamics)

    return LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, pᵣ)
end

#####
##### Prognostic field collection
#####

function collect_prognostic_fields(formulation::LiquidIcePotentialTemperatureFormulation,
                                   dynamics,
                                   momentum,
                                   moisture_density,
                                   microphysical_fields,
                                   tracers)

    ρθ = formulation.potential_temperature_density
    thermodynamic_variables = (ρθ=ρθ, ρqᵗ=moisture_density)
    dynamics_fields = dynamics_prognostic_fields(dynamics)
    return merge(dynamics_fields, momentum, thermodynamic_variables, microphysical_fields, tracers)
end

#####
##### Show methods
#####

function Base.summary(::LiquidIcePotentialTemperatureFormulation)
    return "LiquidIcePotentialTemperatureFormulation"
end

function Base.show(io::IO, formulation::LiquidIcePotentialTemperatureFormulation)
    print(io, summary(formulation))
    if formulation.potential_temperature_density !== nothing
        print(io, '\n')
        print(io, "├── potential_temperature_density: ", prettysummary(formulation.potential_temperature_density), '\n')
        print(io, "└── potential_temperature: ", prettysummary(formulation.potential_temperature))
    end
end

