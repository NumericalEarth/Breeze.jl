#####
##### LiquidIcePotentialTemperatureFormulation
#####

"""
$(TYPEDSIGNATURES)

`LiquidIcePotentialTemperatureFormulation` uses liquid-ice potential temperature density `ρθ`
as the prognostic thermodynamic variable.

Liquid-ice potential temperature is a conserved quantity in moist adiabatic processes and is defined as:

```math
θˡⁱ = T \\left( \\frac{p^{st}}{p} \\right)^{Rᵐ/cᵖᵐ} \\exp\\left( -\\frac{ℒˡᵣ qˡ + ℒⁱᵣ qⁱ}{cᵖᵐ T} \\right)
```

Recovering temperature from θˡⁱ may require an iterative inversion, depending on the
dynamics: with prognostic-density (compressible) dynamics, temperature solves the implicit
relation `T = (ρRᵐT/pˢᵗ)^κ θ + ΔL/cᵖᵐ`. The inversion is controlled by `temperature_solver`:

* `DefaultTemperatureSolver()` (default): resolved at materialization to
  [`default_temperature_solver(dynamics)`](@ref Breeze.AtmosphereModels.default_temperature_solver) —
  `nothing` for anelastic dynamics (closed-form inversion) and `NewtonSolver()` for
  compressible dynamics.
* [`NewtonSolver`](@ref Breeze.Solvers.NewtonSolver): tolerance-based Newton iteration.
* [`FixedIterations`](@ref Breeze.Solvers.FixedIterations): a fixed number of Newton steps with no convergence test,
  which unrolls to straight-line code (required for Reactant tracing and cheap
  reverse-mode differentiation).
* `nothing`: the non-iterated closed-form inversion.
"""
struct LiquidIcePotentialTemperatureFormulation{F, T, S}
    potential_temperature_density :: F  # ρθ (prognostic)
    potential_temperature :: T          # θ = ρθ / ρ (diagnostic)
    temperature_solver :: S             # solver for the θˡⁱ→T inversion (or Nothing when closed-form)
end

"""
$(TYPEDSIGNATURES)

Return a `LiquidIcePotentialTemperatureFormulation` with the given `temperature_solver`.
The prognostic and diagnostic fields are materialized later in the model constructor.

```jldoctest
using Breeze

LiquidIcePotentialTemperatureFormulation(temperature_solver = FixedIterations(2))

# output
LiquidIcePotentialTemperatureFormulation
└── temperature_solver: FixedIterations(2)
```
"""
LiquidIcePotentialTemperatureFormulation(; temperature_solver = DefaultTemperatureSolver()) =
    LiquidIcePotentialTemperatureFormulation(nothing, nothing, temperature_solver)

Adapt.adapt_structure(to, formulation::LiquidIcePotentialTemperatureFormulation) =
    LiquidIcePotentialTemperatureFormulation(adapt(to, formulation.potential_temperature_density),
                                             adapt(to, formulation.potential_temperature),
                                             formulation.temperature_solver)

function BoundaryConditions.fill_halo_regions!(formulation::LiquidIcePotentialTemperatureFormulation)
    fill_halo_regions!(formulation.potential_temperature)
    return nothing
end

#####
##### Field naming interface
#####

AtmosphereModels.prognostic_thermodynamic_field_names(::LiquidIcePotentialTemperatureFormulation) = tuple(:ρθ)
AtmosphereModels.additional_thermodynamic_field_names(::LiquidIcePotentialTemperatureFormulation) = tuple(:θ)
AtmosphereModels.thermodynamic_density_name(::LiquidIcePotentialTemperatureFormulation) = :ρθ
AtmosphereModels.thermodynamic_density(formulation::LiquidIcePotentialTemperatureFormulation) = formulation.potential_temperature_density

# Val-based versions for pre-materialization (called via Symbol fallback in interface)
AtmosphereModels.prognostic_thermodynamic_field_names(::Val{:LiquidIcePotentialTemperature}) = tuple(:ρθ)
AtmosphereModels.additional_thermodynamic_field_names(::Val{:LiquidIcePotentialTemperature}) = tuple(:θ)
AtmosphereModels.thermodynamic_density_name(::Val{:LiquidIcePotentialTemperature}) = :ρθ

Oceananigans.fields(formulation::LiquidIcePotentialTemperatureFormulation) = (; θ=formulation.potential_temperature)
Oceananigans.prognostic_fields(formulation::LiquidIcePotentialTemperatureFormulation) = (; ρθ=formulation.potential_temperature_density)

#####
##### Materialization
#####

AtmosphereModels.materialize_formulation(::Val{:LiquidIcePotentialTemperature}, dynamics, grid, boundary_conditions) =
    materialize_formulation(LiquidIcePotentialTemperatureFormulation(), dynamics, grid, boundary_conditions)

function AtmosphereModels.materialize_formulation(formulation::LiquidIcePotentialTemperatureFormulation,
                                                  dynamics, grid, boundary_conditions)
    potential_temperature_density = CenterField(grid, boundary_conditions=boundary_conditions.ρθ)
    potential_temperature = CenterField(grid)  # θ = ρθ / ρ (diagnostic)
    temperature_solver = materialize_temperature_solver(formulation.temperature_solver, dynamics, grid)
    return LiquidIcePotentialTemperatureFormulation(potential_temperature_density,
                                                    potential_temperature,
                                                    temperature_solver)
end

# `DefaultTemperatureSolver` defers the choice of solver to the dynamics: the need for an
# iterative θˡⁱ→T inversion is dictated by the intersection of the formulation and the
# dynamics (anelastic: closed-form, no solver; compressible: implicit, Newton by default).
materialize_temperature_solver(::DefaultTemperatureSolver, dynamics, grid) =
    materialize_solver(default_temperature_solver(dynamics), eltype(grid))

materialize_temperature_solver(temperature_solver, dynamics, grid) =
    materialize_solver(temperature_solver, eltype(grid))

#####
##### Auxiliary variable computation
#####

function AtmosphereModels.compute_auxiliary_thermodynamic_variables!(formulation::LiquidIcePotentialTemperatureFormulation, dynamics, i, j, k, grid)
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

Build a `LiquidIcePotentialTemperatureState` at grid point `(i, j, k)` from the
given `formulation`, `dynamics`, and pre-computed moisture mass fractions `q`.
"""
function AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid,
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

function AtmosphereModels.collect_prognostic_fields(formulation::LiquidIcePotentialTemperatureFormulation,
                                                    dynamics,
                                                    momentum,
                                                    moisture_density,
                                                    moisture_name,
                                                    microphysical_fields,
                                                    tracers)

    ρθ = formulation.potential_temperature_density
    thermodynamic_variables = merge((ρθ=ρθ,), NamedTuple{(moisture_name,)}((moisture_density,)))
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
        print(io, "├── potential_temperature: ", prettysummary(formulation.potential_temperature), '\n')
    else
        print(io, '\n')
    end
    print(io, "└── temperature_solver: ", summary(formulation.temperature_solver))
end
